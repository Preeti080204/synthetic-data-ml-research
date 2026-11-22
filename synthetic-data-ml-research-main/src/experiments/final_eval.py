"""Score the frozen plan on the held-out test partition.

Importing this module does not load the test frame. ``run_authorized_test``
loads it only after ``preflight`` returns no errors.
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from experiments.cache import file_sha256, source_tree_fingerprint
from experiments.design import (
    CLASSIFIERS,
    GENERATORS,
    augmentation_synthetic_count,
    budget_count,
    compose_mixture,
    mixture_counts,
    synthetic_count,
)
from experiments.final_guard import freeze_path, preflight
from experiments.tracking import utc_now, write_record
from utils.paths import project_root, results_dir


def run_authorized_test() -> Path:
    errors = preflight()
    if errors:
        raise RuntimeError("Final-test preflight failed: " + "; ".join(errors))
    from data.preprocessing import split_xy
    from experiments.data import (
        hyperparameters_for_seed,
        load_modeling_frames,
        load_test_frame,
        modeling_columns,
        selected_hyperparameters,
    )
    from experiments.design import stratified_sample
    from generators.tabular import load_synthetic
    from models.baseline import build_model_pipeline
    from utils.paths import load_config

    freeze = json.loads(freeze_path().read_text(encoding="utf-8"))
    train, _validation = load_modeling_frames()
    test = load_test_frame()
    selected = selected_hyperparameters()
    phase1 = load_config()
    seeds = [int(seed) for seed in freeze["seeds"]]
    shares = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
    output = results_dir() / "final_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    fingerprint = source_tree_fingerprint()
    rows.extend(_main(
        train, test, seeds, shares, selected, phase1, split_xy, stratified_sample,
        load_synthetic, fingerprint,
    ))
    rows.extend(_scarcity(
        train, test, seeds, freeze, selected, phase1, split_xy, stratified_sample,
        load_synthetic, modeling_columns, fingerprint,
    ))
    frame = pd.DataFrame(rows)
    metrics_path = output / "test_run_metrics.csv"
    frame.to_csv(metrics_path, index=False)
    record = {
        "status": "completed",
        "timestamp": utc_now(),
        "runs": int(len(frame)),
        "source_tree_fingerprint": source_tree_fingerprint(),
        "git_commit": None,
        "git_note": "The project directory was not a Git repository at audit time.",
        "freeze_sha256": file_sha256(freeze_path()),
        "test_partition_rows": int(len(test)),
        "analysis_plan": "paired comparisons follow docs/ANALYSIS_PLAN.md; no test-based selection",
    }
    write_record(output / "reproducibility.json", record)
    from reporting.final_report import write_final_report

    write_final_report(frame)
    return metrics_path


def _score(pipeline, features, target) -> dict:
    probabilities = pipeline.predict_proba(features)[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    labels = target.to_numpy()
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "threshold": 0.5,
        "partition": "test",
    }


def _fit_row(train, test, classifier, params, phase1, split_xy, lineage) -> dict:
    from models.baseline import build_model_pipeline

    destination = results_dir() / "final_evaluation" / "runs" / f"{lineage['experiment_id']}.json"
    if destination.is_file():
        cached = json.loads(destination.read_text(encoding="utf-8"))
        if (
            cached.get("status") == "completed"
            and cached.get("partition") == "test"
            and cached.get("experiment_id") == lineage["experiment_id"]
            and cached.get("roc_auc") is not None
        ):
            cached["classifier"] = classifier
            cached["cache"] = "hit"
            return cached
    features, target = split_xy(train)
    test_features, test_target = split_xy(test)
    pipeline = build_model_pipeline(classifier, params, phase1)
    pipeline.fit(features, target)
    payload = dict(lineage)
    payload.update(_score(pipeline, test_features, test_target))
    payload["status"] = "completed"
    payload["timestamp"] = utc_now()
    payload["classifier"] = classifier
    write_record(destination, payload)
    return payload


def _main(train, test, seeds, shares, selected, phase1, split_xy, stratified_sample, load_synthetic, fingerprint):
    from experiments.data import hyperparameters_for_seed
    from experiments.train import experiment_id
    from generators.tabular import generation_dir

    root = project_root() / "data" / "synthetic" / "main"
    rows = []
    for seed in seeds:
        params = {name: hyperparameters_for_seed(name, seed, selected) for name in CLASSIFIERS}
        n_real, n_syn = mixture_counts(len(train), 0.0)
        real_frame = stratified_sample(train, n_real, seed)
        for classifier in CLASSIFIERS:
            identifier = experiment_id("final_main", "real", 0.0, classifier, seed)
            rows.append(_fit_row(
                real_frame, test, classifier, params[classifier], phase1, split_xy,
                _lineage(identifier, "main", "real", 0.0, seed, None, n_real, n_syn, None, fingerprint, classifier),
            ))
        for kind in GENERATORS:
            synthetic = load_synthetic(root, kind, seed)
            synthetic_hash = file_sha256(generation_dir(root, kind, seed) / "synthetic.csv")
            for share in shares:
                if share == 0:
                    continue
                n_real, n_syn = mixture_counts(len(train), share)
                mixture = compose_mixture(train, synthetic, n_real, n_syn, seed)
                for classifier in CLASSIFIERS:
                    identifier = experiment_id("final_main", kind, share, classifier, seed)
                    rows.append(_fit_row(
                        mixture, test, classifier, params[classifier], phase1, split_xy,
                        _lineage(identifier, "main", kind, share, seed, None, n_real, n_syn, synthetic_hash, fingerprint, classifier),
                    ))
    return rows


def _scarcity(train, test, seeds, freeze, selected, phase1, split_xy, stratified_sample, load_synthetic, modeling_columns, fingerprint):
    from experiments.data import hyperparameters_for_seed
    from experiments.train import experiment_id
    from generators.tabular import generation_dir

    rows = []
    selection = freeze["selected_augmentation"]["generators"]
    shared = float(freeze["selected_augmentation"]["shared_augmentation_share"])
    root = project_root() / "data" / "synthetic" / "scarcity"
    for budget in (0.25, 0.50, 0.75):
        n_budget = budget_count(len(train), budget)
        for seed in seeds:
            subset = stratified_sample(train, n_budget, seed)
            params = {name: hyperparameters_for_seed(name, seed, selected) for name in CLASSIFIERS}
            for classifier in CLASSIFIERS:
                identifier = experiment_id("final_scarcity_real", "real", 0.0, classifier, seed, budget)
                rows.append(_fit_row(
                    subset, test, classifier, params[classifier], phase1, split_xy,
                    _lineage(identifier, "scarcity_real", "real", 0.0, seed, budget, n_budget, 0, None, fingerprint, classifier),
                ))
            for kind in GENERATORS:
                generator_share = float(selection[kind]["augmentation_share"])
                plans = (
                    [("generator_specific_and_shared", generator_share)]
                    if generator_share == shared
                    else [("generator_specific", generator_share), ("shared_ratio", shared)]
                )
                folder = root / f"budget_{budget:.2f}"
                synthetic = load_synthetic(folder, kind, seed)
                synthetic_hash = file_sha256(generation_dir(folder, kind, seed) / "synthetic.csv")
                for _comparison, share in plans:
                    n_syn = augmentation_synthetic_count(n_budget, share)
                    enlarged = pd.concat(
                        [subset, stratified_sample(synthetic, n_syn, seed)],
                        ignore_index=True,
                    ).sample(frac=1.0, random_state=seed).reset_index(drop=True)
                    n_syn_fixed = synthetic_count(n_budget, share)
                    replaced = compose_mixture(subset, synthetic, n_budget - n_syn_fixed, n_syn_fixed, seed)
                    for classifier in CLASSIFIERS:
                        aug_id = experiment_id("final_scarcity_augmentation", kind, share, classifier, seed, budget)
                        rows.append(_fit_row(
                            enlarged, test, classifier, params[classifier], phase1, split_xy,
                            _lineage(aug_id, "scarcity_augmentation", kind, share, seed, budget, n_budget, n_syn, synthetic_hash, fingerprint, classifier),
                        ))
                        rep_id = experiment_id("final_scarcity_replacement", kind, share, classifier, seed, budget)
                        rows.append(_fit_row(
                            replaced, test, classifier, params[classifier], phase1, split_xy,
                            _lineage(rep_id, "scarcity_replacement", kind, share, seed, budget, n_budget - n_syn_fixed, n_syn_fixed, synthetic_hash, fingerprint, classifier),
                        ))
    return rows


def _lineage(identifier, stage, generator, share, seed, budget, n_real, n_syn, synthetic_hash, fingerprint, classifier) -> dict:
    return {
        "experiment_id": identifier,
        "stage": stage,
        "generator": generator,
        "classifier": classifier,
        "synthetic_share": float(share),
        "seed": int(seed),
        "budget": None if budget is None else float(budget),
        "n_real": int(n_real),
        "n_synthetic": int(n_syn),
        "synthetic_sha256": synthetic_hash,
        "source_tree_fingerprint": fingerprint,
        "git_commit": None,
        "test_based_selection": False,
    }
