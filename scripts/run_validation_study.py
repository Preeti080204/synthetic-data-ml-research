"""Run the validation-stage study.

The held-out test partition is not loaded. Completed generator fits and
classifier runs are skipped on a later invocation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from experiments.data import (
    hyperparameters_for_seed,
    load_modeling_frames,
    modeling_columns,
    selected_hyperparameters,
    verify_phase1_validation,
)
from experiments.design import (
    CLASSIFIERS,
    GENERATORS,
    augmentation_synthetic_count,
    budget_count,
    experiment_config,
    generator_config,
    mixture_counts,
    share_token,
    stratified_sample,
    synthetic_count,
    synthetic_shares,
)
from experiments.tracking import append_jsonl, utc_now, write_record
from experiments.train import experiment_id, run_classifier
from generators.tabular import generate_cached, load_synthetic
from quality.fidelity import evaluate_quality
from reporting import figures
from analysis.summarize import paired_differences, select_augmentation_shares, summarize_runs
from utils.paths import (
    data_processed_dir,
    models_dir,
    project_root,
    results_dir,
    sha256_file,
    tables_dir,
    write_json,
)

EXPECTED_MAIN_COUNTS = {
    0.0: (34189, 0),
    0.10: (34189 - 3419, 3419),
    0.25: (34189 - 8547, 8547),
    0.50: (34189 - 17094, 17094),
    0.75: (34189 - 25642, 25642),
    1.0: (0, 34189),
}


def _logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("validation_study")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream)
    logger.propagate = False
    return logger


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def _roots(smoke: bool) -> dict[str, Path]:
    if smoke:
        base = results_dir() / "smoke"
        return {
            "synthetic": base / "synthetic",
            "metrics": base / "raw_metrics",
            "tables": base / "tables",
            "figures": base / "figures",
            "statistics": base / "statistical_tests",
            "logs": base / "logs" / "study.log",
            "jsonl": base / "logs" / "experiments.jsonl",
        }
    return {
        "synthetic": project_root() / "data" / "synthetic",
        "metrics": results_dir() / "raw_metrics",
        "tables": results_dir() / "tables" / "study",
        "figures": results_dir() / "figures" / "study",
        "statistics": results_dir() / "statistical_tests",
        "logs": results_dir() / "experiment_logs" / "validation_study.log",
        "jsonl": results_dir() / "experiment_logs" / "experiments.jsonl",
    }


def _fingerprints() -> dict:
    root = project_root()
    return {
        "adult_combined": sha256_file(data_processed_dir() / "adult_combined.csv"),
        "split_indices": sha256_file(data_processed_dir() / "split_indices.csv"),
        "selected_configurations": sha256_file(tables_dir() / "selected_configurations.json"),
        "experiments_yaml": sha256_file(root / "configs" / "experiments.yaml"),
        "generators_yaml": sha256_file(root / "configs" / "generators.yaml"),
        "analysis_plan": sha256_file(root / "docs" / "ANALYSIS_PLAN.md"),
    }


def _flatten_records(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("status") != "completed" or "metrics" not in payload:
            continue
        row = {
            key: value
            for key, value in payload.items()
            if key not in {"metrics", "hyperparameters", "artifact_paths"}
            and not isinstance(value, (dict, list))
        }
        row.update(payload["metrics"])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _fit_ctgan_subprocess(real, seed, root, n_rows, overrides, logger) -> dict:
    """Fit CTGAN in a fresh process.

    Loading the saved XGBoost pipeline in this process and then fitting CTGAN
    deadlocks PyTorch. A child process that never loads that pipeline does not.
    """
    from generators.tabular import generation_dir

    destination = generation_dir(root, "ctgan", seed)
    destination.mkdir(parents=True, exist_ok=True)
    input_path = destination / "_fit_input.csv"
    real.to_csv(input_path, index=False)
    script = """
import json, os, sys
from pathlib import Path
os.environ["SDV_CTGAN_SUBPROCESS"] = "1"
sys.path.insert(0, "src")
import pandas as pd
from generators.tabular import generate_cached
input_path, root, seed, n_rows, overrides_path = sys.argv[1:]
frame = pd.read_csv(input_path)
overrides = json.loads(Path(overrides_path).read_text()) if overrides_path != "-" else None
generate_cached(frame, "ctgan", int(seed), Path(root), n_rows=int(n_rows), overrides=overrides)
"""
    overrides_path = "-"
    if overrides:
        overrides_file = destination / "_overrides.json"
        overrides_file.write_text(json.dumps(overrides), encoding="utf-8")
        overrides_path = str(overrides_file)
    env = os.environ.copy()
    env["SDV_CTGAN_SUBPROCESS"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(input_path),
            str(root),
            str(seed),
            str(n_rows),
            overrides_path,
        ],
        cwd=project_root(),
        env=env,
        check=False,
    )
    meta_path = destination / "generation.json"
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = {
            "status": "failed",
            "generator": "ctgan",
            "seed": int(seed),
            "error": f"CTGAN subprocess exited {completed.returncode} without a generation record.",
        }
    if completed.returncode != 0 and payload.get("status") != "completed":
        payload["status"] = "failed"
        payload["error"] = payload.get("error") or f"CTGAN subprocess exited {completed.returncode}."
    logger.info("CTGAN subprocess exit %s", completed.returncode)
    return payload


def _fit_generator(real, kind, seed, root, n_rows, overrides, logger) -> dict:
    logger.info("generator %s seed %s rows %s", kind, seed, n_rows)
    if kind == "ctgan" and os.environ.get("SDV_CTGAN_SUBPROCESS") != "1":
        result = _fit_ctgan_subprocess(real, seed, root, n_rows, overrides, logger)
    else:
        result = generate_cached(real, kind, seed, root, n_rows=n_rows, overrides=overrides)
    error = str(result.get("error", "")).lower()
    if result.get("status") != "completed" and kind == "ctgan" and any(
        token in error for token in ("mps", "cuda", "gpu")
    ):
        logger.info("retrying CTGAN on CPU after accelerator error")
        cpu_overrides = dict(overrides or {})
        cpu_overrides["enable_gpu"] = False
        result = generate_cached(real, kind, seed, root, n_rows=n_rows, overrides=cpu_overrides)
    logger.info(
        "generator %s seed %s status %s seconds %.1f",
        kind,
        seed,
        result.get("status"),
        float(result.get("fit_seconds", 0.0)),
    )
    return result


def _run_one(train, validation, classifier, params, path, context, logger, jsonl: Path) -> dict:
    logger.info("fit %s", context["experiment_id"])
    context = dict(context)
    context["rss_mb_before"] = _rss_mb()
    try:
        result = run_classifier(train, validation, classifier, params, path, context)
    except Exception as exc:
        result = dict(context)
        result.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "timestamp": utc_now(),
            }
        )
        write_record(path, result)
    result["rss_mb_after"] = _rss_mb()
    append_jsonl(jsonl, {key: result.get(key) for key in (
        "experiment_id", "status", "timestamp", "generator", "synthetic_share",
        "classifier", "seed", "training_seconds", "cache",
    )})
    return result


def run_study(smoke: bool = False) -> None:
    roots = _roots(smoke)
    for key in ("synthetic", "metrics", "tables", "figures", "statistics"):
        roots[key].mkdir(parents=True, exist_ok=True)
    logger = _logger(roots["logs"])
    logging.getLogger("sdv").setLevel(logging.ERROR)
    fingerprints = _fingerprints()
    train, validation = load_modeling_frames()
    if not smoke:
        verification = verify_phase1_validation(validation)
        write_json(roots["tables"] / "phase1_verification.json", verification)
        logger.info("Phase 1 validation metrics matched the saved pipelines.")
        for share, expected in EXPECTED_MAIN_COUNTS.items():
            if mixture_counts(len(train), share) != expected:
                raise RuntimeError(f"Mixture counts for share {share} drifted.")
    else:
        train = stratified_sample(train, 400, 42)
        validation = stratified_sample(validation, 200, 42)
        logger.info("Smoke mode uses 400 training rows and 200 validation rows.")

    exp = experiment_config()
    gen = generator_config()
    seeds = [42] if smoke else [int(seed) for seed in exp["seeds"]]
    shares = [0.0, 0.5, 1.0] if smoke else synthetic_shares(exp)
    budgets = [0.50] if smoke else [float(budget) for budget in exp["scarcity"]["budgets"]]
    selected = selected_hyperparameters()
    overrides = {"epochs": 1, "batch_size": 40, "verbose": False} if smoke else None
    synthetic_root = roots["synthetic"] / ("smoke" if smoke else "main")

    for seed in seeds:
        for kind in GENERATORS:
            result = _fit_generator(
                modeling_columns(train), kind, seed, synthetic_root, len(train), overrides, logger
            )
            if result.get("status") != "completed":
                logger.error("Generator failed: %s", result.get("error"))

    for seed in seeds:
        params_by_model = {
            name: hyperparameters_for_seed(name, seed, selected) for name in CLASSIFIERS
        }
        n_real, n_syn = mixture_counts(len(train), 0.0)
        real_frame = stratified_sample(train, n_real, seed)
        for classifier in CLASSIFIERS:
            identifier = experiment_id("main", "real", 0.0, classifier, seed)
            context = _context(
                identifier, "main", "real", 0.0, seed, None, "real_only",
                n_real, n_syn, fingerprints,
            )
            _run_one(
                real_frame, validation, classifier, params_by_model[classifier],
                roots["metrics"] / f"{identifier}.json", context, logger, roots["jsonl"],
            )
        for kind in GENERATORS:
            try:
                synthetic = load_synthetic(synthetic_root, kind, seed)
            except (FileNotFoundError, RuntimeError) as exc:
                logger.error("Skipping %s seed %s: %s", kind, seed, exc)
                continue
            for share in shares:
                if share == 0:
                    continue
                n_real, n_syn = mixture_counts(len(train), share)
                mixture = _mixture(train, synthetic, n_real, n_syn, seed)
                for classifier in CLASSIFIERS:
                    identifier = experiment_id("main", kind, share, classifier, seed)
                    context = _context(
                        identifier, "main", kind, share, seed, None, "mixture",
                        n_real, n_syn, fingerprints,
                    )
                    _run_one(
                        mixture, validation, classifier, params_by_model[classifier],
                        roots["metrics"] / f"{identifier}.json", context, logger, roots["jsonl"],
                    )

    _quality(train, synthetic_root, seeds, roots, logger)
    runs = _flatten_records(sorted(roots["metrics"].glob("*.json")))
    if runs.empty:
        raise RuntimeError("No completed classifier runs were found.")
    main_runs = runs.loc[runs["stage"] == "main"].copy()
    selection = select_augmentation_shares(main_runs.loc[main_runs["generator"] != "real"])
    write_json(roots["tables"] / "selected_augmentation_ratios.json", selection)
    logger.info("Selected augmentation ratios: %s", json.dumps(selection["generators"]))
    _scarcity(
        train, validation, seeds, budgets, selection, selected, overrides,
        roots, fingerprints, logger,
    )
    runs = _flatten_records(sorted(roots["metrics"].glob("*.json")))
    _aggregate(runs, roots, logger)
    if not smoke:
        _figures(train, validation, synthetic_root, seeds, roots, logger)
        _freeze(roots, fingerprints, selection, gen, logger)
    logger.info("Validation-stage run finished. Test metrics were not computed.")


def _mixture(real, synthetic, n_real, n_syn, seed):
    from experiments.design import compose_mixture

    return compose_mixture(real, synthetic, n_real, n_syn, seed)


def _context(identifier, stage, generator, share, seed, budget, comparison, n_real, n_syn, fingerprints):
    return {
        "experiment_id": identifier,
        "stage": stage,
        "generator": generator,
        "synthetic_share": float(share),
        "real_share": float(1.0 - share) if stage == "main" else None,
        "seed": int(seed),
        "budget": None if budget is None else float(budget),
        "comparison": comparison,
        "n_real": int(n_real),
        "n_synthetic": int(n_syn),
        "dataset_fingerprint": fingerprints["adult_combined"],
        "split_fingerprint": fingerprints["split_indices"],
        "config_fingerprint": fingerprints["experiments_yaml"],
        "artifact_paths": {},
        "test_metrics_computed": False,
    }


def _quality(train, synthetic_root, seeds, roots, logger) -> None:
    rows = []
    numeric_rows = []
    categorical_rows = []
    association_rows = []
    for seed in seeds:
        for kind in GENERATORS:
            try:
                synthetic = load_synthetic(synthetic_root, kind, seed)
            except (FileNotFoundError, RuntimeError):
                continue
            logger.info("quality %s seed %s", kind, seed)
            report = evaluate_quality(modeling_columns(train), synthetic, kind, seed)
            real_corr = report.pop("real_correlation")
            syn_corr = report.pop("synthetic_correlation")
            real_corr.to_csv(roots["tables"] / "real_spearman.csv")
            syn_corr.to_csv(roots["tables"] / f"spearman_{kind}_seed_{seed}.csv")
            for row in report["numeric"]:
                numeric_rows.append({"generator": kind, "seed": seed, **row})
            for row in report["categorical"]:
                categorical_rows.append({"generator": kind, "seed": seed, **row})
            for row in report["target_associations"]:
                association_rows.append({"generator": kind, "seed": seed, **row})
            summary = {
                key: value
                for key, value in report.items()
                if key not in {"numeric", "categorical", "target_associations"}
            }
            summary["duplicates"] = report["duplicates"]
            write_json(roots["tables"] / f"quality_{kind}_seed_{seed}.json", summary)
            rows.append(
                {
                    "generator": kind,
                    "seed": seed,
                    "mean_categorical_tv": report["mean_categorical_tv"],
                    "mean_ks": report["mean_ks"],
                    "spearman_frobenius": report["spearman_frobenius"],
                    "real_positive_rate": report["real_positive_rate"],
                    "synthetic_positive_rate": report["synthetic_positive_rate"],
                    "exact_training_match_rate": report["duplicates"]["exact_training_match_rate"],
                    "exact_internal_duplicate_rate": report["duplicates"]["exact_internal_duplicate_rate"],
                    "near_duplicate_rate": report["duplicates"]["near_duplicate"]["synthetic_near_duplicate_rate"],
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(roots["tables"] / "quality_summary.csv", index=False)
        pd.DataFrame(numeric_rows).to_csv(roots["tables"] / "numeric_fidelity.csv", index=False)
        pd.DataFrame(categorical_rows).to_csv(roots["tables"] / "categorical_fidelity.csv", index=False)
        pd.DataFrame(association_rows).to_csv(roots["tables"] / "target_associations.csv", index=False)


def _scarcity(train, validation, seeds, budgets, selection, selected, overrides, roots, fingerprints, logger):
    root = roots["synthetic"] / "scarcity"
    for budget in budgets:
        n_budget = budget_count(len(train), budget)
        for seed in seeds:
            subset = stratified_sample(train, n_budget, seed)
            params_by_model = {
                name: hyperparameters_for_seed(name, seed, selected) for name in CLASSIFIERS
            }
            for classifier in CLASSIFIERS:
                identifier = experiment_id("scarcity_real", "real", 0.0, classifier, seed, budget)
                context = _context(
                    identifier, "scarcity_real", "real", 0.0, seed, budget, "real_only",
                    n_budget, 0, fingerprints,
                )
                _run_one(
                    subset, validation, classifier, params_by_model[classifier],
                    roots["metrics"] / f"{identifier}.json", context, logger, roots["jsonl"],
                )
            for kind in GENERATORS:
                generator_share = float(selection["generators"][kind]["augmentation_share"])
                shared_share = float(selection["shared_augmentation_share"])
                if generator_share == shared_share:
                    plans = [("generator_specific_and_shared", generator_share)]
                else:
                    plans = [
                        ("generator_specific", generator_share),
                        ("shared_ratio", shared_share),
                    ]
                needed = sorted({share for _, share in plans})
                n_generate = max(
                    max(augmentation_synthetic_count(n_budget, share) for share in needed),
                    max(synthetic_count(n_budget, share) for share in needed),
                )
                result = _fit_generator(
                    modeling_columns(subset), kind, seed, root / f"budget_{budget:.2f}",
                    n_generate, overrides, logger,
                )
                if result.get("status") != "completed":
                    logger.error("Scarcity generator failed: %s", result.get("error"))
                    continue
                synthetic = load_synthetic(root / f"budget_{budget:.2f}", kind, seed)
                for comparison, share in plans:
                    n_syn = augmentation_synthetic_count(n_budget, share)
                    enlarged = pd.concat(
                        [subset, stratified_sample(synthetic, n_syn, seed)],
                        ignore_index=True,
                    ).sample(frac=1.0, random_state=seed).reset_index(drop=True)
                    for classifier in CLASSIFIERS:
                        run_id = experiment_id(
                            "scarcity_augmentation", kind, share, classifier, seed, budget
                        )
                        context = _context(
                            run_id, "scarcity_augmentation", kind, share, seed, budget,
                            comparison, n_budget, n_syn, fingerprints,
                        )
                        _run_one(
                            enlarged, validation, classifier, params_by_model[classifier],
                            roots["metrics"] / f"{run_id}.json", context, logger, roots["jsonl"],
                        )
                    n_syn_fixed = synthetic_count(n_budget, share)
                    n_real_fixed = n_budget - n_syn_fixed
                    replaced = _mixture(subset, synthetic, n_real_fixed, n_syn_fixed, seed)
                    for classifier in CLASSIFIERS:
                        run_id = experiment_id(
                            "scarcity_replacement", kind, share, classifier, seed, budget
                        )
                        context = _context(
                            run_id, "scarcity_replacement", kind, share, seed, budget,
                            comparison, n_real_fixed, n_syn_fixed, fingerprints,
                        )
                        _run_one(
                            replaced, validation, classifier, params_by_model[classifier],
                            roots["metrics"] / f"{run_id}.json", context, logger, roots["jsonl"],
                        )


def _aggregate(runs: pd.DataFrame, roots, logger) -> None:
    runs.to_csv(roots["tables"] / "run_metrics.csv", index=False)
    summary = summarize_runs(runs)
    summary.to_csv(roots["tables"] / "metric_summary.csv", index=False)
    pieces = []
    if (runs["stage"] == "main").any():
        pieces.append(paired_differences(runs.loc[runs["stage"] == "main"], baseline_stage="main"))
    for stage in ("scarcity_augmentation", "scarcity_replacement"):
        if (runs["stage"] == stage).any():
            stage_runs = pd.concat(
                [runs.loc[runs["stage"] == stage], runs.loc[runs["stage"] == "scarcity_real"]],
                ignore_index=True,
            )
            # Paired differences look for synthetic_share == 0 inside baseline_stage.
            # Real-only scarcity rows already have share 0 and stage scarcity_real,
            # so compare within a rewritten stage label.
            rewritten = stage_runs.copy()
            rewritten.loc[rewritten["stage"] == "scarcity_real", "stage"] = stage
            pieces.append(paired_differences(rewritten, baseline_stage=stage))
    paired = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if len(paired):
        paired.to_csv(roots["statistics"] / "paired_differences.csv", index=False)
    logger.info("Aggregated %s completed runs.", len(runs))


def _figures(train, validation, synthetic_root, seeds, roots, logger) -> None:
    class_table = pd.read_csv(tables_dir() / "class_distribution.csv")
    missing_table = pd.read_csv(tables_dir() / "missing_values.csv")
    figures.plot_class_distribution(class_table, roots["figures"] / "01_target_distribution.png")
    figures.plot_missingness(missing_table, roots["figures"] / "02_missing_values.png")
    model_paths = {name: models_dir() / f"{name}.joblib" for name in CLASSIFIERS}
    figures.plot_baseline_roc(validation, model_paths, roots["figures"] / "03_baseline_roc.png")
    figures.plot_baseline_confusion(validation, model_paths, roots["figures"] / "04_baseline_confusion.png")
    summary = pd.read_csv(roots["tables"] / "metric_summary.csv")
    runs = pd.read_csv(roots["tables"] / "run_metrics.csv")
    paired = pd.read_csv(roots["statistics"] / "paired_differences.csv")
    figures.plot_ratio_curves(summary, roots["figures"] / "05_ratio_curves.png")
    figures.plot_generator_comparison(summary, roots["figures"] / "06_generator_comparison.png")
    figures.plot_differences(paired, roots["figures"] / "07_differences.png")
    figures.plot_seed_variability(runs, roots["figures"] / "08_seed_variability.png")
    synthetic_examples = {}
    for kind in GENERATORS:
        synthetic_examples[kind] = load_synthetic(synthetic_root, kind, seeds[0])
    figures.plot_numeric_overlay(
        modeling_columns(train), synthetic_examples, roots["figures"] / "09_numeric_distributions.png"
    )
    categorical = pd.read_csv(roots["tables"] / "categorical_fidelity.csv")
    categorical = categorical.loc[categorical["feature"] != "income"]
    figures.plot_categorical_tv(categorical, roots["figures"] / "10_categorical_tv.png")
    real_corr = pd.read_csv(roots["tables"] / "real_spearman.csv", index_col=0)
    syn_corrs = {
        kind: pd.read_csv(roots["tables"] / f"spearman_{kind}_seed_{seeds[0]}.csv", index_col=0)
        for kind in GENERATORS
    }
    figures.plot_correlation_heatmaps(real_corr, syn_corrs, roots["figures"] / "11_correlation_heatmaps.png")
    quality = pd.read_csv(roots["tables"] / "quality_summary.csv")
    figures.plot_target_balance(quality, roots["figures"] / "12_target_balance.png")
    utility = (
        runs.loc[(runs["stage"] == "main") & (runs["synthetic_share"] == 1.0)]
        .groupby(["generator", "seed"], as_index=False)["roc_auc"]
        .mean()
        .rename(columns={"roc_auc": "utility_roc_auc"})
    )
    points = quality.merge(utility, on=["generator", "seed"], how="inner")
    points.to_csv(roots["tables"] / "fidelity_utility_points.csv", index=False)
    figures.plot_fidelity_utility(points, roots["figures"] / "13_fidelity_utility.png")
    figures.plot_scarcity(summary, roots["figures"] / "14_scarcity.png")
    figures.plot_research_summary(paired, roots["figures"] / "15_research_summary.png")
    logger.info("Wrote study figures to %s", roots["figures"])


def _freeze(roots, fingerprints, selection, generator_settings, logger) -> None:
    manifest = {
        "stage": "validation_frozen",
        "timestamp": utc_now(),
        "test_metrics_computed": False,
        "final_test_authorized": False,
        "fingerprints": fingerprints,
        "seeds": generator_settings["seeds"],
        "ctgan_epochs": generator_settings["ctgan"]["epochs"],
        "gaussian_copula": {
            "default_distribution": generator_settings["gaussian_copula"]["default_distribution"],
            "enforce_min_max_values": generator_settings["gaussian_copula"]["enforce_min_max_values"],
            "enforce_rounding": generator_settings["gaussian_copula"]["enforce_rounding"],
        },
        "selected_augmentation": selection,
        "analysis_plan": str(project_root() / "docs" / "ANALYSIS_PLAN.md"),
        "note": (
            "Validation results and this manifest were frozen before any "
            "held-out test metric was computed."
        ),
    }
    destination = results_dir() / "frozen"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "validation_freeze.json", manifest)
    logger.info("Wrote freeze manifest.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the validation-stage synthetic-data study.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny cached pipeline check.")
    args = parser.parse_args()
    run_study(smoke=args.smoke)


if __name__ == "__main__":
    main()
