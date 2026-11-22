"""Integrity checks for the saved held-out evaluation.

The checks read existing files. They do not load the test frame and they do
not refit a model.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.cache import file_sha256
from experiments.design import (
    CLASSIFIERS,
    GENERATORS,
    augmentation_synthetic_count,
    budget_count,
    mixture_counts,
    study_seeds,
    synthetic_count,
)
from experiments.final_guard import authorization_path, freeze_path
from utils.paths import project_root, results_dir

N_TRAIN = 34189


class IntegrityError(RuntimeError):
    pass


def audit_saved_test() -> dict:
    """Return the audit record. Raise if a required check fails."""
    failures: list[str] = []
    freeze = json.loads(freeze_path().read_text(encoding="utf-8"))
    authorization = _authorization_record()
    reproducibility = json.loads(
        (results_dir() / "final_evaluation" / "reproducibility.json").read_text(encoding="utf-8")
    )
    runs = pd.read_csv(results_dir() / "final_evaluation" / "test_run_metrics.csv")
    if "classifier" not in runs.columns:
        failures.append("The test metric table has no classifier column.")
    if len(runs) != 390:
        failures.append(f"Expected 390 test runs, found {len(runs)}.")
    if not (runs["partition"] == "test").all():
        failures.append("A saved test row is not labelled as the test partition.")
    if not (runs["status"] == "completed").all():
        failures.append("A saved test row is not completed.")
    for column in ("roc_auc", "f1", "precision", "recall", "balanced_accuracy"):
        if runs[column].isna().any():
            failures.append(f"Missing values in {column}.")
    if set(int(seed) for seed in runs["seed"].unique()) != set(study_seeds()):
        failures.append("The test seeds are not the five frozen seeds.")
    if runs["test_based_selection"].astype(str).str.lower().isin(["true", "1"]).any():
        failures.append("A test row records test-based selection.")
    grouped = runs.groupby(
        ["stage", "generator", "synthetic_share", "classifier", "budget"], dropna=False
    )["seed"].nunique()
    if not (grouped == 5).all():
        failures.append("A test configuration does not have five seeds.")
    failures.extend(_main_counts(runs))
    failures.extend(_scarcity_counts(runs))
    failures.extend(_synthetic_hashes(runs))
    freeze_hash = file_sha256(freeze_path())
    if reproducibility.get("freeze_sha256") != freeze_hash:
        failures.append("The evaluation record does not match the current freeze hash.")
    if authorization.get("freeze_sha256") != freeze_hash:
        failures.append("The authorization record does not match the current freeze hash.")
    if authorization.get("authorized") is not True:
        failures.append("The authorization record is not marked authorized.")
    if reproducibility.get("runs") != 390:
        failures.append("The reproducibility record does not say 390 runs.")
    if reproducibility.get("test_partition_rows") != 7327:
        failures.append("The recorded test partition size is not 7,327.")
    record = {
        "runs": int(len(runs)),
        "failures": failures,
        "freeze_sha256": freeze_hash,
        "authorization_reviewer": authorization.get("reviewer"),
        "authorization_statement": authorization.get("statement"),
        "evaluation_timestamp": reproducibility.get("timestamp"),
        "source_tree_fingerprint": reproducibility.get("source_tree_fingerprint"),
        "git_commit": reproducibility.get("git_commit"),
        "test_based_selection": False,
        "freeze_still_says_test_metrics_computed": freeze.get("test_metrics_computed"),
    }
    if failures:
        raise IntegrityError("; ".join(failures))
    destination = results_dir() / "final_evaluation" / "integrity_audit.json"
    destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _authorization_record() -> dict:
    """Read the local authorization file, or the published status if it is absent.

    The raw authorization file is not part of the public repository. The
    post-evaluation status is the sanitized record a clone can check.
    """
    path = authorization_path()
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    status_path = project_root() / "results" / "frozen" / "post_evaluation_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    held = status["held_out_evaluation"]
    return {
        "authorized": held.get("status") == "completed_and_verified",
        "reviewer": held.get("authorization_reviewer_as_stored"),
        "statement": "Recorded in results/frozen/post_evaluation_status.json.",
        "freeze_sha256": status.get("validation_freeze_sha256"),
    }


def _main_counts(runs: pd.DataFrame) -> list[str]:
    failures = []
    main = runs.loc[runs["stage"] == "main"]
    if len(main) != 165:
        failures.append(f"Expected 165 main test runs, found {len(main)}.")
    for row in main[["generator", "synthetic_share", "n_real", "n_synthetic"]].drop_duplicates().itertuples(index=False):
        expected_real, expected_syn = mixture_counts(N_TRAIN, float(row.synthetic_share))
        if (int(row.n_real), int(row.n_synthetic)) != (expected_real, expected_syn):
            failures.append(
                f"Main counts for {row.generator} at {row.synthetic_share} are "
                f"{int(row.n_real)}/{int(row.n_synthetic)}, expected {expected_real}/{expected_syn}."
            )
    shares = sorted(float(share) for share in main["synthetic_share"].unique())
    if shares != [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]:
        failures.append(f"Main shares are {shares}.")
    generators = set(main.loc[main["synthetic_share"] > 0, "generator"])
    if generators != set(GENERATORS):
        failures.append(f"Main generators are {generators}.")
    if set(main["classifier"]) != set(CLASSIFIERS):
        failures.append("Main classifiers are incomplete.")
    return failures


def _scarcity_counts(runs: pd.DataFrame) -> list[str]:
    failures = []
    real = runs.loc[runs["stage"] == "scarcity_real"]
    aug = runs.loc[runs["stage"] == "scarcity_augmentation"]
    repl = runs.loc[runs["stage"] == "scarcity_replacement"]
    if (len(real), len(aug), len(repl)) != (45, 90, 90):
        failures.append(f"Scarcity counts are real {len(real)}, augmentation {len(aug)}, replacement {len(repl)}.")
    budgets = {0.25: budget_count(N_TRAIN, 0.25), 0.50: budget_count(N_TRAIN, 0.50), 0.75: budget_count(N_TRAIN, 0.75)}
    for budget, n_budget in budgets.items():
        aug_rows = aug.loc[np.isclose(aug["budget"], budget)]
        repl_rows = repl.loc[np.isclose(repl["budget"], budget)]
        real_rows = real.loc[np.isclose(real["budget"], budget)]
        expected_added = augmentation_synthetic_count(n_budget, 0.10)
        expected_fixed = synthetic_count(n_budget, 0.10)
        if not (real_rows["n_real"] == n_budget).all() or not (real_rows["n_synthetic"] == 0).all():
            failures.append(f"Scarcity real-only budget {budget} has the wrong size.")
        if not np.isclose(aug_rows["synthetic_share"], 0.10).all():
            failures.append(f"Augmentation budget {budget} is not the frozen 10 percent share.")
        if not (aug_rows["n_real"] == n_budget).all() or not (aug_rows["n_synthetic"] == expected_added).all():
            failures.append(f"Augmentation budget {budget} row counts do not match the frozen rule.")
        if not (repl_rows["n_synthetic"] == expected_fixed).all():
            failures.append(f"Replacement budget {budget} synthetic count is wrong.")
        if not (repl_rows["n_real"] == n_budget - expected_fixed).all():
            failures.append(f"Replacement budget {budget} real count is wrong.")
    return failures


def _synthetic_hashes(runs: pd.DataFrame) -> list[str]:
    failures = []
    root = project_root() / "data" / "synthetic"
    seen: dict[str, str] = {}
    for row in runs.itertuples(index=False):
        if row.generator == "real":
            continue
        if pd.isna(row.budget):
            path = root / "main" / row.generator / f"seed_{int(row.seed)}" / "synthetic.csv"
        else:
            path = (
                root / "scarcity" / f"budget_{float(row.budget):.2f}" / row.generator / f"seed_{int(row.seed)}" / "synthetic.csv"
            )
        key = str(path)
        if key not in seen:
            if not path.is_file():
                failures.append(f"Missing synthetic file {path}.")
                continue
            seen[key] = file_sha256(path)
        if str(row.synthetic_sha256) != seen[key]:
            failures.append(f"Synthetic hash mismatch for {row.experiment_id}.")
    return failures
