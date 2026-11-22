"""Refuse final-test scoring unless every authorization check passes.

This module does not import the held-out test frame.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.cache import file_sha256
from experiments.design import GENERATORS, study_seeds
from utils.paths import project_root, results_dir


def authorization_path() -> Path:
    return results_dir() / "frozen" / "FINAL_TEST_AUTHORIZED"


def freeze_path() -> Path:
    return results_dir() / "frozen" / "validation_freeze.json"


def preflight() -> list[str]:
    """Return refusal reasons. An empty list means scoring may load the test set."""
    errors: list[str] = []
    freeze_file = freeze_path()
    if not freeze_file.is_file():
        return ["The validation freeze manifest is missing."]
    freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
    authorization = authorization_path()
    if not authorization.is_file():
        errors.append("results/frozen/FINAL_TEST_AUTHORIZED is missing.")
        return errors
    try:
        payload = json.loads(authorization.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        errors.append("The authorization file is not valid JSON.")
        return errors
    if payload.get("authorized") is not True:
        errors.append("The authorization record does not set authorized to true.")
    if not str(payload.get("reviewer", "")).strip():
        errors.append("The authorization record has no reviewer.")
    statement = str(payload.get("statement", ""))
    if "held-out test" not in statement:
        errors.append("The authorization statement does not name the held-out test.")
    if payload.get("freeze_sha256") != file_sha256(freeze_file):
        errors.append("The authorization record does not match the freeze manifest hash.")
    fingerprints = freeze.get("fingerprints", {})
    expected = {
        "adult_combined": project_root() / "data" / "processed" / "adult_combined.csv",
        "split_indices": project_root() / "data" / "processed" / "split_indices.csv",
        "selected_configurations": project_root() / "results" / "tables" / "selected_configurations.json",
        "experiments_yaml": project_root() / "configs" / "experiments.yaml",
        "generators_yaml": project_root() / "configs" / "generators.yaml",
        "analysis_plan": project_root() / "docs" / "ANALYSIS_PLAN.md",
    }
    for key, path in expected.items():
        if not path.is_file() or file_sha256(path) != fingerprints.get(key):
            errors.append(f"Frozen fingerprint mismatch: {key}.")
    if freeze.get("test_metrics_computed") is not False:
        errors.append("The freeze manifest does not say that test metrics are uncomputed.")
    seeds = freeze.get("seeds")
    if seeds != study_seeds():
        errors.append("The freeze seeds do not match the study seeds.")
    selection = freeze.get("selected_augmentation", {})
    for kind in GENERATORS:
        share = selection.get("generators", {}).get(kind, {}).get("augmentation_share")
        if share is None:
            errors.append(f"The freeze has no augmentation share for {kind}.")
    errors.extend(_synthetic_files(seeds or []))
    return errors


def _synthetic_files(seeds: list) -> list[str]:
    errors = []
    root = project_root() / "data" / "synthetic"
    for kind in GENERATORS:
        for seed in seeds:
            path = root / "main" / kind / f"seed_{seed}" / "synthetic.csv"
            meta = path.with_name("generation.json")
            if not path.is_file() or not _completed(meta):
                errors.append(f"Missing completed main synthetic data for {kind} seed {seed}.")
    for budget in (0.25, 0.50, 0.75):
        for kind in GENERATORS:
            for seed in seeds:
                folder = root / "scarcity" / f"budget_{budget:.2f}" / kind / f"seed_{seed}"
                if not (folder / "synthetic.csv").is_file() or not _completed(folder / "generation.json"):
                    errors.append(f"Missing scarcity synthetic data for budget {budget} {kind} seed {seed}.")
    return errors


def _completed(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("status") == "completed"
