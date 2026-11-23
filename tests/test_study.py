"""Design, sampling, aggregation, and test-set guards for the validation study."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.summarize import interval_label, select_augmentation_shares, summarize_runs
from experiments.design import (
    augmentation_synthetic_count,
    compose_mixture,
    mixture_counts,
    stratified_sample,
    study_seeds,
    synthetic_count,
    synthetic_shares,
)
from experiments.tracking import is_complete, write_record
from models.evaluation import evaluate_binary_model
from utils.paths import project_root, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_mixture_counts_match_the_frozen_training_size():
    expected = {
        0.0: (34189, 0),
        0.10: (30770, 3419),
        0.25: (25642, 8547),
        0.50: (17095, 17094),
        0.75: (8547, 25642),
        1.0: (0, 34189),
    }
    for share, counts in expected.items():
        assert mixture_counts(34189, share) == counts
        assert sum(counts) == 34189


def test_python_round_half_to_even_for_the_half_row():
    assert synthetic_count(34189, 0.5) == 17094
    assert int(round(17094.5)) == 17094


def test_study_seeds_and_shares_are_fixed():
    assert study_seeds() == [42, 123, 2024, 3407, 777]
    assert synthetic_shares() == [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]


def test_stratified_sample_does_not_duplicate_or_oversample():
    frame = pd.DataFrame(
        {
            "income": ["<=50K"] * 80 + [">50K"] * 20,
            "age": np.arange(100),
        }
    )
    sample = stratified_sample(frame, 25, seed=42)
    assert len(sample) == 25
    assert sample["age"].is_unique
    with pytest.raises(ValueError, match="oversample"):
        stratified_sample(frame, 101, seed=1)


def test_mixture_keeps_the_requested_size():
    real = pd.DataFrame({"income": ["<=50K"] * 30 + [">50K"] * 10, "age": np.arange(40)})
    synthetic = pd.DataFrame(
        {"income": ["<=50K"] * 30 + [">50K"] * 10, "age": np.arange(100, 140)}
    )
    mixed = compose_mixture(real, synthetic, 20, 20, seed=7)
    assert len(mixed) == 40
    assert mixed["age"].nunique() == 40


def test_augmentation_count_enlarges_the_budget():
    assert augmentation_synthetic_count(100, 0.25) == 33
    enlarged = 100 + augmentation_synthetic_count(100, 0.25)
    assert augmentation_synthetic_count(100, 0.25) / enlarged == pytest.approx(0.25, abs=0.01)
    with pytest.raises(ValueError):
        augmentation_synthetic_count(100, 1.0)


def test_zero_variance_interval_is_not_invented():
    summary = summarize_runs(
        pd.DataFrame(
            {
                "stage": ["main", "main"],
                "generator": ["real", "real"],
                "synthetic_share": [0.0, 0.0],
                "classifier": ["logistic_regression", "logistic_regression"],
                "seed": [1, 2],
                "roc_auc": [0.9, 0.9],
                "f1": [0.5, 0.5],
                "precision": [0.4, 0.4],
                "recall": [0.3, 0.3],
                "balanced_accuracy": [0.6, 0.6],
            }
        )
    )
    assert summary.iloc[0]["roc_auc_std"] == 0
    assert summary.iloc[0]["roc_auc_ci_low"] == summary.iloc[0]["roc_auc_ci_high"]
    assert interval_label(0.0, 0.0) == "no_difference_point"
    assert interval_label(-0.01, 0.02) == "inconclusive"
    assert interval_label(0.01, 0.02) == "higher"


def test_ratio_rule_breaks_ties_and_rejects_synthetic_only_as_augmentation():
    rows = []
    for generator, scores in {
        "gaussian_copula": {0.1: 0.90, 0.25: 0.90, 1.0: 0.95},
        "ctgan": {0.1: 0.80, 0.25: 0.85, 1.0: 0.70},
    }.items():
        for share, score in scores.items():
            for classifier in ("logistic_regression", "random_forest", "xgboost"):
                rows.append(
                    {
                        "generator": generator,
                        "synthetic_share": share,
                        "classifier": classifier,
                        "roc_auc": score,
                    }
                )
    selected = select_augmentation_shares(pd.DataFrame(rows))
    assert selected["generators"]["gaussian_copula"]["augmentation_share"] == 0.1
    assert selected["generators"]["ctgan"]["augmentation_share"] == 0.25
    assert selected["shared_augmentation_share"] == 0.25


def test_validation_guard_rejects_the_test_partition():
    with pytest.raises(RuntimeError, match="test"):
        evaluate_binary_model(None, pd.DataFrame(), pd.Series(dtype=int), partition="test")


def test_phase1_artifacts_keep_their_recorded_hashes():
    manifest = json.loads((ROOT / "docs" / "phase1_artifact_hashes.json").read_text())
    for relative, digest in manifest["files"].items():
        if relative == "reports/phase_1_report.md":
            # Historical hash is retained for provenance; the old report is retired.
            assert not (ROOT / relative).exists()
            continue
        assert sha256_file(ROOT / relative) == digest


def test_checkpoint_round_trip(tmp_path: Path):
    path = tmp_path / "run.json"
    write_record(path, {"status": "completed", "experiment_id": "example"})
    assert is_complete(path)
    write_record(path, {"status": "failed"})
    assert not is_complete(path)


def test_final_report_refuses_without_test_metrics(monkeypatch, tmp_path):
    import reporting.final_report as report

    monkeypatch.setattr(report, "results_dir", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        report.write_final_report()


def test_preflight_refuses_without_authorization(monkeypatch, tmp_path):
    import experiments.data as data
    import experiments.final_guard as guard

    def fail_if_called():
        raise AssertionError("The test frame was loaded during preflight.")

    monkeypatch.setattr(data, "load_test_frame", fail_if_called)
    monkeypatch.setattr(guard, "authorization_path", lambda: tmp_path / "missing")
    errors = guard.preflight()
    assert errors
    assert any("AUTHORIZED" in item for item in errors)


def test_preflight_refuses_when_the_authorization_file_is_absent(monkeypatch, tmp_path):
    import experiments.final_guard as guard

    monkeypatch.setattr(guard, "authorization_path", lambda: tmp_path / "missing")
    errors = guard.preflight()
    assert any("AUTHORIZED" in item for item in errors)


def test_final_test_script_refuses_without_the_flag():
    script = project_root() / "scripts" / "run_final_test.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=project_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "Refusing" in completed.stderr
    assert not (project_root() / "results" / "tables" / "test_metrics.csv").exists()


def test_saved_split_counts_are_unchanged():
    table = pd.read_csv(ROOT / "data" / "processed" / "split_indices.csv")
    counts = table["partition"].value_counts().to_dict()
    assert counts == {"train": 34189, "validation": 7326, "test": 7327}
    assert table.groupby("partition")["source_row_id"].nunique().to_dict() == counts
