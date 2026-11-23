"""Tests that preserved baseline results and the final report are present."""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytestmark = pytest.mark.artifacts


def test_phase1_outputs_exist(project_root):
    required = [
        "data/processed/adult_combined.csv",
        "data/processed/split_indices.csv",
        "results/tables/validation_metrics.csv",
        "results/tables/hyperparameter_search.csv",
        "results/tables/class_distribution.csv",
        "results/tables/missing_values.csv",
        "results/tables/numeric_summary.csv",
        "results/tables/categorical_frequencies.csv",
        "results/tables/partition_summary.csv",
        "results/tables/validation_confusion_matrices.csv",
        "results/figures/class_distribution.png",
        "results/figures/missing_values.png",
        "results/figures/roc_curves.png",
        "results/figures/confusion_matrices.png",
        "results/figures/model_comparison.png",
        "models/logistic_regression.joblib",
        "models/random_forest.joblib",
        "models/xgboost.joblib",
        "results/test_set_status.json",
        "results/phase1_run.log",
    ]
    for relative in required:
        path = project_root / relative
        assert path.is_file(), relative
        assert path.stat().st_size > 0, relative


def test_validation_metrics_and_report_agree(project_root):
    metrics = pd.read_csv(project_root / "results" / "tables" / "validation_metrics.csv")
    assert set(metrics["model"]) == {"logistic_regression", "random_forest", "xgboost"}
    for row in metrics.itertuples(index=False):
        for value in (row.roc_auc, row.f1, row.precision, row.recall, row.balanced_accuracy):
            assert 0.0 < value < 1.0
    status = json.loads((project_root / "results" / "test_set_status.json").read_text())
    assert status["test_metrics_computed"] is False
    assert status["test_predictions_computed"] is False
    assert status["test_used_for_model_selection"] is False
    assert status["validation_metrics_computed"] is True
    assert not (project_root / "results" / "tables" / "test_metrics.csv").exists()
    log = (project_root / "results" / "phase1_run.log").read_text(encoding="utf-8")
    assert "Test-set performance was not computed." in log


def test_processed_dataset_and_split_files(project_root):
    combined = pd.read_csv(project_root / "data" / "processed" / "adult_combined.csv")
    assert len(combined) == 48842
    splits = pd.read_csv(project_root / "data" / "processed" / "split_indices.csv")
    assert set(splits["partition"]) == {"train", "validation", "test"}
    assert splits["source_row_id"].is_unique
    counts = splits["partition"].value_counts().to_dict()
    assert counts["train"] + counts["validation"] + counts["test"] == 48842
    pdf = project_root / "reports" / "final" / "Preeti_Gorial_Synthetic_Data_Research_Report.pdf"
    assert pdf.stat().st_size > 20_000
