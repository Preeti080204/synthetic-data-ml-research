"""Tests for baseline fitting and the Phase 1 evaluation guard."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from data.preprocessing import split_xy
from models.baseline import fit_baselines
from models.evaluation import evaluate_binary_model


def _tiny_config(config: dict) -> dict:
    updated = {
        "seed": config["seed"],
        "decision_threshold": config["decision_threshold"],
        "selection_metric": config["selection_metric"],
        "tie_breaker_metric": config["tie_breaker_metric"],
        "preprocessing": config["preprocessing"],
        "models": {
            "logistic_regression": {
                "display_name": "Logistic Regression",
                "fixed": {"solver": "lbfgs", "max_iter": 300, "random_state": 42},
                "grid": [{"C": 1.0, "class_weight": None}],
            },
            "random_forest": {
                "display_name": "Random Forest",
                "fixed": {
                    "n_estimators": 12,
                    "max_features": "sqrt",
                    "bootstrap": True,
                    "n_jobs": 1,
                    "random_state": 42,
                },
                "grid": [{"max_depth": 4, "min_samples_leaf": 5}],
            },
            "xgboost": {
                "display_name": "XGBoost",
                "fixed": {
                    "n_estimators": 12,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "tree_method": "hist",
                    "n_jobs": 1,
                    "random_state": 42,
                    "verbosity": 0,
                },
                "grid": [{"max_depth": 2, "learning_rate": 0.1}],
            },
        },
    }
    return updated


def _stratified_subset(frame, n_rows: int, seed: int):
    subset, _ = train_test_split(
        frame,
        train_size=n_rows,
        stratify=frame["income"],
        random_state=seed,
    )
    return subset.reset_index(drop=True)


def test_fit_baselines_does_not_accept_a_test_partition():
    parameters = set(inspect.signature(fit_baselines).parameters)
    assert "test" not in parameters
    assert "test_frame" not in parameters
    assert "x_test" not in parameters


def test_evaluation_refuses_the_test_partition(adult_df, splits):
    from data.preprocessing import build_preprocessor
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    train = _stratified_subset(adult_df.iloc[splits["train"]], 400, 42)
    x_train, y_train = split_xy(train)
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor("logistic_regression")),
            ("classifier", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )
    pipeline.fit(x_train, y_train)
    with pytest.raises(RuntimeError, match="Phase 1 refuses"):
        evaluate_binary_model(pipeline, x_train, y_train, partition="test")


def test_three_models_train_and_return_valid_validation_metrics(adult_df, splits, config, tmp_path):
    train = _stratified_subset(adult_df.iloc[splits["train"]], 1200, 42)
    validation = _stratified_subset(adult_df.iloc[splits["validation"]], 400, 42)
    result = fit_baselines(
        train,
        validation,
        _tiny_config(config),
        tmp_path / "models",
        tmp_path / "tables",
    )
    assert set(result["models"]) == {"logistic_regression", "random_forest", "xgboost"}
    assert result["best_model"] in result["models"]
    metrics_path = tmp_path / "tables" / "validation_metrics.csv"
    assert metrics_path.is_file()
    for name, item in result["models"].items():
        for metric in ("roc_auc", "f1", "precision", "recall", "balanced_accuracy"):
            value = item["metrics"][metric]
            assert np.isfinite(value), name
            assert 0.0 <= value <= 1.0, name
        assert item["metrics"]["roc_auc"] > 0.65
        assert (tmp_path / "models" / f"{name}.joblib").is_file()
        assert item["scored_on"] == "validation"
        assert item["refit_on_validation"] is False
