"""Baseline model specifications, small validation search, and fitting."""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from data.preprocessing import build_preprocessor, split_xy
from models.evaluation import evaluate_binary_model

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def build_estimator(model_name: str, params: dict):
    if model_name == "logistic_regression":
        return LogisticRegression(**params)
    if model_name == "random_forest":
        return RandomForestClassifier(**params)
    if model_name == "xgboost":
        return XGBClassifier(**params)
    raise ValueError(f"Unknown model name: {model_name}")


def build_model_pipeline(model_name: str, params: dict, config: dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(model_name, config)),
            ("classifier", build_estimator(model_name, params)),
        ]
    )


def _candidate_params(model_cfg: dict) -> list[dict]:
    fixed = dict(model_cfg["fixed"])
    candidates = []
    for overrides in model_cfg["grid"]:
        merged = dict(fixed)
        merged.update(overrides)
        candidates.append(merged)
    return candidates


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def fit_baselines(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: dict,
    model_dir: Path,
    tables_dir: Path,
) -> dict:
    """Fit every declared configuration on the training partition only.

    Configuration choice uses validation ROC-AUC, with validation F1 as the
    tie-breaker. The selected pipeline is the training-set fit. It is not
    refit on the validation rows. The function does not receive or score a
    test partition.
    """
    x_train, y_train = split_xy(train_frame)
    x_val, y_val = split_xy(validation_frame)
    threshold = float(config["decision_threshold"])
    model_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    search_rows: list[dict] = []
    selected: dict[str, dict] = {}

    for model_name, model_cfg in config["models"].items():
        best: dict | None = None
        search_started = time.perf_counter()
        for candidate_index, params in enumerate(_candidate_params(model_cfg), start=1):
            pipeline = build_model_pipeline(model_name, params, config)
            caught: list[warnings.WarningMessage] = []
            started = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pipeline.fit(x_train, y_train)
                error = ""
                scores = evaluate_binary_model(
                    pipeline,
                    x_val,
                    y_val,
                    partition="validation",
                    threshold=threshold,
                )
            except Exception as exc:  # recorded so a failed candidate is not silent
                error = f"{type(exc).__name__}: {exc}"
                scores = {
                    "roc_auc": float("nan"),
                    "f1": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "balanced_accuracy": float("nan"),
                }
            elapsed = time.perf_counter() - started
            warning_text = [
                f"{item.category.__name__}: {item.message}" for item in caught
            ]
            row = {
                "model": model_name,
                "display_name": model_cfg.get("display_name", DISPLAY_NAMES[model_name]),
                "candidate": candidate_index,
                "params": _json_ready(params),
                "train_seconds": elapsed,
                "roc_auc": scores["roc_auc"],
                "f1": scores["f1"],
                "precision": scores["precision"],
                "recall": scores["recall"],
                "balanced_accuracy": scores["balanced_accuracy"],
                "warnings": warning_text,
                "error": error,
            }
            search_rows.append(row)
            if error:
                continue
            if best is None or _is_better(scores, best["metrics"]):
                best = {
                    "pipeline": pipeline,
                    "params": params,
                    "metrics": scores,
                    "train_seconds": elapsed,
                    "warnings": warning_text,
                    "candidate": candidate_index,
                }
        search_seconds = time.perf_counter() - search_started
        if best is None:
            raise RuntimeError(f"Every candidate failed for {model_name}.")

        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(best["pipeline"], model_path)
        n_transformed = int(best["pipeline"].named_steps["preprocess"].transform(x_train.iloc[:1]).shape[1])
        selected[model_name] = {
            "name": model_name,
            "display_name": model_cfg.get("display_name", DISPLAY_NAMES[model_name]),
            "params": _json_ready(best["params"]),
            "metrics": best["metrics"],
            "train_seconds": best["train_seconds"],
            "search_seconds": search_seconds,
            "n_candidates": len(model_cfg["grid"]),
            "selected_candidate": best["candidate"],
            "n_transformed_features": n_transformed,
            "warnings": best["warnings"],
            "model_path": str(model_path),
            "decision_threshold": threshold,
            "positive_class": ">50K",
            "trained_on": "train",
            "scored_on": "validation",
            "refit_on_validation": False,
        }

    search_frame = pd.DataFrame(search_rows)
    search_frame.to_csv(tables_dir / "hyperparameter_search.csv", index=False)
    metrics_frame = pd.DataFrame(
        [
            {
                "model": item["name"],
                "display_name": item["display_name"],
                "roc_auc": item["metrics"]["roc_auc"],
                "f1": item["metrics"]["f1"],
                "precision": item["metrics"]["precision"],
                "recall": item["metrics"]["recall"],
                "balanced_accuracy": item["metrics"]["balanced_accuracy"],
                "train_seconds": item["train_seconds"],
                "search_seconds": item["search_seconds"],
                "n_transformed_features": item["n_transformed_features"],
                "selected_candidate": item["selected_candidate"],
                "params": item["params"],
            }
            for item in selected.values()
        ]
    )
    metrics_frame.to_csv(tables_dir / "validation_metrics.csv", index=False)

    ranking = sorted(
        selected.values(),
        key=lambda item: (item["metrics"]["roc_auc"], item["metrics"]["f1"]),
        reverse=True,
    )
    return {
        "models": selected,
        "ranking": [item["name"] for item in ranking],
        "best_model": ranking[0]["name"],
        "selection_metric": config["selection_metric"],
        "tie_breaker_metric": config["tie_breaker_metric"],
        "decision_threshold": threshold,
        "search_rows": search_rows,
    }


def _is_better(candidate: dict, incumbent: dict) -> bool:
    if candidate["roc_auc"] > incumbent["roc_auc"] + 1e-12:
        return True
    if abs(candidate["roc_auc"] - incumbent["roc_auc"]) <= 1e-12:
        return candidate["f1"] > incumbent["f1"] + 1e-12
    return False
