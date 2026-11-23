"""Load the frozen Phase 1 partitions without redrawing them.

The modeling loader returns train and validation only. The test partition is
loaded by a separate function used by the protected final-evaluation script.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd

from data.preprocessing import split_xy
from data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from models.evaluation import evaluate_binary_model
from utils.paths import data_processed_dir, models_dir, sha256_file, tables_dir


def combined_frame() -> pd.DataFrame:
    path = data_processed_dir() / "adult_combined.csv"
    frame = pd.read_csv(path)
    if len(frame) != 48842:
        raise ValueError(f"Expected 48,842 pooled rows, found {len(frame)}.")
    return frame


def split_table() -> pd.DataFrame:
    path = data_processed_dir() / "split_indices.csv"
    table = pd.read_csv(path)
    counts = table["partition"].value_counts().to_dict()
    if counts.get("train") != 34189 or counts.get("validation") != 7326 or counts.get("test") != 7327:
        raise ValueError(f"Frozen split counts changed: {counts}.")
    return table


def _ordered_partition(frame: pd.DataFrame, table: pd.DataFrame, name: str) -> pd.DataFrame:
    ordered = table.loc[table["partition"] == name].sort_values("order_in_partition")
    indexed = frame.set_index("source_row_id", drop=False)
    missing = [row_id for row_id in ordered["source_row_id"] if row_id not in indexed.index]
    if missing:
        raise ValueError(f"{len(missing)} {name} ids are missing from the pooled table.")
    subset = indexed.loc[ordered["source_row_id"].tolist()].reset_index(drop=True)
    if len(subset) != len(ordered):
        raise ValueError(f"{name} partition length does not match the split file.")
    return subset


def load_modeling_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the frozen training and validation frames, in split order.

    The test partition is not loaded.
    """
    frame = combined_frame()
    table = split_table()
    train = _ordered_partition(frame, table, "train")
    validation = _ordered_partition(frame, table, "validation")
    if set(train["source_row_id"]).intersection(validation["source_row_id"]):
        raise ValueError("Train and validation share source-row identifiers.")
    return train, validation


def load_test_frame() -> pd.DataFrame:
    """Load the held-out test frame. Predictive code must not call this."""
    frame = combined_frame()
    table = split_table()
    return _ordered_partition(frame, table, "test")


def verify_phase1_validation(validation: pd.DataFrame, tolerance: float = 1e-6) -> dict:
    """Rescore the saved Phase 1 pipelines on the validation partition.

    The comparison is against ``validation_metrics.csv``. A mismatch aborts the
    study so later experiments are not built on a drifted baseline.
    """
    saved = pd.read_csv(tables_dir() / "validation_metrics.csv")
    features, target = split_xy(validation)
    checks = []
    for model_name in ("logistic_regression", "random_forest", "xgboost"):
        pipeline = joblib.load(models_dir() / f"{model_name}.joblib")
        scores = evaluate_binary_model(
            pipeline, features, target, partition="validation", threshold=0.5
        )
        reference = saved.loc[saved["model"] == model_name].iloc[0]
        deltas = {}
        for metric in ("roc_auc", "f1", "precision", "recall", "balanced_accuracy"):
            deltas[metric] = float(scores[metric] - float(reference[metric]))
            if abs(deltas[metric]) > tolerance:
                raise RuntimeError(
                    f"Phase 1 {model_name} {metric} drifted by {deltas[metric]}."
                )
        checks.append({"model": model_name, "scores": scores, "deltas": deltas})
    return {
        "status": "matched",
        "tolerance": tolerance,
        "models": checks,
        "validation_metrics_sha256": sha256_file(tables_dir() / "validation_metrics.csv"),
    }


def selected_hyperparameters() -> dict:
    path = tables_dir() / "selected_configurations.json"
    with path.open(encoding="utf-8") as handle:
        selected = json.load(handle)
    return selected


def hyperparameters_for_seed(model_name: str, seed: int, selected: dict | None = None) -> dict:
    selected = selected_hyperparameters() if selected is None else selected
    params = dict(selected[model_name]["params"])
    params["random_state"] = int(seed)
    return params


def modeling_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing {missing}.")
    return frame.loc[:, columns].copy()
