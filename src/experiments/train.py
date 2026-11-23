"""Fit the Phase 1 classifiers on a study training set and score validation only."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from data.preprocessing import split_xy
from data.schema import NEGATIVE_CLASS, POSITIVE_CLASS
from experiments.design import (
    CLASSIFIERS,
    compose_mixture,
    experiment_config,
    mixture_counts,
    share_token,
)
from experiments.tracking import is_complete, read_json, utc_now, write_record
from models.baseline import build_model_pipeline
from models.evaluation import evaluate_binary_model
from utils.paths import load_config


def experiment_id(
    stage: str,
    generator: str,
    share: float,
    classifier: str,
    seed: int,
    budget: float | None = None,
) -> str:
    parts = [stage, generator, f"share-{share_token(share)}"]
    if budget is not None:
        parts.insert(2, f"budget-{budget:.2f}")
    parts.extend([classifier, f"seed-{int(seed)}"])
    return "__".join(parts)


def run_classifier(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    classifier: str,
    params: dict,
    record_path: Path,
    context: dict,
) -> dict:
    """Fit one pipeline and write its validation metrics.

    A completed record is returned unchanged. The test partition is not an
    argument of this function.
    """
    if is_complete(record_path):
        cached = read_json(record_path)
        from experiments.cache import classifier_cache_hit

        if classifier_cache_hit(cached, context, params):
            cached["cache"] = "hit"
            cached["cache_validation"] = "matched stored fingerprints and hyperparameters"
            return cached

    phase1 = load_config()
    features, target = split_xy(train_frame)
    val_features, val_target = split_xy(validation_frame)
    labels = set(train_frame["income"].astype(str).unique())
    if not labels.issubset({POSITIVE_CLASS, NEGATIVE_CLASS}):
        raise ValueError(f"Training labels are not binary Adult labels: {labels}.")
    started = time.perf_counter()
    pipeline = build_model_pipeline(classifier, params, phase1)
    pipeline.fit(features, target)
    metrics = evaluate_binary_model(
        pipeline,
        val_features,
        val_target,
        partition="validation",
        threshold=float(experiment_config()["decision_threshold"]),
    )
    elapsed = time.perf_counter() - started
    payload = dict(context)
    payload.update(
        {
            "status": "completed",
            "timestamp": utc_now(),
            "classifier": classifier,
            "hyperparameters": params,
            "training_seconds": elapsed,
            "metrics": metrics,
            "n_train": int(len(train_frame)),
            "positive_rate_train": float((target == 1).mean()),
            "scored_on": "validation",
            "test_metrics_computed": False,
            "cache": "miss",
        }
    )
    write_record(record_path, payload)
    return payload


def main_training_frame(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame | None,
    share: float,
    seed: int,
) -> pd.DataFrame:
    n_real, n_synthetic = mixture_counts(len(real_train), share)
    if n_synthetic == 0:
        return compose_mixture(real_train, real_train.iloc[0:0].copy(), n_real, 0, seed)
    if synthetic is None:
        raise ValueError("A positive synthetic share requires a synthetic frame.")
    return compose_mixture(real_train, synthetic, n_real, n_synthetic, seed)


def classifier_names() -> tuple[str, ...]:
    return CLASSIFIERS
