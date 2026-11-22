"""Counts, seeds, and sampling for the validation study.

Integer counts use Python ``round`` (half to even). The real count is the
complement of the synthetic count. Real rows are never duplicated to fill a
shortfall.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from utils.paths import project_root

CLASSIFIERS = ("logistic_regression", "random_forest", "xgboost")
GENERATORS = ("gaussian_copula", "ctgan")
METRICS = ("roc_auc", "f1", "precision", "recall", "balanced_accuracy")


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


def generator_config() -> dict:
    return load_yaml(project_root() / "configs" / "generators.yaml")


def experiment_config() -> dict:
    return load_yaml(project_root() / "configs" / "experiments.yaml")


def reporting_config() -> dict:
    return load_yaml(project_root() / "configs" / "reporting.yaml")


def study_seeds(config: dict | None = None) -> list[int]:
    config = experiment_config() if config is None else config
    seeds = [int(seed) for seed in config["seeds"]]
    if seeds != [42, 123, 2024, 3407, 777]:
        raise ValueError(f"The study seeds are fixed. Received {seeds}.")
    return seeds


def synthetic_shares(config: dict | None = None) -> list[float]:
    config = experiment_config() if config is None else config
    shares = [float(share) for share in config["synthetic_shares"]]
    if shares != [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]:
        raise ValueError(f"The synthetic shares are fixed. Received {shares}.")
    return shares


def share_token(share: float) -> str:
    return f"{float(share):.2f}"


def synthetic_count(n_rows: int, share: float) -> int:
    """Return the synthetic row count for a fixed-size mixture.

    A share of 0 returns 0. A share of 1 returns ``n_rows``. Other shares use
    Python's round-half-to-even rule.
    """
    if n_rows < 0:
        raise ValueError("n_rows must be non-negative.")
    share = float(share)
    if share <= 0:
        return 0
    if share >= 1:
        return int(n_rows)
    return int(round(share * n_rows))


def mixture_counts(n_rows: int, share: float) -> tuple[int, int]:
    """Return ``(n_real, n_synthetic)`` summing to ``n_rows``."""
    n_synthetic = synthetic_count(n_rows, share)
    n_real = int(n_rows) - n_synthetic
    if n_real < 0:
        raise ValueError("Synthetic count exceeds the training size.")
    return n_real, n_synthetic


def budget_count(n_train: int, budget: float) -> int:
    return int(round(float(budget) * int(n_train)))


def augmentation_synthetic_count(n_budget: int, share: float) -> int:
    """Synthetic rows to add while keeping every real budget row.

    The synthetic fraction of the enlarged set is ``share``. A share of 0 or 1
    is rejected because this helper is only for genuine augmentation.
    """
    share = float(share)
    if not 0 < share < 1:
        raise ValueError("Augmentation share must be strictly between 0 and 1.")
    return int(round(n_budget * share / (1.0 - share)))


def stratified_sample(frame: pd.DataFrame, n_rows: int, seed: int, target: str = "income") -> pd.DataFrame:
    """Draw ``n_rows`` without replacement, preserving class shares as far as possible.

    The function raises if the request exceeds the frame. It does not duplicate
    rows. If one class cannot supply its allocation, the unused rows of the
    other class fill the gap, and only when those unused rows exist.
    """
    if target not in frame.columns:
        raise ValueError(f"Stratified sample requires column {target!r}.")
    n_rows = int(n_rows)
    if n_rows < 0:
        raise ValueError("n_rows must be non-negative.")
    if n_rows > len(frame):
        raise ValueError(
            f"Refusing to oversample: requested {n_rows} rows from {len(frame)}."
        )
    if n_rows == 0:
        return frame.iloc[0:0].copy()
    if n_rows == len(frame):
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    labels = frame[target].astype(str)
    counts = labels.value_counts()
    raw = {label: n_rows * (int(count) / len(frame)) for label, count in counts.items()}
    allocated = {label: int(np.floor(value)) for label, value in raw.items()}
    remainder = n_rows - sum(allocated.values())
    order = sorted(
        raw,
        key=lambda label: (raw[label] - allocated[label], str(label)),
        reverse=True,
    )
    for label in order:
        if remainder <= 0:
            break
        allocated[label] += 1
        remainder -= 1

    pieces: list[pd.DataFrame] = []
    used: list[pd.Index] = []
    shortfall = 0
    for label, count in allocated.items():
        pool = frame.loc[labels == label]
        take = min(int(count), len(pool))
        shortfall += int(count) - take
        if take:
            piece = pool.sample(n=take, random_state=seed)
            pieces.append(piece)
            used.append(piece.index)
    if shortfall:
        used_index = used[0]
        for extra in used[1:]:
            used_index = used_index.append(extra)
        leftover = frame.drop(index=used_index)
        if len(leftover) < shortfall:
            raise ValueError(
                f"Cannot draw {n_rows} rows without replacement "
                f"(shortfall {shortfall - len(leftover)})."
            )
        pieces.append(leftover.sample(n=shortfall, random_state=seed))
    combined = pd.concat(pieces, axis=0)
    if len(combined) != n_rows:
        raise RuntimeError("Stratified sample returned an unexpected row count.")
    if combined.index.has_duplicates:
        raise RuntimeError("Stratified sample duplicated a source index.")
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def compose_mixture(
    real_frame: pd.DataFrame,
    synthetic_frame: pd.DataFrame,
    n_real: int,
    n_synthetic: int,
    seed: int,
) -> pd.DataFrame:
    """Sample, stack, and shuffle a fixed-size training mixture."""
    real_part = stratified_sample(real_frame, n_real, seed)
    if n_synthetic == 0:
        synthetic_part = synthetic_frame.iloc[0:0].copy()
    else:
        synthetic_part = stratified_sample(synthetic_frame, n_synthetic, seed)
    combined = pd.concat([real_part, synthetic_part], ignore_index=True)
    if len(combined) != n_real + n_synthetic:
        raise RuntimeError("Mixture length does not equal the requested size.")
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
