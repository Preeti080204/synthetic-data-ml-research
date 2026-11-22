"""Reproducible 70/15/15 stratified partition of the pooled Adult data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.schema import (
    FEATURE_COLUMNS,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    TARGET_COLUMN,
)


def make_splits(frame: pd.DataFrame, config: dict) -> dict[str, np.ndarray]:
    """Stratify the pooled sample into train, validation, and test indices.

    The split is applied to positional indices of ``frame``. Membership and
    order are fully determined by the seed and by the row order of ``frame``.
    Scikit-learn converts a fractional test size with ``ceil``, so the
    validation and test counts can differ by one observation.
    """
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        raise ValueError("frame must use a contiguous RangeIndex before splitting.")

    seed = int(config["seed"])
    train_fraction = float(config["split"]["train"])
    validation_fraction = float(config["split"]["validation"])
    test_fraction = float(config["split"]["test"])
    total = train_fraction + validation_fraction + test_fraction
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1. Received {total}.")

    y = frame[TARGET_COLUMN].to_numpy()
    indices = np.arange(len(frame))
    holdout_fraction = validation_fraction + test_fraction
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=holdout_fraction,
        stratify=y,
        random_state=seed,
    )
    relative_test = test_fraction / holdout_fraction
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test,
        stratify=y[temp_idx],
        random_state=seed,
    )
    splits = {
        "train": np.asarray(train_idx, dtype=int),
        "validation": np.asarray(val_idx, dtype=int),
        "test": np.asarray(test_idx, dtype=int),
    }
    _assert_partition_integrity(frame, splits)
    return splits


def _assert_partition_integrity(frame: pd.DataFrame, splits: dict[str, np.ndarray]) -> None:
    covered = np.concatenate([splits["train"], splits["validation"], splits["test"]])
    if len(covered) != len(frame) or len(np.unique(covered)) != len(frame):
        raise ValueError("Partitions do not form a partition of the row index.")
    ids = frame["source_row_id"].to_numpy()
    id_sets = {name: set(ids[idx].tolist()) for name, idx in splits.items()}
    if id_sets["train"] & id_sets["validation"]:
        raise ValueError("Train and validation share source_row_id values.")
    if id_sets["train"] & id_sets["test"]:
        raise ValueError("Train and test share source_row_id values.")
    if id_sets["validation"] & id_sets["test"]:
        raise ValueError("Validation and test share source_row_id values.")


def save_splits(frame: pd.DataFrame, splits: dict[str, np.ndarray], path: Path) -> None:
    """Persist partition membership in split order so later phases can reuse it."""
    pieces = []
    for name, idx in splits.items():
        piece = pd.DataFrame(
            {
                "partition": name,
                "order_in_partition": np.arange(len(idx)),
                "row_index": idx,
                "source_row_id": frame.loc[idx, "source_row_id"].to_numpy(),
                "source_file": frame.loc[idx, "source_file"].to_numpy(),
                "income": frame.loc[idx, TARGET_COLUMN].to_numpy(),
            }
        )
        pieces.append(piece)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(pieces, ignore_index=True).to_csv(path, index=False)


def load_saved_splits(path: Path) -> dict[str, np.ndarray]:
    table = pd.read_csv(path)
    splits: dict[str, np.ndarray] = {}
    for name, group in table.groupby("partition", sort=False):
        ordered = group.sort_values("order_in_partition")
        splits[str(name)] = ordered["row_index"].to_numpy(dtype=int)
    expected = {"train", "validation", "test"}
    if set(splits) != expected:
        raise ValueError(f"Saved split is missing partitions: {expected - set(splits)}")
    return splits


def class_counts(labels: pd.Series) -> dict[str, int]:
    counts = labels.value_counts()
    return {
        NEGATIVE_CLASS: int(counts.get(NEGATIVE_CLASS, 0)),
        POSITIVE_CLASS: int(counts.get(POSITIVE_CLASS, 0)),
    }


def audit_partitions(frame: pd.DataFrame, splits: dict[str, np.ndarray]) -> dict:
    """Check sizes, class balance, source-row identity, and duplicate content.

    This audit reads partition membership and feature values. It does not fit a
    model and does not compute predictive performance on any partition.
    """
    n_rows = len(frame)
    ids = frame["source_row_id"].to_numpy()
    id_sets = {name: set(ids[idx].tolist()) for name, idx in splits.items()}
    overlap = {
        "train_validation": len(id_sets["train"] & id_sets["validation"]),
        "train_test": len(id_sets["train"] & id_sets["test"]),
        "validation_test": len(id_sets["validation"] & id_sets["test"]),
    }

    sizes = {name: int(len(idx)) for name, idx in splits.items()}
    proportions = {name: sizes[name] / n_rows for name in sizes}
    label_counts = {}
    label_proportions = {}
    for name, idx in splits.items():
        counts = class_counts(frame.loc[idx, TARGET_COLUMN])
        label_counts[name] = counts
        total = sum(counts.values())
        label_proportions[name] = {label: counts[label] / total for label in counts}

    partition = pd.Series(index=frame.index, dtype="object")
    for name, idx in splits.items():
        partition.iloc[idx] = name

    content_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    content = frame.loc[:, content_columns].copy()
    content["partition"] = partition.to_numpy()

    exact_duplicate_extra_rows = int(content.duplicated(content_columns).sum())
    exact_duplicate_rows_involved = int(
        content.duplicated(content_columns, keep=False).sum()
    )
    grouped = content.groupby(content_columns, dropna=False)["partition"]
    partition_nunique = grouped.nunique()
    group_sizes = grouped.size()
    spanning = partition_nunique > 1
    cross_partition_duplicate_groups = int(spanning.sum())
    cross_partition_duplicate_rows = int(group_sizes.loc[spanning].sum()) if spanning.any() else 0

    feature_label_nunique = content.groupby(FEATURE_COLUMNS, dropna=False)[TARGET_COLUMN].nunique()
    conflicting_feature_groups = int((feature_label_nunique > 1).sum())

    overall_counts = class_counts(frame[TARGET_COLUMN])
    return {
        "n_rows": int(n_rows),
        "sizes": sizes,
        "proportions": proportions,
        "overlap_source_row_id": overlap,
        "class_counts": label_counts,
        "class_proportions": label_proportions,
        "overall_class_counts": overall_counts,
        "overall_class_proportions": {
            label: overall_counts[label] / n_rows for label in overall_counts
        },
        "exact_duplicate_extra_rows": exact_duplicate_extra_rows,
        "exact_duplicate_rows_involved": exact_duplicate_rows_involved,
        "cross_partition_duplicate_groups": cross_partition_duplicate_groups,
        "cross_partition_duplicate_rows": cross_partition_duplicate_rows,
        "conflicting_feature_groups": conflicting_feature_groups,
        "test_predictive_metrics_computed": False,
    }


def save_partition_tables(audit: dict, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    size_rows = []
    for name in ("train", "validation", "test"):
        size_rows.append(
            {
                "partition": name,
                "n_rows": audit["sizes"][name],
                "proportion": audit["proportions"][name],
                "n_le_50k": audit["class_counts"][name][NEGATIVE_CLASS],
                "n_gt_50k": audit["class_counts"][name][POSITIVE_CLASS],
                "proportion_le_50k": audit["class_proportions"][name][NEGATIVE_CLASS],
                "proportion_gt_50k": audit["class_proportions"][name][POSITIVE_CLASS],
            }
        )
    pd.DataFrame(size_rows).to_csv(tables_dir / "partition_summary.csv", index=False)

    duplicate_summary = {
        "exact_duplicate_extra_rows": audit["exact_duplicate_extra_rows"],
        "exact_duplicate_rows_involved": audit["exact_duplicate_rows_involved"],
        "cross_partition_duplicate_groups": audit["cross_partition_duplicate_groups"],
        "cross_partition_duplicate_rows": audit["cross_partition_duplicate_rows"],
        "conflicting_feature_groups": audit["conflicting_feature_groups"],
        "overlap_source_row_id": audit["overlap_source_row_id"],
        "note": (
            "Source-row overlap counts shared source_row_id values. "
            "Cross-partition duplicate counts are identical feature and label "
            "contents that occur in more than one partition. Those identical "
            "contents are distinct source rows."
        ),
    }
    from utils.paths import write_json

    write_json(tables_dir / "partition_audit.json", duplicate_summary)
