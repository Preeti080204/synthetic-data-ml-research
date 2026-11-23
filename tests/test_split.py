"""Tests for the stratified 70/15/15 partition."""

from __future__ import annotations

import numpy as np

from data.schema import POSITIVE_CLASS, TARGET_COLUMN
from data.split_data import audit_partitions, make_splits


def test_split_is_deterministic(adult_df, config):
    first = make_splits(adult_df, config)
    second = make_splits(adult_df, config)
    for name in ("train", "validation", "test"):
        assert np.array_equal(first[name], second[name])


def test_partition_proportions_and_coverage(adult_df, splits):
    n_rows = len(adult_df)
    train, validation, test = (len(splits[name]) for name in ("train", "validation", "test"))
    assert train + validation + test == n_rows
    assert abs(train / n_rows - 0.70) < 0.002
    assert abs(validation / n_rows - 0.15) < 0.002
    assert abs(test / n_rows - 0.15) < 0.002
    covered = np.concatenate([splits["train"], splits["validation"], splits["test"]])
    assert len(np.unique(covered)) == n_rows


def test_no_source_row_overlap(adult_df, splits):
    ids = adult_df["source_row_id"].to_numpy()
    sets = {name: set(ids[idx].tolist()) for name, idx in splits.items()}
    assert sets["train"].isdisjoint(sets["validation"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["validation"].isdisjoint(sets["test"])


def test_stratified_class_balance(adult_df, splits):
    overall = float((adult_df[TARGET_COLUMN] == POSITIVE_CLASS).mean())
    for name, idx in splits.items():
        share = float((adult_df.loc[idx, TARGET_COLUMN] == POSITIVE_CLASS).mean())
        assert abs(share - overall) < 0.005, name


def test_audit_reports_zero_identity_overlap(adult_df, splits):
    audit = audit_partitions(adult_df, splits)
    assert audit["overlap_source_row_id"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    assert audit["test_predictive_metrics_computed"] is False
