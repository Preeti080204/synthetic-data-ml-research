"""Tests for Adult file loading and label cleaning."""

from __future__ import annotations

import pandas as pd

from data.schema import (
    EXPECTED_RAW_SHA256,
    EXPECTED_ROW_COUNTS,
    EXPECTED_ZIP_SHA256,
    FEATURE_COLUMNS,
    NEGATIVE_CLASS,
    NUMERIC_FEATURES,
    POSITIVE_CLASS,
    TARGET_COLUMN,
)
from utils.paths import resolve_zip_path, sha256_file


def test_raw_checksums_are_unchanged(project_root):
    raw = project_root / "data" / "raw"
    for name, digest in EXPECTED_RAW_SHA256.items():
        assert sha256_file(raw / name) == digest
    zip_path = resolve_zip_path()
    if zip_path.is_file():
        assert sha256_file(zip_path) == EXPECTED_ZIP_SHA256


def test_combined_dimensions_and_sources(loaded, adult_df):
    assert loaded.metadata["rows_removed"] == 0
    assert len(adult_df) == EXPECTED_ROW_COUNTS["combined"]
    assert loaded.metadata["n_features"] == 14
    counts = adult_df["source_file"].value_counts().to_dict()
    assert counts["adult.data"] == EXPECTED_ROW_COUNTS["adult.data"]
    assert counts["adult.test"] == EXPECTED_ROW_COUNTS["adult.test"]
    assert adult_df["source_row_id"].is_unique
    assert adult_df.loc[0, "source_row_id"] == "adult.data:1"
    test_rows = adult_df.loc[adult_df["source_file"] == "adult.test"]
    assert test_rows.iloc[0]["source_line"] == 2


def test_official_feature_names(adult_df):
    assert list(adult_df.columns[-15:]) == FEATURE_COLUMNS + [TARGET_COLUMN]


def test_target_normalization(loaded, adult_df):
    assert set(adult_df[TARGET_COLUMN].unique()) == {NEGATIVE_CLASS, POSITIVE_CLASS}
    assert adult_df[TARGET_COLUMN].str.endswith(".").sum() == 0
    assert loaded.metadata["trailing_period_labels"] == EXPECTED_ROW_COUNTS["adult.test"]


def test_missing_value_recognition(adult_df):
    for column in ("workclass", "occupation", "native-country"):
        assert adult_df[column].isna().sum() > 0
        assert (adult_df[column] == "?").sum() == 0
    assert int(adult_df[NUMERIC_FEATURES].isna().sum().sum()) == 0


def test_numeric_dtypes(adult_df):
    for column in NUMERIC_FEATURES:
        assert pd.api.types.is_numeric_dtype(adult_df[column])
