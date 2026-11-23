"""A tiny generator fit. Excluded from the fast suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.schema import FEATURE_COLUMNS, NEGATIVE_CLASS, POSITIVE_CLASS, TARGET_COLUMN
from generators.tabular import generate_cached

pytestmark = pytest.mark.integration


def _toy_frame() -> pd.DataFrame:
    rows = []
    for index in range(80):
        label = POSITIVE_CLASS if index % 4 == 0 else NEGATIVE_CLASS
        rows.append(
            {
                "age": 20 + (index % 40),
                "workclass": "Private" if index % 2 == 0 else "Missing",
                "fnlwgt": 10000 + index,
                "education": "HS-grad",
                "education-num": 9,
                "marital-status": "Never-married",
                "occupation": "Sales",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Female" if index % 2 else "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
                TARGET_COLUMN: label,
            }
        )
    return pd.DataFrame(rows)


def test_tiny_generators_emit_valid_targets(tmp_path: Path):
    frame = _toy_frame()
    for kind in ("gaussian_copula", "ctgan"):
        result = generate_cached(
            frame,
            kind,
            seed=42,
            root=tmp_path,
            n_rows=80,
            overrides={"epochs": 1, "batch_size": 40, "verbose": False},
        )
        assert result["status"] == "completed", result.get("error")
        synthetic = pd.read_csv(tmp_path / kind / "seed_42" / "synthetic.csv")
        assert list(synthetic.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
        assert set(synthetic[TARGET_COLUMN]).issubset({POSITIVE_CLASS, NEGATIVE_CLASS})
        assert len(synthetic) == 80
        second = generate_cached(frame, kind, seed=42, root=tmp_path, n_rows=80)
        assert second["cache"] == "hit"
