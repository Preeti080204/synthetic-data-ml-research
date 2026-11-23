"""Learned preprocessing that is fit on a training partition only."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    POSITIVE_CLASS,
    TARGET_COLUMN,
)


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the binary target.

    The positive class ``>50K`` is encoded as 1. Metadata columns are excluded.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing feature columns: {missing}")
    if TARGET_COLUMN not in frame.columns:
        raise ValueError("Frame is missing the income target.")
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    target = (frame[TARGET_COLUMN] == POSITIVE_CLASS).astype(int)
    target.name = TARGET_COLUMN
    return features, target


def build_preprocessor(model_name: str, config: dict | None = None) -> ColumnTransformer:
    """Build an unfitted preprocessor for one baseline model.

    Numerical missing values use the training-set median. Categorical missing
    values become an explicit ``Missing`` level so that the missingness pattern
    is retained. One-hot encoding ignores levels that were absent from the
    training partition. Logistic regression also standardizes numerical
    columns. Random Forest and XGBoost share the same imputation and encoding
    and do not standardize, because their splits do not require a common scale.
    """
    if model_name not in {"logistic_regression", "random_forest", "xgboost"}:
        raise ValueError(f"Unknown model name: {model_name}")

    fill_value = "Missing"
    if config is not None:
        fill_value = config["preprocessing"]["categorical_fill_value"]

    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if model_name == "logistic_regression":
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=fill_value)),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("cat", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transformer_by_name(preprocessor: ColumnTransformer, name: str) -> Pipeline:
    """Return a named column pipeline, preferring the fitted clone when present."""
    fitted = getattr(preprocessor, "transformers_", None)
    source = fitted if fitted is not None else preprocessor.transformers
    for transformer_name, transformer, _columns in source:
        if transformer_name == name:
            return transformer
    raise KeyError(name)
