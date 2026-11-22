"""Tests that learned preprocessing is fit on training rows only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.preprocessing import build_preprocessor, split_xy, transformer_by_name
from data.schema import NUMERIC_FEATURES


def _copy_with_forced_weight(frame: pd.DataFrame, fill: float) -> pd.DataFrame:
    copied = frame.copy()
    copied["fnlwgt"] = fill
    copied.loc[copied.index[0], "fnlwgt"] = np.nan
    return copied


def test_scaler_is_applied_only_for_logistic_regression(config):
    logistic = build_preprocessor("logistic_regression", config)
    forest = build_preprocessor("random_forest", config)
    boosting = build_preprocessor("xgboost", config)
    assert "scaler" in transformer_by_name(logistic, "num").named_steps
    assert "scaler" not in transformer_by_name(forest, "num").named_steps
    assert "scaler" not in transformer_by_name(boosting, "num").named_steps
    for model_name in ("logistic_regression", "random_forest", "xgboost"):
        preprocessor = build_preprocessor(model_name, config)
        imputer = transformer_by_name(preprocessor, "num").named_steps["imputer"]
        categorical = transformer_by_name(preprocessor, "cat")
        assert imputer.strategy == "median"
        assert categorical.named_steps["imputer"].strategy == "constant"
        assert categorical.named_steps["imputer"].fill_value == "Missing"
        assert categorical.named_steps["encoder"].handle_unknown == "ignore"


def test_imputer_and_scaler_do_not_learn_from_validation(adult_df, splits, config):
    # Equal-sized slices make the pooled median differ from the training median.
    train = _copy_with_forced_weight(adult_df.iloc[splits["train"][:40]], 10.0)
    validation = _copy_with_forced_weight(adult_df.iloc[splits["validation"][:40]], 1000.0)
    x_train, _ = split_xy(train)
    x_val, _ = split_xy(validation)
    preprocessor = build_preprocessor("logistic_regression", config)
    preprocessor.fit(x_train)
    numeric = transformer_by_name(preprocessor, "num")
    weight_index = NUMERIC_FEATURES.index("fnlwgt")
    learned_median = float(numeric.named_steps["imputer"].statistics_[weight_index])
    learned_mean = float(numeric.named_steps["scaler"].mean_[weight_index])
    assert learned_median == float(np.nanmedian(x_train["fnlwgt"]))
    combined_median = float(np.nanmedian(pd.concat([x_train["fnlwgt"], x_val["fnlwgt"]])))
    assert learned_median != combined_median

    leaked = build_preprocessor("logistic_regression", config)
    leaked.fit(pd.concat([x_train, x_val], ignore_index=True))
    leaked_median = float(
        transformer_by_name(leaked, "num").named_steps["imputer"].statistics_[weight_index]
    )
    assert leaked_median == combined_median
    assert leaked_median != learned_median

    transformed = preprocessor.transform(x_val)
    assert transformed.shape[0] == len(x_val)
    assert float(numeric.named_steps["imputer"].statistics_[weight_index]) == learned_median
    assert float(numeric.named_steps["scaler"].mean_[weight_index]) == learned_mean


def test_transformed_width_is_stable_and_unknown_levels_are_ignored(adult_df, splits, config):
    x_train, _ = split_xy(adult_df.iloc[splits["train"]])
    x_val, _ = split_xy(adult_df.iloc[splits["validation"]])
    widths = []
    for model_name in ("logistic_regression", "random_forest", "xgboost"):
        preprocessor = build_preprocessor(model_name, config)
        preprocessor.fit(x_train)
        train_matrix = preprocessor.transform(x_train.iloc[:20])
        val_matrix = preprocessor.transform(x_val.iloc[:20])
        assert train_matrix.shape[1] == val_matrix.shape[1]
        widths.append(train_matrix.shape[1])
        novel = x_val.iloc[:1].copy()
        novel["native-country"] = "Atlantis"
        encoded = preprocessor.transform(novel)
        assert encoded.shape[1] == train_matrix.shape[1]
    assert len(set(widths)) == 1
