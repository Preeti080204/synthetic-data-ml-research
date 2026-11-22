"""Compare a synthetic training file with the real training partition.

These checks do not use the validation or test partitions and they are not a
privacy guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from data.schema import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN

CONTENT_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def _as_content(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in CONTENT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Quality frame is missing {missing}.")
    content = frame.loc[:, CONTENT_COLUMNS].copy()
    for column in CATEGORICAL_FEATURES + [TARGET_COLUMN]:
        content[column] = content[column].where(content[column].notna(), "Missing").astype(str)
    return content


def total_variation(real: pd.Series, synthetic: pd.Series) -> float:
    real_p = real.value_counts(normalize=True)
    syn_p = synthetic.value_counts(normalize=True)
    levels = sorted(set(real_p.index).union(syn_p.index))
    distance = 0.5 * sum(abs(float(real_p.get(level, 0.0)) - float(syn_p.get(level, 0.0))) for level in levels)
    return float(distance)


def cramers_v(left: pd.Series, right: pd.Series) -> float:
    table = pd.crosstab(left, right)
    if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n_obs = float(table.to_numpy().sum())
    degrees = min(table.shape[0] - 1, table.shape[1] - 1)
    if n_obs == 0 or degrees == 0:
        return float("nan")
    return float(np.sqrt(chi2 / (n_obs * degrees)))


def numeric_report(real: pd.DataFrame, synthetic: pd.DataFrame) -> list[dict]:
    rows = []
    for column in NUMERIC_FEATURES:
        real_values = pd.to_numeric(real[column], errors="coerce").dropna().to_numpy(dtype=float)
        syn_values = pd.to_numeric(synthetic[column], errors="coerce").dropna().to_numpy(dtype=float)
        quantiles = (0.1, 0.5, 0.9)
        row = {
            "feature": column,
            "ks_statistic": float(stats.ks_2samp(real_values, syn_values).statistic),
            "wasserstein": float(stats.wasserstein_distance(real_values, syn_values)),
            "real_mean": float(np.mean(real_values)),
            "synthetic_mean": float(np.mean(syn_values)),
            "real_std": float(np.std(real_values, ddof=1)),
            "synthetic_std": float(np.std(syn_values, ddof=1)),
        }
        for quantile in quantiles:
            row[f"real_q{int(quantile * 100)}"] = float(np.quantile(real_values, quantile))
            row[f"synthetic_q{int(quantile * 100)}"] = float(np.quantile(syn_values, quantile))
        rows.append(row)
    return rows


def categorical_report(real: pd.DataFrame, synthetic: pd.DataFrame) -> list[dict]:
    rows = []
    for column in list(CATEGORICAL_FEATURES) + [TARGET_COLUMN]:
        real_counts = real[column].value_counts(normalize=True)
        syn_counts = synthetic[column].value_counts(normalize=True)
        rare = real_counts[real_counts < 0.01]
        rows.append(
            {
                "feature": column,
                "total_variation": total_variation(real[column], synthetic[column]),
                "real_missing_rate": float((real[column] == "Missing").mean()),
                "synthetic_missing_rate": float((synthetic[column] == "Missing").mean()),
                "real_n_levels": int(real[column].nunique()),
                "synthetic_n_levels": int(synthetic[column].nunique()),
                "unseen_levels": int(len(set(syn_counts.index) - set(real_counts.index))),
                "rare_real_levels": int(len(rare)),
                "mean_abs_rare_frequency_gap": float(
                    np.mean([abs(float(real_counts[level]) - float(syn_counts.get(level, 0.0))) for level in rare.index])
                )
                if len(rare)
                else 0.0,
            }
        )
    return rows


def dependence_report(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    real_corr = real[list(NUMERIC_FEATURES)].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    syn_corr = synthetic[list(NUMERIC_FEATURES)].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    difference = real_corr.to_numpy(dtype=float) - syn_corr.to_numpy(dtype=float)
    associations = []
    for column in CATEGORICAL_FEATURES:
        associations.append(
            {
                "feature": column,
                "real_cramers_v": cramers_v(real[column], real[TARGET_COLUMN]),
                "synthetic_cramers_v": cramers_v(synthetic[column], synthetic[TARGET_COLUMN]),
            }
        )
    for row in associations:
        row["absolute_difference"] = abs(row["real_cramers_v"] - row["synthetic_cramers_v"])
    return {
        "spearman_frobenius": float(np.linalg.norm(difference, ord="fro")),
        "real_correlation": real_corr,
        "synthetic_correlation": syn_corr,
        "target_associations": associations,
    }


def duplicate_report(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    real_keys = set(map(tuple, real[CONTENT_COLUMNS].astype(str).to_numpy()))
    synthetic_keys = list(map(tuple, synthetic[CONTENT_COLUMNS].astype(str).to_numpy()))
    exact_matches = int(sum(key in real_keys for key in synthetic_keys))
    internal = int(synthetic.duplicated(CONTENT_COLUMNS).sum())
    near = _near_duplicate_rate(real, synthetic)
    return {
        "synthetic_rows": int(len(synthetic)),
        "exact_internal_duplicate_rows": internal,
        "exact_internal_duplicate_rate": internal / len(synthetic),
        "exact_training_matches": exact_matches,
        "exact_training_match_rate": exact_matches / len(synthetic),
        "near_duplicate": near,
        "privacy_guarantee": False,
    }


def _encoded(real: pd.DataFrame, synthetic: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    categorical = real[list(CATEGORICAL_FEATURES) + [TARGET_COLUMN]].astype(str)
    syn_categorical = synthetic[list(CATEGORICAL_FEATURES) + [TARGET_COLUMN]].astype(str)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(categorical)
    scaler = StandardScaler()
    real_numeric = scaler.fit_transform(real[list(NUMERIC_FEATURES)].apply(pd.to_numeric, errors="coerce").fillna(0.0))
    syn_numeric = scaler.transform(
        synthetic[list(NUMERIC_FEATURES)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )
    real_matrix = np.hstack([real_numeric, encoder.transform(categorical)])
    syn_matrix = np.hstack([syn_numeric, encoder.transform(syn_categorical)])
    return real_matrix, syn_matrix


def _near_duplicate_rate(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    """Euclidean near-duplicates on a documented subsample.

    The threshold is the first percentile of training-to-training nearest
    neighbor distances. A synthetic row closer than that threshold is counted
    as a near duplicate of the training subsample. This is a memorization
    indicator, not a privacy test.
    """
    real_matrix, syn_matrix = _encoded(real, synthetic)
    reference_n = min(8000, len(real_matrix))
    query_n = min(2000, len(syn_matrix), len(real_matrix))
    rng = np.random.default_rng(0)
    reference_idx = rng.choice(len(real_matrix), size=reference_n, replace=False)
    reference = real_matrix[reference_idx]
    train_query_idx = rng.choice(len(real_matrix), size=query_n, replace=False)
    train_query = real_matrix[train_query_idx]
    syn_query_idx = rng.choice(len(syn_matrix), size=query_n, replace=False)
    syn_query = syn_matrix[syn_query_idx]
    neighbors = NearestNeighbors(n_neighbors=2, metric="euclidean")
    neighbors.fit(reference)
    train_distances, _ = neighbors.kneighbors(train_query)
    # The closest neighbor can be the query row itself when it was also drawn
    # into the reference. Use the second neighbor for the training baseline.
    baseline = train_distances[:, 1]
    threshold = float(np.quantile(baseline, 0.01))
    syn_neighbors = NearestNeighbors(n_neighbors=1, metric="euclidean")
    syn_neighbors.fit(reference)
    syn_distances, _ = syn_neighbors.kneighbors(syn_query)
    syn_distances = syn_distances[:, 0]
    return {
        "encoding": "ordinal categories and standard-scaled numerics, both fit on the real training partition",
        "distance": "euclidean",
        "reference_rows": int(reference_n),
        "query_rows": int(query_n),
        "threshold_rule": "first percentile of training-to-training second-neighbor distances",
        "threshold": threshold,
        "synthetic_near_duplicate_rate": float(np.mean(syn_distances <= threshold)),
        "median_synthetic_distance": float(np.median(syn_distances)),
        "median_training_distance": float(np.median(baseline)),
        "privacy_guarantee": False,
    }


def evaluate_quality(real_frame: pd.DataFrame, synthetic_frame: pd.DataFrame, generator: str, seed: int) -> dict:
    real = _as_content(real_frame)
    synthetic = _as_content(synthetic_frame)
    numeric = numeric_report(real, synthetic)
    categorical = categorical_report(real, synthetic)
    dependence = dependence_report(real, synthetic)
    duplicates = duplicate_report(real, synthetic)
    target_real = float((real[TARGET_COLUMN] == ">50K").mean())
    target_syn = float((synthetic[TARGET_COLUMN] == ">50K").mean())
    mean_tv = float(np.mean([row["total_variation"] for row in categorical if row["feature"] != TARGET_COLUMN]))
    mean_ks = float(np.mean([row["ks_statistic"] for row in numeric]))
    return {
        "generator": generator,
        "seed": int(seed),
        "numeric": numeric,
        "categorical": categorical,
        "spearman_frobenius": dependence["spearman_frobenius"],
        "target_associations": dependence["target_associations"],
        "real_correlation": dependence["real_correlation"],
        "synthetic_correlation": dependence["synthetic_correlation"],
        "duplicates": duplicates,
        "real_positive_rate": target_real,
        "synthetic_positive_rate": target_syn,
        "mean_categorical_tv": mean_tv,
        "mean_ks": mean_ks,
        "test_partition_used": False,
    }
