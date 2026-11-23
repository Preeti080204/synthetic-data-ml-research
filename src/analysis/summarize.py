"""Means, paired differences, and interval labels for the validation study."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from experiments.design import METRICS

PRIMARY = "roc_auc"


def _student_interval(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    n_obs = int(values.size)
    mean = float(np.mean(values)) if n_obs else float("nan")
    if n_obs < 2:
        return {
            "n": n_obs,
            "mean": mean,
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "interval": "undefined",
        }
    std = float(np.std(values, ddof=1))
    if std == 0:
        return {
            "n": n_obs,
            "mean": mean,
            "std": 0.0,
            "ci_low": mean,
            "ci_high": mean,
            "interval": "degenerate_zero_variance",
        }
    half = float(stats.t.ppf(0.975, n_obs - 1) * std / np.sqrt(n_obs))
    return {
        "n": n_obs,
        "mean": mean,
        "std": std,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "interval": "student_t_95",
    }


def interval_label(low: float, high: float) -> str:
    if np.isnan(low) or np.isnan(high):
        return "undefined"
    if low == high:
        if low > 0:
            return "higher_point"
        if low < 0:
            return "lower_point"
        return "no_difference_point"
    if low > 0:
        return "higher"
    if high < 0:
        return "lower"
    return "inconclusive"


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample standard deviation of each metric."""
    group_columns = [
        column
        for column in ("stage", "generator", "synthetic_share", "classifier", "budget", "comparison")
        if column in runs.columns
    ]
    rows = []
    for keys, group in runs.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n"] = int(len(group))
        for metric in METRICS:
            summary = _student_interval(group[metric].to_numpy(dtype=float))
            row[f"{metric}_mean"] = summary["mean"]
            row[f"{metric}_std"] = summary["std"]
            row[f"{metric}_ci_low"] = summary["ci_low"]
            row[f"{metric}_ci_high"] = summary["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(runs: pd.DataFrame, baseline_stage: str = "main") -> pd.DataFrame:
    """Paired seed differences against the matched real-only run."""
    rows = []
    group_columns = ["stage", "generator", "synthetic_share", "classifier"]
    if "budget" in runs.columns:
        group_columns.append("budget")
    if "comparison" in runs.columns:
        group_columns.append("comparison")
    baseline = runs.loc[(runs["synthetic_share"] == 0) & (runs["stage"] == baseline_stage)].copy()
    candidates = runs.loc[runs["synthetic_share"] > 0].copy()
    for keys, group in candidates.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(group_columns, keys))
        matched_base = baseline
        if "budget" in identity and pd.notna(identity["budget"]):
            matched_base = baseline.loc[np.isclose(baseline["budget"].astype(float), float(identity["budget"]))]
        matched_base = matched_base.loc[matched_base["classifier"] == identity["classifier"]]
        for metric in METRICS:
            diffs = []
            seeds = []
            for seed, seed_rows in group.groupby("seed"):
                base_rows = matched_base.loc[matched_base["seed"] == seed]
                if len(seed_rows) != 1 or len(base_rows) != 1:
                    continue
                diffs.append(float(seed_rows.iloc[0][metric] - base_rows.iloc[0][metric]))
                seeds.append(int(seed))
            summary = _student_interval(np.asarray(diffs, dtype=float))
            p_value = None
            if summary["n"] >= 2 and summary["std"] not in (0, 0.0) and summary["std"] == summary["std"]:
                left = group.drop_duplicates("seed").set_index("seed").loc[seeds, metric]
                right = matched_base.drop_duplicates("seed").set_index("seed").loc[seeds, metric]
                p_value = float(stats.ttest_rel(left, right).pvalue)
            row = dict(identity)
            row.update(
                {
                    "metric": metric,
                    "seeds": seeds,
                    "mean_difference": summary["mean"],
                    "std_difference": summary["std"],
                    "ci_low": summary["ci_low"],
                    "ci_high": summary["ci_high"],
                    "interval_type": summary["interval"],
                    "direction": interval_label(summary["ci_low"], summary["ci_high"]),
                    "paired_t_pvalue": p_value,
                    "pvalue_role": "descriptive_only",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def select_augmentation_shares(main_runs: pd.DataFrame) -> dict:
    """Apply the pre-specified ratio rule to main-experiment validation ROC-AUC."""
    positive = main_runs.loc[main_runs["synthetic_share"] > 0].copy()
    by_generator = {}
    for generator, group in positive.groupby("generator"):
        means = group.groupby("synthetic_share")[PRIMARY].mean().sort_index()
        winner = _best_share(means)
        below = means.loc[means.index < 1]
        augmentation = _best_share(below) if winner == 1.0 else winner
        by_generator[str(generator)] = {
            "mean_roc_auc_by_share": {str(share): float(value) for share, value in means.items()},
            "best_share_including_synthetic_only": float(winner),
            "augmentation_share": float(augmentation),
        }
    below_one = positive.loc[positive["synthetic_share"] < 1]
    shared_means = below_one.groupby("synthetic_share")[PRIMARY].mean().sort_index()
    return {
        "rule": (
            "Highest mean validation ROC-AUC across classifiers and seeds. "
            "Ties take the smaller share. A share of 1.0 is not an augmentation ratio."
        ),
        "generators": by_generator,
        "shared_augmentation_share": float(_best_share(shared_means)),
        "shared_mean_roc_auc_by_share": {str(share): float(value) for share, value in shared_means.items()},
    }


def _best_share(means: pd.Series) -> float:
    if means.empty:
        raise ValueError("The ratio rule received no candidate shares.")
    best_value = float(means.max())
    winners = [float(share) for share, value in means.items() if np.isclose(value, best_value)]
    return float(min(winners))
