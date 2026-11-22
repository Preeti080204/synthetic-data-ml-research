"""Dataset inspection tables and exploratory figures.

Target-conditional plots are restricted to the training partition. Marginal
descriptions of the pooled sample do not train a model and do not score the
test partition.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from data.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NEGATIVE_CLASS,
    NUMERIC_FEATURES,
    OFFICIAL_CATEGORIES,
    POSITIVE_CLASS,
    TARGET_COLUMN,
)


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200, "font.size": 11})


def inspect_dataset(
    frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> dict:
    _style()
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    class_summary = _class_summary(frame)
    missing = _missing_summary(frame)
    numeric = _numeric_summary(frame)
    frequencies = _categorical_frequencies(frame)
    unexpected = _unexpected_values(frame)
    education = _education_mapping(frame)
    duplicates = _content_duplicates(frame)
    ranges = _numeric_range_audit(frame)

    class_summary.to_csv(tables_dir / "class_distribution.csv", index=False)
    missing.to_csv(tables_dir / "missing_values.csv", index=False)
    numeric.to_csv(tables_dir / "numeric_summary.csv", index=False)
    frequencies.to_csv(tables_dir / "categorical_frequencies.csv", index=False)
    unexpected.to_csv(tables_dir / "unexpected_values.csv", index=False)
    education["mapping"].to_csv(tables_dir / "education_mapping.csv", index=False)

    _plot_class_distribution(class_summary, figures_dir / "class_distribution.png")
    _plot_missing(missing, figures_dir / "missing_values.png")
    _plot_training_numeric(train_frame, figures_dir / "numeric_by_income_training.png")
    _plot_training_correlation(train_frame, figures_dir / "correlation_heatmap_training.png")
    _plot_rate(
        train_frame,
        "education",
        education["order"],
        "Share with income > $50K by education",
        figures_dir / "income_rate_education_training.png",
    )
    _plot_rate(
        train_frame,
        "occupation",
        None,
        "Share with income > $50K by occupation",
        figures_dir / "income_rate_occupation_training.png",
    )
    _plot_rate(
        train_frame,
        "marital-status",
        None,
        "Share with income > $50K by marital status",
        figures_dir / "income_rate_marital_status_training.png",
    )

    both_positive = int(((frame["capital-gain"] > 0) & (frame["capital-loss"] > 0)).sum())
    return {
        "class_distribution": class_summary.to_dict(orient="records"),
        "missing": missing.to_dict(orient="records"),
        "missing_cooccurrence": dict(missing.attrs["cooccurrence"]),
        "numeric_summary": numeric.to_dict(orient="records"),
        "n_unexpected_levels": int((unexpected["issue"] == "unexpected_level").sum())
        if len(unexpected)
        else 0,
        "unexpected_records": unexpected.to_dict(orient="records"),
        "education_one_to_one": bool(education["one_to_one"]),
        "n_education_levels": int(len(education["mapping"])),
        "duplicates": duplicates,
        "numeric_range_violations": ranges,
        "capital_gain_and_loss_both_positive": both_positive,
        "figures": {
            "class_distribution": "results/figures/class_distribution.png",
            "missing_values": "results/figures/missing_values.png",
            "numeric_by_income_training": "results/figures/numeric_by_income_training.png",
            "correlation_heatmap_training": "results/figures/correlation_heatmap_training.png",
            "income_rate_education_training": "results/figures/income_rate_education_training.png",
            "income_rate_occupation_training": "results/figures/income_rate_occupation_training.png",
            "income_rate_marital_status_training": "results/figures/income_rate_marital_status_training.png",
        },
    }


def _class_summary(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame[TARGET_COLUMN].value_counts()
    rows = []
    for label in (NEGATIVE_CLASS, POSITIVE_CLASS):
        count = int(counts.get(label, 0))
        rows.append(
            {
                "income": label,
                "count": count,
                "proportion": count / len(frame),
            }
        )
    return pd.DataFrame(rows)


def _missing_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in FEATURE_COLUMNS:
        count = int(frame[column].isna().sum())
        rows.append(
            {
                "feature": column,
                "missing_count": count,
                "missing_percent": 100.0 * count / len(frame),
            }
        )
    summary = pd.DataFrame(rows)
    work = frame["workclass"].isna()
    occupation = frame["occupation"].isna()
    country = frame["native-country"].isna()
    summary.attrs["cooccurrence"] = {
        "workclass": int(work.sum()),
        "occupation": int(occupation.sum()),
        "native_country": int(country.sum()),
        "workclass_and_occupation": int((work & occupation).sum()),
        "any_feature": int(frame[FEATURE_COLUMNS].isna().any(axis=1).sum()),
    }
    return summary


def _numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in NUMERIC_FEATURES:
        series = frame[column]
        observed = series.dropna()
        quantiles = observed.quantile([0.25, 0.5, 0.75])
        rows.append(
            {
                "feature": column,
                "count": int(observed.shape[0]),
                "missing": int(series.isna().sum()),
                "mean": float(observed.mean()),
                "std": float(observed.std(ddof=1)),
                "min": float(observed.min()),
                "q1": float(quantiles.loc[0.25]),
                "median": float(quantiles.loc[0.5]),
                "q3": float(quantiles.loc[0.75]),
                "max": float(observed.max()),
                "skewness": float(stats.skew(observed.to_numpy(), bias=False)),
            }
        )
    return pd.DataFrame(rows)


def _categorical_frequencies(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in CATEGORICAL_FEATURES:
        filled = frame[column].fillna("Missing")
        counts = filled.value_counts(dropna=False)
        for category, count in counts.items():
            rows.append(
                {
                    "feature": column,
                    "category": category,
                    "count": int(count),
                    "percent": 100.0 * int(count) / len(frame),
                }
            )
    return pd.DataFrame(rows)


def _unexpected_values(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, official in OFFICIAL_CATEGORIES.items():
        observed = set(frame[column].dropna().unique())
        unknown = sorted(observed - official)
        for level in unknown:
            rows.append(
                {
                    "feature": column,
                    "issue": "unexpected_level",
                    "value": level,
                    "count": int((frame[column] == level).sum()),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["feature", "issue", "value", "count"])
    return pd.DataFrame(rows)


def _numeric_range_audit(frame: pd.DataFrame) -> dict[str, int]:
    checks = {
        "age_below_17": int((frame["age"] < 17).sum()),
        "age_above_100": int((frame["age"] > 100).sum()),
        "fnlwgt_at_or_below_1": int((frame["fnlwgt"] <= 1).sum()),
        "education_num_outside_1_16": int(
            ((frame["education-num"] < 1) | (frame["education-num"] > 16)).sum()
        ),
        "capital_gain_negative": int((frame["capital-gain"] < 0).sum()),
        "capital_loss_negative": int((frame["capital-loss"] < 0).sum()),
        "hours_nonpositive": int((frame["hours-per-week"] <= 0).sum()),
        "hours_above_168": int((frame["hours-per-week"] > 168).sum()),
    }
    return checks


def _education_mapping(frame: pd.DataFrame) -> dict:
    mapping = (
        frame.groupby("education", dropna=False)["education-num"]
        .agg(["min", "max", "nunique", "count"])
        .reset_index()
        .sort_values("min")
    )
    one_to_one = bool((mapping["nunique"] == 1).all()) and mapping["education"].is_unique
    order = mapping.sort_values("min")["education"].tolist()
    return {"mapping": mapping, "one_to_one": one_to_one, "order": order}


def _content_duplicates(frame: pd.DataFrame) -> dict:
    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    extra = int(frame.duplicated(columns).sum())
    involved = int(frame.duplicated(columns, keep=False).sum())
    conflicts = frame.groupby(FEATURE_COLUMNS, dropna=False)[TARGET_COLUMN].nunique()
    return {
        "exact_duplicate_extra_rows": extra,
        "exact_duplicate_rows_involved": involved,
        "conflicting_feature_groups": int((conflicts > 1).sum()),
    }


def _plot_class_distribution(summary: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5.6, 4.2))
    colors = ["#4C78A8", "#F58518"]
    bars = axis.bar(summary["income"], summary["count"], color=colors)
    axis.set_ylabel("Observations")
    axis.set_xlabel("Income class")
    axis.set_title("Income class distribution")
    for bar, row in zip(bars, summary.itertuples(index=False)):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{row.count:,}\n({100 * row.proportion:.1f}%)",
            ha="center",
            va="bottom",
        )
    axis.set_ylim(0, summary["count"].max() * 1.18)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_missing(summary: pd.DataFrame, path: Path) -> None:
    present = summary.loc[summary["missing_count"] > 0].sort_values("missing_count")
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.barh(present["feature"], present["missing_percent"], color="#4C78A8")
    axis.set_xlabel("Missing values (%)")
    axis.set_title("Features with missing values")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_training_numeric(train_frame: pd.DataFrame, path: Path) -> None:
    columns = ["age", "hours-per-week", "education-num", "capital-gain"]
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 6.4))
    for axis, column in zip(axes.ravel(), columns):
        if column == "capital-gain":
            share = (
                train_frame.assign(positive_gain=train_frame[column] > 0)
                .groupby(TARGET_COLUMN, observed=True)["positive_gain"]
                .mean()
                .reindex([NEGATIVE_CLASS, POSITIVE_CLASS])
            )
            axis.bar(share.index.astype(str), share.to_numpy(), color=["#4C78A8", "#F58518"])
            axis.set_ylim(0, 1)
            axis.set_ylabel("Share with capital-gain > 0")
            axis.set_xlabel("Income")
            continue
        else:
            sns.boxplot(
                data=train_frame,
                x=TARGET_COLUMN,
                y=column,
                hue=TARGET_COLUMN,
                order=[NEGATIVE_CLASS, POSITIVE_CLASS],
                hue_order=[NEGATIVE_CLASS, POSITIVE_CLASS],
                palette=["#4C78A8", "#F58518"],
                legend=False,
                ax=axis,
            )
            axis.set_ylabel(column)
        axis.set_xlabel("Income")
    figure.suptitle("Training-partition numeric distributions by income", y=1.02)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_training_correlation(train_frame: pd.DataFrame, path: Path) -> None:
    matrix = pd.DataFrame(index=NUMERIC_FEATURES, columns=NUMERIC_FEATURES, dtype=float)
    for left in NUMERIC_FEATURES:
        for right in NUMERIC_FEATURES:
            rho, _ = stats.spearmanr(
                train_frame[left].to_numpy(),
                train_frame[right].to_numpy(),
                nan_policy="omit",
            )
            matrix.loc[left, right] = float(rho)
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    sns.heatmap(
        matrix.astype(float),
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        ax=axis,
    )
    axis.set_title("Training-partition Spearman correlations")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_rate(
    train_frame: pd.DataFrame,
    column: str,
    order: list[str] | None,
    title: str,
    path: Path,
) -> None:
    grouped = (
        train_frame.groupby(column, dropna=False)[TARGET_COLUMN]
        .agg(
            n="size",
            positive_rate=lambda series: float((series == POSITIVE_CLASS).mean()),
        )
        .reset_index()
    )
    grouped[column] = grouped[column].fillna("Missing").astype(str)
    if order is not None:
        grouped["_order"] = grouped[column].map({name: index for index, name in enumerate(order)})
        grouped = grouped.sort_values("_order")
    else:
        grouped = grouped.sort_values("positive_rate")
    figure_height = max(3.8, 0.32 * len(grouped) + 1.2)
    figure, axis = plt.subplots(figsize=(7.4, figure_height))
    axis.barh(grouped[column], grouped["positive_rate"], color="#4C78A8")
    axis.set_xlim(0, 1)
    axis.set_xlabel("Share with income > $50K")
    axis.set_title(title + " (training partition)")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
