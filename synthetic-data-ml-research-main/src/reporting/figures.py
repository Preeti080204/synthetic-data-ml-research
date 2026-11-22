"""Publication figures for the validation study.

Axes that represent a rate or a ROC curve run from 0 to 1. Difference plots
include zero. Uncertainty bars are sample standard deviations across seeds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, confusion_matrix

from data.preprocessing import split_xy
from models.baseline import DISPLAY_NAMES
from models.evaluation import positive_scores

NAVY = "#1B3A4B"
TEAL = "#1F7A8C"
MUTED = "#5C7C99"
GREY = "#6B7280"
LIGHT = "#E6EEF2"
MODEL_COLORS = {
    "logistic_regression": NAVY,
    "random_forest": TEAL,
    "xgboost": MUTED,
}
GENERATOR_COLORS = {"gaussian_copula": NAVY, "ctgan": TEAL, "real": GREY}
GENERATOR_LABELS = {
    "gaussian_copula": "Gaussian copula",
    "ctgan": "CTGAN",
    "real": "Real only",
}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": NAVY,
            "axes.labelcolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "text.color": "#2D343A",
            "axes.titleweight": "regular",
            "font.size": 11,
        }
    )


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def plot_class_distribution(table: pd.DataFrame, path: Path) -> None:
    _style()
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    colors = [NAVY, TEAL]
    axis.bar(table["income"], table["proportion"], color=colors[: len(table)], width=0.62)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Share of pooled observations")
    axis.set_xlabel("Income class")
    axis.set_title("Target-class distribution")
    for label, proportion, count in zip(table["income"], table["proportion"], table["count"]):
        axis.text(label, proportion + 0.03, f"{proportion:.1%}\n(n={int(count):,})", ha="center", color=NAVY)
    save_figure(figure, path)


def plot_missingness(table: pd.DataFrame, path: Path) -> None:
    _style()
    present = table.loc[table["missing_count"] > 0].sort_values("missing_percent")
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    axis.barh(present["feature"], present["missing_percent"], color=TEAL)
    axis.set_xlabel("Missing values (% of pooled observations)")
    axis.set_title("Features with missing values")
    save_figure(figure, path)


def plot_baseline_roc(validation: pd.DataFrame, model_paths: dict[str, Path], path: Path) -> None:
    _style()
    features, target = split_xy(validation)
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    for name, model_path in model_paths.items():
        import joblib

        pipeline = joblib.load(model_path)
        scores = positive_scores(pipeline, features)
        RocCurveDisplay.from_predictions(
            target,
            scores,
            name=DISPLAY_NAMES[name],
            ax=axis,
            curve_kwargs={"color": MODEL_COLORS[name]},
        )
    axis.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1, label="Chance")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title("Validation ROC curves, Phase 1 models")
    axis.legend(frameon=False, loc="lower right")
    save_figure(figure, path)


def plot_baseline_confusion(validation: pd.DataFrame, model_paths: dict[str, Path], path: Path) -> None:
    _style()
    features, target = split_xy(validation)
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    labels = ["<=50K", ">50K"]
    for axis, (name, model_path) in zip(axes, model_paths.items()):
        import joblib

        pipeline = joblib.load(model_path)
        scores = positive_scores(pipeline, features)
        predicted = (scores >= 0.5).astype(int)
        matrix = confusion_matrix(target, predicted, labels=[0, 1])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap=sns.light_palette(TEAL, as_cmap=True),
            cbar=False,
            ax=axis,
            xticklabels=labels,
            yticklabels=labels,
        )
        axis.set_title(DISPLAY_NAMES[name])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Observed")
    figure.suptitle("Validation confusion matrices at threshold 0.5", color=NAVY)
    save_figure(figure, path)


def plot_ratio_curves(summary: pd.DataFrame, path: Path) -> None:
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    main = summary.loc[summary["stage"] == "main"].copy()
    for axis, generator in zip(axes, ("gaussian_copula", "ctgan")):
        subset = main.loc[(main["generator"] == generator) | (main["synthetic_share"] == 0)]
        for classifier, color in MODEL_COLORS.items():
            rows = subset.loc[subset["classifier"] == classifier].sort_values("synthetic_share")
            if rows.empty:
                continue
            axis.errorbar(
                rows["synthetic_share"],
                rows["roc_auc_mean"],
                yerr=rows["roc_auc_std"].fillna(0),
                color=color,
                marker="o",
                linewidth=1.5,
                capsize=3,
                label=DISPLAY_NAMES[classifier],
            )
        axis.set_ylim(0, 1)
        axis.set_xlabel("Synthetic share")
        axis.set_title(GENERATOR_LABELS[generator])
    axes[0].set_ylabel("Validation ROC-AUC")
    axes[1].legend(frameon=False, loc="lower right")
    figure.suptitle("Validation ROC-AUC by synthetic share", color=NAVY)
    save_figure(figure, path)


def plot_generator_comparison(summary: pd.DataFrame, path: Path) -> None:
    _style()
    main = summary.loc[(summary["stage"] == "main") & (summary["synthetic_share"] > 0)]
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    shares = sorted(main["synthetic_share"].unique())
    width = 0.35
    positions = np.arange(len(shares))
    for offset, generator in ((-width / 2, "gaussian_copula"), (width / 2, "ctgan")):
        means = []
        stds = []
        for share in shares:
            rows = main.loc[(main["generator"] == generator) & (main["synthetic_share"] == share)]
            means.append(float(rows["roc_auc_mean"].mean()) if len(rows) else np.nan)
            stds.append(float(rows["roc_auc_std"].mean()) if len(rows) else 0)
        axis.bar(
            positions + offset,
            means,
            width=width,
            yerr=stds,
            color=GENERATOR_COLORS[generator],
            label=GENERATOR_LABELS[generator],
            capsize=3,
        )
    axis.set_xticks(positions, [f"{share:.0%}" for share in shares])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean validation ROC-AUC across classifiers")
    axis.set_xlabel("Synthetic share")
    axis.set_title("Gaussian copula and CTGAN")
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    save_figure(figure, path)


def plot_differences(paired: pd.DataFrame, path: Path) -> None:
    _style()
    roc = paired.loc[(paired["stage"] == "main") & (paired["metric"] == "roc_auc")].copy()
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)
    for axis, generator in zip(axes, ("gaussian_copula", "ctgan")):
        subset = roc.loc[roc["generator"] == generator]
        for classifier, color in MODEL_COLORS.items():
            rows = subset.loc[subset["classifier"] == classifier].sort_values("synthetic_share")
            low = rows["ci_low"].fillna(rows["mean_difference"])
            high = rows["ci_high"].fillna(rows["mean_difference"])
            axis.errorbar(
                rows["synthetic_share"],
                rows["mean_difference"],
                yerr=[rows["mean_difference"] - low, high - rows["mean_difference"]],
                color=color,
                marker="o",
                capsize=3,
                label=DISPLAY_NAMES[classifier],
            )
        axis.axhline(0, color=GREY, linewidth=1)
        axis.set_xlabel("Synthetic share")
        axis.set_title(GENERATOR_LABELS[generator])
    axes[0].set_ylabel("Paired difference in validation ROC-AUC")
    axes[1].legend(frameon=False)
    figure.suptitle("Difference from the matched-seed real-only baseline", color=NAVY)
    save_figure(figure, path)


def plot_seed_variability(runs: pd.DataFrame, path: Path) -> None:
    _style()
    main = runs.loc[runs["stage"] == "main"].copy()
    figure, axis = plt.subplots(figsize=(8.4, 4.6))
    plot_frame = main.loc[main["synthetic_share"].isin([0.0, 0.5, 1.0])].copy()
    plot_frame["condition"] = plot_frame.apply(
        lambda row: "Real only"
        if row["synthetic_share"] == 0
        else f"{GENERATOR_LABELS[row['generator']]} {row['synthetic_share']:.0%}",
        axis=1,
    )
    plot_frame["model"] = plot_frame["classifier"].map(DISPLAY_NAMES)
    order = [
        "Real only",
        "Gaussian copula 50%",
        "Gaussian copula 100%",
        "CTGAN 50%",
        "CTGAN 100%",
    ]
    present = [item for item in order if item in set(plot_frame["condition"])]
    sns.boxplot(
        data=plot_frame,
        x="condition",
        y="roc_auc",
        order=present,
        hue="model",
        hue_order=list(DISPLAY_NAMES.values()),
        palette=list(MODEL_COLORS.values()),
        ax=axis,
        fliersize=0,
        linewidth=0.8,
    )
    sns.stripplot(
        data=plot_frame,
        x="condition",
        y="roc_auc",
        order=present,
        hue="model",
        hue_order=list(DISPLAY_NAMES.values()),
        palette=list(MODEL_COLORS.values()),
        ax=axis,
        dodge=True,
        size=3,
        legend=False,
    )
    axis.set_ylim(0, 1)
    axis.set_xlabel("")
    axis.set_ylabel("Validation ROC-AUC")
    axis.set_title("Five-seed spread at selected training compositions")
    axis.tick_params(axis="x", rotation=20)
    axis.legend(frameon=False, title="")
    save_figure(figure, path)


def plot_numeric_overlay(real: pd.DataFrame, synthetic_by_generator: dict[str, pd.DataFrame], path: Path) -> None:
    _style()
    columns = ["age", "education-num", "hours-per-week", "capital-gain"]
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.0))
    for axis, column in zip(axes.ravel(), columns):
        real_values = pd.to_numeric(real[column], errors="coerce")
        axis.hist(real_values, bins=30, density=True, color=GREY, alpha=0.45, label="Real training")
        for generator, frame in synthetic_by_generator.items():
            values = pd.to_numeric(frame[column], errors="coerce")
            axis.hist(
                values,
                bins=30,
                density=True,
                histtype="step",
                linewidth=1.6,
                color=GENERATOR_COLORS[generator],
                label=GENERATOR_LABELS[generator],
            )
        axis.set_title(column)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Numerical distributions, real training versus one synthetic seed", color=NAVY)
    save_figure(figure, path)


def plot_categorical_tv(quality: pd.DataFrame, path: Path) -> None:
    _style()
    figure, axis = plt.subplots(figsize=(8.0, 4.6))
    grouped = (
        quality.groupby(["generator", "feature"], as_index=False)["total_variation"].mean()
    )
    features = list(grouped["feature"].unique())
    positions = np.arange(len(features))
    width = 0.38
    for offset, generator in ((-width / 2, "gaussian_copula"), (width / 2, "ctgan")):
        rows = grouped.loc[grouped["generator"] == generator].set_index("feature").reindex(features)
        axis.bar(
            positions + offset,
            rows["total_variation"],
            width=width,
            color=GENERATOR_COLORS[generator],
            label=GENERATOR_LABELS[generator],
        )
    axis.set_xticks(positions, features, rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean total variation")
    axis.set_title("Categorical distance from the real training distribution")
    axis.legend(frameon=False)
    save_figure(figure, path)


def plot_correlation_heatmaps(real_corr: pd.DataFrame, synthetic_corrs: dict[str, pd.DataFrame], path: Path) -> None:
    _style()
    frames = [("Real training", real_corr)] + [
        (GENERATOR_LABELS[name], matrix) for name, matrix in synthetic_corrs.items()
    ]
    figure, axes = plt.subplots(1, len(frames), figsize=(5.1 * len(frames), 5.0))
    if len(frames) == 1:
        axes = [axes]
    for axis, (title, matrix) in zip(axes, frames):
        sns.heatmap(
            matrix,
            ax=axis,
            vmin=-1,
            vmax=1,
            cmap="RdBu_r",
            square=True,
            cbar=axis is axes[-1],
            annot=False,
        )
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=45, labelsize=7)
        axis.tick_params(axis="y", labelsize=7)
        for label in axis.get_xticklabels():
            label.set_ha("right")
    figure.suptitle("Spearman correlations among numerical features", color=NAVY)
    save_figure(figure, path)


def plot_target_balance(quality_summary: pd.DataFrame, path: Path) -> None:
    _style()
    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    grouped = quality_summary.groupby("generator")[["real_positive_rate", "synthetic_positive_rate"]].mean()
    positions = np.arange(len(grouped))
    axis.bar(positions - 0.18, grouped["real_positive_rate"], width=0.36, color=GREY, label="Real training")
    axis.bar(positions + 0.18, grouped["synthetic_positive_rate"], width=0.36, color=TEAL, label="Synthetic")
    axis.set_xticks(positions, [GENERATOR_LABELS.get(name, name) for name in grouped.index])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Share with income >50K")
    axis.set_title("Target balance")
    axis.legend(frameon=False)
    save_figure(figure, path)


def plot_fidelity_utility(points: pd.DataFrame, path: Path) -> None:
    _style()
    figure, axis = plt.subplots(figsize=(6.6, 4.6))
    for generator, color in GENERATOR_COLORS.items():
        rows = points.loc[points["generator"] == generator]
        if rows.empty:
            continue
        axis.scatter(
            rows["mean_categorical_tv"],
            rows["utility_roc_auc"],
            color=color,
            s=42,
            label=GENERATOR_LABELS[generator],
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean categorical total variation (lower is closer)")
    axis.set_ylabel("Mean validation ROC-AUC of synthetic-only training")
    axis.set_title("Fidelity and synthetic-only utility")
    axis.legend(frameon=False)
    save_figure(figure, path)


def plot_scarcity(summary: pd.DataFrame, path: Path) -> None:
    _style()
    scarcity = summary.loc[summary["stage"] == "scarcity_augmentation"].copy()
    if scarcity.empty:
        return
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    real = summary.loc[summary["stage"] == "scarcity_real"]
    budgets = sorted(real["budget"].dropna().unique())
    for classifier, color in MODEL_COLORS.items():
        rows = real.loc[real["classifier"] == classifier].sort_values("budget")
        axis.plot(
            rows["budget"],
            rows["roc_auc_mean"],
            color=color,
            marker="o",
            label=f"{DISPLAY_NAMES[classifier]}, real only",
        )
    for generator, color in (("gaussian_copula", NAVY), ("ctgan", TEAL)):
        rows = (
            scarcity.loc[scarcity["generator"] == generator]
            .groupby("budget")["roc_auc_mean"]
            .mean()
            .reindex(budgets)
        )
        axis.plot(
            budgets,
            rows.to_numpy(),
            color=color,
            marker="s",
            linestyle="--",
            label=f"{GENERATOR_LABELS[generator]}, augmented",
        )
    axis.set_ylim(0, 1)
    axis.set_xlabel("Real-data budget as a share of the frozen training set")
    axis.set_ylabel("Validation ROC-AUC")
    axis.set_title("Data-scarcity comparison")
    axis.legend(frameon=False, fontsize=8)
    save_figure(figure, path)


def plot_research_summary(paired: pd.DataFrame, path: Path) -> None:
    _style()
    roc = paired.loc[(paired["stage"] == "main") & (paired["metric"] == "roc_auc")]
    best = (
        roc.sort_values("mean_difference", ascending=False)
        .groupby(["generator", "classifier"], as_index=False)
        .first()
    )
    figure, axis = plt.subplots(figsize=(8.0, 4.6))
    labels = [
        f"{GENERATOR_LABELS[row.generator]}\n{DISPLAY_NAMES[row.classifier]}\n{row.synthetic_share:.0%}"
        for row in best.itertuples()
    ]
    colors = [GENERATOR_COLORS[row.generator] for row in best.itertuples()]
    axis.bar(np.arange(len(best)), best["mean_difference"], color=colors)
    axis.errorbar(
        np.arange(len(best)),
        best["mean_difference"],
        yerr=[best["mean_difference"] - best["ci_low"], best["ci_high"] - best["mean_difference"]],
        fmt="none",
        ecolor=NAVY,
        capsize=3,
    )
    axis.axhline(0, color=GREY, linewidth=1)
    axis.set_xticks(np.arange(len(best)), labels)
    axis.set_ylabel("Best paired ROC-AUC difference")
    axis.set_title("Largest mean gain versus real-only training, by model and generator")
    save_figure(figure, path)
