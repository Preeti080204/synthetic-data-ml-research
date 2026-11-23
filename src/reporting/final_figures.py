"""Figures for the held-out evaluation. They read saved metric tables only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from reporting.figures import (
    GENERATOR_COLORS,
    GENERATOR_LABELS,
    MODEL_COLORS,
    NAVY,
    TEAL,
    _style,
    save_figure,
)
from models.baseline import DISPLAY_NAMES


def plot_test_ratios(runs: pd.DataFrame, path: Path) -> None:
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    main = runs.loc[runs["stage"] == "main"]
    for axis, generator in zip(axes, ("gaussian_copula", "ctgan")):
        for classifier, color in MODEL_COLORS.items():
            owned = main.loc[(main["generator"] == generator) & (main["classifier"] == classifier)]
            real = main.loc[(main["generator"] == "real") & (main["classifier"] == classifier), "roc_auc"]
            points = [(0.0, float(real.mean()), float(real.std(ddof=1)))]
            for share, group in owned.groupby("synthetic_share"):
                points.append((float(share), float(group["roc_auc"].mean()), float(group["roc_auc"].std(ddof=1))))
            points = sorted(points)
            axis.errorbar(
                [item[0] for item in points],
                [item[1] for item in points],
                yerr=[item[2] for item in points],
                color=color,
                marker="o",
                label=DISPLAY_NAMES[classifier],
                capsize=2,
            )
        axis.set_ylim(0.75, 1.0)
        axis.set_title(GENERATOR_LABELS[generator])
        axis.set_xlabel("Synthetic share")
    axes[0].set_ylabel("Test ROC-AUC")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    save_figure(figure, path)


def plot_validation_versus_test(validation: pd.DataFrame, test: pd.DataFrame, path: Path) -> None:
    _style()
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    keys = ["generator", "synthetic_share", "classifier"]
    left = (
        validation.loc[validation["stage"] == "main"]
        .groupby(keys)["roc_auc"]
        .mean()
        .rename("validation")
    )
    right = test.loc[test["stage"] == "main"].groupby(keys)["roc_auc"].mean().rename("test")
    joined = pd.concat([left, right], axis=1).dropna().reset_index()
    for generator, color in GENERATOR_COLORS.items():
        subset = joined.loc[joined["generator"] == generator]
        axis.scatter(subset["validation"], subset["test"], color=color, label=GENERATOR_LABELS[generator], s=36)
    limits = (0.82, 0.96)
    axis.plot(limits, limits, color="#9AA5B1", linestyle="--", linewidth=1)
    axis.set_xlim(*limits)
    axis.set_ylim(*limits)
    axis.set_xlabel("Mean validation ROC-AUC")
    axis.set_ylabel("Mean test ROC-AUC")
    axis.set_title("Same configuration, two partitions")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    save_figure(figure, path)


def plot_test_differences(paired: pd.DataFrame, path: Path) -> None:
    _style()
    frame = paired.loc[(paired["stage"] == "main") & (paired["metric"] == "roc_auc")].copy()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    for axis, generator in zip(axes, ("gaussian_copula", "ctgan")):
        subset = frame.loc[frame["generator"] == generator]
        for classifier, color in MODEL_COLORS.items():
            rows = subset.loc[subset["classifier"] == classifier].sort_values("synthetic_share")
            axis.errorbar(
                rows["synthetic_share"],
                rows["mean_difference"],
                yerr=[rows["mean_difference"] - rows["ci_low"], rows["ci_high"] - rows["mean_difference"]],
                color=color,
                marker="o",
                label=DISPLAY_NAMES[classifier],
                capsize=2,
            )
        axis.axhline(0, color="#9AA5B1", linewidth=1)
        axis.set_title(GENERATOR_LABELS[generator])
        axis.set_xlabel("Synthetic share")
    axes[0].set_ylabel("Test ROC-AUC minus real-only")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    save_figure(figure, path)


def plot_test_scarcity(runs: pd.DataFrame, path: Path) -> None:
    _style()
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    aug = runs.loc[runs["stage"] == "scarcity_augmentation"]
    real = runs.loc[runs["stage"] == "scarcity_real"]
    for classifier, color in MODEL_COLORS.items():
        real_rows = real.loc[real["classifier"] == classifier].groupby("budget")["roc_auc"].mean()
        axis.plot(real_rows.index, real_rows.values, color=color, marker="o", label=f"{DISPLAY_NAMES[classifier]}, real only")
    for generator, marker in (("gaussian_copula", "s"), ("ctgan", "D")):
        subset = aug.loc[aug["generator"] == generator].groupby("budget")["roc_auc"].mean()
        axis.plot(
            subset.index,
            subset.values,
            color=GENERATOR_COLORS[generator],
            marker=marker,
            linestyle="--",
            label=f"{GENERATOR_LABELS[generator]}, augmented",
        )
    axis.set_xlabel("Real-data budget")
    axis.set_ylabel("Mean test ROC-AUC")
    axis.set_ylim(0.75, 1.0)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    figure.tight_layout()
    save_figure(figure, path)
