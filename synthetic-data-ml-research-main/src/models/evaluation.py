"""Validation metrics and Phase 1 figures.

Predictive evaluation accepts only the validation partition. Calling it with
the test partition raises an error so a later edit cannot score the holdout
silently during Phase 1.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

POSITIVE_CODE = 1
PHASE1_ALLOWED_PARTITIONS = {"validation"}


def positive_scores(pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return P(income > 50K) using the fitted classifier's class order."""
    classifier = pipeline.named_steps["classifier"]
    probabilities = pipeline.predict_proba(features)
    classes = list(classifier.classes_)
    if POSITIVE_CODE not in classes:
        raise ValueError(f"Positive class {POSITIVE_CODE} is missing from {classes}.")
    return probabilities[:, classes.index(POSITIVE_CODE)]


def evaluate_binary_model(
    pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    partition: str,
    threshold: float = 0.5,
) -> dict:
    """Score one fitted pipeline. Phase 1 allows the validation partition only."""
    if partition not in PHASE1_ALLOWED_PARTITIONS:
        raise RuntimeError(
            "Phase 1 refuses to compute predictive metrics on the "
            f"{partition!r} partition. The held-out test set stays unused "
            "until a later, separately approved evaluation."
        )
    scores = positive_scores(pipeline, features)
    y_true = target.to_numpy()
    y_pred = (scores >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "f1": float(f1_score(y_true, y_pred, pos_label=POSITIVE_CODE, zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=POSITIVE_CODE, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, pos_label=POSITIVE_CODE, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "partition": partition,
    }


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )


def save_model_figures(
    fitted: dict[str, object],
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    metrics_by_model: dict[str, dict],
    figures_dir: Path,
    tables_dir: Path,
    threshold: float = 0.5,
) -> dict[str, str]:
    """Write ROC curves, confusion matrices, the comparison chart, and tables."""
    _style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    y_true = validation_target.to_numpy()
    colors = {
        "logistic_regression": "#4C78A8",
        "random_forest": "#F58518",
        "xgboost": "#54A24B",
    }
    score_map = {}
    prediction_map = {}
    for name, pipeline in fitted.items():
        scores = positive_scores(pipeline, validation_features)
        score_map[name] = scores
        prediction_map[name] = (scores >= threshold).astype(int)

    # ROC curves
    figure, axis = plt.subplots(figsize=(6.6, 5.4))
    for name, scores in score_map.items():
        fpr, tpr, _ = roc_curve(y_true, scores, pos_label=POSITIVE_CODE)
        auc = metrics_by_model[name]["roc_auc"]
        label = f"{metrics_by_model[name]['display_name']} (AUC {auc:.3f})"
        axis.plot(fpr, tpr, color=colors[name], lw=2, label=label)
    axis.plot([0, 1], [0, 1], color="#9A9A9A", lw=1, linestyle="--", label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Validation ROC curves")
    axis.legend(loc="lower right", frameon=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    figure.tight_layout()
    roc_path = figures_dir / "roc_curves.png"
    figure.savefig(roc_path, bbox_inches="tight")
    plt.close(figure)

    # Confusion matrices
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
    matrix_rows = []
    for axis, name in zip(axes, fitted):
        matrix = confusion_matrix(y_true, prediction_map[name], labels=[0, 1])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axis,
            xticklabels=["Pred. <=50K", "Pred. >50K"],
            yticklabels=["True <=50K", "True >50K"],
        )
        axis.set_title(metrics_by_model[name]["display_name"])
        matrix_rows.append(
            {
                "model": name,
                "true_le_50k_pred_le_50k": int(matrix[0, 0]),
                "true_le_50k_pred_gt_50k": int(matrix[0, 1]),
                "true_gt_50k_pred_le_50k": int(matrix[1, 0]),
                "true_gt_50k_pred_gt_50k": int(matrix[1, 1]),
            }
        )
    figure.suptitle("Validation confusion matrices at threshold 0.5", y=1.04)
    figure.tight_layout()
    cm_path = figures_dir / "confusion_matrices.png"
    figure.savefig(cm_path, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(matrix_rows).to_csv(tables_dir / "validation_confusion_matrices.csv", index=False)

    # Comparison chart
    metric_names = ["roc_auc", "f1", "precision", "recall", "balanced_accuracy"]
    metric_labels = ["ROC-AUC", "F1", "Precision", "Recall", "Balanced accuracy"]
    plot_rows = []
    for name, metrics in metrics_by_model.items():
        for key, label in zip(metric_names, metric_labels):
            plot_rows.append(
                {
                    "Model": metrics["display_name"],
                    "Metric": label,
                    "Value": metrics[key],
                }
            )
    plot_frame = pd.DataFrame(plot_rows)
    hue_order = [metrics_by_model[name]["display_name"] for name in fitted]
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    sns.barplot(
        data=plot_frame,
        x="Metric",
        y="Value",
        hue="Model",
        hue_order=hue_order,
        palette=["#4C78A8", "#F58518", "#54A24B"],
        ax=axis,
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Validation score")
    axis.set_xlabel("")
    axis.set_title("Validation metric comparison")
    axis.legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout()
    comparison_path = figures_dir / "model_comparison.png"
    figure.savefig(comparison_path, bbox_inches="tight")
    plt.close(figure)

    return {
        "roc_curves": str(roc_path),
        "confusion_matrices": str(cm_path),
        "model_comparison": str(comparison_path),
    }
