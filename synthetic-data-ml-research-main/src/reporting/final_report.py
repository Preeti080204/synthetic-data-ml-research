"""Write the final report from a completed test-metric table.

The function refuses to create the final documents when that table is absent.
Every number is computed from the saved test metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.summarize import paired_differences
from reporting.academic_docx import write_docx
from reporting.academic_pdf import write_pdf
from utils.paths import project_root, results_dir

CLASSIFIERS = ("logistic_regression", "random_forest", "xgboost")
DISPLAY = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "gaussian_copula": "Gaussian copula",
    "ctgan": "CTGAN",
    "real": "Real only",
}


def write_final_report(metrics: pd.DataFrame | None = None) -> tuple[Path, Path]:
    path = results_dir() / "final_evaluation" / "test_run_metrics.csv"
    if metrics is None:
        if not path.is_file():
            raise FileNotFoundError(
                "Final documents were not written. No authorized test metrics exist at "
                f"{path}."
            )
        metrics = pd.read_csv(path)
    metrics = _with_classifier(metrics)
    if metrics.empty or "partition" not in metrics.columns or not (metrics["partition"] == "test").all():
        raise ValueError("Final documents require a non-empty table scored on the test partition.")
    paired = _paired(metrics)
    paired_path = results_dir() / "final_evaluation" / "paired_differences.csv"
    paired_path.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(paired_path, index=False)
    if "classifier" not in pd.read_csv(path, nrows=0).columns:
        metrics.to_csv(path, index=False)
    from reporting.final_audit import audit_saved_test
    from reporting.final_document import build_final_markdown
    from reporting.final_figures import (
        plot_test_differences,
        plot_test_ratios,
        plot_test_scarcity,
        plot_validation_versus_test,
    )

    audit = audit_saved_test()
    validation = pd.read_csv(results_dir() / "tables" / "study" / "run_metrics.csv")
    validation_paired = pd.read_csv(results_dir() / "statistical_tests" / "paired_differences.csv")
    figure_dir = results_dir() / "figures" / "final"
    plot_test_ratios(metrics, figure_dir / "11_test_ratio_curves.png")
    plot_validation_versus_test(validation, metrics, figure_dir / "12_validation_vs_test.png")
    plot_test_differences(paired, figure_dir / "13_test_differences.png")
    plot_test_scarcity(metrics, figure_dir / "14_test_scarcity.png")
    bundle = {
        "test": metrics,
        "paired": paired,
        "validation": validation,
        "validation_paired": validation_paired,
        "quality": pd.read_csv(results_dir() / "tables" / "study" / "quality_summary.csv"),
        "points": pd.read_csv(results_dir() / "tables" / "study" / "fidelity_utility_points.csv"),
        "phase1": pd.read_csv(results_dir() / "tables" / "validation_metrics.csv"),
        "classes": pd.read_csv(results_dir() / "tables" / "class_distribution.csv"),
        "partition_audit": json.loads((results_dir() / "tables" / "partition_audit.json").read_text()),
        "audit": audit,
    }
    template = build_final_markdown(bundle)
    destination = project_root() / "reports" / "final"
    destination.mkdir(parents=True, exist_ok=True)
    pdf_path = destination / "Preeti_Gorial_Synthetic_Data_Research_Report.pdf"
    docx_path = destination / "Preeti_Gorial_Synthetic_Data_Research_Report.docx"
    markdown = template
    for _ in range(6):
        pdf = write_pdf(markdown, pdf_path, project_root(), running_header="Final report")
        updated = _apply_pages(template, pdf.heading_pages, pdf.caption_pages, pdf.table_pages)
        if updated == markdown:
            break
        markdown = updated
    else:
        write_pdf(markdown, pdf_path, project_root(), running_header="Final report")
    write_docx(markdown, docx_path, project_root())
    # PDF/DOCX resolve figure paths from the repository root. Markdown is
    # viewed inside reports/final/, so its links must be relative to that folder.
    # Only adjust the published Markdown links; do not alter research content.
    repository_markdown = markdown.replace(
        "](results/figures/", "](../../results/figures/"
    )
    (destination / "Preeti_Gorial_Synthetic_Data_Research_Report.md").write_text(
        repository_markdown, encoding="utf-8"
    )
    if "{{" in markdown:
        raise RuntimeError("The final report still has unresolved page placeholders.")
    return pdf_path, docx_path


def _apply_pages(text, headings, captions, tables) -> str:
    for _level, title, page in headings:
        text = text.replace(f"{{{{H:{title}}}}}", str(page))
    for caption, page in captions:
        text = text.replace(f"{{{{F:{caption}}}}}", str(page))
    for caption, page in tables:
        text = text.replace(f"{{{{T:{caption}}}}}", str(page))
    return text


def _with_classifier(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    if "classifier" in frame.columns and frame["classifier"].notna().all():
        return frame

    def parse(identifier: str) -> str:
        for part in str(identifier).split("__"):
            if part in CLASSIFIERS:
                return part
        raise ValueError(f"No classifier name in {identifier}.")

    frame["classifier"] = frame["experiment_id"].map(parse)
    return frame


def _paired(metrics: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    main = metrics.loc[metrics["stage"] == "main"]
    if len(main):
        pieces.append(paired_differences(main, baseline_stage="main"))
    for stage in ("scarcity_augmentation", "scarcity_replacement"):
        if not (metrics["stage"] == stage).any():
            continue
        combined = pd.concat(
            [metrics.loc[metrics["stage"] == stage], metrics.loc[metrics["stage"] == "scarcity_real"]],
            ignore_index=True,
        )
        rewritten = combined.copy()
        rewritten.loc[rewritten["stage"] == "scarcity_real", "stage"] = stage
        pieces.append(paired_differences(rewritten, baseline_stage=stage))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _text(metrics: pd.DataFrame, paired: pd.DataFrame) -> str:
    roc = paired.loc[paired["metric"] == "roc_auc"] if len(paired) else paired
    main = roc.loc[roc["stage"] == "main"] if len(roc) else roc
    aug = roc.loc[roc["stage"] == "scarcity_augmentation"] if len(roc) else roc
    repl = roc.loc[roc["stage"] == "scarcity_replacement"] if len(roc) else roc
    lines = [
        ":::cover",
        "# Can Synthetic Data Improve Machine Learning?",
        "## Final held-out evaluation",
        "Preeti Gorial",
        "Projet Étudiant",
        "Université de Technologie de Troyes (UTT)",
        "Academic year 2026–2027",
        "Every number below is computed from results/final_evaluation/test_run_metrics.csv.",
        ":::",
        "",
        "# 1. Test evaluation",
        "",
        f"The authorized scorer wrote {len(metrics)} completed test runs. "
        "The configurations, seeds, hyperparameters, and synthetic files are the frozen validation plan. "
        "No configuration was selected from these test metrics.",
        "",
        _count_table(metrics),
        "",
        "# 2. Mean test ROC-AUC",
        "",
        "Parentheses are sample standard deviations across the five seeds.",
        "",
        _mean_table(metrics.loc[metrics["stage"] == "main"]),
        "",
        "# 3. Paired comparisons",
        "",
        "Each difference is the test ROC-AUC of a synthetic condition minus the matched-seed real-only run. "
        "The interval is a 95 percent Student-t interval. An interval above zero is higher, an interval below zero is lower, and an interval covering zero is inconclusive.",
        "",
        _direction_paragraph(main, "the main experiment"),
        "",
        _paired_table(main),
        "",
        "# 4. Data-scarcity test results",
        "",
        _direction_paragraph(aug, "scarcity augmentation"),
        "",
        _practical_note(aug),
        "",
        _paired_table(aug),
        "",
        _direction_paragraph(repl, "fixed-size scarcity replacement"),
        "",
        _paired_table(repl),
        "",
        "# 5. Conclusion",
        "",
        _conclusion(main, aug),
        "",
    ]
    return "\n".join(line for line in lines if line is not None)


def _count_table(metrics: pd.DataFrame) -> str:
    counts = metrics.groupby(["stage", "generator", "synthetic_share"], dropna=False).size()
    rows = ["| Stage | Generator | Share | Runs |", "| --- | --- | --- | --- |"]
    for (stage, generator, share), count in counts.items():
        rows.append(f"| {stage} | {DISPLAY.get(generator, generator)} | {float(share):.0%} | {int(count)} |")
    return "\n".join(rows)


def _mean_table(main: pd.DataFrame) -> str:
    if main.empty:
        return "No main-experiment test runs were saved."
    rows = ["| Generator | Share | Model | Mean ROC-AUC |", "| --- | --- | --- | --- |"]
    grouped = main.groupby(["generator", "synthetic_share", "classifier"], dropna=False)["roc_auc"]
    for (generator, share, classifier), values in grouped:
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        rows.append(
            f"| {DISPLAY.get(generator, generator)} | {float(share):.0%} | "
            f"{DISPLAY.get(classifier, classifier)} | {mean:.4f} ({std:.4f}) |"
        )
    return "\n".join(rows)


def _paired_table(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "No paired contrasts were saved."
    rows = [
        "| Generator | Share | Model | Budget | Mean difference [interval] | Reading |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    ordered = frame.sort_values(["generator", "synthetic_share", "classifier", "budget"], na_position="first")
    for row in ordered.itertuples(index=False):
        budget = "" if pd.isna(row.budget) else f"{float(row.budget):.0%}"
        rows.append(
            "| "
            + " | ".join(
                [
                    DISPLAY.get(row.generator, row.generator),
                    f"{float(row.synthetic_share):.0%}",
                    DISPLAY.get(row.classifier, row.classifier),
                    budget,
                    f"{row.mean_difference:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}]",
                    row.direction,
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _direction_paragraph(frame: pd.DataFrame, label: str) -> str:
    if frame is None or frame.empty:
        return f"No paired ROC-AUC contrasts were available for {label}."
    counts = frame["direction"].value_counts().to_dict()
    return (
        f"For {label}, {int(counts.get('higher', 0))} ROC-AUC intervals lay entirely above zero, "
        f"{int(counts.get('lower', 0))} lay entirely below zero, and "
        f"{int(counts.get('inconclusive', 0))} covered zero."
    )


def _practical_note(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return ""
    notes = []
    for row in frame.loc[frame["direction"] == "higher"].itertuples(index=False):
        difference = float(row.mean_difference)
        if abs(difference) < 0.001:
            budget = "" if pd.isna(row.budget) else f" at budget {float(row.budget):.0%}"
            notes.append(
                f"The higher {DISPLAY.get(row.generator, row.generator)} contrast for "
                f"{DISPLAY.get(row.classifier, row.classifier)}{budget} has mean difference {difference:.6f}. "
                "A ROC-AUC change of this size is not a meaningful practical gain."
            )
    return " ".join(notes)


def _conclusion(main: pd.DataFrame, aug: pd.DataFrame) -> str:
    main_text = _direction_paragraph(main, "the held-out main comparison")
    aug_text = _direction_paragraph(aug, "the held-out scarcity augmentation comparison")
    return (
        f"{main_text} {aug_text} "
        "These statements describe the frozen configurations on the held-out test partition. "
        "They were not used to choose a generator, a share, or a classifier."
    )
