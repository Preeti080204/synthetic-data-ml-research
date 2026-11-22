"""Inspect saved research outputs without fitting a model.

The default command prints a summary computed from the saved tables.
It does not retrain a generator, rescore the held-out test, or change a frozen file.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

NAMES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "gaussian_copula": "Gaussian copula",
    "ctgan": "CTGAN",
    "real": "Real only",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv else "summary"
    if command in {"-h", "--help", "help"}:
        return _help()
    if command == "summary":
        return summary()
    if command == "figures":
        return figures()
    if command == "report":
        return report()
    if command == "test":
        return tests()
    if command == "reproduce":
        return reproduce(argv[1:])
    print(f"Unknown command: {command}")
    return _help() or 2


def summary() -> int:
    validation = pd.read_csv(ROOT / "results" / "tables" / "study" / "run_metrics.csv")
    test = pd.read_csv(ROOT / "results" / "final_evaluation" / "test_run_metrics.csv")
    phase1 = pd.read_csv(ROOT / "results" / "tables" / "validation_metrics.csv")
    paired = pd.read_csv(ROOT / "results" / "final_evaluation" / "paired_differences.csv")
    validation_paired = pd.read_csv(ROOT / "results" / "statistical_tests" / "paired_differences.csv")
    split = pd.read_csv(ROOT / "data" / "processed" / "split_indices.csv")
    counts = split["partition"].value_counts()
    print("Can Synthetic Data Improve Machine Learning?")
    print("Author: Preeti Gorial")
    print()
    print("Dataset")
    print(f"  Pooled rows: {len(split)}")
    print(
        "  Train / validation / test: "
        f"{int(counts['train'])} / {int(counts['validation'])} / {int(counts['test'])}"
    )
    print()
    print("Models and generators")
    print("  Classifiers: " + ", ".join(NAMES[name] for name in sorted(validation["classifier"].unique())))
    print("  Generators: " + ", ".join(NAMES[name] for name in ("gaussian_copula", "ctgan")))
    preferred = [42, 123, 2024, 3407, 777]
    present = {int(seed) for seed in validation["seed"].unique()}
    seeds = [seed for seed in preferred if seed in present]
    seeds.extend(sorted(present.difference(preferred)))
    print("  Seeds: " + ", ".join(str(seed) for seed in seeds))
    print()
    print("Completed experiments")
    print(f"  Validation runs: {len(validation)}")
    print(f"  Held-out test runs: {len(test)}")
    print()
    print("Baseline validation ROC-AUC, Phase 1, one split")
    for _, row in phase1.iterrows():
        print(f"  {row['display_name']}: {row['roc_auc']:.6f}")
    print()
    print("Held-out test, real-only mean ROC-AUC")
    real = test.loc[(test["stage"] == "main") & (test["synthetic_share"] == 0)]
    for classifier, group in real.groupby("classifier"):
        print(f"  {NAMES.get(classifier, classifier)}: {group['roc_auc'].mean():.6f}")
    print()
    print("Synthetic-share comparison, paired ROC-AUC intervals")
    _print_counts("  Validation, main", validation_paired, "main")
    _print_counts("  Test, main", paired, "main")
    print()
    print("Generator comparison, mean paired test ROC-AUC difference, shares below 100%")
    below = paired.loc[
        (paired["stage"] == "main")
        & (paired["metric"] == "roc_auc")
        & (paired["synthetic_share"] < 1)
    ]
    for generator, group in below.groupby("generator"):
        print(f"  {NAMES.get(generator, generator)}: {group['mean_difference'].mean():.6f}")
    print()
    print("Data scarcity, paired test ROC-AUC intervals")
    _print_counts("  Augmentation", paired, "scarcity_augmentation")
    _print_counts("  Replacement", paired, "scarcity_replacement")
    print()
    print("Research questions, from the saved test intervals")
    main_counts = _direction_counts(paired, "main")
    print(
        "  RQ1. Changing the synthetic share does not raise test ROC-AUC above the "
        f"matched real-only run: {main_counts.get('higher', 0)} intervals are higher, "
        f"{main_counts.get('lower', 0)} are lower, and "
        f"{main_counts.get('inconclusive', 0)} cover zero."
    )
    print(
        "  RQ2. Both generators are below the real-only baseline on the mixed-share "
        "test contrasts. The mean paired differences are printed above."
    )
    aug = _direction_counts(paired, "scarcity_augmentation")
    print(
        "  RQ3. Adding synthetic rows beyond a smaller real budget does not raise test "
        f"ROC-AUC: {aug.get('higher', 0)} higher, {aug.get('lower', 0)} lower, "
        f"{aug.get('inconclusive', 0)} inconclusive."
    )
    print(
        "  RQ4. Quality and utility are compared in the saved fidelity table and in "
        "the final report. The summary command does not reinterpret that correlation."
    )
    print()
    print("Locations")
    print("  Validation figures: results/figures/study/")
    print("  Test figures: results/figures/final/")
    print("  Final report: reports/final/Preeti_Gorial_Synthetic_Data_Research_Report.pdf")
    return 0


def figures() -> int:
    from reporting import figures as study_figures
    from reporting.final_figures import (
        plot_test_differences,
        plot_test_ratios,
        plot_test_scarcity,
        plot_validation_versus_test,
    )

    study = ROOT / "results" / "figures" / "study"
    study.mkdir(parents=True, exist_ok=True)
    summary_table = pd.read_csv(ROOT / "results" / "tables" / "study" / "metric_summary.csv")
    runs = pd.read_csv(ROOT / "results" / "tables" / "study" / "run_metrics.csv")
    paired = pd.read_csv(ROOT / "results" / "statistical_tests" / "paired_differences.csv")
    classes = pd.read_csv(ROOT / "results" / "tables" / "class_distribution.csv")
    missing = pd.read_csv(ROOT / "results" / "tables" / "missing_values.csv")
    categorical = pd.read_csv(ROOT / "results" / "tables" / "study" / "categorical_fidelity.csv")
    categorical = categorical.loc[categorical["feature"] != "income"]
    quality = pd.read_csv(ROOT / "results" / "tables" / "study" / "quality_summary.csv")
    points = pd.read_csv(ROOT / "results" / "tables" / "study" / "fidelity_utility_points.csv")
    study_figures.plot_class_distribution(classes, study / "01_target_distribution.png")
    study_figures.plot_missingness(missing, study / "02_missing_values.png")
    study_figures.plot_ratio_curves(summary_table, study / "05_ratio_curves.png")
    study_figures.plot_generator_comparison(summary_table, study / "06_generator_comparison.png")
    study_figures.plot_differences(paired, study / "07_differences.png")
    study_figures.plot_seed_variability(runs, study / "08_seed_variability.png")
    study_figures.plot_categorical_tv(categorical, study / "10_categorical_tv.png")
    real_corr = pd.read_csv(ROOT / "results" / "tables" / "study" / "real_spearman.csv", index_col=0)
    syn_corrs = {
        kind: pd.read_csv(ROOT / "results" / "tables" / "study" / f"spearman_{kind}_seed_42.csv", index_col=0)
        for kind in ("gaussian_copula", "ctgan")
    }
    study_figures.plot_correlation_heatmaps(real_corr, syn_corrs, study / "11_correlation_heatmaps.png")
    study_figures.plot_target_balance(quality, study / "12_target_balance.png")
    study_figures.plot_fidelity_utility(points, study / "13_fidelity_utility.png")
    study_figures.plot_scarcity(summary_table, study / "14_scarcity.png")
    study_figures.plot_research_summary(paired, study / "15_research_summary.png")
    test = pd.read_csv(ROOT / "results" / "final_evaluation" / "test_run_metrics.csv")
    test_paired = pd.read_csv(ROOT / "results" / "final_evaluation" / "paired_differences.csv")
    final = ROOT / "results" / "figures" / "final"
    plot_test_ratios(test, final / "11_test_ratio_curves.png")
    plot_validation_versus_test(runs, test, final / "12_validation_vs_test.png")
    plot_test_differences(test_paired, final / "13_test_differences.png")
    plot_test_scarcity(test, final / "14_test_scarcity.png")
    print("Regenerated summary figures from saved tables.")
    print("Left unchanged: baseline ROC and confusion figures, which read saved model files,")
    print("and the numeric-distribution overlay, which reads synthetic training tables.")
    print(f"Study figures: {study}")
    print(f"Test figures: {final}")
    return 0


def report() -> int:
    from reporting.final_report import write_final_report

    pdf_path, docx_path = write_final_report()
    print(pdf_path)
    print(docx_path)
    return 0


def tests() -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest"],
        cwd=ROOT,
        check=False,
    )
    return int(completed.returncode)


def reproduce(argv: list[str]) -> int:
    print("Full reproduction is computationally expensive.")
    print("It can refit CTGAN when a cache check fails. It is not the default command.")
    print("It does not score the held-out test and it does not rewrite results/frozen/validation_freeze.json.")
    print()
    print("Documented pipeline, from the repository root:")
    print("  uv sync --all-groups")
    print("  uv run python run_phase1.py")
    print("  uv run python scripts/run_validation_study.py")
    print()
    print("The held-out evaluation is already saved. This interface will not run scripts/run_final_test.py.")
    if "--execute" not in argv:
        print("Not executed.")
        return 0
    if "--held-out-test" in argv or any(token.endswith("run_final_test.py") for token in argv):
        print("Refusing to score or rescore the held-out test.")
        return 2
    freeze = ROOT / "results" / "frozen" / "validation_freeze.json"
    if not freeze.is_file():
        print("Refusing because the original freeze file is missing.")
        return 2
    command = [sys.executable, str(ROOT / "scripts" / "run_validation_study.py")]
    print("Executing the cached validation pipeline only:")
    print("  " + " ".join(command))
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return int(completed.returncode)


def _print_counts(label: str, paired: pd.DataFrame, stage: str) -> None:
    counts = _direction_counts(paired, stage)
    print(
        f"{label}: lower {counts.get('lower', 0)}, "
        f"inconclusive {counts.get('inconclusive', 0)}, "
        f"higher {counts.get('higher', 0)}"
    )


def _direction_counts(paired: pd.DataFrame, stage: str) -> dict:
    rows = paired.loc[(paired["stage"] == stage) & (paired["metric"] == "roc_auc")]
    return rows["direction"].value_counts().to_dict()


def _help() -> int:
    print(
        """Usage, from the repository root:

  uv run python scripts/run_project.py
  uv run python scripts/run_project.py summary
  uv run python scripts/run_project.py figures
  uv run python scripts/run_project.py report
  uv run python scripts/run_project.py test
  uv run python scripts/run_project.py reproduce
  uv run python scripts/run_project.py help

summary, figures, report, and test read or rebuild saved outputs.
They do not retrain CTGAN and they do not rerun the held-out test.
reproduce prints the expensive pipeline and does not run it unless --execute is present.
--execute runs the validation study only. It refuses a held-out rescore.
"""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
