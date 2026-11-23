"""Markdown source for the final report. Numbers come from saved tables."""

from __future__ import annotations

import json

import pandas as pd
from scipy import stats

from models.baseline import DISPLAY_NAMES

DISPLAY = {
    **DISPLAY_NAMES,
    "gaussian_copula": "Gaussian copula",
    "ctgan": "CTGAN",
    "real": "Real only",
}


def build_final_markdown(bundle: dict) -> str:
    test = bundle["test"]
    paired = bundle["paired"]
    validation = bundle["validation"]
    val_paired = bundle["validation_paired"]
    parts = [
        _cover(bundle),
        _front(bundle),
        _introduction(),
        _literature(),
        _questions(),
        _dataset(bundle),
        _protocol(),
        _baselines(bundle),
        _generators(),
        _design(),
        _validation_results(validation, val_paired),
        _quality(bundle),
        _scarcity_validation(val_paired),
        _test_results(test, paired),
        _comparison(val_paired, paired),
        _discussion(paired, bundle),
        _limitations(bundle),
        _conclusion(paired),
        _references(),
        _appendix(bundle),
    ]
    return "\n\n".join(parts)


def _cover(bundle: dict) -> str:
    audit = bundle["audit"]
    return f"""
:::cover
# Can Synthetic Data Improve Machine Learning?
## Final research report
Preeti Gorial
Projet Étudiant
Université de Technologie de Troyes (UTT)
Academic year 2026–2027
Held-out evaluation timestamp: {audit["evaluation_timestamp"]}
The test metrics were read from results/final_evaluation/test_run_metrics.csv.
:::
""".strip()


def _front(bundle: dict) -> str:
    abstract = _abstract(bundle)
    toc = [
        ("Abstract", "{{H:Abstract}}"),
        ("1. Introduction", "{{H:1. Introduction}}"),
        ("2. Literature review", "{{H:2. Literature review}}"),
        ("3. Research questions and hypotheses", "{{H:3. Research questions and hypotheses}}"),
        ("4. Dataset", "{{H:4. Dataset}}"),
        ("5. Protocol", "{{H:5. Protocol}}"),
        ("6. Baseline models", "{{H:6. Baseline models}}"),
        ("7. Generators", "{{H:7. Generators}}"),
        ("8. Experimental design", "{{H:8. Experimental design}}"),
        ("9. Validation results", "{{H:9. Validation results}}"),
        ("10. Synthetic-data quality", "{{H:10. Synthetic-data quality}}"),
        ("11. Scarcity on the validation partition", "{{H:11. Scarcity on the validation partition}}"),
        ("12. Held-out test evaluation", "{{H:12. Held-out test evaluation}}"),
        ("13. Validation compared with the test set", "{{H:13. Validation compared with the test set}}"),
        ("14. Discussion", "{{H:14. Discussion}}"),
        ("15. Limitations", "{{H:15. Limitations}}"),
        ("16. Conclusion", "{{H:16. Conclusion}}"),
        ("References", "{{H:References}}"),
        ("Appendix A. Reproduction", "{{H:Appendix A. Reproduction}}"),
    ]
    figures = [
        ("Figure 1. Target-class distribution in the pooled Adult data.", "{{F:Figure 1. Target-class distribution in the pooled Adult data.}}"),
        ("Figure 2. Validation ROC curves of the frozen Phase 1 models.", "{{F:Figure 2. Validation ROC curves of the frozen Phase 1 models.}}"),
        ("Figure 3. Validation ROC-AUC by synthetic share.", "{{F:Figure 3. Validation ROC-AUC by synthetic share.}}"),
        ("Figure 4. Mean validation ROC-AUC of the Gaussian copula and CTGAN.", "{{F:Figure 4. Mean validation ROC-AUC of the Gaussian copula and CTGAN.}}"),
        ("Figure 5. Numerical distributions for the real training partition and seed 42.", "{{F:Figure 5. Numerical distributions for the real training partition and seed 42.}}"),
        ("Figure 6. Mean categorical total variation from the real training distribution.", "{{F:Figure 6. Mean categorical total variation from the real training distribution.}}"),
        ("Figure 7. Spearman correlations among numerical features.", "{{F:Figure 7. Spearman correlations among numerical features.}}"),
        ("Figure 8. Share of rows with income above 50,000 dollars.", "{{F:Figure 8. Share of rows with income above 50,000 dollars.}}"),
        ("Figure 9. Categorical fidelity and synthetic-only validation ROC-AUC.", "{{F:Figure 9. Categorical fidelity and synthetic-only validation ROC-AUC.}}"),
        ("Figure 10. Validation ROC-AUC under real-data budgets.", "{{F:Figure 10. Validation ROC-AUC under real-data budgets.}}"),
        ("Figure 11. Test ROC-AUC by synthetic share.", "{{F:Figure 11. Test ROC-AUC by synthetic share.}}"),
        ("Figure 12. Mean validation ROC-AUC against mean test ROC-AUC.", "{{F:Figure 12. Mean validation ROC-AUC against mean test ROC-AUC.}}"),
        ("Figure 13. Paired test ROC-AUC differences from the matched-seed real-only run.", "{{F:Figure 13. Paired test ROC-AUC differences from the matched-seed real-only run.}}"),
        ("Figure 14. Test ROC-AUC under real-data budgets.", "{{F:Figure 14. Test ROC-AUC under real-data budgets.}}"),
    ]
    tables = [
        ("Table 1. Pooled income class.", "{{T:Table 1. Pooled income class.}}"),
        ("Table 2. Frozen Phase 1 validation metrics.", "{{T:Table 2. Frozen Phase 1 validation metrics.}}"),
        ("Table 3. Mean fidelity across five generator seeds.", "{{T:Table 3. Mean fidelity across five generator seeds.}}"),
        ("Table 4. Mean test ROC-AUC in the main experiment.", "{{T:Table 4. Mean test ROC-AUC in the main experiment.}}"),
        ("Table 5. Paired test ROC-AUC differences in the main experiment.", "{{T:Table 5. Paired test ROC-AUC differences in the main experiment.}}"),
        ("Table 6. Paired test ROC-AUC differences for scarcity augmentation.", "{{T:Table 6. Paired test ROC-AUC differences for scarcity augmentation.}}"),
        ("Table 7. Paired test ROC-AUC differences for fixed-size replacement.", "{{T:Table 7. Paired test ROC-AUC differences for fixed-size replacement.}}"),
        ("Table 8. Validation and test interval readings for ROC-AUC.", "{{T:Table 8. Validation and test interval readings for ROC-AUC.}}"),
    ]
    return "\n\n".join(
        [
            "# Abstract",
            abstract,
            "Keywords: synthetic data; Gaussian copula; CTGAN; Adult Census Income; held-out evaluation.",
            "# Contents",
            "Page numbers in this list are the pages of this PDF. The Word file also contains a table-of-contents field.",
            _table(["Section", "Page"], toc),
            "# List of figures",
            _table(["Figure", "Page"], figures),
            "# List of tables",
            _table(["Table", "Page"], tables),
        ]
    )


def _abstract(bundle: dict) -> str:
    test_main = _roc(bundle["paired"], "main")
    val_main = _roc(bundle["validation_paired"], "main")
    return (
        "This report asks whether synthetic rows improve income classification on the UCI Adult Census Income table. "
        "A Gaussian copula and CTGAN were fit on the frozen training partition. Three classifiers were trained at "
        "synthetic shares of 0, 10, 25, 50, 75, and 100 percent, and again when only 25, 50, or 75 percent of the real "
        f"training rows were available. { _counts(val_main, 'On the validation partition') } "
        f"{ _counts(test_main, 'On the held-out test partition') } "
        "Five seeds are repeated draws on one split. They are not five datasets. "
        "The ten studies named in the approved proposal were checked against publisher or repository records. None of them is this Adult protocol."
    )


def _introduction() -> str:
    return """
# 1. Introduction

The practical question is whether rows drawn from a generator trained on a census table can improve a classifier that predicts whether income exceeds 50,000 dollars. A positive answer would matter when real rows are scarce or costly. A negative answer matters too, because synthetic rows can look plausible and still move the decision boundary in the wrong direction.

The study uses one public extract, two required generators, and three classifiers whose hyperparameters were frozen before the synthetic experiments. The validation partition was used to describe the mixtures and to apply a pre-written rule for the scarcity share. The held-out test partition was scored only after that plan was frozen. Test scores were not used to choose an epoch budget, a share, a generator, or a classifier.
""".strip()


def _literature() -> str:
    return """
# 2. Literature review

The approved proposal names ten studies. Each bibliographic record below was checked against the publisher, proceedings, or arXiv page. Placeholder links in the proposal were not copied. Where a record has no DOI, none is invented. Becker and Kohavi (1996) are cited separately because they document the Adult extract required by the later experimental instructions. They are not one of the ten.

Manousakas and Aydöre (2023) compared synthetic tabular data for sharing, augmentation, class balancing, and summarisation. They report marginal gains for balancing on some datasets and conclude that the evidence does not support a general claim that synthetic tabular data improves machine-learning training. Their generators and datasets are not this protocol.

Xu, Skoularidou, Cuesta-Infante, and Veeramachaneni (2019) introduce CTGAN for mixed continuous and categorical columns. On their benchmark, CTGAN outperformed the Bayesian-network baselines on most of the real datasets they studied. That result is about generation quality under their own epoch budget and utility metric. It is not an estimate of the shares used here.

Fonseca and Bacao (2023) review 70 generation algorithms across six machine-learning problem types and treat quality metrics as part of that map. Hernandez, Epelde, Alberdi, Cilla, and Rankin (2022) review 34 papers on synthetic tabular health records, with emphasis on GANs and privacy, and conclude that no single method or metric covers that setting. Jordon et al. (2022) survey motivations and limits. They state that synthetic data is not automatically private and is not a substitute for evaluating a final tool on real data.

Pereira et al. (2024) compare differentially private marginal and GAN generators on utility and fairness. In their pipelines, marginal generators surpassed the GAN generators they tested, and some matched real-data utility. Decruyenaere et al. (2024) show that treating synthetic rows as if they had been observed can inflate false-positive rates in statistical inference, even when estimates are unbiased. Kurakova and Homayouni (2025) propose a medical evaluation framework covering quality, privacy, usability, and computation, and show that duplicate amplification and out-of-range values can be missed by statistical similarity alone. Adams et al. (2025) compare fidelity, utility, and privacy on three patient datasets. Achterberg, Haas, van Dijk, and Spruit (2025) show that a task-specific generator can improve prediction without maximising resemblance to the source table.

| Proposal study | What it contributes | Boundary for this experiment |
| --- | --- | --- |
| Manousakas and Aydöre (2023) | Augmentation is not generally useful | Different generators and datasets |
| Xu et al. (2019) | CTGAN method | Different benchmark and epoch budget |
| Fonseca and Bacao (2023) | Field map and quality metrics | Review, not this grid |
| Hernandez et al. (2022) | Health-record synthesis review | Different domain |
| Pereira et al. (2024) | Private synthesis, utility, fairness | Differential privacy is not used here |
| Decruyenaere et al. (2024) | Inferential utility is a separate question | This study scores prediction |
| Kurakova and Homayouni (2025) | Multidimensional medical evaluation | Smaller diagnostic set here |
| Achterberg et al. (2025) | Utility need not follow fidelity | Their generator was not fit |
| Adams et al. (2025) | Fidelity, utility, and privacy can trade off | Patient data and privacy attacks |
| Jordon et al. (2022) | Synthetic data is not automatically private | No privacy guarantee is claimed |

The proposal left the dataset open and listed scarcity budgets of 100, 50, 25, and 10 percent. The 22 October instructions fix the dataset as Adult income and replace those budgets with 25, 50, and 75 percent. The experiment follows the later instructions. The narrower gap is then the one in the frozen analysis plan: on this split, do these two generators and these three classifiers change ROC-AUC across the specified shares and budgets, and do the fidelity checks move with that utility? The studies above do not answer that question.
""".strip()


def _questions() -> str:
    return """
# 3. Research questions and hypotheses

The questions and the decision rules were written before the utility results were interpreted.

RQ1. Does the synthetic share of a fixed-size training set change ROC-AUC relative to real training rows alone?

RQ2. Do the Gaussian copula and CTGAN differ in that comparison?

RQ3. When only 25, 50, or 75 percent of the real training rows are available, does augmentation beyond that budget change ROC-AUC relative to a model trained on the budget alone?

RQ4. Does closer distributional fidelity accompany higher synthetic-only ROC-AUC?

The primary metric is ROC-AUC. For a paired difference across five seeds, the report gives the mean, the sample standard deviation, and a 95 percent Student-t interval with four degrees of freedom. An interval entirely above zero is higher on these seeds. An interval entirely below zero is lower. An interval covering zero is inconclusive. A paired t statistic is descriptive only. A zero sample standard deviation is reported as a point.

The directional statement for RQ1, written before the results, was that at least one share in {10, 25, 50, 75} percent would have a paired ROC-AUC interval entirely above zero. A 100 percent run is synthetic-only training and is not called augmentation. RQ3 uses the same interval rule for augmentation minus the budget model. RQ4 is descriptive.
""".strip()


def _dataset(bundle: dict) -> str:
    classes = bundle["classes"]
    rows = [
        (row.income, f"{int(row.count):,}", f"{row.proportion:.2%}")
        for row in classes.itertuples(index=False)
    ]
    audit = bundle["partition_audit"]
    return f"""
# 4. Dataset

The table is the UCI Adult Census Income extract (Becker & Kohavi, 1996). The pooled file has 48,842 rows, fourteen predictors, and a binary income target. The frozen split places 34,189 rows in training, 7,326 in validation, and 7,327 in test. Source-row overlap is zero. The pooled table contains {audit["exact_duplicate_extra_rows"]} extra exact duplicate rows and {audit["cross_partition_duplicate_groups"]} cross-partition duplicate groups covering {audit["cross_partition_duplicate_rows"]} rows. {audit["conflicting_feature_groups"]} feature groups have conflicting labels. These records were kept. `fnlwgt` is a predictor, not a sample weight.

Table 1. Pooled income class.

{_table(["Income", "Rows", "Share"], rows)}

![Figure 1. Target-class distribution in the pooled Adult data.](results/figures/study/01_target_distribution.png)
""".strip()


def _protocol() -> str:
    return """
# 5. Protocol

Preprocessing is the Phase 1 pipeline and is fit on the training rows of that run only. Numerical medians and the categorical level Missing are not learned from validation or test rows. Logistic regression standardizes numerical columns. The tree models do not. The decision threshold for F1, precision, recall, and balanced accuracy is 0.5. ROC-AUC uses the predicted probability of income above 50,000 dollars.

The main experiment keeps the training size at N = 34,189. The synthetic count is Python's round of share times N, and the real count is N minus that number. Real and synthetic rows are stratified samples without replacement. The 0 percent condition is one shared real-only baseline. The 100 percent condition contains no real training rows.

Scarcity budgets are 25, 50, and 75 percent of the frozen training partition. Augmentation keeps every real budget row and adds synthetic rows. Replacement holds the training size at the budget. The two are not pooled. The augmentation share was selected on the validation partition by the pre-written rule and is 10 percent for both generators. The test set was not used to make that choice.
""".strip()


def _baselines(bundle: dict) -> str:
    phase1 = bundle["phase1"]
    rows = [
        (
            row.display_name,
            f"{row.roc_auc:.4f}",
            f"{row.f1:.4f}",
            f"{row.precision:.4f}",
            f"{row.recall:.4f}",
            f"{row.balanced_accuracy:.4f}",
        )
        for row in phase1.itertuples(index=False)
    ]
    return f"""
# 6. Baseline models

Phase 1 selected logistic regression, random forest, and XGBoost by validation ROC-AUC. Those hyperparameters stay fixed. Each repetition changes only `random_state`. Table 2 repeats the saved one-split validation scores. They are not test scores, and they are not the five-seed means. The five-seed real-only runs reshuffle the same training rows.

Table 2. Frozen Phase 1 validation metrics.

{_table(["Model", "ROC-AUC", "F1", "Precision", "Recall", "Balanced accuracy"], rows)}

![Figure 2. Validation ROC curves of the frozen Phase 1 models.](results/figures/study/03_baseline_roc.png)
""".strip()


def _generators() -> str:
    return """
# 7. Generators

Both generators are fit on real training rows only, including the income target. The Gaussian copula uses beta marginals, minimum and maximum enforcement, and rounding. Beta marginals are the recorded SDV default. They are not a claim that the Adult features follow a beta distribution. The dependence model is a Gaussian copula.

CTGAN uses the library architecture: embedding dimension 128, generator and discriminator widths (256, 256), learning rates 0.0002, batch size 500, and pac size 10. Training uses 80 epochs on CPU. The library default is 300 epochs. A two-epoch benchmark on the Apple GPU took 18.9 seconds, but an 80-epoch GPU fit stalled before any classifier was trained. One CPU epoch on the full training partition took about 13 seconds. The epoch budget was written down from that timing check before any full-data utility metric was used to choose it. The shorter budget can leave CTGAN under-trained.

Invalid target labels are counted. Up to three extra sampling rounds may be used. Rows are not duplicated to fill a shortfall. Exact matches to training rows are a memorization check, not a privacy guarantee.
""".strip()


def _design() -> str:
    return """
# 8. Experimental design

The validation stage has 165 main runs and 225 scarcity runs, 390 in total. The held-out stage repeats that matrix on the same test partition: real-only; shares of 10, 25, 50, 75, and 100 percent; both generators; three classifiers; five seeds; and the frozen 10 percent scarcity augmentation and replacement runs at three budgets.

Seeds are 42, 123, 2024, 3407, and 777. A comparison uses the same seed for the synthetic run and the real-only run. The sample standard deviation is taken across those five seeds. With four degrees of freedom, a small difference often has an interval that covers zero or that excludes zero by a margin too small to matter for a classifier.
""".strip()


def _validation_results(validation: pd.DataFrame, paired: pd.DataFrame) -> str:
    roc = _roc(paired, "main")
    real = validation.loc[(validation["stage"] == "main") & (validation["synthetic_share"] == 0)]
    bits = []
    zero_note = ""
    for classifier, group in real.groupby("classifier"):
        std = float(group["roc_auc"].std(ddof=1))
        bits.append(f"{DISPLAY[classifier]} {group['roc_auc'].mean():.4f} (SD {std:.4f})")
        if std == 0:
            zero_note = f" {DISPLAY[classifier]} has a sample standard deviation of zero across these seeds."
    return f"""
# 9. Validation results

The five-seed real-only validation means are {"; ".join(bits)}.{zero_note}

{_counts(roc, "For main-experiment validation ROC-AUC")} The one inconclusive contrast is the interval that covers zero. The higher count is zero. Shares below 100 percent and the six synthetic-only contrasts are included in that sentence. Raising the synthetic share lowers validation ROC-AUC for both generators. CTGAN declines less than the Gaussian copula.

![Figure 3. Validation ROC-AUC by synthetic share.](results/figures/study/05_ratio_curves.png)

![Figure 4. Mean validation ROC-AUC of the Gaussian copula and CTGAN.](results/figures/study/06_generator_comparison.png)

Secondary validation metrics do not reverse the ROC-AUC pattern into a practical gain. Their paired files are in `results/statistical_tests/paired_differences.csv`. {_secondary(paired, "main", "validation")}
""".strip()


def _quality(bundle: dict) -> str:
    quality = bundle["quality"]
    points = bundle["points"]
    rows = []
    for generator, group in quality.groupby("generator"):
        rows.append(
            (
                DISPLAY.get(generator, generator),
                f"{group['mean_ks'].mean():.3f}",
                f"{group['mean_categorical_tv'].mean():.3f}",
                f"{group['synthetic_positive_rate'].mean():.3f}",
                f"{group['exact_training_match_rate'].mean():.4f}",
                f"{group['near_duplicate_rate'].mean():.3f}",
            )
        )
    rho = stats.spearmanr(points["mean_categorical_tv"], points["utility_roc_auc"])
    return f"""
# 10. Synthetic-data quality

Quality is computed against the real training partition. The test partition is not used. Table 3 gives the mean of each generator across its five seeds.

Table 3. Mean fidelity across five generator seeds.

{_table(["Generator", "Mean KS", "Mean TV", "Synthetic positive rate", "Exact train match", "Near-duplicate rate"], rows)}

![Figure 5. Numerical distributions for the real training partition and seed 42.](results/figures/study/09_numeric_distributions.png)

![Figure 6. Mean categorical total variation from the real training distribution.](results/figures/study/10_categorical_tv.png)

![Figure 7. Spearman correlations among numerical features.](results/figures/study/11_correlation_heatmaps.png)

![Figure 8. Share of rows with income above 50,000 dollars.](results/figures/study/12_target_balance.png)

Across the {len(points)} generator-seed points, the Spearman correlation between mean categorical total variation and synthetic-only validation ROC-AUC is {rho.statistic:.3f}. The descriptive two-sided p-value is {rho.pvalue:.3f}. The Gaussian copula is closer on categories and weaker as a synthetic-only training set. CTGAN is closer on the Kolmogorov-Smirnov distance and has the higher synthetic-only ROC-AUC. Closer category frequencies do not, in this run, mean a more useful training set. Exact training matches are absent. That check is not a privacy guarantee.

![Figure 9. Categorical fidelity and synthetic-only validation ROC-AUC.](results/figures/study/13_fidelity_utility.png)
""".strip()


def _scarcity_validation(paired: pd.DataFrame) -> str:
    aug = _roc(paired, "scarcity_augmentation")
    repl = _roc(paired, "scarcity_replacement")
    note = _practical(aug)
    return f"""
# 11. Scarcity on the validation partition

The pre-written rule selects a 10 percent synthetic share for both generators. Augmentation keeps the real budget and adds rows. Replacement holds the size fixed.

{_counts(aug, "For validation augmentation ROC-AUC")} {note}

{_counts(repl, "For validation replacement ROC-AUC")}

![Figure 10. Validation ROC-AUC under real-data budgets.](results/figures/study/14_scarcity.png)
""".strip()


def _test_results(test: pd.DataFrame, paired: pd.DataFrame) -> str:
    roc = _roc(paired, "main")
    aug = _roc(paired, "scarcity_augmentation")
    repl = _roc(paired, "scarcity_replacement")
    means = _mean_roc_table(test.loc[test["stage"] == "main"])
    return f"""
# 12. Held-out test evaluation

The saved test file contains {len(test)} completed runs, all labelled with the test partition. The authorization record names the held-out test and matches the freeze-file hash. Its reviewer field is the text saved when the file was created at the user's request on 17 November 2026. No configuration was changed after these scores were known.

The five-seed real-only test means are {_real_means(test)}.

Table 4. Mean test ROC-AUC in the main experiment.

{means}

![Figure 11. Test ROC-AUC by synthetic share.](results/figures/final/11_test_ratio_curves.png)

![Figure 12. Mean validation ROC-AUC against mean test ROC-AUC.](results/figures/final/12_validation_vs_test.png)

{_counts(roc, "For main-experiment test ROC-AUC")}

Table 5. Paired test ROC-AUC differences in the main experiment.

{_paired_table(roc)}

![Figure 13. Paired test ROC-AUC differences from the matched-seed real-only run.](results/figures/final/13_test_differences.png)

{_secondary(paired, "main", "test")}

{_counts(aug, "For test augmentation ROC-AUC")} {_practical(aug)}

Table 6. Paired test ROC-AUC differences for scarcity augmentation.

{_paired_table(aug)}

{_counts(repl, "For test replacement ROC-AUC")}

Table 7. Paired test ROC-AUC differences for fixed-size replacement.

{_paired_table(repl)}

![Figure 14. Test ROC-AUC under real-data budgets.](results/figures/final/14_test_scarcity.png)
""".strip()


def _comparison(val_paired: pd.DataFrame, test_paired: pd.DataFrame) -> str:
    left = _roc(val_paired, "main")
    right = _roc(test_paired, "main")
    keys = ["generator", "synthetic_share", "classifier"]
    merged = left.merge(right, on=keys, suffixes=("_validation", "_test"))
    disagree = merged.loc[merged["direction_validation"] != merged["direction_test"]]
    rows = [
        (
            "Main",
            _count_word(left, "lower"),
            _count_word(left, "inconclusive"),
            _count_word(left, "higher"),
            _count_word(right, "lower"),
            _count_word(right, "inconclusive"),
            _count_word(right, "higher"),
        )
    ]
    for label, stage in (
        ("Augmentation", "scarcity_augmentation"),
        ("Replacement", "scarcity_replacement"),
    ):
        a = _roc(val_paired, stage)
        b = _roc(test_paired, stage)
        rows.append(
            (
                label,
                _count_word(a, "lower"),
                _count_word(a, "inconclusive"),
                _count_word(a, "higher"),
                _count_word(b, "lower"),
                _count_word(b, "inconclusive"),
                _count_word(b, "higher"),
            )
        )
    detail = "Every main ROC-AUC contrast has the same interval reading on both partitions."
    if len(disagree):
        names = [
            f"{DISPLAY[row.generator]} {DISPLAY[row.classifier]} at {row.synthetic_share:.0%} "
            f"({row.direction_validation} on validation, {row.direction_test} on test)"
            for row in disagree.itertuples(index=False)
        ]
        detail = "The main ROC-AUC readings that differ are: " + "; ".join(names) + "."
    higher = _roc(val_paired, "scarcity_augmentation")
    higher = higher.loc[higher["direction"] == "higher"]
    higher_text = "No validation augmentation contrast was higher."
    if len(higher):
        difference = float(higher.iloc[0]["mean_difference"])
        higher_text = (
            f"The validation augmentation contrast that was higher, with mean difference {difference:.6f}, "
            "is not a test-set gain."
        )
    return f"""
# 13. Validation compared with the test set

Table 8. Validation and test interval readings for ROC-AUC.

{_table(["Analysis", "Validation lower", "Validation inconclusive", "Validation higher", "Test lower", "Test inconclusive", "Test higher"], rows)}

{detail} {higher_text} Figure 12 plots the mean ROC-AUC of each main configuration on the two partitions. Points near the diagonal kept their rank. Points below the diagonal scored lower on the test partition than on the validation partition.
""".strip()


def _discussion(paired: pd.DataFrame, bundle: dict) -> str:
    test_main = _roc(paired, "main")
    below = test_main.loc[test_main["synthetic_share"] < 1]
    copula = below.loc[below["generator"] == "gaussian_copula", "mean_difference"].mean()
    ctgan = below.loc[below["generator"] == "ctgan", "mean_difference"].mean()
    aug = _roc(paired, "scarcity_augmentation")
    return f"""
# 14. Discussion

## 14.1 Synthetic share

RQ1 asked whether some synthetic share would raise ROC-AUC. {_counts(test_main, "On the test set")} The pre-written hope of at least one mixed share above the real-only baseline is not supported. Performance falls as the synthetic share rises, and the 100 percent condition is the weakest. That condition replaces the real rows. It is not evidence about a small supplement by itself, but the 10, 25, 50, and 75 percent rows are already lower as well.

## 14.2 Generator choice

RQ2 asked which generator is less harmful under this protocol. For shares below 100 percent, the mean of the paired test ROC-AUC differences is {ctgan:.4f} for CTGAN and {copula:.4f} for the Gaussian copula. CTGAN loses less. Neither generator improves on real-only training. This is a description of one table and five seeds, not a ranking to be reused on another dataset.

## 14.3 Scarce real data

RQ3 asked whether synthetic rows help when real rows are fewer. {_counts(aug, "On the test augmentation contrasts")} Adding synthetic rows increases the training size and still does not raise ROC-AUC above the budget baseline by a margin the interval can support. Fixed-size replacement, which holds the training size constant, does not show a gain either. The validation result that was higher by a fraction of a thousandth of an ROC-AUC point does not reappear as a test-set improvement.

## 14.4 Fidelity and utility

RQ4 asked whether fidelity and utility move together. They do not, in the direction a user would hope. The copula matches categories more closely and produces the weaker synthetic-only classifier. CTGAN is farther from the category frequencies and closer as a training set, while both remain below the real-only baseline. The positive Spearman correlation in Section 10 is a description of ten generator-seed points from one dataset. It is not a licence to choose a generator by total variation.
""".strip()


def _limitations(bundle: dict) -> str:
    audit = bundle["partition_audit"]
    return f"""
# 15. Limitations

Five seeds on one split give a t interval with four degrees of freedom. Repeating the seed does not create five independent datasets. A result on Adult does not establish the same pattern for another table, another class balance, or another generator.

CTGAN used 80 epochs rather than the library default of 300. The shorter budget was fixed from runtime, before utility results, and it can leave the generator under-trained. The Gaussian copula uses beta marginals because that is the library default, not because the features were shown to be beta.

The same validation partition was used for Phase 1 model selection and for the scarcity-ratio rule. The rule was written before the utility numbers were read, and the test set was not used to pick the 10 percent share, but the validation scores are not a fresh sample.

The pooled table has {audit["cross_partition_duplicate_groups"]} cross-partition duplicate groups. Keeping them preserves the frozen split and can make validation scores slightly optimistic when an identical row sits in both training and validation.

Exact-match and near-duplicate rates depend on the encoding and the threshold. They are not a privacy audit.

Privacy, fairness, and a second dataset were outside the protocol. The duplicate checks are not the privacy evaluation discussed by Jordon et al. (2022), Pereira et al. (2024), or Adams et al. (2025).

A coding assistant drafted code and report prose. It did not choose hyperparameters, shares, or generators, and it did not use the held-out test for selection. Every number in this report is computed from the saved tables.
""".strip()


def _conclusion(paired: pd.DataFrame) -> str:
    return f"""
# 16. Conclusion

{_counts(_roc(paired, "main"), "On the held-out test partition")} Synthetic augmentation did not improve ROC-AUC for these generators, classifiers, and shares. The same conclusion holds for the validation partition, with one inconclusive mixed-share interval there and none on the test set. Scarcity does not reverse the finding. Fidelity and utility disagree: the generator that is closer in category frequencies is the weaker training set.

These statements describe the frozen Adult experiment. They are not a general claim that synthetic data cannot help, and they are not a reason to retune epochs or shares on the test set.
""".strip()


def _references() -> str:
    return """
# References

Achterberg, J., Haas, M., van Dijk, B., & Spruit, M. (2025). Fidelity-agnostic synthetic data generation improves utility while retaining privacy. *Patterns, 6*(10), 101287. https://doi.org/10.1016/j.patter.2025.101287

Adams, T., Birkenbihl, C., Otte, K., Ng, H. G., Rieling, J. A., Näher, A.-F., Sax, U., Prasser, F., Fröhlich, H., & Alzheimer’s Disease Neuroimaging Initiative. (2025). On the fidelity versus privacy and utility trade-off of synthetic patient data. *iScience, 28*(5), 112382. https://doi.org/10.1016/j.isci.2025.112382

Becker, B., & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

Decruyenaere, A., Dehaene, H., Rabaey, P., Polet, C., Decruyenaere, J., Vansteelandt, S., & Demeester, T. (2024). The real deal behind the artificial appeal: Inferential utility of tabular synthetic data. *Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence, 244*, 966–996. https://proceedings.mlr.press/v244/decruyenaere24a.html

Fonseca, J., & Bacao, F. (2023). Tabular and latent space synthetic data generation: A literature review. *Journal of Big Data, 10*, 115. https://doi.org/10.1186/s40537-023-00792-7

Hernandez, M., Epelde, G., Alberdi, A., Cilla, R., & Rankin, D. (2022). Synthetic data generation for tabular health records: A systematic review. *Neurocomputing, 493*, 28–45. https://doi.org/10.1016/j.neucom.2022.04.053

Jordon, J., Szpruch, L., Houssiau, F., Bottarelli, M., Cherubin, G., Maple, C., Cohen, S. N., & Weller, A. (2022). Synthetic data—what, why and how? arXiv. https://doi.org/10.48550/arXiv.2205.03257

Kurakova, A., & Homayouni, H. (2025). A comprehensive evaluation framework for synthetic medical tabular data generation. *Journal of Biomedical Informatics, 171*, 104939. https://doi.org/10.1016/j.jbi.2025.104939

Manousakas, D., & Aydöre, S. (2023). On the usefulness of synthetic tabular data generation. arXiv. https://doi.org/10.48550/arXiv.2306.15636

Pereira, M., Kshirsagar, M., Mukherjee, S., Dodhia, R., Lavista Ferres, J., & de Sousa, R. (2024). Assessment of differentially private synthetic data for utility and fairness in end-to-end machine learning pipelines for tabular data. *PLOS ONE, 19*(2), e0297271. https://doi.org/10.1371/journal.pone.0297271

Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling tabular data using conditional GAN. *Advances in Neural Information Processing Systems, 32*. https://doi.org/10.5555/3454287.3454946
""".strip()


def _appendix(bundle: dict) -> str:
    audit = bundle["audit"]
    return f"""
# Appendix A. Reproduction

Validation metrics: `results/tables/study/run_metrics.csv`. Test metrics: `results/final_evaluation/test_run_metrics.csv`. Paired test differences: `results/final_evaluation/paired_differences.csv`. The freeze hash recorded with the evaluation is `{audit["freeze_sha256"]}`. The source-tree fingerprint recorded at evaluation time is `{audit["source_tree_fingerprint"]}`. The project directory was not a Git repository, so there is no commit hash.

The validation command is `uv run python scripts/run_validation_study.py`. The held-out command is `uv run python scripts/run_final_test.py --final-test`. That command was not run again for this report. The original freeze file still records `test_metrics_computed: false`. The separate file `results/frozen/post_evaluation_status.json` records that the previously authorized test evaluation was later completed and checked. Tests are `uv run pytest`. The integration test fits a toy generator and is excluded from the fast suite.

CTGAN epochs remain 80. Seeds remain 42, 123, 2024, 3407, and 777. The scarcity share remains 10 percent.
""".strip()


def _roc(paired: pd.DataFrame, stage: str) -> pd.DataFrame:
    return paired.loc[(paired["stage"] == stage) & (paired["metric"] == "roc_auc")].copy()


def _counts(frame: pd.DataFrame, label: str) -> str:
    counts = frame["direction"].value_counts().to_dict() if len(frame) else {}
    return (
        f"{label}, {int(counts.get('higher', 0))} ROC-AUC intervals lay entirely above zero, "
        f"{int(counts.get('lower', 0))} lay entirely below zero, and "
        f"{int(counts.get('inconclusive', 0))} covered zero."
    )


def _count_word(frame: pd.DataFrame, direction: str) -> str:
    if frame.empty:
        return "0"
    return str(int((frame["direction"] == direction).sum()))


def _practical(frame: pd.DataFrame) -> str:
    notes = []
    for row in frame.loc[frame["direction"] == "higher"].itertuples(index=False):
        if abs(float(row.mean_difference)) < 0.001:
            notes.append(
                f"The higher {DISPLAY[row.generator]} contrast for {DISPLAY[row.classifier]} "
                f"has mean difference {float(row.mean_difference):.6f}. "
                "A ROC-AUC change of this size is not a meaningful practical gain."
            )
    return " ".join(notes)


def _secondary(paired: pd.DataFrame, stage: str, label: str) -> str:
    bits = []
    for metric in ("f1", "precision", "recall", "balanced_accuracy"):
        frame = paired.loc[(paired["stage"] == stage) & (paired["metric"] == metric)]
        counts = frame["direction"].value_counts().to_dict()
        bits.append(
            f"{metric} {int(counts.get('lower', 0))} lower, "
            f"{int(counts.get('inconclusive', 0))} inconclusive, "
            f"{int(counts.get('higher', 0))} higher"
        )
    return f"On the {label} partition, the paired secondary-metric readings are: " + "; ".join(bits) + "."


def _real_means(test: pd.DataFrame) -> str:
    real = test.loc[(test["stage"] == "main") & (test["generator"] == "real")]
    bits = []
    for classifier, group in real.groupby("classifier"):
        bits.append(f"{DISPLAY[classifier]} {group['roc_auc'].mean():.4f} (SD {group['roc_auc'].std(ddof=1):.4f})")
    return "; ".join(bits)


def _mean_roc_table(main: pd.DataFrame) -> str:
    rows = []
    grouped = main.groupby(["generator", "synthetic_share", "classifier"], dropna=False)["roc_auc"]
    for (generator, share, classifier), values in grouped:
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        rows.append(
            (
                DISPLAY.get(generator, generator),
                f"{float(share):.0%}",
                DISPLAY.get(classifier, classifier),
                f"{float(values.mean()):.4f} ({std:.4f})",
            )
        )
    return _table(["Generator", "Share", "Model", "Mean ROC-AUC (SD)"], rows)


def _paired_table(frame: pd.DataFrame) -> str:
    rows = []
    ordered = frame.sort_values(["generator", "synthetic_share", "classifier"])
    for row in ordered.itertuples(index=False):
        rows.append(
            (
                DISPLAY.get(row.generator, row.generator),
                f"{float(row.synthetic_share):.0%}",
                DISPLAY.get(row.classifier, row.classifier),
                f"{row.mean_difference:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}]",
                row.direction,
            )
        )
    return _table(["Generator", "Share", "Model", "Mean difference [interval]", "Reading"], rows)


def _table(header: list[str], rows: list[tuple]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)
