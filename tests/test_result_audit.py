"""Independent checks of the saved validation tables."""

import numpy as np
import pandas as pd
from scipy import stats

from experiments.design import augmentation_synthetic_count, mixture_counts, synthetic_count
from utils.paths import project_root


def _runs():
    return pd.read_csv(project_root() / "results" / "tables" / "study" / "run_metrics.csv")


def _paired():
    return pd.read_csv(project_root() / "results" / "statistical_tests" / "paired_differences.csv")


def test_completed_run_count_and_metrics():
    runs = _runs()
    assert len(runs) == 390
    assert (runs["status"] == "completed").all()
    for column in ("roc_auc", "f1", "precision", "recall", "balanced_accuracy"):
        assert runs[column].notna().all()
    assert (runs["partition"] == "validation").all()
    assert (runs["n_train"] == runs["n_real"] + runs["n_synthetic"]).all()
    assert (~runs["test_metrics_computed"].astype(str).str.lower().isin(["true", "1"])).all()


def test_five_seeds_and_main_training_size():
    runs = _runs()
    counted = runs.groupby(
        ["stage", "generator", "synthetic_share", "classifier", "budget"], dropna=False
    )["seed"].nunique()
    assert (counted == 5).all()
    main = runs.loc[runs["stage"] == "main"]
    assert (main["n_train"] == 34189).all()
    for row in main[["synthetic_share", "n_real", "n_synthetic"]].drop_duplicates().itertuples(index=False):
        expected_real, expected_syn = mixture_counts(34189, float(row.synthetic_share))
        assert (int(row.n_real), int(row.n_synthetic)) == (expected_real, expected_syn)


def test_paired_intervals_match_an_independent_calculation():
    runs = _runs()
    paired = _paired()
    main = paired.loc[(paired["stage"] == "main") & (paired["metric"] == "roc_auc")]
    assert (main["direction"] == "lower").sum() == 29
    assert (main["direction"] == "inconclusive").sum() == 1
    assert (main["direction"] == "higher").sum() == 0
    aug = paired.loc[(paired["stage"] == "scarcity_augmentation") & (paired["metric"] == "roc_auc")]
    assert (aug["direction"] == "lower").sum() == 15
    assert (aug["direction"] == "inconclusive").sum() == 2
    assert (aug["direction"] == "higher").sum() == 1
    higher = aug.loc[aug["direction"] == "higher"].iloc[0]
    assert 0 < float(higher.mean_difference) < 0.001

    baseline = runs.loc[(runs["stage"] == "main") & (runs["synthetic_share"] == 0)]
    synthetic = runs.loc[(runs["stage"] == "main") & (runs["synthetic_share"] > 0)]
    for row in main.itertuples(index=False):
        left = synthetic.loc[
            (synthetic["generator"] == row.generator)
            & np.isclose(synthetic["synthetic_share"], row.synthetic_share)
            & (synthetic["classifier"] == row.classifier)
        ].sort_values("seed")
        right = baseline.loc[baseline["classifier"] == row.classifier].sort_values("seed")
        assert list(left["seed"]) == list(right["seed"])
        difference = left["roc_auc"].to_numpy() - right["roc_auc"].to_numpy()
        mean = float(difference.mean())
        std = float(difference.std(ddof=1))
        half = float(stats.t.ppf(0.975, len(difference) - 1) * std / np.sqrt(len(difference)))
        assert abs(mean - float(row.mean_difference)) < 1e-12
        assert abs(std - float(row.std_difference)) < 1e-12
        assert abs((mean - half) - float(row.ci_low)) < 1e-12
        assert abs((mean + half) - float(row.ci_high)) < 1e-12


def test_scarcity_definitions():
    runs = _runs()
    aug = runs.loc[runs["stage"] == "scarcity_augmentation"]
    repl = runs.loc[runs["stage"] == "scarcity_replacement"]
    assert set(np.round(aug["budget"].unique(), 2)) == {0.25, 0.5, 0.75}
    assert (aug["n_train"] > aug["n_real"]).all()
    assert np.isclose(aug["synthetic_share"], 0.10).all()
    assert (repl["n_train"] == repl["n_real"] + repl["n_synthetic"]).all()
    budgets = {0.25: 8547, 0.50: 17094, 0.75: 25642}
    for budget, n_budget in budgets.items():
        assert int(round(34189 * budget)) == n_budget
        subset = aug.loc[np.isclose(aug["budget"], budget)]
        expected_syn = augmentation_synthetic_count(n_budget, 0.10)
        assert (subset["n_real"] == n_budget).all()
        assert (subset["n_synthetic"] == expected_syn).all()
        replaced = repl.loc[np.isclose(repl["budget"], budget)]
        expected_fixed = synthetic_count(n_budget, 0.10)
        assert (replaced["n_synthetic"] == expected_fixed).all()
        assert (replaced["n_real"] == n_budget - expected_fixed).all()
