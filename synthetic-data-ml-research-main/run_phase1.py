"""Execute the Phase 1 real-data baseline.

The test partition is audited for composition and source-row identity only.
This script does not fit on it, tune on it, or score it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data.inspect import inspect_dataset
from data.load_data import load_adult_combined, save_processed_dataset
from data.preprocessing import split_xy
from data.split_data import audit_partitions, make_splits, save_partition_tables, save_splits
from models.baseline import fit_baselines
from models.evaluation import save_model_figures
from utils.environment import collect_environment
from utils.paths import (
    data_processed_dir,
    ensure_project_dirs,
    figures_dir,
    load_config,
    models_dir,
    results_dir,
    tables_dir,
    write_json,
)


def _configure_logging() -> None:
    log_path = results_dir() / "phase1_run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )


def main() -> None:
    ensure_project_dirs()
    _configure_logging()
    for noisy in ("fontTools", "fontTools.subset", "PIL"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    logger = logging.getLogger("phase1")
    config = load_config()
    seed = int(config["seed"])
    np.random.seed(seed)

    logger.info("Loading Adult Census Income files.")
    loaded = load_adult_combined()
    frame = loaded.frame
    processed_path = data_processed_dir() / "adult_combined.csv"
    save_processed_dataset(frame, processed_path)
    logger.info(
        "Pooled dataset: %s rows, %s predictors. Trailing periods normalized: %s.",
        loaded.metadata["n_rows"],
        loaded.metadata["n_features"],
        loaded.metadata["trailing_period_labels"],
    )

    logger.info("Creating the stratified 70/15/15 partition with seed %s.", seed)
    splits = make_splits(frame, config)
    split_path = data_processed_dir() / "split_indices.csv"
    save_splits(frame, splits, split_path)
    audit = audit_partitions(frame, splits)
    save_partition_tables(audit, tables_dir())
    if any(count != 0 for count in audit["overlap_source_row_id"].values()):
        raise RuntimeError(f"Source-row overlap detected: {audit['overlap_source_row_id']}")
    logger.info(
        "Partition sizes: train=%s validation=%s test=%s.",
        audit["sizes"]["train"],
        audit["sizes"]["validation"],
        audit["sizes"]["test"],
    )

    train_frame = frame.iloc[splits["train"]].reset_index(drop=True)
    validation_frame = frame.iloc[splits["validation"]].reset_index(drop=True)
    # The test rows stay inside `frame` for the identity audit above.
    # They are not passed to inspection plots that condition on income,
    # and they are not passed to fitting or scoring.

    logger.info("Writing inspection tables and figures.")
    inspection = inspect_dataset(
        frame,
        train_frame,
        tables_dir(),
        figures_dir(),
    )
    if (
        inspection["duplicates"]["exact_duplicate_extra_rows"]
        != audit["exact_duplicate_extra_rows"]
    ):
        raise RuntimeError("Duplicate-row counts disagree across audit functions.")

    logger.info("Fitting baseline models on the training partition.")
    modeling = fit_baselines(
        train_frame,
        validation_frame,
        config,
        models_dir(),
        tables_dir(),
    )
    logger.info("Selected model by validation ROC-AUC: %s.", modeling["best_model"])
    for name in modeling["ranking"]:
        metrics = modeling["models"][name]["metrics"]
        logger.info(
            "%s validation ROC-AUC=%.4f F1=%.4f precision=%.4f recall=%.4f "
            "balanced_accuracy=%.4f",
            name,
            metrics["roc_auc"],
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            metrics["balanced_accuracy"],
        )

    pipelines = {
        name: joblib.load(modeling["models"][name]["model_path"])
        for name in config["models"]
    }
    metrics_by_model = {}
    for name, item in modeling["models"].items():
        metrics_by_model[name] = {"display_name": item["display_name"], **item["metrics"]}
    x_val, y_val = split_xy(validation_frame)
    figure_paths = save_model_figures(
        pipelines,
        x_val,
        y_val,
        metrics_by_model,
        figures_dir(),
        tables_dir(),
        threshold=float(config["decision_threshold"]),
    )

    environment = collect_environment()
    test_status = {
        "phase": 1,
        "test_predictions_computed": False,
        "test_metrics_computed": False,
        "test_used_for_model_selection": False,
        "test_used_for_preprocessing_fit": False,
        "validation_metrics_computed": True,
        "identity_audit_includes_test_rows": True,
        "identity_audit_purpose": (
            "Confirm that source-row identifiers do not overlap and report "
            "identical feature-label contents that fall in more than one "
            "partition. No predictive metric was calculated on the test partition."
        ),
    }
    write_json(results_dir() / "test_set_status.json", test_status)
    write_json(tables_dir() / "selected_configurations.json", modeling["models"])
    write_json(results_dir() / "environment.json", environment)

    summary = {
        "seed": seed,
        "load": loaded.metadata,
        "inspection": inspection,
        "audit": audit,
        "modeling": {
            "models": modeling["models"],
            "ranking": modeling["ranking"],
            "best_model": modeling["best_model"],
            "selection_metric": modeling["selection_metric"],
            "tie_breaker_metric": modeling["tie_breaker_metric"],
            "decision_threshold": modeling["decision_threshold"],
            "search_rows": modeling["search_rows"],
        },
        "environment": environment,
        "test_set_status": test_status,
        "figure_paths": figure_paths,
    }
    write_json(results_dir() / "phase1_summary.json", summary)

    # Only the final research report is published. Save baseline models and
    # metrics without producing a redundant Phase 1 PDF or Markdown file.
    logger.info(
        "Phase 1 complete. Test-set performance was not computed. "
        "Validation metrics are not test metrics."
    )


if __name__ == "__main__":
    main()
