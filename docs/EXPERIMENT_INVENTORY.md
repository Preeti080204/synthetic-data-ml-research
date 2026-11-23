# Experiment inventory

The held-out evaluation was executed once and saved. It was not rerun for the final report.

| Requirement | Code | Configuration | Artifact | Test | Status |
| --- | --- | --- | --- | --- | --- |
| Preserve the Phase 1 split and metrics | `src/experiments/data.py` | `configs/phase1.json` seed 42 | `data/processed/split_indices.csv`, `docs/phase1_artifact_hashes.json` | `tests/test_study.py` | Complete. Hashes match. |
| Gaussian copula, training rows only, target included | `src/generators/tabular.py` | `configs/generators.yaml` | `data/synthetic/main/gaussian_copula/` | `tests/test_generator_integration.py` | Complete for five seeds. |
| CTGAN, 80 epochs, CPU | `src/generators/tabular.py`, subprocess in `scripts/run_validation_study.py` | `configs/generators.yaml` | `data/synthetic/main/ctgan/` | integration test uses one epoch on toy rows | Complete. MPS fit stalled and was not used. 300 library-default epochs were not run. |
| Six synthetic shares, fixed N = 34,189 | `src/experiments/design.py` | `configs/experiments.yaml` | `results/tables/study/run_metrics.csv` | mixture-count test | Complete. 165 main runs. |
| Three classifiers, Phase 1 hyperparameters | `src/experiments/train.py` | `results/tables/selected_configurations.json` | `results/raw_metrics/` | Phase 1 model tests plus study sampling tests | Complete. No retune. |
| Five seeds and paired intervals | `src/analysis/summarize.py` | seeds in both YAML files | `results/statistical_tests/paired_differences.csv` | interval and ratio-rule tests | Complete. n = 5 is treated as descriptive. |
| Quality, duplicates, fidelity versus utility | `src/quality/fidelity.py` | `configs/experiments.yaml` quality block | `results/tables/study/quality_summary.csv`, figures 9–13 | not a separate unit test of the full files | Complete on the training partition. Not a privacy audit. |
| Scarcity at 25, 50, 75 percent | `scripts/run_validation_study.py` | ratio rule in `configs/experiments.yaml` and `docs/ANALYSIS_PLAN.md` | scarcity rows in `run_metrics.csv` | budget arithmetic is in the design tests | Complete. Augmentation and replacement are separate. Selected share is 0.10. |
| Held-out test evaluation | `scripts/run_final_test.py`, `src/experiments/final_eval.py` | freeze manifest and `results/frozen/FINAL_TEST_AUTHORIZED` | `results/final_evaluation/test_run_metrics.csv` | `src/reporting/final_audit.py` | Executed once. 390 test runs passed the integrity audit. The scorer was not run again. |
| Figures | `src/reporting/figures.py` | `configs/reporting.yaml` | `results/figures/study/` | rendered with the report | Complete. 15 figure families, PNG, PDF, and SVG. |
| Validation-stage evidence | `scripts/run_validation_study.py` | `configs/experiments.yaml` | `results/tables/study/run_metrics.csv` and `results/figures/study/` | Saved-run integrity tests | Complete. Intermediate report exports were retired; preserved metrics remain. |
| Final report | `src/reporting/final_report.py` | `configs/reporting.yaml` | `reports/final/Preeti_Gorial_Synthetic_Data_Research_Report.pdf` and `.docx` | built from the saved test table | Written from the saved 390 test runs. |
| Documentation | `README.md`, `REPRODUCIBILITY.md`, `DATA_DICTIONARY.md` | — | this inventory | `uv run pytest` | Complete. |

The 22 October instructions specify Adult and scarcity budgets of 25, 50, and 75 percent. Those instructions take priority over the proposal’s open dataset choice and its 10 and 100 percent scarcity budgets. The ten proposal studies are reviewed in the final report. The private source files are not published.
