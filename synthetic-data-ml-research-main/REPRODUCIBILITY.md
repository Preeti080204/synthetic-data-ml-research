# Reproducing the validation study

The project uses Python 3.12 and uv. Commands are run from `Synthetic_Data_ML`.

## Inspect the saved results

These commands do not retrain a generator and do not rescore the held-out test:

```bash
uv run python scripts/run_project.py
uv run python scripts/run_project.py figures
uv run python scripts/run_project.py report
uv run python scripts/run_project.py test
```

`./run.sh` and `.\run.ps1` call the same script. `reproduce` only prints the pipeline unless `--execute` is added. `--execute` runs `scripts/run_validation_study.py` and refuses `scripts/run_final_test.py`.

## Environment

```bash
uv sync --all-groups
```

The study environment includes SDV, PyYAML, and python-docx in addition to the Phase 1 packages. Phase 1 was originally executed with pandas 3.1.1. Installing SDV moved the environment to pandas 2.3.3. The Phase 1 tests were re-run after that change. The saved Phase 1 result files were not rewritten.

## Phase 1 baseline

```bash
uv run python run_phase1.py
uv run pytest
```

Do not change the split seed in `configs/phase1.json`. The synthetic-data study reads `data/processed/split_indices.csv` and does not call the split function for its primary partition.

## Validation study

```bash
uv run python scripts/run_validation_study.py
```

This command fits the Gaussian copula and CTGAN on the real training partition, runs the six synthetic shares, the quality checks, and the 25/50/75 percent scarcity analyses, then writes the validation report. Completed generator fits and classifier runs are skipped on a later run. It does not load the test partition and it does not compute test metrics.

A tiny pipeline check, which is not a scientific result, is:

```bash
uv run python scripts/run_validation_study.py --smoke
```

Smoke outputs stay under `results/smoke/` and are gitignored.

## Tests

```bash
uv run pytest
uv run pytest -m integration -o addopts=-q
```

The fast suite excludes the integration test. The integration test fits both generators for one epoch on 80 toy rows.

## Held-out evaluation

The held-out run has already been executed. Its metrics are `results/final_evaluation/test_run_metrics.csv`. Do not run the scorer again unless that file fails `src/reporting/final_audit.py`. A later run reuses completed test records instead of refitting.

```bash
uv run python scripts/run_final_test.py
```

Without `--final-test` the command exits before reading the test frame. The authorized form also requires `results/frozen/FINAL_TEST_AUTHORIZED`. That file was created on 17 November 2026 at the user's request. Its reviewer field is "User request, 2026-11-17". The final report is rebuilt from the saved table:

```bash
uv run python -c "from reporting.final_report import write_final_report; write_final_report()"
```

## Main outputs

- Per-run metrics: `results/raw_metrics/`
- Aggregates: `results/tables/study/`
- Paired intervals: `results/statistical_tests/paired_differences.csv`
- Figures: `results/figures/study/`
- Synthetic tables: `data/synthetic/main/` and `data/synthetic/scarcity/`
- Validation metrics and scientific figures: `results/tables/study/` and `results/figures/study/`.
- Test metrics: `results/final_evaluation/test_run_metrics.csv`
- Final report: `reports/final/Preeti_Gorial_Synthetic_Data_Research_Report.pdf`
- Freeze manifest: `results/frozen/validation_freeze.json`

CTGAN epochs are fixed in `configs/generators.yaml` at 80. That choice was made from a timing benchmark before utility results and is not a tuning result.
