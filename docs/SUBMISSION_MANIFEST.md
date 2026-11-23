# Submission manifest

This document records the **published GitHub repository**, not the location of a historical local ZIP.

Repository: https://github.com/Geek-MJ/synthetic-data-ml-research

The canonical research evidence is retained in Git. Superseded interim and Phase 1 reports are intentionally not published. See [Release verification](RELEASE_CHECKLIST.md) for read-only checks.

## Main submission documents

- `README.md` — research overview, summary results, and quick-start commands.
- `reports/final/Preeti_Gorial_Synthetic_Data_Research_Report.pdf` — academic submission report.
- `reports/final/Preeti_Gorial_Synthetic_Data_Research_Report.docx` — editable report.
- `reports/final/Preeti_Gorial_Synthetic_Data_Research_Report.md` — inspectable source with GitHub-relative figure links.
- `docs/PROFESSOR_REQUIREMENTS_AUDIT.md` — mapping of the nine professor requirements.
- `REPRODUCIBILITY.md`, `DATA_DICTIONARY.md`, and `docs/ANALYSIS_PLAN.md` — research protocol and reproducibility.
- `src/`, `scripts/`, `configs/`, `notebooks/`, `tests/`, `pyproject.toml`, `uv.lock`, `run.sh`, `run.ps1`.

## Essential experiment evidence: intentionally preserved

- `data/raw/` and `data/processed/` with the original pooled data and frozen split indices.
- `data/synthetic/` including generator configurations, fitted synthesizers, and synthetic datasets.
- `models/` including all existing Phase 1 models.
- `results/raw_metrics/`, `results/tables/`, `results/statistical_tests/`, `results/figures/`.
- `results/final_evaluation/` including 390 saved test results and the integrity audit.
- `results/frozen/validation_freeze.json` (the original pre-test snapshot).
- `results/frozen/post_evaluation_status.json` (records the completed test after the freeze).
- `results/provenance/legacy_cache_ledger.json` and the Phase 1 artifact hashes.
- The single **final report**, supplied as PDF, editable DOCX, and source Markdown in `reports/final/`. Historical Phase 1 and interim reports were retired, while underlying evidence remains.

The original freeze must remain unchanged. The separate post-evaluation status explains why its pre-test flags remain false. No experiment or model was retrained as part of repository cleanup.

## Excluded from public Git

- `.venv/`, Python caches, notebook checkpoints, macOS metadata, and `results/smoke/`.
- `docs/source_documents/`: the private approved proposal and the professor's correspondence.
- `results/frozen/FINAL_TEST_AUTHORIZED`: local authorization file.
- Local credentials and environment variables as defined in `.gitignore`.

The public report and the nine-section audit include the scientific synthesis of the private sources, without publishing the private documents.

## Review and release safety

- The completed results are intended to be inspected using `uv run python scripts/run_project.py summary`.
- The Markdown report's figures must resolve relative to `reports/final/`.
- The GitHub Actions fast test suite must pass before merging changes to the default branch.
- The generator-fitting integration test and Windows PowerShell execution require separate environments; they are not covered by the Linux fast workflow.
- Do **not** remove scientific artifacts simply because there are many files. No model, raw data, synthetic data, metric, frozen record, or report has been removed in this documentation cleanup.

## Packaging

A separate ZIP may be built from a clean checkout for the university submission. Do not publish historical absolute paths such as a private user's Desktop or local submission directory as though they were portable artifacts.
