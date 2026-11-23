# Release verification

This repository publishes the completed study and its preserved experiment evidence. The academic report is [here](../reports/final/Preeti_Gorial_Synthetic_Data_Research_Report.pdf).

## Fast checks (do not retrain generators or rescore the test)

```bash
uv sync --frozen --all-groups
uv run --frozen pytest
uv run --frozen python scripts/run_project.py summary
uv run --frozen python scripts/run_project.py figures
```

The fast pytest suite excludes the generator-fitting integration test. GitHub Actions runs the locked fast suite and saved-results summary for every pull request and push to `main`. The figures command only regenerates cached-result plots; it does not modify frozen metrics.

## Research evidence that must be retained

- `data/processed/split_indices.csv` and the real/synthetic data required for reproducibility.
- `models/` and `data/synthetic/` completed model/generator artifacts.
- `configs/`, `results/tables/`, `results/raw_metrics/`, `results/statistical_tests/`, and `results/final_evaluation/`.
- `results/frozen/validation_freeze.json` (historical pre-test freeze) and `results/frozen/post_evaluation_status.json` (the subsequent completed test record).
- `reports/final/` (one complete report in PDF, DOCX, and Markdown), `src/`, `scripts/`, `tests/`, and the dependency lockfile.

The original freeze intentionally retains its pre-test status. Never rewrite it to make the status appear current; the separate post-evaluation record is authoritative for the subsequent test.

## Private and local-only content

`docs/source_documents/` contains the approved proposal and professor correspondence on the author's computer; it is not published. `results/frozen/FINAL_TEST_AUTHORIZED` is also intentionally excluded. The report and [requirements audit](PROFESSOR_REQUIREMENTS_AUDIT.md) preserve their scientific conclusions without publishing private correspondence.

Do not run the final-test scorer again to validate a documentation-only change. Do not retrain CTGAN for routine review.

## Cross-platform scope

The PowerShell entrypoint is provided for Windows, but Windows execution is not verified by the Linux GitHub Actions job. The workflow verifies Python 3.12 on Linux only.

## GitHub contents

Reproducible research artifacts are intentionally included, which makes this a larger repository than a typical Python library. No scientific dataset, model, frozen manifest, or completed experiment should be removed just to reduce file count.
