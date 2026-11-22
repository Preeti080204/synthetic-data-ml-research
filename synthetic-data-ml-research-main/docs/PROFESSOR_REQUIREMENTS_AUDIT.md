# Requirements audit

Checked on 23 November 2026 against the approved proposal and the 22 October experimental instructions. The private source files stay in `docs/source_documents/` and are gitignored. This note records the methodological requirements, not the correspondence.

Where the proposal and the later instructions differ, the experiment follows the later instructions. Frozen configurations were not changed for this audit.

| Section | Requirement | Evidence | Status |
| --- | --- | --- | --- |
| 1. Dataset and problem | Adult Census Income; binary income above $50,000; document source, features, missing values, class balance, and preprocessing | `DATA_DICTIONARY.md`, `results/tables/class_distribution.csv`, `results/tables/partition_audit.json` | Met |
| 2. Split and protocol | Stratified 70/15/15 split; test rows real and unused for fitting or selection; preprocessing fit on training rows; same partitions for every condition | `data/processed/split_indices.csv`: 34,189 / 7,326 / 7,327. Source-row overlap is zero | Met |
| 3. Baselines | Logistic regression, random forest, and XGBoost on real training rows. Report ROC-AUC, F1, precision, recall, and balanced accuracy. Keep configurations fixed after validation selection | `results/tables/validation_metrics.csv` and the saved hyperparameters | Met. ROC-AUC is the pre-specified primary metric; the other four are reported |
| 4. Generators | Gaussian copula and CTGAN, fit only on real training rows, including the target, with one synthetic file per generator at the training size | `data/synthetic/`, `configs/generators.yaml` | Met. CTGAN uses the documented 80 CPU epochs, not the library default of 300 |
| 5. Main comparison | Shares 0, 10, 25, 50, 75, and 100 percent at constant training size. 100 percent is synthetic-only. Final comparison on the real test set | 165 main validation runs and 165 main test runs | Met. Validation was used to select the scarcity share before the test set was scored |
| 6. Repetition | Five shared seeds; mean, standard deviation, and interval comparisons; separate consistent changes from small ones | Seeds 42, 123, 2024, 3407, and 777. Paired Student-t intervals with four degrees of freedom | Met. The five seeds are one split, not five datasets |
| 7. Synthetic quality | Numerical and categorical distributions, dependence, target balance, invalid or duplicate rows, and the link to utility | `results/tables/study/quality_summary.csv` and Figures 5–9 | Met. Duplicate checks are not a privacy audit |
| 8. Limited real data | After the main experiment, use 25, 50, and 75 percent of the real training rows. Compare real-only training with the augmentation setting selected from the main experiment. Keep validation and test membership fixed | Both generators selected a 10 percent share. Scarcity tables cover augmentation and replacement | Met. The proposal’s 10 and 100 percent scarcity budgets were not used, because the later instructions specify 25, 50, and 75 percent |
| 9. Reproducibility | Python pipeline, versions, seeds, configurations, all four research questions, and limitations on one dataset | `REPRODUCIBILITY.md`, `uv.lock`, final report Sections 14–16 | Met |

## Literature

All ten proposal studies were identified and their bibliographic records were checked. Placeholder DOI text in the proposal was removed. Decruyenaere et al. (2024) has a proceedings URL and no DOI on the PMLR record. No DOI was invented. The reconciled review is in the final report.
