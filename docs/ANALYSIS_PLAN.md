# Pre-specified validation analysis plan

This plan was written before the synthetic-data utility experiments were interpreted. It is the analysis that the validation report follows. It does not use the held-out test set.

## Primary question

Can synthetic tabular data improve income classification on the UCI Adult Census Income data, relative to real training data alone, when the same validation partition is used for every comparison?

## Estimands

- Primary metric: validation ROC-AUC at threshold-free ranking. The positive class is `>50K`.
- Secondary metrics at probability threshold 0.5: F1, precision, recall, and balanced accuracy.
- The training-set size in the main experiment is the frozen training count, N = 34,189.
- Synthetic and real counts use Python `round` (half to even). The real count is N minus the synthetic count. The two counts sum to N.
- The 0% condition is one shared real-only baseline. The 100% condition is synthetic-only training and is not called augmentation.

## Controls

- The split is the existing `data/processed/split_indices.csv` from seed 42. It is not redrawn.
- Classifier hyperparameters are the Phase 1 selected configurations. The only change across repetitions is `random_state`, which is the repetition seed.
- Preprocessing is the Phase 1 pipeline and is fit on the training rows of that run only.
- Generators are fit on real training rows only. Scarcity generators are fit on the real budget subset only.
- CTGAN epochs are the value in `configs/generators.yaml`. They are not changed after looking at validation AUC.

## Seeds and pairing

Seeds are 42, 123, 2024, 3407, and 777. Comparisons use the same seed for the real-only run and the synthetic run. Variability is the sample standard deviation across these seeds. If that standard deviation is zero, the report gives the point and does not add noise.

## Uncertainty

For a paired difference, report the mean, the sample standard deviation, and a 95% Student-t interval with n − 1 degrees of freedom. A paired t statistic may be reported as a description. With five seeds it is not treated as a confirmatory significance test.

An interval that lies entirely above zero is described as higher validation ROC-AUC on these five seeds. An interval that lies entirely below zero is described as lower. An interval that covers zero is described as inconclusive on these five seeds. The same rule is applied to the secondary metrics when they are discussed.

## Scarcity

Budgets are 25%, 50%, and 75% of the frozen training partition. Validation and test membership do not change.

After the main validation table exists, and not before, each generator’s augmentation share is the share in {0.10, 0.25, 0.50, 0.75, 1.00} with the highest mean validation ROC-AUC across the three classifiers and five seeds. Ties take the smaller share. If the winner is 1.00, augmentation uses the best share strictly below 1.00.

Augmentation keeps every real row in the budget and adds synthetic rows so the synthetic fraction of the enlarged set equals that share. The comparison baseline is a real-only model trained on the same budget. This is a change in training-set size.

Replacement is a separate table. Training size stays at the budget size. Synthetic rows replace real rows. The two analyses are not pooled into one estimate.

A shared share, excluding 1.00, is also selected by the same mean across both generators. Both generators are run at that share so their augmented training sets have the same size.

## Quality

Quality compares each synthetic training file with the real training partition. The test partition is not used. Metrics are Kolmogorov-Smirnov and Wasserstein distances, selected quantiles, total variation for categories, the Frobenius norm of the Spearman-correlation difference, Cramér’s V with the income target, exact duplicate counts, and a near-duplicate rate. The near-duplicate distance is Euclidean distance after training-fit ordinal encoding and standard scaling. The threshold is the first percentile of training-to-training nearest-neighbor distances on a documented subsample. These checks are not a privacy guarantee.

Fidelity-versus-utility plots are descriptive. The runs share a dataset, so a correlation is not an independent confirmation.

## What this plan will not do

- It will not score the test set.
- It will not retune models, epochs, or ratios from test metrics.
- It will not drop duplicate Adult rows from the frozen split.
- It will not claim that five seeds identify a small effect.
