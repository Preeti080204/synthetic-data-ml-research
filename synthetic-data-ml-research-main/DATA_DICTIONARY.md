# Data dictionary

The Adult Census Income extract used in this project follows the UCI attribute names. The target is binary. A question mark in a source file is missing data. After the Phase 1 preprocessing contract, a missing categorical value is the explicit level `Missing`.

| Column | Role | Type | Description |
| --- | --- | --- | --- |
| age | predictor | numerical | Age in years. |
| workclass | predictor | categorical | Employment class. Missing values occur in the source file. |
| fnlwgt | predictor | numerical | Census final weight. It is a survey design weight, not a personal attribute, and it is not used as `sample_weight`. |
| education | predictor | categorical | Highest educational attainment, as a label. |
| education-num | predictor | numerical | Ordered code for the same attainment as `education`. |
| marital-status | predictor | categorical | Marital status. |
| occupation | predictor | categorical | Occupational category. Missing values occur in the source file. |
| relationship | predictor | categorical | Relationship role within the household. |
| race | predictor | categorical | Race category in the 1994 census extract. |
| sex | predictor | categorical | Sex recorded in the 1994 census extract. |
| capital-gain | predictor | numerical | Capital gains. The mass at zero is expected. |
| capital-loss | predictor | numerical | Capital losses. The mass at zero is expected. |
| hours-per-week | predictor | numerical | Usual hours worked per week. |
| native-country | predictor | categorical | Country of origin. Missing values occur in the source file. |
| income | target | categorical | `>50K` is the positive class and is coded as 1. `<=50K` is the negative class. |
| source_file | metadata | text | `adult.data` or `adult.test`. Not a model input. |
| source_line | metadata | integer | Line in the source file. Not a model input. |
| source_row_id | metadata | text | `file:line` identifier. Not a model input. |

## Partitions

`data/processed/split_indices.csv` is the only primary split. It has 34,189 training rows, 7,326 validation rows, and 7,327 test rows. The study does not draw a new primary split. Synthetic files contain the fourteen predictors and `income` only.

## Synthetic files

A completed synthetic file has one row per requested sample and an income label in `{<=50K, >50K}`. Invalid labels are logged in `generation.json` and are not silently deleted. Generator metadata, the fitted synthesizer, and any exported loss table sit beside the CSV.

## Held-out evaluation

`results/final_evaluation/test_run_metrics.csv` has one row per completed test run. `partition` is `test`. `synthetic_share`, `n_real`, `n_synthetic`, `seed`, `budget`, and `classifier` identify the frozen condition. `results/final_evaluation/paired_differences.csv` stores the matched-seed differences. These files are not validation metrics.

## What is not a column

The split seed, the repetition seed, the synthetic share, and the real-data budget are experiment fields. They are stored in the run records, not in the Adult table.
