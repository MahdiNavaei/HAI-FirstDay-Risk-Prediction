# Feature definitions

Feature construction is implemented in `src/modeling/prompt3_runtime.py` and
the deterministic source cleaning in `src/modeling/run_prompt3.py`.

## Feature Set A

`Age`, `Sex`, and `Triage level`.

## Feature Set B

Feature Set A plus the first-day numeric and categorical clinical fields:

`SPo2`, `BPMin`, `BPMax`, `PR`, `RR`, `T`, `BS`, `BS.1`, `WBC`, `HB`, `HCT`,
`PLT`, `ESR`, `CRP`, `UREA`, `CR`, `NA`, and `K`.

## Feature Set C (frozen primary)

Feature Set B plus one missingness indicator for every Feature Set B field.
The indicator is `1` when the source value is blank or missing and `0`
otherwise. Indicators are generated within the sklearn pipeline before
training-fold preprocessing.

## Excluded fields

The primary representation excludes `diagnosis`, `Department`, `RBC`, `PT`,
`Row`, and `Year`. These fields are not predictors in the frozen primary
model. Feature Set D adds a deterministic complaint grouping only for
sensitivity analysis and is not the primary representation.

## Preprocessing

Numeric values use median imputation fitted on the training fold. Categorical
values use an explicit missing category and one-hot encoding fitted on the
training fold. XGBoost uses no scaling, resampling, or class weighting in the
frozen primary pipeline.
