# Study governance

This record carries the frozen analysis boundary into the public code release.
It is a release-level governance summary, not a substitute for source-data
adjudication or clinical validation.

## Frozen analysis specification

- Outcome: released binary `Label` used as provided; operational definition
  remains unverified.
- Development pool: `Year` in `1402`, `1403`, or `1404`.
- Held-out period: `1404-2`, evaluated only after model, preprocessing,
  calibration, and alert-budget decisions were frozen.
- Primary model: XGBoost / Feature Set C.
- Calibration: no recalibration; raw frozen probabilities are evaluated.
- Primary scalar metric: Average Precision (AP), implemented with
  `sklearn.metrics.average_precision_score`.
- Alert budgets: 1%, 2%, 5%, and 10%; top 5% is primary.

## Safeguards

Preprocessing is fitted inside training folds. The held-out period is not used
for model fitting, tuning, feature selection, calibration, threshold selection,
or model rescue. Natural outcome prevalence is retained. Subgroup,
missingness/shift, explanation-stability, and negative-control analyses are
descriptive robustness analyses and do not authorize changes to the frozen
model.

## Interpretation boundary

The release supports risk ranking, aggregate calibration, enrichment, and
alert-workload reporting. It does not support claims of clinical utility,
causal risk factors, fairness certification, patient-independent validation,
true temporal validation, external validation, prospective effectiveness, or
deployment readiness.

## Known limitations retained at release

- The operational `Label` definition and direct `Label`-to-INIS/NISS mapping
  are unverified.
- Exact field-level timestamps and patient independence are unavailable.
- `1404-2` chronology is too coarse for a true temporal-validation claim.
- The study is a single-release retrospective analysis without external or
  prospective validation.
