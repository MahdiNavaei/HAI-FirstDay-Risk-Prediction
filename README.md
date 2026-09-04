# Early Prediction of Hospital-Acquired Infection from First-Day EHR Data

This repository contains the reproducible analysis code, frozen
configurations, and aggregate reproducibility materials accompanying the
manuscript:

_Early Prediction of Hospital-Acquired Infection from First-Day EHR Data Under
Extreme Class Imbalance: A Frozen Held-Out Evaluation of Average Precision,
Calibration, and Alert Burden_

Manuscript prepared for submission. The repository does not contain the source
dataset or encounter-level prediction outputs.

## Overview

The study is a retrospective secondary analysis of a rare binary outcome in
119,743 encounters. The released `Label` is used as provided by the source
dataset. The analysis reconstructs the source-described admission/first-24-hour
window, builds leakage-controlled first-day representations, develops models on
the development pool, and evaluates the frozen primary specification on the
pre-specified `1404-2` held-out period.

The primary model is XGBoost with Feature Set C: authorized first-day clinical
variables plus explicit missingness indicators. The primary scalar metric is
Average Precision (AP), implemented with
`sklearn.metrics.average_precision_score`. Calibration and fixed alert budgets
are reported alongside discrimination.

## Study design

- Retrospective, encounter-level secondary analysis.
- Full cohort: 119,743 encounters and 1,567 positive labels.
- Source-described admission and first-24-hour data window; exact field-level
  timestamps are unavailable.
- Development pool: 95,997 encounters and 1,228 positives.
- Frozen held-out period `1404-2`: 23,746 encounters and 339 positives.
- Final model: XGBoost / Feature Set C, with no post-held-out model tuning.
- Primary alert budget: top 5%; secondary budgets: 1%, 2%, and 10%.
- Evaluation uses natural outcome prevalence and fixed rank-based alert budgets.

## Key held-out results

The frozen aggregate reference results are recorded in
[`results/expected_results.json`](results/expected_results.json):

- Average Precision (AP): **0.0525**.
- Held-out prevalence: **0.0143**.
- Prevalence-relative AP: **3.68-fold**.
- AUROC: **0.7895**.
- Brier score: **0.0138**.
- At the top 5% alert budget: **21.24%** of HAI events captured and **6.06%**
  PPV, corresponding to 72 of 339 events among 1,188 alerts.

These results do not establish clinical utility, external validation,
patient-independent validation, prospective effectiveness, causal risk factors,
or deployment readiness.

## Dataset

The source dataset is not included. Obtain it directly from the original
source:

- Hospital-Acquired Infection (HAI) Prediction Dataset, HAI-release 1.0.1.
- [Zenodo DOI 10.5281/zenodo.21831931](https://doi.org/10.5281/zenodo.21831931).
- Source-reported license: CC BY-NC 4.0.

The locally analyzed raw-file SHA-256 was
`e60a63201eceb772329247b561a8dd996d40e33750abfc333e845d6598b08060`. The
analysis established a strong association with HAI-release 1.0.1, but exact
byte-level identity with the Zenodo archive was not independently verified.
Users must obtain the data under the original source terms.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduction

The raw CSV path is always supplied by the user; it is never hard-coded into
the public release. The following commands use a placeholder path:

```bash
python scripts/prepare_data.py --data /path/to/Hospital_infection_data.csv
python scripts/reproduce_development.py --data /path/to/Hospital_infection_data.csv
python scripts/reproduce_heldout.py --data /path/to/Hospital_infection_data.csv
python scripts/reproduce_tables.py --results results/local/heldout_results.json
python scripts/reproduce_figures.py --results results/local/heldout_results.json
```

Development and held-out execution can require substantial CPU and memory.
Generated models, tables, figures, logs, intermediate files, and aggregate
local outputs are ignored by Git. The held-out script keeps predictions in
memory and writes aggregate metrics only. It does not tune or recalibrate after
observing held-out outcomes, and it never overwrites the frozen reference in
`results/expected_results.json`.

`reproduce_development.py` runs the existing development-only comparison and
robustness preparation workflow. The primary held-out command is deliberately
separate so the held-out period remains a governed, one-time-style evaluation
boundary in the released workflow.

## Repository structure

```text
src/modeling/       Actual cleaning, feature, modeling, and robustness code
configs/            Frozen cohort, preprocessing, validation, model, and policy files
scripts/            User-facing reproducibility entry points
docs/               Data provenance, governance, feature, and run instructions
metadata/           Non-secret frozen model metadata
results/            Aggregate expected reference results only
tests/              Lightweight public-release smoke tests
```

## Methodological safeguards

- Fold-safe preprocessing and missingness augmentation.
- Frozen XGBoost / Feature Set C model specification.
- Held-out period untouched until the final pre-specified evaluation.
- No post-test tuning, model selection, recalibration, or threshold rescue.
- Natural-prevalence evaluation.
- Aggregate calibration and uncertainty reporting.
- Fixed 1%, 2%, 5%, and 10% alert-budget evaluation.
- Subgroup, missingness/shift, and explanation-stability analyses.
- Negative controls in the development robustness workflow.

## Limitations

- The operational `Label` definition is not directly documented in the public
  release.
- Direct `Label`-to-INIS/NISS mapping is unverified.
- Patient independence is unavailable; the analysis is encounter-level.
- Exact field-level timestamps are unavailable.
- `1404-2` chronology is insufficient for a true temporal-validation claim.
- The study has no external or prospective validation.
- Subgroup generalization is worse, missingness shift is moderate, and fixed
  alert budgets quantify workload rather than clinical benefit.

## Citation

See [`CITATION.cff`](CITATION.cff) for the software citation metadata. Do not
invent or infer a manuscript DOI from this repository.

## License

### Code license

Original repository code is released under the MIT License. See
[`LICENSE`](LICENSE).

### Dataset license

The source HAI dataset is not part of this repository and remains subject to
its original CC BY-NC 4.0 license.
