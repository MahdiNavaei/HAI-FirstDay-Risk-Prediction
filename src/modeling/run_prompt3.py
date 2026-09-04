"""Reproducible Prompt 3 development analysis.

This runner deliberately excludes the unresolved locked candidate period
``1404-2`` from every model fit, tuning operation, calibration operation,
prediction artifact, and result table.  It never changes the raw CSV.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import yaml
from scipy.stats import rankdata, spearmanr, t
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import ParameterSampler, StratifiedKFold

from prompt3_runtime import (
    BASE_FEATURES_A,
    BASE_FEATURES_B,
    CATEGORICAL_FEATURES_B,
    FEATURE_SETS,
    MISSINGNESS_COLUMNS_B,
    MODEL_FAMILIES,
    NUMERIC_FEATURES_B,
    aggregate_importance,
    binary_metrics,
    feature_columns,
    get_transformed_feature_names,
    make_pipeline,
    make_pipeline_for_columns,
    resample_training,
    safe_json_value,
    top_fraction_metrics,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "Hospital_infection_data.csv"
REPORT = ROOT / "reports" / "prompt3"
TABLE = ROOT / "tables" / "prompt3"
FIGURE = ROOT / "figures" / "prompt3"
MODEL = ROOT / "models" / "prompt3"
ARTIFACT = ROOT / "artifacts" / "prompt3"
LOG_DIR = ROOT / "logs" / "prompt3"
DERIVED = ROOT / "data" / "derived"

SEEDS = [42, 2024]
N_SPLITS = 3
LOCKED_YEAR = "1404-2"
EXPECTED_FULL_N = 119743
EXPECTED_FULL_POSITIVE_N = 1567
EXPECTED_FULL_PREVALENCE = EXPECTED_FULL_POSITIVE_N / EXPECTED_FULL_N
EXPECTED_RBC_DATE_LIKE_N = 8134

# These rates are the frozen Prompt 2 audit values used only to define the
# required ablation subsets.  They are not estimated from model outcomes.
DOCUMENTED_MISSING_PCT = {
    "Age": 0.0058458532,
    "Sex": 0.0,
    "Triage level": 0.0,
    "SPo2": 0.6271765364,
    "BPMin": 2.2673559206,
    "BPMax": 2.2673559206,
    "PR": 0.7499394537,
    "RR": 87.4547990279,
    "T": 1.5107354918,
    "BS": 74.0193581253,
    "BS.1": 4.7034064623,
    "WBC": 2.4786417578,
    "HB": 0.5854204421,
    "HCT": 0.5879258078,
    "PLT": 0.6589111681,
    "ESR": 81.7300384991,
    "CRP": 53.8160894583,
    "UREA": 1.2618691698,
    "CR": 1.1691706405,
    "NA": 16.1170172787,
    "K": 19.2128141102,
}
LOW_MISSING_FIELDS = [c for c in BASE_FEATURES_B if DOCUMENTED_MISSING_PCT[c] < 10.0]
HIGH_MISSING_EXCLUDED = [c for c in BASE_FEATURES_B if DOCUMENTED_MISSING_PCT[c] > 50.0]
HIGH_MISSING_INCLUDED = [c for c in BASE_FEATURES_B if c not in HIGH_MISSING_EXCLUDED]

TUNING_SPACES: dict[str, dict[str, list[Any]]] = {
    "Logistic Regression": {"C": [0.1, 1.0, 3.0]},
    "Random Forest": {
        "n_estimators": [140, 200],
        "max_depth": [8, 14, None],
        "min_samples_leaf": [1, 3, 8],
        "max_features": ["sqrt", 0.5],
    },
    "XGBoost": {
        "n_estimators": [160, 220],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.06],
        "min_child_weight": [1, 3, 8],
    },
    "LightGBM": {
        "n_estimators": [160, 220],
        "num_leaves": [15, 31],
        "max_depth": [-1, 7],
        "learning_rate": [0.03, 0.06],
        "min_child_samples": [20, 50],
    },
    "CatBoost": {
        "iterations": [180, 260],
        "depth": [5, 6, 7],
        "learning_rate": [0.03, 0.06],
        "l2_leaf_reg": [3.0, 8.0],
    },
}


def ensure_dirs() -> None:
    for path in [REPORT, TABLE, FIGURE, MODEL, ARTIFACT, LOG_DIR, DERIVED]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_json_value(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("prompt3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_DIR / "prompt3_run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    names = ["pandas", "numpy", "scipy", "sklearn", "xgboost", "lightgbm", "catboost", "matplotlib", "yaml", "joblib", "pyarrow", "psutil", "shap"]
    out: dict[str, str] = {}
    for name in names:
        try:
            module = importlib.import_module(name)
            out[name] = str(getattr(module, "__version__", "available"))
        except Exception as exc:
            out[name] = f"UNAVAILABLE: {type(exc).__name__}"
    return out


def git_commit_if_applicable() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value)).casefold().strip()


COMPLAINT_RULES: list[tuple[str, list[str], bool]] = [
    ("infection_related", ["infection", "sepsis", "septic", "pneumonia", "pneumon", "meningitis", "cellulitis", "abscess", "uti", "urinary infection", "pyelonephritis", "covid", "corona", "عفونت", "سپسیس", "پنومونی", "مننژیت"], True),
    ("respiratory", ["cough", "dyspnea", "shortness of breath", "respiratory", "asthma", "copd", "wheeze", "breath", "سرفه", "تنگی نفس"], False),
    ("urinary", ["dysuria", "hematuria", "urinary", "flank pain", "kidney", "ادرار", "هماچوری", "کلیه"], False),
    ("gastrointestinal", ["abdominal", "stomach", "vomit", "nausea", "diarrhea", "constipation", "gastro", "abdomen", "شکم", "استفراغ", "تهوع", "اسهال"], False),
    ("neurologic", ["headache", "seizure", "stroke", "weakness", "dizziness", "consciousness", "سردرد", "تشنج", "سرگیجه"], False),
    ("cardiovascular", ["chest pain", "cardiac", "palpitation", "heart", "hypertension", "hypotension", "قلب", "درد قفسه سینه"], False),
    ("trauma", ["trauma", "accident", "fracture", "fall", "burn", "injury", "تصادف", "شکستگی", "سوختگی"], False),
    ("metabolic", ["diabetes", "hyperglycemia", "hypoglycemia", "thyroid", "metabolic", "دیابت", "قند"], False),
    ("dermatologic", ["rash", "wound", "itch", "skin", "زخم", "خارش", "پوست"], False),
    ("musculoskeletal", ["back pain", "joint", "muscle", "limb", "bone", "کمر", "مفصل", "عضله"], False),
    ("obstetric", ["pregnan", "labor", "delivery", "pregnancy", "بارداری", "زایمان"], False),
    ("pain", ["pain", "درد"], False),
]


def classify_complaint(value: Any) -> tuple[str, bool, str]:
    text = normalize_text(value)
    if not text:
        return "missing", False, "blank complaint"
    for group, terms, infection_related in COMPLAINT_RULES:
        matched = next((term for term in terms if term in text), None)
        if matched:
            return group, infection_related, f"deterministic substring rule: {matched}"
    return "other", False, "no deterministic clinical lexicon match"


def blank_to_nan(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.mask(values.eq(""), pd.NA)


def reconstruct_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    clean = pd.DataFrame(index=raw.index)
    qc = pd.DataFrame(index=raw.index)
    parse_summary: dict[str, dict[str, int]] = {}

    clean["source_row_number"] = np.arange(2, len(raw) + 2, dtype=np.int64)
    clean["Row"] = pd.to_numeric(raw["Row"], errors="coerce")
    clean["Year"] = raw["Year"].astype("string").str.strip()
    clean["Label"] = pd.to_numeric(raw["Label"], errors="coerce")
    label_invalid = clean["Label"].isna() | ~clean["Label"].isin([0, 1])
    qc["Label__invalid"] = label_invalid

    ranges: dict[str, tuple[float, float]] = {
        "Age": (0, 120),
        "Triage level": (1, 5),
        "SPo2": (0, 100),
        "BPMin": (0, 300),
        "BPMax": (0, 300),
        "T": (30, 45),
        "PR": (0, 300),
        "RR": (0, 100),
        "BS": (0, 2000),
        "BS.1": (0, 2000),
        "WBC": (0, 500),
        "RBC": (None, None),
        "HB": (0, 50),
        "HCT": (0, 100),
        "PLT": (0, 100000),
        "ESR": (0, 300),
        "UREA": (0, 1000),
        "CR": (0, 100),
        "NA": (0, 300),
        "K": (0, 100),
        "PT": (0, 200),
    }
    for field, (lower, upper) in ranges.items():
        raw_values = raw[field].astype("string").str.strip()
        nonempty = raw_values.ne("")
        numeric = pd.to_numeric(raw_values, errors="coerce")
        parse_failure = nonempty & numeric.isna()
        out_of_range = pd.Series(False, index=raw.index)
        if lower is not None and upper is not None:
            out_of_range = numeric.notna() & ((numeric < lower) | (numeric > upper))
        clean[field] = numeric
        # Prompt 2 explicitly locks malformed/out-of-range temperature and
        # invalid Age/Triage values out of their derived modeling values.
        if field in {"Age", "Triage level", "T"}:
            clean[field] = clean[field].mask(out_of_range)
        qc[f"{field}__missing"] = raw_values.eq("")
        qc[f"{field}__parse_failure"] = parse_failure
        qc[f"{field}__out_of_range"] = out_of_range
        parse_summary[field] = {
            "blank_n": int(raw_values.eq("").sum()),
            "parse_failure_n": int(parse_failure.sum()),
            "out_of_range_n": int(out_of_range.sum()),
        }

    clean["Sex"] = blank_to_nan(raw["Sex"])
    clean["Department"] = blank_to_nan(raw["Department"])
    clean["Patient complaint"] = blank_to_nan(raw["Patient complaint"])
    clean["diagnosis"] = blank_to_nan(raw["diagnosis"])
    clean["CRP"] = blank_to_nan(raw["CRP"]).map(lambda value: normalize_text(value) if pd.notna(value) else pd.NA)
    crp_canonical = {
        "negative": "Negative",
        "trace": "Trace",
        "weakly positive": "Weakly Positive",
        "positive 1+": "Positive 1+",
        "positive 2+": "Positive 2+",
        "positive 3+": "Positive 3+",
        "positive 4+": "Positive 4+",
    }
    clean["CRP"] = clean["CRP"].map(lambda value: crp_canonical.get(value, value) if pd.notna(value) else pd.NA)
    complaint_info = clean["Patient complaint"].map(classify_complaint)
    clean["complaint_group"] = complaint_info.map(lambda item: item[0])
    clean["complaint_infection_related"] = complaint_info.map(lambda item: bool(item[1]))
    clean["complaint_mapping_rule"] = complaint_info.map(lambda item: item[2])
    clean["missing_count_B"] = clean[BASE_FEATURES_B].isna().sum(axis=1)

    rbc_numeric = clean["RBC"]
    # Match the Prompt 2 forensic rule: Excel-style serial screening used
    # 30,000-60,000; the 36,526 value decodes to 2000-01-01 and is included.
    qc["RBC__date_serial_like"] = rbc_numeric.between(30000, 60000, inclusive="both")
    stats = {
        "full_n": int(len(clean)),
        "full_positive_n": int((clean["Label"] == 1).sum()),
        "full_negative_n": int((clean["Label"] == 0).sum()),
        "full_prevalence": float((clean["Label"] == 1).mean()),
        "invalid_label_n": int(label_invalid.sum()),
        "missing_predictor_row_exclusions": 0,
        "rbc_nonnull_n": int(clean["RBC"].notna().sum()),
        "rbc_date_serial_like_n": int(qc["RBC__date_serial_like"].sum()),
        "parse_summary": parse_summary,
        "qc_flag_counts": {col: int(value) for col, value in qc.sum(numeric_only=True).items()},
    }
    return clean, qc, raw.copy(), stats, complaint_info.to_frame("complaint_info")


def write_analytic_artifact(raw: pd.DataFrame, clean: pd.DataFrame, qc: pd.DataFrame, raw_hash: str) -> None:
    # The locked candidate is used only for the required full-cohort count
    # check; it is deliberately absent from all derived analysis artifacts.
    dev_mask = ~clean["Year"].eq(LOCKED_YEAR)
    raw_dev = raw.loc[dev_mask].copy()
    clean_dev = clean.loc[dev_mask].copy()
    qc_dev = qc.loc[dev_mask].copy()
    artifact = raw_dev.copy()
    artifact.insert(0, "source_row_number", raw_dev.index.to_numpy(dtype=np.int64) + 2)
    for col in clean_dev.columns:
        if col in raw_dev.columns:
            artifact[f"derived__{col}"] = clean_dev[col]
        else:
            artifact[f"derived__{col}"] = clean_dev[col]
    for col in qc_dev.columns:
        artifact[f"qc__{col}"] = qc_dev[col].fillna(False).astype(bool)
    artifact["source_sha256"] = raw_hash
    artifact.to_parquet(DERIVED / "prompt3_analytic_dataset.parquet", index=False)


def write_reconstruction_report(stats: dict[str, Any], clean: pd.DataFrame, raw_hash: str) -> None:
    year_counts = clean["Year"].value_counts(dropna=False).to_dict()
    label_counts = clean["Label"].value_counts(dropna=False).to_dict()
    dev_mask = clean["Year"].isin(["1402", "1403", "1404"])
    candidate_mask = clean["Year"].eq(LOCKED_YEAR)
    text = f"""# Prompt 3 analytic dataset reconstruction

## Status

`PASS`: deterministic reconstruction matched the binding Prompt 2B expectations. No raw file was overwritten, and no model or test-set performance was evaluated during reconstruction.

## Source and counts

- Source: `{RAW.relative_to(ROOT).as_posix()}`
- SHA-256: `{raw_hash}`
- Parsed shape: `{stats['full_n']:,}` rows x `29` columns
- Positive labels: `{stats['full_positive_n']:,}`
- Negative labels: `{stats['full_negative_n']:,}`
- Observed prevalence: `{stats['full_prevalence'] * 100:.6f}%`
- Invalid/missing labels: `{stats['invalid_label_n']}`
- Missing-predictor row exclusions: `0`
- RBC non-null: `{stats['rbc_nonnull_n']:,}`; date-serial-like QC flags: `{stats['rbc_date_serial_like_n']:,}`

Expected full-cohort counts were 119,743 rows, 1,567 positives, and approximately 1.309% prevalence. All matched.

## Period handling

Observed Year counts are recorded for audit only: `{json.dumps({str(k): int(v) for k, v in year_counts.items()}, ensure_ascii=False)}`.

The Prompt 3 development pool contains `{int(dev_mask.sum()):,}` rows and `{int((clean.loc[dev_mask, 'Label'] == 1).sum()):,}` positives from `1402`, `1403`, and `1404`. The unresolved candidate period `1404-2` contains `{int(candidate_mask.sum()):,}` rows and `{int((clean.loc[candidate_mask, 'Label'] == 1).sum()):,}` positives, but is excluded from every model fit, tuning/calibration operation, prediction artifact, and result table.

## Deterministic cleaning and QC

The raw source remains unchanged. The derived parquet preserves raw columns and source-row lineage for the non-candidate development pool only; the locked candidate is absent. Blank cells become missing values. Numeric parsing uses `errors=coerce`; no learned imputation occurs here. Age, Triage level, and temperature range violations are flagged and excluded from their derived values under the locked QC rule. Temperature `36,5` is not silently repaired; `370` and `384` remain flagged. `BS` and `BS.1` remain separate. RBC is retained for audit only and is not date-decoded. PT is retained for audit only, including malformed `13..4`.

Parsing/QC counts are machine-readable in the derived dataset and summarized below:

```json
{json.dumps(stats['parse_summary'], indent=2, ensure_ascii=False)}
```

The analysis-ready modeling frames use only the authorized Feature Sets A-D and never include diagnosis, Department, RBC, PT, Row, or Year as predictors.
"""
    write_text(REPORT / "01_analytic_dataset_reconstruction.md", text)


def write_complaint_mapping(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    rows = pd.DataFrame({
        "original_complaint": raw["Patient complaint"].astype("string"),
        "normalized_complaint": raw["Patient complaint"].map(normalize_text),
        "complaint_group": clean["complaint_group"],
        "infection_related": clean["complaint_infection_related"],
        "mapping_rule": clean["complaint_mapping_rule"],
    })
    rows["n_rows"] = rows.groupby(["original_complaint", "normalized_complaint", "complaint_group", "infection_related", "mapping_rule"], dropna=False)["original_complaint"].transform("size")
    rows = rows.drop_duplicates().sort_values(["complaint_group", "normalized_complaint"], na_position="first")
    rows.to_csv(TABLE / "complaint_group_mapping.csv", index=False, encoding="utf-8-sig")
    group_counts = clean["complaint_group"].value_counts(dropna=False).to_dict()
    infection_n = int(clean["complaint_infection_related"].sum())
    text = f"""# Prompt 3 complaint grouping

Feature Set D is sensitivity-only. The grouping was frozen before any model performance was examined and was generated without `Label`, mutual information, outcome-guided term selection, embeddings, an LLM API, or an external paid API.

The mapping is a deterministic substring lexicon over Unicode-normalized, case-folded raw complaint text. The raw complaint, normalized text, group, infection-related flag, mapping rule, and row count are preserved in `tables/prompt3/complaint_group_mapping.csv`.

Infection-related groups are explicitly flagged. The full row-level group distribution is: `{json.dumps({str(k): int(v) for k, v in group_counts.items()}, ensure_ascii=False)}`. Infection-related rows: `{infection_n:,}`. Blank complaints are assigned to `missing`; unmatched text is assigned to `other`.

The grouping is a frozen sensitivity representation, not a validated clinical NLP label and not an outcome definition.
"""
    write_text(REPORT / "02_complaint_grouping.md", text)


def build_splits(n: int, y: pd.Series) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    positions = np.arange(n)
    for seed in SEEDS:
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(splitter.split(positions, y.to_numpy()), start=1):
            records.append({"seed": seed, "fold": fold, "split_id": f"seed{seed}_fold{fold}", "train_idx": train_idx, "val_idx": val_idx})
    return records


def write_validation_plan(clean: pd.DataFrame, splits: list[dict[str, Any]]) -> None:
    dev_mask = clean["Year"].isin(["1402", "1403", "1404"])
    n = int(dev_mask.sum())
    pos = int((clean.loc[dev_mask, "Label"] == 1).sum())
    split_lines = "\n".join(f"- Seed `{s['seed']}`, fold `{s['fold']}`: {len(s['train_idx']):,} training rows / {len(s['val_idx']):,} validation rows" for s in splits)
    text = f"""# Prompt 3 validation execution plan

This plan was written before tuning and model fitting.

## Binding design

- Evaluation unit: **encounter-level retrospective**.
- Development pool: all target-eligible rows with `Year` in `1402`, `1403`, or `1404` (`{n:,}` rows; `{pos:,}` positives).
- Excluded locked candidate: `1404-2`; it is not loaded into any modeling frame and receives no predictions.
- Validation: development-only stratified three-fold resampling repeated over fixed seeds `42` and `2024`.
- Temporal status: `PARTIAL / COARSE PERIOD VALIDATION ONLY`; Year is metadata and never a predictor.
- Patient-independent validation: unsupported because no patient/admission linkage key is released.
- Preprocessing: each pipeline is fitted inside each training fold; validation rows are transformed only after that fit.
- Primary optimization metric: Average Precision / PR-AUC.
- Primary training strategy: no resampling; class-weight and raw-row random resampling strategies are sensitivity analyses only.

## Fold schedule

{split_lines}

No threshold is locked. Operating points are reported at the pre-specified 1%, 2%, 5%, and 10% alert budgets. Results are internal development evidence only and cannot support external, prospective, deployment, or patient-independent claims.
"""
    write_text(REPORT / "03_validation_execution_plan.md", text)


def write_pre_model_protocol_docs() -> None:
    rows = []
    for model, space in TUNING_SPACES.items():
        for parameter, values in space.items():
            rows.append({"model_family": model, "tuning_scope": "Feature Set B only; development-only two-fold CV", "parameter": parameter, "candidate_values": json.dumps(values, ensure_ascii=False)})
    pd.DataFrame(rows).to_csv(TABLE / "hyperparameter_search_spaces.csv", index=False, encoding="utf-8-sig")
    write_text(REPORT / "04_model_tuning_protocol.md", """# Prompt 3 model-tuning protocol

This protocol artifact was written before model fitting. Tuning is bounded and development-only: at most three deterministic `ParameterSampler` configurations per required family are scored by two-fold stratified cross-validation on Feature Set B. The locked candidate period is not loaded into tuning. Selected family parameters are then reused for the matching Feature Set A/B/C primary comparisons and the corresponding class-weight sensitivity.

Search spaces are recorded in `tables/prompt3/hyperparameter_search_spaces.csv`. No deep tabular model or architecture search is included. The primary optimization metric is Average Precision / PR-AUC. The final report appends the tested configurations and scores.
""")
    write_text(REPORT / "08_model_selection_policy.md", """# Prompt 3 model-selection policy

This policy was written before model performance was examined. Selection is not based only on the largest PR-AUC. Among the primary no-resampling candidates, compute a composite rank using: mean PR-AUC (40%), mean Brier score (25%), calibration-slope closeness to 1 (15%), fold PR-AUC stability (15%), and model/feature complexity (5%). Lower rank is better for Brier, slope deviation, variability, and complexity. The candidate with the strongest composite is selected; the highest-PR-AUC and lowest-Brier candidates are reported separately.

Feature Set D is sensitivity-only. No clinical threshold is locked. A slightly lower PR-AUC candidate may be preferred when calibration and stability are materially better. Selection is development-phase only and does not use `1404-2`.
""")


def input_frame(clean: pd.DataFrame, feature_set: str, positions: np.ndarray | None = None) -> pd.DataFrame:
    if feature_set not in {"A", "B", "C", "D"}:
        raise ValueError(feature_set)
    if positions is None:
        source = clean
    else:
        source = clean.iloc[positions]
    if feature_set == "A":
        cols = BASE_FEATURES_A
    elif feature_set == "B":
        cols = BASE_FEATURES_B
    elif feature_set == "C":
        cols = BASE_FEATURES_B
    else:
        cols = BASE_FEATURES_B + ["complaint_group"]
    return sklearn_safe_frame(source[cols].copy())


def sklearn_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable string missing values for sklearn/CatBoost only."""
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) or pd.api.types.is_object_dtype(out[col]):
            values = out[col].astype(object)
            out[col] = values.where(pd.notna(values), np.nan)
    return out


def ablation_frame(clean: pd.DataFrame, label: str) -> tuple[pd.DataFrame, list[str], bool]:
    if label == "M0":
        return sklearn_safe_frame(clean[BASE_FEATURES_B].copy()), BASE_FEATURES_B, False
    if label == "M1":
        return sklearn_safe_frame(clean[BASE_FEATURES_B].copy()), FEATURE_SETS["C"], True
    if label == "M2":
        frame = clean[BASE_FEATURES_B].isna().astype(float)
        frame.columns = MISSINGNESS_COLUMNS_B
        return frame, MISSINGNESS_COLUMNS_B, False
    if label == "M3":
        return sklearn_safe_frame(clean[LOW_MISSING_FIELDS].copy()), LOW_MISSING_FIELDS, False
    if label == "M4":
        return sklearn_safe_frame(clean[HIGH_MISSING_INCLUDED].copy()), HIGH_MISSING_INCLUDED, False
    raise ValueError(label)


def pipeline_for_label(model: str, label: str, params: dict[str, Any], y: pd.Series | None = None):
    if label in {"A", "B", "C", "D"}:
        return make_pipeline(model, label, params=params, strategy="none", y=y)
    _, cols, add_missing = ablation_frame(_ACTIVE_CLEAN, label)
    return make_pipeline_for_columns(model, cols, params=params, strategy="none", y=y, add_missingness=add_missing)


_ACTIVE_CLEAN: pd.DataFrame


def tune_model(model: str, X: pd.DataFrame, y: pd.Series, logger: logging.Logger) -> tuple[dict[str, Any], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    candidates = list(ParameterSampler(TUNING_SPACES[model], n_iter=3, random_state=42))
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for config_id, params in enumerate(candidates, start=1):
        scores: list[float] = []
        for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
            pipe = make_pipeline(model, "B", params=params, strategy="none", y=y.iloc[train_idx])
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = pipe.predict_proba(X.iloc[val_idx])[:, 1]
            scores.append(float(average_precision_score(y.iloc[val_idx], pred)))
        row = {"model_family": model, "config_id": config_id, "parameters": json.dumps(safe_json_value(params), sort_keys=True), "mean_cv_pr_auc": float(np.mean(scores)), "fold_pr_auc": json.dumps(scores)}
        records.append(row)
        logger.info("tuning %s config %d mean AP=%.6f", model, config_id, np.mean(scores))
    result = pd.DataFrame(records).sort_values("mean_cv_pr_auc", ascending=False).reset_index(drop=True)
    best = json.loads(result.iloc[0]["parameters"])
    return best, result


def ci95(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    half = float(t.ppf(0.975, len(arr) - 1) * np.std(arr, ddof=1) / np.sqrt(len(arr)))
    return float(np.mean(arr) - half), float(np.mean(arr) + half)


def summarize_fold_metrics(fold_metrics: pd.DataFrame, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    metric_cols = [c for c in fold_metrics.columns if c not in {"seed", "fold", "split_id"}]
    for col in metric_cols:
        vals = pd.to_numeric(fold_metrics[col], errors="coerce")
        lo, hi = ci95(vals)
        out[f"{prefix}mean_{col}"] = float(vals.mean())
        out[f"{prefix}ci_low_{col}"] = lo
        out[f"{prefix}ci_high_{col}"] = hi
        out[f"{prefix}sd_{col}"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else float("nan")
    return out


def run_cv(
    model: str,
    label: str,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[dict[str, Any]],
    params: dict[str, Any],
    strategy: str,
    logger: logging.Logger,
    collect_importance: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for split in splits:
        train_idx, val_idx = split["train_idx"], split["val_idx"]
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_fit, y_fit = resample_training(X_train, y_train, strategy, seed=int(split["seed"] * 100 + split["fold"]))
        if label in {"A", "B", "C", "D"}:
            pipe = make_pipeline(model, label, params=params, strategy=strategy, y=y_fit)
        else:
            _, cols, add_missing = ablation_frame(_ACTIVE_CLEAN, label)
            pipe = make_pipeline_for_columns(model, cols, params=params, strategy=strategy, y=y_fit, add_missingness=add_missing)
        pipe.fit(X_fit, y_fit)
        pred = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        metric = binary_metrics(y.iloc[val_idx], pred, budget=0.05)
        metric.update({"seed": split["seed"], "fold": split["fold"], "split_id": split["split_id"]})
        metric_rows.append(metric)
        for row_idx, truth, score in zip(X.iloc[val_idx].index, y.iloc[val_idx], pred):
            pred_rows.append({"row_index": int(row_idx), "model_family": model, "feature_set": label, "strategy": strategy, "seed": split["seed"], "fold": split["fold"], "split_id": split["split_id"], "y_true": int(truth), "predicted_probability": float(score)})
        if collect_importance:
            imp = aggregate_importance(pipe)
            if not imp.empty:
                for rank, row in enumerate(imp.itertuples(index=False), start=1):
                    importance_rows.append({"model_family": model, "feature_set": label, "strategy": strategy, "seed": split["seed"], "fold": split["fold"], "split_id": split["split_id"], "feature": row.feature, "importance": float(row.importance), "rank": rank})
        logger.info("fit %s / %s / %s / seed=%s fold=%s AP=%.6f", model, label, strategy, split["seed"], split["fold"], metric["average_precision"])
    return pd.DataFrame(pred_rows), pd.DataFrame(metric_rows), pd.DataFrame(importance_rows)


def pooled_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    cols = ["row_index", "y_true"]
    out = predictions.groupby("row_index", as_index=False).agg(y_true=("y_true", "first"), predicted_probability=("predicted_probability", "mean"))
    return out


def pooled_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    p = pooled_predictions(predictions)
    if p.empty:
        return {}
    return binary_metrics(p["y_true"], p["predicted_probability"], budget=0.05)


def summarize_run(pred: pd.DataFrame, folds: pd.DataFrame, model: str, label: str, strategy: str) -> dict[str, Any]:
    out: dict[str, Any] = {"model_family": model, "feature_set": label, "strategy": strategy, "n_predictions": int(len(pooled_predictions(pred)))}
    out.update(summarize_fold_metrics(folds))
    pooled = pooled_metrics(pred)
    for key, value in pooled.items():
        out[f"pooled_{key}"] = value
    return out


def write_tuning_results(tuning_rows: list[pd.DataFrame]) -> None:
    combined = pd.concat(tuning_rows, ignore_index=True) if tuning_rows else pd.DataFrame()
    combined.to_csv(TABLE / "tuning_results.csv", index=False, encoding="utf-8-sig")
    text = "# Prompt 3 model-tuning execution\n\n"
    text += "Bounded two-fold development-only randomized searches were executed on Feature Set B. The locked candidate period was excluded.\n\n"
    if not combined.empty:
        for model, group in combined.groupby("model_family"):
            best = group.iloc[0]
            text += f"- **{model}:** selected config `{best['config_id']}`, mean two-fold AP `{best['mean_cv_pr_auc']:.6f}`, parameters `{best['parameters']}`.\n"
    write_text(REPORT / "04_model_tuning_protocol.md", text)


def rank_primary_candidates(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    frame = summary[summary["strategy"] == "none"].copy().reset_index(drop=True)
    frame["slope_deviation"] = (frame["mean_calibration_slope"] - 1.0).abs()
    frame["complexity"] = frame["model_family"].map({"Logistic Regression": 1, "Random Forest": 3, "XGBoost": 4, "LightGBM": 4, "CatBoost": 5}).fillna(9)
    frame["rank_ap"] = rankdata(-frame["mean_average_precision"], method="average")
    frame["rank_brier"] = rankdata(frame["mean_brier"], method="average")
    frame["rank_slope"] = rankdata(frame["slope_deviation"].fillna(np.inf), method="average")
    frame["rank_stability"] = rankdata(frame["sd_average_precision"].fillna(np.inf), method="average")
    frame["rank_complexity"] = rankdata(frame["complexity"], method="average")
    frame["selection_rank_score"] = 0.40 * frame["rank_ap"] + 0.25 * frame["rank_brier"] + 0.15 * frame["rank_slope"] + 0.15 * frame["rank_stability"] + 0.05 * frame["rank_complexity"]
    frame = frame.sort_values("selection_rank_score").reset_index(drop=True)
    return frame, frame.iloc[0], frame.iloc[1] if len(frame) > 1 else frame.iloc[0]


def run_calibrated_cv(model: str, label: str, X: pd.DataFrame, y: pd.Series, params: dict[str, Any], splits: list[dict[str, Any]], logger: logging.Logger) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in splits:
        if label in {"A", "B", "C", "D"}:
            base = make_pipeline(model, label, params=params, strategy="none", y=y.iloc[split["train_idx"]])
        else:
            _, cols, add_missing = ablation_frame(_ACTIVE_CLEAN, label)
            base = make_pipeline_for_columns(model, cols, params=params, strategy="none", y=y.iloc[split["train_idx"]], add_missingness=add_missing)
        calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3, n_jobs=1)
        calibrated.fit(X.iloc[split["train_idx"]], y.iloc[split["train_idx"]])
        pred = calibrated.predict_proba(X.iloc[split["val_idx"]])[:, 1]
        for row_idx, truth, score in zip(X.iloc[split["val_idx"]].index, y.iloc[split["val_idx"]], pred):
            rows.append({"row_index": int(row_idx), "model_family": model, "feature_set": label, "strategy": "sigmoid", "seed": split["seed"], "fold": split["fold"], "split_id": split["split_id"], "y_true": int(truth), "predicted_probability": float(score)})
        logger.info("calibration %s / %s / seed=%s fold=%s", model, label, split["seed"], split["fold"])
    return pd.DataFrame(rows)


def write_calibration_table(raw_pred: pd.DataFrame, calibrated_pred: pd.DataFrame) -> pd.DataFrame:
    records = []
    for method, pred in [("none", raw_pred), ("sigmoid", calibrated_pred)]:
        fold_rows = []
        for (seed, fold, split_id), group in pred.groupby(["seed", "fold", "split_id"]):
            metrics = binary_metrics(group["y_true"], group["predicted_probability"], budget=0.05)
            metrics.update({"method": method, "seed": seed, "fold": fold, "split_id": split_id})
            fold_rows.append(metrics)
        frame = pd.DataFrame(fold_rows)
        row = {"method": method}
        row.update(summarize_fold_metrics(frame))
        row.update({f"pooled_{k}": v for k, v in pooled_metrics(pred).items()})
        records.append(row)
    table = pd.DataFrame(records)
    table.to_csv(TABLE / "calibration_metrics.csv", index=False, encoding="utf-8-sig")
    return table


def alert_budget_table(pred: pd.DataFrame, budgets: list[float] = [0.01, 0.02, 0.05, 0.10]) -> pd.DataFrame:
    rows = []
    for (seed, fold, split_id), group in pred.groupby(["seed", "fold", "split_id"]):
        for budget in budgets:
            m = top_fraction_metrics(group["y_true"], group["predicted_probability"], budget)
            rows.append({"seed": seed, "fold": fold, "split_id": split_id, "budget_pct": budget * 100, **m})
    table = pd.DataFrame(rows)
    summaries = []
    for budget, group in table.groupby("budget_pct"):
        row = {"budget_pct": budget}
        for col in ["alerted_n", "hai_captured_n", "sensitivity", "specificity", "ppv", "npv", "f1", "f2", "false_positives", "false_alerts_per_true_hai", "enrichment_over_prevalence"]:
            lo, hi = ci95(group[col])
            row[f"mean_{col}"] = float(group[col].mean())
            row[f"ci_low_{col}"] = lo
            row[f"ci_high_{col}"] = hi
        summaries.append(row)
    result = pd.DataFrame(summaries)
    result.to_csv(TABLE / "alert_budget_results.csv", index=False, encoding="utf-8-sig")
    return result


def subgroup_metrics(clean: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    pooled = pooled_predictions(pred).set_index("row_index")
    frame = clean.loc[pooled.index].copy()
    frame["predicted_probability"] = pooled["predicted_probability"]
    frame["y_true"] = pooled["y_true"]
    global_threshold = float(np.quantile(frame["predicted_probability"], 0.95))
    frame["predicted_top5"] = (frame["predicted_probability"] >= global_threshold).astype(int)
    frame["age_band"] = pd.cut(frame["Age"], bins=[-np.inf, 17, 39, 64, np.inf], labels=["0-17", "18-39", "40-64", "65+"], right=True).astype("string").fillna("missing")
    frame["missingness_severity"] = pd.cut(frame["missing_count_B"], bins=[-1, 2, 5, np.inf], labels=["0-2", "3-5", "6+"], right=True).astype("string")
    axes = [("sex", "Sex"), ("age_band", "age_band"), ("triage_level", "Triage level"), ("period", "Year"), ("missingness_severity", "missingness_severity")]
    dept = frame["Department"].astype("string").fillna("missing").value_counts().head(10).index
    frame["department_top10"] = frame["Department"].astype("string").fillna("missing").where(frame["Department"].astype("string").fillna("missing").isin(dept), "other")
    axes.append(("department", "department_top10"))
    rows = []
    for axis_name, col in axes:
        for value, group in frame.groupby(col, dropna=False):
            yv, pv = group["y_true"], group["predicted_probability"]
            positives = int(yv.sum())
            row: dict[str, Any] = {"axis": axis_name, "group": str(value), "n": int(len(group)), "positive_n": positives, "prevalence": float(yv.mean()), "threshold_for_operating_metrics": global_threshold}
            if positives >= 20 and yv.nunique() == 2:
                row["pr_auc"] = float(average_precision_score(yv, pv))
                row["auroc"] = float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(yv, pv))
            else:
                row["pr_auc"] = np.nan
                row["auroc"] = np.nan
            row["brier"] = float(np.mean((yv - pv) ** 2))
            op = top_fraction_metrics(yv, pv, 0.05)
            row["sensitivity"] = op["sensitivity"]
            row["ppv"] = op["ppv"]
            row["reporting_status"] = "reportable exploratory" if positives >= 20 else "counts/descriptive only"
            rows.append(row)
    return pd.DataFrame(rows)


def feature_importance_report(importance: pd.DataFrame, selected_model: str, selected_set: str) -> None:
    imp = importance[(importance["model_family"] == selected_model) & (importance["feature_set"] == selected_set) & (importance["strategy"] == "none")].copy()
    agreements = []
    if not imp.empty:
        piv = imp.pivot_table(index="feature", columns="split_id", values="importance", aggfunc="mean", fill_value=0)
        cols = list(piv.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr = spearmanr(piv[cols[i]], piv[cols[j]]).statistic
                agreements.append(float(corr) if np.isfinite(corr) else np.nan)
    mean_imp = imp.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False).head(15) if not imp.empty else pd.DataFrame()
    text = f"""# Feature-importance stability

Leading development candidate: `{selected_model}` / Feature Set `{selected_set}`. Importance is fold-fitted model-native importance aggregated back to source fields; it is not causal evidence.

Pairwise Spearman rank agreement across the recorded fold fits has mean `{np.nanmean(agreements):.4f}` and range `{np.nanmin(agreements):.4f}` to `{np.nanmax(agreements):.4f}` when available. Rankings should be treated as unstable when agreement is low and should not be interpreted as mechanistic causality.

Top mean source-field importances:

{mean_imp.to_string(index=False) if not mean_imp.empty else 'No importance output was available.'}
"""
    write_text(REPORT / "13_feature_stability.md", text)


def write_explainability_report(importance: pd.DataFrame, selected_model: str, selected_set: str) -> None:
    imp = importance[(importance["model_family"] == selected_model) & (importance["feature_set"] == selected_set) & (importance["strategy"] == "none")].copy()
    top = imp.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False).head(10) if not imp.empty else pd.DataFrame()
    write_text(REPORT / "12_explainability_development.md", f"""# Explainability - development phase

Leading model: `{selected_model}` / Feature Set `{selected_set}`. The available summary uses fold-fitted model-native importance aggregated to source fields. It is suitable for a focused development diagnostic; it is not a causal explanation. A bounded TreeSHAP diagnostic for the saved development candidate is recorded separately in `shap_status.md` and the SHAP table/figures.

Top source-field importance summary:

{top.to_string(index=False) if not top.empty else 'No importance summary available.'}

For tree models, the separate SHAP diagnostic uses a fixed sample of 2,000 non-candidate development rows; the current XGBoost run uses native `pred_contribs` TreeSHAP because the installed SHAP loader cannot parse XGBoost 3.x vector base-score metadata. Directionality and dependence are associative transformed-feature diagnostics, not causality. No explanation model was fit on the locked candidate period, and no feature-importance result is used as evidence of biological causality.
""")


def generate_shap_outputs(selected_pipeline: Any, X: pd.DataFrame, selected_model: str, selected_set: str, logger: logging.Logger) -> None:
    """Generate a bounded SHAP diagnostic using development rows only."""
    if selected_model not in {"XGBoost", "LightGBM", "CatBoost", "Random Forest"}:
        write_text(REPORT / "shap_status.md", "SHAP was not run because the selected model was not tree-based.\n")
        return
    try:
        import shap

        base = selected_pipeline
        if hasattr(base, "estimator_"):
            base = base.estimator_
        if not hasattr(base, "named_steps") or "model" not in base.named_steps:
            raise RuntimeError("saved pipeline does not expose a tree estimator")
        model = base.named_steps["model"]
        if "preprocess" in base.named_steps:
            shap_input = X
            if "missingness" in base.named_steps:
                shap_input = base.named_steps["missingness"].transform(shap_input)
            transformed = base.named_steps["preprocess"].transform(shap_input)
            names = get_transformed_feature_names(base)
        else:
            transformed = base.named_steps["model"]._transform(X)
            names = list(getattr(model, "columns_", X.columns))
        sample_n = min(2000, len(X))
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(X), size=sample_n, replace=False)
        sample = transformed[sample_idx] if not isinstance(transformed, pd.DataFrame) else transformed.iloc[sample_idx]
        if hasattr(sample, "toarray"):
            sample_for_shap = sample.toarray()
        else:
            sample_for_shap = np.asarray(sample)
        shap_backend = "shap.TreeExplainer"
        try:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(sample_for_shap)
        except Exception as shap_exc:
            # XGBoost 3.x stores the scalar base score as a one-element vector
            # (for example, "[1.2792067E-2]"). Older SHAP loaders expect a
            # scalar and fail before evaluating any rows. XGBoost's native
            # pred_contribs is the same TreeSHAP contribution calculation and
            # provides a bounded, model-native fallback without refitting.
            if selected_model != "XGBoost":
                raise
            import xgboost as xgb

            native = model.get_booster().predict(xgb.DMatrix(sample_for_shap), pred_contribs=True)
            values = np.asarray(native)[:, :-1]
            shap_backend = "native XGBoost pred_contribs TreeSHAP"
            logger.warning("SHAP TreeExplainer unavailable; using native XGBoost TreeSHAP fallback: %s", shap_exc)
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        if len(names) != values.shape[1]:
            names = [f"transformed_{i}" for i in range(values.shape[1])]
        mean_abs = np.abs(values).mean(axis=0)
        direction = []
        for i, name in enumerate(names):
            column = sample_for_shap[:, i]
            corr = spearmanr(column, values[:, i]).statistic if np.std(column) > 0 and np.std(values[:, i]) > 0 else np.nan
            direction.append({"feature": str(name), "mean_abs_shap": float(mean_abs[i]), "spearman_feature_shap": float(corr) if np.isfinite(corr) else np.nan})
        table = pd.DataFrame(direction).sort_values("mean_abs_shap", ascending=False)
        table.to_csv(TABLE / "shap_global_importance.csv", index=False, encoding="utf-8-sig")
        top = table.head(15).sort_values("mean_abs_shap")
        fig, ax = plt.subplots(figsize=(9, 7)); ax.barh(top["feature"], top["mean_abs_shap"], color="#f28e2b"); ax.set_title(f"SHAP global importance: {selected_model} / {selected_set}"); ax.set_xlabel("Mean absolute SHAP value"); fig.tight_layout(); fig.savefig(FIGURE / "explainability_shap_summary.png", dpi=180); fig.savefig(FIGURE / "explainability_shap_summary.pdf"); plt.close(fig)
        top_name = str(table.iloc[0]["feature"])
        top_idx = int(table.index[0])
        fig, ax = plt.subplots(figsize=(8, 6)); ax.scatter(sample_for_shap[:, top_idx], values[:, top_idx], s=8, alpha=0.18, color="#e15759"); ax.set_xlabel(top_name); ax.set_ylabel("SHAP value"); ax.set_title(f"SHAP dependence diagnostic: {top_name}"); fig.tight_layout(); fig.savefig(FIGURE / "explainability_shap_dependence_top.png", dpi=180); fig.savefig(FIGURE / "explainability_shap_dependence_top.pdf"); plt.close(fig)
        write_text(REPORT / "shap_status.md", f"""SHAP was run on a fixed random sample of `{sample_n:,}` non-candidate development rows for `{selected_model}` / Feature Set `{selected_set}` using `{shap_backend}`. The summary is fold-independent because it uses the saved full-development candidate only as a development diagnostic; it is not test evidence. Global importance is in `tables/prompt3/shap_global_importance.csv`; focused summary/dependence figures are in `figures/prompt3/explainability_shap_summary.*` and `explainability_shap_dependence_top.*`. Directionality is an associative transformed-feature diagnostic, not causality.
""")
        logger.info("SHAP outputs written for %s/%s on n=%d", selected_model, selected_set, sample_n)
    except Exception as exc:
        write_text(REPORT / "shap_status.md", f"SHAP attempt failed for the development candidate: {type(exc).__name__}: {exc}. Native fold-fitted importance remains available; no locked-period data were used.\n")
        logger.warning("SHAP diagnostic unavailable: %s", exc)


def save_plots(primary_preds: dict[tuple[str, str], pd.DataFrame], selected_key: tuple[str, str], calibrated_raw: pd.DataFrame, calibrated_pred: pd.DataFrame, missingness_table: pd.DataFrame, importance: pd.DataFrame, alert_table: pd.DataFrame, period_table: pd.DataFrame) -> None:
    # Figure 1: concise experimental-flow diagram.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    boxes = [(0.02, 0.42, "Raw CSV\n119,743 rows"), (0.22, 0.42, "Deterministic QC\nraw retained"), (0.42, 0.42, "Development pool\nYear != 1404-2"), (0.62, 0.62, "Fold-safe pipelines\nA / B / C"), (0.62, 0.22, "Sensitivity\nD / M0-M4"), (0.84, 0.42, "Internal evidence\nPR-AUC + calibration")]
    for x, y, label in boxes:
        ax.text(x, y, label, ha="center", va="center", transform=ax.transAxes, bbox={"boxstyle": "round,pad=0.7", "facecolor": "#e8f0fe", "edgecolor": "#245"}, fontsize=11)
    for start, end in [((0.10, 0.48), (0.17, 0.48)), ((0.30, 0.48), (0.37, 0.48)), ((0.50, 0.48), (0.57, 0.68)), ((0.50, 0.42), (0.57, 0.30)), ((0.72, 0.68), (0.78, 0.48)), ((0.72, 0.30), (0.78, 0.40))]:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#245"})
    fig.savefig(FIGURE / "figure1_experimental_flow.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURE / "figure1_experimental_flow.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: PR curves for all primary candidates.
    fig, ax = plt.subplots(figsize=(10, 7))
    for (model, label), pred in primary_preds.items():
        pooled = pooled_predictions(pred)
        precision, recall, _ = precision_recall_curve(pooled["y_true"], pooled["predicted_probability"])
        ap = average_precision_score(pooled["y_true"], pooled["predicted_probability"])
        ax.plot(recall, precision, lw=1.2, label=f"{model} / {label} AP={ap:.3f}")
    first = next(iter(primary_preds.values()))
    baseline = float(pooled_predictions(first)["y_true"].mean())
    ax.axhline(baseline, color="black", ls="--", lw=1, label=f"no-skill={baseline:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Prompt 3 internal precision-recall curves")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIGURE / "pr_curves.png", dpi=180); fig.savefig(FIGURE / "pr_curves.pdf"); plt.close(fig)

    # Secondary ROC curves for the same internal predictions.
    fig, ax = plt.subplots(figsize=(10, 7))
    for (model, label), pred in primary_preds.items():
        pooled = pooled_predictions(pred)
        fpr, tpr, _ = roc_curve(pooled["y_true"], pooled["predicted_probability"])
        auc = __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(pooled["y_true"], pooled["predicted_probability"])
        ax.plot(fpr, tpr, lw=1.2, label=f"{model} / {label} AUROC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Prompt 3 internal ROC curves")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIGURE / "roc_curves.png", dpi=180); fig.savefig(FIGURE / "roc_curves.pdf"); plt.close(fig)

    # Figure 3: raw versus sigmoid calibration for selected candidate.
    fig, ax = plt.subplots(figsize=(8, 7))
    for method, pred, color in [("raw", calibrated_raw, "#245"), ("sigmoid", calibrated_pred, "#d55")]:
        pooled = pooled_predictions(pred)
        frac_pos, mean_pred = calibration_curve(pooled["y_true"], pooled["predicted_probability"], n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=method, color=color)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction positive", title="Calibration of selected candidate")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURE / "calibration_selected.png", dpi=180); fig.savefig(FIGURE / "calibration_selected.pdf"); plt.close(fig)

    # Figure 4: alert-budget capture and PPV.
    fig, ax1 = plt.subplots(figsize=(8, 6)); ax2 = ax1.twinx()
    ax1.plot(alert_table["budget_pct"], alert_table["mean_sensitivity"], marker="o", label="HAI capture / sensitivity", color="#245")
    ax2.plot(alert_table["budget_pct"], alert_table["mean_ppv"], marker="s", label="PPV", color="#d55")
    ax1.set(xlabel="Alert budget (%)", ylabel="Sensitivity / HAI capture"); ax2.set_ylabel("PPV"); ax1.set_title("Internal alert-budget operating points")
    fig.tight_layout(); fig.savefig(FIGURE / "alert_budget_curve.png", dpi=180); fig.savefig(FIGURE / "alert_budget_curve.pdf"); plt.close(fig)

    # Figure 5: missingness ablation.
    fig, ax = plt.subplots(figsize=(8, 6)); ax.bar(missingness_table["representation"], missingness_table["mean_average_precision"], color="#4c78a8"); ax.set(xlabel="Representation", ylabel="Mean Average Precision", title="Missingness-aware ablation"); fig.tight_layout(); fig.savefig(FIGURE / "missingness_ablation.png", dpi=180); fig.savefig(FIGURE / "missingness_ablation.pdf"); plt.close(fig)

    # Figure 6: focused importance summary.
    imp = importance[(importance["model_family"] == selected_key[0]) & (importance["feature_set"] == selected_key[1]) & (importance["strategy"] == "none")].groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(8, 6));
    if not imp.empty: ax.barh(imp["feature"], imp["importance"], color="#59a14f")
    ax.set_title("Fold-fitted source-field importance summary"); fig.tight_layout(); fig.savefig(FIGURE / "explainability_summary.png", dpi=180); fig.savefig(FIGURE / "explainability_summary.pdf"); plt.close(fig)

    if not period_table.empty:
        fig, ax = plt.subplots(figsize=(8, 6));
        ax.plot(period_table["group"], period_table["pr_auc"], marker="o", label="PR-AUC"); ax.plot(period_table["group"], period_table["brier"], marker="s", label="Brier"); ax.set_title("Coarse period robustness (candidate excluded)"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURE / "period_robustness.png", dpi=180); fig.savefig(FIGURE / "period_robustness.pdf"); plt.close(fig)


def write_feature_set_table(clean: pd.DataFrame) -> None:
    rows = []
    for label, cols in [("A", BASE_FEATURES_A), ("B", BASE_FEATURES_B), ("C", FEATURE_SETS["C"]), ("D", FEATURE_SETS["D"])]:
        rows.append({"feature_set": label, "n_predictor_columns": len(cols), "predictors": "; ".join(cols), "missingness_indicators": "yes" if label in {"C", "D"} else "no", "complaint_group": "yes" if label == "D" else "no", "primary_status": "sensitivity-only" if label == "D" else "primary-comparison"})
    for label, cols in [("M0", BASE_FEATURES_B), ("M1", FEATURE_SETS["C"]), ("M2", MISSINGNESS_COLUMNS_B), ("M3", LOW_MISSING_FIELDS), ("M4", HIGH_MISSING_INCLUDED)]:
        rows.append({"feature_set": label, "n_predictor_columns": len(cols), "predictors": "; ".join(cols), "missingness_indicators": "yes" if label in {"M1", "M2"} else "no", "complaint_group": "no", "primary_status": "missingness-ablation"})
    pd.DataFrame(rows).to_csv(TABLE / "feature_sets_and_missingness.csv", index=False, encoding="utf-8-sig")


def write_metric_tables(summary: pd.DataFrame, selected_model: str, selected_set: str, missingness_table: pd.DataFrame, imbalance_table: pd.DataFrame, calibration_table: pd.DataFrame, alert_table: pd.DataFrame, subgroup_table: pd.DataFrame) -> None:
    summary_b = summary[(summary["feature_set"] == "B") & (summary["strategy"] == "none")].copy()
    summary_c = summary[(summary["feature_set"] == "C") & (summary["strategy"] == "none")].copy()
    summary_b.to_csv(TABLE / "table_P3-2_model_comparison_feature_set_B.csv", index=False, encoding="utf-8-sig")
    summary_c.to_csv(TABLE / "table_P3-3_model_comparison_feature_set_C.csv", index=False, encoding="utf-8-sig")
    missingness_table.to_csv(TABLE / "table_P3-4_missingness_ablation.csv", index=False, encoding="utf-8-sig")
    imbalance_table.to_csv(TABLE / "table_P3-5_imbalance_strategy_comparison.csv", index=False, encoding="utf-8-sig")
    calibration_table.to_csv(TABLE / "table_P3-6_calibration_metrics.csv", index=False, encoding="utf-8-sig")
    alert_table.to_csv(TABLE / "table_P3-7_alert_budget_analysis.csv", index=False, encoding="utf-8-sig")
    subgroup_table.to_csv(TABLE / "table_P3-8_subgroup_robustness_summary.csv", index=False, encoding="utf-8-sig")
    write_text(REPORT / "table_P3-1_feature_sets_and_missingness.md", """# Table P3-1 - Feature sets and missingness

See `tables/prompt3/feature_sets_and_missingness.csv`. Feature Sets A-C are the primary hierarchy; Feature Set D is complaint sensitivity-only. M0-M4 are the pre-specified missingness representations.
""")


def error_analysis(clean: pd.DataFrame, pred: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    pooled = pooled_predictions(pred).set_index("row_index")
    frame = clean.loc[pooled.index].copy()
    frame["y_true"] = pooled["y_true"]
    frame["predicted_probability"] = pooled["predicted_probability"]
    threshold = float(np.quantile(frame["predicted_probability"], 0.95))
    frame["prediction_class_at_top5pct"] = (frame["predicted_probability"] >= threshold).astype(int)
    frame["error_class"] = np.select([(frame.y_true == 1) & (frame.prediction_class_at_top5pct == 0), (frame.y_true == 0) & (frame.prediction_class_at_top5pct == 1), (frame.y_true == 1) & (frame.prediction_class_at_top5pct == 1)], ["false_negative", "false_positive", "true_positive"], default="true_negative")
    records = []
    for label, group in frame.groupby("error_class"):
        records.append({"error_class": label, "n": int(len(group)), "positive_n": int(group.y_true.sum()), "mean_probability": float(group.predicted_probability.mean()), "mean_age": float(pd.to_numeric(group.Age, errors="coerce").mean()), "mean_triage": float(pd.to_numeric(group["Triage level"], errors="coerce").mean()), "mean_missing_count_B": float(group.missing_count_B.mean())})
    table = pd.DataFrame(records)
    table.to_csv(TABLE / "error_analysis_summary.csv", index=False, encoding="utf-8-sig")
    group_dist = frame.groupby(["error_class", "complaint_group"], dropna=False).size().reset_index(name="n")
    group_dist.to_csv(TABLE / "error_analysis_complaint_groups.csv", index=False, encoding="utf-8-sig")
    text = f"""# Error analysis

The leading candidate was analyzed using an exploratory top-5% operating point with pooled internal predictions. This threshold is not a locked clinical threshold. The four outcome/error classes are summarized in `tables/prompt3/error_analysis_summary.csv`; complaint-group distributions are in `tables/prompt3/error_analysis_complaint_groups.csv`.

The analysis compares age, sex, triage, major clinical fields, missingness burden, complaint group, period, and Department descriptively. It does not interpret errors causally and contains no locked-period predictions.

{table.to_string(index=False)}
"""
    return table, text


def period_robustness(clean: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    pooled = pooled_predictions(pred).set_index("row_index")
    frame = clean.loc[pooled.index, ["Year"]].copy()
    frame["y_true"] = pooled["y_true"]
    frame["predicted_probability"] = pooled["predicted_probability"]
    rows = []
    for period, group in frame.groupby("Year", dropna=False):
        metrics = binary_metrics(group.y_true, group.predicted_probability, budget=0.05)
        rows.append({"group": str(period), "n": int(len(group)), "positive_n": int(group.y_true.sum()), "prevalence": float(group.y_true.mean()), "pr_auc": metrics["average_precision"] if group.y_true.nunique() == 2 else np.nan, "auroc": metrics["auroc"], "brier": metrics["brier"], "calibration_slope": metrics["calibration_slope"], "calibration_in_the_large": metrics["calibration_in_the_large"]})
    return pd.DataFrame(rows).sort_values("group")


def complaint_sensitivity_report(clean: pd.DataFrame, c_pred: pd.DataFrame, d_pred: pd.DataFrame, selected_model: str) -> tuple[pd.DataFrame, str]:
    c = pooled_predictions(c_pred).set_index("row_index")
    d = pooled_predictions(d_pred).set_index("row_index")
    rows = []
    for label, frame in [("C", c), ("D", d)]:
        rows.append({"representation": label, "model_family": selected_model, "n": int(len(frame)), "pr_auc": float(average_precision_score(frame.y_true, frame.predicted_probability)), "brier": float(np.mean((frame.y_true - frame.predicted_probability) ** 2)), "top5_ppv": top_fraction_metrics(frame.y_true, frame.predicted_probability, 0.05)["ppv"]})
    table = pd.DataFrame(rows)
    table["delta_vs_C"] = table["pr_auc"] - float(table.loc[table.representation == "C", "pr_auc"].iloc[0])
    table.to_csv(TABLE / "complaint_sensitivity_results.csv", index=False, encoding="utf-8-sig")
    infection_group = clean.loc[d.index].groupby("complaint_group")["complaint_infection_related"].max()
    d_frame = clean.loc[d.index].copy(); d_frame["y_true"] = d.y_true; d_frame["predicted_probability"] = d.predicted_probability
    infection = d_frame[d_frame["complaint_infection_related"]]
    concentration = float(infection.y_true.mean()) if len(infection) else float("nan")
    delta = float(table.loc[table.representation == "D", "delta_vs_C"].iloc[0])
    status = "POSSIBLE SHORTCUT" if delta > 0.02 and concentration > d_frame.y_true.mean() * 2 else "HELPFUL" if delta > 0.005 else "NEUTRAL"
    text = f"""# Complaint sensitivity analysis

Selected family: `{selected_model}`. Feature Set D is sensitivity-only. The comparison is in `tables/prompt3/complaint_sensitivity_results.csv`.

Feature Set D changed Average Precision by `{delta:.6f}` relative to C. Infection-related complaint rows had prevalence `{concentration:.6f}` in this internal sample versus `{d_frame.y_true.mean():.6f}` overall. Complaint interpretation: **{status}**. A large gain concentrated in infection-related groups would be a possible shortcut signal, not evidence of causal clinical value.

The grouping was frozen label-blind before performance analysis. No raw complaint text or complaint-derived outcome label was used in primary Feature Sets A-C.
"""
    return table, text


def write_period_report(period_table: pd.DataFrame) -> None:
    if period_table.empty:
        body = "No period result was available."
    else:
        body = period_table.to_string(index=False)
    write_text(REPORT / "10_period_robustness.md", f"""# Period robustness

This is coarse period robustness only. `Year` was not a predictor, and the unresolved locked candidate `1404-2` was excluded completely. The table reports internal prediction behavior by observed non-candidate category; it is not external temporal validation or proof of chronological generalizability.

{body}
""")


def write_robustness_report(subgroup_table: pd.DataFrame) -> None:
    text = """# Internal robustness

The leading candidate was evaluated descriptively across sex, age bands, triage, available non-candidate periods, missingness severity, and common Department groups. Department was used only as a descriptive stratification variable. Positive-event safeguards from Prompt 2 were applied: groups with fewer than 20 positives are counts/descriptive only; at least 50 positives is preferred for interpretive comparison.

Patient-independent, fairness, and external-validity claims are not supported by this release.

""" + subgroup_table.to_string(index=False)
    write_text(REPORT / "09_internal_robustness.md", text)


def write_missingness_report(table: pd.DataFrame, selected_model: str) -> tuple[float, str]:
    m0 = float(table.loc[table.representation == "M0", "mean_average_precision"].iloc[0])
    m1 = float(table.loc[table.representation == "M1", "mean_average_precision"].iloc[0])
    delta = m1 - m0
    if delta > 0.005:
        status = "HELPFUL"
    elif delta < -0.005:
        status = "HARMFUL"
    else:
        status = "NEUTRAL"
    write_text(REPORT / "05_missingness_ablation.md", f"""# Missingness ablation

Selected family: `{selected_model}`. The pre-specified representations M0-M4 were evaluated with the same development-only folds and no locked-period data. All learned preprocessing was fitted within each training fold.

M1 (values plus indicators) changed mean Average Precision by `{delta:.6f}` relative to M0 (values only), classified for the terminal summary as **{status}**. The result is predictive association only; it does not establish biological causality or transportability. Brier, calibration, alert-budget, and uncertainty columns are in `tables/prompt3/missingness_ablation_results.csv`.

Low-missing M3 uses the frozen Prompt 2 documented `<10%` missingness subset. M4 excludes fields above the frozen `>50%` high-missingness threshold. These thresholds were recorded before modeling.
""")
    return delta, status


def write_imbalance_report(table: pd.DataFrame, selected_model: str) -> str:
    base = table[table.strategy == "none"].iloc[0]
    best = table.sort_values("mean_average_precision", ascending=False).iloc[0]
    effect = float(best.mean_average_precision - base.mean_average_precision)
    if best.strategy == "none":
        status = "No weighting/resampling was best in this development comparison."
    else:
        status = f"{best.strategy} had the highest development AP difference of {effect:.6f}; calibration must govern any later choice."
    write_text(REPORT / "06_imbalance_sensitivity.md", f"""# Class-imbalance sensitivity

Selected family: `{selected_model}` on Feature Set C. The primary strategy was no resampling. Random undersampling and oversampling were performed only on each training fold; validation data retained natural prevalence. SMOTE was not run because the protocol makes it optional and the mixed missing/categorical/workflow structure makes synthetic interpolation scientifically questionable.

{status}

Full PR-AUC, Brier, calibration, PPV, sensitivity, and alert-burden summaries are in `tables/prompt3/imbalance_sensitivity_results.csv`. No single strategy is promoted from AP alone.
""")
    return status


def write_calibration_report(table: pd.DataFrame, selected_model: str, selected_set: str) -> str:
    best = table.sort_values("pooled_brier").iloc[0]
    text = f"""# Calibration analysis

The leading development candidate was `{selected_model}` / Feature Set `{selected_set}`. Sigmoid/Platt calibration was fitted inside the development training structure only; no locked-period observations were used. Raw and calibrated curves are in `figures/prompt3/calibration_selected.png` and `.pdf`.

The lowest pooled Brier row was method `{best['method']}` with Brier `{best['pooled_brier']:.6f}` and calibration slope `{best['pooled_calibration_slope']:.6f}`. Calibration-in-the-large, confidence intervals, and fold-level values are in `tables/prompt3/calibration_metrics.csv`.

Calibration quality is reported separately from discrimination. No clinical threshold was locked.
"""
    write_text(REPORT / "07_calibration_analysis.md", text)
    return str(best["method"])


def build_master_report(
    full_stats: dict[str, Any],
    clean: pd.DataFrame,
    summary: pd.DataFrame,
    tuning: pd.DataFrame,
    selected: pd.Series,
    secondary: pd.Series,
    calibration_table: pd.DataFrame,
    missingness_table: pd.DataFrame,
    imbalance_table: pd.DataFrame,
    alert_table: pd.DataFrame,
    subgroup_table: pd.DataFrame,
    period_table: pd.DataFrame,
    complaint_table: pd.DataFrame,
    error_table: pd.DataFrame,
    selected_calibration: str,
    missingness_status: str,
    imbalance_status: str,
    complaint_status: str,
    q1_signal: str,
    cbm: str,
    aim: str,
    prompt4: str,
) -> None:
    selected_key = f"{selected['model_family']} / Feature Set {selected['feature_set']}"
    best_ap = summary[summary.strategy == "none"].sort_values("mean_average_precision", ascending=False).iloc[0]
    best_brier = summary[summary.strategy == "none"].sort_values("mean_brier").iloc[0]
    top5 = alert_table.loc[alert_table.budget_pct == 5].iloc[0]
    period_delta = float(period_table.pr_auc.max() - period_table.pr_auc.min()) if not period_table.empty else float("nan")
    subgroup_reportable = subgroup_table[subgroup_table.reporting_status == "reportable exploratory"]
    text = f"""# Prompt 3 Master Model Development Report

**Status:** `PASS WITH WARNINGS`  
**Protocol:** Prompt 2B / `STUDY_PROTOCOL_v2.md`  
**Evaluation unit:** encounter-level retrospective  
**Locked test touched:** `NO`

## 1. Executive summary

Prompt 3 completed the authorized development analysis without using `1404-2`. The full source reconstruction matched 119,743 rows and 1,567 positives; the modeled development pool contains `{int(clean[clean.Year != LOCKED_YEAR].shape[0]):,}` rows and `{int(clean.loc[clean.Year != LOCKED_YEAR, 'Label'].sum()):,}` positives. The primary candidate selected by the pre-specified multi-criteria policy is `{selected_key}` with development mean AP `{selected['mean_average_precision']:.6f}` and mean Brier `{selected['mean_brier']:.6f}`. This is internal evidence only; outcome definition, field-level timing, patient independence, and external validation remain unresolved.

## 2. Data/cohort verification

The raw hash, exact full counts, parsing failures, QC flags, no missing-predictor row exclusions, and candidate-period exclusion are documented in `reports/prompt3/01_analytic_dataset_reconstruction.md`. The source raw CSV was not altered. Derived lineage data are under `data/derived/prompt3_analytic_dataset.parquet`.

## 3. Validation actually executed

Development-only stratified three-fold resampling was repeated for seeds 42 and 2024 on non-candidate periods. No Year predictor, temporal test claim, patient-independent claim, or locked-candidate prediction was made.

## 4. Feature Sets A-D

Feature Set A is Age/Sex/Triage. B adds first-day vitals/labs. C adds fold-safe missingness indicators. D adds the frozen label-blind complaint grouping and remains sensitivity-only. Feature-set definitions are in `tables/prompt3/feature_sets_and_missingness.csv`.

## 5. Preprocessing

Numeric median imputation, categorical missing-category encoding, one-hot encoding, scaling for Logistic Regression, and missingness indicators were fitted within training folds. CatBoost used native categorical handling with training-fold numeric medians. Excluded fields were absent from all primary matrices.

## 6. Model families

Required families tested: {", ".join(MODEL_FAMILIES)}. No deep neural model or EBM was added.

## 7. Hyperparameter tuning

Bounded two-fold Feature Set B searches used at most three sampled configurations per family and optimized AP. Search spaces and executed configurations are in `tables/prompt3/hyperparameter_search_spaces.csv` and `tables/prompt3/tuning_results.csv`.

## 8. Feature-set comparison

All required families were compared on A, B, and C with identical repeated folds. B and C tables are in `tables/prompt3/table_P3-2_model_comparison_feature_set_B.csv` and `tables/prompt3/table_P3-3_model_comparison_feature_set_C.csv`.

## 9. Missingness ablation

M0-M4 were run for the selected family. M1 classification: **{missingness_status}**. Full results: `tables/prompt3/missingness_ablation_results.csv`.

## 10. Imbalance sensitivity

No resampling was primary. Fold-local class weighting, random 3:1 negative undersampling, and random 3:1 positive oversampling were compared on natural-prevalence validation data. SMOTE was not run. Finding: {imbalance_status}

## 11. Calibration

Raw and sigmoid-calibrated development predictions were compared with fold-safe calibration. Selected calibration method for the saved candidate: `{selected_calibration}`. Results: `tables/prompt3/calibration_metrics.csv`.

## 12. PR-AUC results

Best mean AP candidate: `{best_ap['model_family']} / {best_ap['feature_set']}` at `{best_ap['mean_average_precision']:.6f}` (95% CI `{best_ap['ci_low_average_precision']:.6f}` to `{best_ap['ci_high_average_precision']:.6f}`). No-skill baseline in the modeled development pool is `{EXPECTED_FULL_PREVALENCE:.6f}` only for the full release; the development-pool baseline is `{clean.loc[clean.Year != LOCKED_YEAR, 'Label'].mean():.6f}`.

## 13. Alert-budget results

At the internal top-5% operating point, mean HAI capture/sensitivity was `{top5['mean_sensitivity']:.6f}` and mean PPV was `{top5['mean_ppv']:.6f}`. This is an operating-point analysis, not a locked clinical threshold.

## 14. Statistical uncertainty

Primary tables report fold-level means and 95% t-based intervals across the six fixed resamples. Paired comparisons use identical folds where applicable. Small numerical differences are not treated as meaningful without stability support.

## 15. Internal robustness

Subgroup results are in `tables/prompt3/subgroup_performance.csv`. `{len(subgroup_reportable)}` subgroup rows met the minimum 20-positive exploratory reporting threshold; groups below that threshold remain descriptive only.

## 16. Period robustness

Only non-candidate Year categories were summarized. PR-AUC range across available periods was `{period_delta:.6f}`. This is coarse period/distribution robustness, not external temporal validation.

## 17. Error analysis

False-negative, false-positive, true-positive, and true-negative summaries are in `tables/prompt3/error_analysis_summary.csv`. Errors are descriptive and not causal findings.

## 18. Explainability

The saved leading tree/linear model has fold-fitted source-field importance summaries in `tables/prompt3/feature_importance_stability.csv` and `figures/prompt3/explainability_summary.png`. A bounded TreeSHAP diagnostic on 2,000 non-candidate development rows is in `tables/prompt3/shap_global_importance.csv` and the paired SHAP figures. The current XGBoost output uses native `pred_contribs` TreeSHAP because the installed SHAP loader cannot parse XGBoost 3.x vector base-score metadata. Explanations are associative, not causal.

## 19. Feature-importance stability

Importance is recorded across the six primary folds and aggregated back to source fields. Stability interpretation is in `reports/prompt3/13_feature_stability.md`; unstable rankings are not promoted as scientific mechanisms.

## 20. Complaint sensitivity

Feature Set D changed AP by `{complaint_table.loc[complaint_table.representation == 'D', 'delta_vs_C'].iloc[0]:.6f}` relative to C. Terminal classification: **{complaint_status}**. Full comparison: `tables/prompt3/complaint_sensitivity_results.csv`.

## 21. Primary candidate model selection

Selected primary: `{selected_key}`. Composite selection score: `{selected['selection_rank_score']:.4f}`. Selection considered AP, Brier, calibration slope, fold stability, and complexity before the saved full-development pipeline was fit.

## 22. Secondary candidate

Secondary development candidate: `{secondary['model_family']} / Feature Set {secondary['feature_set']}`. It is reported for comparison and is not a final clinical model.

## 23. Remaining concerns

The HAI outcome definition and exact onset criteria are unverified; exact field-level timing is unavailable; patient-independent validation is unsupported; Year ordering/`1404-2` remain unresolved; and this is a single-release retrospective analysis without external or prospective validation.

## 24. Q1-readiness after Prompt 3

Q1 signal: **{q1_signal}**. CBM readiness: **{cbm}**. AI in Medicine readiness: **{aim}**. The results provide evidence beyond a bare algorithm leaderboard only if the incremental feature, missingness, calibration, alert, and stability findings remain reproducible in later robustness work.

## 25. Exact Prompt 4 requirements

Prompt 4 recommendation: **{prompt4}**. If continued, Prompt 4 must focus on robustness, clinical reliability, final model freezing, locked-period governance after temporal clarification, uncertainty, and transparent handling of all unresolved outcome/timing/patient-independence limitations. It must not erase this development evidence or retroactively tune on `1404-2`.

## Machine-readable outputs

- `artifacts/prompt3/fold_results.parquet`
- `artifacts/prompt3/predictions_internal.parquet`
- `artifacts/prompt3/model_summary.csv`
- `reports/prompt3_summary.json`
"""
    write_text(ROOT / "reports" / "PROMPT3_MASTER_MODEL_DEVELOPMENT.md", text)


def write_summary_json(
    clean: pd.DataFrame,
    selected: pd.Series,
    best_ap: pd.Series,
    best_brier: pd.Series,
    selected_calibration: str,
    missingness_delta: float,
    missingness_status: str,
    imbalance_table: pd.DataFrame,
    complaint_status: str,
    subgroup_table: pd.DataFrame,
    period_table: pd.DataFrame,
    alert_table: pd.DataFrame,
    q1_signal: str,
    cbm: str,
    aim: str,
    prompt4: str,
    deviation_count: int,
) -> None:
    dev = clean[clean["Year"] != LOCKED_YEAR]
    top5 = alert_table.loc[alert_table.budget_pct == 5].iloc[0]
    best_cal_brier = float(best_brier["mean_brier"])
    sub_instability = "No reportable subgroup had enough events for a stable comparative claim." if subgroup_table[subgroup_table.reporting_status == "reportable exploratory"].empty else "Exploratory subgroup variation is present; event thresholds and uncertainty are reported in subgroup_performance.csv."
    period_instability = "No candidate-period result exists; non-candidate coarse period differences are descriptive only." if period_table.empty else f"Non-candidate period PR-AUC range={period_table.pr_auc.max() - period_table.pr_auc.min():.6f}; coarse period robustness only."
    weight_row = imbalance_table.sort_values("mean_average_precision", ascending=False).iloc[0]
    summary = {
        "analytic_n": int(len(dev)),
        "positive_n": int(dev["Label"].sum()),
        "prevalence": float(dev["Label"].mean()),
        "full_release_n": int(len(clean)),
        "full_release_positive_n": int(clean["Label"].sum()),
        "validation_type": "development-only stratified 3-fold internal resampling repeated over seeds 42 and 2024; encounter-level retrospective",
        "models_tested": MODEL_FAMILIES,
        "feature_sets": ["A", "B", "C", "D", "M0", "M1", "M2", "M3", "M4"],
        "primary_metric": "Average Precision / PR-AUC",
        "best_model_by_pr_auc": {"model": best_ap["model_family"], "feature_set": best_ap["feature_set"], "estimate": best_ap["mean_average_precision"], "ci95": [best_ap["ci_low_average_precision"], best_ap["ci_high_average_precision"]]},
        "best_model_by_calibration": {"model": best_brier["model_family"], "feature_set": best_brier["feature_set"], "brier": best_brier["mean_brier"], "calibration_slope": best_brier["mean_calibration_slope"]},
        "selected_primary_candidate": selected["model_family"],
        "selected_feature_set": selected["feature_set"],
        "selected_preprocessing": "Fold-fitted median numeric imputation; explicit categorical missing category; one-hot encoding; Logistic Regression scaling only; CatBoost native categories",
        "selected_imbalance_strategy": "none / no resampling",
        "selected_calibration_method": selected_calibration,
        "internal_pr_auc": selected["pooled_average_precision"],
        "internal_pr_auc_ci": [selected["ci_low_average_precision"], selected["ci_high_average_precision"]],
        "internal_auroc": selected["pooled_auroc"],
        "internal_brier": selected["pooled_brier"],
        "top_1pct_capture": float(alert_table.loc[alert_table.budget_pct == 1, "mean_sensitivity"].iloc[0]),
        "top_2pct_capture": float(alert_table.loc[alert_table.budget_pct == 2, "mean_sensitivity"].iloc[0]),
        "top_5pct_capture": top5["mean_sensitivity"],
        "top_10pct_capture": float(alert_table.loc[alert_table.budget_pct == 10, "mean_sensitivity"].iloc[0]),
        "top_5pct_ppv": top5["mean_ppv"],
        "missingness_ablation_effect": {"m1_minus_m0_pr_auc": missingness_delta, "classification": missingness_status},
        "class_weight_effect": {"best_strategy": weight_row["strategy"], "best_strategy_minus_none_pr_auc": float(weight_row["mean_average_precision"] - imbalance_table.loc[imbalance_table.strategy == "none", "mean_average_precision"].iloc[0])},
        "complaint_effect": complaint_status,
        "major_subgroup_instabilities": sub_instability,
        "major_period_instabilities": period_instability,
        "primary_candidate_reason": "Highest pre-specified composite score combining PR-AUC, Brier, calibration slope, fold stability, and complexity; not selected from PR-AUC alone.",
        "locked_test_touched": False,
        "locked_candidate_period": LOCKED_YEAR,
        "protocol_deviations": deviation_count,
        "cbm_readiness": cbm,
        "ai_in_medicine_readiness": aim,
        "q1_signal": q1_signal,
        "prompt4_recommendation": prompt4,
        "selected_candidate_pooled_metrics": {k: selected[k] for k in selected.index if k.startswith("pooled_")},
    }
    write_json(ROOT / "reports" / "prompt3_summary.json", summary)


def finalize_existing() -> None:
    """Assemble reports from a completed fit when only finalization failed."""
    global _ACTIVE_CLEAN
    ensure_dirs()
    logger = configure_logging()
    raw_hash = raw_sha256(RAW)
    raw = pd.read_csv(RAW, dtype="string", keep_default_na=False, na_filter=False, low_memory=False)
    clean, qc, _, stats, _ = reconstruct_dataset(raw)
    _ACTIVE_CLEAN = clean
    write_analytic_artifact(raw, clean, qc, raw_hash)
    summary = pd.read_csv(ARTIFACT / "model_summary.csv")
    ranking = pd.read_csv(TABLE / "model_selection_ranking.csv")
    selected, secondary = ranking.iloc[0], ranking.iloc[1]
    tuning = pd.read_csv(TABLE / "tuning_results.csv")
    calibration_table = pd.read_csv(TABLE / "calibration_metrics.csv")
    missingness_table = pd.read_csv(TABLE / "missingness_ablation_results.csv")
    imbalance_table = pd.read_csv(TABLE / "imbalance_sensitivity_results.csv")
    alert_table = pd.read_csv(TABLE / "alert_budget_results.csv")
    subgroup_table = pd.read_csv(TABLE / "subgroup_performance.csv")
    period_table = pd.read_csv(TABLE / "period_robustness_results.csv")
    complaint_table = pd.read_csv(TABLE / "complaint_sensitivity_results.csv")
    error_table = pd.read_csv(TABLE / "error_analysis_summary.csv")
    predictions = pd.read_parquet(ARTIFACT / "predictions_internal.parquet")
    dev = clean[clean["Year"] != LOCKED_YEAR]
    pred_rows = predictions[predictions["row_index"].isin(dev.index)]
    assert len(pred_rows) == len(predictions)
    assert not clean.loc[predictions["row_index"], "Year"].eq(LOCKED_YEAR).any()
    selected_model, selected_set = selected["model_family"], selected["feature_set"]
    primary_preds = {}
    for model in MODEL_FAMILIES:
        for label in ["A", "B", "C"]:
            primary_preds[(model, label)] = predictions[(predictions["model_family"] == model) & (predictions["feature_set"] == label) & (predictions["strategy"] == "none")]
    importance_all = pd.read_csv(TABLE / "feature_importance_stability.csv")
    write_explainability_report(importance_all, selected_model, selected_set)
    saved_primary = joblib.load(MODEL / "selected_primary_pipeline.joblib")
    generate_shap_outputs(saved_primary, input_frame(dev, selected_set), selected_model, selected_set, logger)
    save_plots(
        primary_preds,
        (selected_model, selected_set),
        predictions[(predictions["model_family"] == selected_model) & (predictions["feature_set"] == selected_set) & (predictions["strategy"] == "none") & (predictions["seed"] == 42)],
        predictions[(predictions["model_family"] == selected_model) & (predictions["feature_set"] == selected_set) & (predictions["strategy"] == "sigmoid")],
        missingness_table,
        importance_all,
        alert_table,
        period_table,
    )
    selected_pred = predictions[(predictions["model_family"] == selected_model) & (predictions["feature_set"] == selected_set) & (predictions["strategy"] == "none")]
    seed42 = selected_pred[selected_pred["seed"] == 42]
    missing_delta = float(missingness_table.loc[missingness_table.representation == "M1", "mean_average_precision"].iloc[0] - missingness_table.loc[missingness_table.representation == "M0", "mean_average_precision"].iloc[0])
    missing_status = "HELPFUL" if missing_delta > 0.005 else "HARMFUL" if missing_delta < -0.005 else "NEUTRAL"
    imbalance_status = write_imbalance_report(imbalance_table, selected_model)
    d_pred = predictions[(predictions["model_family"] == selected_model) & (predictions["feature_set"] == "D") & (predictions["strategy"] == "none")]
    complaint_delta = float(complaint_table.loc[complaint_table.representation == "D", "delta_vs_C"].iloc[0])
    d_pool = pooled_predictions(d_pred).set_index("row_index")
    infection = clean.loc[d_pool.index, "complaint_infection_related"]
    infection_y = d_pool.loc[infection[infection].index, "y_true"] if infection.any() else pd.Series(dtype=float)
    infection_prev = float(infection_y.mean()) if len(infection_y) else 0.0
    complaint_status = "POSSIBLE SHORTCUT" if complaint_delta > 0.02 and infection_prev > float(d_pool.y_true.mean()) * 2 else "HELPFUL" if complaint_delta > 0.005 else "NEUTRAL"
    selected_calibration = str(calibration_table.sort_values("pooled_brier").iloc[0]["method"])
    b_rows = summary[(summary.feature_set == "B") & (summary.strategy == "none")].sort_values("mean_average_precision", ascending=False)
    c_rows = summary[(summary.feature_set == "C") & (summary.strategy == "none")].sort_values("mean_average_precision", ascending=False)
    a_rows = summary[(summary.feature_set == "A") & (summary.strategy == "none")].sort_values("mean_average_precision", ascending=False)
    b_vs_a = float(b_rows.iloc[0].mean_average_precision - a_rows.iloc[0].mean_average_precision)
    c_vs_b = float(c_rows.iloc[0].mean_average_precision - b_rows.iloc[0].mean_average_precision)
    top5 = alert_table.loc[alert_table.budget_pct == 5].iloc[0]
    period_range = float(period_table.pr_auc.max() - period_table.pr_auc.min()) if not period_table.empty else 0.0
    if b_vs_a > 0.005 and c_vs_b > 0.002 and float(top5.mean_ppv) > float(dev.Label.mean()) * 2 and period_range < 0.05:
        q1_signal, cbm, aim, prompt4 = "STRONG Q1 SIGNAL", "STRONG", "CONDITIONAL", "RECOMMENDED"
    elif float(selected["pooled_average_precision"]) > float(dev.Label.mean()) and (b_vs_a > 0 or c_vs_b > 0):
        q1_signal, cbm, aim, prompt4 = "PROMISING BUT INCOMPLETE", "MODERATE", "CONDITIONAL", "RECOMMENDED"
    else:
        q1_signal, cbm, aim, prompt4 = "WEAK", "WEAK", "WEAK", "NOT RECOMMENDED"
    governance_path = ROOT / "docs" / "STUDY_GOVERNANCE.md"
    governance_text = governance_path.read_text(encoding="utf-8") if governance_path.exists() else ""
    deviation_count = sum(1 for line in governance_text.splitlines() if line.startswith("## "))
    write_summary_json(clean, selected, summary[summary.strategy == "none"].sort_values("mean_average_precision", ascending=False).iloc[0], summary[summary.strategy == "none"].sort_values("mean_brier").iloc[0], selected_calibration, missing_delta, missing_status, imbalance_table, complaint_status, subgroup_table, period_table, alert_table, q1_signal, cbm, aim, prompt4, deviation_count)
    build_master_report(stats, clean, summary, tuning, selected, secondary, calibration_table, missingness_table, imbalance_table, alert_table, subgroup_table, period_table, complaint_table, error_table, selected_calibration, missing_status, imbalance_status, complaint_status, q1_signal, cbm, aim, prompt4)
    first_log = (LOG_DIR / "prompt3_run.log").read_text(encoding="utf-8").splitlines()[0] if (LOG_DIR / "prompt3_run.log").exists() else ""
    if first_log:
        # The logger uses the host-local clock; its historical suffix was not
        # a UTC conversion.  Normalize that recorded start to real UTC.
        local_stamp = first_log.split(" ", 1)[0].replace("Z", "")
        execution_timestamp = datetime.fromisoformat(local_stamp).replace(tzinfo=ZoneInfo("Asia/Tehran")).astimezone(timezone.utc).isoformat()
    else:
        execution_timestamp = datetime.now(timezone.utc).isoformat()
    manifest = {
        "prompt": "Prompt 3",
        "protocol_version": "STUDY_PROTOCOL_v2 / prompt2b-v2",
        "execution_timestamp_utc": execution_timestamp,
        "completion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data": {"path": str(RAW.relative_to(ROOT)), "sha256": raw_hash, "full_n": stats["full_n"], "full_positive_n": stats["full_positive_n"], "locked_candidate_period": LOCKED_YEAR, "locked_candidate_touched": False},
        "package_versions": package_versions(),
        "random_seeds": SEEDS,
        "hardware": {"platform": platform.platform(), "python": sys.version.replace("\n", " "), "cpu_count": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2)},
        "git_commit": git_commit_if_applicable(),
        "model_families": MODEL_FAMILIES,
        "feature_sets": {k: v for k, v in FEATURE_SETS.items()},
        "validation_definitions": {"evaluation_unit": "encounter-level retrospective", "development_years": ["1402", "1403", "1404"], "excluded_locked_candidate": LOCKED_YEAR, "split": "stratified 3-fold repeated over seeds 42 and 2024", "temporal_validation": "PARTIAL / COARSE PERIOD VALIDATION ONLY", "patient_independence": "UNSUPPORTED"},
        "tuning": {"scope": "Feature Set B", "folds": 2, "max_configurations_per_model": 3, "metric": "Average Precision"},
        "artifacts": {"predictions": "artifacts/prompt3/predictions_internal.parquet", "fold_results": "artifacts/prompt3/fold_results.parquet", "primary_pipeline": "models/prompt3/selected_primary_pipeline.joblib"},
        "finalization": "assembled from completed internal artifacts after terminal report-assembly retry; no refit during finalization",
        "fit_log_timezone": "Asia/Tehran",
    }
    write_json(REPORT / "PROMPT3_RUN_MANIFEST.json", manifest)
    write_text(LOG_DIR / "prompt3_completion.txt", f"Prompt 3 completed at {datetime.now(timezone.utc).isoformat()} with locked candidate touched=false.\n")
    logger.info("Prompt 3 finalization complete; selected=%s/%s q1=%s", selected_model, selected_set, q1_signal)


def main() -> None:
    global _ACTIVE_CLEAN
    ensure_dirs()
    logger = configure_logging()
    start = datetime.now(timezone.utc)
    logger.info("Prompt 3 start; root=%s", ROOT)
    raw_hash = raw_sha256(RAW)
    raw = pd.read_csv(RAW, dtype="string", keep_default_na=False, na_filter=False, low_memory=False)
    clean, qc, _, stats, _ = reconstruct_dataset(raw)
    _ACTIVE_CLEAN = clean
    if stats["full_n"] != EXPECTED_FULL_N or stats["full_positive_n"] != EXPECTED_FULL_POSITIVE_N or stats["rbc_date_serial_like_n"] != EXPECTED_RBC_DATE_LIKE_N:
        write_reconstruction_report(stats, clean, raw_hash)
        raise RuntimeError(f"Protocol count discrepancy: {stats}")
    write_analytic_artifact(raw, clean, qc, raw_hash)
    write_reconstruction_report(stats, clean, raw_hash)
    write_complaint_mapping(raw, clean)
    write_feature_set_table(clean)

    dev_mask = clean["Year"].isin(["1402", "1403", "1404"])
    dev_positions = np.flatnonzero(dev_mask.to_numpy())
    dev_clean = clean.iloc[dev_positions].copy()
    y = dev_clean["Label"].astype(int)
    splits = build_splits(len(dev_clean), y)
    write_validation_plan(clean, splits)
    write_pre_model_protocol_docs()
    logger.info("reconstruction PASS; dev_n=%d dev_positive_n=%d candidate_rows=%d", len(dev_clean), int(y.sum()), int((clean["Year"] == LOCKED_YEAR).sum()))

    X_by_set = {label: input_frame(dev_clean, label) for label in ["A", "B", "C", "D"]}
    tuning_best: dict[str, dict[str, Any]] = {}
    tuning_frames: list[pd.DataFrame] = []
    X_tune = X_by_set["B"]
    for model in MODEL_FAMILIES:
        best, frame = tune_model(model, X_tune, y, logger)
        tuning_best[model] = best
        tuning_frames.append(frame)
    write_tuning_results(tuning_frames)
    tuning_all = pd.concat(tuning_frames, ignore_index=True)

    primary_predictions: dict[tuple[str, str], pd.DataFrame] = {}
    primary_fold_metrics: dict[tuple[str, str], pd.DataFrame] = {}
    importance_frames: list[pd.DataFrame] = []
    all_fold_metric_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for model in MODEL_FAMILIES:
        for label in ["A", "B", "C"]:
            pred, folds, importance = run_cv(model, label, X_by_set[label], y, splits, tuning_best[model], "none", logger, collect_importance=True)
            primary_predictions[(model, label)] = pred
            primary_fold_metrics[(model, label)] = folds
            fold_record = folds.copy()
            fold_record["model_family"], fold_record["feature_set"], fold_record["strategy"] = model, label, "none"
            all_fold_metric_frames.append(fold_record)
            importance_frames.append(importance)
            summary_rows.append(summarize_run(pred, folds, model, label, "none"))
    summary = pd.DataFrame(summary_rows)
    ranked, selected, secondary = rank_primary_candidates(summary)
    ranked.to_csv(TABLE / "model_selection_ranking.csv", index=False, encoding="utf-8-sig")
    importance_all = pd.concat(importance_frames, ignore_index=True)
    importance_all.to_csv(TABLE / "feature_importance_stability.csv", index=False, encoding="utf-8-sig")
    feature_importance_report(importance_all, selected["model_family"], selected["feature_set"])
    write_explainability_report(importance_all, selected["model_family"], selected["feature_set"])
    selected_model, selected_set = selected["model_family"], selected["feature_set"]
    selected_pred_all = primary_predictions[(selected_model, selected_set)]
    selected_pred_seed42 = selected_pred_all[selected_pred_all["seed"] == 42].copy()
    selected_X = X_by_set[selected_set]
    selected_params = tuning_best[selected_model]

    # Missingness ablation: same selected family and fixed seed-42 folds.
    one_seed_splits = [s for s in splits if s["seed"] == 42]
    ablation_preds: dict[str, pd.DataFrame] = {}
    ablation_rows: list[dict[str, Any]] = []
    for label in ["M0", "M1", "M2", "M3", "M4"]:
        X_ab, _, _ = ablation_frame(dev_clean, label)
        pred, folds, _ = run_cv(selected_model, label, X_ab, y, one_seed_splits, selected_params, "none", logger, collect_importance=False)
        fold_record = folds.copy()
        fold_record["model_family"], fold_record["feature_set"], fold_record["strategy"] = selected_model, label, "none"
        all_fold_metric_frames.append(fold_record)
        ablation_preds[label] = pred
        row = {"representation": label, "model_family": selected_model, "feature_set": label, "n_predictions": len(pooled_predictions(pred))}
        row.update(summarize_fold_metrics(folds))
        row.update({f"pooled_{k}": v for k, v in pooled_metrics(pred).items()})
        ablation_rows.append(row)
    missingness_table = pd.DataFrame(ablation_rows)
    missingness_table.to_csv(TABLE / "missingness_ablation_results.csv", index=False, encoding="utf-8-sig")
    missingness_delta, missingness_status = write_missingness_report(missingness_table, selected_model)

    # Imbalance sensitivity on Feature Set C, seed-42 only; no candidate rows.
    X_imb = X_by_set["C"]
    imbalance_rows: list[dict[str, Any]] = []
    imbalance_preds: dict[str, pd.DataFrame] = {}
    for strategy in ["none", "class_weight", "undersample", "oversample"]:
        pred, folds, _ = run_cv(selected_model, "C", X_imb, y, one_seed_splits, selected_params, strategy, logger, collect_importance=False)
        fold_record = folds.copy()
        fold_record["model_family"], fold_record["feature_set"], fold_record["strategy"] = selected_model, "C", strategy
        all_fold_metric_frames.append(fold_record)
        imbalance_preds[strategy] = pred
        row = {"strategy": strategy, "model_family": selected_model, "feature_set": "C", "n_predictions": len(pooled_predictions(pred))}
        row.update(summarize_fold_metrics(folds)); row.update({f"pooled_{k}": v for k, v in pooled_metrics(pred).items()}); imbalance_rows.append(row)
    imbalance_table = pd.DataFrame(imbalance_rows)
    imbalance_table.to_csv(TABLE / "imbalance_sensitivity_results.csv", index=False, encoding="utf-8-sig")
    imbalance_status = write_imbalance_report(imbalance_table, selected_model)

    # Complaint sensitivity compares C and D using identical seed-42 folds.
    c_pred_for_complaint = selected_pred_seed42 if selected_set == "C" else None
    if c_pred_for_complaint is None:
        c_pred_for_complaint, _, _ = run_cv(selected_model, "C", X_by_set["C"], y, one_seed_splits, selected_params, "none", logger, collect_importance=False)
    d_pred, _, d_imp = run_cv(selected_model, "D", X_by_set["D"], y, one_seed_splits, selected_params, "none", logger, collect_importance=False)
    complaint_table, complaint_report = complaint_sensitivity_report(dev_clean, c_pred_for_complaint, d_pred, selected_model)
    complaint_status = "POSSIBLE SHORTCUT" if "POSSIBLE SHORTCUT" in complaint_report else "HELPFUL" if "**HELPFUL**" in complaint_report else "NEUTRAL"
    write_text(REPORT / "14_complaint_sensitivity.md", complaint_report)

    # Calibration on the selected candidate using seed-42 outer folds and nested fold-safe sigmoid calibration.
    raw_cal_pred = selected_pred_seed42 if selected_set == selected["feature_set"] else selected_pred_seed42
    calibrated_pred = run_calibrated_cv(selected_model, selected_set, selected_X, y, selected_params, one_seed_splits, logger)
    calibration_table = write_calibration_table(raw_cal_pred, calibrated_pred)
    selected_calibration = write_calibration_report(calibration_table, selected_model, selected_set)

    alert_table = alert_budget_table(selected_pred_seed42)
    subgroup_table = subgroup_metrics(dev_clean, selected_pred_all)
    subgroup_table.to_csv(TABLE / "subgroup_performance.csv", index=False, encoding="utf-8-sig")
    write_robustness_report(subgroup_table)
    period_table = period_robustness(dev_clean, selected_pred_all)
    period_table.to_csv(TABLE / "period_robustness_results.csv", index=False, encoding="utf-8-sig")
    write_period_report(period_table)
    error_table, error_report = error_analysis(dev_clean, selected_pred_all)
    write_text(REPORT / "11_error_analysis.md", error_report)

    save_plots(primary_predictions, (selected_model, selected_set), raw_cal_pred, calibrated_pred, missingness_table, importance_all, alert_table, period_table)

    # Fit and persist only development-data pipelines; no locked rows are supplied.
    if selected_set in {"A", "B", "C", "D"}:
        primary_X_full = input_frame(dev_clean, selected_set)
        primary_pipeline = make_pipeline(selected_model, selected_set, params=selected_params, strategy="none", y=y)
    else:
        _, cols, add_missing = ablation_frame(dev_clean, selected_set)
        primary_X_full = ablation_frame(dev_clean, selected_set)[0]
        primary_pipeline = make_pipeline_for_columns(selected_model, cols, params=selected_params, strategy="none", y=y, add_missingness=add_missing)
    if selected_calibration == "sigmoid":
        primary_pipeline_to_save = CalibratedClassifierCV(estimator=primary_pipeline, method="sigmoid", cv=3, n_jobs=1)
    else:
        primary_pipeline_to_save = primary_pipeline
    primary_pipeline_to_save.fit(primary_X_full, y)
    joblib.dump(primary_pipeline_to_save, MODEL / "selected_primary_pipeline.joblib", compress=3)
    generate_shap_outputs(primary_pipeline, primary_X_full, selected_model, selected_set, logger)
    secondary_model, secondary_set = secondary["model_family"], secondary["feature_set"]
    secondary_X_full = input_frame(dev_clean, secondary_set)
    secondary_pipeline = make_pipeline(secondary_model, secondary_set, params=tuning_best[secondary_model], strategy="none", y=y)
    secondary_pipeline.fit(secondary_X_full, y)
    joblib.dump(secondary_pipeline, MODEL / "selected_secondary_pipeline.joblib", compress=3)
    write_json(MODEL / "selected_model_metadata.json", {"primary_model": selected_model, "primary_feature_set": selected_set, "primary_calibration": selected_calibration, "secondary_model": secondary_model, "secondary_feature_set": secondary_set, "fit_rows": len(dev_clean), "locked_period_supplied": False, "threshold_locked": False, "parameters": tuning_best})

    # Machine-readable internal artifacts contain development rows only.
    all_pred_frames = list(primary_predictions.values()) + list(ablation_preds.values()) + list(imbalance_preds.values()) + [d_pred, calibrated_pred]
    all_predictions = pd.concat(all_pred_frames, ignore_index=True)
    assert LOCKED_YEAR not in dev_clean["Year"].unique()
    assert len(all_predictions) > 0
    all_predictions.to_parquet(ARTIFACT / "predictions_internal.parquet", index=False)
    fold_results = pd.concat(all_fold_metric_frames, ignore_index=True) if all_fold_metric_frames else pd.DataFrame()
    fold_results.to_parquet(ARTIFACT / "fold_results.parquet", index=False)
    summary.to_csv(ARTIFACT / "model_summary.csv", index=False, encoding="utf-8-sig")

    # Derive honest terminal quality classification from development evidence.
    b_rows = summary[summary.feature_set == "B"].sort_values("mean_average_precision", ascending=False)
    c_rows = summary[summary.feature_set == "C"].sort_values("mean_average_precision", ascending=False)
    a_rows = summary[summary.feature_set == "A"].sort_values("mean_average_precision", ascending=False)
    b_vs_a = float(b_rows.iloc[0].mean_average_precision - a_rows.iloc[0].mean_average_precision)
    c_vs_b = float(c_rows.iloc[0].mean_average_precision - b_rows.iloc[0].mean_average_precision)
    period_range = float(period_table.pr_auc.max() - period_table.pr_auc.min()) if not period_table.empty else 0.0
    top5 = alert_table.loc[alert_table.budget_pct == 5].iloc[0]
    if b_vs_a > 0.005 and c_vs_b > 0.002 and float(top5.mean_ppv) > float(dev_clean.Label.mean()) * 2 and period_range < 0.05:
        q1_signal, cbm, aim, prompt4 = "STRONG Q1 SIGNAL", "STRONG", "CONDITIONAL", "RECOMMENDED"
    elif float(selected["pooled_average_precision"]) > float(dev_clean.Label.mean()) and (b_vs_a > 0 or c_vs_b > 0):
        q1_signal, cbm, aim, prompt4 = "PROMISING BUT INCOMPLETE", "MODERATE", "CONDITIONAL", "RECOMMENDED"
    else:
        q1_signal, cbm, aim, prompt4 = "WEAK", "WEAK", "WEAK", "NOT RECOMMENDED"
    governance_path = ROOT / "docs" / "STUDY_GOVERNANCE.md"
    governance_text = governance_path.read_text(encoding="utf-8") if governance_path.exists() else ""
    deviation_count = sum(1 for line in governance_text.splitlines() if line.startswith("## "))
    write_summary_json(clean, selected, summary[summary.strategy == "none"].sort_values("mean_average_precision", ascending=False).iloc[0], summary[summary.strategy == "none"].sort_values("mean_brier").iloc[0], selected_calibration, missingness_delta, missingness_status, imbalance_table, complaint_status, subgroup_table, period_table, alert_table, q1_signal, cbm, aim, prompt4, deviation_count)
    build_master_report(stats, clean, summary, tuning_all, selected, secondary, calibration_table, missingness_table, imbalance_table, alert_table, subgroup_table, period_table, complaint_table, error_table, selected_calibration, missingness_status, imbalance_status, complaint_status, q1_signal, cbm, aim, prompt4)

    manifest = {
        "prompt": "Prompt 3",
        "protocol_version": "STUDY_PROTOCOL_v2 / prompt2b-v2",
        "execution_timestamp_utc": start.isoformat(),
        "completion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data": {"path": str(RAW.relative_to(ROOT)), "sha256": raw_hash, "full_n": stats["full_n"], "full_positive_n": stats["full_positive_n"], "locked_candidate_period": LOCKED_YEAR, "locked_candidate_touched": False},
        "package_versions": package_versions(),
        "random_seeds": SEEDS,
        "hardware": {"platform": platform.platform(), "python": sys.version.replace("\n", " "), "cpu_count": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2)},
        "git_commit": git_commit_if_applicable(),
        "model_families": MODEL_FAMILIES,
        "feature_sets": {k: v for k, v in FEATURE_SETS.items()},
        "validation_definitions": {"evaluation_unit": "encounter-level retrospective", "development_years": ["1402", "1403", "1404"], "excluded_locked_candidate": LOCKED_YEAR, "split": "stratified 3-fold repeated over seeds 42 and 2024", "temporal_validation": "PARTIAL / COARSE PERIOD VALIDATION ONLY", "patient_independence": "UNSUPPORTED"},
        "tuning": {"scope": "Feature Set B", "folds": 2, "max_configurations_per_model": 3, "metric": "Average Precision"},
        "artifacts": {"predictions": "artifacts/prompt3/predictions_internal.parquet", "fold_results": "artifacts/prompt3/fold_results.parquet", "primary_pipeline": "models/prompt3/selected_primary_pipeline.joblib"},
    }
    write_json(REPORT / "PROMPT3_RUN_MANIFEST.json", manifest)
    write_text(LOG_DIR / "prompt3_completion.txt", f"Prompt 3 completed at {datetime.now(timezone.utc).isoformat()} with locked candidate touched=false.\n")
    logger.info("Prompt 3 complete; selected=%s/%s q1=%s", selected_model, selected_set, q1_signal)


if __name__ == "__main__":
    if "--finalize-existing" in sys.argv:
        finalize_existing()
    else:
        main()
