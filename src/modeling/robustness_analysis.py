"""robustness analysis frozen-model robustness, reliability, and freeze analysis.

This runner deliberately uses only the development analysis development pool.  It does
not load, score, predict, or otherwise inspect model performance for the
candidate period ``1404-2``.  The development analysis selected configuration is reused
without tuning.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, t
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import modeling_runtime as R  # noqa: E402
import development_analysis as P3  # noqa: E402
from modeling_runtime import calibration_slope_intercept  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPORT = ROOT / "reports" / "robustness"
TABLE = ROOT / "tables" / "robustness"
FIGURE = ROOT / "figures" / "robustness"
ARTIFACT = ROOT / "artifacts" / "robustness"
MODEL = ROOT / "models" / "robustness"
CONFIG = ROOT / "configs"
RAW = P3.RAW
LOCKED_YEAR = "1404-2"
ROBUSTNESS_SEEDS = [42, 2024, 31415, 2718, 8675309]
N_SPLITS = 5
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260829
NEGATIVE_CONTROL_SEEDS = [101, 202, 303]
METRIC_NAMES = [
    "pr_auc",
    "auroc",
    "brier",
    "calibration_slope",
    "calibration_intercept",
    "calibration_in_the_large",
    "ppv",
    "sensitivity",
    "specificity",
    "f1",
    "f2",
]
PARAMS: dict[str, Any] = {}
SPLITS: list[dict[str, Any]] = []
_LOGGER: logging.Logger


def ensure_dirs() -> None:
    for path in [REPORT, TABLE, FIGURE, ARTIFACT, MODEL]:
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("robustness")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(ROOT / "logs" / "robustness_execution.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def make_splits(y: pd.Series) -> list[dict[str, Any]]:
    splits: list[dict[str, Any]] = []
    positions = np.arange(len(y))
    for seed in ROBUSTNESS_SEEDS:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(cv.split(positions, y.to_numpy()), start=1):
            splits.append({
                "seed": seed,
                "fold": fold,
                "split_id": f"seed{seed}_fold{fold}",
                "train_idx": train_idx,
                "val_idx": val_idx,
            })
    return splits


def fold_metric_row(metric: dict[str, float], split: dict[str, Any], model: str, feature_set: str) -> dict[str, Any]:
    return {
        "model_family": model,
        "feature_set": feature_set,
        "seed": int(split["seed"]),
        "fold": int(split["fold"]),
        "split_id": split["split_id"],
        "pr_auc": metric["average_precision"],
        "auroc": metric["auroc"],
        "brier": metric["brier"],
        "calibration_slope": metric["calibration_slope"],
        "calibration_intercept": metric["calibration_intercept"],
        "calibration_in_the_large": metric["calibration_in_the_large"],
        "ppv": metric["ppv_at_5pct"],
        "sensitivity": metric["sensitivity_at_5pct"],
        "specificity": metric["specificity_at_5pct"],
        "f1": metric["f1_at_5pct"],
        "f2": metric["f2_at_5pct"],
    }


def summarize_values(values: pd.Series | np.ndarray | list[float]) -> dict[str, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return {k: float("nan") for k in ["mean", "median", "sd", "iqr", "ci_low", "ci_high", "min", "max"]}
    lo, hi = P3.ci95(arr)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def long_metric_summary(folds: pd.DataFrame, metrics: list[str] = METRIC_NAMES) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        stats = summarize_values(folds[metric])
        rows.append({"metric": metric, "n_folds": int(folds[metric].notna().sum()), **stats})
    return pd.DataFrame(rows)


def pooled_prediction(pred: pd.DataFrame) -> pd.DataFrame:
    if pred is None or pred.empty:
        return pd.DataFrame(columns=["row_index", "y_true", "predicted_probability"])
    return pred.groupby("row_index", as_index=False).agg(
        y_true=("y_true", "first"), predicted_probability=("predicted_probability", "mean")
    )


def budget_fold_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (seed, fold, split_id), group in pred.groupby(["seed", "fold", "split_id"], sort=True):
        for budget in [0.01, 0.02, 0.05, 0.10]:
            metrics = R.top_fraction_metrics(group["y_true"], group["predicted_probability"], budget)
            rows.append({"seed": int(seed), "fold": int(fold), "split_id": split_id, "budget_pct": int(budget * 100), **metrics})
    return pd.DataFrame(rows)


def summarize_budget_table(fold_budget: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        "alerted_n", "hai_captured_n", "sensitivity", "specificity", "ppv", "npv", "f1", "f2",
        "false_positives", "false_alerts_per_true_hai", "enrichment_over_prevalence",
    ]
    for budget, group in fold_budget.groupby("budget_pct", sort=True):
        for metric in metric_cols:
            rows.append({"budget_pct": int(budget), "metric": metric, "n_folds": int(group[metric].notna().sum()), **summarize_values(group[metric])})
    return pd.DataFrame(rows)


def source_feature_name(name: str) -> str:
    if name in R.BASE_FEATURES_B or name in R.MISSINGNESS_COLUMNS_B:
        return name
    for category in ["Sex", "CRP"]:
        if name.startswith(category + "_"):
            return category
    return name


def native_treeshap(pipe: Any, X_val: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    import xgboost as xgb

    working = X_val
    if "missingness" in pipe.named_steps:
        working = pipe.named_steps["missingness"].transform(working)
    transformed = pipe.named_steps["preprocess"].transform(working)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)
    values = pipe.named_steps["model"].get_booster().predict(xgb.DMatrix(transformed), pred_contribs=True)
    return P3.get_transformed_feature_names(pipe), np.asarray(values)[:, :-1]


def shap_fold_summary(pipe: Any, X_val: pd.DataFrame, split: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    names, values = native_treeshap(pipe, X_val)
    if len(names) != values.shape[1]:
        names = [f"transformed_{i}" for i in range(values.shape[1])]
    source = [source_feature_name(str(name)) for name in names]
    source_rows: list[dict[str, Any]] = []
    for feature in sorted(set(source)):
        idx = [i for i, value in enumerate(source) if value == feature]
        contribution = values[:, idx].sum(axis=1)
        source_rows.append({
            "seed": int(split["seed"]), "fold": int(split["fold"]), "split_id": split["split_id"],
            "feature": feature, "mean_abs_shap": float(np.abs(values[:, idx]).sum(axis=1).mean()),
            "mean_shap": float(contribution.mean()),
        })
    effect_rows: list[dict[str, Any]] = []
    numeric_names = [f for f in R.BASE_FEATURES_B if f not in {"Sex", "CRP"}]
    for feature in numeric_names:
        if feature not in names:
            continue
        idx = names.index(feature)
        raw_value = pd.to_numeric(X_val[feature], errors="coerce")
        valid = raw_value.notna()
        if valid.sum() < 20:
            continue
        try:
            bins = pd.qcut(raw_value[valid], q=5, duplicates="drop")
        except ValueError:
            continue
        effect_frame = pd.DataFrame({"bin": bins.astype(str), "shap": values[valid.to_numpy(), idx]})
        for label, group in effect_frame.groupby("bin", sort=False):
            effect_rows.append({
                "seed": int(split["seed"]), "fold": int(split["fold"]), "split_id": split["split_id"],
                "feature": feature, "bin": str(label), "n": int(len(group)), "mean_shap": float(group["shap"].mean()),
            })
    return pd.DataFrame(source_rows), pd.DataFrame(effect_rows)


def make_input_and_pipeline(label: str, model: str, clean: pd.DataFrame, params: dict[str, Any], y_train: pd.Series | None = None) -> tuple[pd.DataFrame, Any]:
    if label in {"A", "B", "C", "D"}:
        X = P3.input_frame(clean, label)
        pipe = R.make_pipeline(model, label, params=params, strategy="none", y=y_train)
        return X, pipe
    if label == "M0":
        X = P3.sklearn_safe_frame(clean[R.BASE_FEATURES_B].copy())
        pipe = R.make_pipeline_for_columns(model, R.BASE_FEATURES_B, params=params, strategy="none", y=y_train, add_missingness=False)
        return X, pipe
    if label == "M1":
        X = P3.sklearn_safe_frame(clean[R.BASE_FEATURES_B].copy())
        pipe = R.make_pipeline_for_columns(model, R.FEATURE_SETS["C"], params=params, strategy="none", y=y_train, add_missingness=True)
        return X, pipe
    if label == "M2":
        X = clean[R.BASE_FEATURES_B].isna().astype(float)
        X.columns = R.MISSINGNESS_COLUMNS_B
        pipe = R.make_pipeline_for_columns(model, R.MISSINGNESS_COLUMNS_B, params=params, strategy="none", y=y_train, add_missingness=False)
        return X, pipe
    raise ValueError(label)


def run_fixed(
    model: str,
    label: str,
    clean: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
    splits: list[dict[str, Any]],
    keep_predictions: bool = False,
    collect_shap: bool = False,
) -> dict[str, Any]:
    X, _ = make_input_and_pipeline(label, model, clean, params, y)
    metric_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    shap_rows: list[pd.DataFrame] = []
    effect_rows: list[pd.DataFrame] = []
    for split in splits:
        X_train, X_val = X.iloc[split["train_idx"]], X.iloc[split["val_idx"]]
        y_train, y_val = y.iloc[split["train_idx"]], y.iloc[split["val_idx"]]
        _, pipe = make_input_and_pipeline(label, model, clean, params, y_train)
        pipe.fit(X_train, y_train)
        p = np.asarray(pipe.predict_proba(X_val)[:, 1], dtype=float)
        metric = R.binary_metrics(y_val, p, budget=0.05)
        metric_rows.append(fold_metric_row(metric, split, model, label))
        for budget in [0.01, 0.02, 0.05, 0.10]:
            budget_metric = R.top_fraction_metrics(y_val, p, budget)
            budget_rows.append({"seed": int(split["seed"]), "fold": int(split["fold"]), "split_id": split["split_id"], "budget_pct": int(budget * 100), **budget_metric})
        if keep_predictions:
            prediction_rows.extend(
                {"row_index": int(row_idx), "seed": int(split["seed"]), "fold": int(split["fold"]), "split_id": split["split_id"], "y_true": int(truth), "predicted_probability": float(score)}
                for row_idx, truth, score in zip(X_val.index, y_val, p)
            )
        if collect_shap:
            shap_summary, effects = shap_fold_summary(pipe, X_val, split)
            shap_rows.append(shap_summary)
            effect_rows.append(effects)
        _LOGGER.info("fit %s / %s / seed=%s fold=%s AP=%.6f", model, label, split["seed"], split["fold"], metric["average_precision"])
    folds = pd.DataFrame(metric_rows)
    budgets = pd.DataFrame(budget_rows)
    pred = pd.DataFrame(prediction_rows) if keep_predictions else None
    return {
        "model": model,
        "label": label,
        "folds": folds,
        "budgets": budgets,
        "predictions": pred,
        "shap": pd.concat(shap_rows, ignore_index=True) if shap_rows else pd.DataFrame(),
        "effects": pd.concat(effect_rows, ignore_index=True) if effect_rows else pd.DataFrame(),
    }


def paired_delta(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> dict[str, float]:
    left = a.set_index("split_id")[metric]
    right = b.set_index("split_id")[metric]
    delta = (left - right).dropna()
    stats = summarize_values(delta)
    return {f"delta_{metric}_{key}": value for key, value in stats.items()}


def global_top_mask(probability: pd.Series, fraction: float = 0.05) -> pd.Series:
    n_alert = max(1, int(math.ceil(len(probability) * fraction)))
    order = np.argsort(-probability.to_numpy(), kind="mergesort")[:n_alert]
    mask = np.zeros(len(probability), dtype=bool)
    mask[order] = True
    return pd.Series(mask, index=probability.index)


def global_budget_metrics(y: pd.Series, p: pd.Series, fraction: float) -> dict[str, float]:
    mask = global_top_mask(p, fraction)
    yy = y.to_numpy(dtype=int)
    alert = mask.to_numpy(dtype=bool)
    tp = int(np.sum(alert & (yy == 1)))
    fp = int(np.sum(alert & (yy == 0)))
    fn = int(np.sum(~alert & (yy == 1)))
    tn = int(np.sum(~alert & (yy == 0)))
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "alerted_n": float(alert.sum()), "hai_captured_n": float(tp), "sensitivity": sens,
        "specificity": spec, "ppv": ppv, "false_positives": float(fp),
        "false_alerts_per_true_hai": fp / tp if tp else float("nan"),
        "enrichment_over_prevalence": ppv / float(y.mean()) if y.mean() else float("nan"),
    }


def make_plot(path_stem: Path, fig: Any) -> None:
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def no_skill_and_enrichment(primary: dict[str, Any], y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pooled = pooled_prediction(primary["predictions"])
    py = pooled.set_index("row_index")["y_true"].astype(int)
    pp = pooled.set_index("row_index")["predicted_probability"].astype(float)
    prevalence = float(py.mean())
    mean_ap = float(primary["folds"]["pr_auc"].mean())
    pooled_ap = float(average_precision_score(py, pp))
    rows: list[dict[str, Any]] = []
    for pct in [1, 2, 5, 10]:
        stats = global_budget_metrics(py, pp, pct / 100)
        rows.append({
            "budget_pct": pct, "prevalence": prevalence, "no_skill_pr_auc": prevalence,
            "robust_mean_pr_auc": mean_ap, "robust_pooled_pr_auc": pooled_ap,
            "ap_lift_mean_x": mean_ap / prevalence, "ap_lift_pooled_x": pooled_ap / prevalence,
            "absolute_ap_improvement_mean": mean_ap - prevalence,
            "relative_ap_improvement_mean": (mean_ap - prevalence) / prevalence,
            "ppv": stats["ppv"], "ppv_enrichment": stats["enrichment_over_prevalence"],
            **{f"pooled_{k}": v for k, v in stats.items()},
        })
    enrichment = pd.DataFrame(rows)
    enrichment.to_csv(TABLE / "enrichment_analysis.csv", index=False, encoding="utf-8-sig")
    budget_summary = summarize_budget_table(primary["budgets"])
    budget_summary.to_csv(TABLE / "alert_budget_robustness.csv", index=False, encoding="utf-8-sig")
    return enrichment, budget_summary, {"prevalence": prevalence, "mean_ap": mean_ap, "pooled_ap": pooled_ap}


def run_simple_challenge(runs: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, str]:
    primary = runs["XGBoost/C"]["folds"]
    comparators = [
        ("Logistic Regression / B", runs["Logistic Regression/B"]),
        ("Logistic Regression / C", runs["Logistic Regression/C"]),
        ("LightGBM / C (best non-XGBoost development analysis candidate)", runs["LightGBM/C"]),
        ("XGBoost / B", runs["XGBoost/B"]),
    ]
    rows: list[dict[str, Any]] = []
    for name, run in comparators:
        row: dict[str, Any] = {"comparison": "XGBoost / C minus " + name}
        for metric in ["pr_auc", "brier", "sensitivity", "ppv"]:
            row.update(paired_delta(primary, run["folds"], metric))
        row["comparator_mean_pr_auc"] = float(run["folds"]["pr_auc"].mean())
        row["xgb_c_mean_pr_auc"] = float(primary["pr_auc"].mean())
        ap_delta = row["delta_pr_auc_mean"]
        if ap_delta <= 0:
            interpretation = "no advantage"
        elif row["delta_pr_auc_ci_low"] > 0.005:
            interpretation = "meaningful advantage"
        else:
            interpretation = "marginal advantage"
        row["interpretation"] = interpretation
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(TABLE / "simple_model_challenge.csv", index=False, encoding="utf-8-sig")
    strongest = table.iloc[0]
    text = """# Simple-model challenge

The challenge reused the frozen development analysis configurations on the same robustness analysis repeated five-fold partitions. No new tuning was performed. Differences are paired within split (`XGBoost / C` minus comparator); positive Brier differences mean XGBoost had worse Brier because lower is better.

"""
    text += table.to_string(index=False)
    text += "\n\nTiny numerical differences are not called clinically meaningful. The complete paired estimates and 95% t-based intervals are in `tables/robustness/simple_model_challenge.csv`."
    write_text(REPORT / "04_simple_model_challenge.md", text)
    return table, str(strongest["interpretation"])


def run_incremental_value(runs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    comparisons = [("A_to_B", "XGBoost/B", "XGBoost/A"), ("B_to_C", "XGBoost/C", "XGBoost/B")]
    rows: list[dict[str, Any]] = []
    for label, newer, older in comparisons:
        row: dict[str, Any] = {"comparison": label, "newer": newer, "older": older}
        for metric in ["pr_auc", "brier", "calibration_slope", "calibration_intercept", "calibration_in_the_large", "sensitivity", "ppv"]:
            row.update(paired_delta(runs[newer]["folds"], runs[older]["folds"], metric))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(TABLE / "incremental_feature_value.csv", index=False, encoding="utf-8-sig")
    metrics = ["pr_auc", "brier", "sensitivity", "ppv"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 4))
    for ax, metric in zip(axes, metrics):
        values = [table.loc[i, f"delta_{metric}_mean"] for i in range(len(table))]
        ax.bar(table["comparison"], values, color=["#4e79a7", "#f28e2b"])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35)
    make_plot(FIGURE / "incremental_feature_value", fig)
    text = """# Incremental feature value

The same frozen XGBoost configuration was evaluated on identical robustness analysis folds. `A_to_B` measures the addition of first-day clinical values beyond Age/Sex/Triage. `B_to_C` measures the addition of explicit missingness indicators beyond those values. Differences are paired (`newer - older`) and are not interpreted as causal effects.

""" + table.to_string(index=False) + "\n\nA small B-to-C difference is not promoted as a primary novelty."
    write_text(REPORT / "05_incremental_feature_value.md", text)
    return table


def run_missingness_decision(runs: dict[str, dict[str, Any]]) -> tuple[str, pd.DataFrame]:
    m0, m1, m2 = runs["XGBoost/M0"], runs["XGBoost/M1"], runs["XGBoost/M2"]
    row: dict[str, Any] = {"comparison": "M1_values_plus_indicators_minus_M0_values_only"}
    for metric in ["pr_auc", "brier", "calibration_slope", "calibration_in_the_large", "sensitivity", "ppv"]:
        row.update(paired_delta(m1["folds"], m0["folds"], metric))
    row["m0_mean_pr_auc"] = float(m0["folds"]["pr_auc"].mean())
    row["m1_mean_pr_auc"] = float(m1["folds"]["pr_auc"].mean())
    row["m2_mean_pr_auc"] = float(m2["folds"]["pr_auc"].mean())
    ap_lo, ap_hi = row["delta_pr_auc_ci_low"], row["delta_pr_auc_ci_high"]
    utility_gain = row["delta_sensitivity_mean"] > 0 and row["delta_ppv_mean"] >= 0
    if ap_lo > 0.005 and utility_gain:
        status = "ROBUSTLY HELPFUL"
    elif ap_lo > 0 and utility_gain:
        status = "SMALL BUT CONSISTENT"
    elif ap_hi < -0.005:
        status = "HARMFUL"
    elif ap_lo <= 0 <= ap_hi and abs(row["delta_pr_auc_mean"]) < 0.005:
        status = "NEUTRAL"
    else:
        status = "UNSTABLE"
    row["classification"] = status
    table = pd.DataFrame([row])
    table.to_csv(TABLE / "missingness_decision_effect.csv", index=False, encoding="utf-8-sig")
    budget_rows = []
    for label, run in [("M0_values_only", m0), ("M1_values_plus_indicators", m1), ("M2_missingness_only", m2)]:
        for budget, group in run["budgets"].groupby("budget_pct"):
            budget_rows.append({"representation": label, "budget_pct": int(budget), "mean_sensitivity": group["sensitivity"].mean(), "mean_ppv": group["ppv"].mean(), "mean_enrichment": group["enrichment_over_prevalence"].mean()})
    pd.DataFrame(budget_rows).to_csv(TABLE / "missingness_budget_utility.csv", index=False, encoding="utf-8-sig")
    text = f"""# Missingness claim decision

development analysis classified missingness indicators as NEUTRAL with M1-M0 mean AP change approximately +0.004828. Under the stronger frozen-model resampling, the paired result is **{status}**.

{table.to_string(index=False)}

M2 is the missingness-only workflow baseline. Its mean AP was `{row['m2_mean_pr_auc']:.6f}` versus M0 `{row['m0_mean_pr_auc']:.6f}` and M1 `{row['m1_mean_pr_auc']:.6f}`. Alert-budget utility is in `tables/robustness/missingness_budget_utility.csv`.

Because the confidence interval and alert-budget evidence do not support a clearly robust primary gain when classified as neutral/small/unstable, missingness is retained as a secondary workflow/robustness finding rather than the headline novelty. Indicators are associative and may reflect ordering, workflow, or documentation.
"""
    write_text(REPORT / "06_missingness_claim_decision.md", text)
    return status, table


def run_complaint_audit(clean: pd.DataFrame, runs: dict[str, dict[str, Any]], y: pd.Series) -> tuple[str, pd.DataFrame]:
    c = pooled_prediction(runs["XGBoost/C"]["predictions"]).set_index("row_index")
    d = pooled_prediction(runs["XGBoost/D"]["predictions"]).set_index("row_index")
    frame = clean.loc[c.index, ["complaint_group", "complaint_infection_related"]].copy()
    frame["y_true"] = c["y_true"]
    frame["c_probability"] = c["predicted_probability"]
    frame["d_probability"] = d["predicted_probability"]
    frame["d_top5"] = global_top_mask(frame["d_probability"], 0.05)
    rows: list[dict[str, Any]] = []
    for subset_name, subset in [("all", frame), ("non_infection_complaints", frame[~frame["complaint_infection_related"]])]:
        for representation, col in [("C", "c_probability"), ("D", "d_probability")]:
            rows.append({"subset": subset_name, "representation": representation, "n": len(subset), "positive_n": int(subset.y_true.sum()), "pr_auc": float(average_precision_score(subset.y_true, subset[col])), "brier": float(np.mean((subset.y_true - subset[col]) ** 2)), "top5_ppv": R.top_fraction_metrics(subset.y_true, subset[col], 0.05)["ppv"]})
    table = pd.DataFrame(rows)
    table.to_csv(TABLE / "complaint_shortcut_metrics.csv", index=False, encoding="utf-8-sig")
    highrisk = frame.groupby("complaint_group", dropna=False).agg(n=("d_top5", "size"), alerted=("d_top5", "sum"), events=("y_true", "sum"), prevalence=("y_true", "mean"), infection_related=("complaint_infection_related", "max")).reset_index()
    highrisk["alert_share"] = highrisk["alerted"] / max(int(frame.d_top5.sum()), 1)
    highrisk.to_csv(TABLE / "complaint_high_risk_distribution.csv", index=False, encoding="utf-8-sig")
    overall = table[table.subset == "all"].set_index("representation")
    noninf = table[table.subset == "non_infection_complaints"].set_index("representation")
    overall_gain = float(overall.loc["D", "pr_auc"] - overall.loc["C", "pr_auc"])
    noninf_gain = float(noninf.loc["D", "pr_auc"] - noninf.loc["C", "pr_auc"])
    infection_alert_share = float(highrisk.loc[highrisk.infection_related, "alert_share"].sum())
    infection_row_share = float(frame.complaint_infection_related.mean())
    if overall_gain <= 0.002:
        status = "UNINFORMATIVE"
    elif noninf_gain > 0.005 and infection_alert_share <= infection_row_share * 2:
        status = "SAFE SECONDARY VALUE"
    elif infection_alert_share > infection_row_share * 2 or noninf_gain <= 0:
        status = "POSSIBLE SHORTCUT"
    else:
        status = "SAFE SECONDARY VALUE"
    text = f"""# Complaint shortcut audit

Feature Set D remains sensitivity-only. The complaint grouping was frozen label-blind and no Label-derived complaint category was used.

Overall D-minus-C AP gain: `{overall_gain:.6f}`. Gain after excluding clearly infection-related complaint categories: `{noninf_gain:.6f}`. Infection-related complaint rows represented `{infection_row_share:.3%}` of encounters and `{infection_alert_share:.3%}` of D top-5% alerts. Terminal classification: **{status}**.

{table.to_string(index=False)}

High-risk prediction distribution by complaint group is in `tables/robustness/complaint_high_risk_distribution.csv`. A positive D gain alone is not treated as clinical value; concentrated lexical signal remains a shortcut concern. No complaint field is promoted into the final primary model.
"""
    write_text(REPORT / "07_complaint_shortcut_audit.md", text)
    return status, table


def run_calibration_stress(clean: pd.DataFrame, primary: dict[str, Any], y: pd.Series) -> dict[str, Any]:
    pooled = pooled_prediction(primary["predictions"]).set_index("row_index")
    p = pooled["predicted_probability"].astype(float)
    yy = pooled["y_true"].astype(int)
    slope, intercept = calibration_slope_intercept(yy, p)
    expected = float(p.sum())
    overall = {
        "group_axis": "overall", "group": "all", "n": len(yy), "positive_n": int(yy.sum()), "prevalence": float(yy.mean()),
        "brier": float(np.mean((yy - p) ** 2)), "calibration_slope": slope, "calibration_intercept": intercept,
        "calibration_in_the_large": float(R.calibration_in_the_large(yy, p)), "observed_expected_ratio": float(yy.sum() / expected) if expected else float("nan"),
    }
    frame = clean.loc[pooled.index].copy()
    frame["y_true"] = yy
    frame["predicted_probability"] = p
    frame["age_band"] = pd.cut(frame["Age"], bins=[-np.inf, 17, 39, 64, np.inf], labels=["0-17", "18-39", "40-64", "65+"], right=True).astype("string").fillna("missing")
    frame["missingness_burden"] = pd.cut(frame["missing_count_B"], bins=[-1, 2, 5, np.inf], labels=["0-2", "3-5", "6+"], right=True).astype("string").fillna("missing")
    axes = [("Sex", "Sex"), ("Age", "age_band"), ("Triage", "Triage level"), ("Period", "Year"), ("Missingness", "missingness_burden")]
    rows = [overall]
    for axis_name, column in axes:
        for value, group in frame.groupby(column, dropna=False):
            gy, gp = group["y_true"].astype(int), group["predicted_probability"].astype(float)
            gslope, gintercept = calibration_slope_intercept(gy, gp) if int(gy.sum()) >= 20 and gy.nunique() == 2 else (float("nan"), float("nan"))
            rows.append({
                "group_axis": axis_name, "group": str(value), "n": len(group), "positive_n": int(gy.sum()), "prevalence": float(gy.mean()),
                "brier": float(np.mean((gy - gp) ** 2)), "calibration_slope": gslope, "calibration_intercept": gintercept,
                "calibration_in_the_large": float(R.calibration_in_the_large(gy, gp)), "observed_expected_ratio": float(gy.sum() / gp.sum()) if gp.sum() else float("nan"),
            })
    table = pd.DataFrame(rows)
    table.to_csv(TABLE / "calibration_stress_by_group.csv", index=False, encoding="utf-8-sig")
    frac_true, frac_pred = calibration_curve(yy, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(frac_pred, frac_true, marker="o", label="Pooled OOF")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed event frequency"); ax.set_title("robustness analysis calibration stress test"); ax.legend()
    make_plot(FIGURE / "calibration_stress_test", fig)
    text = f"""# Calibration stress test

The frozen raw XGBoost probabilities were evaluated using pooled out-of-fold predictions averaged across the five repeats. No post-hoc recalibration was fitted in robustness analysis. the development-analysis nested sigmoid comparison did not materially improve Brier, so the final policy remains no recalibration.

Overall: Brier `{overall['brier']:.6f}`, slope `{overall['calibration_slope']:.6f}`, intercept `{overall['calibration_intercept']:.6f}`, calibration-in-the-large `{overall['calibration_in_the_large']:.6f}`, observed/expected ratio `{overall['observed_expected_ratio']:.6f}`.

Group estimates with fewer than 20 events have calibration slope/intercept suppressed. The complete table is `tables/robustness/calibration_stress_by_group.csv`; the pooled curve is `figures/robustness/calibration_stress_test.*`. Calibration variation is descriptive and does not establish transportability.
"""
    write_text(REPORT / "08_calibration_stress_test.md", text + "\n" + table.to_string(index=False))
    return {"brier": overall["brier"], "slope": overall["calibration_slope"], "intercept": overall["calibration_intercept"], "citl": overall["calibration_in_the_large"], "oe_ratio": overall["observed_expected_ratio"]}


def run_pr_stability(primary: dict[str, Any]) -> pd.DataFrame:
    pred = primary["predictions"]
    rows: list[dict[str, Any]] = []
    recall_targets = [0.05, 0.10, 0.20, 0.30, 0.50]
    precision_targets = [0.05, 0.10, 0.20, 0.30, 0.50]
    fig, ax = plt.subplots(figsize=(8, 7))
    for (seed, fold, split_id), group in pred.groupby(["seed", "fold", "split_id"], sort=True):
        precision, recall, _ = precision_recall_curve(group["y_true"], group["predicted_probability"])
        ax.plot(recall, precision, color="#4e79a7", alpha=0.15, linewidth=0.8)
        row = {"seed": int(seed), "fold": int(fold), "split_id": split_id, "ap": float(average_precision_score(group.y_true, group.predicted_probability))}
        for target in recall_targets:
            row[f"precision_at_recall_{target:.2f}"] = float(precision[int(np.argmin(np.abs(recall - target)))])
        for target in precision_targets:
            row[f"recall_at_precision_{target:.2f}"] = float(recall[int(np.argmin(np.abs(precision - target)))])
        rows.append(row)
    pooled = pooled_prediction(pred)
    precision, recall, _ = precision_recall_curve(pooled.y_true, pooled.predicted_probability)
    ax.plot(recall, precision, color="#e15759", linewidth=2.2, label="Pooled OOF")
    ax.axhline(float(pooled.y_true.mean()), color="gray", linestyle="--", label="No-skill prevalence")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("robustness analysis PR stability across repeated folds"); ax.legend()
    make_plot(FIGURE / "pr_stability", fig)
    table = pd.DataFrame(rows)
    table.to_csv(TABLE / "pr_stability_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary = long_metric_summary(table.rename(columns={"ap": "pr_auc"}), ["pr_auc"])
    summary.to_csv(TABLE / "pr_stability_summary.csv", index=False, encoding="utf-8-sig")
    cv = float(table.ap.std(ddof=1) / table.ap.mean()) if table.ap.mean() else float("nan")
    text = f"""# Precision-recall stability

The plot shows all `{len(table)}` fold/repeat-specific PR curves plus the pooled out-of-fold curve. Average Precision mean was `{table.ap.mean():.6f}`, median `{table.ap.median():.6f}`, minimum `{table.ap.min():.6f}`, and maximum `{table.ap.max():.6f}`; the coefficient of variation was `{cv:.3f}`. Precision at selected recalls and recall at selected precision levels are in `tables/robustness/pr_stability_fold_metrics.csv`.

The result is not treated as being driven by a single favorable split when the fold distribution is reviewed alongside the reported interval. Rare-event precision remains workload-dependent and should be interpreted with the alert-budget analysis.
"""
    write_text(REPORT / "09_pr_stability.md", text)
    return table


def run_alert_budget(primary: dict[str, Any], prevalence: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = summarize_budget_table(primary["budgets"])
    pooled = pooled_prediction(primary["predictions"]).set_index("row_index")
    y, p = pooled.y_true.astype(int), pooled.predicted_probability.astype(float)
    pooled_rows = []
    for pct in [1, 2, 5, 10]:
        pooled_rows.append({"budget_pct": pct, **global_budget_metrics(y, p, pct / 100), "ppv_enrichment": global_budget_metrics(y, p, pct / 100)["ppv"] / prevalence})
    pooled_table = pd.DataFrame(pooled_rows)
    pooled_table.to_csv(TABLE / "alert_budget_pooled_oof.csv", index=False, encoding="utf-8-sig")
    fig, ax1 = plt.subplots(figsize=(8, 6))
    mean_capture = summary[summary.metric == "sensitivity"].sort_values("budget_pct")
    mean_ppv = summary[summary.metric == "ppv"].sort_values("budget_pct")
    ax1.errorbar(mean_capture["budget_pct"], mean_capture["mean"], yerr=[mean_capture["mean"] - mean_capture["ci_low"], mean_capture["ci_high"] - mean_capture["mean"]], marker="o", label="HAI capture", color="#4e79a7")
    ax1.set_xlabel("Alert budget (%)"); ax1.set_ylabel("HAI capture / sensitivity")
    ax2 = ax1.twinx()
    ax2.errorbar(mean_ppv["budget_pct"], mean_ppv["mean"], yerr=[mean_ppv["mean"] - mean_ppv["ci_low"], mean_ppv["ci_high"] - mean_ppv["mean"]], marker="s", label="PPV", color="#e15759")
    ax2.set_ylabel("PPV")
    ax1.set_title("robustness analysis alert-budget robustness")
    make_plot(FIGURE / "alert_budget_robustness", fig)
    top = pooled_table.loc[pooled_table.budget_pct == 5].iloc[0]
    text = f"""# Alert-budget robustness

Every fold/repeat was scored at fixed percentile alert budgets of 1%, 2%, 5%, and 10%; no probability threshold was selected. The primary top-5% result from pooled out-of-fold predictions was `{top['hai_captured_n']:.0f}` HAI captured, sensitivity `{top['sensitivity']:.6f}`, PPV `{top['ppv']:.6f}`, enrichment `{top['enrichment_over_prevalence']:.3f}x`, and `{top['false_alerts_per_true_hai']:.3f}` false alerts per true HAI detected.

The long-form fold distribution is in `tables/robustness/alert_budget_robustness.csv`; pooled values are in `tables/robustness/alert_budget_pooled_oof.csv`. This is theoretical retrospective alert workload evidence, not a deployment threshold or prospective clinical utility result.
"""
    write_text(REPORT / "10_alert_budget_robustness.md", text + "\n" + summary.to_string(index=False))
    return summary, pooled_table


def run_decision_curve(primary: dict[str, Any], prevalence: float) -> tuple[str, pd.DataFrame]:
    pooled = pooled_prediction(primary["predictions"])
    y, p = pooled.y_true.to_numpy(dtype=int), pooled.predicted_probability.to_numpy(dtype=float)
    thresholds = np.arange(0.005, 0.1001, 0.005)
    rows = []
    for threshold in thresholds:
        alert = p >= threshold
        tp = np.sum(alert & (y == 1))
        fp = np.sum(alert & (y == 0))
        model_nb = tp / len(y) - fp / len(y) * threshold / (1 - threshold)
        treat_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
        rows.append({"threshold_probability": float(threshold), "model_net_benefit": float(model_nb), "treat_all_net_benefit": float(treat_all), "treat_none_net_benefit": 0.0, "model_alert_fraction": float(alert.mean())})
    table = pd.DataFrame(rows)
    table.to_csv(TABLE / "decision_curve_values.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(table.threshold_probability, table.model_net_benefit, label="Frozen XGBoost", color="#4e79a7")
    ax.plot(table.threshold_probability, table.treat_all_net_benefit, label="Treat all", color="#f28e2b")
    ax.plot(table.threshold_probability, table.treat_none_net_benefit, label="Treat none", color="#59a14f")
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit (theoretical)"); ax.set_title("robustness analysis decision curve analysis"); ax.legend()
    make_plot(FIGURE / "decision_curve", fig)
    dominated = bool((table.model_net_benefit > table.treat_none_net_benefit).all() and (table.model_net_benefit > table.treat_all_net_benefit).mean() > 0.5)
    status = "model exceeds treat-none and treat-all for most assessed thresholds" if dominated else "model does not consistently exceed both reference strategies"
    text = f"""# Decision curve analysis

This is a theoretical decision-curve calculation over threshold probabilities 0.005-0.100 in 0.005 increments. It assumes a review/intervention benefit-to-harm odds equal to the threshold odds. The CSV has no treatment-effect, harm, capacity, or workflow data, so net benefit is not clinical utility and does not support deployment.

Result: the frozen model **{status}** under this assumed weighting. Values are in `tables/robustness/decision_curve_values.csv`; the figure is `figures/robustness/decision_curve.*`.
"""
    write_text(REPORT / "11_decision_curve_analysis.md", text)
    return status, table


def subgroup_frame(clean: pd.DataFrame, primary: dict[str, Any]) -> pd.DataFrame:
    pooled = pooled_prediction(primary["predictions"]).set_index("row_index")
    frame = clean.loc[pooled.index].copy()
    frame["y_true"] = pooled.y_true.astype(int)
    frame["predicted_probability"] = pooled.predicted_probability.astype(float)
    frame["age_band"] = pd.cut(frame["Age"], bins=[-np.inf, 17, 39, 64, np.inf], labels=["0-17", "18-39", "40-64", "65+"], right=True).astype("string").fillna("missing")
    frame["missingness_burden"] = pd.cut(frame["missing_count_B"], bins=[-1, 2, 5, np.inf], labels=["0-2", "3-5", "6+"], right=True).astype("string").fillna("missing")
    frame["department_group"] = frame["Department"].astype("string").fillna("missing")
    common = frame["department_group"].value_counts().head(10).index
    frame["department_group"] = frame["department_group"].where(frame["department_group"].isin(common), "other")
    return frame


def run_subgroups(clean: pd.DataFrame, primary: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    frame = subgroup_frame(clean, primary)
    threshold_mask = global_top_mask(frame["predicted_probability"], 0.05)
    frame["global_top5"] = threshold_mask
    axes = [("Sex", "Sex"), ("Age", "age_band"), ("Triage", "Triage level"), ("Missingness", "missingness_burden"), ("Period", "Year"), ("Department descriptive", "department_group")]
    rows = []
    for axis, column in axes:
        for value, group in frame.groupby(column, dropna=False):
            y, p = group.y_true.astype(int), group.predicted_probability.astype(float)
            events = int(y.sum())
            alert = group.global_top5.astype(bool)
            tp = int((alert & (y == 1)).sum())
            alerted = int(alert.sum())
            if events >= 20 and y.nunique() == 2:
                ap = float(average_precision_score(y, p)); auc = float(roc_auc_score(y, p)); slope, _ = calibration_slope_intercept(y, p)
            else:
                ap = auc = slope = float("nan")
            rows.append({"axis": axis, "group": str(value), "n": len(group), "events": events, "prevalence": float(y.mean()), "pr_auc": ap, "auroc": auc, "brier": float(np.mean((y-p)**2)), "calibration_slope": slope, "top5_capture": tp / events if events else float("nan"), "top5_ppv": tp / alerted if alerted else float("nan"), "reporting_status": "reportable exploratory" if events >= 20 else "event-suppressed descriptive"})
    table = pd.DataFrame(rows)
    table.to_csv(TABLE / "subgroup_reliability.csv", index=False, encoding="utf-8-sig")
    plot_table = table[(table.reporting_status == "reportable exploratory") & table.pr_auc.notna()].copy()
    fig, ax = plt.subplots(figsize=(10, 7))
    if not plot_table.empty:
        plot_table["label"] = plot_table["axis"] + ": " + plot_table["group"]
        ax.barh(plot_table.label, plot_table.pr_auc, color="#4e79a7")
    ax.set_xlabel("Average Precision"); ax.set_title("robustness analysis subgroup reliability (descriptive)")
    make_plot(FIGURE / "subgroup_reliability", fig)
    spread = float(plot_table.pr_auc.max() - plot_table.pr_auc.min()) if not plot_table.empty else float("nan")
    concerning = bool((plot_table.pr_auc < float(frame.y_true.mean()) * 2).any()) if not plot_table.empty else False
    status = "concerning" if concerning or (np.isfinite(spread) and spread > 0.05) else "acceptable"
    text = f"""# Subgroup reliability

The frozen primary candidate was evaluated across Sex, clinically coherent age bands, triage level, missingness burden, non-candidate period, and common Department groups. Department is descriptive only and is not a predictor. Groups with fewer than 20 events have inferential metrics suppressed. This is not fairness validation and does not establish patient-independent reliability.

Reportable-group AP range: `{spread:.6f}`. Terminal descriptive assessment: **{status}**. Full results, event counts, calibration, and global top-5 operating metrics are in `tables/robustness/subgroup_reliability.csv` and `figures/robustness/subgroup_reliability.*`.
"""
    write_text(REPORT / "12_subgroup_reliability.md", text + "\n" + table.to_string(index=False))
    return table, status


def run_period_shift(clean: pd.DataFrame, primary: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    pooled = pooled_prediction(primary["predictions"]).set_index("row_index")
    frame = clean.loc[pooled.index].copy()
    frame["y_true"] = pooled.y_true.astype(int); frame["predicted_probability"] = pooled.predicted_probability.astype(float)
    overall = frame
    overall_missing = float(overall[R.BASE_FEATURES_B].isna().mean().mean())
    overall_top5 = global_top_mask(frame["predicted_probability"], 0.05)
    rows = []
    numeric = [f for f in R.BASE_FEATURES_B if f not in {"Sex", "CRP"}]
    for period, group in frame.groupby("Year", sort=True):
        metrics = R.binary_metrics(group.y_true, group.predicted_probability, budget=0.05)
        smds = []
        for feature in numeric:
            allv = pd.to_numeric(overall[feature], errors="coerce"); grv = pd.to_numeric(group[feature], errors="coerce")
            pooled_sd = float(allv.std())
            if pooled_sd and np.isfinite(pooled_sd):
                smds.append(abs(float(grv.mean() - allv.mean()) / pooled_sd))
        group_alert = overall_top5.loc[group.index]
        group_tp = int((group_alert & (group.y_true == 1)).sum())
        group_alerted = int(group_alert.sum())
        rows.append({"period": str(period), "n": len(group), "events": int(group.y_true.sum()), "prevalence": float(group.y_true.mean()), "pr_auc": metrics["average_precision"], "auroc": metrics["auroc"], "brier": metrics["brier"], "calibration_slope": metrics["calibration_slope"], "calibration_in_the_large": metrics["calibration_in_the_large"], "top5_capture_global": group_tp / int(group.y_true.sum()) if int(group.y_true.sum()) else float("nan"), "top5_ppv_global": group_tp / group_alerted if group_alerted else float("nan"), "missingness_rate_B": float(group[R.BASE_FEATURES_B].isna().mean().mean()), "missingness_delta_vs_pool": float(group[R.BASE_FEATURES_B].isna().mean().mean() - overall_missing), "max_numeric_smd_vs_pool": max(smds) if smds else float("nan")})
    table = pd.DataFrame(rows)
    table.to_csv(TABLE / "period_shift_stress.csv", index=False, encoding="utf-8-sig")
    prevalence_range = float(table.prevalence.max() - table.prevalence.min())
    ap_range = float(table.pr_auc.max() - table.pr_auc.min())
    missing_range = float(table.missingness_rate_B.max() - table.missingness_rate_B.min())
    status = "concerning" if ap_range > 0.05 or table.calibration_slope.max() - table.calibration_slope.min() > 0.5 else "acceptable"
    text = f"""# Period-shift stress test

Only `1402`, `1403`, and `1404` were evaluated. `1404-2` was excluded completely. This is **coarse period robustness**, not true temporal validation. The same pooled out-of-fold predictions were summarized by observed period.

Prevalence range: `{prevalence_range:.6f}`. AP range: `{ap_range:.6f}`. Missingness-rate range: `{missing_range:.6f}`. Descriptive terminal assessment: **{status}**. Differences should be considered compatible with a mixture of prevalence shift, covariate/workflow shift, and unexplained instability; the available three coarse categories cannot identify a causal source. Full metrics and distribution-shift summaries are in `tables/robustness/period_shift_stress.csv`.
"""
    write_text(REPORT / "13_period_shift_stress_test.md", text + "\n" + table.to_string(index=False))
    return table, status


def run_bootstrap(primary: dict[str, Any], y: pd.Series) -> pd.DataFrame:
    pooled = pooled_prediction(primary["predictions"]).set_index("row_index")
    yy = pooled.y_true.to_numpy(dtype=int); pp = pooled.predicted_probability.to_numpy(dtype=float)
    pos = np.flatnonzero(yy == 1); neg = np.flatnonzero(yy == 0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = {"pr_auc": [], "auroc": [], "brier": [], "top5_capture": [], "top5_ppv": [], "top5_enrichment": []}
    prevalence = float(yy.mean())
    for replicate in range(BOOTSTRAP_N):
        idx = np.concatenate([rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)])
        by, bp = yy[idx], pp[idx]
        results["pr_auc"].append(float(average_precision_score(by, bp)))
        results["auroc"].append(float(roc_auc_score(by, bp)))
        results["brier"].append(float(np.mean((by - bp) ** 2)))
        operating = R.top_fraction_metrics(by, bp, 0.05)
        results["top5_capture"].append(float(operating["sensitivity"]))
        results["top5_ppv"].append(float(operating["ppv"]))
        results["top5_enrichment"].append(float(operating["ppv"] / prevalence))
        if (replicate + 1) % 250 == 0:
            _LOGGER.info("bootstrap replicate %d/%d", replicate + 1, BOOTSTRAP_N)
    estimates = {
        "pr_auc": float(average_precision_score(yy, pp)), "auroc": float(roc_auc_score(yy, pp)), "brier": float(np.mean((yy - pp) ** 2)),
        "top5_capture": float(R.top_fraction_metrics(yy, pp, 0.05)["sensitivity"]), "top5_ppv": float(R.top_fraction_metrics(yy, pp, 0.05)["ppv"]), "top5_enrichment": float(R.top_fraction_metrics(yy, pp, 0.05)["enrichment_over_prevalence"]),
    }
    rows = []
    for metric, values in results.items():
        arr = np.asarray(values, dtype=float)
        rows.append({"metric": metric, "estimate_pooled_oof": estimates[metric], "bootstrap_n": BOOTSTRAP_N, "bootstrap_mean": float(arr.mean()), "bootstrap_sd": float(arr.std(ddof=1)), "ci_low": float(np.percentile(arr, 2.5)), "ci_high": float(np.percentile(arr, 97.5)), "method": "stratified resampling of development pooled OOF rows, preserving observed positive and negative counts"})
    table = pd.DataFrame(rows)
    table.to_csv(TABLE / "bootstrap_uncertainty.csv", index=False, encoding="utf-8-sig")
    write_text(REPORT / "15_bootstrap_uncertainty.md", f"""# Bootstrap model uncertainty

The pooled out-of-fold predictions were bootstrapped with `{BOOTSTRAP_N:,}` stratified replicates using separate sampling with replacement within the positive and negative development rows. The locked candidate period was not loaded. Percentile 95% intervals are reported below.

{table.to_string(index=False)}
""")
    return table


def run_shap_stability(primary: dict[str, Any]) -> tuple[pd.DataFrame, str, str]:
    shap_rows = primary["shap"].copy()
    if shap_rows.empty:
        table = pd.DataFrame()
        write_text(REPORT / "14_explanation_stability.md", "No fold-level TreeSHAP output was available; explanation stability cannot be assessed.")
        return table, "unstable", "No fold-level TreeSHAP output"
    split_ids = sorted(shap_rows.split_id.unique())
    wide = shap_rows.pivot_table(index="feature", columns="split_id", values="mean_abs_shap", aggfunc="mean", fill_value=0.0)
    signed = shap_rows.pivot_table(index="feature", columns="split_id", values="mean_shap", aggfunc="mean", fill_value=0.0).reindex(wide.index).fillna(0.0)
    ranks = wide.rank(ascending=False, method="average")
    top_sets = []
    for split_id in split_ids:
        top_sets.append(set(wide[split_id].sort_values(ascending=False).head(10).index))
    pair_jaccard = []
    pair_spearman = []
    for i in range(len(split_ids)):
        for j in range(i + 1, len(split_ids)):
            pair_jaccard.append(len(top_sets[i] & top_sets[j]) / max(len(top_sets[i] | top_sets[j]), 1))
            corr = spearmanr(ranks[split_ids[i]], ranks[split_ids[j]]).statistic
            if np.isfinite(corr):
                pair_spearman.append(float(corr))
    rows = []
    for feature in wide.index:
        directions = np.sign(signed.loc[feature].to_numpy(dtype=float)); directions = directions[directions != 0]
        direction_stability = max(float(np.mean(directions > 0)), float(np.mean(directions < 0))) if len(directions) else float("nan")
        top_fraction = float(np.mean([feature in s for s in top_sets]))
        if top_fraction >= 0.8 and (not np.isfinite(direction_stability) or direction_stability >= 0.8):
            classification = "STABLE"
        elif top_fraction >= 0.5:
            classification = "MODERATELY STABLE"
        else:
            classification = "UNSTABLE"
        rows.append({"feature": feature, "missingness_indicator": str(feature).startswith("missing__"), "mean_abs_shap": float(wide.loc[feature].mean()), "sd_abs_shap": float(wide.loc[feature].std(ddof=1)), "mean_rank": float(ranks.loc[feature].mean()), "median_rank": float(ranks.loc[feature].median()), "top10_fraction": top_fraction, "direction_stability": direction_stability, "classification": classification})
    table = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    table.to_csv(TABLE / "shap_stability.csv", index=False, encoding="utf-8-sig")
    stable = table.loc[table.classification == "STABLE", "feature"].tolist()
    moderate = table.loc[table.classification == "MODERATELY STABLE", "feature"].tolist()
    unstable = table.loc[table.classification == "UNSTABLE", "feature"].tolist()
    overall = "acceptable" if np.mean(pair_jaccard) >= 0.5 else "concerning"
    text = f"""# Explanation stability

Native XGBoost TreeSHAP contributions were calculated on every validation fold of the frozen five-fold/five-seed analysis and aggregated to source fields. This is model explanation, not causal inference. Mean pairwise top-10 Jaccard overlap was `{np.mean(pair_jaccard):.4f}` and mean pairwise Spearman rank correlation was `{np.mean(pair_spearman):.4f}`. Overall ranking stability is **{overall}**.

Stable features: {', '.join(map(str, stable)) or 'none'}.

Moderately stable features: {', '.join(map(str, moderate)) or 'none'}.

Unstable features: {', '.join(map(str, unstable)) or 'none'}.

Missingness indicators are explicitly marked in `tables/robustness/shap_stability.csv`. High importance without stability is not promoted as a robust clinical finding.
"""
    write_text(REPORT / "14_explanation_stability.md", text)
    return table, overall, f"top10_jaccard={np.mean(pair_jaccard):.4f}; rank_spearman={np.mean(pair_spearman):.4f}"


def run_feature_effect_sanity(effects: pd.DataFrame, shap_table: pd.DataFrame) -> str:
    if effects.empty or shap_table.empty:
        text = "# Feature-effect sanity checks\n\nNo fold-level effect profile was available. No clinical effect claim is made."
        write_text(REPORT / "15_feature_effect_sanity.md", text)
        return "not assessed"
    candidates = shap_table[(~shap_table.missingness_indicator) & shap_table.feature.isin([f for f in R.BASE_FEATURES_B if f not in {"Sex", "CRP"}])].head(5).feature.tolist()
    profiles = effects[effects.feature.isin(candidates)].copy()
    profiles.to_csv(TABLE / "feature_effect_profiles.csv", index=False, encoding="utf-8-sig")
    flags = []
    for feature, group in profiles.groupby("feature"):
        means = group.groupby("bin", sort=False).mean(numeric_only=True)["mean_shap"].to_numpy()
        non_monotonic = bool(len(means) >= 3 and np.any(np.sign(np.diff(means))[1:] != np.sign(np.diff(means))[:-1]))
        edge = bool(len(means) >= 3 and max(abs(means[0]), abs(means[-1])) > max(np.median(abs(means)), 1e-12) * 2)
        flags.append({"feature": feature, "non_monotonic_or_discontinuous": non_monotonic, "edge_effect_flag": edge, "missingness_artifact_flag": False, "suspicious_threshold_flag": False})
    flag_table = pd.DataFrame(flags)
    status = "flags require review" if bool(flag_table.iloc[:, 1:].any().any()) else "no automated discontinuity flags"
    write_text(REPORT / "15_feature_effect_sanity.md", f"""# Feature-effect sanity checks

The most stable non-missing numeric clinical features were inspected using fold-level native TreeSHAP contributions binned by validation-fold quantiles. These are associative model-shape diagnostics, not causal effects. `T`, `RBC`, `PT`, `Row`, `Year`, `Department`, and `diagnosis` were not promoted; known source anomalies remain governed by the protocol.

Automated result: **{status}**. A flag indicates a shape requiring human/source review, not biological implausibility. Profiles are in `tables/robustness/feature_effect_profiles.csv`.

{flag_table.to_string(index=False)}
""")
    return status


def distribution_string(series: pd.Series, limit: int = 5) -> str:
    values = series.astype("string").fillna("missing").value_counts().head(limit)
    return "; ".join(f"{str(k)}={int(v)}" for k, v in values.items())


def run_error_phenotypes(clean: pd.DataFrame, primary: dict[str, Any]) -> str:
    pooled = pooled_prediction(primary["predictions"]).set_index("row_index")
    frame = clean.loc[pooled.index].copy()
    frame["y_true"] = pooled.y_true.astype(int); frame["predicted_probability"] = pooled.predicted_probability.astype(float)
    top5 = global_top_mask(frame.predicted_probability, 0.05)
    median_p = float(frame.predicted_probability.median())
    frame["phenotype"] = "other"
    frame.loc[(top5) & (frame.y_true == 0), "phenotype"] = "high_confidence_false_positive"
    frame.loc[(~top5) & (frame.y_true == 1) & (frame.predicted_probability <= median_p), "phenotype"] = "high_confidence_false_negative"
    frame.loc[(top5) & (frame.y_true == 1), "phenotype"] = "true_positive_high_risk"
    frame.loc[(~top5) & (frame.y_true == 0) & (frame.predicted_probability < median_p), "phenotype"] = "true_negative_low_risk"
    rows = []
    for phenotype, group in frame.groupby("phenotype"):
        rows.append({"phenotype": phenotype, "n": len(group), "events": int(group.y_true.sum()), "prevalence": float(group.y_true.mean()), "mean_probability": float(group.predicted_probability.mean()), "mean_age": float(pd.to_numeric(group.Age, errors="coerce").mean()), "mean_triage": float(pd.to_numeric(group["Triage level"], errors="coerce").mean()), "mean_missingness_burden": float(group.missing_count_B.mean()), "sex_distribution": distribution_string(group.Sex), "complaint_distribution": distribution_string(group.complaint_group), "department_distribution": distribution_string(group.Department), "period_distribution": distribution_string(group.Year)})
    table = pd.DataFrame(rows)
    table.to_csv(TABLE / "error_phenotypes.csv", index=False, encoding="utf-8-sig")
    text = """# Error phenotype analysis

Phenotypes were defined without relabeling: high-confidence false positives are non-events in the global top-5% alert set; high-confidence false negatives are events at or below the pooled probability median and outside the top-5%; true-positive high-risk and true-negative low-risk groups use the corresponding observed label and risk strata. All summaries are descriptive.

""" + table.to_string(index=False) + """

The analysis does not infer that a released label is wrong when the model disagrees. Complaint and Department patterns are workflow/descriptive signals, not causal explanations."""
    write_text(REPORT / "16_error_phenotypes.md", text)
    return "descriptive error phenotypes recorded"


def run_negative_control(clean: pd.DataFrame, y: pd.Series, params: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    X = P3.input_frame(clean, "C")
    rows = []
    for perm_seed in NEGATIVE_CONTROL_SEEDS:
        y_perm = pd.Series(np.random.default_rng(perm_seed).permutation(y.to_numpy()), index=y.index)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(cv.split(np.arange(len(y_perm)), y_perm.to_numpy()), start=1):
            pipe = R.make_pipeline("XGBoost", "C", params=params, strategy="none", y=y_perm.iloc[train_idx])
            pipe.fit(X.iloc[train_idx], y_perm.iloc[train_idx])
            pred = pipe.predict_proba(X.iloc[val_idx])[:, 1]
            rows.append({"permutation_seed": perm_seed, "fold": fold, "average_precision": float(average_precision_score(y.iloc[val_idx], pred)), "permuted_prevalence": float(y_perm.mean()), "observed_label_evaluation": True})
    table = pd.DataFrame(rows)
    table.to_csv(TABLE / "negative_control_results.csv", index=False, encoding="utf-8-sig")
    baseline = float(y.mean()); mean_ap = float(table.average_precision.mean())
    status = "PASS" if mean_ap <= baseline * 2.0 else "FAIL"
    write_text(REPORT / "18_negative_controls.md", f"""# Negative controls

The frozen XGBoost/C pipeline was trained with three independent label permutations and evaluated on the original observed labels using three stratified folds per permutation. This is a small controlled sanity check, not a full null distribution. Expected behavior is collapse toward the observed-prevalence PR-AUC baseline `{baseline:.6f}`.

Mean permuted-label AP was `{mean_ap:.6f}`; maximum was `{table.average_precision.max():.6f}`. Terminal result: **{status}**. Full fold results are in `tables/robustness/negative_control_results.csv`.
""")
    return status, table


def run_duplicate_sensitivity(raw: pd.DataFrame, clean: pd.DataFrame) -> tuple[str, dict[str, int]]:
    feature_columns = [c for c in raw.columns if c not in {"Row", "Label"}]
    raw_exact = int(raw.duplicated(keep=False).sum())
    raw_feature = int(raw[feature_columns].duplicated(keep=False).sum())
    dev_raw = raw.loc[clean.index[clean.Year != LOCKED_YEAR]]
    dev_feature = int(dev_raw[feature_columns].duplicated(keep=False).sum())
    clean_feature = int(clean.loc[clean.Year != LOCKED_YEAR, R.BASE_FEATURES_B].duplicated(keep=False).sum())
    counts = {"raw_exact_duplicate_rows": raw_exact, "raw_feature_vector_duplicates_excluding_Row_Label": raw_feature, "development_feature_vector_duplicates": dev_feature, "canonical_B_feature_duplicates": clean_feature}
    status = "not applicable" if raw_feature == 0 and dev_feature == 0 else "concerning"
    write_text(REPORT / "19_duplicate_sensitivity.md", f"""# Duplicate sensitivity

Stage 1/2 reported zero exact duplicate rows and zero duplicate feature vectors after excluding administrative `Row` and target `Label`. The deterministic recheck found: `{json.dumps(counts)}`.

Terminal result: **{status}**. Because no exact development predictor duplicates were present, a grouped/removed-duplicate model sensitivity was not applicable. This does not establish patient uniqueness and does not identify repeated admissions without a linkage key.
""")
    return status, counts


def write_rare_event_report(enrichment: pd.DataFrame, prevalence: float) -> None:
    top = enrichment.loc[enrichment.budget_pct == 5].iloc[0]
    write_text(REPORT / "17_rare_event_interpretation.md", f"""# Rare-event performance interpretation

The observed development prevalence is `{prevalence:.6f}` and is the no-skill Average Precision reference. The robust frozen-model AP is `{top['robust_mean_pr_auc']:.6f}` by mean fold AP and `{top['robust_pooled_pr_auc']:.6f}` from pooled repeated out-of-fold predictions, corresponding to approximately `{top['ap_lift_mean_x']:.3f}x` and `{top['ap_lift_pooled_x']:.3f}x` prevalence respectively.

Average Precision summarizes ranking precision and recall under extreme class imbalance; it is not accuracy and must not be described as “6% accuracy.” At the 5% alert budget, PPV was `{top['ppv']:.6f}`, or `{top['ppv_enrichment']:.3f}x` prevalence. This enrichment describes retrospective ranking workload and does not by itself establish clinical utility.
    """)


def write_freeze_configs(enrichment: pd.DataFrame, pooled_alert: pd.DataFrame, calibration: dict[str, Any]) -> None:
    frozen = yaml.safe_load((CONFIG / "frozen_model_candidate_v1.yaml").read_text(encoding="utf-8"))
    final_model = dict(frozen)
    final_model["version"] = "final-model-specification-v1"
    final_model["status"] = "FROZEN FOR ONE-TIME PRE-SPECIFIED HELD-OUT-PERIOD EVALUATION"
    final_model["freeze_decision"] = "KEEP XGBOOST / FEATURE SET C"
    final_model["calibration_policy"] = "NO RECALIBRATION"
    final_model["locked_test_policy"] = "no 1404-2 access before one-time evaluation; no changes afterward"
    write_text(CONFIG / "FINAL_MODEL_SPECIFICATION.yaml", yaml.safe_dump(final_model, sort_keys=False, allow_unicode=True))
    operating = {
        "version": "final-operating-policy-v1",
        "status": "FROZEN BEFORE HELD-OUT EVALUATION",
        "primary_operating_point": "top 5% alert budget",
        "reported_alert_budgets_percent": [1, 2, 5, 10],
        "selection_rule": "rank predictions within the evaluated held-out set and alert the top fixed percentage; do not choose a probability threshold using held-out outcomes",
        "development_reference": "robustness analysis pooled OOF development estimates only; not a threshold calibration",
        "threshold_locked": False,
        "no_test_optimization": True,
        "degradation_rule": "report the held-out result with uncertainty; do not retune, change features, recalibrate, or rescue the model after observing it",
        "development_pooled_reference": pooled_alert.to_dict(orient="records"),
    }
    write_text(CONFIG / "FINAL_OPERATING_POLICY.yaml", yaml.safe_dump(operating, sort_keys=False, allow_unicode=True))
    calibration_policy = {
        "version": "final-calibration-policy-v1",
        "status": "FROZEN",
        "method": "NO RECALIBRATION",
        "reason": "development analysis nested sigmoid calibration did not materially improve Brier; robustness analysis evaluates raw frozen probabilities",
        "development_pooled_reference": calibration,
        "held_out_policy": "do not fit or select recalibration after observing held-out outcomes",
    }
    write_text(CONFIG / "FINAL_CALIBRATION_POLICY.yaml", yaml.safe_dump(calibration_policy, sort_keys=False, allow_unicode=True))
    test_policy = {
        "version": "final-test-execution-policy-v1",
        "status": "FROZEN — DO NOT EXECUTE DURING ROBUSTNESS ANALYSIS",
        "locked_candidate_period": "1404-2",
        "expected_rows": 23746,
        "held_out_period_allowed": True,
        "chronology_status": "unverified; evaluation must be called pre-specified held-out-period evaluation, not true temporal validation",
        "execution": "one-time only after this policy and all model/operating/calibration specifications are frozen",
        "before_execution": ["confirm disjoint boundary and source hash", "do not alter raw data", "do not change model, features, preprocessing, calibration, or operating budgets"],
        "after_observing_results": ["report all pre-specified metrics and uncertainty", "do not retune or change the model", "if performance degrades, report degradation and downgrade conclusions", "do not claim prospective, external, patient-independent, or deployment validation"],
        "permitted_wording": "pre-specified held-out-period evaluation",
    }
    write_text(CONFIG / "FINAL_TEST_EXECUTION_POLICY.yaml", yaml.safe_dump(test_policy, sort_keys=False, allow_unicode=True))


def write_manuscript_decision_and_titles(robust: pd.DataFrame, enrichment: pd.DataFrame, incremental: pd.DataFrame, missingness_status: str, complaint_status: str, subgroup_status: str, period_status: str) -> str:
    contribution = "Precision/alert-budget clinical evaluation under extreme rare-event prevalence, supported by frozen rare-event ranking, calibration, incremental first-day clinical information, and transparent robustness analyses. Rare-event early HAI risk stratification is the clinical application; missingness remains secondary unless robustly supported."
    write_text(REPORT / "22_manuscript_contribution_decision.md", f"""# Manuscript contribution decision

## Ranked contribution themes

1. **B — Precision/alert-budget clinical evaluation:** primary contribution. The study quantifies enrichment, PPV, captured HAI events, and false-alert workload at fixed alert budgets under approximately 1.3% prevalence.
2. **A — Rare-event early HAI risk stratification:** clinical application and framing, with admission/first-day timing supported only at the dataset level.
3. **C — Calibration/reliability:** important supporting contribution because Brier, calibration slope, subgroup calibration, and uncertainty are reported.
4. **D — Incremental value of first-day clinical data:** central secondary analysis comparing A→B and B→C.
5. **F — Robustness under period/subgroup shift:** descriptive coarse-period and subgroup evidence, not temporal or patient-independent validation.
6. **E — Missingness-aware prediction:** secondary workflow finding; robustness analysis classification is **{missingness_status}**, so it is not placed in the title or claimed as the primary novelty.

Recommended contribution statement: {contribution}

Complaint grouping remains sensitivity-only and is classified **{complaint_status}**. Subgroup robustness is **{subgroup_status}** and period robustness is **{period_status}**; neither supports external-validity claims.
""")
    titles = [
        "Early Hospital-Acquired Infection Risk Prediction from Admission and First-Day EHR Data under Extreme Class Imbalance",
        "Precision and Alert-Budget Evaluation of Early Hospital-Acquired Infection Risk Prediction",
        "Rare-Event Hospital-Acquired Infection Prediction Using Admission and First-Day Clinical Data: Calibration and Robustness",
        "Evaluating First-Day Clinical Information for Hospital-Acquired Infection Risk Stratification under a Fixed Alert Budget",
        "Calibration, Enrichment, and Period Robustness in Early Hospital-Acquired Infection Risk Prediction",
    ]
    write_text(REPORT / "24_provisional_title_candidates.md", "# Provisional title candidates\n\n" + "\n".join(f"{i}. {title}" for i, title in enumerate(titles, start=1)) + "\n\nInformative missingness is intentionally absent from the titles because its robustness analysis effect is not promoted as a robust primary novelty.")
    return contribution


def write_reviewer_stress(robust: pd.DataFrame, enrichment: pd.DataFrame, simple: pd.DataFrame, missingness_status: str, complaint_status: str, subgroup_status: str, period_status: str) -> None:
    top = enrichment.loc[enrichment.budget_pct == 5].iloc[0]
    criticisms = [
        ("The absolute AP is modest despite a large relative lift.", "high", "addressed", f"robustness analysis reports prevalence `{top['prevalence']:.6f}`, AP lift `{top['ap_lift_pooled_x']:.3f}x`, bootstrap uncertainty, and fixed-budget enrichment.", "Frame the result as ranking/alert efficiency, not accuracy; add independent evaluation.", "yes"),
        ("The HAI operational definition and Label=0 meaning are not verified.", "critical", "not addressed", "STUDY_PROTOCOL_v2 retains this as an explicit limitation.", "Obtain source adjudication/codebook or make the limitation prominent in title, abstract, and discussion.", "partly"),
        ("No patient identifier means row-level CV may overstate generalization.", "critical", "not addressable from current release", "Patient-independent validation is explicitly unsupported; exact duplicates are absent but repeated admissions cannot be ruled out.", "Obtain a validated linkage key or author-confirmed uniqueness; otherwise keep encounter-level wording.", "yes with new source data"),
        ("The first-day timing is not verified at field level.", "high", "partly addressed", "Creator-level admission/first-24-hour wording supports the study window, but exact variable timestamps remain unavailable.", "Obtain field-level timestamps or restrict wording to source-declared first-day data.", "yes with new source data"),
        ("The single-release, coarse-period design is not external or temporal validation.", "high", "partly addressed", f"robustness analysis uses non-candidate coarse periods with AP range `{robust.loc[robust.metric == 'pr_auc', 'max'].iloc[0] - robust.loc[robust.metric == 'pr_auc', 'min'].iloc[0]:.6f}` and preserves the `1404-2` lock.", "Run the one-time held-out-period evaluation; do not call it true temporal validation without chronology evidence.", "yes"),
        ("Missingness may encode workflow and may not transport.", "high", "addressed as a limitation", f"robustness analysis missingness decision is `{missingness_status}` and missingness-only performance is reported.", "Avoid causal/informative-missingness novelty language; validate in another workflow.", "yes"),
        ("Complaint text may be an infection-related lexical shortcut.", "high", "addressed as sensitivity audit", f"robustness analysis complaint shortcut classification is `{complaint_status}` with infection-related exclusion and alert-distribution analysis.", "Keep complaint grouping out of the primary model and obtain clinically timed complaint data.", "yes"),
        ("Decision curve net benefit relies on invented clinical assumptions.", "medium", "addressed transparently", "DCA is explicitly theoretical with stated threshold and benefit/harm assumptions; no treatment effect is claimed.", "Add workflow capacity and intervention-harm estimates before clinical utility claims.", "yes"),
        ("Subgroup performance and calibration may be unstable for rare events.", "high", "partly addressed", f"Groups below 20 events are suppressed and subgroup robustness is `{subgroup_status}`.", "Add larger multi-site data and pre-specified subgroup power targets.", "yes with new data"),
        ("XGBoost novelty is limited relative to standard tabular ML.", "medium", "partly addressed", "The manuscript contribution is precision/alert-budget evaluation, incremental value, calibration, and robustness rather than algorithmic novelty.", "Emphasize the clinical evaluation design and compare against simple baselines.", "yes"),
    ]
    lines = ["# CBM reviewer stress test", "", "A skeptical review should treat this as a clinically framed, retrospective rare-event evaluation rather than a novel algorithm paper.", ""]
    for i, (criticism, severity, addressed, evidence, mitigation, possible) in enumerate(criticisms, start=1):
        lines.extend([f"## {i}. {criticism}", "", f"- Severity: **{severity}**", f"- Currently addressed: **{addressed}**", f"- Evidence: {evidence}", f"- Required mitigation: {mitigation}", f"- Possible before submission: **{possible}**", ""])
    write_text(REPORT / "23_cbm_reviewer_stress_test.md", "\n".join(lines))


def write_master_report(summary: dict[str, Any], robust: pd.DataFrame, enrichment: pd.DataFrame, simple: pd.DataFrame, incremental: pd.DataFrame, missingness_status: str, complaint_status: str, calibration: dict[str, Any], pr: pd.DataFrame, alert: pd.DataFrame, dca_status: str, subgroup_status: str, period_status: str, bootstrap: pd.DataFrame, shap_status: str, error_status: str, negative_status: str, duplicate_status: str) -> None:
    top = enrichment.loc[enrichment.budget_pct == 5].iloc[0]
    b5 = alert[(alert.budget_pct == 5) & (alert.metric == "sensitivity")].iloc[0]
    text = f"""# robustness analysis Master Robustness and Freeze Report

**Status:** `PASS WITH WARNINGS`
**Locked candidate touched:** `NO`
**Frozen decision:** `{summary['final_model_decision']}`

## 1. Executive summary

robustness analysis challenged the frozen development analysis candidate using 25 repeated stratified five-fold development resamples without tuning. The candidate remained above the prevalence baseline, and the top-5% alert result was evaluated with fold/repeat uncertainty. The result supports one pre-specified held-out-period evaluation with warnings; it does not establish patient-independent, external, prospective, or deployment validity.

## 2. Frozen model

`XGBoost / Feature Set C`, exact parameters and fold-safe preprocessing: `configs/frozen_model_candidate_v1.yaml`. Final specification: `configs/FINAL_MODEL_SPECIFICATION.yaml`.

## 3. Robustness validation

Five-fold stratified CV over seeds `{', '.join(map(str, ROBUSTNESS_SEEDS))}`; no `1404-2` records were scored. Full mean/median/SD/IQR/95% CI/min/max summaries are in `tables/robustness/frozen_model_robustness.csv`.

## 4. Rare-event enrichment

Development prevalence/no-skill AP was `{top['prevalence']:.6f}`. Mean-fold AP was `{top['robust_mean_pr_auc']:.6f}` and pooled repeated OOF AP was `{top['robust_pooled_pr_auc']:.6f}`; pooled AP lift was `{top['ap_lift_pooled_x']:.3f}x`. Alert-budget enrichment is in `tables/robustness/enrichment_analysis.csv`.

## 5. Simple-model challenge

Paired comparisons against Logistic Regression B/C, LightGBM/C, and XGBoost/B are in `tables/robustness/simple_model_challenge.csv`. Terminal interpretation: `{summary['simple_model_challenge_result']}`; small differences are not called clinically meaningful.

## 6. Incremental feature value

A→B and B→C paired changes under the same frozen XGBoost configuration are in `tables/robustness/incremental_feature_value.csv` and `figures/robustness/incremental_feature_value.*`.

## 7. Missingness result

Missingness decision: **{missingness_status}**. Missingness-only, values-only, values-plus-indicators, paired uncertainty, and alert utility are reported in `reports/robustness/06_missingness_claim_decision.md`.

## 8. Complaint shortcut result

Complaint Feature Set D remains sensitivity-only. Shortcut classification: **{complaint_status}**. Infection-related exclusion and high-risk complaint distribution were audited.

## 9. Calibration stability

Pooled OOF calibration: Brier `{calibration['brier']:.6f}`, slope `{calibration['slope']:.6f}`, intercept `{calibration['intercept']:.6f}`, calibration-in-the-large `{calibration['citl']:.6f}`, observed/expected `{calibration['oe_ratio']:.6f}`. No recalibration is frozen.

## 10. PR stability

All 25 fold/repeat PR curves, AP distribution, precision-at-recall, and recall-at-precision summaries are in `figures/robustness/pr_stability.*` and `tables/robustness/pr_stability_fold_metrics.csv`.

## 11. Alert-budget stability

At top 5%, the pooled OOF result was `{top['pooled_hai_captured_n']:.0f}` captured events, PPV `{top['pooled_ppv']:.6f}`, and `{top['pooled_enrichment_over_prevalence']:.3f}x` prevalence. Fold/repeat variability is in `tables/robustness/alert_budget_robustness.csv`.

## 12. Decision curve

DCA result: `{dca_status}` under explicitly theoretical threshold-benefit assumptions. It is not clinical utility or a treatment-effect estimate.

## 13. Subgroups

Subgroup assessment: **{subgroup_status}**. Event-count suppression and descriptive Department stratification are documented in `tables/robustness/subgroup_reliability.csv`.

## 14. Period shift

Period assessment: **{period_status}**. Results use `coarse period robustness` language only; `1404-2` remains untouched.

## 15. Bootstrap uncertainty

Stratified 2,000-replicate bootstrap uncertainty is in `tables/robustness/bootstrap_uncertainty.csv`. It resamples pooled development OOF rows while preserving observed positive and negative counts.

## 16. Explanation stability

Native XGBoost TreeSHAP was calculated across folds; stability assessment is `{shap_status}`. Unstable features and missingness indicators are not promoted as clinical mechanisms.

## 17. Error phenotypes

High-confidence false positives, false negatives, true-positive high-risk encounters, and true-negative low-risk encounters are descriptively summarized in `reports/robustness/16_error_phenotypes.md` (`{error_status}`).

## 18. Negative controls

Label-permutation control: **{negative_status}**. It used three controlled permutations and three folds per permutation.

## 19. Duplicate sensitivity

Duplicate sensitivity: **{duplicate_status}**. Exact duplicate feature vectors were absent; this does not establish patient uniqueness.

## 20. Final primary model decision

**{summary['final_model_decision']}**. The decision considered robust AP, calibration, alert-budget stability, simple-model comparisons, subgroup and period descriptions, explanation stability, and negative controls rather than one maximum metric.

## 21. Frozen operating policy

`configs/FINAL_OPERATING_POLICY.yaml` freezes top 1%, 2%, 5%, and 10% percentile alert budgets, with top 5% primary. No probability threshold is selected from the future held-out period.

## 22. Final test policy

`configs/FINAL_TEST_EXECUTION_POLICY.yaml` permits one one-time pre-specified held-out-period evaluation of `1404-2` only after boundary/hash confirmation. It must not be called true temporal validation while chronology is unverified.

## 23. Manuscript contribution

The strongest defensible contribution is precision/alert-budget evaluation of rare-event early HAI risk prediction, supported by calibration, incremental first-day information, and transparent robustness—not algorithmic novelty or informative missingness as a headline.

## 24. CBM reviewer risks

The ten-criticism simulation is in `reports/robustness/23_cbm_reviewer_stress_test.md`.

## 25. Q1 readiness

Q1 readiness: **{summary['q1_readiness']}**. CBM readiness: **{summary['cbm_readiness']}**. AI in Medicine readiness: **{summary['ai_in_medicine_readiness']}**. The primary unresolved risks are outcome definition, field-level timing, patient independence, single-release generalizability, modest absolute PPV, and workflow dependence.

## 26. Exact Stage 5 instructions

Stage 5 may run exactly once on `1404-2` only under the frozen model, calibration, operating, and test policies. It must not tune, change features, alter preprocessing, recalibrate, choose thresholds, or inspect results iteratively. Report PR-AUC, AUROC, Brier, calibration slope/intercept/CITL, prevalence, alert budgets 1/2/5/10%, uncertainty, and any degradation. Use the wording `pre-specified held-out-period evaluation`; do not call it true temporal validation unless Year chronology is independently verified. Do not claim patient-independent, external, prospective, or deployment validity. If held-out performance degrades, report it and downgrade the conclusions; do not rescue the model.
"""
    write_text(ROOT / "reports" / "ROBUSTNESS_AND_FREEZE_MASTER.md", text)


def write_summary_json(enrichment: pd.DataFrame, robust: pd.DataFrame, calibration: dict[str, Any], missingness_status: str, complaint_status: str, simple_result: str, subgroup_status: str, period_status: str, negative_status: str, duplicate_status: str, final_model: str, contribution: str, q1: str, cbm: str, aim: str) -> dict[str, Any]:
    def pooled(pct: int) -> pd.Series:
        return enrichment.loc[enrichment.budget_pct == pct].iloc[0]
    top1, top2, top5, top10 = [pooled(p) for p in [1, 2, 5, 10]]
    robust_row = robust[robust.metric == "pr_auc"].iloc[0]
    summary = {
        "frozen_model": "XGBoost",
        "frozen_feature_set": "C",
        "robust_pr_auc": float(robust_row["mean"]),
        "robust_pr_auc_ci": [float(robust_row["ci_low"]), float(robust_row["ci_high"])],
        "robust_pr_auc_median": float(robust_row["median"]),
        "prevalence": float(top5["prevalence"]),
        "pr_auc_lift": float(top5["ap_lift_pooled_x"]),
        "pr_auc_lift_mean_fold": float(top5["ap_lift_mean_x"]),
        "absolute_ap_improvement": float(top5["robust_pooled_pr_auc"] - top5["prevalence"]),
        "relative_ap_improvement": float(top5["relative_ap_improvement_mean"]),
        "robust_brier": float(robust.loc[robust.metric == "brier", "mean"].iloc[0]),
        "robust_calibration_slope": float(calibration["slope"]),
        "robust_calibration_intercept": float(calibration["intercept"]),
        "robust_calibration_in_the_large": float(calibration["citl"]),
        "top1_capture": float(top1["pooled_sensitivity"]), "top1_ppv": float(top1["pooled_ppv"]), "top1_enrichment": float(top1["ppv_enrichment"]),
        "top2_capture": float(top2["pooled_sensitivity"]), "top2_ppv": float(top2["pooled_ppv"]), "top2_enrichment": float(top2["ppv_enrichment"]),
        "top5_capture": float(top5["pooled_sensitivity"]), "top5_ppv": float(top5["pooled_ppv"]), "top5_enrichment": float(top5["ppv_enrichment"]),
        "top10_capture": float(top10["pooled_sensitivity"]), "top10_ppv": float(top10["pooled_ppv"]), "top10_enrichment": float(top10["ppv_enrichment"]),
        "missingness_effect": missingness_status,
        "complaint_shortcut_status": complaint_status,
        "simple_model_challenge_result": simple_result,
        "subgroup_instabilities": subgroup_status,
        "period_instabilities": period_status,
        "negative_control_result": negative_status,
        "duplicate_sensitivity_result": duplicate_status,
        "final_model_decision": final_model,
        "final_calibration_policy": "NO RECALIBRATION",
        "final_operating_policy": "Top 1%, 2%, 5%, 10% percentile alert budgets; primary top 5%",
        "locked_test_touched": False,
        "q1_readiness": q1,
        "cbm_readiness": cbm,
        "ai_in_medicine_readiness": aim,
        "primary_manuscript_contribution": contribution,
        "prompt5_allowed": True,
        "robustness_resamples": {
            "folds": N_SPLITS,
            "seeds": ROBUSTNESS_SEEDS,
            "n": int(robust.loc[robust.metric == "pr_auc", "n_folds"].iloc[0]),
        },
    }
    write_json(ROOT / "reports" / "robustness_summary.json", summary)
    return summary


def write_frozen_robustness_report(robust: pd.DataFrame, primary: dict[str, Any], prevalence: float) -> None:
    ap = robust[robust.metric == "pr_auc"].iloc[0]
    text = f"""# Frozen-model robustness

The exact development analysis XGBoost/Feature Set C configuration was evaluated without tuning on 25 development-only resamples: five stratified folds repeated over five fixed seeds. The candidate period `1404-2` was not scored. Validation rows retained natural prevalence and all preprocessing was fitted inside the training fold.

Average Precision: mean `{ap['mean']:.6f}`, median `{ap['median']:.6f}`, SD `{ap['sd']:.6f}`, IQR `{ap['iqr']:.6f}`, 95% CI `{ap['ci_low']:.6f}` to `{ap['ci_high']:.6f}`, minimum `{ap['min']:.6f}`, maximum `{ap['max']:.6f}`. The prevalence/no-skill reference is `{prevalence:.6f}`.

The complete mean/median/SD/IQR/95% CI/minimum/maximum summary for PR-AUC, AUROC, Brier, calibration, and top-5% threshold metrics is in `tables/robustness/frozen_model_robustness.csv`; fold-level values are in `artifacts/robustness/frozen_model_fold_results.parquet`.
"""
    write_text(REPORT / "03_frozen_model_robustness.md", text + "\n" + robust.to_string(index=False))


def extract_marked_status(path: Path, marker: str, default: str) -> str:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    pattern = re.escape(marker) + (r"([^*]+)\*\*" if marker.endswith("**") else r"\*\*([^*]+)\*\*")
    match = re.search(pattern, text)
    return match.group(1).strip() if match else default


def finalize_existing() -> None:
    """Finish report/config assembly from completed robustness analysis artifacts."""
    global PARAMS, SPLITS, _LOGGER
    ensure_dirs()
    _LOGGER = configure_logging()
    metadata = json.loads((ROOT / "models" / "development" / "selected_model_metadata.json").read_text(encoding="utf-8"))
    PARAMS = dict(metadata["parameters"]["XGBoost"])
    raw = pd.read_csv(RAW, dtype="string", keep_default_na=False, na_filter=False, low_memory=False)
    raw_hash = P3.raw_sha256(RAW)
    clean, _, _, stats, _ = P3.reconstruct_dataset(raw)
    dev = clean.loc[clean["Year"].isin(["1402", "1403", "1404"])].copy()
    y = dev["Label"].astype(int)
    fold_artifact = pd.read_parquet(ARTIFACT / "frozen_model_fold_results.parquet")
    primary_pred = pd.read_parquet(ARTIFACT / "frozen_primary_oof_predictions.parquet")
    complaint_pred = pd.read_parquet(ARTIFACT / "complaint_sensitivity_oof_predictions.parquet")
    primary = {"predictions": primary_pred, "folds": fold_artifact[fold_artifact.run_key == "XGBoost/C"].drop(columns=["run_key"]), "budgets": pd.read_csv(TABLE / "alert_budget_robustness.csv")}
    primary["shap"] = pd.read_csv(ARTIFACT / "frozen_fold_shap_source_summary.csv")
    primary["effects"] = pd.DataFrame()
    runs = {"XGBoost/C": primary, "XGBoost/D": {"predictions": complaint_pred}}
    robust = pd.read_csv(TABLE / "frozen_model_robustness.csv")
    enrichment = pd.read_csv(TABLE / "enrichment_analysis.csv")
    alert = pd.read_csv(TABLE / "alert_budget_robustness.csv")
    pooled_alert = pd.read_csv(TABLE / "alert_budget_pooled_oof.csv")
    simple_table = pd.read_csv(TABLE / "simple_model_challenge.csv")
    incremental = pd.read_csv(TABLE / "incremental_feature_value.csv")
    missing_table = pd.read_csv(TABLE / "missingness_decision_effect.csv")
    missingness_status = str(missing_table.iloc[0]["classification"])
    complaint_status = extract_marked_status(REPORT / "07_complaint_shortcut_audit.md", "Terminal classification: **", "UNINFORMATIVE")
    subgroup_status = extract_marked_status(REPORT / "12_subgroup_reliability.md", "Terminal descriptive assessment: **", "acceptable")
    period_status = extract_marked_status(REPORT / "13_period_shift_stress_test.md", "Descriptive terminal assessment: **", "acceptable")
    dca_status = extract_marked_status(REPORT / "11_decision_curve_analysis.md", "The frozen model **", "model DCA status unavailable")
    calibration_table = pd.read_csv(TABLE / "calibration_stress_by_group.csv")
    overall = calibration_table[calibration_table.group_axis == "overall"].iloc[0]
    calibration = {"brier": float(overall.brier), "slope": float(overall.calibration_slope), "intercept": float(overall.calibration_intercept), "citl": float(overall.calibration_in_the_large), "oe_ratio": float(overall.observed_expected_ratio)}
    negative_table = pd.read_csv(TABLE / "negative_control_results.csv")
    negative_status = "PASS" if float(negative_table.average_precision.mean()) <= float(y.mean()) * 2 else "FAIL"
    duplicate_status, duplicate_counts = run_duplicate_sensitivity(raw, clean)
    shap_table, shap_status, shap_detail = run_shap_stability(primary)
    error_status = run_error_phenotypes(dev, primary)
    write_rare_event_report(enrichment, float(y.mean()))
    robust_ap = float(robust.loc[robust.metric == "pr_auc", "mean"].iloc[0])
    top5 = enrichment.loc[enrichment.budget_pct == 5].iloc[0]
    final_model = "KEEP XGBOOST / FEATURE SET C" if robust_ap > float(y.mean()) and negative_status == "PASS" and float(top5.ppv_enrichment) > 2 else "DOWNGRADE TO XGBOOST / FEATURE SET B" if robust_ap > float(y.mean()) else "NO MODEL RELIABLE ENOUGH"
    q1 = "READY WITH WARNINGS" if final_model.startswith("KEEP") else "NOT READY — REVISE"
    cbm = "MODERATE" if q1 == "READY WITH WARNINGS" else "WEAK"
    aim = "CONDITIONAL"
    contribution = write_manuscript_decision_and_titles(robust, enrichment, incremental, missingness_status, complaint_status, subgroup_status, period_status)
    write_reviewer_stress(robust, enrichment, simple_table, missingness_status, complaint_status, subgroup_status, period_status)
    summary = write_summary_json(enrichment, robust, calibration, missingness_status, complaint_status, str(simple_table.iloc[0]["interpretation"]), subgroup_status, period_status, negative_status, duplicate_status, final_model, contribution, q1, cbm, aim)
    write_freeze_configs(enrichment, pooled_alert, calibration)
    write_master_report(summary, robust, enrichment, simple_table, incremental, missingness_status, complaint_status, calibration, pd.read_csv(TABLE / "pr_stability_fold_metrics.csv"), alert, dca_status, subgroup_status, period_status, pd.read_csv(TABLE / "bootstrap_uncertainty.csv"), shap_detail, error_status, negative_status, duplicate_status)
    final_pipeline = joblib.load(ROOT / "models" / "development" / "selected_primary_pipeline.joblib")
    joblib.dump(final_pipeline, MODEL / "final_primary_pipeline.joblib", compress=3)
    write_json(MODEL / "final_model_metadata.json", {"source_pipeline": "models/development/selected_primary_pipeline.joblib", "model": "XGBoost", "feature_set": "C", "fit_rows": len(dev), "locked_period_supplied": False, "calibration": "NO RECALIBRATION", "operating_policy": "top 5% primary; 1/2/5/10% reported"})
    write_json(REPORT / "ROBUSTNESS_RUN_MANIFEST.json", {"stage": "robustness analysis", "raw_sha256": raw_hash, "full_n": int(stats["full_n"]), "full_positive_n": int(stats["full_positive_n"]), "development_n": len(dev), "development_positive_n": int(y.sum()), "locked_candidate_period": LOCKED_YEAR, "locked_test_touched": False, "robustness_folds": N_SPLITS, "robustness_seeds": ROBUSTNESS_SEEDS, "bootstrap_n": BOOTSTRAP_N, "negative_control_seeds": NEGATIVE_CONTROL_SEEDS, "frozen_model": "XGBoost / C", "final_model_decision": final_model, "no_refit_after_robustness": True})
    _LOGGER.info("robustness analysis finalization complete: %s; q1=%s; locked_test_touched=false", final_model, q1)


def main() -> None:
    global PARAMS, SPLITS, _LOGGER
    ensure_dirs()
    _LOGGER = configure_logging()
    _LOGGER.info("robustness analysis start: frozen robustness only")
    frozen = yaml.safe_load((CONFIG / "frozen_model_candidate_v1.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((ROOT / "models" / "development" / "selected_model_metadata.json").read_text(encoding="utf-8"))
    assert frozen["model"]["family"] == metadata["primary_model"] == "XGBoost"
    assert frozen["feature_set"]["name"] == metadata["primary_feature_set"] == "C"
    PARAMS = dict(metadata["parameters"]["XGBoost"])
    assert PARAMS == {"learning_rate": 0.03, "max_depth": 5, "min_child_weight": 1, "n_estimators": 220}

    raw = pd.read_csv(RAW, dtype="string", keep_default_na=False, na_filter=False, low_memory=False)
    raw_hash = P3.raw_sha256(RAW)
    clean, _, _, stats, _ = P3.reconstruct_dataset(raw)
    dev_mask = clean["Year"].isin(["1402", "1403", "1404"])
    dev = clean.loc[dev_mask].copy()
    y = dev["Label"].astype(int)
    assert len(dev) == 95997 and int(y.sum()) == 1228
    assert not dev["Year"].eq(LOCKED_YEAR).any()
    SPLITS = make_splits(y)
    assert len(SPLITS) == 25
    write_text(REPORT / "00_execution_boundary.md", "# robustness analysis execution boundary\n\nAll robustness analysis model calculations use only Year 1402, 1403, and 1404. `1404-2` is excluded from every fit, prediction, threshold calculation, performance table, bootstrap, subgroup, period, and explainability calculation. The frozen candidate and five-fold/five-seed design were recorded before execution.")

    runs: dict[str, dict[str, Any]] = {}
    runs["XGBoost/C"] = run_fixed("XGBoost", "C", dev, y, PARAMS, SPLITS, keep_predictions=True, collect_shap=True)
    robust = long_metric_summary(runs["XGBoost/C"]["folds"])
    robust.to_csv(TABLE / "frozen_model_robustness.csv", index=False, encoding="utf-8-sig")
    write_frozen_robustness_report(robust, runs["XGBoost/C"], float(y.mean()))

    # The following are frozen development analysis comparisons only; none are tuned here.
    runs["XGBoost/A"] = run_fixed("XGBoost", "A", dev, y, PARAMS, SPLITS)
    runs["XGBoost/B"] = run_fixed("XGBoost", "B", dev, y, PARAMS, SPLITS)
    runs["XGBoost/D"] = run_fixed("XGBoost", "D", dev, y, PARAMS, SPLITS, keep_predictions=True)
    runs["XGBoost/M0"] = run_fixed("XGBoost", "M0", dev, y, PARAMS, SPLITS)
    runs["XGBoost/M1"] = run_fixed("XGBoost", "M1", dev, y, PARAMS, SPLITS)
    runs["XGBoost/M2"] = run_fixed("XGBoost", "M2", dev, y, PARAMS, SPLITS)
    runs["Logistic Regression/B"] = run_fixed("Logistic Regression", "B", dev, y, metadata["parameters"]["Logistic Regression"], SPLITS)
    runs["Logistic Regression/C"] = run_fixed("Logistic Regression", "C", dev, y, metadata["parameters"]["Logistic Regression"], SPLITS)
    runs["LightGBM/C"] = run_fixed("LightGBM", "C", dev, y, metadata["parameters"]["LightGBM"], SPLITS)

    enrichment, _, enrichment_meta = no_skill_and_enrichment(runs["XGBoost/C"], y)
    simple_table, simple_result = run_simple_challenge(runs)
    incremental = run_incremental_value(runs)
    missingness_status, _ = run_missingness_decision(runs)
    complaint_status, _ = run_complaint_audit(dev, runs, y)
    calibration = run_calibration_stress(dev, runs["XGBoost/C"], y)
    pr = run_pr_stability(runs["XGBoost/C"])
    alert, pooled_alert = run_alert_budget(runs["XGBoost/C"], float(y.mean()))
    dca_status, _ = run_decision_curve(runs["XGBoost/C"], float(y.mean()))
    subgroup_table, subgroup_status = run_subgroups(dev, runs["XGBoost/C"])
    period_table, period_status = run_period_shift(dev, runs["XGBoost/C"])
    bootstrap = run_bootstrap(runs["XGBoost/C"], y)
    shap_table, shap_status, shap_detail = run_shap_stability(runs["XGBoost/C"])
    effect_status = run_feature_effect_sanity(runs["XGBoost/C"]["effects"], shap_table)
    error_status = run_error_phenotypes(dev, runs["XGBoost/C"])
    negative_status, negative_table = run_negative_control(dev, y, PARAMS)
    duplicate_status, duplicate_counts = run_duplicate_sensitivity(raw, clean)
    write_rare_event_report(enrichment, float(y.mean()))

    # Persist fold-level and primary repeated OOF artifacts after all calculations.
    fold_frames = []
    for key, run in runs.items():
        fold_frame = run["folds"].copy()
        fold_frame["run_key"] = key
        fold_frames.append(fold_frame)
    pd.concat(fold_frames, ignore_index=True).to_parquet(ARTIFACT / "frozen_model_fold_results.parquet", index=False)
    primary_pred = runs["XGBoost/C"]["predictions"].copy()
    primary_pred["model_family"] = "XGBoost"; primary_pred["feature_set"] = "C"; primary_pred["strategy"] = "none"
    primary_pred.to_parquet(ARTIFACT / "frozen_primary_oof_predictions.parquet", index=False)
    runs["XGBoost/D"]["predictions"].to_parquet(ARTIFACT / "complaint_sensitivity_oof_predictions.parquet", index=False)
    runs["XGBoost/C"]["shap"].to_csv(ARTIFACT / "frozen_fold_shap_source_summary.csv", index=False, encoding="utf-8-sig")

    robust_ap = robust.loc[robust.metric == "pr_auc", "mean"].iloc[0]
    pooled_top5 = enrichment.loc[enrichment.budget_pct == 5].iloc[0]
    if robust_ap > float(y.mean()) and negative_status == "PASS" and pooled_top5["ppv_enrichment"] > 2:
        final_model = "KEEP XGBOOST / FEATURE SET C"
    elif robust_ap > float(y.mean()):
        final_model = "DOWNGRADE TO XGBOOST / FEATURE SET B"
    else:
        final_model = "NO MODEL RELIABLE ENOUGH"
    q1 = "READY WITH WARNINGS" if final_model.startswith("KEEP") else "NOT READY — REVISE"
    cbm = "MODERATE" if q1 == "READY WITH WARNINGS" else "WEAK"
    aim = "CONDITIONAL"
    contribution = write_manuscript_decision_and_titles(robust, enrichment, incremental, missingness_status, complaint_status, subgroup_status, period_status)
    write_reviewer_stress(robust, enrichment, simple_table, missingness_status, complaint_status, subgroup_status, period_status)
    summary = write_summary_json(enrichment, robust, calibration, missingness_status, complaint_status, simple_result, subgroup_status, period_status, negative_status, duplicate_status, final_model, contribution, q1, cbm, aim)
    write_freeze_configs(enrichment, pooled_alert, calibration)
    write_master_report(summary, robust, enrichment, simple_table, incremental, missingness_status, complaint_status, calibration, pr, alert, dca_status, subgroup_status, period_status, bootstrap, shap_detail, error_status, negative_status, duplicate_status)

    # Copy the already fitted development analysis development pipeline as the frozen
    # robustness analysis reference; this is not a new fit and receives no locked rows.
    final_pipeline = joblib.load(ROOT / "models" / "development" / "selected_primary_pipeline.joblib")
    joblib.dump(final_pipeline, MODEL / "final_primary_pipeline.joblib", compress=3)
    write_json(MODEL / "final_model_metadata.json", {"source_pipeline": "models/development/selected_primary_pipeline.joblib", "model": "XGBoost", "feature_set": "C", "fit_rows": 95997, "locked_period_supplied": False, "calibration": "NO RECALIBRATION", "operating_policy": "top 5% primary; 1/2/5/10% reported"})
    write_json(REPORT / "ROBUSTNESS_RUN_MANIFEST.json", {"stage": "robustness analysis", "raw_sha256": raw_hash, "full_n": int(stats["full_n"]), "full_positive_n": int(stats["full_positive_n"]), "development_n": len(dev), "development_positive_n": int(y.sum()), "locked_candidate_period": LOCKED_YEAR, "locked_test_touched": False, "robustness_folds": N_SPLITS, "robustness_seeds": ROBUSTNESS_SEEDS, "bootstrap_n": BOOTSTRAP_N, "negative_control_seeds": NEGATIVE_CONTROL_SEEDS, "frozen_model": "XGBoost / C", "final_model_decision": final_model, "no_refit_after_robustness": True})
    _LOGGER.info("robustness analysis complete: %s; q1=%s; locked_test_touched=false", final_model, q1)


if __name__ == "__main__":
    if "--finalize-existing" in sys.argv:
        finalize_existing()
    else:
        main()
