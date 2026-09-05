"""Runtime helpers for the development analysis.

The module intentionally contains no data-loading side effects.  All learned
transformations are fitted by the estimator/pipeline on the current training
fold.  Raw-source cleaning and analysis orchestration live in development_analysis.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


BASE_FEATURES_A = ["Age", "Sex", "Triage level"]
NUMERIC_FEATURES_B = [
    "Age",
    "Triage level",
    "SPo2",
    "BPMin",
    "BPMax",
    "PR",
    "RR",
    "T",
    "BS",
    "BS.1",
    "WBC",
    "HB",
    "HCT",
    "PLT",
    "ESR",
    "UREA",
    "CR",
    "NA",
    "K",
]
CATEGORICAL_FEATURES_B = ["Sex", "CRP"]
BASE_FEATURES_B = NUMERIC_FEATURES_B + CATEGORICAL_FEATURES_B
MISSINGNESS_COLUMNS_B = [f"missing__{c}" for c in BASE_FEATURES_B]
FEATURE_SETS = {
    "A": BASE_FEATURES_A,
    "B": BASE_FEATURES_B,
    "C": BASE_FEATURES_B + MISSINGNESS_COLUMNS_B,
    "D": BASE_FEATURES_B + MISSINGNESS_COLUMNS_B + ["complaint_group"],
}
MODEL_FAMILIES = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]


class MissingnessAugmenter(BaseEstimator, TransformerMixin):
    """Add one row-level missingness indicator for every eligible base field."""

    def __init__(self, base_features: Iterable[str]):
        self.base_features = tuple(base_features)

    def fit(self, X: pd.DataFrame, y: Any = None):
        self.feature_names_in_ = list(X.columns)
        missing = [c for c in self.base_features if c not in self.feature_names_in_]
        if missing:
            raise ValueError(f"Missing required fields for indicators: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        out = X.copy()
        for col in self.base_features:
            if col not in out.columns:
                raise ValueError(f"Missing required field during transform: {col}")
            values = out[col]
            if pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
                miss = values.isna() | values.astype("string").str.strip().eq("")
            else:
                miss = values.isna()
            out[f"missing__{col}"] = miss.astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        base = list(input_features if input_features is not None else self.feature_names_in_)
        return np.asarray(base + list(MISSINGNESS_COLUMNS_B), dtype=object)


class CatBoostFoldSafeClassifier(BaseEstimator, ClassifierMixin):
    """CatBoost wrapper with fold-fitted numeric medians and native categories."""

    def __init__(
        self,
        iterations: int = 250,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 5.0,
        random_seed: int = 42,
        auto_class_weights: str | None = None,
        thread_count: int = 1,
        verbose: bool = False,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.auto_class_weights = auto_class_weights
        self.thread_count = thread_count
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: Any):
        X = pd.DataFrame(X).copy()
        self.columns_ = list(X.columns)
        self.categorical_columns_ = [
            c for c in self.columns_ if c in {"Sex", "CRP", "complaint_group"}
        ]
        self.numeric_columns_ = [c for c in self.columns_ if c not in self.categorical_columns_]
        self.numeric_medians_ = {}
        for col in self.numeric_columns_:
            vals = pd.to_numeric(X[col], errors="coerce")
            median = float(vals.median()) if vals.notna().any() else 0.0
            self.numeric_medians_[col] = median
        self.cat_indices_ = [self.columns_.index(c) for c in self.categorical_columns_]
        Xt = self._transform(X)
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            auto_class_weights=self.auto_class_weights,
            thread_count=self.thread_count,
            verbose=self.verbose,
            loss_function="Logloss",
            eval_metric="AUC",
            allow_writing_files=False,
        )
        self.model_.fit(Xt, np.asarray(y), cat_features=self.cat_indices_, verbose=False)
        self.classes_ = np.asarray([0, 1])
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in self.numeric_columns_:
            out[col] = pd.to_numeric(X[col], errors="coerce").fillna(self.numeric_medians_[col])
        for col in self.categorical_columns_:
            values = X[col].astype("string").fillna("__MISSING__").str.strip()
            out[col] = values.mask(values.eq(""), "__MISSING__").astype(str)
        return out[self.columns_]

    def predict_proba(self, X: pd.DataFrame):
        check_is_fitted(self, ["model_", "columns_"])
        Xt = self._transform(pd.DataFrame(X))
        return np.asarray(self.model_.predict_proba(Xt))

    def predict(self, X: pd.DataFrame):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        check_is_fitted(self, "model_")
        return self.model_.get_feature_importance()


@dataclass
class ModelSpec:
    name: str
    params: dict[str, Any]
    strategy: str = "none"


def feature_columns(feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return list(FEATURE_SETS[feature_set])


def _numeric_and_categorical(columns: list[str]) -> tuple[list[str], list[str]]:
    categorical = [c for c in columns if c in {"Sex", "CRP", "complaint_group"}]
    numeric = [c for c in columns if c not in categorical]
    return numeric, categorical


def make_estimator(model_name: str, params: dict[str, Any] | None = None, strategy: str = "none", y: Any = None):
    params = dict(params or {})
    if model_name == "Logistic Regression":
        params.setdefault("C", 1.0)
        params.setdefault("max_iter", 350)
        params.setdefault("solver", "liblinear")
        params.setdefault("random_state", 42)
        params["class_weight"] = "balanced" if strategy == "class_weight" else None
        return LogisticRegression(**params)
    if model_name == "Random Forest":
        params.setdefault("n_estimators", 180)
        params.setdefault("max_depth", 14)
        params.setdefault("min_samples_leaf", 2)
        params.setdefault("max_features", "sqrt")
        params.setdefault("random_state", 42)
        params.setdefault("n_jobs", 1)
        params["class_weight"] = "balanced" if strategy == "class_weight" else None
        return RandomForestClassifier(**params)
    if model_name == "XGBoost":
        params.setdefault("n_estimators", 220)
        params.setdefault("max_depth", 4)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("min_child_weight", 2)
        params.setdefault("subsample", 0.85)
        params.setdefault("colsample_bytree", 0.85)
        params.setdefault("reg_lambda", 1.0)
        params.setdefault("random_state", 42)
        params.setdefault("n_jobs", 1)
        params.setdefault("tree_method", "hist")
        params.setdefault("eval_metric", "logloss")
        if strategy == "class_weight":
            if y is None:
                raise ValueError("XGBoost class weighting needs training labels")
            yy = np.asarray(y)
            params["scale_pos_weight"] = float((yy == 0).sum() / max((yy == 1).sum(), 1))
        else:
            params["scale_pos_weight"] = 1.0
        return XGBClassifier(**params)
    if model_name == "LightGBM":
        params.setdefault("n_estimators", 220)
        params.setdefault("num_leaves", 31)
        params.setdefault("max_depth", -1)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("min_child_samples", 30)
        params.setdefault("subsample", 0.85)
        params.setdefault("colsample_bytree", 0.85)
        params.setdefault("reg_lambda", 1.0)
        params.setdefault("random_state", 42)
        params.setdefault("n_jobs", 1)
        params.setdefault("verbosity", -1)
        params["class_weight"] = "balanced" if strategy == "class_weight" else None
        return LGBMClassifier(**params)
    if model_name == "CatBoost":
        params.setdefault("iterations", 250)
        params.setdefault("depth", 6)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("l2_leaf_reg", 5.0)
        params.setdefault("random_seed", 42)
        params.setdefault("thread_count", 1)
        params.setdefault("verbose", False)
        params["auto_class_weights"] = "Balanced" if strategy == "class_weight" else None
        return CatBoostFoldSafeClassifier(**params)
    raise ValueError(f"Unknown model family: {model_name}")


def make_pipeline_for_columns(
    model_name: str,
    columns: list[str],
    params: dict[str, Any] | None = None,
    strategy: str = "none",
    y: Any = None,
    add_missingness: bool = False,
):
    columns = list(columns)
    estimator = make_estimator(model_name, params=params, strategy=strategy, y=y)
    steps: list[tuple[str, Any]] = []
    if add_missingness:
        steps.append(("missingness", MissingnessAugmenter(BASE_FEATURES_B)))
        columns_after = columns
        # The raw X frame contains only base predictors (and complaint_group
        # for Set D); the augmenter creates the indicator columns in-fold.
        columns = [c for c in columns if not c.startswith("missing__")]
    else:
        columns_after = columns
    if model_name == "CatBoost":
        steps.append(("model", estimator))
        return Pipeline(steps)
    numeric, categorical = _numeric_and_categorical(columns_after)
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if model_name == "Logistic Regression":
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))
    cat_steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", Pipeline(cat_steps), categorical),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )
    steps.extend([("preprocess", preprocessor), ("model", estimator)])
    return Pipeline(steps)


def make_pipeline(model_name: str, feature_set: str, params: dict[str, Any] | None = None, strategy: str = "none", y: Any = None):
    return make_pipeline_for_columns(
        model_name,
        feature_columns(feature_set),
        params=params,
        strategy=strategy,
        y=y,
        add_missingness=feature_set in {"C", "D"},
    )


def resample_training(X: pd.DataFrame, y: pd.Series, strategy: str, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Apply only training-fold random resampling to preserve mixed field semantics."""
    if strategy == "none" or strategy == "class_weight":
        return X, y
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y).astype(int)
    pos = np.flatnonzero(y_arr == 1)
    neg = np.flatnonzero(y_arr == 0)
    if len(pos) == 0 or len(neg) == 0:
        return X, y
    target_neg = min(len(neg), len(pos) * 3)
    target_pos = len(pos)
    if strategy == "undersample":
        chosen_neg = rng.choice(neg, size=target_neg, replace=False)
        chosen = np.concatenate([pos, chosen_neg])
    elif strategy == "oversample":
        target_pos = min(len(pos) * 3, len(neg))
        chosen_pos = rng.choice(pos, size=target_pos, replace=True)
        chosen = np.concatenate([neg, chosen_pos])
    else:
        raise ValueError(f"Unknown resampling strategy: {strategy}")
    rng.shuffle(chosen)
    return X.iloc[chosen].copy(), y.iloc[chosen].copy()


def top_fraction_metrics(y_true: Iterable[int], proba: Iterable[float], fraction: float = 0.05) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    n_alert = max(1, int(np.ceil(len(y) * fraction)))
    chosen = np.argsort(-p, kind="mergesort")[:n_alert]
    pred = np.zeros(len(y), dtype=int)
    pred[chosen] = 1
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if ppv + sensitivity else np.nan
    beta2 = 4.0
    f2 = (1 + beta2) * ppv * sensitivity / (beta2 * ppv + sensitivity) if beta2 * ppv + sensitivity else np.nan
    return {
        "alerted_n": float(n_alert),
        "hai_captured_n": float(tp),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "f2": float(f2),
        "false_positives": float(fp),
        "false_alerts_per_true_hai": float(fp / tp) if tp else np.nan,
        "enrichment_over_prevalence": float((ppv / y.mean()) if y.mean() else np.nan),
    }


def calibration_in_the_large(y_true: Iterable[int], proba: Iterable[float]) -> float:
    y = np.asarray(y_true).astype(float)
    p = np.clip(np.asarray(proba, dtype=float), 1e-7, 1 - 1e-7)
    offset = logit(p)
    target = float(y.mean())
    fn = lambda intercept: float(expit(offset + intercept).mean() - target)
    try:
        return float(brentq(fn, -50, 50))
    except ValueError:
        return float("nan")


def calibration_slope_intercept(y_true: Iterable[int], proba: Iterable[float]) -> tuple[float, float]:
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(proba, dtype=float), 1e-7, 1 - 1e-7)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    model.fit(logit(p).reshape(-1, 1), y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def binary_metrics(y_true: Iterable[int], proba: Iterable[float], budget: float = 0.05) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    out = {
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "calibration_in_the_large": calibration_in_the_large(y, p),
    }
    try:
        out["auroc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["auroc"] = float("nan")
    slope, intercept = calibration_slope_intercept(y, p)
    out["calibration_slope"] = slope
    out["calibration_intercept"] = intercept
    out.update({f"{k}_at_{int(budget * 100)}pct": v for k, v in top_fraction_metrics(y, p, budget).items()})
    return out


def get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    if "preprocess" not in pipeline.named_steps:
        model = pipeline.named_steps["model"]
        if hasattr(model, "columns_"):
            return list(model.columns_)
        return []
    pre = pipeline.named_steps["preprocess"]
    try:
        return [str(x) for x in pre.get_feature_names_out()]
    except Exception:
        return []


def aggregate_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Return fold-fitted importance, aggregating one-hot levels to source fields."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_") and not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["feature", "importance"])
    if hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).reshape(-1))
    else:
        values = np.asarray(model.feature_importances_).reshape(-1)
    names = get_transformed_feature_names(pipeline)
    if len(names) != len(values):
        names = [f"transformed_{i}" for i in range(len(values))]
    rows: dict[str, float] = {}
    for name, value in zip(names, values):
        source = name
        for prefix in ("numeric__", "categorical__"):
            if source.startswith(prefix):
                source = source[len(prefix):]
        source = source.split("_")[0] if source.startswith(("Sex_", "CRP_", "complaint_group_")) else source
        rows[source] = rows.get(source, 0.0) + float(value)
    return pd.DataFrame({"feature": list(rows), "importance": list(rows.values())}).sort_values("importance", ascending=False)


def safe_json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return [safe_json_value(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json_value(v) for v in value]
    return value
