# -*- coding: utf-8 -*-
"""
07_train_baselines.py

Baseline-модели для оконного EEG PM/POW датасета.

Вход:
    data/processed/windowed_pm_dataset_w10.parquet

Задачи:
    1. Classification:
        X = POW.*
        y = label_q5

    2. Regression:
        X = POW.*
        y = target_main

Важно:
    PM.* не используются как признаки, потому что target_main построен из PM.Focus.
    target_* и label_* тоже не используются как признаки.
    subject_id используется только как group для GroupKFold / subject-aware validation.

Новые возможности:
    --feature-mode raw_pow / log_pow / raw_plus_log_pow
    --enable-cross-source-no-overlap

Запуск:
    D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\07_train_baselines.py --fast --feature-mode log_pow --output-prefix baseline_pow_w10_log_fast

Быстрый тест:
    D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\07_train_baselines.py --max-rows 10000 --fast --feature-mode log_pow --enable-cross-source-no-overlap --output-prefix baseline_pow_w10_log_test
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler


try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = None
    spearmanr = None


RANDOM_STATE = 42

DEFAULT_CLASS_TARGET = "label_q5"
DEFAULT_REG_TARGET = "target_main"

META_COLS = {
    "record_id",
    "source",
    "subject_id",
    "day",
    "part",
    "datetime_from_name",
    "t_center",
    "t_start",
    "t_end",
}

TARGET_PREFIXES = ("target_", "label_")
LEAKAGE_PREFIXES = ("PM.",)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def df_to_markdown_safe(df: pd.DataFrame, index: bool = True) -> str:
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return df.to_string(index=index)


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def sanitize_filename(s: str) -> str:
    out = str(s)
    for ch in ["\\", "/", ":", "*", "?", '"', "<", ">", "|", " "]:
        out = out.replace(ch, "_")
    return out


def infer_pow_feature_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if not c.startswith("POW."):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def transform_features(
    df: pd.DataFrame,
    raw_feature_cols: List[str],
    feature_mode: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Создает матрицу признаков без утечки.

    feature_mode:
        raw_pow:
            исходные POW.* агрегаты

        log_pow:
            log1p(max(x, 0)) для POW.*.
            Это полезно из-за тяжелых хвостов и больших выбросов в POW.

        raw_plus_log_pow:
            исходные POW.* + log1p(max(x, 0)).
    """
    X_raw = df[raw_feature_cols].copy()

    if feature_mode == "raw_pow":
        return X_raw, raw_feature_cols

    if feature_mode == "log_pow":
        X_log = np.log1p(X_raw.clip(lower=0))
        X_log.columns = [f"log1p_{c}" for c in raw_feature_cols]
        return X_log, X_log.columns.tolist()

    if feature_mode == "raw_plus_log_pow":
        X_log = np.log1p(X_raw.clip(lower=0))
        X_log.columns = [f"log1p_{c}" for c in raw_feature_cols]
        X = pd.concat([X_raw, X_log], axis=1)
        return X, X.columns.tolist()

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def make_classification_models(fast: bool = False) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    models["logreg_robust"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    models["hgb_clf"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=80 if fast else 200,
                    max_leaf_nodes=31,
                    l2_regularization=0.01,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    if not fast:
        models["rf_clf"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    try:
        from lightgbm import LGBMClassifier

        models["lgbm_clf"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=150 if fast else 500,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    except Exception:
        pass

    return models


def make_regression_models(fast: bool = False) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    models["ridge_robust"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )

    models["hgb_reg"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=80 if fast else 200,
                    max_leaf_nodes=31,
                    l2_regularization=0.01,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    if not fast:
        models["rf_reg"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=250,
                        max_depth=None,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    try:
        from lightgbm import LGBMRegressor

        models["lgbm_reg"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=150 if fast else 500,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    except Exception:
        pass

    return models


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)

    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }

    if pearsonr is not None:
        try:
            out["pearson"] = float(pearsonr(y_true, y_pred)[0])
        except Exception:
            out["pearson"] = np.nan
    else:
        out["pearson"] = np.nan

    if spearmanr is not None:
        try:
            out["spearman"] = float(spearmanr(y_true, y_pred)[0])
        except Exception:
            out["spearman"] = np.nan
    else:
        out["spearman"] = np.nan

    return out


def prepare_classification_data(
    df: pd.DataFrame,
    raw_feature_cols: List[str],
    target_col: str,
    min_windows_per_subject: int,
    feature_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    data = df.copy()
    data = data[data[target_col].notna()].copy()
    data[target_col] = data[target_col].astype(int)

    counts = data["subject_id"].value_counts()
    keep_subjects = counts[counts >= min_windows_per_subject].index
    data = data[data["subject_id"].isin(keep_subjects)].copy()

    X, _ = transform_features(data, raw_feature_cols, feature_mode)
    y = data[target_col].to_numpy()

    return data, X, y


def prepare_regression_data(
    df: pd.DataFrame,
    raw_feature_cols: List[str],
    target_col: str,
    min_windows_per_subject: int,
    feature_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    data = df.copy()
    data = data[data[target_col].notna()].copy()
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data[data[target_col].notna()].copy()

    counts = data["subject_id"].value_counts()
    keep_subjects = counts[counts >= min_windows_per_subject].index
    data = data[data["subject_id"].isin(keep_subjects)].copy()

    X, _ = transform_features(data, raw_feature_cols, feature_mode)
    y = data[target_col].to_numpy(dtype=float)

    return data, X, y


def get_base_prediction_columns(data: pd.DataFrame, target_col: str) -> List[str]:
    cols = ["record_id", "source", "subject_id", "day", target_col]
    return [c for c in cols if c in data.columns]


def run_random_split_classification(
    data: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    target_col: str,
    models: Dict[str, Any],
    test_size: float,
) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame]]:
    metrics_rows = []
    pred_parts = []

    train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    for model_name, model in models.items():
        print(f"[classification/random] model={model_name}")

        t0 = time.time()
        est = clone(model)
        est.fit(X.iloc[train_idx], y[train_idx])
        y_pred = est.predict(X.iloc[test_idx])
        elapsed = time.time() - t0

        row = {
            "task": "classification",
            "validation": "random_split",
            "model": model_name,
            "fold": 0,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "elapsed_s": elapsed,
        }
        row.update(classification_metrics(y[test_idx], y_pred))
        metrics_rows.append(row)

        pred = data.iloc[test_idx][get_base_prediction_columns(data, target_col)].copy()
        pred["validation"] = "random_split"
        pred["model"] = model_name
        pred["fold"] = 0
        pred["y_true"] = y[test_idx]
        pred["y_pred"] = y_pred
        pred_parts.append(pred)

    return metrics_rows, pred_parts


def run_groupkfold_classification(
    data: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    target_col: str,
    models: Dict[str, Any],
    n_splits: int,
) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame]]:
    metrics_rows = []
    pred_parts = []

    groups = data["subject_id"].astype(str).to_numpy()
    n_splits_eff = min(n_splits, len(np.unique(groups)))

    gkf = GroupKFold(n_splits=n_splits_eff)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        print(f"[classification/groupkfold] fold={fold}/{n_splits_eff}")

        for model_name, model in models.items():
            print(f"  model={model_name}")

            t0 = time.time()
            est = clone(model)
            est.fit(X.iloc[train_idx], y[train_idx])
            y_pred = est.predict(X.iloc[test_idx])
            elapsed = time.time() - t0

            row = {
                "task": "classification",
                "validation": "groupkfold_subject",
                "model": model_name,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_train_subjects": len(np.unique(groups[train_idx])),
                "n_test_subjects": len(np.unique(groups[test_idx])),
                "elapsed_s": elapsed,
            }
            row.update(classification_metrics(y[test_idx], y_pred))
            metrics_rows.append(row)

            pred = data.iloc[test_idx][get_base_prediction_columns(data, target_col)].copy()
            pred["validation"] = "groupkfold_subject"
            pred["model"] = model_name
            pred["fold"] = fold
            pred["y_true"] = y[test_idx]
            pred["y_pred"] = y_pred
            pred_parts.append(pred)

    return metrics_rows, pred_parts


def run_cross_source_classification(
    data: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    target_col: str,
    models: Dict[str, Any],
    no_subject_overlap: bool,
) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame]]:
    metrics_rows = []
    pred_parts = []

    sources = sorted(data["source"].dropna().unique().tolist())
    if len(sources) < 2:
        return metrics_rows, pred_parts

    for train_source in sources:
        for test_source in sources:
            if train_source == test_source:
                continue

            train_mask = data["source"] == train_source
            test_mask = data["source"] == test_source

            if no_subject_overlap:
                train_subjects = set(data.loc[train_mask, "subject_id"].astype(str))
                test_mask = test_mask & ~data["subject_id"].astype(str).isin(train_subjects)
                validation_name = f"cross_source_no_overlap_train_{train_source}_test_{test_source}"
            else:
                validation_name = f"cross_source_train_{train_source}_test_{test_source}"

            train_idx = np.where(train_mask.to_numpy())[0]
            test_idx = np.where(test_mask.to_numpy())[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                print(f"[WARN] skip {validation_name}: n_train={len(train_idx)}, n_test={len(test_idx)}")
                continue

            for model_name, model in models.items():
                print(f"[classification/{validation_name}] model={model_name}")

                t0 = time.time()
                est = clone(model)
                est.fit(X.iloc[train_idx], y[train_idx])
                y_pred = est.predict(X.iloc[test_idx])
                elapsed = time.time() - t0

                row = {
                    "task": "classification",
                    "validation": validation_name,
                    "model": model_name,
                    "fold": 0,
                    "train_source": train_source,
                    "test_source": test_source,
                    "no_subject_overlap": no_subject_overlap,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_train_subjects": data.iloc[train_idx]["subject_id"].nunique(),
                    "n_test_subjects": data.iloc[test_idx]["subject_id"].nunique(),
                    "elapsed_s": elapsed,
                }
                row.update(classification_metrics(y[test_idx], y_pred))
                metrics_rows.append(row)

                pred = data.iloc[test_idx][get_base_prediction_columns(data, target_col)].copy()
                pred["validation"] = validation_name
                pred["model"] = model_name
                pred["fold"] = 0
                pred["y_true"] = y[test_idx]
                pred["y_pred"] = y_pred
                pred_parts.append(pred)

    return metrics_rows, pred_parts


def run_random_split_regression(
    data: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    target_col: str,
    models: Dict[str, Any],
    test_size: float,
) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame]]:
    metrics_rows = []
    pred_parts = []

    train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    for model_name, model in models.items():
        print(f"[regression/random] model={model_name}")

        t0 = time.time()
        est = clone(model)
        est.fit(X.iloc[train_idx], y[train_idx])
        y_pred = est.predict(X.iloc[test_idx])
        elapsed = time.time() - t0

        row = {
            "task": "regression",
            "validation": "random_split",
            "model": model_name,
            "fold": 0,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "elapsed_s": elapsed,
        }
        row.update(regression_metrics(y[test_idx], y_pred))
        metrics_rows.append(row)

        pred = data.iloc[test_idx][get_base_prediction_columns(data, target_col)].copy()
        pred["validation"] = "random_split"
        pred["model"] = model_name
        pred["fold"] = 0
        pred["y_true"] = y[test_idx]
        pred["y_pred"] = y_pred
        pred_parts.append(pred)

    return metrics_rows, pred_parts


def run_groupkfold_regression(
    data: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    target_col: str,
    models: Dict[str, Any],
    n_splits: int,
) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame]]:
    metrics_rows = []
    pred_parts = []

    groups = data["subject_id"].astype(str).to_numpy()
    n_splits_eff = min(n_splits, len(np.unique(groups)))

    gkf = GroupKFold(n_splits=n_splits_eff)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        print(f"[regression/groupkfold] fold={fold}/{n_splits_eff}")

        for model_name, model in models.items():
            print(f"  model={model_name}")

            t0 = time.time()
            est = clone(model)
            est.fit(X.iloc[train_idx], y[train_idx])
            y_pred = est.predict(X.iloc[test_idx])
            elapsed = time.time() - t0

            row = {
                "task": "regression",
                "validation": "groupkfold_subject",
                "model": model_name,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_train_subjects": len(np.unique(groups[train_idx])),
                "n_test_subjects": len(np.unique(groups[test_idx])),
                "elapsed_s": elapsed,
            }
            row.update(regression_metrics(y[test_idx], y_pred))
            metrics_rows.append(row)

            pred = data.iloc[test_idx][get_base_prediction_columns(data, target_col)].copy()
            pred["validation"] = "groupkfold_subject"
            pred["model"] = model_name
            pred["fold"] = fold
            pred["y_true"] = y[test_idx]
            pred["y_pred"] = y_pred
            pred_parts.append(pred)

    return metrics_rows, pred_parts


def run_cross_source_regression(
    data: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    target_col: str,
    models: Dict[str, Any],
    no_subject_overlap: bool,
) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame]]:
    metrics_rows = []
    pred_parts = []

    sources = sorted(data["source"].dropna().unique().tolist())
    if len(sources) < 2:
        return metrics_rows, pred_parts

    for train_source in sources:
        for test_source in sources:
            if train_source == test_source:
                continue

            train_mask = data["source"] == train_source
            test_mask = data["source"] == test_source

            if no_subject_overlap:
                train_subjects = set(data.loc[train_mask, "subject_id"].astype(str))
                test_mask = test_mask & ~data["subject_id"].astype(str).isin(train_subjects)
                validation_name = f"cross_source_no_overlap_train_{train_source}_test_{test_source}"
            else:
                validation_name = f"cross_source_train_{train_source}_test_{test_source}"

            train_idx = np.where(train_mask.to_numpy())[0]
            test_idx = np.where(test_mask.to_numpy())[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                print(f"[WARN] skip {validation_name}: n_train={len(train_idx)}, n_test={len(test_idx)}")
                continue

            for model_name, model in models.items():
                print(f"[regression/{validation_name}] model={model_name}")

                t0 = time.time()
                est = clone(model)
                est.fit(X.iloc[train_idx], y[train_idx])
                y_pred = est.predict(X.iloc[test_idx])
                elapsed = time.time() - t0

                row = {
                    "task": "regression",
                    "validation": validation_name,
                    "model": model_name,
                    "fold": 0,
                    "train_source": train_source,
                    "test_source": test_source,
                    "no_subject_overlap": no_subject_overlap,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_train_subjects": data.iloc[train_idx]["subject_id"].nunique(),
                    "n_test_subjects": data.iloc[test_idx]["subject_id"].nunique(),
                    "elapsed_s": elapsed,
                }
                row.update(regression_metrics(y[test_idx], y_pred))
                metrics_rows.append(row)

                pred = data.iloc[test_idx][get_base_prediction_columns(data, target_col)].copy()
                pred["validation"] = validation_name
                pred["model"] = model_name
                pred["fold"] = 0
                pred["y_true"] = y[test_idx]
                pred["y_pred"] = y_pred
                pred_parts.append(pred)

    return metrics_rows, pred_parts


def aggregate_metrics(metrics_df: pd.DataFrame, task: str) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    if task == "classification":
        metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    else:
        metric_cols = ["mae", "rmse", "r2", "pearson", "spearman"]

    rows = []

    for (validation, model), g in metrics_df.groupby(["validation", "model"], dropna=False):
        row = {
            "task": task,
            "validation": validation,
            "model": model,
            "folds": len(g),
        }

        for c in metric_cols:
            if c in g.columns:
                row[f"{c}_mean"] = float(g[c].mean())
                row[f"{c}_std"] = float(g[c].std()) if len(g) > 1 else 0.0
                row[f"{c}_min"] = float(g[c].min())
                row[f"{c}_max"] = float(g[c].max())

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["validation", "model"]).reset_index(drop=True)


def plot_classification_summary(agg_df: pd.DataFrame, fig_dir: Path) -> List[Path]:
    paths = []

    if agg_df.empty or "macro_f1_mean" not in agg_df.columns:
        return paths

    for validation in agg_df["validation"].unique():
        sub = agg_df[agg_df["validation"] == validation].copy()

        path = fig_dir / f"classification_macro_f1_{sanitize_filename(validation)}.png"

        plt.figure(figsize=(10, 5))
        plt.bar(sub["model"], sub["macro_f1_mean"])
        plt.title(f"Classification macro-F1: {validation}")
        plt.xlabel("model")
        plt.ylabel("macro-F1")
        plt.xticks(rotation=45, ha="right")
        save_plot(path)
        paths.append(path)

    return paths


def plot_regression_summary(agg_df: pd.DataFrame, fig_dir: Path) -> List[Path]:
    paths = []

    if agg_df.empty or "rmse_mean" not in agg_df.columns:
        return paths

    for validation in agg_df["validation"].unique():
        sub = agg_df[agg_df["validation"] == validation].copy()

        path = fig_dir / f"regression_rmse_{sanitize_filename(validation)}.png"

        plt.figure(figsize=(10, 5))
        plt.bar(sub["model"], sub["rmse_mean"])
        plt.title(f"Regression RMSE: {validation}")
        plt.xlabel("model")
        plt.ylabel("RMSE")
        plt.xticks(rotation=45, ha="right")
        save_plot(path)
        paths.append(path)

    return paths


def plot_confusion_matrix_for_best(
    pred_df: pd.DataFrame,
    metrics_agg: pd.DataFrame,
    fig_dir: Path,
) -> Optional[Path]:
    if pred_df.empty or metrics_agg.empty or "macro_f1_mean" not in metrics_agg.columns:
        return None

    g = metrics_agg[metrics_agg["validation"] == "groupkfold_subject"].copy()
    if g.empty:
        g = metrics_agg.copy()

    best = g.sort_values("macro_f1_mean", ascending=False).iloc[0]
    validation = best["validation"]
    model = best["model"]

    sub = pred_df[(pred_df["validation"] == validation) & (pred_df["model"] == model)].copy()
    if sub.empty:
        return None

    y_true = sub["y_true"].astype(int).to_numpy()
    y_pred = sub["y_pred"].astype(int).to_numpy()
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    path = fig_dir / f"best_confusion_matrix_{sanitize_filename(validation)}_{sanitize_filename(model)}.png"

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion matrix: {model}, {validation}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    save_plot(path)
    return path


def plot_regression_scatter_for_best(
    pred_df: pd.DataFrame,
    metrics_agg: pd.DataFrame,
    fig_dir: Path,
) -> Optional[Path]:
    if pred_df.empty or metrics_agg.empty or "rmse_mean" not in metrics_agg.columns:
        return None

    g = metrics_agg[metrics_agg["validation"] == "groupkfold_subject"].copy()
    if g.empty:
        g = metrics_agg.copy()

    best = g.sort_values("rmse_mean", ascending=True).iloc[0]
    validation = best["validation"]
    model = best["model"]

    sub = pred_df[(pred_df["validation"] == validation) & (pred_df["model"] == model)].copy()
    if sub.empty:
        return None

    if len(sub) > 10000:
        sub = sub.sample(10000, random_state=RANDOM_STATE)

    path = fig_dir / f"best_regression_scatter_{sanitize_filename(validation)}_{sanitize_filename(model)}.png"

    plt.figure(figsize=(7, 6))
    plt.scatter(sub["y_true"], sub["y_pred"], s=8, alpha=0.4)
    plt.title(f"Regression: {model}, {validation}")
    plt.xlabel("True target_main")
    plt.ylabel("Predicted target_main")

    vmin = min(sub["y_true"].min(), sub["y_pred"].min())
    vmax = max(sub["y_true"].max(), sub["y_pred"].max())
    plt.plot([vmin, vmax], [vmin, vmax], linestyle="--")

    save_plot(path)
    return path


def make_report(
    report_path: Path,
    dataset_path: Path,
    feature_mode: str,
    raw_feature_cols: List[str],
    final_feature_count: int,
    class_metrics: pd.DataFrame,
    class_agg: pd.DataFrame,
    reg_metrics: pd.DataFrame,
    reg_agg: pd.DataFrame,
    class_data_info: Dict[str, Any],
    reg_data_info: Dict[str, Any],
    figures: List[Path],
    root: Path,
) -> None:
    lines = []

    lines.append("# Baseline report")
    lines.append("")
    lines.append(f"Dataset: `{dataset_path}`")
    lines.append("")

    lines.append("## Feature policy")
    lines.append("")
    lines.append("- Used as base features: `POW.*` aggregated columns only.")
    lines.append("- Excluded from features: `PM.*`, `target_*`, `label_*`, `source`, `subject_id`, `record_id`, time/meta columns.")
    lines.append("- Reason: `target_main` is derived from `PM.Focus.Scaled`; including PM columns would cause target leakage.")
    lines.append("")
    lines.append(f"- Feature mode: **{feature_mode}**")
    lines.append(f"- Raw POW feature columns: **{len(raw_feature_cols)}**")
    lines.append(f"- Final feature columns after transform: **{final_feature_count}**")
    lines.append("")

    lines.append("## Classification data")
    lines.append("")
    lines.append(df_to_markdown_safe(pd.DataFrame([class_data_info]), index=False))
    lines.append("")

    lines.append("## Regression data")
    lines.append("")
    lines.append(df_to_markdown_safe(pd.DataFrame([reg_data_info]), index=False))
    lines.append("")

    lines.append("## Classification aggregated metrics")
    lines.append("")
    lines.append(df_to_markdown_safe(class_agg, index=False) if not class_agg.empty else "_No classification metrics._")
    lines.append("")

    lines.append("## Classification fold metrics")
    lines.append("")
    lines.append(df_to_markdown_safe(class_metrics.head(100), index=False) if not class_metrics.empty else "_No classification fold metrics._")
    lines.append("")

    lines.append("## Regression aggregated metrics")
    lines.append("")
    lines.append(df_to_markdown_safe(reg_agg, index=False) if not reg_agg.empty else "_No regression metrics._")
    lines.append("")

    lines.append("## Regression fold metrics")
    lines.append("")
    lines.append(df_to_markdown_safe(reg_metrics.head(100), index=False) if not reg_metrics.empty else "_No regression fold metrics._")
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    for fig in figures:
        try:
            rel = fig.relative_to(root)
        except Exception:
            rel = fig
        lines.append(f"- `{rel}`")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("1. Random split is only a sanity check and likely overestimates performance.")
    lines.append("2. GroupKFold by `subject_id` is the main baseline validation scheme.")
    lines.append("3. Cross-source validation with subject overlap estimates source transfer but may be optimistic.")
    lines.append("4. Cross-source validation without subject overlap is stricter and should be used for transfer conclusions.")
    lines.append("5. If GroupKFold metrics are much lower than random split metrics, the task has strong subject-specific effects.")
    lines.append("6. If linear models fail on raw POW but improve on log POW, the issue is likely heavy-tailed spectral features.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=r"D:\PycharmProjects\eeg-cognitive-state-nir",
        help="Корень проекта.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=r"data\processed\windowed_pm_dataset_w10.parquet",
        help="Путь к датасету относительно root.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="baseline_pow_w10",
    )
    parser.add_argument(
        "--class-target",
        type=str,
        default=DEFAULT_CLASS_TARGET,
    )
    parser.add_argument(
        "--reg-target",
        type=str,
        default=DEFAULT_REG_TARGET,
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=["raw_pow", "log_pow", "raw_plus_log_pow"],
        default="raw_pow",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--min-windows-per-subject",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Для быстрого теста. Сэмплирует строки после загрузки.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Быстрый режим: меньше моделей и итераций.",
    )
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--skip-groupkfold", action="store_true")
    parser.add_argument("--skip-cross-source", action="store_true")
    parser.add_argument(
        "--enable-cross-source-no-overlap",
        action="store_true",
        help="Дополнительно запускает cross-source validation с удалением пересекающихся subject_id из test.",
    )

    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    root = Path(args.root).resolve()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path

    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"
    fig_dir = reports_dir / "figures" / args.output_prefix

    ensure_dir(processed_dir)
    ensure_dir(reports_dir)
    ensure_dir(fig_dir)

    print("=" * 80)
    print("Train baseline models")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Dataset: {dataset_path}")
    print(f"Output prefix: {args.output_prefix}")
    print(f"Feature mode: {args.feature_mode}")

    df = pd.read_parquet(dataset_path)

    if args.max_rows is not None and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Sampled max_rows={args.max_rows}")

    required = ["source", "subject_id", "record_id", args.class_target, args.reg_target]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw_feature_cols = infer_pow_feature_cols(df)
    if not raw_feature_cols:
        raise RuntimeError("No POW.* feature columns found.")

    _, final_feature_cols = transform_features(df.head(5), raw_feature_cols, args.feature_mode)

    leakage_cols = [
        c for c in raw_feature_cols
        if c.startswith("PM.") or c.startswith("target_") or c.startswith("label_")
    ]
    if leakage_cols:
        raise RuntimeError(f"Leakage columns in features: {leakage_cols[:20]}")

    print(f"Rows: {len(df)}")
    print(f"Columns: {df.shape[1]}")
    print(f"Raw POW feature columns: {len(raw_feature_cols)}")
    print(f"Final feature columns: {len(final_feature_cols)}")

    class_data, X_class, y_class = prepare_classification_data(
        df=df,
        raw_feature_cols=raw_feature_cols,
        target_col=args.class_target,
        min_windows_per_subject=args.min_windows_per_subject,
        feature_mode=args.feature_mode,
    )

    reg_data, X_reg, y_reg = prepare_regression_data(
        df=df,
        raw_feature_cols=raw_feature_cols,
        target_col=args.reg_target,
        min_windows_per_subject=args.min_windows_per_subject,
        feature_mode=args.feature_mode,
    )

    class_data_info = {
        "rows": len(class_data),
        "subjects": class_data["subject_id"].nunique(),
        "records": class_data["record_id"].nunique(),
        "sources": json.dumps(class_data["source"].value_counts().to_dict(), ensure_ascii=False),
        "class_distribution": json.dumps(class_data[args.class_target].value_counts().sort_index().to_dict(), ensure_ascii=False),
        "min_windows_per_subject": args.min_windows_per_subject,
        "feature_mode": args.feature_mode,
    }

    reg_data_info = {
        "rows": len(reg_data),
        "subjects": reg_data["subject_id"].nunique(),
        "records": reg_data["record_id"].nunique(),
        "sources": json.dumps(reg_data["source"].value_counts().to_dict(), ensure_ascii=False),
        "target_mean": float(reg_data[args.reg_target].mean()),
        "target_std": float(reg_data[args.reg_target].std()),
        "target_min": float(reg_data[args.reg_target].min()),
        "target_median": float(reg_data[args.reg_target].median()),
        "target_max": float(reg_data[args.reg_target].max()),
        "min_windows_per_subject": args.min_windows_per_subject,
        "feature_mode": args.feature_mode,
    }

    print("\nClassification data:")
    print(pd.DataFrame([class_data_info]).to_string(index=False))

    print("\nRegression data:")
    print(pd.DataFrame([reg_data_info]).to_string(index=False))

    clf_models = make_classification_models(fast=args.fast)
    reg_models = make_regression_models(fast=args.fast)

    print(f"\nClassification models: {list(clf_models.keys())}")
    print(f"Regression models: {list(reg_models.keys())}")

    class_metrics_rows: List[Dict[str, Any]] = []
    class_pred_parts: List[pd.DataFrame] = []

    reg_metrics_rows: List[Dict[str, Any]] = []
    reg_pred_parts: List[pd.DataFrame] = []

    if not args.skip_random:
        rows, preds = run_random_split_classification(
            data=class_data,
            X=X_class,
            y=y_class,
            target_col=args.class_target,
            models=clf_models,
            test_size=args.test_size,
        )
        class_metrics_rows.extend(rows)
        class_pred_parts.extend(preds)

        rows, preds = run_random_split_regression(
            data=reg_data,
            X=X_reg,
            y=y_reg,
            target_col=args.reg_target,
            models=reg_models,
            test_size=args.test_size,
        )
        reg_metrics_rows.extend(rows)
        reg_pred_parts.extend(preds)

    if not args.skip_groupkfold:
        rows, preds = run_groupkfold_classification(
            data=class_data,
            X=X_class,
            y=y_class,
            target_col=args.class_target,
            models=clf_models,
            n_splits=args.n_splits,
        )
        class_metrics_rows.extend(rows)
        class_pred_parts.extend(preds)

        rows, preds = run_groupkfold_regression(
            data=reg_data,
            X=X_reg,
            y=y_reg,
            target_col=args.reg_target,
            models=reg_models,
            n_splits=args.n_splits,
        )
        reg_metrics_rows.extend(rows)
        reg_pred_parts.extend(preds)

    if not args.skip_cross_source:
        rows, preds = run_cross_source_classification(
            data=class_data,
            X=X_class,
            y=y_class,
            target_col=args.class_target,
            models=clf_models,
            no_subject_overlap=False,
        )
        class_metrics_rows.extend(rows)
        class_pred_parts.extend(preds)

        rows, preds = run_cross_source_regression(
            data=reg_data,
            X=X_reg,
            y=y_reg,
            target_col=args.reg_target,
            models=reg_models,
            no_subject_overlap=False,
        )
        reg_metrics_rows.extend(rows)
        reg_pred_parts.extend(preds)

        if args.enable_cross_source_no_overlap:
            rows, preds = run_cross_source_classification(
                data=class_data,
                X=X_class,
                y=y_class,
                target_col=args.class_target,
                models=clf_models,
                no_subject_overlap=True,
            )
            class_metrics_rows.extend(rows)
            class_pred_parts.extend(preds)

            rows, preds = run_cross_source_regression(
                data=reg_data,
                X=X_reg,
                y=y_reg,
                target_col=args.reg_target,
                models=reg_models,
                no_subject_overlap=True,
            )
            reg_metrics_rows.extend(rows)
            reg_pred_parts.extend(preds)

    class_metrics = pd.DataFrame(class_metrics_rows)
    reg_metrics = pd.DataFrame(reg_metrics_rows)

    class_preds = pd.concat(class_pred_parts, ignore_index=True) if class_pred_parts else pd.DataFrame()
    reg_preds = pd.concat(reg_pred_parts, ignore_index=True) if reg_pred_parts else pd.DataFrame()

    class_agg = aggregate_metrics(class_metrics, task="classification")
    reg_agg = aggregate_metrics(reg_metrics, task="regression")

    class_metrics_path = processed_dir / f"{args.output_prefix}_classification_metrics.csv"
    reg_metrics_path = processed_dir / f"{args.output_prefix}_regression_metrics.csv"
    class_agg_path = processed_dir / f"{args.output_prefix}_classification_metrics_agg.csv"
    reg_agg_path = processed_dir / f"{args.output_prefix}_regression_metrics_agg.csv"

    class_preds_path = processed_dir / f"{args.output_prefix}_classification_predictions.parquet"
    reg_preds_path = processed_dir / f"{args.output_prefix}_regression_predictions.parquet"

    class_metrics.to_csv(class_metrics_path, index=False, encoding="utf-8-sig")
    reg_metrics.to_csv(reg_metrics_path, index=False, encoding="utf-8-sig")
    class_agg.to_csv(class_agg_path, index=False, encoding="utf-8-sig")
    reg_agg.to_csv(reg_agg_path, index=False, encoding="utf-8-sig")

    if not class_preds.empty:
        class_preds.to_parquet(class_preds_path, index=False)
    if not reg_preds.empty:
        reg_preds.to_parquet(reg_preds_path, index=False)

    figures: List[Path] = []
    figures.extend(plot_classification_summary(class_agg, fig_dir))
    figures.extend(plot_regression_summary(reg_agg, fig_dir))

    cm_path = plot_confusion_matrix_for_best(class_preds, class_agg, fig_dir)
    if cm_path is not None:
        figures.append(cm_path)

    scat_path = plot_regression_scatter_for_best(reg_preds, reg_agg, fig_dir)
    if scat_path is not None:
        figures.append(scat_path)

    report_path = reports_dir / f"{args.output_prefix}_report.md"

    make_report(
        report_path=report_path,
        dataset_path=dataset_path,
        feature_mode=args.feature_mode,
        raw_feature_cols=raw_feature_cols,
        final_feature_count=len(final_feature_cols),
        class_metrics=class_metrics,
        class_agg=class_agg,
        reg_metrics=reg_metrics,
        reg_agg=reg_agg,
        class_data_info=class_data_info,
        reg_data_info=reg_data_info,
        figures=figures,
        root=root,
    )

    print("\nSaved:")
    print(f"  {class_metrics_path}")
    print(f"  {reg_metrics_path}")
    print(f"  {class_agg_path}")
    print(f"  {reg_agg_path}")
    print(f"  {class_preds_path}")
    print(f"  {reg_preds_path}")
    print(f"  {report_path}")
    print(f"  figures: {fig_dir}")

    print("\nClassification aggregated metrics:")
    print(class_agg.to_string(index=False) if not class_agg.empty else "empty")

    print("\nRegression aggregated metrics:")
    print(reg_agg.to_string(index=False) if not reg_agg.empty else "empty")

    print("\nDone.")


if __name__ == "__main__":
    main()