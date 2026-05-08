# -*- coding: utf-8 -*-
"""
09_train_multi_pm_baselines.py

Быстрый baseline для предсказания всех PM-метрик Emotiv.

Идея:
    Вместо одного target_main = PM.Focus.Scaled обучаем отдельную регрессионную
    модель для каждой PM.*.Scaled метрики:

        PM.Attention.Scaled
        PM.Engagement.Scaled
        PM.Excitement.Scaled
        PM.Stress.Scaled
        PM.Relaxation.Scaled
        PM.Interest.Scaled
        PM.Focus.Scaled

Важно:
    PM.* используются только как target.
    PM.* не используются как input features, чтобы не было утечки.

Поддерживаемые feature_set:
    pow
    eeg
    pow_plus_eeg

Поддерживаемые feature_mode для POW:
    raw_pow
    log_pow
    raw_plus_log_pow

Логирование:
    Каждый запуск сохраняется в отдельную папку:
        reports/runs/<run_id>/

Внутри:
    train.log
    config.json
    target_fold_metrics.csv
    target_metrics_agg.csv
    target_summary.csv
    predictions.parquet
    report.md

Пример быстрого запуска:
    D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\09_train_multi_pm_baselines.py ^
      --dataset data\\processed\\windowed_eeg_pm_dataset_w10.parquet ^
      --feature-set pow_plus_eeg ^
      --feature-mode log_pow ^
      --models hgb,lgbm ^
      --validation groupkfold ^
      --run-name multi_pm_pow_plus_eeg_w10_log

С cross-source no-overlap:
    D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\09_train_multi_pm_baselines.py ^
      --dataset data\\processed\\windowed_eeg_pm_dataset_w10.parquet ^
      --feature-set pow_plus_eeg ^
      --feature-mode log_pow ^
      --models hgb,lgbm ^
      --validation groupkfold ^
      --enable-cross-source-no-overlap ^
      --run-name multi_pm_pow_plus_eeg_w10_log_xsource
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = None
    spearmanr = None


RANDOM_STATE = 42

PM_TARGET_BASES = [
    "PM.Attention.Scaled",
    "PM.Engagement.Scaled",
    "PM.Excitement.Scaled",
    "PM.Stress.Scaled",
    "PM.Relaxation.Scaled",
    "PM.Interest.Scaled",
    "PM.Focus.Scaled",
]


@dataclass
class RunConfig:
    root: str
    dataset: str
    run_name: str
    run_id: str
    feature_set: str
    feature_mode: str
    models: str
    validation: str
    enable_cross_source_no_overlap: bool
    n_splits: int
    test_size: float
    min_windows_per_subject: int
    max_rows: Optional[int]
    save_predictions: bool
    seed: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("multi_pm_baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def df_to_markdown_safe(df: pd.DataFrame, index: bool = True) -> str:
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return df.to_string(index=index)


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def infer_pow_feature_cols(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if c.startswith("POW.") and pd.api.types.is_numeric_dtype(df[c])
    ]


def infer_eeg_feature_cols(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if c.startswith("EEG.") and "__" in c and pd.api.types.is_numeric_dtype(df[c])
    ]


def transform_pow_features(
    df: pd.DataFrame,
    pow_cols: List[str],
    feature_mode: str,
) -> pd.DataFrame:
    if not pow_cols:
        return pd.DataFrame(index=df.index)

    x_raw = df[pow_cols].copy()

    if feature_mode == "raw_pow":
        return x_raw

    if feature_mode == "log_pow":
        x_log = np.log1p(x_raw.clip(lower=0))
        x_log.columns = [f"log1p_{c}" for c in pow_cols]
        return x_log

    if feature_mode == "raw_plus_log_pow":
        x_log = np.log1p(x_raw.clip(lower=0))
        x_log.columns = [f"log1p_{c}" for c in pow_cols]
        return pd.concat([x_raw, x_log], axis=1)

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def build_feature_frame(
    df: pd.DataFrame,
    feature_set: str,
    feature_mode: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    pow_cols = infer_pow_feature_cols(df)
    eeg_cols = infer_eeg_feature_cols(df)

    parts = []

    if feature_set in {"pow", "pow_plus_eeg"}:
        if not pow_cols:
            raise RuntimeError("feature_set requires POW features, but no POW.* columns found.")
        parts.append(transform_pow_features(df, pow_cols, feature_mode))

    if feature_set in {"eeg", "pow_plus_eeg"}:
        if not eeg_cols:
            raise RuntimeError("feature_set requires EEG features, but no EEG.* feature columns found.")
        parts.append(df[eeg_cols].copy())

    if not parts:
        raise RuntimeError(f"No features selected for feature_set={feature_set}")

    x = pd.concat(parts, axis=1)

    leakage_cols = [
        c for c in x.columns
        if c.startswith("PM.") or c.startswith("target_") or c.startswith("label_")
    ]
    if leakage_cols:
        raise RuntimeError(f"Leakage columns found in X: {leakage_cols[:20]}")

    info = {
        "pow_available": len(pow_cols),
        "eeg_available": len(eeg_cols),
        "features_used": x.shape[1],
    }

    return x, info


def find_pm_target_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Возвращает словарь:
        short_target_name -> actual_column_name

    Предпочитаем агрегированную колонку __mean, потому что именно она
    соответствует целевому значению PM в окне.
    """
    targets = {}

    for base in PM_TARGET_BASES:
        short_name = (
            base.replace("PM.", "")
            .replace(".Scaled", "")
            .lower()
        )

        candidates = [
            f"{base}__mean",
            f"{base}__last",
            base,
        ]

        found = None
        for c in candidates:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                found = c
                break

        if found is not None:
            targets[short_name] = found

    return targets


def make_models(model_names: List[str]) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    if "ridge" in model_names:
        models["ridge_robust"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )

    if "hgb" in model_names:
        models["hgb_reg"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=200,
                        max_leaf_nodes=31,
                        l2_regularization=0.01,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    if "rf" in model_names:
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

    if "lgbm" in model_names:
        try:
            from lightgbm import LGBMRegressor

            models["lgbm_reg"] = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        LGBMRegressor(
                            n_estimators=500,
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

    if not models:
        raise RuntimeError(
            "No models were created. Check --models. "
            "Allowed: ridge,hgb,lgbm,rf"
        )

    return models


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

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


def filter_target_data(
    df: pd.DataFrame,
    x: pd.DataFrame,
    target_col: str,
    min_windows_per_subject: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    data = df.copy()
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data[data[target_col].notna()].copy()

    counts = data["subject_id"].value_counts()
    keep_subjects = counts[counts >= min_windows_per_subject].index
    data = data[data["subject_id"].isin(keep_subjects)].copy()

    x_target = x.loc[data.index].copy()
    y = data[target_col].to_numpy(dtype=float)

    return data, x_target, y


def make_groupkfold_splits(
    data: pd.DataFrame,
    n_splits: int,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    groups = data["subject_id"].astype(str).to_numpy()
    n_splits_eff = min(n_splits, len(np.unique(groups)))

    gkf = GroupKFold(n_splits=n_splits_eff)
    dummy_y = np.zeros(len(data))

    splits = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(data, dummy_y, groups=groups), start=1):
        splits.append((train_idx, test_idx, f"groupkfold_subject_fold_{fold}"))

    return splits


def make_random_split(
    data: pd.DataFrame,
    test_size: float,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    idx = np.arange(len(data))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )
    return [(train_idx, test_idx, "random_split")]


def make_cross_source_no_overlap_splits(
    data: pd.DataFrame,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    sources = sorted(data["source"].dropna().unique().tolist())
    splits = []

    if len(sources) < 2:
        return splits

    for train_source in sources:
        for test_source in sources:
            if train_source == test_source:
                continue

            train_mask = data["source"] == train_source
            test_mask = data["source"] == test_source

            train_subjects = set(data.loc[train_mask, "subject_id"].astype(str))
            test_mask = test_mask & ~data["subject_id"].astype(str).isin(train_subjects)

            train_idx = np.where(train_mask.to_numpy())[0]
            test_idx = np.where(test_mask.to_numpy())[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            name = f"cross_source_no_overlap_train_{train_source}_test_{test_source}"
            splits.append((train_idx, test_idx, name))

    return splits


def run_one_target(
    target_name: str,
    target_col: str,
    df: pd.DataFrame,
    x_all: pd.DataFrame,
    models: Dict[str, Any],
    validation: str,
    enable_cross_source_no_overlap: bool,
    n_splits: int,
    test_size: float,
    min_windows_per_subject: int,
    save_predictions: bool,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data, x, y = filter_target_data(
        df=df,
        x=x_all,
        target_col=target_col,
        min_windows_per_subject=min_windows_per_subject,
    )

    logger.info(
        "Target=%s | column=%s | rows=%d | subjects=%d | records=%d | y_mean=%.6f | y_std=%.6f",
        target_name,
        target_col,
        len(data),
        data["subject_id"].nunique(),
        data["record_id"].nunique(),
        float(np.mean(y)),
        float(np.std(y)),
    )

    if validation == "groupkfold":
        splits = make_groupkfold_splits(data, n_splits=n_splits)
    elif validation == "random":
        splits = make_random_split(data, test_size=test_size)
    else:
        raise ValueError(f"Unknown validation: {validation}")

    if enable_cross_source_no_overlap:
        splits.extend(make_cross_source_no_overlap_splits(data))

    metrics_rows = []
    pred_parts = []

    for split_id, (train_idx, test_idx, split_name) in enumerate(splits, start=1):
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        train_subjects = data.iloc[train_idx]["subject_id"].nunique()
        test_subjects = data.iloc[test_idx]["subject_id"].nunique()

        for model_name, model in models.items():
            logger.info(
                "Target=%s | split=%s | model=%s | n_train=%d | n_test=%d",
                target_name,
                split_name,
                model_name,
                len(train_idx),
                len(test_idx),
            )

            t0 = time.time()

            estimator = clone(model)
            estimator.fit(x_train, y_train)
            y_pred = estimator.predict(x_test)

            elapsed = time.time() - t0

            m = regression_metrics(y_test, y_pred)

            row = {
                "target_name": target_name,
                "target_col": target_col,
                "validation": split_name,
                "split_id": split_id,
                "model": model_name,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_train_subjects": train_subjects,
                "n_test_subjects": test_subjects,
                "elapsed_s": elapsed,
            }
            row.update(m)
            metrics_rows.append(row)

            logger.info(
                "Target=%s | split=%s | model=%s | RMSE=%.6f | R2=%.6f | Spearman=%.6f | elapsed=%.1fs",
                target_name,
                split_name,
                model_name,
                row["rmse"],
                row["r2"],
                row["spearman"],
                elapsed,
            )

            if save_predictions:
                pred = data.iloc[test_idx][
                    ["record_id", "source", "subject_id", "day", target_col]
                ].copy()
                pred["target_name"] = target_name
                pred["target_col"] = target_col
                pred["validation"] = split_name
                pred["split_id"] = split_id
                pred["model"] = model_name
                pred["y_true"] = y_test
                pred["y_pred"] = y_pred
                pred_parts.append(pred)

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()

    return metrics_df, preds_df


def aggregate_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    metric_cols = ["mae", "rmse", "r2", "pearson", "spearman"]

    rows = []

    for (target_name, validation, model), g in metrics_df.groupby(
        ["target_name", "validation", "model"],
        dropna=False,
    ):
        row = {
            "target_name": target_name,
            "validation": validation,
            "model": model,
            "folds": len(g),
            "n_test_total": int(g["n_test"].sum()),
            "elapsed_s_total": float(g["elapsed_s"].sum()),
        }

        for c in metric_cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_std"] = float(g[c].std()) if len(g) > 1 else 0.0
            row[f"{c}_min"] = float(g[c].min())
            row[f"{c}_max"] = float(g[c].max())

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["validation", "target_name", "model"]
    ).reset_index(drop=True)


def make_target_summary(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Делает компактную таблицу: лучший model для каждого target на GroupKFold.
    """
    if agg_df.empty:
        return pd.DataFrame()

    group_df = agg_df[agg_df["validation"].str.startswith("groupkfold_subject", na=False)].copy()

    # Если валидация называется fold-wise, агрегатор оставляет отдельные fold names.
    # Для удобства соберем среднее по всем groupkfold folds для target/model.
    if group_df.empty:
        group_df = agg_df.copy()

    metric_cols = ["mae_mean", "rmse_mean", "r2_mean", "pearson_mean", "spearman_mean"]

    compact_rows = []

    for (target_name, model), g in group_df.groupby(["target_name", "model"], dropna=False):
        row = {
            "target_name": target_name,
            "model": model,
        }
        for c in metric_cols:
            if c in g.columns:
                row[c] = float(g[c].mean())
        compact_rows.append(row)

    compact = pd.DataFrame(compact_rows)

    if compact.empty:
        return compact

    # Основной ранжирующий критерий: spearman выше, rmse ниже.
    compact = compact.sort_values(
        ["target_name", "spearman_mean", "r2_mean", "rmse_mean"],
        ascending=[True, False, False, True],
    )

    best = compact.groupby("target_name", as_index=False).head(1)
    best = best.sort_values(["spearman_mean", "r2_mean"], ascending=False).reset_index(drop=True)

    return best


def plot_target_spearman(summary_df: pd.DataFrame, figures_dir: Path) -> Optional[Path]:
    if summary_df.empty or "spearman_mean" not in summary_df.columns:
        return None

    path = figures_dir / "target_best_groupkfold_spearman.png"

    s = summary_df.sort_values("spearman_mean", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(s["target_name"], s["spearman_mean"])
    plt.title("Best GroupKFold Spearman by PM target")
    plt.xlabel("PM target")
    plt.ylabel("Spearman")
    plt.xticks(rotation=45, ha="right")
    save_plot(path)

    return path


def plot_target_r2(summary_df: pd.DataFrame, figures_dir: Path) -> Optional[Path]:
    if summary_df.empty or "r2_mean" not in summary_df.columns:
        return None

    path = figures_dir / "target_best_groupkfold_r2.png"

    s = summary_df.sort_values("r2_mean", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(s["target_name"], s["r2_mean"])
    plt.title("Best GroupKFold R2 by PM target")
    plt.xlabel("PM target")
    plt.ylabel("R2")
    plt.xticks(rotation=45, ha="right")
    save_plot(path)

    return path


def make_report(
    report_path: Path,
    config: RunConfig,
    dataset_info: Dict[str, Any],
    feature_info: Dict[str, int],
    targets: Dict[str, str],
    metrics_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    figures: List[Path],
    run_dir: Path,
) -> None:
    lines = []

    lines.append("# Multi-PM baseline report")
    lines.append("")
    lines.append(f"Run ID: `{config.run_id}`")
    lines.append(f"Run directory: `{run_dir}`")
    lines.append("")

    lines.append("## Goal")
    lines.append("")
    lines.append(
        "This experiment trains separate regression baselines for all available "
        "`PM.*.Scaled` metrics. PM columns are used only as weak targets, not as input features."
    )
    lines.append("")

    lines.append("## Configuration")
    lines.append("")
    cfg_df = pd.DataFrame([asdict(config)]).T.reset_index()
    cfg_df.columns = ["parameter", "value"]
    lines.append(df_to_markdown_safe(cfg_df, index=False))
    lines.append("")

    lines.append("## Dataset info")
    lines.append("")
    lines.append(df_to_markdown_safe(pd.DataFrame([dataset_info]), index=False))
    lines.append("")

    lines.append("## Feature info")
    lines.append("")
    lines.append(df_to_markdown_safe(pd.DataFrame([feature_info]), index=False))
    lines.append("")

    lines.append("## PM targets")
    lines.append("")
    target_df = pd.DataFrame(
        [{"target_name": k, "column": v} for k, v in targets.items()]
    )
    lines.append(df_to_markdown_safe(target_df, index=False))
    lines.append("")

    lines.append("## Best target summary")
    lines.append("")
    if summary_df.empty:
        lines.append("_No summary._")
    else:
        lines.append(df_to_markdown_safe(summary_df, index=False))
    lines.append("")

    lines.append("## Aggregated metrics")
    lines.append("")
    if agg_df.empty:
        lines.append("_No aggregated metrics._")
    else:
        lines.append(df_to_markdown_safe(agg_df, index=False))
    lines.append("")

    lines.append("## Fold metrics preview")
    lines.append("")
    if metrics_df.empty:
        lines.append("_No fold metrics._")
    else:
        lines.append(df_to_markdown_safe(metrics_df.head(120), index=False))
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    for fig in figures:
        try:
            rel = fig.relative_to(run_dir)
        except Exception:
            rel = fig
        lines.append(f"- `{rel}`")
    lines.append("")

    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("1. PM metrics are treated as weak labels produced by Emotiv, not as expert ground truth.")
    lines.append("2. PM columns are excluded from X to avoid direct target leakage.")
    lines.append("3. The main comparison criterion is GroupKFold by subject_id.")
    lines.append("4. Spearman is important because it measures whether the model preserves the ordering of cognitive state intensity.")
    lines.append("5. Targets with higher GroupKFold Spearman/R2 are better candidates for the main project target.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default=r"D:\PycharmProjects\eeg-cognitive-state-nir")
    parser.add_argument("--dataset", type=str, default=r"data\processed\windowed_eeg_pm_dataset_w10.parquet")
    parser.add_argument("--run-name", type=str, default="multi_pm_baseline")
    parser.add_argument("--feature-set", type=str, choices=["pow", "eeg", "pow_plus_eeg"], default="pow_plus_eeg")
    parser.add_argument("--feature-mode", type=str, choices=["raw_pow", "log_pow", "raw_plus_log_pow"], default="log_pow")
    parser.add_argument("--models", type=str, default="hgb,lgbm", help="Comma-separated: ridge,hgb,lgbm,rf")
    parser.add_argument("--validation", type=str, choices=["groupkfold", "random"], default="groupkfold")
    parser.add_argument("--enable-cross-source-no-overlap", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-windows-per-subject", type=int, default=30)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)

    args = parser.parse_args()

    np.random.seed(args.seed)

    root = Path(args.root).resolve()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{args.run_name}_{args.feature_set}_{args.feature_mode}"
    run_id = run_id.replace(" ", "_")

    run_dir = root / "reports" / "runs" / run_id
    figures_dir = run_dir / "figures"

    ensure_dir(run_dir)
    ensure_dir(figures_dir)

    logger = setup_logging(run_dir)

    config = RunConfig(
        root=str(root),
        dataset=str(dataset_path),
        run_name=args.run_name,
        run_id=run_id,
        feature_set=args.feature_set,
        feature_mode=args.feature_mode,
        models=args.models,
        validation=args.validation,
        enable_cross_source_no_overlap=args.enable_cross_source_no_overlap,
        n_splits=args.n_splits,
        test_size=args.test_size,
        min_windows_per_subject=args.min_windows_per_subject,
        max_rows=args.max_rows,
        save_predictions=args.save_predictions,
        seed=args.seed,
    )

    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("=" * 80)
    logger.info("Multi-PM baseline")
    logger.info("=" * 80)
    logger.info("Run directory: %s", run_dir)
    logger.info("Dataset: %s", dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_parquet(dataset_path)

    if args.max_rows is not None and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=args.seed).reset_index(drop=True)
        logger.info("Sampled max_rows=%d", args.max_rows)

    required = ["record_id", "source", "subject_id", "day"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    dataset_info = {
        "rows": len(df),
        "columns": df.shape[1],
        "records": df["record_id"].nunique(),
        "subjects": df["subject_id"].nunique(),
        "sources": json.dumps(df["source"].value_counts().to_dict(), ensure_ascii=False),
    }

    logger.info("Dataset info: %s", dataset_info)

    targets = find_pm_target_columns(df)
    if not targets:
        raise RuntimeError(
            "No PM target columns found. Expected columns like PM.Focus.Scaled__mean."
        )

    logger.info("Found PM targets: %s", targets)

    x_all, feature_info = build_feature_frame(
        df=df,
        feature_set=args.feature_set,
        feature_mode=args.feature_mode,
    )

    feature_info = {
        **feature_info,
        "feature_set": args.feature_set,
        "feature_mode": args.feature_mode,
    }

    logger.info("Feature info: %s", feature_info)

    model_names = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    models = make_models(model_names)

    logger.info("Models: %s", list(models.keys()))

    all_metrics = []
    all_predictions = []

    for target_name, target_col in targets.items():
        logger.info("-" * 80)
        logger.info("Processing target: %s (%s)", target_name, target_col)

        metrics_df, preds_df = run_one_target(
            target_name=target_name,
            target_col=target_col,
            df=df,
            x_all=x_all,
            models=models,
            validation=args.validation,
            enable_cross_source_no_overlap=args.enable_cross_source_no_overlap,
            n_splits=args.n_splits,
            test_size=args.test_size,
            min_windows_per_subject=args.min_windows_per_subject,
            save_predictions=args.save_predictions,
            logger=logger,
        )

        all_metrics.append(metrics_df)

        if args.save_predictions and not preds_df.empty:
            all_predictions.append(preds_df)

    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    agg_df = aggregate_metrics(metrics_df)
    summary_df = make_target_summary(agg_df)

    metrics_path = run_dir / "target_fold_metrics.csv"
    agg_path = run_dir / "target_metrics_agg.csv"
    summary_path = run_dir / "target_summary.csv"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    agg_df.to_csv(agg_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    predictions_path = None
    if args.save_predictions and all_predictions:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_path = run_dir / "predictions.parquet"
        predictions_df.to_parquet(predictions_path, index=False)

    figures = []

    fig = plot_target_spearman(summary_df, figures_dir)
    if fig is not None:
        figures.append(fig)

    fig = plot_target_r2(summary_df, figures_dir)
    if fig is not None:
        figures.append(fig)

    report_path = run_dir / "report.md"

    make_report(
        report_path=report_path,
        config=config,
        dataset_info=dataset_info,
        feature_info=feature_info,
        targets=targets,
        metrics_df=metrics_df,
        agg_df=agg_df,
        summary_df=summary_df,
        figures=figures,
        run_dir=run_dir,
    )

    logger.info("=" * 80)
    logger.info("Saved outputs")
    logger.info("=" * 80)
    logger.info("Run directory: %s", run_dir)
    logger.info("Fold metrics: %s", metrics_path)
    logger.info("Aggregated metrics: %s", agg_path)
    logger.info("Target summary: %s", summary_path)
    if predictions_path is not None:
        logger.info("Predictions: %s", predictions_path)
    logger.info("Report: %s", report_path)

    logger.info("Best target summary:\n%s", summary_df.to_string(index=False) if not summary_df.empty else "empty")
    logger.info("Done.")


if __name__ == "__main__":
    main()