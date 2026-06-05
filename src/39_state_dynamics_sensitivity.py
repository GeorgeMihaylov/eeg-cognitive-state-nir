from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline


PM_TARGETS = {
    "attention": "PM.Attention.Scaled__mean",
    "engagement": "PM.Engagement.Scaled__mean",
    "excitement": "PM.Excitement.Scaled__mean",
    "stress": "PM.Stress.Scaled__mean",
    "relaxation": "PM.Relaxation.Scaled__mean",
    "interest": "PM.Interest.Scaled__mean",
    "focus": "PM.Focus.Scaled__mean",
}

ID_CANDIDATES = {
    "source": ["source", "dataset", "data_source"],
    "subject_id": ["subject_id", "subject", "participant_id", "user_id"],
    "record_id": ["record_id", "record", "session_id", "file_id"],
    "window_id": ["window_id", "window_index", "window_idx"],
    "window_start": ["t_start", "window_start", "start_time", "start"],
    "window_end": ["t_end", "window_end", "end_time", "end"],
}

NON_FEATURE_KEYWORDS = [
    "pm.",
    "pm_",
    "label",
    "target",
    "class",
    "fold",
    "split",
    "source",
    "subject",
    "record",
    "session",
    "window",
    "file",
    "path",
    "timestamp",
    "datetime",
    "date",
    "time",
    "annotation",
    "marker",
]


@dataclass
class Config:
    dataset: Path
    output_dir: Path
    run_name: str
    rolling_windows: list[int]
    trend_thresholds: list[float]
    regression_targets: list[str]
    classification_targets: list[str]
    validation_modes: list[str]
    n_splits: int
    test_size: float
    feature_set: str
    max_features: int | None
    max_rows: int | None
    random_state: int
    fast: bool
    no_plots: bool


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("state_dynamics_sensitivity")


def read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)

    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def find_first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    colset = set(columns)

    for candidate in candidates:
        if candidate in colset:
            return candidate

    return None


def detect_id_columns(columns: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}

    for logical_name, candidates in ID_CANDIDATES.items():
        col = find_first_existing(columns, candidates)
        if col is not None:
            found[logical_name] = col

    return found


def detect_pm_columns(columns: list[str]) -> dict[str, str]:
    colset = set(columns)
    found: dict[str, str] = {}

    for short_name, expected_col in PM_TARGETS.items():
        if expected_col in colset:
            found[short_name] = expected_col
            continue

        candidates = [
            c
            for c in columns
            if short_name.lower() in c.lower()
            and "pm" in c.lower()
            and ("scaled" in c.lower() or "mean" in c.lower())
        ]

        if candidates:
            found[short_name] = candidates[0]

    return found


def choose_group_columns(id_cols: dict[str, str]) -> list[str]:
    group_cols = []

    if "source" in id_cols:
        group_cols.append(id_cols["source"])
    if "subject_id" in id_cols:
        group_cols.append(id_cols["subject_id"])
    if "record_id" in id_cols:
        group_cols.append(id_cols["record_id"])

    if not group_cols:
        raise ValueError("Need at least source/subject_id/record_id columns for dynamics computation.")

    return group_cols


def choose_sort_columns(id_cols: dict[str, str]) -> list[str]:
    if "window_start" in id_cols:
        return [id_cols["window_start"]]

    if "window_id" in id_cols:
        return [id_cols["window_id"]]

    return []


def is_feature_column(col: str, id_cols: dict[str, str], pm_cols: dict[str, str]) -> bool:
    if col in set(id_cols.values()):
        return False

    if col in set(pm_cols.values()):
        return False

    low = col.lower()

    if low.startswith("pm_"):
        return False

    for keyword in NON_FEATURE_KEYWORDS:
        if keyword in low:
            return False

    return True


def select_feature_columns(
    df: pd.DataFrame,
    id_cols: dict[str, str],
    pm_cols: dict[str, str],
    feature_set: str,
    max_features: int | None,
) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in numeric_cols if is_feature_column(c, id_cols=id_cols, pm_cols=pm_cols)]

    if feature_set == "numeric":
        selected = candidates

    elif feature_set == "pow":
        selected = [
            c for c in candidates
            if "pow" in c.lower()
            or "band" in c.lower()
            or "delta" in c.lower()
            or "theta" in c.lower()
            or "alpha" in c.lower()
            or "beta" in c.lower()
            or "gamma" in c.lower()
        ]

    elif feature_set == "eeg":
        selected = [
            c for c in candidates
            if "eeg" in c.lower()
            and "pow" not in c.lower()
            and "band" not in c.lower()
        ]

    elif feature_set == "pow_plus_eeg":
        selected = [
            c for c in candidates
            if "eeg" in c.lower()
            or "pow" in c.lower()
            or "band" in c.lower()
            or "delta" in c.lower()
            or "theta" in c.lower()
            or "alpha" in c.lower()
            or "beta" in c.lower()
            or "gamma" in c.lower()
        ]

        if len(selected) < 10:
            selected = candidates

    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    selected = list(dict.fromkeys(selected))

    if max_features is not None and len(selected) > max_features:
        variances = df[selected].var(numeric_only=True).sort_values(ascending=False)
        selected = variances.head(max_features).index.tolist()

    return selected


def add_pm_base_columns(df: pd.DataFrame, pm_cols: dict[str, str]) -> pd.DataFrame:
    out = df.copy()

    for name, col in pm_cols.items():
        out[f"pm_{name}"] = pd.to_numeric(out[col], errors="coerce")

    return out


def add_dynamics_for_params(
    df: pd.DataFrame,
    pm_names: list[str],
    id_cols: dict[str, str],
    rolling_window: int,
    trend_threshold: float,
) -> pd.DataFrame:
    group_cols = choose_group_columns(id_cols)
    sort_cols = choose_sort_columns(id_cols)

    sort_by = list(dict.fromkeys(group_cols + sort_cols))
    if sort_by:
        df = df.sort_values(sort_by).reset_index(drop=True)

    pieces = []

    for _, group in df.groupby(group_cols, dropna=False, sort=False):
        group = group.copy()

        for name in pm_names:
            pm_col = f"pm_{name}"

            slow = (
                group[pm_col]
                .rolling(
                    window=rolling_window,
                    min_periods=max(2, rolling_window // 2),
                    center=True,
                )
                .mean()
            )

            delta_next = group[pm_col].shift(-1) - group[pm_col]

            trend = pd.Series("stable", index=group.index, dtype="object")
            trend[delta_next > trend_threshold] = "up"
            trend[delta_next < -trend_threshold] = "down"
            trend[delta_next.isna()] = np.nan

            group[f"pm_{name}_slow"] = slow
            group[f"pm_{name}_fast"] = group[pm_col] - slow
            group[f"pm_{name}_delta_next"] = delta_next
            group[f"pm_{name}_trend_next"] = trend

        pieces.append(group)

    return pd.concat(pieces, ignore_index=True)


def make_regressor(random_state: int, fast: bool) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=80 if fast else 200,
                    learning_rate=0.06,
                    l2_regularization=0.01,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_classifier(random_state: int, fast: bool) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=80 if fast else 200,
                    learning_rate=0.06,
                    l2_regularization=0.01,
                    random_state=random_state,
                ),
            ),
        ]
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "pearson": np.nan,
            "spearman": np.nan,
        }

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
        "pearson": pd.Series(y_true).corr(pd.Series(y_pred), method="pearson"),
        "spearman": pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def split_indices(
    df: pd.DataFrame,
    validation_mode: str,
    id_cols: dict[str, str],
    test_size: float,
    n_splits: int,
    random_state: int,
):
    indices = np.arange(len(df))

    if validation_mode == "random_split":
        train_idx, val_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        yield 1, train_idx, val_idx
        return

    if validation_mode == "groupkfold_subject":
        if "subject_id" not in id_cols:
            raise ValueError("subject_id column is required for groupkfold_subject.")

        groups = df[id_cols["subject_id"]].astype(str).fillna("unknown").values
        unique_groups = np.unique(groups)
        actual_splits = min(n_splits, len(unique_groups))

        if actual_splits < 2:
            return

        splitter = GroupKFold(n_splits=actual_splits)

        for fold, (train_idx, val_idx) in enumerate(splitter.split(indices, groups=groups), start=1):
            yield fold, train_idx, val_idx

        return

    raise ValueError(f"Unsupported validation mode: {validation_mode}")


def evaluate_regression_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    id_cols: dict[str, str],
    validation_modes: list[str],
    n_splits: int,
    test_size: float,
    random_state: int,
    fast: bool,
) -> pd.DataFrame:
    keep_cols = feature_cols + [target_col] + [c for c in id_cols.values() if c in df.columns]
    task_df = df[keep_cols].copy()

    task_df[target_col] = pd.to_numeric(task_df[target_col], errors="coerce")
    task_df = task_df[np.isfinite(task_df[target_col])].reset_index(drop=True)

    rows = []

    if len(task_df) < 100:
        return pd.DataFrame()

    for validation_mode in validation_modes:
        for fold, train_idx, val_idx in split_indices(
            task_df,
            validation_mode=validation_mode,
            id_cols=id_cols,
            test_size=test_size,
            n_splits=n_splits,
            random_state=random_state,
        ):
            X_train = task_df.iloc[train_idx][feature_cols]
            y_train = task_df.iloc[train_idx][target_col].to_numpy(dtype=float)

            X_val = task_df.iloc[val_idx][feature_cols]
            y_val = task_df.iloc[val_idx][target_col].to_numpy(dtype=float)

            model = make_regressor(random_state=random_state, fast=fast)

            started = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - started

            started = time.perf_counter()
            pred = model.predict(X_val)
            predict_time = time.perf_counter() - started

            metrics = regression_metrics(y_val, pred)

            rows.append(
                {
                    "task_type": "regression",
                    "target": target_col,
                    "validation": validation_mode,
                    "fold": fold,
                    "model": "hgb_reg",
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "fit_time_sec": fit_time,
                    "predict_time_sec": predict_time,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def evaluate_classification_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    id_cols: dict[str, str],
    validation_modes: list[str],
    n_splits: int,
    test_size: float,
    random_state: int,
    fast: bool,
) -> pd.DataFrame:
    keep_cols = feature_cols + [target_col] + [c for c in id_cols.values() if c in df.columns]
    task_df = df[keep_cols].copy()

    task_df = task_df[task_df[target_col].isin(["up", "stable", "down"])].reset_index(drop=True)

    rows = []

    if len(task_df) < 100:
        return pd.DataFrame()

    for validation_mode in validation_modes:
        for fold, train_idx, val_idx in split_indices(
            task_df,
            validation_mode=validation_mode,
            id_cols=id_cols,
            test_size=test_size,
            n_splits=n_splits,
            random_state=random_state,
        ):
            y_train = task_df.iloc[train_idx][target_col].values
            y_val = task_df.iloc[val_idx][target_col].values

            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                continue

            X_train = task_df.iloc[train_idx][feature_cols]
            X_val = task_df.iloc[val_idx][feature_cols]

            model = make_classifier(random_state=random_state, fast=fast)

            started = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - started

            started = time.perf_counter()
            pred = model.predict(X_val)
            predict_time = time.perf_counter() - started

            metrics = classification_metrics(y_val, pred)

            rows.append(
                {
                    "task_type": "classification",
                    "target": target_col,
                    "validation": validation_mode,
                    "fold": fold,
                    "model": "hgb_clf",
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "fit_time_sec": fit_time,
                    "predict_time_sec": predict_time,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def aggregate_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame()

    group_cols = [
        "rolling_window",
        "trend_threshold",
        "task_type",
        "target",
        "validation",
        "model",
    ]

    metric_cols = [
        c
        for c in fold_df.columns
        if c not in group_cols
        and c not in {
            "fold",
            "n_train",
            "n_val",
            "fit_time_sec",
            "predict_time_sec",
        }
        and pd.api.types.is_numeric_dtype(fold_df[c])
    ]

    rows = []

    for keys, sub in fold_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["folds"] = int(sub["fold"].nunique())
        row["n_val_total"] = int(sub["n_val"].sum())
        row["fit_time_sec_sum"] = float(sub["fit_time_sec"].sum())

        for metric in metric_cols:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_std"] = sub[metric].std()
            row[f"{metric}_min"] = sub[metric].min()
            row[f"{metric}_max"] = sub[metric].max()

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["task_type", "target", "validation", "rolling_window", "trend_threshold"]
    ).reset_index(drop=True)


def build_best_params(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows = []

    group_cols = ["task_type", "target", "validation"]

    for keys, sub in summary_df.groupby(group_cols, dropna=False):
        task_type, target, validation = keys

        if task_type == "regression":
            metric = "r2_mean"
        else:
            metric = "balanced_accuracy_mean"

        if metric not in sub.columns:
            continue

        best = sub.sort_values(metric, ascending=False, na_position="last").iloc[0].to_dict()
        best["selection_metric"] = metric
        best["selection_value"] = best.get(metric, np.nan)
        rows.append(best)

    return pd.DataFrame(rows)


def build_stability_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows = []

    group_cols = ["task_type", "target", "validation"]

    for keys, sub in summary_df.groupby(group_cols, dropna=False):
        task_type, target, validation = keys

        if task_type == "regression":
            metric = "r2_mean"
        else:
            metric = "balanced_accuracy_mean"

        if metric not in sub.columns:
            continue

        values = pd.to_numeric(sub[metric], errors="coerce").dropna()
        if values.empty:
            continue

        rows.append(
            {
                "task_type": task_type,
                "target": target,
                "validation": validation,
                "metric": metric,
                "n_param_settings": int(len(values)),
                "metric_mean_across_params": values.mean(),
                "metric_std_across_params": values.std(),
                "metric_min_across_params": values.min(),
                "metric_max_across_params": values.max(),
                "metric_range_across_params": values.max() - values.min(),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["task_type", "target", "validation"]
    ).reset_index(drop=True)


def build_label_distribution(df: pd.DataFrame, classification_targets: list[str]) -> pd.DataFrame:
    rows = []

    for target_col in classification_targets:
        if target_col not in df.columns:
            continue

        counts = df[target_col].value_counts(dropna=False)
        total = counts.sum()

        for label, count in counts.items():
            rows.append(
                {
                    "target": target_col,
                    "label": str(label),
                    "count": int(count),
                    "fraction": float(count / total) if total else np.nan,
                }
            )

    return pd.DataFrame(rows)


def plot_sensitivity(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if summary_df.empty:
        return

    for task_type, metric_col, ylabel in [
        ("regression", "r2_mean", "R²"),
        ("classification", "balanced_accuracy_mean", "Balanced accuracy"),
    ]:
        sub = summary_df[
            (summary_df["task_type"] == task_type)
            & (summary_df["validation"] == "groupkfold_subject")
        ].copy()

        if sub.empty or metric_col not in sub.columns:
            continue

        for target, target_df in sub.groupby("target"):
            fig, ax = plt.subplots(figsize=(8, 5))

            for threshold, th_df in target_df.groupby("trend_threshold"):
                th_df = th_df.sort_values("rolling_window")
                ax.plot(
                    th_df["rolling_window"],
                    th_df[metric_col],
                    marker="o",
                    label=f"threshold={threshold}",
                )

            ax.set_title(f"{target}: {task_type} sensitivity")
            ax.set_xlabel("Rolling window")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            safe_target = str(target).replace("/", "_").replace("\\", "_")
            fig.tight_layout()
            fig.savefig(output_dir / f"sensitivity_{task_type}_{safe_target}.png", dpi=160)
            plt.close(fig)


def write_report(
    output_dir: Path,
    config: Config,
    dataset_info: dict,
    feature_cols: list[str],
    summary_df: pd.DataFrame,
    best_params: pd.DataFrame,
    stability: pd.DataFrame,
    label_distribution: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# State dynamics sensitivity report")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append(
        "Check whether the conclusions about slow/background and trend/change-direction states depend on one fixed rolling window or trend threshold."
    )
    lines.append("")
    lines.append("## Input")
    lines.append("")
    lines.append(f"- Dataset: `{config.dataset}`")
    lines.append(f"- Output dir: `{config.output_dir}`")
    lines.append(f"- Rows loaded: `{dataset_info['rows_loaded']}`")
    lines.append(f"- Rows used: `{dataset_info['rows_used']}`")
    lines.append(f"- Rolling windows: `{config.rolling_windows}`")
    lines.append(f"- Trend thresholds: `{config.trend_thresholds}`")
    lines.append(f"- Regression targets: `{config.regression_targets}`")
    lines.append(f"- Classification targets: `{config.classification_targets}`")
    lines.append(f"- Validation modes: `{config.validation_modes}`")
    lines.append(f"- Feature set: `{config.feature_set}`")
    lines.append(f"- Features used: `{len(feature_cols)}`")
    lines.append("")
    lines.append("## Best parameter settings")
    lines.append("")
    if not best_params.empty:
        display_cols = [
            c
            for c in [
                "task_type",
                "target",
                "validation",
                "rolling_window",
                "trend_threshold",
                "selection_metric",
                "selection_value",
                "r2_mean",
                "spearman_mean",
                "balanced_accuracy_mean",
                "macro_f1_mean",
            ]
            if c in best_params.columns
        ]
        lines.append(best_params[display_cols].to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No best parameter rows.")
    lines.append("")
    lines.append("## Stability across parameter settings")
    lines.append("")
    if not stability.empty:
        lines.append(stability.to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No stability rows.")
    lines.append("")
    lines.append("## Trend label distribution")
    lines.append("")
    if not label_distribution.empty:
        lines.append(label_distribution.to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No label distribution rows.")
    lines.append("")
    lines.append("## Working interpretation")
    lines.append("")
    lines.append("- Low metric range across parameters means the conclusion is stable.")
    lines.append("- If a target is strong only for one threshold/window, it should be treated cautiously.")
    lines.append("- For trend targets, label distribution must be checked because severe imbalance can inflate results.")
    lines.append("- The default project setting was rolling_window=5 and trend_threshold=0.02; this report tests whether that choice is fragile.")
    lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_list_arg(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_list_arg(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_list_arg(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for PM dynamics parameters."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/windowed_eeg_pm_dataset_w10.csv"),
        help="Input windowed EEG/PM dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/state_dynamics/sensitivity"),
        help="Output directory.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="state_dynamics_sensitivity",
        help="Run name.",
    )
    parser.add_argument(
        "--rolling-windows",
        type=str,
        default="3,5,7,10",
        help="Comma-separated rolling window values.",
    )
    parser.add_argument(
        "--trend-thresholds",
        type=str,
        default="0.01,0.02,0.03",
        help="Comma-separated trend thresholds.",
    )
    parser.add_argument(
        "--regression-targets",
        type=str,
        default="excitement,relaxation,stress,focus",
        help="Comma-separated PM targets for slow regression sensitivity.",
    )
    parser.add_argument(
        "--classification-targets",
        type=str,
        default="stress,focus,attention,interest,relaxation",
        help="Comma-separated PM targets for trend classification sensitivity.",
    )
    parser.add_argument(
        "--validation-modes",
        type=str,
        default="groupkfold_subject",
        help="Comma-separated validation modes: random_split,groupkfold_subject.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold splits.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Random split test size.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="pow_plus_eeg",
        choices=["numeric", "pow", "eeg", "pow_plus_eeg"],
        help="Feature selection mode.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=448,
        help="Optional max features selected by variance.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row sample before sensitivity loops.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster model settings.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plots.",
    )

    args = parser.parse_args()

    return Config(
        dataset=args.dataset,
        output_dir=args.output_dir,
        run_name=args.run_name,
        rolling_windows=parse_int_list_arg(args.rolling_windows),
        trend_thresholds=parse_float_list_arg(args.trend_thresholds),
        regression_targets=parse_list_arg(args.regression_targets),
        classification_targets=parse_list_arg(args.classification_targets),
        validation_modes=parse_list_arg(args.validation_modes),
        n_splits=args.n_splits,
        test_size=args.test_size,
        feature_set=args.feature_set,
        max_features=args.max_features,
        max_rows=args.max_rows,
        random_state=args.random_state,
        fast=args.fast,
        no_plots=args.no_plots,
    )


def main() -> None:
    logger = setup_logging()
    config = parse_args()

    config.dataset = config.dataset.resolve()
    config.output_dir = config.output_dir.resolve()
    figures_dir = config.output_dir / "figures"

    config.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("State dynamics sensitivity analysis")
    logger.info("=" * 80)
    logger.info("Dataset: %s", config.dataset)
    logger.info("Output dir: %s", config.output_dir)
    logger.info("Rolling windows: %s", config.rolling_windows)
    logger.info("Trend thresholds: %s", config.trend_thresholds)

    if not config.dataset.exists():
        raise FileNotFoundError(f"Dataset was not found: {config.dataset}")

    df = read_dataset(config.dataset)
    rows_loaded = len(df)

    columns = list(df.columns)
    id_cols = detect_id_columns(columns)
    pm_cols_all = detect_pm_columns(columns)

    required_pm_names = sorted(set(config.regression_targets + config.classification_targets))
    missing_pm = [name for name in required_pm_names if name not in pm_cols_all]

    if missing_pm:
        logger.warning("Missing PM targets will be skipped: %s", missing_pm)

    pm_cols = {
        name: col
        for name, col in pm_cols_all.items()
        if name in required_pm_names
    }

    if not pm_cols:
        raise ValueError("No requested PM columns were detected.")

    if config.max_rows is not None and len(df) > config.max_rows:
        df = df.sample(n=config.max_rows, random_state=config.random_state).reset_index(drop=True)
        logger.info("Sampled rows: %d", len(df))

    df = add_pm_base_columns(df, pm_cols=pm_cols)

    feature_cols = select_feature_columns(
        df=df,
        id_cols=id_cols,
        pm_cols=pm_cols,
        feature_set=config.feature_set,
        max_features=config.max_features,
    )

    if not feature_cols:
        raise ValueError("No feature columns were selected.")

    logger.info("Detected ID columns: %s", id_cols)
    logger.info("Detected PM columns: %s", pm_cols)
    logger.info("Selected features: %d", len(feature_cols))

    fold_parts = []
    label_distribution_parts = []

    for rolling_window in config.rolling_windows:
        for trend_threshold in config.trend_thresholds:
            logger.info(
                "Parameter setting: rolling_window=%s trend_threshold=%s",
                rolling_window,
                trend_threshold,
            )

            param_df = add_dynamics_for_params(
                df=df,
                pm_names=list(pm_cols.keys()),
                id_cols=id_cols,
                rolling_window=rolling_window,
                trend_threshold=trend_threshold,
            )

            regression_target_cols = [
                f"pm_{name}_slow"
                for name in config.regression_targets
                if f"pm_{name}_slow" in param_df.columns
            ]

            classification_target_cols = [
                f"pm_{name}_trend_next"
                for name in config.classification_targets
                if f"pm_{name}_trend_next" in param_df.columns
            ]

            label_dist = build_label_distribution(param_df, classification_target_cols)
            if not label_dist.empty:
                label_dist.insert(0, "rolling_window", rolling_window)
                label_dist.insert(1, "trend_threshold", trend_threshold)
                label_distribution_parts.append(label_dist)

            for target_col in regression_target_cols:
                logger.info("Regression target: %s", target_col)

                res = evaluate_regression_target(
                    df=param_df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    id_cols=id_cols,
                    validation_modes=config.validation_modes,
                    n_splits=config.n_splits,
                    test_size=config.test_size,
                    random_state=config.random_state,
                    fast=config.fast,
                )

                if not res.empty:
                    res.insert(0, "rolling_window", rolling_window)
                    res.insert(1, "trend_threshold", trend_threshold)
                    fold_parts.append(res)

            for target_col in classification_target_cols:
                logger.info("Classification target: %s", target_col)

                res = evaluate_classification_target(
                    df=param_df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    id_cols=id_cols,
                    validation_modes=config.validation_modes,
                    n_splits=config.n_splits,
                    test_size=config.test_size,
                    random_state=config.random_state,
                    fast=config.fast,
                )

                if not res.empty:
                    res.insert(0, "rolling_window", rolling_window)
                    res.insert(1, "trend_threshold", trend_threshold)
                    fold_parts.append(res)

    fold_df = pd.concat(fold_parts, ignore_index=True, sort=False) if fold_parts else pd.DataFrame()

    if fold_df.empty:
        raise RuntimeError("No sensitivity fold metrics were produced.")

    summary_df = aggregate_fold_metrics(fold_df)
    best_params = build_best_params(summary_df)
    stability = build_stability_table(summary_df)
    label_distribution = (
        pd.concat(label_distribution_parts, ignore_index=True, sort=False)
        if label_distribution_parts
        else pd.DataFrame()
    )

    fold_df.to_csv(config.output_dir / "sensitivity_fold_metrics.csv", index=False)
    summary_df.to_csv(config.output_dir / "sensitivity_summary.csv", index=False)
    best_params.to_csv(config.output_dir / "sensitivity_best_params.csv", index=False)
    stability.to_csv(config.output_dir / "sensitivity_stability.csv", index=False)
    label_distribution.to_csv(config.output_dir / "trend_label_distribution.csv", index=False)

    dataset_info = {
        "dataset": str(config.dataset),
        "rows_loaded": int(rows_loaded),
        "rows_used": int(len(df)),
        "id_columns": id_cols,
        "pm_columns": pm_cols,
        "rolling_windows": config.rolling_windows,
        "trend_thresholds": config.trend_thresholds,
        "regression_targets": config.regression_targets,
        "classification_targets": config.classification_targets,
        "validation_modes": config.validation_modes,
        "feature_set": config.feature_set,
        "n_features": int(len(feature_cols)),
        "max_rows": config.max_rows,
    }

    save_json(
        config.output_dir / "summary.json",
        {
            "run_name": config.run_name,
            "output_dir": str(config.output_dir),
            **dataset_info,
            "n_fold_rows": int(len(fold_df)),
            "n_summary_rows": int(len(summary_df)),
            "n_best_param_rows": int(len(best_params)),
            "n_stability_rows": int(len(stability)),
        },
    )

    if not config.no_plots:
        plot_sensitivity(summary_df, figures_dir)

    write_report(
        output_dir=config.output_dir,
        config=config,
        dataset_info=dataset_info,
        feature_cols=feature_cols,
        summary_df=summary_df,
        best_params=best_params,
        stability=stability,
        label_distribution=label_distribution,
    )

    logger.info("=" * 80)
    logger.info("Saved state dynamics sensitivity outputs")
    logger.info("=" * 80)
    logger.info("Fold metrics: %s", config.output_dir / "sensitivity_fold_metrics.csv")
    logger.info("Summary: %s", config.output_dir / "sensitivity_summary.csv")
    logger.info("Best params: %s", config.output_dir / "sensitivity_best_params.csv")
    logger.info("Stability: %s", config.output_dir / "sensitivity_stability.csv")
    logger.info("Trend label distribution: %s", config.output_dir / "trend_label_distribution.csv")
    logger.info("Report: %s", config.output_dir / "report.md")

    with pd.option_context("display.max_rows", 40, "display.max_columns", 20, "display.width", 180):
        logger.info("Best params:\n%s", best_params.to_string(index=False))

    logger.info("Done.")


if __name__ == "__main__":
    main()