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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


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
    targets: list[str]
    target_modes: list[str]
    validation_modes: list[str]
    models: list[str]
    feature_set: str
    trend_threshold: float
    rolling_window: int
    n_splits: int
    test_size: float
    random_state: int
    max_rows: int | None
    max_features: int | None
    fast: bool
    no_plots: bool


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("pm_dynamics_baselines")


def read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
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
        raise ValueError("Need at least source/subject_id/record_id columns for dynamics targets.")

    return group_cols


def choose_sort_columns(id_cols: dict[str, str]) -> list[str]:
    if "window_start" in id_cols:
        return [id_cols["window_start"]]
    if "window_id" in id_cols:
        return [id_cols["window_id"]]
    return []


def compute_group_dynamics(
    group: pd.DataFrame,
    pm_names: list[str],
    rolling_window: int,
    trend_threshold: float,
) -> pd.DataFrame:
    group = group.copy()

    for name in pm_names:
        col = f"pm_{name}"

        next_values = group[col].shift(-1)
        delta = next_values - group[col]

        group[f"pm_{name}_absolute"] = group[col]
        group[f"pm_{name}_delta_next"] = delta

        trend = pd.Series("stable", index=group.index, dtype="object")
        trend[delta > trend_threshold] = "up"
        trend[delta < -trend_threshold] = "down"
        trend[delta.isna()] = np.nan

        trend_code = pd.Series(0.0, index=group.index)
        trend_code[trend == "up"] = 1.0
        trend_code[trend == "down"] = -1.0
        trend_code[trend.isna()] = np.nan

        slow = (
            group[col]
            .rolling(window=rolling_window, min_periods=max(2, rolling_window // 2), center=True)
            .mean()
        )

        group[f"pm_{name}_trend_next"] = trend
        group[f"pm_{name}_trend_next_code"] = trend_code
        group[f"pm_{name}_slow"] = slow
        group[f"pm_{name}_fast"] = group[col] - slow
        group[f"pm_{name}_valid_transition"] = group[col].notna() & next_values.notna()

    return group


def add_pm_dynamics(
    df: pd.DataFrame,
    id_cols: dict[str, str],
    pm_cols: dict[str, str],
    rolling_window: int,
    trend_threshold: float,
) -> pd.DataFrame:
    out = df.copy()

    for short_name, col in pm_cols.items():
        out[f"pm_{short_name}"] = pd.to_numeric(out[col], errors="coerce")

    group_cols = choose_group_columns(id_cols)
    sort_cols = choose_sort_columns(id_cols)
    sort_by = list(dict.fromkeys(group_cols + sort_cols))

    if sort_by:
        out = out.sort_values(sort_by).reset_index(drop=True)

    pieces = []
    for _, group in out.groupby(group_cols, dropna=False, sort=False):
        pieces.append(
            compute_group_dynamics(
                group=group,
                pm_names=list(pm_cols.keys()),
                rolling_window=rolling_window,
                trend_threshold=trend_threshold,
            )
        )

    return pd.concat(pieces, ignore_index=True)


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
        # Simple variance-based feature preselection.
        variances = df[selected].var(numeric_only=True).sort_values(ascending=False)
        selected = variances.head(max_features).index.tolist()

    return selected


def make_regression_model(name: str, random_state: int, fast: bool):
    if name == "hgb_reg":
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

    if name == "rf_reg":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=80 if fast else 200,
                        max_depth=12 if fast else None,
                        min_samples_leaf=3,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if name == "robust_linear_reg":
        from sklearn.linear_model import Ridge

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", Ridge(alpha=1.0, random_state=random_state)),
            ]
        )

    raise ValueError(f"Unknown regression model: {name}")


def make_classification_model(name: str, random_state: int, fast: bool):
    if name == "hgb_clf":
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

    if name == "rf_clf":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=80 if fast else 200,
                        max_depth=12 if fast else None,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if name == "logistic_robust":
        from sklearn.linear_model import LogisticRegression

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown classification model: {name}")


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


def get_split_indices(
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
        splitter = GroupKFold(n_splits=actual_splits)

        for fold, (train_idx, val_idx) in enumerate(splitter.split(indices, groups=groups), start=1):
            yield fold, train_idx, val_idx
        return

    if validation_mode == "groupkfold_record":
        if "record_id" not in id_cols:
            raise ValueError("record_id column is required for groupkfold_record.")
        groups = df[id_cols["record_id"]].astype(str).fillna("unknown").values
        unique_groups = np.unique(groups)
        actual_splits = min(n_splits, len(unique_groups))
        splitter = GroupKFold(n_splits=actual_splits)

        for fold, (train_idx, val_idx) in enumerate(splitter.split(indices, groups=groups), start=1):
            yield fold, train_idx, val_idx
        return

    raise ValueError(f"Unsupported validation mode: {validation_mode}")


def get_cross_source_splits(df: pd.DataFrame, id_cols: dict[str, str]):
    if "source" not in id_cols:
        return []

    source_col = id_cols["source"]
    sources = sorted(df[source_col].dropna().astype(str).unique().tolist())

    splits = []
    if len(sources) < 2:
        return splits

    for train_source in sources:
        for test_source in sources:
            if train_source == test_source:
                continue

            train_idx = df.index[df[source_col].astype(str) == train_source].to_numpy()
            val_idx = df.index[df[source_col].astype(str) == test_source].to_numpy()

            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((f"cross_source_train_{train_source}_test_{test_source}", train_idx, val_idx))

    return splits


def prepare_task_dataframe(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    target_mode: str,
) -> tuple[pd.DataFrame, str, str]:
    if target_mode == "absolute":
        target_col = f"pm_{target}_absolute"
        task_type = "regression"
    elif target_mode == "delta":
        target_col = f"pm_{target}_delta_next"
        task_type = "regression"
    elif target_mode == "fast":
        target_col = f"pm_{target}_fast"
        task_type = "regression"
    elif target_mode == "slow":
        target_col = f"pm_{target}_slow"
        task_type = "regression"
    elif target_mode == "trend":
        target_col = f"pm_{target}_trend_next"
        task_type = "classification"
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    task_cols = feature_cols + [target_col]
    task_df = df[task_cols].copy()
    task_df["__orig_index"] = df.index

    if task_type == "classification":
        task_df = task_df[task_df[target_col].isin(["up", "stable", "down"])].copy()
    else:
        task_df[target_col] = pd.to_numeric(task_df[target_col], errors="coerce")
        task_df = task_df[np.isfinite(task_df[target_col])].copy()

    task_df = task_df.reset_index(drop=True)
    return task_df, target_col, task_type


def evaluate_one_split(
    task_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    task_type: str,
    model_name: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    random_state: int,
    fast: bool,
) -> dict:
    train_df = task_df.iloc[train_idx]
    val_df = task_df.iloc[val_idx]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols]
    y_val = val_df[target_col].values

    if task_type == "regression":
        model = make_regression_model(model_name, random_state=random_state, fast=fast)
    else:
        model = make_classification_model(model_name, random_state=random_state, fast=fast)

    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - started

    started = time.perf_counter()
    y_pred = model.predict(X_val)
    predict_time = time.perf_counter() - started

    if task_type == "regression":
        metrics = regression_metrics(y_val.astype(float), y_pred.astype(float))
    else:
        metrics = classification_metrics(y_val, y_pred)

    result = {
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "fit_time_sec": fit_time,
        "predict_time_sec": predict_time,
        **metrics,
    }

    return result


def aggregate_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["target", "target_mode", "task_type", "validation", "model"]

    metric_cols = [
        c for c in fold_df.columns
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
        row["predict_time_sec_sum"] = float(sub["predict_time_sec"].sum())

        for metric in metric_cols:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_std"] = sub[metric].std()
            row[f"{metric}_min"] = sub[metric].min()
            row[f"{metric}_max"] = sub[metric].max()

        rows.append(row)

    out = pd.DataFrame(rows)

    sort_cols = []
    if "r2_mean" in out.columns:
        sort_cols.append("r2_mean")
    if "balanced_accuracy_mean" in out.columns:
        sort_cols.append("balanced_accuracy_mean")
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=False, na_position="last")

    return out.reset_index(drop=True)


def plot_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    reg = summary_df[summary_df["task_type"] == "regression"].copy()
    if not reg.empty and "r2_mean" in reg.columns:
        pivot = reg.pivot_table(
            index="target",
            columns="target_mode",
            values="r2_mean",
            aggfunc="max",
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Best R² by target and target mode")
        ax.set_ylabel("R²")
        ax.set_xlabel("PM target")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "best_r2_by_target_mode.png", dpi=160)
        plt.close(fig)

    clf = summary_df[summary_df["task_type"] == "classification"].copy()
    if not clf.empty and "balanced_accuracy_mean" in clf.columns:
        pivot = clf.pivot_table(
            index="target",
            columns="validation",
            values="balanced_accuracy_mean",
            aggfunc="max",
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Best trend balanced accuracy by target and validation")
        ax.set_ylabel("Balanced accuracy")
        ax.set_xlabel("PM target")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "best_trend_balanced_accuracy.png", dpi=160)
        plt.close(fig)


def write_report(
    output_dir: Path,
    config: Config,
    dataset_info: dict,
    feature_cols: list[str],
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# PM dynamics baseline report")
    lines.append("")
    lines.append("## Input")
    lines.append("")
    lines.append(f"- Dataset: `{config.dataset}`")
    lines.append(f"- Output dir: `{config.output_dir}`")
    lines.append(f"- Run name: `{config.run_name}`")
    lines.append(f"- Rows loaded: `{dataset_info['rows_loaded']}`")
    lines.append(f"- Rows used after optional sampling: `{dataset_info['rows_used']}`")
    lines.append(f"- Feature set: `{config.feature_set}`")
    lines.append(f"- Features used: `{len(feature_cols)}`")
    lines.append(f"- Targets: `{config.targets}`")
    lines.append(f"- Target modes: `{config.target_modes}`")
    lines.append(f"- Validation modes: `{config.validation_modes}`")
    lines.append(f"- Models: `{config.models}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if not summary_df.empty:
        display_cols = [
            c for c in [
                "target",
                "target_mode",
                "task_type",
                "validation",
                "model",
                "folds",
                "n_val_total",
                "mae_mean",
                "rmse_mean",
                "r2_mean",
                "spearman_mean",
                "balanced_accuracy_mean",
                "macro_f1_mean",
            ]
            if c in summary_df.columns
        ]
        lines.append(summary_df[display_cols].to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No summary rows.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This run compares three main target formulations:")
    lines.append("")
    lines.append("- `absolute`: predict PM value at current window.")
    lines.append("- `delta`: predict PM(t+1) - PM(t).")
    lines.append("- `trend`: classify next change as `up`, `stable`, or `down`.")
    lines.append("")
    lines.append("Main questions:")
    lines.append("")
    lines.append("1. Which PM metrics are best predicted as absolute values?")
    lines.append("2. Which PM metrics are better represented by dynamics?")
    lines.append("3. How much does quality drop under subject-wise validation?")
    lines.append("4. Do delta/trend targets transfer better than absolute PM targets?")
    lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_list_arg(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train baselines for absolute PM, delta PM and trend targets."
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
        default=Path("reports/state_dynamics/pm_w10_baselines"),
        help="Output directory.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="pm_dynamics_baselines",
        help="Run name.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="all",
        help="Comma-separated PM targets or 'all'.",
    )
    parser.add_argument(
        "--target-modes",
        type=str,
        default="absolute,delta,trend",
        help="Comma-separated target modes: absolute,delta,trend,fast,slow.",
    )
    parser.add_argument(
        "--validation-modes",
        type=str,
        default="random_split,groupkfold_subject",
        help="Comma-separated validation modes: random_split,groupkfold_subject,groupkfold_record,cross_source.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="hgb_reg,hgb_clf",
        help="Comma-separated models. Regression: hgb_reg,rf_reg,robust_linear_reg. Classification: hgb_clf,rf_clf,logistic_robust.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="pow_plus_eeg",
        choices=["numeric", "pow", "eeg", "pow_plus_eeg"],
        help="Feature selection mode.",
    )
    parser.add_argument(
        "--trend-threshold",
        type=float,
        default=0.02,
        help="Trend threshold for PM delta.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Rolling window for fast/slow components.",
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
        "--random-state",
        type=int,
        default=42,
        help="Random state.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row sample after dynamics target construction.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Optional max features selected by variance.",
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

    if args.targets == "all":
        targets = list(PM_TARGETS.keys())
    else:
        targets = parse_list_arg(args.targets)

    target_modes = parse_list_arg(args.target_modes)
    validation_modes = parse_list_arg(args.validation_modes)
    models = parse_list_arg(args.models)

    return Config(
        dataset=args.dataset,
        output_dir=args.output_dir,
        run_name=args.run_name,
        targets=targets,
        target_modes=target_modes,
        validation_modes=validation_modes,
        models=models,
        feature_set=args.feature_set,
        trend_threshold=args.trend_threshold,
        rolling_window=args.rolling_window,
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_state,
        max_rows=args.max_rows,
        max_features=args.max_features,
        fast=args.fast,
        no_plots=args.no_plots,
    )


def main() -> None:
    logger = setup_logging()
    config = parse_args()

    config.dataset = config.dataset.resolve()
    config.output_dir = config.output_dir.resolve()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Train PM dynamics baselines")
    logger.info("=" * 80)
    logger.info("Dataset: %s", config.dataset)
    logger.info("Output dir: %s", config.output_dir)
    logger.info("Targets: %s", config.targets)
    logger.info("Target modes: %s", config.target_modes)
    logger.info("Validation modes: %s", config.validation_modes)
    logger.info("Models: %s", config.models)

    if not config.dataset.exists():
        raise FileNotFoundError(f"Dataset was not found: {config.dataset}")

    df = read_dataset(config.dataset)
    rows_loaded = len(df)

    columns = list(df.columns)
    id_cols = detect_id_columns(columns)
    pm_cols = detect_pm_columns(columns)

    available_targets = [t for t in config.targets if t in pm_cols]
    missing_targets = [t for t in config.targets if t not in pm_cols]

    if missing_targets:
        logger.warning("Missing PM targets will be skipped: %s", missing_targets)

    if not available_targets:
        raise ValueError(f"No requested PM targets were found. Detected PM columns: {pm_cols}")

    pm_cols = {k: v for k, v in pm_cols.items() if k in available_targets}

    logger.info("Detected ID columns: %s", id_cols)
    logger.info("Detected PM columns: %s", pm_cols)

    df = add_pm_dynamics(
        df=df,
        id_cols=id_cols,
        pm_cols=pm_cols,
        rolling_window=config.rolling_window,
        trend_threshold=config.trend_threshold,
    )

    if config.max_rows is not None and len(df) > config.max_rows:
        df = df.sample(n=config.max_rows, random_state=config.random_state).reset_index(drop=True)
        logger.info("Sampled rows after dynamics construction: %d", len(df))

    feature_cols = select_feature_columns(
        df=df,
        id_cols=id_cols,
        pm_cols=pm_cols,
        feature_set=config.feature_set,
        max_features=config.max_features,
    )

    if not feature_cols:
        raise ValueError("No feature columns were selected.")

    logger.info("Selected feature columns: %d", len(feature_cols))

    fold_rows = []

    for target in available_targets:
        logger.info("-" * 80)
        logger.info("Target: %s", target)

        for target_mode in config.target_modes:
            task_df, target_col, task_type = prepare_task_dataframe(
                df=df,
                feature_cols=feature_cols,
                target=target,
                target_mode=target_mode,
            )

            if len(task_df) < 100:
                logger.warning("Skip %s/%s: too few rows (%d)", target, target_mode, len(task_df))
                continue

            # Reattach ID columns for split generation.
            id_keep = [c for c in id_cols.values() if c in df.columns]
            orig_index = task_df["__orig_index"].to_numpy()

            task_with_ids = pd.concat(
                [
                    df.loc[orig_index, id_keep].reset_index(drop=True)
                    if id_keep else pd.DataFrame(index=task_df.index),
                    task_df.drop(columns=["__orig_index"]).reset_index(drop=True),
                ],
                axis=1,
            )

            task_df = task_df.drop(columns=["__orig_index"]).reset_index(drop=True)

            task_models = []
            for model_name in config.models:
                if task_type == "regression" and model_name.endswith("_reg"):
                    task_models.append(model_name)
                if task_type == "classification" and (model_name.endswith("_clf") or model_name == "logistic_robust"):
                    task_models.append(model_name)

            if not task_models:
                logger.warning("No compatible models for %s/%s (%s)", target, target_mode, task_type)
                continue

            logger.info(
                "Target mode=%s | task_type=%s | rows=%d | target_col=%s | models=%s",
                target_mode,
                task_type,
                len(task_df),
                target_col,
                task_models,
            )

            for validation_mode in config.validation_modes:
                if validation_mode == "cross_source":
                    split_iter = [
                        (name, train_idx, val_idx)
                        for name, train_idx, val_idx in get_cross_source_splits(task_with_ids, id_cols=id_cols)
                    ]
                    if not split_iter:
                        logger.warning("No cross-source splits available.")
                        continue
                else:
                    split_iter = [
                        (validation_mode, train_idx, val_idx)
                        for _, train_idx, val_idx in get_split_indices(
                            task_with_ids,
                            validation_mode=validation_mode,
                            id_cols=id_cols,
                            test_size=config.test_size,
                            n_splits=config.n_splits,
                            random_state=config.random_state,
                        )
                    ]

                for split_no, (validation_name, train_idx, val_idx) in enumerate(split_iter, start=1):
                    for model_name in task_models:
                        logger.info(
                            "[%s/%s] validation=%s fold=%d model=%s n_train=%d n_val=%d",
                            target,
                            target_mode,
                            validation_name,
                            split_no,
                            model_name,
                            len(train_idx),
                            len(val_idx),
                        )

                        try:
                            metrics = evaluate_one_split(
                                task_df=task_df,
                                feature_cols=feature_cols,
                                target_col=target_col,
                                task_type=task_type,
                                model_name=model_name,
                                train_idx=train_idx,
                                val_idx=val_idx,
                                random_state=config.random_state,
                                fast=config.fast,
                            )
                        except Exception as exc:
                            logger.exception(
                                "Failed %s/%s/%s/%s: %s",
                                target,
                                target_mode,
                                validation_name,
                                model_name,
                                exc,
                            )
                            continue

                        fold_rows.append(
                            {
                                "target": target,
                                "target_mode": target_mode,
                                "target_col": target_col,
                                "task_type": task_type,
                                "validation": validation_name,
                                "fold": split_no,
                                "model": model_name,
                                **metrics,
                            }
                        )

    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        raise RuntimeError("No fold metrics were produced.")

    summary_df = aggregate_metrics(fold_df)

    fold_path = config.output_dir / "fold_metrics.csv"
    summary_path = config.output_dir / "summary.csv"
    feature_path = config.output_dir / "feature_columns.json"
    report_path = config.output_dir / "report.md"

    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    feature_path.write_text(
        json.dumps(
            {
                "feature_set": config.feature_set,
                "n_features": len(feature_cols),
                "feature_columns": feature_cols,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset_info = {
        "rows_loaded": int(rows_loaded),
        "rows_used": int(len(df)),
        "id_columns": id_cols,
        "pm_columns": pm_cols,
    }

    (config.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset": str(config.dataset),
                "output_dir": str(config.output_dir),
                "run_name": config.run_name,
                "dataset_info": dataset_info,
                "targets": available_targets,
                "target_modes": config.target_modes,
                "validation_modes": config.validation_modes,
                "models": config.models,
                "feature_set": config.feature_set,
                "n_features": len(feature_cols),
                "trend_threshold": config.trend_threshold,
                "rolling_window": config.rolling_window,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not config.no_plots:
        plot_summary(summary_df, config.output_dir)

    write_report(
        output_dir=config.output_dir,
        config=config,
        dataset_info=dataset_info,
        feature_cols=feature_cols,
        fold_df=fold_df,
        summary_df=summary_df,
    )

    logger.info("=" * 80)
    logger.info("Saved PM dynamics baseline outputs")
    logger.info("=" * 80)
    logger.info("Fold metrics: %s", fold_path)
    logger.info("Summary: %s", summary_path)
    logger.info("Features: %s", feature_path)
    logger.info("Report: %s", report_path)
    logger.info("")
    logger.info("Top summary rows:")
    with pd.option_context("display.max_rows", 30, "display.max_columns", 20, "display.width", 180):
        logger.info("\n%s", summary_df.head(20).to_string(index=False))
    logger.info("Done.")


if __name__ == "__main__":
    main()