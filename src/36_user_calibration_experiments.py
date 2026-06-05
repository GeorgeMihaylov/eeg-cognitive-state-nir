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
from sklearn.model_selection import train_test_split
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
    "slow_pm_",
    "slow_pca_",
    "latent",
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
    latent_dataset: Path
    raw_dataset: Path
    output_dir: Path
    run_name: str
    regression_targets: list[str]
    classification_targets: list[str]
    calibration_fracs: list[float]
    subject_dependent_train_frac: float
    trend_threshold: float
    rolling_window: int
    min_subject_rows: int
    feature_set: str
    max_features: int | None
    random_state: int
    fast: bool
    no_plots: bool


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("user_calibration")


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
        raise ValueError("Need source/subject_id/record_id columns to compute trends.")

    return group_cols


def choose_sort_columns(id_cols: dict[str, str]) -> list[str]:
    if "window_start" in id_cols:
        return [id_cols["window_start"]]
    if "window_id" in id_cols:
        return [id_cols["window_id"]]
    return []


def is_feature_column(col: str, id_cols: dict[str, str], pm_cols: dict[str, str] | None = None) -> bool:
    if col in set(id_cols.values()):
        return False

    if pm_cols is not None and col in set(pm_cols.values()):
        return False

    low = col.lower()

    for keyword in NON_FEATURE_KEYWORDS:
        if keyword in low:
            return False

    return True


def select_feature_columns(
    df: pd.DataFrame,
    id_cols: dict[str, str],
    feature_set: str,
    max_features: int | None,
    pm_cols: dict[str, str] | None = None,
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


def add_trend_targets(
    df: pd.DataFrame,
    id_cols: dict[str, str],
    pm_cols: dict[str, str],
    requested_targets: list[str],
    trend_threshold: float,
) -> pd.DataFrame:
    out = df.copy()

    available = [t for t in requested_targets if t in pm_cols]

    for target in available:
        out[f"pm_{target}"] = pd.to_numeric(out[pm_cols[target]], errors="coerce")

    group_cols = choose_group_columns(id_cols)
    sort_cols = choose_sort_columns(id_cols)

    sort_by = list(dict.fromkeys(group_cols + sort_cols))
    if sort_by:
        out = out.sort_values(sort_by).reset_index(drop=True)

    pieces = []

    for _, group in out.groupby(group_cols, dropna=False, sort=False):
        group = group.copy()

        for target in available:
            col = f"pm_{target}"
            delta = group[col].shift(-1) - group[col]

            trend = pd.Series("stable", index=group.index, dtype="object")
            trend[delta > trend_threshold] = "up"
            trend[delta < -trend_threshold] = "down"
            trend[delta.isna()] = np.nan

            group[f"{target}_trend"] = trend

        pieces.append(group)

    return pd.concat(pieces, ignore_index=True)


def subject_calibration_splits(
    df: pd.DataFrame,
    subject_col: str,
    subject_value: str,
    calibration_fracs: list[float],
    subject_dependent_train_frac: float,
    random_state: int,
):
    subject_mask = df[subject_col].astype(str) == str(subject_value)
    subject_idx = df.index[subject_mask].to_numpy()
    other_idx = df.index[~subject_mask].to_numpy()

    if len(subject_idx) < 10:
        return

    for frac in calibration_fracs:
        frac = float(frac)

        if frac <= 0:
            yield {
                "mode": "zero_shot",
                "calibration_frac": 0.0,
                "train_idx": other_idx,
                "eval_idx": subject_idx,
            }
            continue

        if frac >= 1:
            continue

        calib_idx, eval_idx = train_test_split(
            subject_idx,
            train_size=frac,
            random_state=random_state,
            shuffle=True,
        )

        train_idx = np.concatenate([other_idx, calib_idx])

        yield {
            "mode": f"calibration_{int(round(frac * 100))}pct",
            "calibration_frac": frac,
            "train_idx": train_idx,
            "eval_idx": eval_idx,
        }

    if 0 < subject_dependent_train_frac < 1:
        train_idx, eval_idx = train_test_split(
            subject_idx,
            train_size=subject_dependent_train_frac,
            random_state=random_state,
            shuffle=True,
        )

        yield {
            "mode": "subject_dependent",
            "calibration_frac": subject_dependent_train_frac,
            "train_idx": train_idx,
            "eval_idx": eval_idx,
        }


def evaluate_regression_task(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    subject_col: str,
    calibration_fracs: list[float],
    subject_dependent_train_frac: float,
    random_state: int,
    fast: bool,
    min_subject_rows: int,
) -> pd.DataFrame:
    task_cols = feature_cols + [target_col, subject_col]
    task_df = df[task_cols].copy()

    task_df[target_col] = pd.to_numeric(task_df[target_col], errors="coerce")
    task_df = task_df[np.isfinite(task_df[target_col])].reset_index(drop=True)

    subjects = sorted(task_df[subject_col].dropna().astype(str).unique().tolist())

    rows = []

    for subject in subjects:
        n_subject = int((task_df[subject_col].astype(str) == str(subject)).sum())

        if n_subject < min_subject_rows:
            continue

        for split in subject_calibration_splits(
            df=task_df,
            subject_col=subject_col,
            subject_value=subject,
            calibration_fracs=calibration_fracs,
            subject_dependent_train_frac=subject_dependent_train_frac,
            random_state=random_state,
        ):
            train_idx = split["train_idx"]
            eval_idx = split["eval_idx"]

            if len(train_idx) < 20 or len(eval_idx) < 10:
                continue

            model = make_regressor(random_state=random_state, fast=fast)

            X_train = task_df.iloc[train_idx][feature_cols]
            y_train = task_df.iloc[train_idx][target_col].values

            X_eval = task_df.iloc[eval_idx][feature_cols]
            y_eval = task_df.iloc[eval_idx][target_col].values

            started = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - started

            started = time.perf_counter()
            pred = model.predict(X_eval)
            predict_time = time.perf_counter() - started

            metrics = regression_metrics(y_eval.astype(float), pred.astype(float))

            rows.append(
                {
                    "task_type": "regression",
                    "target": target_col,
                    "subject_id": subject,
                    "mode": split["mode"],
                    "calibration_frac": split["calibration_frac"],
                    "n_subject_total": n_subject,
                    "n_train": int(len(train_idx)),
                    "n_eval": int(len(eval_idx)),
                    "model": "hgb_reg",
                    "fit_time_sec": fit_time,
                    "predict_time_sec": predict_time,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def evaluate_classification_task(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    subject_col: str,
    calibration_fracs: list[float],
    subject_dependent_train_frac: float,
    random_state: int,
    fast: bool,
    min_subject_rows: int,
) -> pd.DataFrame:
    task_cols = feature_cols + [target_col, subject_col]
    task_df = df[task_cols].copy()
    task_df = task_df[task_df[target_col].isin(["up", "stable", "down"])].reset_index(drop=True)

    subjects = sorted(task_df[subject_col].dropna().astype(str).unique().tolist())

    rows = []

    for subject in subjects:
        n_subject = int((task_df[subject_col].astype(str) == str(subject)).sum())

        if n_subject < min_subject_rows:
            continue

        for split in subject_calibration_splits(
            df=task_df,
            subject_col=subject_col,
            subject_value=subject,
            calibration_fracs=calibration_fracs,
            subject_dependent_train_frac=subject_dependent_train_frac,
            random_state=random_state,
        ):
            train_idx = split["train_idx"]
            eval_idx = split["eval_idx"]

            if len(train_idx) < 20 or len(eval_idx) < 10:
                continue

            y_train = task_df.iloc[train_idx][target_col].values
            y_eval = task_df.iloc[eval_idx][target_col].values

            if len(np.unique(y_train)) < 2 or len(np.unique(y_eval)) < 2:
                continue

            model = make_classifier(random_state=random_state, fast=fast)

            X_train = task_df.iloc[train_idx][feature_cols]
            X_eval = task_df.iloc[eval_idx][feature_cols]

            started = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - started

            started = time.perf_counter()
            pred = model.predict(X_eval)
            predict_time = time.perf_counter() - started

            metrics = classification_metrics(y_eval, pred)

            rows.append(
                {
                    "task_type": "classification",
                    "target": target_col,
                    "subject_id": subject,
                    "mode": split["mode"],
                    "calibration_frac": split["calibration_frac"],
                    "n_subject_total": n_subject,
                    "n_train": int(len(train_idx)),
                    "n_eval": int(len(eval_idx)),
                    "model": "hgb_clf",
                    "fit_time_sec": fit_time,
                    "predict_time_sec": predict_time,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def aggregate_results(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame()

    group_cols = ["task_type", "target", "mode", "calibration_frac", "model"]

    metric_cols = [
        c for c in fold_df.columns
        if c not in group_cols
        and c not in {
            "subject_id",
            "n_subject_total",
            "n_train",
            "n_eval",
            "fit_time_sec",
            "predict_time_sec",
        }
        and pd.api.types.is_numeric_dtype(fold_df[c])
    ]

    rows = []

    for keys, sub in fold_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["subjects"] = int(sub["subject_id"].nunique())
        row["n_eval_total"] = int(sub["n_eval"].sum())
        row["fit_time_sec_sum"] = float(sub["fit_time_sec"].sum())

        for metric in metric_cols:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_std"] = sub[metric].std()
            row[f"{metric}_min"] = sub[metric].min()
            row[f"{metric}_max"] = sub[metric].max()

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["task_type", "target", "calibration_frac"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def compute_calibration_gain(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows = []

    for (task_type, target, model), sub in summary_df.groupby(["task_type", "target", "model"], dropna=False):
        zero = sub[sub["mode"] == "zero_shot"]

        if zero.empty:
            continue

        zero_row = zero.iloc[0]

        if task_type == "regression":
            base_metric = zero_row.get("r2_mean", np.nan)
            metric_name = "r2_mean"
        else:
            base_metric = zero_row.get("balanced_accuracy_mean", np.nan)
            metric_name = "balanced_accuracy_mean"

        for _, row in sub.iterrows():
            current = row.get(metric_name, np.nan)

            rows.append(
                {
                    "task_type": task_type,
                    "target": target,
                    "model": model,
                    "mode": row["mode"],
                    "calibration_frac": row["calibration_frac"],
                    "metric_name": metric_name,
                    "zero_shot_metric": base_metric,
                    "current_metric": current,
                    "absolute_gain_vs_zero_shot": current - base_metric if pd.notna(current) and pd.notna(base_metric) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def plot_calibration_curves(gain_df: pd.DataFrame, output_dir: Path) -> None:
    if gain_df.empty:
        return

    for task_type, metric_name, ylabel in [
        ("regression", "r2_mean", "R²"),
        ("classification", "balanced_accuracy_mean", "Balanced accuracy"),
    ]:
        sub = gain_df[
            (gain_df["task_type"] == task_type)
            & (gain_df["metric_name"] == metric_name)
            & (gain_df["mode"] != "subject_dependent")
        ].copy()

        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))

        for target, target_df in sub.groupby("target"):
            target_df = target_df.sort_values("calibration_frac")
            ax.plot(
                target_df["calibration_frac"],
                target_df["current_metric"],
                marker="o",
                label=target,
            )

        ax.set_title(f"User calibration: {task_type}")
        ax.set_xlabel("Calibration fraction")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

        fig.tight_layout()
        fig.savefig(output_dir / f"calibration_curve_{task_type}.png", dpi=160)
        plt.close(fig)


def write_report(
    output_dir: Path,
    config: Config,
    regression_summary: pd.DataFrame,
    classification_summary: pd.DataFrame,
    gain_df: pd.DataFrame,
    dataset_info: dict,
) -> None:
    lines = []

    lines.append("# User calibration experiments")
    lines.append("")
    lines.append("## Input")
    lines.append("")
    lines.append(f"- Latent dataset: `{config.latent_dataset}`")
    lines.append(f"- Raw dataset: `{config.raw_dataset}`")
    lines.append(f"- Output dir: `{config.output_dir}`")
    lines.append(f"- Regression targets: `{config.regression_targets}`")
    lines.append(f"- Classification targets: `{config.classification_targets}`")
    lines.append(f"- Calibration fractions: `{config.calibration_fracs}`")
    lines.append(f"- Subject-dependent train fraction: `{config.subject_dependent_train_frac}`")
    lines.append(f"- Feature set: `{config.feature_set}`")
    lines.append(f"- Max features: `{config.max_features}`")
    lines.append("")
    lines.append("## Dataset info")
    lines.append("")
    for key, value in dataset_info.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")

    lines.append("## Regression calibration summary")
    lines.append("")
    if not regression_summary.empty:
        show_cols = [
            c for c in [
                "target",
                "mode",
                "calibration_frac",
                "subjects",
                "n_eval_total",
                "r2_mean",
                "r2_std",
                "spearman_mean",
                "mae_mean",
                "rmse_mean",
            ]
            if c in regression_summary.columns
        ]
        lines.append(regression_summary[show_cols].to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No regression summary rows.")

    lines.append("")
    lines.append("## Classification calibration summary")
    lines.append("")
    if not classification_summary.empty:
        show_cols = [
            c for c in [
                "target",
                "mode",
                "calibration_frac",
                "subjects",
                "n_eval_total",
                "balanced_accuracy_mean",
                "balanced_accuracy_std",
                "macro_f1_mean",
                "macro_recall_mean",
                "macro_precision_mean",
            ]
            if c in classification_summary.columns
        ]
        lines.append(classification_summary[show_cols].to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No classification summary rows.")

    lines.append("")
    lines.append("## Calibration gain vs zero-shot")
    lines.append("")
    if not gain_df.empty:
        show_cols = [
            c for c in [
                "task_type",
                "target",
                "mode",
                "calibration_frac",
                "metric_name",
                "zero_shot_metric",
                "current_metric",
                "absolute_gain_vs_zero_shot",
            ]
            if c in gain_df.columns
        ]
        lines.append(gain_df[show_cols].to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No calibration gain rows.")

    lines.append("")
    lines.append("## Working interpretation")
    lines.append("")
    lines.append("The experiment compares zero-shot transfer to a new subject against small subject-specific calibration.")
    lines.append("")
    lines.append("- Regression targets: slow latent coordinates and selected slow PM targets.")
    lines.append("- Classification targets: PM trend labels.")
    lines.append("- Main question: whether 5–20% subject calibration improves transfer to the held-out part of the same user.")
    lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_list_arg(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_float_list_arg(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Run user calibration experiments for slow latent states and PM trend targets."
    )
    parser.add_argument(
        "--latent-dataset",
        type=Path,
        default=Path("reports/slow_latent_states/pm_w10/slow_pm_latent_states_w10.parquet"),
        help="Dataset produced by src/35_build_and_train_slow_latent_states.py.",
    )
    parser.add_argument(
        "--raw-dataset",
        type=Path,
        default=Path("data/processed/windowed_eeg_pm_dataset_w10.csv"),
        help="Original windowed EEG/PM dataset for trend targets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/user_calibration/pm_w10"),
        help="Output directory.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="user_calibration_pm_w10",
        help="Run name.",
    )
    parser.add_argument(
        "--regression-targets",
        type=str,
        default="slow_pca_1,slow_pca_2,slow_pca_4,slow_pm_excitement,slow_pm_relaxation,slow_pm_stress,slow_pm_focus",
        help="Comma-separated regression targets from latent dataset.",
    )
    parser.add_argument(
        "--classification-targets",
        type=str,
        default="stress,focus,attention,interest",
        help="Comma-separated PM targets for trend classification.",
    )
    parser.add_argument(
        "--calibration-fracs",
        type=str,
        default="0,0.05,0.10,0.20",
        help="Comma-separated calibration fractions. 0 means zero-shot.",
    )
    parser.add_argument(
        "--subject-dependent-train-frac",
        type=float,
        default=0.5,
        help="Train fraction for subject-dependent baseline.",
    )
    parser.add_argument(
        "--trend-threshold",
        type=float,
        default=0.02,
        help="PM trend threshold.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Reserved for compatibility.",
    )
    parser.add_argument(
        "--min-subject-rows",
        type=int,
        default=80,
        help="Minimum rows per subject for calibration experiments.",
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
        latent_dataset=args.latent_dataset,
        raw_dataset=args.raw_dataset,
        output_dir=args.output_dir,
        run_name=args.run_name,
        regression_targets=parse_list_arg(args.regression_targets),
        classification_targets=parse_list_arg(args.classification_targets),
        calibration_fracs=parse_float_list_arg(args.calibration_fracs),
        subject_dependent_train_frac=args.subject_dependent_train_frac,
        trend_threshold=args.trend_threshold,
        rolling_window=args.rolling_window,
        min_subject_rows=args.min_subject_rows,
        feature_set=args.feature_set,
        max_features=args.max_features,
        random_state=args.random_state,
        fast=args.fast,
        no_plots=args.no_plots,
    )


def main() -> None:
    logger = setup_logging()
    config = parse_args()

    config.latent_dataset = config.latent_dataset.resolve()
    config.raw_dataset = config.raw_dataset.resolve()
    config.output_dir = config.output_dir.resolve()
    figures_dir = config.output_dir / "figures"

    config.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("User calibration experiments")
    logger.info("=" * 80)
    logger.info("Latent dataset: %s", config.latent_dataset)
    logger.info("Raw dataset: %s", config.raw_dataset)
    logger.info("Output dir: %s", config.output_dir)

    if not config.latent_dataset.exists():
        raise FileNotFoundError(f"Latent dataset not found: {config.latent_dataset}")

    if not config.raw_dataset.exists():
        raise FileNotFoundError(f"Raw dataset not found: {config.raw_dataset}")

    # ---------------------------------------------------------------------
    # Regression calibration: slow latent and slow PM targets.
    # ---------------------------------------------------------------------
    latent_df = read_dataset(config.latent_dataset)
    latent_id_cols = detect_id_columns(list(latent_df.columns))

    if "subject_id" not in latent_id_cols:
        raise ValueError("subject_id column is required in latent dataset.")

    latent_feature_cols = select_feature_columns(
        latent_df,
        id_cols=latent_id_cols,
        pm_cols=None,
        feature_set=config.feature_set,
        max_features=config.max_features,
    )

    available_regression_targets = [
        t for t in config.regression_targets
        if t in latent_df.columns
    ]

    missing_regression_targets = [
        t for t in config.regression_targets
        if t not in latent_df.columns
    ]

    if missing_regression_targets:
        logger.warning("Missing regression targets will be skipped: %s", missing_regression_targets)

    logger.info("Regression feature columns: %d", len(latent_feature_cols))
    logger.info("Regression targets: %s", available_regression_targets)

    regression_rows = []

    for target in available_regression_targets:
        logger.info("Regression calibration target: %s", target)

        res = evaluate_regression_task(
            df=latent_df,
            feature_cols=latent_feature_cols,
            target_col=target,
            subject_col=latent_id_cols["subject_id"],
            calibration_fracs=config.calibration_fracs,
            subject_dependent_train_frac=config.subject_dependent_train_frac,
            random_state=config.random_state,
            fast=config.fast,
            min_subject_rows=config.min_subject_rows,
        )

        if not res.empty:
            regression_rows.append(res)

    regression_fold_metrics = (
        pd.concat(regression_rows, ignore_index=True)
        if regression_rows
        else pd.DataFrame()
    )

    # ---------------------------------------------------------------------
    # Classification calibration: PM trend targets.
    # ---------------------------------------------------------------------
    raw_df = read_dataset(config.raw_dataset)
    raw_id_cols = detect_id_columns(list(raw_df.columns))
    raw_pm_cols = detect_pm_columns(list(raw_df.columns))

    if "subject_id" not in raw_id_cols:
        raise ValueError("subject_id column is required in raw dataset.")

    raw_df = add_trend_targets(
        df=raw_df,
        id_cols=raw_id_cols,
        pm_cols=raw_pm_cols,
        requested_targets=config.classification_targets,
        trend_threshold=config.trend_threshold,
    )

    raw_feature_cols = select_feature_columns(
        raw_df,
        id_cols=raw_id_cols,
        pm_cols=raw_pm_cols,
        feature_set=config.feature_set,
        max_features=config.max_features,
    )

    available_classification_targets = [
        f"{target}_trend"
        for target in config.classification_targets
        if f"{target}_trend" in raw_df.columns
    ]

    logger.info("Classification feature columns: %d", len(raw_feature_cols))
    logger.info("Classification targets: %s", available_classification_targets)

    classification_rows = []

    for target_col in available_classification_targets:
        logger.info("Classification calibration target: %s", target_col)

        res = evaluate_classification_task(
            df=raw_df,
            feature_cols=raw_feature_cols,
            target_col=target_col,
            subject_col=raw_id_cols["subject_id"],
            calibration_fracs=config.calibration_fracs,
            subject_dependent_train_frac=config.subject_dependent_train_frac,
            random_state=config.random_state,
            fast=config.fast,
            min_subject_rows=config.min_subject_rows,
        )

        if not res.empty:
            classification_rows.append(res)

    classification_fold_metrics = (
        pd.concat(classification_rows, ignore_index=True)
        if classification_rows
        else pd.DataFrame()
    )

    # ---------------------------------------------------------------------
    # Aggregate and save.
    # ---------------------------------------------------------------------
    all_fold_metrics = pd.concat(
        [regression_fold_metrics, classification_fold_metrics],
        ignore_index=True,
        sort=False,
    )

    summary = aggregate_results(all_fold_metrics)
    gain = compute_calibration_gain(summary)

    regression_summary = summary[summary["task_type"] == "regression"].copy()
    classification_summary = summary[summary["task_type"] == "classification"].copy()

    all_fold_metrics.to_csv(config.output_dir / "calibration_fold_metrics.csv", index=False)
    summary.to_csv(config.output_dir / "calibration_summary.csv", index=False)
    regression_summary.to_csv(config.output_dir / "calibration_summary_regression.csv", index=False)
    classification_summary.to_csv(config.output_dir / "calibration_summary_classification.csv", index=False)
    gain.to_csv(config.output_dir / "calibration_gain_vs_zero_shot.csv", index=False)

    dataset_info = {
        "latent_rows": int(len(latent_df)),
        "raw_rows": int(len(raw_df)),
        "latent_features": int(len(latent_feature_cols)),
        "raw_features": int(len(raw_feature_cols)),
        "available_regression_targets": available_regression_targets,
        "available_classification_targets": available_classification_targets,
        "latent_id_columns": latent_id_cols,
        "raw_id_columns": raw_id_cols,
        "raw_pm_columns": raw_pm_cols,
    }

    (config.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_name": config.run_name,
                "latent_dataset": str(config.latent_dataset),
                "raw_dataset": str(config.raw_dataset),
                "output_dir": str(config.output_dir),
                "regression_targets_requested": config.regression_targets,
                "classification_targets_requested": config.classification_targets,
                "calibration_fracs": config.calibration_fracs,
                "subject_dependent_train_frac": config.subject_dependent_train_frac,
                "trend_threshold": config.trend_threshold,
                "min_subject_rows": config.min_subject_rows,
                "feature_set": config.feature_set,
                "max_features": config.max_features,
                "dataset_info": dataset_info,
                "n_fold_rows": int(len(all_fold_metrics)),
                "n_summary_rows": int(len(summary)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not config.no_plots:
        plot_calibration_curves(gain, figures_dir)

    write_report(
        output_dir=config.output_dir,
        config=config,
        regression_summary=regression_summary,
        classification_summary=classification_summary,
        gain_df=gain,
        dataset_info=dataset_info,
    )

    logger.info("=" * 80)
    logger.info("Saved user calibration outputs")
    logger.info("=" * 80)
    logger.info("Fold metrics: %s", config.output_dir / "calibration_fold_metrics.csv")
    logger.info("Summary: %s", config.output_dir / "calibration_summary.csv")
    logger.info("Regression summary: %s", config.output_dir / "calibration_summary_regression.csv")
    logger.info("Classification summary: %s", config.output_dir / "calibration_summary_classification.csv")
    logger.info("Gain vs zero-shot: %s", config.output_dir / "calibration_gain_vs_zero_shot.csv")
    logger.info("Report: %s", config.output_dir / "report.md")
    logger.info("Done.")


if __name__ == "__main__":
    main()