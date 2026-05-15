#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WESAD protocol-control experiment.

Purpose:
    Check whether WESAD stress classification quality is driven by:
        1. physiological signals: EDA / BVP / TEMP;
        2. movement/protocol confounding: ACC;
        3. threshold choice.

Main comparisons:
    all          = EDA + BVP + TEMP + ACC
    no_acc       = EDA + BVP + TEMP
    acc_only     = ACC only
    bvp_only     = BVP only
    eda_only     = EDA only
    temp_only    = TEMP only
    eda_bvp      = EDA + BVP
    eda_bvp_temp = EDA + BVP + TEMP
    bvp_temp     = BVP + TEMP

Validation:
    GroupKFold by subject_id.

Typical command:
    D:\miniconda3\envs\eeg_nir\python.exe src\21_train_wesad_protocol_control.py `
      --dataset data\processed\wesad_windowed_stress_dataset.parquet `
      --fast `
      --models logistic_robust,lgbm_clf `
      --run-name wesad_protocol_control

Fast test:
    D:\miniconda3\envs\eeg_nir\python.exe src\21_train_wesad_protocol_control.py `
      --dataset data\processed\wesad_windowed_stress_dataset.parquet `
      --fast `
      --fold-limit 2 `
      --models logistic_robust `
      --run-name wesad_protocol_control_test
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


NON_FEATURE_COLUMNS = {
    "subject_id",
    "subject_num",
    "window_id",
    "start_sec",
    "end_sec",
    "center_sec",
    "window_size_sec",
    "step_size_sec",
    "stress_binary",
    "wesad_multiclass",
    "wesad_label",
    "wesad_label_name",
    "valid_fraction",
    "majority_fraction",
    "majority_original_label",
    "majority_original_label_count",
    "n_label_samples",
}

CONTROL_GROUPS = {
    "all": ["eda_", "bvp_", "temp_", "acc_"],
    "no_acc": ["eda_", "bvp_", "temp_"],
    "acc_only": ["acc_"],
    "bvp_only": ["bvp_"],
    "eda_only": ["eda_"],
    "temp_only": ["temp_"],
    "eda_bvp": ["eda_", "bvp_"],
    "eda_bvp_temp": ["eda_", "bvp_", "temp_"],
    "bvp_temp": ["bvp_", "temp_"],
}


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def make_run_dir(root: Path, run_name: str, output_root: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in run_name)
    run_dir = resolve_path(root, output_root) / f"{ts}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def infer_all_feature_columns(df: pd.DataFrame) -> List[str]:
    out = []
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if col.startswith("label_count_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return sorted(out)


def select_columns(all_cols: List[str], prefixes: List[str]) -> List[str]:
    return sorted([c for c in all_cols if any(c.startswith(p) for p in prefixes)])


def build_feature_groups(all_feature_cols: List[str], requested_groups: Optional[str]) -> Dict[str, List[str]]:
    requested = None
    if requested_groups:
        requested = [g.strip() for g in requested_groups.split(",") if g.strip()]

    groups: Dict[str, List[str]] = {}
    for name, prefixes in CONTROL_GROUPS.items():
        if requested is not None and name not in requested:
            continue
        cols = select_columns(all_feature_cols, prefixes)
        if cols:
            groups[name] = cols

    if requested is not None:
        missing = [g for g in requested if g not in groups]
        if missing:
            raise ValueError(f"Requested groups not available: {missing}. Available: {list(groups)}")

    return groups


def build_models(fast: bool, selected_models: Optional[str], random_state: int) -> Dict[str, object]:
    models: Dict[str, object] = {}

    models["logistic_robust"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
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
                    max_iter=120 if fast else 250,
                    learning_rate=0.07 if fast else 0.05,
                    max_leaf_nodes=15 if fast else 31,
                    l2_regularization=0.1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    models["rf_clf"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=150 if fast else 400,
                    max_depth=10 if fast else None,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    if HAS_LIGHTGBM:
        models["lgbm_clf"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=200 if fast else 500,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary",
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )

    if selected_models:
        requested = [m.strip() for m in selected_models.split(",") if m.strip()]
        unknown = [m for m in requested if m not in models]
        if unknown:
            raise ValueError(f"Unknown models: {unknown}. Available: {list(models)}")
        models = {m: models[m] for m in requested}

    return models


def get_scores(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.reshape(-1)
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    return model.predict(X).astype(float)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_stress": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_stress": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_stress": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }

    if len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = np.nan
        try:
            out["average_precision"] = float(average_precision_score(y_true, y_score))
        except Exception:
            out["average_precision"] = np.nan
    else:
        out["roc_auc"] = np.nan
        out["average_precision"] = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["tn"] = int(cm[0, 0])
    out["fp"] = int(cm[0, 1])
    out["fn"] = int(cm[1, 0])
    out["tp"] = int(cm[1, 1])
    return out


def aggregate_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "precision_stress",
        "recall_stress",
        "f1_stress",
        "roc_auc",
        "average_precision",
    ]

    rows = []
    for (feature_group, model), g in metrics_df.groupby(["feature_group", "model"]):
        row = {
            "feature_group": feature_group,
            "model": model,
            "folds": int(g["fold"].nunique()),
            "n_val_total": int(g["n_val"].sum()),
            "n_features": int(g["n_features"].iloc[0]),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0))
            row[f"{col}_min"] = float(g[col].min())
            row[f"{col}_max"] = float(g[col].max())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["balanced_accuracy_mean", "macro_f1_mean", "roc_auc_mean"],
        ascending=False,
    )


def threshold_sweep(predictions: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for (feature_group, model), g in predictions.groupby(["feature_group", "model"]):
        y_true = g["y_true"].astype(int).to_numpy()
        y_score = g["y_score"].astype(float).to_numpy()

        for th in thresholds:
            y_pred = (y_score >= th).astype(int)
            metrics = compute_metrics(y_true, y_pred, y_score)
            row = {"feature_group": feature_group, "model": model, "threshold": float(th)}
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)


def best_thresholds(threshold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature_group, model), g in threshold_df.groupby(["feature_group", "model"]):
        row = {"feature_group": feature_group, "model": model}
        for objective in ["balanced_accuracy", "macro_f1", "f1_stress"]:
            idx = g[objective].idxmax()
            best = g.loc[idx]
            row[f"best_threshold_by_{objective}"] = float(best["threshold"])
            row[f"best_{objective}"] = float(best[objective])
            row[f"at_{objective}_precision_stress"] = float(best["precision_stress"])
            row[f"at_{objective}_recall_stress"] = float(best["recall_stress"])
            row[f"at_{objective}_roc_auc"] = float(best["roc_auc"])
            row[f"at_{objective}_average_precision"] = float(best["average_precision"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("best_balanced_accuracy", ascending=False)


def build_protocol_comparison(summary: pd.DataFrame, best_th: pd.DataFrame) -> pd.DataFrame:
    rows = []
    models = sorted(summary["model"].unique().tolist())

    for model in models:
        s = summary[summary["model"] == model].set_index("feature_group")
        t = best_th[best_th["model"] == model].set_index("feature_group") if not best_th.empty else pd.DataFrame()

        row = {"model": model}

        for group in ["all", "no_acc", "acc_only", "bvp_only", "eda_only", "eda_bvp_temp"]:
            if group in s.index:
                row[f"{group}_balanced_accuracy"] = float(s.loc[group, "balanced_accuracy_mean"])
                row[f"{group}_roc_auc"] = float(s.loc[group, "roc_auc_mean"])
                row[f"{group}_f1_stress"] = float(s.loc[group, "f1_stress_mean"])
            if not t.empty and group in t.index:
                row[f"{group}_best_threshold"] = float(t.loc[group, "best_threshold_by_balanced_accuracy"])
                row[f"{group}_best_balanced_accuracy"] = float(t.loc[group, "best_balanced_accuracy"])

        if "all" in s.index and "no_acc" in s.index:
            row["delta_all_minus_no_acc_balanced_accuracy"] = (
                float(s.loc["all", "balanced_accuracy_mean"]) -
                float(s.loc["no_acc", "balanced_accuracy_mean"])
            )
        if "no_acc" in s.index and "acc_only" in s.index:
            row["delta_no_acc_minus_acc_only_balanced_accuracy"] = (
                float(s.loc["no_acc", "balanced_accuracy_mean"]) -
                float(s.loc["acc_only", "balanced_accuracy_mean"])
            )
        if "bvp_only" in s.index and "acc_only" in s.index:
            row["delta_bvp_only_minus_acc_only_balanced_accuracy"] = (
                float(s.loc["bvp_only", "balanced_accuracy_mean"]) -
                float(s.loc["acc_only", "balanced_accuracy_mean"])
            )

        if not t.empty and "all" in t.index and "no_acc" in t.index:
            row["delta_all_minus_no_acc_best_balanced_accuracy"] = (
                float(t.loc["all", "best_balanced_accuracy"]) -
                float(t.loc["no_acc", "best_balanced_accuracy"])
            )
        if not t.empty and "no_acc" in t.index and "acc_only" in t.index:
            row["delta_no_acc_minus_acc_only_best_balanced_accuracy"] = (
                float(t.loc["no_acc", "best_balanced_accuracy"]) -
                float(t.loc["acc_only", "best_balanced_accuracy"])
            )

        rows.append(row)

    return pd.DataFrame(rows)


def plot_metric_by_group(summary: pd.DataFrame, metric: str, figures_dir: Path) -> Optional[Path]:
    if metric not in summary.columns:
        return None

    groups = [g for g in CONTROL_GROUPS.keys() if g in set(summary["feature_group"])]
    models = sorted(summary["model"].unique().tolist())

    fig, ax = plt.subplots(figsize=(max(10, len(groups) * 1.1), 5))
    x = np.arange(len(groups))
    width = 0.82 / max(1, len(models))

    for i, model in enumerate(models):
        g = summary[summary["model"] == model].set_index("feature_group")
        values = [g.loc[group, metric] if group in g.index else np.nan for group in groups]
        ax.bar(x + i * width, values, width=width, label=model)

    ax.set_title(f"{metric} by feature group and model")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(groups, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    path = figures_dir / f"{metric}_by_feature_group_model.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_best_threshold_metric(best_th: pd.DataFrame, metric: str, figures_dir: Path) -> Optional[Path]:
    col = f"best_{metric}"
    if best_th.empty or col not in best_th.columns:
        return None

    df = best_th.copy()
    df["label"] = df["feature_group"] + "\n" + df["model"]
    df = df.sort_values(col, ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.25)))
    ax.barh(df["label"], df[col])
    ax.set_title(f"Threshold-optimized {metric}")
    ax.set_xlabel(col)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = figures_dir / f"best_threshold_{metric}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_no_acc_vs_acc(summary: pd.DataFrame, figures_dir: Path) -> Optional[Path]:
    metric = "balanced_accuracy_mean"
    required_groups = {"no_acc", "acc_only", "all"}
    if not required_groups.issubset(set(summary["feature_group"])):
        return None

    models = sorted(summary["model"].unique().tolist())
    rows = []
    for model in models:
        g = summary[summary["model"] == model].set_index("feature_group")
        if not required_groups.issubset(set(g.index)):
            continue
        rows.append(
            {
                "model": model,
                "all": float(g.loc["all", metric]),
                "no_acc": float(g.loc["no_acc", metric]),
                "acc_only": float(g.loc["acc_only", metric]),
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, df["all"], width=width, label="all")
    ax.bar(x, df["no_acc"], width=width, label="no_acc")
    ax.bar(x + width, df["acc_only"], width=width, label="acc_only")
    ax.set_title("Protocol-control comparison: all vs no_acc vs acc_only")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    path = figures_dir / "protocol_control_all_no_acc_acc_only.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def build_report(
    args: argparse.Namespace,
    dataset_path: Path,
    run_dir: Path,
    df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    summary: pd.DataFrame,
    best_th: pd.DataFrame,
    protocol_cmp: pd.DataFrame,
    figure_paths: List[Path],
) -> str:
    lines = []
    lines.append("# WESAD protocol-control report")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(vars(args), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Rows: **{len(df)}**")
    lines.append(f"- Columns: **{df.shape[1]}**")
    lines.append(f"- Subjects: **{df['subject_id'].nunique()}**")
    lines.append("")
    lines.append("## Target distribution")
    lines.append("")
    counts = df["stress_binary"].value_counts().sort_index().reset_index()
    counts.columns = ["stress_binary", "n_windows"]
    counts["class_name"] = counts["stress_binary"].map({0: "non_stress", 1: "stress"})
    lines.append(counts[["stress_binary", "class_name", "n_windows"]].to_markdown(index=False))
    lines.append("")
    lines.append("## Feature groups")
    lines.append("")
    group_rows = [{"feature_group": name, "n_features": len(cols)} for name, cols in feature_groups.items()]
    lines.append(pd.DataFrame(group_rows).to_markdown(index=False))
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Threshold-optimized summary")
    lines.append("")
    lines.append(best_th.to_markdown(index=False) if not best_th.empty else "No threshold analysis.")
    lines.append("")
    lines.append("## Protocol-control comparison")
    lines.append("")
    lines.append(protocol_cmp.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation rules")
    lines.append("")
    lines.append("- If `acc_only` is close to `no_acc` or `all`, WESAD contains movement/protocol information useful for stress classification.")
    lines.append("- If `no_acc` remains strong, physiological channels are sufficient without movement features.")
    lines.append("- If `all` is worse than `no_acc`, adding ACC may introduce cross-subject instability.")
    lines.append("- If `bvp_only` is close to `no_acc`, BVP/PPG is the strongest practical wearable proxy.")
    lines.append("- If `EDA` underperforms, it may require better SCR/SCL decomposition and lag-aware features.")
    lines.append("")
    lines.append("## Practical conclusion template")
    lines.append("")
    lines.append("```text")
    lines.append("WESAD supports wearable stress detection, but ACC-only performance must be treated as a protocol/movement confounding indicator.")
    lines.append("For PM.Stress alignment, ACC should be used primarily as a control channel, while BVP/EDA/TEMP should be treated as physiological channels.")
    lines.append("```")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for p in figure_paths:
        try:
            rel = p.relative_to(run_dir)
        except Exception:
            rel = p
        lines.append(f"- `{rel}`")
    lines.append("")
    lines.append("## Recommended next step")
    lines.append("")
    lines.append("Proceed to eye-tracking baseline after this control:")
    lines.append("")
    lines.append("```text")
    lines.append("22_prepare_colet_dataset.py")
    lines.append("23_train_colet_workload_baseline.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WESAD protocol-control experiment.")
    parser.add_argument("--root", type=str, default=".", help="Project root.")
    parser.add_argument("--dataset", type=str, default="data/processed/wesad_windowed_stress_dataset.parquet")
    parser.add_argument("--output-root", type=str, default="reports/wearable_pm_alignment/runs")
    parser.add_argument("--run-name", type=str, default="wesad_protocol_control")
    parser.add_argument("--models", type=str, default="logistic_robust,lgbm_clf")
    parser.add_argument("--groups", type=str, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold-limit", type=int, default=0)
    parser.add_argument("--validation", type=str, default="groupkfold_subject", choices=["groupkfold_subject", "random_split"])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-predictions", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    root = Path(args.root).resolve()
    dataset_path = resolve_path(root, args.dataset)
    run_dir = make_run_dir(root, args.run_name, args.output_root)
    figures_dir = run_dir / "figures"

    df = load_dataset(dataset_path)
    if "stress_binary" not in df.columns:
        raise ValueError("Dataset must contain stress_binary.")
    if "subject_id" not in df.columns:
        raise ValueError("Dataset must contain subject_id.")

    all_feature_cols = infer_all_feature_columns(df)
    feature_groups = build_feature_groups(all_feature_cols, args.groups)
    models = build_models(args.fast, args.models, args.random_state)

    y = df["stress_binary"].astype(int).to_numpy()
    groups = df["subject_id"].astype(str).to_numpy()

    if args.validation == "groupkfold_subject":
        n_splits = min(args.n_splits, df["subject_id"].nunique())
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(df, y, groups=groups))
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.random_state)
        splits = list(splitter.split(df, y))

    if args.fold_limit and args.fold_limit > 0:
        splits = splits[: args.fold_limit]

    print("=" * 80)
    print("WESAD protocol-control experiment")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Run dir: {run_dir}")
    print(f"Rows: {len(df)}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Validation: {args.validation}")
    print(f"Folds: {len(splits)}")
    print(f"Models: {list(models)}")
    print("Feature groups:")
    for name, cols in feature_groups.items():
        print(f"  {name}: {len(cols)} features")
    print("Target distribution:")
    print(df["stress_binary"].value_counts().sort_index().to_string())
    print("")

    metrics_rows = []
    pred_frames = []

    for feature_group, feature_cols in feature_groups.items():
        X = df[feature_cols].copy()
        print("=" * 80)
        print(f"Feature group: {feature_group} | n_features={len(feature_cols)}")
        print("=" * 80)

        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            print(
                f"[{feature_group}] Fold {fold_idx}/{len(splits)} | "
                f"n_train={len(train_idx)} | n_val={len(val_idx)} | "
                f"train_subjects={len(set(groups[train_idx]))} | val_subjects={len(set(groups[val_idx]))}"
            )

            for model_name, model_template in models.items():
                print(f"  model={model_name}")
                model = clone(model_template)
                model.fit(X_train, y_train)
                y_score = get_scores(model, X_val)
                y_pred = (y_score >= 0.5).astype(int)

                metrics = compute_metrics(y_val, y_pred, y_score)
                row = {
                    "validation": args.validation,
                    "feature_group": feature_group,
                    "model": model_name,
                    "fold": fold_idx,
                    "n_features": int(len(feature_cols)),
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "train_subjects": int(len(set(groups[train_idx]))),
                    "val_subjects": int(len(set(groups[val_idx]))),
                }
                row.update(metrics)
                metrics_rows.append(row)

                if args.save_predictions == "true":
                    pred_frames.append(
                        pd.DataFrame(
                            {
                                "validation": args.validation,
                                "feature_group": feature_group,
                                "model": model_name,
                                "fold": fold_idx,
                                "row_index": val_idx,
                                "subject_id": groups[val_idx],
                                "window_id": df.iloc[val_idx]["window_id"].to_numpy()
                                if "window_id" in df.columns else np.nan,
                                "start_sec": df.iloc[val_idx]["start_sec"].to_numpy()
                                if "start_sec" in df.columns else np.nan,
                                "end_sec": df.iloc[val_idx]["end_sec"].to_numpy()
                                if "end_sec" in df.columns else np.nan,
                                "y_true": y_val,
                                "y_pred": y_pred,
                                "y_score": y_score,
                            }
                        )
                    )

                print(
                    "    "
                    f"balanced_acc={metrics['balanced_accuracy']:.4f} "
                    f"macro_f1={metrics['macro_f1']:.4f} "
                    f"roc_auc={metrics['roc_auc']:.4f} "
                    f"ap={metrics['average_precision']:.4f} "
                    f"f1_stress={metrics['f1_stress']:.4f}"
                )

    fold_metrics = pd.DataFrame(metrics_rows)
    summary = aggregate_metrics(fold_metrics)

    predictions = pd.DataFrame()
    if pred_frames:
        predictions = pd.concat(pred_frames, ignore_index=True)

    thresholds = np.round(np.arange(args.threshold_min, args.threshold_max + 1e-12, args.threshold_step), 6)
    if predictions.empty:
        threshold_df = pd.DataFrame()
        best_th = pd.DataFrame()
    else:
        threshold_df = threshold_sweep(predictions, thresholds)
        best_th = best_thresholds(threshold_df)

    protocol_cmp = build_protocol_comparison(summary, best_th)

    fold_metrics_path = run_dir / "protocol_control_fold_metrics.csv"
    summary_path = run_dir / "protocol_control_summary.csv"
    predictions_path = run_dir / "protocol_control_predictions.parquet"
    threshold_path = run_dir / "threshold_analysis.csv"
    best_threshold_path = run_dir / "best_thresholds.csv"
    protocol_cmp_path = run_dir / "protocol_control_comparison.csv"
    groups_path = run_dir / "protocol_control_feature_groups.json"

    fold_metrics.to_csv(fold_metrics_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    protocol_cmp.to_csv(protocol_cmp_path, index=False, encoding="utf-8")

    if not threshold_df.empty:
        threshold_df.to_csv(threshold_path, index=False, encoding="utf-8")
    if not best_th.empty:
        best_th.to_csv(best_threshold_path, index=False, encoding="utf-8")
    if args.save_predictions == "true" and not predictions.empty:
        predictions.to_parquet(predictions_path, index=False)

    groups_path.write_text(
        json.dumps({k: v for k, v in feature_groups.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    figure_paths = []
    for metric in ["balanced_accuracy_mean", "macro_f1_mean", "roc_auc_mean", "average_precision_mean", "f1_stress_mean"]:
        p = plot_metric_by_group(summary, metric, figures_dir)
        if p:
            figure_paths.append(p)

    if not best_th.empty:
        for metric in ["balanced_accuracy", "macro_f1", "f1_stress"]:
            p = plot_best_threshold_metric(best_th, metric, figures_dir)
            if p:
                figure_paths.append(p)

    p = plot_no_acc_vs_acc(summary, figures_dir)
    if p:
        figure_paths.append(p)

    report_path = run_dir / "report.md"
    report_path.write_text(
        build_report(args, dataset_path, run_dir, df, feature_groups, summary, best_th, protocol_cmp, figure_paths),
        encoding="utf-8",
    )

    source_files = {
        "dataset": str(dataset_path),
        "run_dir": str(run_dir),
        "outputs": {
            "fold_metrics": str(fold_metrics_path),
            "summary": str(summary_path),
            "predictions": str(predictions_path) if args.save_predictions == "true" else None,
            "threshold_analysis": str(threshold_path),
            "best_thresholds": str(best_threshold_path),
            "protocol_control_comparison": str(protocol_cmp_path),
            "feature_groups": str(groups_path),
            "report": str(report_path),
            "figures": [str(p) for p in figure_paths],
        },
    }
    (run_dir / "source_files.json").write_text(json.dumps(source_files, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("=" * 80)
    print("Saved WESAD protocol-control outputs")
    print("=" * 80)
    print(f"Run dir: {run_dir}")
    print(f"Fold metrics: {fold_metrics_path}")
    print(f"Summary: {summary_path}")
    if args.save_predictions == "true":
        print(f"Predictions: {predictions_path}")
    print(f"Threshold analysis: {threshold_path}")
    print(f"Best thresholds: {best_threshold_path}")
    print(f"Protocol comparison: {protocol_cmp_path}")
    print(f"Feature groups: {groups_path}")
    print(f"Report: {report_path}")
    print("")
    print("Protocol-control comparison:")
    print(protocol_cmp.to_string(index=False))
    print("")
    print("Top rows by default balanced accuracy:")
    show_cols = [
        "feature_group",
        "model",
        "n_features",
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "roc_auc_mean",
        "average_precision_mean",
        "f1_stress_mean",
        "recall_stress_mean",
        "precision_stress_mean",
    ]
    print(summary[show_cols].head(20).to_string(index=False))
    print("")
    if not best_th.empty:
        print("Top rows by threshold-optimized balanced accuracy:")
        show_cols2 = [
            "feature_group",
            "model",
            "best_threshold_by_balanced_accuracy",
            "best_balanced_accuracy",
            "at_balanced_accuracy_precision_stress",
            "at_balanced_accuracy_recall_stress",
            "best_threshold_by_f1_stress",
            "best_f1_stress",
        ]
        print(best_th[show_cols2].head(20).to_string(index=False))
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
