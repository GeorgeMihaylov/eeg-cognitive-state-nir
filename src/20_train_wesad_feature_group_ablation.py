#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train WESAD feature-group ablation experiments.

Input:
    data/processed/wesad_windowed_stress_dataset.parquet

Goal:
    Determine which wearable signal groups contribute to stress detection:
        - EDA only
        - BVP only
        - TEMP only
        - ACC only
        - EDA + BVP
        - EDA + BVP + TEMP
        - EDA + BVP + TEMP + ACC
        - all

Validation:
    GroupKFold by subject_id.

Outputs:
    reports/wearable_pm_alignment/runs/<timestamp>_wesad_feature_group_ablation/
        ablation_fold_metrics.csv
        ablation_summary.csv
        ablation_predictions.parquet
        ablation_feature_groups.json
        report.md
        figures/
            balanced_accuracy_by_group.png
            macro_f1_by_group.png
            roc_auc_by_group.png
            average_precision_by_group.png
            f1_stress_by_group.png
            recall_precision_stress_by_group.png
            heatmap_group_model_balanced_accuracy.png
            heatmap_group_model_roc_auc.png

Typical command:
    D:\miniconda3\envs\eeg_nir\python.exe src\20_train_wesad_feature_group_ablation.py `
      --dataset data\processed\wesad_windowed_stress_dataset.parquet `
      --fast `
      --run-name wesad_feature_group_ablation

Fast test:
    D:\miniconda3\envs\eeg_nir\python.exe src\20_train_wesad_feature_group_ablation.py `
      --dataset data\processed\wesad_windowed_stress_dataset.parquet `
      --fast `
      --fold-limit 2 `
      --models logistic_robust,rf_clf `
      --run-name wesad_feature_group_ablation_test
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


DEFAULT_GROUPS = {
    "eda_only": ["eda_"],
    "bvp_only": ["bvp_"],
    "temp_only": ["temp_"],
    "acc_only": ["acc_"],
    "eda_bvp": ["eda_", "bvp_"],
    "eda_bvp_temp": ["eda_", "bvp_", "temp_"],
    "eda_bvp_temp_acc": ["eda_", "bvp_", "temp_", "acc_"],
    "all": ["eda_", "bvp_", "temp_", "acc_"],
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
    feature_cols = []
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if col.startswith("label_count_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return sorted(feature_cols)


def select_feature_group_columns(all_feature_cols: List[str], prefixes: List[str]) -> List[str]:
    selected = []
    for col in all_feature_cols:
        if any(col.startswith(prefix) for prefix in prefixes):
            selected.append(col)
    return sorted(selected)


def build_feature_groups(all_feature_cols: List[str], include_groups: Optional[str]) -> Dict[str, List[str]]:
    groups = {}

    requested = None
    if include_groups:
        requested = [g.strip() for g in include_groups.split(",") if g.strip()]

    for group_name, prefixes in DEFAULT_GROUPS.items():
        if requested is not None and group_name not in requested:
            continue

        cols = select_feature_group_columns(all_feature_cols, prefixes)
        if cols:
            groups[group_name] = cols

    if requested is not None:
        missing = [g for g in requested if g not in groups]
        if missing:
            raise ValueError(
                f"Requested groups not available or empty: {missing}. "
                f"Available groups: {list(groups)}"
            )

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

    pred = model.predict(X)
    return pred.astype(float)


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


def build_best_by_group(summary_df: pd.DataFrame, metric: str = "balanced_accuracy_mean") -> pd.DataFrame:
    rows = []
    for feature_group, g in summary_df.groupby("feature_group"):
        idx = g[metric].idxmax()
        rows.append(g.loc[idx].to_dict())
    return pd.DataFrame(rows).sort_values(metric, ascending=False)


def build_best_by_model(summary_df: pd.DataFrame, metric: str = "balanced_accuracy_mean") -> pd.DataFrame:
    rows = []
    for model, g in summary_df.groupby("model"):
        idx = g[metric].idxmax()
        rows.append(g.loc[idx].to_dict())
    return pd.DataFrame(rows).sort_values(metric, ascending=False)


def save_bar_metric(summary_df: pd.DataFrame, metric: str, figures_dir: Path) -> Optional[Path]:
    if summary_df.empty or metric not in summary_df.columns:
        return None

    # Use best model per feature group for a clean comparison.
    best = build_best_by_group(summary_df, metric=metric)
    best = best.sort_values(metric, ascending=True)

    labels = best["feature_group"] + "\n(" + best["model"] + ")"
    values = best[metric].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(10, max(5, len(best) * 0.45)))
    ax.barh(labels, values)
    ax.set_title(f"Best {metric} by feature group")
    ax.set_xlabel(metric)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = figures_dir / f"{metric}_by_group.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_recall_precision_plot(summary_df: pd.DataFrame, figures_dir: Path) -> Optional[Path]:
    required = {"recall_stress_mean", "precision_stress_mean", "feature_group", "model"}
    if summary_df.empty or not required.issubset(summary_df.columns):
        return None

    best = build_best_by_group(summary_df, metric="balanced_accuracy_mean")
    best = best.sort_values("balanced_accuracy_mean", ascending=True)

    x = np.arange(len(best))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, best["recall_stress_mean"], width=width, label="recall_stress")
    ax.bar(x + width / 2, best["precision_stress_mean"], width=width, label="precision_stress")

    labels = best["feature_group"] + "\n(" + best["model"] + ")"
    ax.set_title("Stress recall and precision by feature group")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    path = figures_dir / "recall_precision_stress_by_group.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_heatmap(summary_df: pd.DataFrame, metric: str, figures_dir: Path) -> Optional[Path]:
    if summary_df.empty or metric not in summary_df.columns:
        return None

    pivot = summary_df.pivot(index="feature_group", columns="model", values=metric)
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(max(7, len(pivot.columns) * 1.4), max(5, len(pivot) * 0.45)))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")

    ax.set_title(f"{metric}: feature group x model")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    path = figures_dir / f"heatmap_group_model_{metric}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def build_report(
    args: argparse.Namespace,
    dataset_path: Path,
    run_dir: Path,
    df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    best_by_group: pd.DataFrame,
    best_by_model: pd.DataFrame,
    figure_paths: List[Path],
) -> str:
    lines: List[str] = []

    lines.append("# WESAD feature-group ablation report")
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
    target_counts = df["stress_binary"].value_counts().sort_index().reset_index()
    target_counts.columns = ["stress_binary", "n_windows"]
    target_counts["class_name"] = target_counts["stress_binary"].map({0: "non_stress", 1: "stress"})
    lines.append(target_counts[["stress_binary", "class_name", "n_windows"]].to_markdown(index=False))
    lines.append("")
    lines.append("## Feature groups")
    lines.append("")
    group_rows = []
    for group_name, cols in feature_groups.items():
        group_rows.append(
            {
                "feature_group": group_name,
                "n_features": len(cols),
                "prefix_examples": ", ".join(sorted(set(c.split("_")[0] for c in cols))),
            }
        )
    lines.append(pd.DataFrame(group_rows).to_markdown(index=False))
    lines.append("")
    lines.append("## Best result by feature group")
    lines.append("")
    lines.append(best_by_group.to_markdown(index=False))
    lines.append("")
    lines.append("## Best feature group by model")
    lines.append("")
    lines.append(best_by_model.to_markdown(index=False))
    lines.append("")
    lines.append("## Full summary")
    lines.append("")
    lines.append(summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Fold metrics")
    lines.append("")
    lines.append(fold_metrics.to_markdown(index=False))
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for path in figure_paths:
        try:
            rel = path.relative_to(run_dir)
        except Exception:
            rel = path
        lines.append(f"- `{rel}`")
    lines.append("")
    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- `EDA-only` estimates the direct electrodermal stress signal.")
    lines.append("- `BVP-only` tests pulse-wave / cardiovascular contribution.")
    lines.append("- `TEMP-only` may capture slow autonomic or protocol-related differences.")
    lines.append("- `ACC-only` is mainly a movement/context control; high ACC-only performance may indicate task/movement confounding.")
    lines.append("- `EDA+BVP+TEMP+ACC` should be treated as the full wrist-physiology baseline.")
    lines.append("- If a single group is close to the full model, the downstream PM-alignment pipeline can start from that signal group.")
    lines.append("")
    lines.append("## Recommended next step")
    lines.append("")
    lines.append("If WESAD ablation confirms that stress is mainly explained by EDA/BVP/TEMP, then proceed to:")
    lines.append("")
    lines.append("```text")
    lines.append("21_prepare_colet_dataset.py")
    lines.append("22_train_colet_workload_baseline.py")
    lines.append("```")
    lines.append("")
    lines.append("If ACC-only is unexpectedly strong, inspect protocol confounding before using WESAD as evidence for physiological stress detection.")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WESAD feature-group ablation experiments.")

    parser.add_argument("--root", type=str, default=".", help="Project root.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/wesad_windowed_stress_dataset.parquet",
        help="Prepared WESAD windowed dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="reports/wearable_pm_alignment/runs",
        help="Run output root.",
    )
    parser.add_argument("--run-name", type=str, default="wesad_feature_group_ablation")

    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names.")
    parser.add_argument("--groups", type=str, default=None, help="Comma-separated feature group names.")
    parser.add_argument("--fast", action="store_true", help="Use faster model settings.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold-limit", type=int, default=0, help="0 means all folds.")
    parser.add_argument(
        "--validation",
        type=str,
        default="groupkfold_subject",
        choices=["groupkfold_subject", "random_split"],
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--save-predictions",
        type=str,
        default="true",
        choices=["true", "false"],
    )

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
        raise ValueError("Dataset must contain stress_binary target column.")
    if "subject_id" not in df.columns:
        raise ValueError("Dataset must contain subject_id column.")

    all_feature_cols = infer_all_feature_columns(df)
    feature_groups = build_feature_groups(all_feature_cols, args.groups)

    if not feature_groups:
        raise RuntimeError("No feature groups selected.")

    models = build_models(
        fast=args.fast,
        selected_models=args.models,
        random_state=args.random_state,
    )

    y = df["stress_binary"].astype(int).to_numpy()
    groups = df["subject_id"].astype(str).to_numpy()

    if args.validation == "groupkfold_subject":
        n_splits = min(args.n_splits, df["subject_id"].nunique())
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(df, y, groups=groups))
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=args.random_state,
        )
        splits = list(splitter.split(df, y))

    if args.fold_limit and args.fold_limit > 0:
        splits = splits[: args.fold_limit]

    print("=" * 80)
    print("Train WESAD feature-group ablation")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Run dir: {run_dir}")
    print(f"Rows: {len(df)}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Validation: {args.validation}")
    print(f"Folds: {len(splits)}")
    print(f"Models: {list(models)}")
    print("Feature groups:")
    for group_name, cols in feature_groups.items():
        print(f"  {group_name}: {len(cols)} features")
    print("Target distribution:")
    print(df["stress_binary"].value_counts().sort_index().to_string())
    print("")

    metrics_rows: List[Dict[str, object]] = []
    pred_frames: List[pd.DataFrame] = []

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
                row: Dict[str, object] = {
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
                    pred_df = pd.DataFrame(
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
                    pred_frames.append(pred_df)

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
    best_by_group = build_best_by_group(summary, metric="balanced_accuracy_mean")
    best_by_model = build_best_by_model(summary, metric="balanced_accuracy_mean")

    fold_metrics_path = run_dir / "ablation_fold_metrics.csv"
    summary_path = run_dir / "ablation_summary.csv"
    best_by_group_path = run_dir / "best_by_feature_group.csv"
    best_by_model_path = run_dir / "best_by_model.csv"
    groups_path = run_dir / "ablation_feature_groups.json"
    predictions_path = run_dir / "ablation_predictions.parquet"

    fold_metrics.to_csv(fold_metrics_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    best_by_group.to_csv(best_by_group_path, index=False, encoding="utf-8")
    best_by_model.to_csv(best_by_model_path, index=False, encoding="utf-8")

    groups_path.write_text(
        json.dumps(
            {group: cols for group, cols in feature_groups.items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.save_predictions == "true" and pred_frames:
        predictions = pd.concat(pred_frames, ignore_index=True)
        predictions.to_parquet(predictions_path, index=False)

    figure_paths: List[Path] = []

    for metric in [
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "roc_auc_mean",
        "average_precision_mean",
        "f1_stress_mean",
    ]:
        p = save_bar_metric(summary, metric, figures_dir)
        if p:
            figure_paths.append(p)

    p = save_recall_precision_plot(summary, figures_dir)
    if p:
        figure_paths.append(p)

    for metric in ["balanced_accuracy_mean", "roc_auc_mean", "f1_stress_mean"]:
        p = save_heatmap(summary, metric, figures_dir)
        if p:
            figure_paths.append(p)

    report_path = run_dir / "report.md"
    report_path.write_text(
        build_report(
            args=args,
            dataset_path=dataset_path,
            run_dir=run_dir,
            df=df,
            feature_groups=feature_groups,
            fold_metrics=fold_metrics,
            summary=summary,
            best_by_group=best_by_group,
            best_by_model=best_by_model,
            figure_paths=figure_paths,
        ),
        encoding="utf-8",
    )

    source_files = {
        "dataset": str(dataset_path),
        "run_dir": str(run_dir),
        "outputs": {
            "fold_metrics": str(fold_metrics_path),
            "summary": str(summary_path),
            "best_by_group": str(best_by_group_path),
            "best_by_model": str(best_by_model_path),
            "feature_groups": str(groups_path),
            "predictions": str(predictions_path) if args.save_predictions == "true" else None,
            "report": str(report_path),
            "figures": [str(p) for p in figure_paths],
        },
    }
    (run_dir / "source_files.json").write_text(
        json.dumps(source_files, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("")
    print("=" * 80)
    print("Saved WESAD feature-group ablation outputs")
    print("=" * 80)
    print(f"Run dir: {run_dir}")
    print(f"Fold metrics: {fold_metrics_path}")
    print(f"Summary: {summary_path}")
    print(f"Best by group: {best_by_group_path}")
    print(f"Best by model: {best_by_model_path}")
    print(f"Feature groups: {groups_path}")
    if args.save_predictions == "true":
        print(f"Predictions: {predictions_path}")
    print(f"Report: {report_path}")
    print("")
    print("Best by feature group:")
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
    print(best_by_group[show_cols].to_string(index=False))
    print("")
    print("Top combinations overall:")
    print(summary[show_cols].head(15).to_string(index=False))
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
