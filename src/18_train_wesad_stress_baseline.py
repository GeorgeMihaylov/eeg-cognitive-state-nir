#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train WESAD wearable stress baselines.

Input:
    data/processed/wesad_windowed_stress_dataset.parquet

Target:
    stress_binary:
        0 -> non-stress
        1 -> stress

Validation:
    GroupKFold by subject_id by default.

Models:
    logistic_robust
    hgb_clf
    rf_clf
    lgbm_clf, if lightgbm is installed

Outputs:
    reports/wearable_pm_alignment/runs/<run_id>/
        fold_metrics.csv
        metrics_summary.csv
        predictions.parquet
        feature_importance.csv
        report.md
        figures/
            metric_by_model.png
            confusion_matrix_<model>.png
            roc_curve_<model>.png
            pr_curve_<model>.png

Typical command:
    D:\miniconda3\envs\eeg_nir\python.exe src\18_train_wesad_stress_baseline.py

Fast test:
    D:\miniconda3\envs\eeg_nir\python.exe src\18_train_wesad_stress_baseline.py `
      --dataset data\processed\wesad_windowed_stress_dataset.parquet `
      --fold-limit 2 `
      --fast `
      --run-name wesad_stress_test
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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


def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    feature_cols = []
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if col.startswith("label_count_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return sorted(feature_cols)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported dataset format: {path.suffix}")


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
    for model, g in metrics_df.groupby("model"):
        row = {
            "model": model,
            "folds": int(g["fold"].nunique()),
            "n_val_total": int(g["n_val"].sum()),
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


def plot_metric_by_model(summary_df: pd.DataFrame, figures_dir: Path) -> Optional[Path]:
    if summary_df.empty:
        return None

    metrics = ["balanced_accuracy_mean", "macro_f1_mean", "roc_auc_mean", "average_precision_mean"]
    plot_df = summary_df[["model"] + metrics].copy()
    plot_df = plot_df.set_index("model")

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("WESAD stress baseline metrics by model")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = figures_dir / "metric_by_model.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_confusion_matrix_for_model(pred_df: pd.DataFrame, model: str, figures_dir: Path) -> Optional[Path]:
    g = pred_df[pred_df["model"] == model]
    if g.empty:
        return None

    cm = confusion_matrix(g["y_true"], g["y_pred"], labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title(f"Confusion matrix: {model}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["non-stress", "stress"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["non-stress", "stress"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    path = figures_dir / f"confusion_matrix_{model}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_roc_pr_for_model(pred_df: pd.DataFrame, model: str, figures_dir: Path) -> List[Path]:
    paths: List[Path] = []
    g = pred_df[pred_df["model"] == model]
    if g.empty or g["y_true"].nunique() < 2:
        return paths

    y_true = g["y_true"].to_numpy()
    y_score = g["y_score"].to_numpy()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC={roc_auc_score(y_true, y_score):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(f"ROC curve: {model}")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = figures_dir / f"roc_curve_{model}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP={ap:.3f}")
    ax.set_title(f"Precision-recall curve: {model}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = figures_dir / f"pr_curve_{model}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    return paths


def extract_feature_importance(model: object, model_name: str, feature_cols: List[str]) -> pd.DataFrame:
    # Pipeline: last step named model
    est = model
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        est = model.named_steps["model"]

    rows = []

    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
        for feature, value in zip(feature_cols, imp):
            rows.append({"model": model_name, "feature": feature, "importance": float(value), "importance_type": "feature_importances"})
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_).reshape(-1)
        for feature, value in zip(feature_cols, coef):
            rows.append({"model": model_name, "feature": feature, "importance": float(abs(value)), "importance_type": "abs_coef"})

    if not rows:
        return pd.DataFrame(columns=["model", "feature", "importance", "importance_type"])

    return pd.DataFrame(rows).sort_values(["model", "importance"], ascending=[True, False])


def build_report(
    args: argparse.Namespace,
    dataset_path: Path,
    run_dir: Path,
    df: pd.DataFrame,
    feature_cols: List[str],
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    figure_paths: List[Path],
) -> str:
    lines = []
    lines.append("# WESAD wearable stress baseline report")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(vars(args), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Rows: **{len(df)}**")
    lines.append(f"- Columns: **{df.shape[1]}**")
    lines.append(f"- Subjects: **{df['subject_id'].nunique()}**")
    lines.append(f"- Feature columns: **{len(feature_cols)}**")
    lines.append("")
    lines.append("## Target distribution")
    lines.append("")
    counts = df["stress_binary"].value_counts().sort_index().reset_index()
    counts.columns = ["stress_binary", "n_windows"]
    counts["class_name"] = counts["stress_binary"].map({0: "non-stress", 1: "stress"})
    lines.append(counts[["stress_binary", "class_name", "n_windows"]].to_markdown(index=False))
    lines.append("")
    lines.append("## Metrics by fold")
    lines.append("")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(summary_df.to_markdown(index=False))
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
    lines.append("- Use `balanced_accuracy`, `macro_f1`, `roc_auc` and `average_precision` as primary metrics because the target is imbalanced.")
    lines.append("- Validation is subject-aware, so windows from the same subject do not appear in both train and validation folds.")
    lines.append("- This is an external wearable baseline. It is not directly comparable to EEG PM prediction yet, but it tests whether EDA/BVP/TEMP/ACC can recover stress-like state.")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WESAD wearable stress baselines.")

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
    parser.add_argument("--run-name", type=str, default="wesad_stress_baseline")

    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names.")
    parser.add_argument("--fast", action="store_true", help="Use faster model settings.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold-limit", type=int, default=0, help="0 means all folds.")
    parser.add_argument("--validation", type=str, default="groupkfold_subject", choices=["groupkfold_subject", "random_split"])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-predictions", type=str, default="true", choices=["true", "false"])

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
        raise ValueError("Dataset must contain subject_id column for GroupKFold.")

    feature_cols = infer_feature_columns(df)
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found.")

    X = df[feature_cols].copy()
    y = df["stress_binary"].astype(int).to_numpy()
    groups = df["subject_id"].astype(str).to_numpy()

    models = build_models(
        fast=args.fast,
        selected_models=args.models,
        random_state=args.random_state,
    )

    print("=" * 80)
    print("Train WESAD wearable stress baselines")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Run dir: {run_dir}")
    print(f"Rows: {len(df)}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Features: {len(feature_cols)}")
    print(f"Validation: {args.validation}")
    print(f"Models: {list(models)}")
    print("Target distribution:")
    print(df["stress_binary"].value_counts().sort_index().to_string())
    print("")

    if args.validation == "groupkfold_subject":
        n_splits = min(args.n_splits, df["subject_id"].nunique())
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(X, y, groups=groups))
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.random_state)
        splits = list(splitter.split(X, y))

    if args.fold_limit and args.fold_limit > 0:
        splits = splits[: args.fold_limit]

    metrics_rows = []
    pred_rows = []
    feature_importance_frames = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        print("-" * 80)
        print(
            f"Fold {fold_idx}/{len(splits)} | "
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
                "fold": fold_idx,
                "model": model_name,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "train_subjects": int(len(set(groups[train_idx]))),
                "val_subjects": int(len(set(groups[val_idx]))),
            }
            row.update(metrics)
            metrics_rows.append(row)

            if args.save_predictions == "true":
                fold_pred = pd.DataFrame(
                    {
                        "validation": args.validation,
                        "fold": fold_idx,
                        "model": model_name,
                        "row_index": val_idx,
                        "subject_id": groups[val_idx],
                        "window_id": df.iloc[val_idx]["window_id"].to_numpy() if "window_id" in df.columns else np.nan,
                        "start_sec": df.iloc[val_idx]["start_sec"].to_numpy() if "start_sec" in df.columns else np.nan,
                        "end_sec": df.iloc[val_idx]["end_sec"].to_numpy() if "end_sec" in df.columns else np.nan,
                        "y_true": y_val,
                        "y_pred": y_pred,
                        "y_score": y_score,
                    }
                )
                pred_rows.append(fold_pred)

            fi = extract_feature_importance(model, model_name, feature_cols)
            if not fi.empty:
                fi["fold"] = fold_idx
                feature_importance_frames.append(fi)

            print(
                "    "
                f"balanced_acc={metrics['balanced_accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f} "
                f"roc_auc={metrics['roc_auc']:.4f} "
                f"ap={metrics['average_precision']:.4f}"
            )

    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = aggregate_metrics(metrics_df)

    metrics_path = run_dir / "fold_metrics.csv"
    summary_path = run_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    pred_df = pd.DataFrame()
    predictions_path = run_dir / "predictions.parquet"
    if pred_rows and args.save_predictions == "true":
        pred_df = pd.concat(pred_rows, ignore_index=True)
        pred_df.to_parquet(predictions_path, index=False)

    feature_importance_path = run_dir / "feature_importance.csv"
    if feature_importance_frames:
        fi_all = pd.concat(feature_importance_frames, ignore_index=True)
        fi_summary = (
            fi_all.groupby(["model", "feature", "importance_type"], as_index=False)["importance"]
            .mean()
            .sort_values(["model", "importance"], ascending=[True, False])
        )
        fi_summary.to_csv(feature_importance_path, index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=["model", "feature", "importance", "importance_type"]).to_csv(
            feature_importance_path, index=False, encoding="utf-8"
        )

    figure_paths: List[Path] = []
    p = plot_metric_by_model(summary_df, figures_dir)
    if p:
        figure_paths.append(p)

    if not pred_df.empty:
        for model_name in summary_df["model"].tolist():
            p = plot_confusion_matrix_for_model(pred_df, model_name, figures_dir)
            if p:
                figure_paths.append(p)
            figure_paths.extend(plot_roc_pr_for_model(pred_df, model_name, figures_dir))

    report_path = run_dir / "report.md"
    report_path.write_text(
        build_report(args, dataset_path, run_dir, df, feature_cols, metrics_df, summary_df, figure_paths),
        encoding="utf-8",
    )

    source_info = {
        "dataset": str(dataset_path),
        "run_dir": str(run_dir),
        "metrics": str(metrics_path),
        "summary": str(summary_path),
        "predictions": str(predictions_path) if args.save_predictions == "true" else None,
        "feature_importance": str(feature_importance_path),
        "report": str(report_path),
        "figures": [str(p) for p in figure_paths],
    }
    (run_dir / "source_files.json").write_text(json.dumps(source_info, indent=2, ensure_ascii=False), encoding="utf-8")

    print("")
    print("=" * 80)
    print("Saved WESAD baseline outputs")
    print("=" * 80)
    print(f"Run dir: {run_dir}")
    print(f"Fold metrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    if args.save_predictions == "true":
        print(f"Predictions: {predictions_path}")
    print(f"Feature importance: {feature_importance_path}")
    print(f"Report: {report_path}")
    print("")
    print("Summary:")
    show_cols = [
        "model",
        "folds",
        "n_val_total",
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "roc_auc_mean",
        "average_precision_mean",
        "f1_stress_mean",
        "recall_stress_mean",
        "precision_stress_mean",
    ]
    print(summary_df[show_cols].to_string(index=False))
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
