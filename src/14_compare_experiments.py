#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare EEG PM prediction experiments.

Purpose:
    Collect and compare main experiments:
        1. tabular baseline: X_t -> PM_t
        2. context-tabular baseline: concat(X_{t-1}, X_t, X_{t+1}) -> PM_t
        3. MHA seq_len=3
        4. MHA seq_len=5 short

The script is tolerant to different result-file formats:
    - *_regression_metrics_agg.csv
    - all_targets_summary.csv
    - all_targets_fold_metrics.csv
    - per-target fold metrics
    - CSV files inside run directories

Typical command:
    D:\\miniconda3\\envs\\eeg_nir\\python.exe src\\14_compare_experiments.py `
      --tabular data\\processed\\baseline_pow_plus_eeg_w10_log_regression_metrics_agg.csv `
      --context reports\\runs\\20260512_144503_context_tabular_len3_fast_all_pow_plus_eeg_len3 `
      --mha-len3 reports\\runs\\<mha_len3_run_dir> `
      --mha-len5 reports\\runs\\<mha_len5_short_run_dir> `
      --output-dir reports\\comparison\\final_pm_experiment_comparison
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PM_TARGET_BY_COL = {
    "PM.Attention.Scaled__mean": "attention",
    "PM.Engagement.Scaled__mean": "engagement",
    "PM.Excitement.Scaled__mean": "excitement",
    "PM.Stress.Scaled__mean": "stress",
    "PM.Relaxation.Scaled__mean": "relaxation",
    "PM.Interest.Scaled__mean": "interest",
    "PM.Focus.Scaled__mean": "focus",
}

PM_ORDER = [
    "attention",
    "engagement",
    "excitement",
    "stress",
    "relaxation",
    "interest",
    "focus",
]

METRICS = ["mae", "rmse", "r2", "pearson", "spearman"]
LOWER_IS_BETTER = {"mae", "rmse"}


@dataclass
class ExperimentSpec:
    name: str
    path: Optional[Path]
    model_filter: Optional[str] = None
    validation_filter: Optional[str] = None
    default_target: Optional[str] = None


def resolve_path(root: Path, value: Optional[str]) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return root / p


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_target_name(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    text = str(value).strip()
    if not text:
        return None

    if text in PM_TARGET_BY_COL:
        return PM_TARGET_BY_COL[text]

    low = text.lower().strip()
    low = low.replace("pm.", "")
    low = low.replace(".scaled__mean", "")
    low = low.replace(".scaled", "")
    low = low.replace("__mean", "")
    low = low.replace("_mean", "")
    low = low.replace("target_", "")
    low = low.replace("pm_", "")
    low = re.sub(r"[^a-z0-9]+", "_", low).strip("_")

    aliases = {
        "att": "attention",
        "attention": "attention",
        "eng": "engagement",
        "engagement": "engagement",
        "exc": "excitement",
        "excitement": "excitement",
        "stress": "stress",
        "relax": "relaxation",
        "relaxation": "relaxation",
        "interest": "interest",
        "focus": "focus",
    }
    return aliases.get(low, low)


def find_candidate_csvs(path: Path) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_file():
        return [path]

    priority_names = [
        "all_targets_summary.csv",
        "summary.csv",
        "metrics_summary.csv",
        "regression_metrics_agg.csv",
        "all_targets_fold_metrics.csv",
    ]

    candidates: List[Path] = []
    for name in priority_names:
        candidates.extend(sorted(path.glob(name)))

    candidates.extend(sorted(path.glob("*regression_metrics_agg.csv")))
    candidates.extend(sorted(path.glob("*summary.csv")))
    candidates.extend(sorted(path.glob("*fold_metrics.csv")))
    candidates.extend(sorted(path.glob("*.csv")))

    seen = set()
    unique = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def read_best_csv(path: Path) -> Tuple[pd.DataFrame, Path]:
    candidates = find_candidate_csvs(path)
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in: {path}")

    scored = []
    for p in candidates:
        try:
            df = pd.read_csv(p)
            cols = set(df.columns)
            score = 0
            if "target" in cols or "target_col" in cols or "pm_target" in cols:
                score += 10
            if any(c.endswith("_mean") for c in cols):
                score += 6
            if {"r2", "spearman", "rmse"}.intersection(cols):
                score += 5
            if "model" in cols:
                score += 3
            if "fold" in cols:
                score += 1
            if p.name == "all_targets_summary.csv":
                score += 10
            if "regression_metrics_agg" in p.name:
                score += 8
            scored.append((score, p, df))
        except Exception:
            continue

    if not scored:
        raise RuntimeError(f"Could not read any CSV from: {path}")

    scored.sort(key=lambda x: (x[0], -len(x[2])), reverse=True)
    _, best_path, best_df = scored[0]
    return best_df, best_path


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["target", "pm_target", "target_name", "metric", "pm_metric"]:
        if col in df.columns:
            return col
    if "target_col" in df.columns:
        return "target_col"
    return None


def detect_validation_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["validation", "validation_mode", "split", "eval_mode"]:
        if col in df.columns:
            return col
    return None


def detect_model_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["model", "model_name", "estimator"]:
        if col in df.columns:
            return col
    return None


def standardize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for metric in METRICS:
        if f"{metric}_mean" in out.columns:
            continue
        candidates = [metric, metric.upper(), metric.capitalize(), f"{metric}_avg", f"mean_{metric}"]
        for c in candidates:
            if c in out.columns:
                rename[c] = f"{metric}_mean"
                break
    return out.rename(columns=rename)


def aggregate_fold_metrics(df: pd.DataFrame) -> pd.DataFrame:
    target_col = detect_target_column(df)
    model_col = detect_model_column(df)
    validation_col = detect_validation_column(df)

    if target_col is None:
        raise ValueError("Cannot aggregate fold metrics: target column was not found.")
    if model_col is None:
        df = df.copy()
        df["model"] = "unknown_model"
        model_col = "model"
    if validation_col is None:
        df = df.copy()
        df["validation"] = "unknown_validation"
        validation_col = "validation"

    group_cols = [target_col, model_col, validation_col]
    group_cols.extend([c for c in ["seq_len", "feature_set", "feature_mode", "task"] if c in df.columns])

    metric_cols = [m for m in METRICS if m in df.columns]
    if not metric_cols:
        metric_cols = [f"{m}_mean" for m in METRICS if f"{m}_mean" in df.columns]
        if metric_cols:
            return df

    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["folds"] = int(grp["fold"].nunique()) if "fold" in grp.columns else int(len(grp))
        row["n_val_total"] = int(grp["n_val"].sum()) if "n_val" in grp.columns else int(len(grp))
        for metric in metric_cols:
            values = pd.to_numeric(grp[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_experiment_df(
    df: pd.DataFrame,
    experiment_name: str,
    validation_filter: Optional[str],
    model_filter: Optional[str],
    default_target: Optional[str] = None,
) -> pd.DataFrame:
    df = df.copy()
    has_mean = any(f"{m}_mean" in df.columns for m in METRICS)
    has_raw_metric = any(m in df.columns for m in METRICS)
    if not has_mean and has_raw_metric:
        df = aggregate_fold_metrics(df)

    df = standardize_metric_columns(df)

    target_col = detect_target_column(df)
    if target_col is None:
        # Legacy single-target files from src/07_train_baselines.py do not store target name.
        # In this project those regression baselines used target_main, i.e. focus by default.
        fallback_target = normalize_target_name(default_target or "focus")
        df = df.copy()
        df["target"] = fallback_target
        target_col = "target"

    model_col = detect_model_column(df)
    if model_col is None:
        df["model"] = "unknown_model"
        model_col = "model"

    validation_col = detect_validation_column(df)
    if validation_col is None:
        df["validation"] = "unknown_validation"
        validation_col = "validation"

    out = df.copy()
    out["target"] = out[target_col].map(normalize_target_name)
    out["model"] = out[model_col].astype(str)
    out["validation"] = out[validation_col].astype(str)
    out["experiment"] = experiment_name
    out = out[out["target"].notna()].copy()

    if "task" in out.columns:
        task_str = out["task"].astype(str).str.lower()
        if task_str.str.contains("regression").any():
            out = out[task_str.str.contains("regression")].copy()

    if validation_filter:
        vf = validation_filter.lower()
        out = out[out["validation"].str.lower().str.contains(vf, regex=False)].copy()
    else:
        val_low = out["validation"].str.lower()
        subject_mask = val_low.str.contains("groupkfold") | val_low.str.contains("subject")
        if subject_mask.any():
            out = out[subject_mask].copy()

    if model_filter:
        requested = [x.strip() for x in model_filter.split(",") if x.strip()]
        out = out[out["model"].isin(requested)].copy()

    for metric in METRICS:
        col = f"{metric}_mean"
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "folds" not in out.columns:
        out["folds"] = np.nan
    if "n_val_total" not in out.columns:
        out["n_val_total"] = np.nan

    keep_cols = ["experiment", "target", "model", "validation", "folds", "n_val_total"] + [f"{m}_mean" for m in METRICS]
    keep_cols.extend([c for c in ["seq_len", "feature_set", "feature_mode"] if c in out.columns])
    return out[keep_cols].reset_index(drop=True)


def choose_best_per_target(df: pd.DataFrame, primary_metric: str, secondary_metric: str) -> pd.DataFrame:
    if df.empty:
        return df
    primary_col = f"{primary_metric}_mean"
    secondary_col = f"{secondary_metric}_mean"
    if secondary_col not in df.columns:
        df[secondary_col] = np.nan

    rows = []
    for _, grp in df.groupby("target", dropna=False):
        sorted_grp = grp.sort_values(
            [primary_col, secondary_col],
            ascending=[primary_metric in LOWER_IS_BETTER, secondary_metric in LOWER_IS_BETTER],
            na_position="last",
        )
        rows.append(sorted_grp.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def sanitize_experiment_prefix(name: str) -> str:
    text = str(name).lower().strip()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def add_experiment_prefix(df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    out = pd.DataFrame()
    prefix = sanitize_experiment_prefix(experiment_name)
    out["target"] = df["target"]
    out[f"{prefix}_model"] = df["model"]
    out[f"{prefix}_validation"] = df["validation"]
    out[f"{prefix}_folds"] = df["folds"]
    out[f"{prefix}_n_val_total"] = df["n_val_total"]
    for metric in METRICS:
        out[f"{prefix}_{metric}"] = df[f"{metric}_mean"]
    return out


def merge_best_tables(best_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for exp_name, df in best_tables.items():
        prefixed = add_experiment_prefix(df, exp_name)
        merged = prefixed if merged is None else merged.merge(prefixed, on="target", how="outer")
    if merged is None:
        return pd.DataFrame()
    merged["target_order"] = merged["target"].map({t: i for i, t in enumerate(PM_ORDER)}).fillna(999)
    return merged.sort_values(["target_order", "target"]).drop(columns=["target_order"]).reset_index(drop=True)


def add_deltas(comparison: pd.DataFrame) -> pd.DataFrame:
    out = comparison.copy()
    pairs = [
        ("context", "tabular", "context_vs_tabular"),
        ("mha_len3", "context", "mha_len3_vs_context"),
        ("mha_len3", "tabular", "mha_len3_vs_tabular"),
        ("mha_len5", "mha_len3", "mha_len5_vs_mha_len3"),
        ("mha_len5", "context", "mha_len5_vs_context"),
    ]
    for left, right, name in pairs:
        for metric in METRICS:
            left_col = f"{left}_{metric}"
            right_col = f"{right}_{metric}"
            if left_col in out.columns and right_col in out.columns:
                out[f"delta_{name}_{metric}"] = out[left_col] - out[right_col]
    return out


def make_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data."
    table_df = df.copy()
    for col in table_df.columns:
        if pd.api.types.is_float_dtype(table_df[col]):
            table_df[col] = table_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return table_df.to_markdown(index=False)


def save_json(path: Path, data: Dict[str, object]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_metric_comparison(comparison: pd.DataFrame, output_dir: Path, metric: str) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    experiments = [p for p in ["tabular", "context", "mha_len3", "mha_len5"] if f"{p}_{metric}" in comparison.columns]
    if not experiments:
        return None
    targets = comparison["target"].tolist()
    x = np.arange(len(targets))
    width = 0.8 / max(len(experiments), 1)
    plt.figure(figsize=(12, 6))
    for i, exp in enumerate(experiments):
        values = pd.to_numeric(comparison[f"{exp}_{metric}"], errors="coerce").to_numpy()
        plt.bar(x + (i - (len(experiments) - 1) / 2) * width, values, width=width, label=exp)
    plt.xticks(x, targets, rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"Experiment comparison: {metric}")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "figures" / f"{metric}_comparison.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_delta(comparison: pd.DataFrame, output_dir: Path, metric: str) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    delta_cols = [c for c in comparison.columns if c.startswith("delta_") and c.endswith(f"_{metric}")]
    if not delta_cols:
        return None
    targets = comparison["target"].tolist()
    x = np.arange(len(targets))
    width = 0.8 / max(len(delta_cols), 1)
    plt.figure(figsize=(13, 6))
    for i, col in enumerate(delta_cols):
        label = col.replace("delta_", "").replace(f"_{metric}", "")
        values = pd.to_numeric(comparison[col], errors="coerce").to_numpy()
        plt.bar(x + (i - (len(delta_cols) - 1) / 2) * width, values, width=width, label=label)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, targets, rotation=45, ha="right")
    plt.ylabel(f"delta {metric}")
    plt.title(f"Metric deltas: {metric}")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "figures" / f"delta_{metric}.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_heatmap(comparison: pd.DataFrame, output_dir: Path, metric: str) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    experiments = [p for p in ["tabular", "context", "mha_len3", "mha_len5"] if f"{p}_{metric}" in comparison.columns]
    if not experiments:
        return None
    matrix = [pd.to_numeric(comparison[f"{exp}_{metric}"], errors="coerce").to_numpy() for exp in experiments]
    data = np.vstack(matrix)
    plt.figure(figsize=(12, 4 + 0.35 * len(experiments)))
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, fraction=0.03, pad=0.04)
    plt.yticks(np.arange(len(experiments)), experiments)
    plt.xticks(np.arange(len(comparison)), comparison["target"].tolist(), rotation=45, ha="right")
    plt.title(f"Target × experiment heatmap: {metric}")
    plt.tight_layout()
    path = output_dir / "figures" / f"target_experiment_heatmap_{metric}.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def save_plots(comparison: pd.DataFrame, output_dir: Path) -> List[Path]:
    ensure_dir(output_dir / "figures")
    paths = []
    for metric in ["r2", "spearman", "rmse", "mae"]:
        for fn in [plot_metric_comparison, plot_delta]:
            p = fn(comparison, output_dir, metric)
            if p:
                paths.append(p)
    for metric in ["r2", "spearman"]:
        p = plot_heatmap(comparison, output_dir, metric)
        if p:
            paths.append(p)
    return paths


def load_experiment(spec: ExperimentSpec) -> Tuple[pd.DataFrame, Optional[str]]:
    if spec.path is None:
        return pd.DataFrame(), None

    raw_df, source_path = read_best_csv(spec.path)
    norm_df = normalize_experiment_df(
        raw_df,
        experiment_name=spec.name,
        validation_filter=spec.validation_filter,
        model_filter=spec.model_filter,
        default_target=spec.default_target,
    )

    if norm_df.empty:
        raise RuntimeError(f"No usable rows after normalization for experiment={spec.name}, source={source_path}")

    return norm_df, str(source_path)


def save_report(
    output_dir: Path,
    args: argparse.Namespace,
    source_files: Dict[str, str],
    best_all: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Final experiment comparison")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(vars(args), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Source files")
    lines.append("")
    for name, path in source_files.items():
        lines.append(f"- **{name}**: `{path}`")
    lines.append("")
    lines.append("## Best model per target and experiment")
    lines.append("")
    lines.append(make_markdown_table(best_all))
    lines.append("")
    lines.append("## Final comparison table")
    lines.append("")
    focus_cols = [
        "target",
        "tabular_model", "tabular_r2", "tabular_spearman",
        "context_model", "context_r2", "context_spearman",
        "mha_len3_model", "mha_len3_r2", "mha_len3_spearman",
        "mha_len5_model", "mha_len5_r2", "mha_len5_spearman",
        "delta_context_vs_tabular_r2",
        "delta_mha_len3_vs_context_r2",
        "delta_mha_len3_vs_tabular_r2",
        "delta_mha_len5_vs_mha_len3_r2",
    ]
    focus_cols = [c for c in focus_cols if c in comparison.columns]
    lines.append(make_markdown_table(comparison[focus_cols] if focus_cols else comparison))
    lines.append("")
    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- Positive `delta_*_r2` means the left experiment has higher R2 than the right experiment.")
    lines.append("- Positive `delta_*_spearman` means the left experiment has higher rank correlation.")
    lines.append("- For `mae` and `rmse`, negative deltas are better because lower error is better.")
    lines.append("- Main question: whether MHA improves over `context-tabular`; if not, the gain is mostly explained by temporal context itself.")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tabular, context-tabular and MHA experiments.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="reports/comparison/final_pm_experiment_comparison")
    parser.add_argument("--tabular", type=str, default=None)
    parser.add_argument("--context", type=str, default=None)
    parser.add_argument("--mha-len3", type=str, default=None)
    parser.add_argument("--mha-len5", type=str, default=None)
    parser.add_argument("--tabular-models", type=str, default=None)
    parser.add_argument("--context-models", type=str, default=None)
    parser.add_argument("--mha-len3-models", type=str, default=None)
    parser.add_argument("--mha-len5-models", type=str, default=None)
    parser.add_argument("--tabular-validation", type=str, default=None)
    parser.add_argument("--context-validation", type=str, default=None)
    parser.add_argument("--mha-len3-validation", type=str, default=None)
    parser.add_argument("--mha-len5-validation", type=str, default=None)
    parser.add_argument("--primary-metric", type=str, default="r2", choices=METRICS)
    parser.add_argument("--secondary-metric", type=str, default="spearman", choices=METRICS)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = resolve_path(root, args.output_dir)
    assert output_dir is not None
    ensure_dir(output_dir)
    ensure_dir(output_dir / "figures")

    specs = [
        ExperimentSpec("tabular", resolve_path(root, args.tabular), args.tabular_models, args.tabular_validation),
        ExperimentSpec("context", resolve_path(root, args.context), args.context_models, args.context_validation),
        ExperimentSpec("mha_len3", resolve_path(root, args.mha_len3), args.mha_len3_models, args.mha_len3_validation),
        ExperimentSpec("mha_len5", resolve_path(root, args.mha_len5), args.mha_len5_models, args.mha_len5_validation),
    ]

    normalized_tables: Dict[str, pd.DataFrame] = {}
    best_tables: Dict[str, pd.DataFrame] = {}
    source_files: Dict[str, str] = {}

    for spec in specs:
        if spec.path is None:
            print(f"[SKIP] {spec.name}: no path provided")
            continue
        print(f"[LOAD] {spec.name}: {spec.path}")
        norm_df, source_path = load_experiment(spec)
        normalized_tables[spec.name] = norm_df
        source_files[spec.name] = source_path or str(spec.path)
        best_df = choose_best_per_target(norm_df, args.primary_metric, args.secondary_metric)
        best_tables[spec.name] = best_df
        print(f"  source: {source_path}")
        print(f"  rows normalized: {len(norm_df)}")
        print(f"  targets: {sorted(best_df['target'].dropna().unique().tolist())}")

    if not normalized_tables:
        raise RuntimeError("No experiment paths were provided or loaded.")

    normalized_all = pd.concat(normalized_tables.values(), ignore_index=True)
    best_all = pd.concat(best_tables.values(), ignore_index=True)
    comparison = add_deltas(merge_best_tables(best_tables))

    normalized_path = output_dir / "normalized_all_experiments.csv"
    best_path = output_dir / "best_models_by_target.csv"
    comparison_path = output_dir / "final_experiment_comparison.csv"
    comparison_md_path = output_dir / "final_experiment_comparison.md"
    source_path = output_dir / "source_files.json"

    normalized_all.to_csv(normalized_path, index=False, encoding="utf-8")
    best_all.to_csv(best_path, index=False, encoding="utf-8")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8")
    comparison_md_path.write_text(make_markdown_table(comparison), encoding="utf-8")
    save_json(source_path, source_files)

    plot_paths = [] if args.no_plots else save_plots(comparison, output_dir)
    save_report(output_dir, args, source_files, best_all, comparison)

    print("")
    print("=" * 80)
    print("Saved comparison outputs")
    print("=" * 80)
    print(f"Output dir: {output_dir}")
    print(f"Normalized: {normalized_path}")
    print(f"Best models: {best_path}")
    print(f"Comparison CSV: {comparison_path}")
    print(f"Comparison MD: {comparison_md_path}")
    print(f"Report: {output_dir / 'report.md'}")
    if plot_paths:
        print("Figures:")
        for p in plot_paths:
            print(f"  {p}")

    focus_cols = [
        "target",
        "tabular_r2",
        "context_r2",
        "mha_len3_r2",
        "mha_len5_r2",
        "delta_context_vs_tabular_r2",
        "delta_mha_len3_vs_context_r2",
        "delta_mha_len3_vs_tabular_r2",
        "delta_mha_len5_vs_mha_len3_r2",
        "tabular_spearman",
        "context_spearman",
        "mha_len3_spearman",
        "mha_len5_spearman",
    ]
    focus_cols = [c for c in focus_cols if c in comparison.columns]
    if focus_cols:
        print("")
        print("Main comparison:")
        print(comparison[focus_cols].to_string(index=False))


if __name__ == "__main__":
    main()
