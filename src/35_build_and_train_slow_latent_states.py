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
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


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
    output_name: str
    n_components: int
    rolling_window: int
    min_valid_pm: int
    validation_modes: list[str]
    n_splits: int
    test_size: float
    random_state: int
    feature_set: str
    max_features: int | None
    max_rows: int | None
    fast: bool
    no_plots: bool


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("slow_latent_states")


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
        raise ValueError("Need at least source/subject_id/record_id for slow PM components.")

    return group_cols


def choose_sort_columns(id_cols: dict[str, str]) -> list[str]:
    if "window_start" in id_cols:
        return [id_cols["window_start"]]

    if "window_id" in id_cols:
        return [id_cols["window_id"]]

    return []


def prepare_pm_frame(df: pd.DataFrame, pm_cols: dict[str, str]) -> pd.DataFrame:
    out = df.copy()

    for short_name, col in pm_cols.items():
        out[f"pm_{short_name}"] = pd.to_numeric(out[col], errors="coerce")

    return out


def add_slow_components(
    df: pd.DataFrame,
    pm_names: list[str],
    group_cols: list[str],
    sort_cols: list[str],
    rolling_window: int,
) -> pd.DataFrame:
    sort_by = list(dict.fromkeys(group_cols + sort_cols))

    if sort_by:
        df = df.sort_values(sort_by).reset_index(drop=True)

    pieces = []

    for _, group in df.groupby(group_cols, dropna=False, sort=False):
        group = group.copy()

        for name in pm_names:
            pm_col = f"pm_{name}"
            slow_col = f"pm_{name}_slow"
            fast_col = f"pm_{name}_fast"

            slow = (
                group[pm_col]
                .rolling(
                    window=rolling_window,
                    min_periods=max(2, rolling_window // 2),
                    center=True,
                )
                .mean()
            )

            group[slow_col] = slow
            group[fast_col] = group[pm_col] - slow

        pieces.append(group)

    return pd.concat(pieces, ignore_index=True)


def clean_slow_pm_matrix(
    df: pd.DataFrame,
    pm_names: list[str],
    min_valid_pm: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    slow_cols = [f"pm_{name}_slow" for name in pm_names]

    slow_df = df[slow_cols].copy()
    slow_df = slow_df.rename(columns={f"pm_{name}_slow": name for name in pm_names})

    for col in slow_df.columns:
        slow_df[col] = pd.to_numeric(slow_df[col], errors="coerce")

    valid_count = slow_df.notna().sum(axis=1)
    mask = valid_count >= min_valid_pm

    slow_clean = slow_df.loc[mask].copy()
    medians = slow_clean.median(axis=0, numeric_only=True)
    slow_clean = slow_clean.fillna(medians).reset_index(drop=True)

    meta_clean = df.loc[mask].copy().reset_index(drop=True)

    return meta_clean, slow_clean


def build_pca(
    slow_pm_scaled: np.ndarray,
    pm_names: list[str],
    n_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PCA]:
    pca = PCA(n_components=n_components, random_state=0)
    coords = pca.fit_transform(slow_pm_scaled)

    coord_cols = [f"slow_pca_{i + 1}" for i in range(n_components)]
    coords_df = pd.DataFrame(coords, columns=coord_cols)

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=pm_names,
        columns=coord_cols,
    ).reset_index(names="pm_metric")

    explained_df = pd.DataFrame(
        {
            "component": coord_cols,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance": pca.explained_variance_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    return coords_df, loadings_df, explained_df, pca


def infer_axis_summary(loadings_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    component_cols = [c for c in loadings_df.columns if c.startswith("slow_pca_")]

    for component in component_cols:
        sub = loadings_df[["pm_metric", component]].copy()
        sub["abs_loading"] = sub[component].abs()
        sub = sub.sort_values("abs_loading", ascending=False)

        top = sub.head(4)
        top_metrics = top["pm_metric"].tolist()
        top_set = set(top_metrics)

        candidate_states = []

        if {"stress", "excitement"} & top_set:
            candidate_states.append("Slow Stress / Arousal")
        if {"attention", "focus"} & top_set:
            candidate_states.append("Slow Workload / Attention")
        if "relaxation" in top_set:
            candidate_states.append("Slow Recovery / Fatigue")
        if {"engagement", "interest"} & top_set:
            candidate_states.append("Slow Engagement / Involvement")

        suggested = " + ".join(candidate_states) if candidate_states else "Unlabeled slow latent axis"

        rows.append(
            {
                "component": component,
                "suggested_latent_state": suggested,
                "manual_interpretation_required": True,
                "top_metrics": ", ".join(top_metrics),
                "top_signed_loadings": "; ".join(
                    f"{r['pm_metric']}={r[component]:.3f}" for _, r in top.iterrows()
                ),
                "top_abs_loading_sum": top["abs_loading"].sum(),
            }
        )

    return pd.DataFrame(rows)


def is_feature_column(col: str, id_cols: dict[str, str], pm_cols: dict[str, str]) -> bool:
    if col in set(id_cols.values()):
        return False

    if col in set(pm_cols.values()):
        return False

    low = col.lower()

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


def train_latent_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    latent_cols: list[str],
    id_cols: dict[str, str],
    validation_modes: list[str],
    n_splits: int,
    test_size: float,
    random_state: int,
    fast: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []

    for target_col in latent_cols:
        task_cols = feature_cols + [target_col] + [c for c in id_cols.values() if c in df.columns]
        task_df = df[task_cols].copy()

        task_df[target_col] = pd.to_numeric(task_df[target_col], errors="coerce")
        task_df = task_df[np.isfinite(task_df[target_col])].reset_index(drop=True)

        for validation_mode in validation_modes:
            for fold, train_idx, val_idx in get_split_indices(
                task_df,
                validation_mode=validation_mode,
                id_cols=id_cols,
                test_size=test_size,
                n_splits=n_splits,
                random_state=random_state,
            ):
                X_train = task_df.iloc[train_idx][feature_cols]
                y_train = task_df.iloc[train_idx][target_col].values

                X_val = task_df.iloc[val_idx][feature_cols]
                y_val = task_df.iloc[val_idx][target_col].values

                model = make_regressor(random_state=random_state, fast=fast)

                started = time.perf_counter()
                model.fit(X_train, y_train)
                fit_time = time.perf_counter() - started

                started = time.perf_counter()
                pred = model.predict(X_val)
                predict_time = time.perf_counter() - started

                metrics = regression_metrics(y_val.astype(float), pred.astype(float))

                fold_rows.append(
                    {
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

    fold_df = pd.DataFrame(fold_rows)

    group_cols = ["target", "validation", "model"]

    rows = []
    for keys, sub in fold_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["folds"] = int(sub["fold"].nunique())
        row["n_val_total"] = int(sub["n_val"].sum())

        for metric in ["mae", "rmse", "r2", "pearson", "spearman"]:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_std"] = sub[metric].std()
            row[f"{metric}_min"] = sub[metric].min()
            row[f"{metric}_max"] = sub[metric].max()

        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(
        ["validation", "r2_mean"],
        ascending=[True, False],
        na_position="last",
    )

    return fold_df, summary_df


def plot_explained_variance(explained_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(explained_df["component"], explained_df["explained_variance_ratio"])
    ax.plot(
        explained_df["component"],
        explained_df["cumulative_explained_variance_ratio"],
        marker="o",
    )

    ax.set_title("Slow PM PCA explained variance")
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance ratio")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_loadings_heatmap(loadings_df: pd.DataFrame, output_path: Path) -> None:
    component_cols = [c for c in loadings_df.columns if c.startswith("slow_pca_")]
    matrix = loadings_df.set_index("pm_metric")[component_cols]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix.values, aspect="auto")

    ax.set_xticks(np.arange(len(component_cols)))
    ax.set_xticklabels(component_cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Slow PM PCA loadings")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_latent_scatter(coords_df: pd.DataFrame, meta_df: pd.DataFrame, output_path: Path) -> None:
    if coords_df.shape[1] < 2:
        return

    x_col = coords_df.columns[0]
    y_col = coords_df.columns[1]

    fig, ax = plt.subplots(figsize=(7, 6))

    source_col = None
    for col in ["source", "dataset", "data_source"]:
        if col in meta_df.columns:
            source_col = col
            break

    if source_col is not None:
        sources = meta_df[source_col].astype(str).fillna("unknown")
        for source in sorted(sources.unique()):
            mask = sources == source
            ax.scatter(
                coords_df.loc[mask, x_col],
                coords_df.loc[mask, y_col],
                s=8,
                alpha=0.45,
                label=source,
            )
        ax.legend(title=source_col)
    else:
        ax.scatter(coords_df[x_col], coords_df[y_col], s=8, alpha=0.45)

    ax.set_title("Slow PM latent space")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_latent_regression_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty or "r2_mean" not in summary_df.columns:
        return

    pivot = summary_df.pivot_table(
        index="target",
        columns="validation",
        values="r2_mean",
        aggfunc="max",
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("EEG/POW → slow latent coordinates")
    ax.set_xlabel("Latent coordinate")
    ax.set_ylabel("R²")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(
    output_dir: Path,
    config: Config,
    dataset_info: dict,
    id_cols: dict[str, str],
    pm_cols: dict[str, str],
    feature_cols: list[str],
    pca_explained: pd.DataFrame,
    axis_summary: pd.DataFrame,
    regression_summary: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# Slow PM latent states report")
    lines.append("")
    lines.append("## Input")
    lines.append("")
    lines.append(f"- Dataset: `{config.dataset}`")
    lines.append(f"- Output dir: `{config.output_dir}`")
    lines.append(f"- Rows loaded: `{dataset_info['rows_loaded']}`")
    lines.append(f"- Rows used for slow PCA: `{dataset_info['rows_used_for_pca']}`")
    lines.append(f"- PM metrics used: `{dataset_info['pm_metrics']}`")
    lines.append(f"- Slow rolling window: `{config.rolling_window}`")
    lines.append(f"- PCA components: `{config.n_components}`")
    lines.append(f"- Features used for latent regression: `{len(feature_cols)}`")
    lines.append("")
    lines.append("## ID columns")
    lines.append("")
    for key, value in id_cols.items():
        lines.append(f"- `{key}` → `{value}`")
    lines.append("")
    lines.append("## PM columns")
    lines.append("")
    for key, value in pm_cols.items():
        lines.append(f"- `{key}` → `{value}`")
    lines.append("")
    lines.append("## PCA explained variance")
    lines.append("")
    lines.append(pca_explained.to_markdown(index=False, floatfmt=".5f"))
    lines.append("")
    lines.append("## Axis summary")
    lines.append("")
    lines.append(axis_summary.to_markdown(index=False, floatfmt=".5f"))
    lines.append("")
    lines.append("## EEG/POW → slow latent coordinates")
    lines.append("")
    if not regression_summary.empty:
        lines.append(regression_summary.to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No regression rows.")
    lines.append("")
    lines.append("## Working interpretation")
    lines.append("")
    lines.append("This experiment connects two project lines:")
    lines.append("")
    lines.append("1. PM dynamics: slow components behave as background-state targets.")
    lines.append("2. Latent states: PM metrics can be compressed into several interpretable coordinates.")
    lines.append("")
    lines.append("If slow latent coordinates are predicted more stably than individual PM targets, they can be used as a common intermediate state representation.")
    lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_list_arg(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Build slow PM latent states and train EEG/POW baselines for latent coordinates."
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
        default=Path("reports/slow_latent_states/pm_w10"),
        help="Output directory.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="slow_pm_latent_states_w10",
        help="Base output name.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=4,
        help="Number of slow PM PCA components.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Rolling window for slow PM components.",
    )
    parser.add_argument(
        "--min-valid-pm",
        type=int,
        default=4,
        help="Minimum valid slow PM metrics per row.",
    )
    parser.add_argument(
        "--validation-modes",
        type=str,
        default="random_split,groupkfold_subject",
        help="Comma-separated validation modes.",
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
        help="Optional sample size after slow components are computed.",
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
        output_name=args.output_name,
        n_components=args.n_components,
        rolling_window=args.rolling_window,
        min_valid_pm=args.min_valid_pm,
        validation_modes=parse_list_arg(args.validation_modes),
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_state,
        feature_set=args.feature_set,
        max_features=args.max_features,
        max_rows=args.max_rows,
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
    logger.info("Build and train slow PM latent states")
    logger.info("=" * 80)
    logger.info("Dataset: %s", config.dataset)
    logger.info("Output dir: %s", config.output_dir)

    if not config.dataset.exists():
        raise FileNotFoundError(f"Dataset was not found: {config.dataset}")

    df = read_dataset(config.dataset)
    rows_loaded = len(df)

    columns = list(df.columns)
    id_cols = detect_id_columns(columns)
    pm_cols = detect_pm_columns(columns)

    if len(pm_cols) < 3:
        raise ValueError(f"Too few PM columns detected: {pm_cols}")

    pm_names = list(pm_cols.keys())
    config.n_components = min(config.n_components, len(pm_names))
    config.min_valid_pm = min(config.min_valid_pm, len(pm_names))

    logger.info("Detected ID columns: %s", id_cols)
    logger.info("Detected PM columns: %s", pm_cols)

    group_cols = choose_group_columns(id_cols)
    sort_cols = choose_sort_columns(id_cols)

    df = prepare_pm_frame(df, pm_cols=pm_cols)
    df = add_slow_components(
        df=df,
        pm_names=pm_names,
        group_cols=group_cols,
        sort_cols=sort_cols,
        rolling_window=config.rolling_window,
    )

    if config.max_rows is not None and len(df) > config.max_rows:
        df = df.sample(n=config.max_rows, random_state=config.random_state).reset_index(drop=True)
        logger.info("Sampled rows after slow component construction: %d", len(df))

    meta_df, slow_pm_df = clean_slow_pm_matrix(
        df=df,
        pm_names=pm_names,
        min_valid_pm=config.min_valid_pm,
    )

    logger.info("Rows used for slow PM PCA: %d", len(slow_pm_df))

    scaler = StandardScaler()
    slow_scaled = scaler.fit_transform(slow_pm_df.values)

    coords_df, loadings_df, explained_df, _ = build_pca(
        slow_pm_scaled=slow_scaled,
        pm_names=list(slow_pm_df.columns),
        n_components=config.n_components,
    )

    axis_summary = infer_axis_summary(loadings_df)

    feature_cols = select_feature_columns(
        meta_df,
        id_cols=id_cols,
        pm_cols=pm_cols,
        feature_set=config.feature_set,
        max_features=config.max_features,
    )

    if not feature_cols:
        raise ValueError("No feature columns were selected.")

    logger.info("Selected feature columns: %d", len(feature_cols))

    id_keep = [c for c in id_cols.values() if c in meta_df.columns]
    latent_dataset = pd.concat(
        [
            meta_df[id_keep].reset_index(drop=True) if id_keep else pd.DataFrame(index=coords_df.index),
            coords_df.reset_index(drop=True),
            slow_pm_df.add_prefix("slow_pm_").reset_index(drop=True),
            meta_df[feature_cols].reset_index(drop=True),
        ],
        axis=1,
    )

    latent_cols = list(coords_df.columns)

    fold_metrics, regression_summary = train_latent_regression(
        df=latent_dataset,
        feature_cols=feature_cols,
        latent_cols=latent_cols,
        id_cols=id_cols,
        validation_modes=config.validation_modes,
        n_splits=config.n_splits,
        test_size=config.test_size,
        random_state=config.random_state,
        fast=config.fast,
    )

    latent_dataset.to_csv(config.output_dir / f"{config.output_name}.csv", index=False)

    try:
        latent_dataset.to_parquet(config.output_dir / f"{config.output_name}.parquet", index=False)
        parquet_saved = True
    except Exception as exc:
        logger.warning("Could not save parquet: %s", exc)
        parquet_saved = False

    slow_pm_df.to_csv(config.output_dir / "slow_pm_metrics_clean.csv", index=False)
    coords_df.to_csv(config.output_dir / "slow_pm_latent_coordinates.csv", index=False)
    loadings_df.to_csv(config.output_dir / "slow_pm_pca_loadings.csv", index=False)
    explained_df.to_csv(config.output_dir / "slow_pm_pca_explained_variance.csv", index=False)
    axis_summary.to_csv(config.output_dir / "slow_pm_axis_summary.csv", index=False)

    fold_metrics.to_csv(config.output_dir / "latent_regression_fold_metrics.csv", index=False)
    regression_summary.to_csv(config.output_dir / "latent_regression_summary.csv", index=False)

    feature_info = {
        "feature_set": config.feature_set,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
    }

    save_json(config.output_dir / "feature_columns.json", feature_info)

    dataset_info = {
        "dataset": str(config.dataset),
        "rows_loaded": int(rows_loaded),
        "rows_after_optional_sampling": int(len(df)),
        "rows_used_for_pca": int(len(slow_pm_df)),
        "pm_metrics": pm_names,
        "id_columns": id_cols,
        "pm_columns": pm_cols,
        "group_columns": group_cols,
        "sort_columns": sort_cols,
        "n_components": int(config.n_components),
        "rolling_window": int(config.rolling_window),
        "min_valid_pm": int(config.min_valid_pm),
        "feature_set": config.feature_set,
        "n_features": int(len(feature_cols)),
        "parquet_saved": parquet_saved,
    }

    save_json(config.output_dir / "summary.json", dataset_info)

    if not config.no_plots:
        plot_explained_variance(
            explained_df,
            figures_dir / "slow_pm_pca_explained_variance.png",
        )
        plot_loadings_heatmap(
            loadings_df,
            figures_dir / "slow_pm_pca_loadings_heatmap.png",
        )
        plot_latent_scatter(
            coords_df,
            meta_df,
            figures_dir / "slow_pm_latent_scatter.png",
        )
        plot_latent_regression_summary(
            regression_summary,
            figures_dir / "latent_regression_r2_summary.png",
        )

    write_report(
        output_dir=config.output_dir,
        config=config,
        dataset_info=dataset_info,
        id_cols=id_cols,
        pm_cols=pm_cols,
        feature_cols=feature_cols,
        pca_explained=explained_df,
        axis_summary=axis_summary,
        regression_summary=regression_summary,
    )

    logger.info("=" * 80)
    logger.info("Saved slow PM latent state outputs")
    logger.info("=" * 80)
    logger.info("Latent dataset: %s", config.output_dir / f"{config.output_name}.csv")
    if parquet_saved:
        logger.info("Latent parquet: %s", config.output_dir / f"{config.output_name}.parquet")
    logger.info("PCA loadings: %s", config.output_dir / "slow_pm_pca_loadings.csv")
    logger.info("Axis summary: %s", config.output_dir / "slow_pm_axis_summary.csv")
    logger.info("Regression summary: %s", config.output_dir / "latent_regression_summary.csv")
    logger.info("Report: %s", config.output_dir / "report.md")
    logger.info("Done.")


if __name__ == "__main__":
    main()