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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


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
    dataset: Path
    output_dir: Path
    run_name: str
    targets: list[str]
    feature_set: str
    models: list[str]
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
    return logging.getLogger("slow_latent_cross_source")


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


def is_feature_column(col: str, id_cols: dict[str, str]) -> bool:
    if col in set(id_cols.values()):
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
) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in numeric_cols if is_feature_column(c, id_cols=id_cols)]

    if feature_set == "numeric":
        selected = candidates

    elif feature_set == "pow":
        selected = [
            c
            for c in candidates
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
            c
            for c in candidates
            if "eeg" in c.lower()
            and "pow" not in c.lower()
            and "band" not in c.lower()
        ]

    elif feature_set == "pow_plus_eeg":
        selected = [
            c
            for c in candidates
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


def make_regressor(model_name: str, random_state: int, fast: bool) -> Pipeline:
    if model_name == "hgb_reg":
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

    if model_name == "rf_reg":
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

    if model_name == "ridge":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", Ridge(alpha=1.0, random_state=random_state)),
            ]
        )

    raise ValueError(f"Unknown model: {model_name}")


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


def get_cross_source_pairs(df: pd.DataFrame, source_col: str) -> list[tuple[str, str]]:
    sources = sorted(df[source_col].dropna().astype(str).unique().tolist())

    pairs = []
    for train_source in sources:
        for test_source in sources:
            if train_source != test_source:
                pairs.append((train_source, test_source))

    return pairs


def evaluate_cross_source(
    df: pd.DataFrame,
    feature_cols: list[str],
    targets: list[str],
    source_col: str,
    models: list[str],
    random_state: int,
    fast: bool,
) -> pd.DataFrame:
    rows = []

    pairs = get_cross_source_pairs(df, source_col=source_col)

    for target in targets:
        task_df = df[feature_cols + [target, source_col]].copy()
        task_df[target] = pd.to_numeric(task_df[target], errors="coerce")
        task_df = task_df[np.isfinite(task_df[target])].reset_index(drop=True)

        if task_df.empty:
            continue

        for train_source, test_source in pairs:
            train_mask = task_df[source_col].astype(str) == str(train_source)
            test_mask = task_df[source_col].astype(str) == str(test_source)

            train_df = task_df[train_mask].copy()
            test_df = task_df[test_mask].copy()

            if len(train_df) < 100 or len(test_df) < 100:
                continue

            X_train = train_df[feature_cols]
            y_train = train_df[target].to_numpy(dtype=float)

            X_test = test_df[feature_cols]
            y_test = test_df[target].to_numpy(dtype=float)

            for model_name in models:
                model = make_regressor(
                    model_name=model_name,
                    random_state=random_state,
                    fast=fast,
                )

                started = time.perf_counter()
                model.fit(X_train, y_train)
                fit_time = time.perf_counter() - started

                started = time.perf_counter()
                pred = model.predict(X_test)
                predict_time = time.perf_counter() - started

                metrics = regression_metrics(y_test, pred)

                rows.append(
                    {
                        "target": target,
                        "train_source": train_source,
                        "test_source": test_source,
                        "direction": f"{train_source} -> {test_source}",
                        "model": model_name,
                        "n_train": int(len(train_df)),
                        "n_test": int(len(test_df)),
                        "fit_time_sec": fit_time,
                        "predict_time_sec": predict_time,
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def build_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame()

    group_cols = ["target", "direction", "train_source", "test_source", "model"]

    metric_cols = [
        c
        for c in fold_df.columns
        if c
        not in {
            "target",
            "direction",
            "train_source",
            "test_source",
            "model",
            "n_train",
            "n_test",
            "fit_time_sec",
            "predict_time_sec",
        }
        and pd.api.types.is_numeric_dtype(fold_df[c])
    ]

    rows = []

    for keys, sub in fold_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["runs"] = int(len(sub))
        row["n_train"] = int(sub["n_train"].iloc[0])
        row["n_test"] = int(sub["n_test"].iloc[0])
        row["fit_time_sec_sum"] = float(sub["fit_time_sec"].sum())
        row["predict_time_sec_sum"] = float(sub["predict_time_sec"].sum())

        for metric in metric_cols:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_std"] = sub[metric].std()
            row[f"{metric}_min"] = sub[metric].min()
            row[f"{metric}_max"] = sub[metric].max()

        rows.append(row)

    out = pd.DataFrame(rows)

    if "r2_mean" in out.columns:
        out = out.sort_values("r2_mean", ascending=False, na_position="last")

    return out.reset_index(drop=True)


def build_target_direction_pivot(summary_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if summary_df.empty or metric not in summary_df.columns:
        return pd.DataFrame()

    return summary_df.pivot_table(
        index="target",
        columns="direction",
        values=metric,
        aggfunc="max",
    ).reset_index()


def plot_metric_by_direction(
    summary_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if summary_df.empty or metric_col not in summary_df.columns:
        return

    pivot = summary_df.pivot_table(
        index="target",
        columns="direction",
        values=metric_col,
        aggfunc="max",
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Slow latent target")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(
    output_dir: Path,
    config: Config,
    dataset_info: dict,
    feature_cols: list[str],
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    r2_pivot: pd.DataFrame,
    spearman_pivot: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# Slow latent cross-source transfer report")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append(
        "Evaluate whether slow PM latent states transfer between source datasets."
    )
    lines.append("")
    lines.append("Cross-source directions:")
    lines.append("")
    lines.append("```text")
    lines.append("gpn_data -> Old_EEG")
    lines.append("Old_EEG -> gpn_data")
    lines.append("```")
    lines.append("")
    lines.append("## Input")
    lines.append("")
    lines.append(f"- Dataset: `{config.dataset}`")
    lines.append(f"- Output dir: `{config.output_dir}`")
    lines.append(f"- Run name: `{config.run_name}`")
    lines.append(f"- Rows loaded: `{dataset_info['rows_loaded']}`")
    lines.append(f"- Rows used: `{dataset_info['rows_used']}`")
    lines.append(f"- Source column: `{dataset_info['source_col']}`")
    lines.append(f"- Sources: `{dataset_info['sources']}`")
    lines.append(f"- Targets: `{config.targets}`")
    lines.append(f"- Feature set: `{config.feature_set}`")
    lines.append(f"- Features used: `{len(feature_cols)}`")
    lines.append(f"- Models: `{config.models}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if not summary_df.empty:
        display_cols = [
            c
            for c in [
                "target",
                "direction",
                "model",
                "n_train",
                "n_test",
                "mae_mean",
                "rmse_mean",
                "r2_mean",
                "pearson_mean",
                "spearman_mean",
            ]
            if c in summary_df.columns
        ]
        lines.append(summary_df[display_cols].to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No summary rows.")
    lines.append("")
    lines.append("## R² pivot")
    lines.append("")
    if not r2_pivot.empty:
        lines.append(r2_pivot.to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No R² pivot.")
    lines.append("")
    lines.append("## Spearman pivot")
    lines.append("")
    if not spearman_pivot.empty:
        lines.append(spearman_pivot.to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No Spearman pivot.")
    lines.append("")
    lines.append("## Working interpretation")
    lines.append("")
    lines.append("- If R² is positive in both directions, the latent state has non-trivial cross-source transfer.")
    lines.append("- If Spearman remains positive while R² is low, the model preserves ranking but not calibration.")
    lines.append("- Strong asymmetry between directions indicates source/domain mismatch.")
    lines.append("- Poor transfer suggests the need for source-specific calibration or domain adaptation.")
    lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_list_arg(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train cross-source regressors for slow latent states."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("reports/slow_latent_states/pm_w10/slow_pm_latent_states_w10.parquet"),
        help="Latent dataset from src/35_build_and_train_slow_latent_states.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/slow_latent_states/cross_source"),
        help="Output directory.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="slow_latent_cross_source",
        help="Run name.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="slow_pca_1,slow_pca_2,slow_pca_3,slow_pca_4",
        help="Comma-separated slow latent targets.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="pow_plus_eeg",
        choices=["numeric", "pow", "eeg", "pow_plus_eeg"],
        help="Feature selection mode.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="hgb_reg",
        help="Comma-separated models: hgb_reg,rf_reg,ridge.",
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
        help="Optional sampling before training.",
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
        targets=parse_list_arg(args.targets),
        feature_set=args.feature_set,
        models=parse_list_arg(args.models),
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
    logger.info("Train slow latent cross-source transfer")
    logger.info("=" * 80)
    logger.info("Dataset: %s", config.dataset)
    logger.info("Output dir: %s", config.output_dir)
    logger.info("Targets: %s", config.targets)
    logger.info("Models: %s", config.models)

    if not config.dataset.exists():
        raise FileNotFoundError(f"Dataset was not found: {config.dataset}")

    df = read_dataset(config.dataset)
    rows_loaded = len(df)

    id_cols = detect_id_columns(list(df.columns))

    if "source" not in id_cols:
        raise ValueError("Source column was not detected. Need source for cross-source transfer.")

    source_col = id_cols["source"]

    available_targets = [t for t in config.targets if t in df.columns]
    missing_targets = [t for t in config.targets if t not in df.columns]

    if missing_targets:
        logger.warning("Missing targets will be skipped: %s", missing_targets)

    if not available_targets:
        raise ValueError(f"No requested targets were found. Requested: {config.targets}")

    if config.max_rows is not None and len(df) > config.max_rows:
        df = df.sample(n=config.max_rows, random_state=config.random_state).reset_index(drop=True)
        logger.info("Sampled rows: %d", len(df))

    feature_cols = select_feature_columns(
        df=df,
        id_cols=id_cols,
        feature_set=config.feature_set,
        max_features=config.max_features,
    )

    if not feature_cols:
        raise ValueError("No feature columns were selected.")

    sources = sorted(df[source_col].dropna().astype(str).unique().tolist())

    if len(sources) < 2:
        raise ValueError(f"Need at least two sources for cross-source transfer. Found: {sources}")

    logger.info("Detected ID columns: %s", id_cols)
    logger.info("Source column: %s", source_col)
    logger.info("Sources: %s", sources)
    logger.info("Selected feature columns: %d", len(feature_cols))

    fold_df = evaluate_cross_source(
        df=df,
        feature_cols=feature_cols,
        targets=available_targets,
        source_col=source_col,
        models=config.models,
        random_state=config.random_state,
        fast=config.fast,
    )

    if fold_df.empty:
        raise RuntimeError("No cross-source metrics were produced.")

    summary_df = build_summary(fold_df)
    r2_pivot = build_target_direction_pivot(summary_df, metric="r2_mean")
    spearman_pivot = build_target_direction_pivot(summary_df, metric="spearman_mean")

    fold_df.to_csv(config.output_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(config.output_dir / "summary.csv", index=False)
    r2_pivot.to_csv(config.output_dir / "r2_pivot.csv", index=False)
    spearman_pivot.to_csv(config.output_dir / "spearman_pivot.csv", index=False)

    feature_info = {
        "feature_set": config.feature_set,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
    }
    save_json(config.output_dir / "feature_columns.json", feature_info)

    dataset_info = {
        "dataset": str(config.dataset),
        "rows_loaded": int(rows_loaded),
        "rows_used": int(len(df)),
        "source_col": source_col,
        "sources": sources,
        "id_columns": id_cols,
        "targets_requested": config.targets,
        "targets_used": available_targets,
        "feature_set": config.feature_set,
        "n_features": int(len(feature_cols)),
        "models": config.models,
    }

    save_json(
        config.output_dir / "summary.json",
        {
            "run_name": config.run_name,
            "output_dir": str(config.output_dir),
            **dataset_info,
            "n_fold_rows": int(len(fold_df)),
            "n_summary_rows": int(len(summary_df)),
        },
    )

    if not config.no_plots:
        plot_metric_by_direction(
            summary_df=summary_df,
            metric_col="r2_mean",
            title="Slow latent cross-source transfer: R²",
            ylabel="R²",
            output_path=figures_dir / "cross_source_r2_by_target.png",
        )
        plot_metric_by_direction(
            summary_df=summary_df,
            metric_col="spearman_mean",
            title="Slow latent cross-source transfer: Spearman",
            ylabel="Spearman",
            output_path=figures_dir / "cross_source_spearman_by_target.png",
        )

    write_report(
        output_dir=config.output_dir,
        config=config,
        dataset_info=dataset_info,
        feature_cols=feature_cols,
        fold_df=fold_df,
        summary_df=summary_df,
        r2_pivot=r2_pivot,
        spearman_pivot=spearman_pivot,
    )

    logger.info("=" * 80)
    logger.info("Saved slow latent cross-source outputs")
    logger.info("=" * 80)
    logger.info("Fold metrics: %s", config.output_dir / "fold_metrics.csv")
    logger.info("Summary: %s", config.output_dir / "summary.csv")
    logger.info("R² pivot: %s", config.output_dir / "r2_pivot.csv")
    logger.info("Spearman pivot: %s", config.output_dir / "spearman_pivot.csv")
    logger.info("Report: %s", config.output_dir / "report.md")
    logger.info("")
    with pd.option_context("display.max_rows", 30, "display.max_columns", 20, "display.width", 180):
        logger.info("Summary:\n%s", summary_df.to_string(index=False))
    logger.info("Done.")


if __name__ == "__main__":
    main()