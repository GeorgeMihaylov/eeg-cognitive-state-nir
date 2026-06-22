from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


@dataclass
class DynamicsConfig:
    dataset: Path
    output_dir: Path
    output_name: str
    rolling_window: int
    trend_threshold: float
    min_valid_pm: int
    sample_size: int | None
    random_state: int
    no_plots: bool


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("pm_state_dynamics")


def read_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        return list(pd.read_parquet(path, columns=[]).columns)

    if path.suffix.lower() in {".csv", ".txt"}:
        return list(pd.read_csv(path, nrows=0).columns)

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


def load_dataset(config: DynamicsConfig, id_cols: dict[str, str], pm_cols: dict[str, str]) -> pd.DataFrame:
    usecols = list(dict.fromkeys(list(id_cols.values()) + list(pm_cols.values())))

    if config.dataset.suffix.lower() == ".parquet":
        df = pd.read_parquet(config.dataset, columns=usecols)
    else:
        df = pd.read_csv(config.dataset, usecols=usecols)

    if config.sample_size is not None and len(df) > config.sample_size:
        df = df.sample(n=config.sample_size, random_state=config.random_state).reset_index(drop=True)

    return df


def choose_group_columns(id_cols: dict[str, str]) -> list[str]:
    group_cols = []
    if "source" in id_cols:
        group_cols.append(id_cols["source"])
    if "subject_id" in id_cols:
        group_cols.append(id_cols["subject_id"])
    if "record_id" in id_cols:
        group_cols.append(id_cols["record_id"])

    if not group_cols:
        raise ValueError(
            "No grouping columns detected. Need at least subject_id or record_id "
            "to compute PM dynamics without mixing independent records."
        )

    return group_cols


def choose_sort_columns(id_cols: dict[str, str]) -> list[str]:
    sort_cols = []
    if "window_start" in id_cols:
        sort_cols.append(id_cols["window_start"])
    elif "window_id" in id_cols:
        sort_cols.append(id_cols["window_id"])

    return sort_cols


def prepare_base_frame(df: pd.DataFrame, id_cols: dict[str, str], pm_cols: dict[str, str]) -> pd.DataFrame:
    keep_cols = list(dict.fromkeys(list(id_cols.values()) + list(pm_cols.values())))
    out = df[keep_cols].copy()

    rename_map = {v: f"pm_{k}" for k, v in pm_cols.items()}
    out = out.rename(columns=rename_map)

    for short_name in pm_cols:
        col = f"pm_{short_name}"
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def add_validity_columns(df: pd.DataFrame, pm_names: list[str], min_valid_pm: int) -> pd.DataFrame:
    pm_cols = [f"pm_{name}" for name in pm_names]
    out = df.copy()
    out["pm_valid_count"] = out[pm_cols].notna().sum(axis=1)
    out["pm_is_valid_for_dynamics"] = out["pm_valid_count"] >= min_valid_pm
    return out


def compute_group_dynamics(
    group: pd.DataFrame,
    pm_names: list[str],
    rolling_window: int,
    trend_threshold: float,
) -> pd.DataFrame:
    group = group.copy()

    for name in pm_names:
        col = f"pm_{name}"

        delta_col = f"pm_{name}_delta_next"
        abs_delta_col = f"pm_{name}_abs_delta_next"
        trend_col = f"pm_{name}_trend_next"
        trend_code_col = f"pm_{name}_trend_next_code"

        slow_col = f"pm_{name}_slow"
        fast_col = f"pm_{name}_fast"
        rolling_std_col = f"pm_{name}_rolling_std"
        valid_transition_col = f"pm_{name}_valid_transition"

        next_values = group[col].shift(-1)
        delta = next_values - group[col]

        group[delta_col] = delta
        group[abs_delta_col] = delta.abs()

        trend = pd.Series("stable", index=group.index, dtype="object")
        trend[delta > trend_threshold] = "up"
        trend[delta < -trend_threshold] = "down"
        trend[delta.isna()] = np.nan

        trend_code = pd.Series(0.0, index=group.index)
        trend_code[trend == "up"] = 1.0
        trend_code[trend == "down"] = -1.0
        trend_code[trend.isna()] = np.nan

        group[trend_col] = trend
        group[trend_code_col] = trend_code

        slow = (
            group[col]
            .rolling(window=rolling_window, min_periods=max(2, rolling_window // 2), center=True)
            .mean()
        )
        rolling_std = (
            group[col]
            .rolling(window=rolling_window, min_periods=max(2, rolling_window // 2), center=True)
            .std()
        )

        group[slow_col] = slow
        group[fast_col] = group[col] - slow
        group[rolling_std_col] = rolling_std

        group[valid_transition_col] = group[col].notna() & next_values.notna()

    return group


def build_dynamics_dataset(
    df: pd.DataFrame,
    pm_names: list[str],
    group_cols: list[str],
    sort_cols: list[str],
    rolling_window: int,
    trend_threshold: float,
) -> pd.DataFrame:
    sort_by = list(dict.fromkeys(group_cols + sort_cols))
    if sort_by:
        df = df.sort_values(sort_by).reset_index(drop=True)

    pieces = []
    grouped = df.groupby(group_cols, dropna=False, sort=False)

    for _, group in grouped:
        group_dyn = compute_group_dynamics(
            group=group,
            pm_names=pm_names,
            rolling_window=rolling_window,
            trend_threshold=trend_threshold,
        )
        pieces.append(group_dyn)

    return pd.concat(pieces, ignore_index=True)


def summarize_pm_targets(df: pd.DataFrame, pm_names: list[str]) -> pd.DataFrame:
    rows = []

    for name in pm_names:
        base = f"pm_{name}"
        delta = f"pm_{name}_delta_next"
        fast = f"pm_{name}_fast"
        slow = f"pm_{name}_slow"
        trend = f"pm_{name}_trend_next"

        trend_counts = df[trend].value_counts(dropna=False).to_dict()

        rows.append(
            {
                "pm_metric": name,
                "n_base_valid": int(df[base].notna().sum()),
                "base_mean": df[base].mean(),
                "base_std": df[base].std(),
                "base_min": df[base].min(),
                "base_median": df[base].median(),
                "base_max": df[base].max(),
                "n_delta_valid": int(df[delta].notna().sum()),
                "delta_mean": df[delta].mean(),
                "delta_std": df[delta].std(),
                "delta_abs_mean": df[delta].abs().mean(),
                "fast_mean": df[fast].mean(),
                "fast_std": df[fast].std(),
                "slow_mean": df[slow].mean(),
                "slow_std": df[slow].std(),
                "trend_up": int(trend_counts.get("up", 0)),
                "trend_stable": int(trend_counts.get("stable", 0)),
                "trend_down": int(trend_counts.get("down", 0)),
                "trend_missing": int(trend_counts.get(np.nan, 0)),
            }
        )

    return pd.DataFrame(rows)


def summarize_by_source(df: pd.DataFrame, pm_names: list[str], source_col: str | None) -> pd.DataFrame:
    if source_col is None or source_col not in df.columns:
        return pd.DataFrame()

    rows = []

    for source, sub in df.groupby(source_col, dropna=False):
        for name in pm_names:
            base = f"pm_{name}"
            delta = f"pm_{name}_delta_next"
            trend = f"pm_{name}_trend_next"

            rows.append(
                {
                    "source": source,
                    "pm_metric": name,
                    "rows": int(len(sub)),
                    "n_base_valid": int(sub[base].notna().sum()),
                    "base_mean": sub[base].mean(),
                    "base_std": sub[base].std(),
                    "n_delta_valid": int(sub[delta].notna().sum()),
                    "delta_abs_mean": sub[delta].abs().mean(),
                    "trend_up": int((sub[trend] == "up").sum()),
                    "trend_stable": int((sub[trend] == "stable").sum()),
                    "trend_down": int((sub[trend] == "down").sum()),
                }
            )

    return pd.DataFrame(rows)


def compute_pm_dynamics_correlations(df: pd.DataFrame, pm_names: list[str]) -> pd.DataFrame:
    cols = []
    for name in pm_names:
        cols.extend(
            [
                f"pm_{name}",
                f"pm_{name}_delta_next",
                f"pm_{name}_fast",
                f"pm_{name}_slow",
            ]
        )

    available = [c for c in cols if c in df.columns]
    corr = df[available].corr(method="spearman")
    corr = corr.reset_index().rename(columns={"index": "metric"})
    return corr


def plot_trend_distribution(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return

    labels = summary_df["pm_metric"].tolist()
    up = summary_df["trend_up"].to_numpy()
    stable = summary_df["trend_stable"].to_numpy()
    down = summary_df["trend_down"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))

    ax.bar(x, down, label="down")
    ax.bar(x, stable, bottom=down, label="stable")
    ax.bar(x, up, bottom=down + stable, label="up")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("PM trend distribution")
    ax.set_ylabel("Windows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_delta_abs(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary_df["pm_metric"], summary_df["delta_abs_mean"])
    ax.set_title("Mean absolute delta by PM metric")
    ax.set_ylabel("Mean |PM(t+1) - PM(t)|")
    ax.set_xlabel("PM metric")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_fast_slow_std(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary_df))
    width = 0.35

    ax.bar(x - width / 2, summary_df["fast_std"], width, label="fast std")
    ax.bar(x + width / 2, summary_df["slow_std"], width, label="slow std")

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["pm_metric"], rotation=45, ha="right")
    ax.set_title("Fast vs slow component variability")
    ax.set_ylabel("Standard deviation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(
    output_dir: Path,
    config: DynamicsConfig,
    dataset_info: dict,
    id_cols: dict[str, str],
    pm_cols: dict[str, str],
    group_cols: list[str],
    sort_cols: list[str],
    target_summary: pd.DataFrame,
    source_summary: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# PM state dynamics report")
    lines.append("")
    lines.append("## Input")
    lines.append("")
    lines.append(f"- Dataset: `{config.dataset}`")
    lines.append(f"- Output dir: `{config.output_dir}`")
    lines.append(f"- Output name: `{config.output_name}`")
    lines.append(f"- Rows loaded: `{dataset_info['rows_loaded']}`")
    lines.append(f"- Rows saved: `{dataset_info['rows_saved']}`")
    lines.append(f"- Rolling window: `{config.rolling_window}`")
    lines.append(f"- Trend threshold: `{config.trend_threshold}`")
    lines.append(f"- Min valid PM: `{config.min_valid_pm}`")
    lines.append("")
    lines.append("## Detected ID columns")
    lines.append("")
    for logical, col in id_cols.items():
        lines.append(f"- `{logical}` → `{col}`")
    lines.append("")
    lines.append("## Grouping and sorting")
    lines.append("")
    lines.append(f"- Group columns: `{group_cols}`")
    lines.append(f"- Sort columns: `{sort_cols}`")
    lines.append("")
    lines.append("## PM columns")
    lines.append("")
    for short_name, col in pm_cols.items():
        lines.append(f"- `{short_name}` → `{col}`")
    lines.append("")
    lines.append("## Target dynamics summary")
    lines.append("")
    lines.append(target_summary.to_markdown(index=False, floatfmt=".5f"))
    lines.append("")

    if not source_summary.empty:
        lines.append("## Source-level summary")
        lines.append("")
        lines.append(source_summary.to_markdown(index=False, floatfmt=".5f"))
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("The output dataset contains several versions of each PM target:")
    lines.append("")
    lines.append("- `pm_<metric>`: absolute PM value at window `t`.")
    lines.append("- `pm_<metric>_delta_next`: change from window `t` to `t+1` inside the same record.")
    lines.append("- `pm_<metric>_trend_next`: categorical trend: `up`, `stable`, `down`.")
    lines.append("- `pm_<metric>_fast`: deviation from local rolling mean.")
    lines.append("- `pm_<metric>_slow`: local rolling mean.")
    lines.append("")
    lines.append("Recommended next step: train comparable baselines for absolute PM, delta PM and trend labels.")
    lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> DynamicsConfig:
    parser = argparse.ArgumentParser(
        description="Build PM dynamics targets: delta, trend, fast and slow components."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/windowed_eeg_pm_dataset_w10.csv"),
        help="Input windowed EEG/PM dataset. CSV or parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/state_dynamics/pm_w10"),
        help="Output directory.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="pm_state_dynamics_w10",
        help="Base output filename without extension.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Rolling window size in number of EEG/PM windows.",
    )
    parser.add_argument(
        "--trend-threshold",
        type=float,
        default=0.02,
        help="Threshold for up/stable/down trend labels.",
    )
    parser.add_argument(
        "--min-valid-pm",
        type=int,
        default=4,
        help="Minimum non-missing PM metrics for row-level validity flag.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size for quick tests. Not recommended for final run.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for sampling.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plots.",
    )
    args = parser.parse_args()

    return DynamicsConfig(
        dataset=args.dataset,
        output_dir=args.output_dir,
        output_name=args.output_name,
        rolling_window=args.rolling_window,
        trend_threshold=args.trend_threshold,
        min_valid_pm=args.min_valid_pm,
        sample_size=args.sample_size,
        random_state=args.random_state,
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
    logger.info("Build PM state dynamics")
    logger.info("=" * 80)
    logger.info("Dataset: %s", config.dataset)
    logger.info("Output dir: %s", config.output_dir)

    if not config.dataset.exists():
        raise FileNotFoundError(f"Dataset was not found: {config.dataset}")

    columns = read_columns(config.dataset)
    id_cols = detect_id_columns(columns)
    pm_cols = detect_pm_columns(columns)

    if len(pm_cols) < 3:
        raise ValueError(f"Too few PM columns detected: {pm_cols}")

    config.min_valid_pm = min(config.min_valid_pm, len(pm_cols))

    group_cols = choose_group_columns(id_cols)
    sort_cols = choose_sort_columns(id_cols)

    logger.info("Detected ID columns: %s", id_cols)
    logger.info("Detected PM columns: %s", pm_cols)
    logger.info("Group columns: %s", group_cols)
    logger.info("Sort columns: %s", sort_cols)

    df = load_dataset(config, id_cols=id_cols, pm_cols=pm_cols)
    logger.info("Loaded rows: %d", len(df))

    base_df = prepare_base_frame(df, id_cols=id_cols, pm_cols=pm_cols)
    pm_names = list(pm_cols.keys())

    base_df = add_validity_columns(base_df, pm_names=pm_names, min_valid_pm=config.min_valid_pm)

    dyn_df = build_dynamics_dataset(
        df=base_df,
        pm_names=pm_names,
        group_cols=group_cols,
        sort_cols=sort_cols,
        rolling_window=config.rolling_window,
        trend_threshold=config.trend_threshold,
    )

    target_summary = summarize_pm_targets(dyn_df, pm_names=pm_names)
    source_summary = summarize_by_source(
        dyn_df,
        pm_names=pm_names,
        source_col=id_cols.get("source"),
    )
    corr_df = compute_pm_dynamics_correlations(dyn_df, pm_names=pm_names)

    output_csv = config.output_dir / f"{config.output_name}.csv"
    output_parquet = config.output_dir / f"{config.output_name}.parquet"

    dyn_df.to_csv(output_csv, index=False)

    try:
        dyn_df.to_parquet(output_parquet, index=False)
        parquet_saved = True
    except Exception as exc:
        logger.warning("Could not save parquet: %s", exc)
        parquet_saved = False

    target_summary.to_csv(config.output_dir / "pm_dynamics_target_summary.csv", index=False)
    source_summary.to_csv(config.output_dir / "pm_dynamics_source_summary.csv", index=False)
    corr_df.to_csv(config.output_dir / "pm_dynamics_spearman_correlations.csv", index=False)

    summary = {
        "dataset": str(config.dataset),
        "output_dir": str(config.output_dir),
        "output_name": config.output_name,
        "rows_loaded": int(len(df)),
        "rows_saved": int(len(dyn_df)),
        "pm_metrics": pm_names,
        "id_columns": id_cols,
        "pm_columns": pm_cols,
        "group_columns": group_cols,
        "sort_columns": sort_cols,
        "rolling_window": int(config.rolling_window),
        "trend_threshold": float(config.trend_threshold),
        "min_valid_pm": int(config.min_valid_pm),
        "parquet_saved": parquet_saved,
    }
    save_json(config.output_dir / "summary.json", summary)

    if not config.no_plots:
        plot_trend_distribution(
            target_summary,
            figures_dir / "trend_distribution_by_pm.png",
        )
        plot_delta_abs(
            target_summary,
            figures_dir / "mean_abs_delta_by_pm.png",
        )
        plot_fast_slow_std(
            target_summary,
            figures_dir / "fast_slow_std_by_pm.png",
        )

    write_report(
        output_dir=config.output_dir,
        config=config,
        dataset_info=summary,
        id_cols=id_cols,
        pm_cols=pm_cols,
        group_cols=group_cols,
        sort_cols=sort_cols,
        target_summary=target_summary,
        source_summary=source_summary,
    )

    logger.info("=" * 80)
    logger.info("Saved PM state dynamics outputs")
    logger.info("=" * 80)
    logger.info("Dynamics CSV: %s", output_csv)
    if parquet_saved:
        logger.info("Dynamics parquet: %s", output_parquet)
    logger.info("Target summary: %s", config.output_dir / "pm_dynamics_target_summary.csv")
    logger.info("Source summary: %s", config.output_dir / "pm_dynamics_source_summary.csv")
    logger.info("Report: %s", config.output_dir / "report.md")
    logger.info("Done.")


if __name__ == "__main__":
    main()