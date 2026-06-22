from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler


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
    "window_start": ["window_start", "start_time", "t_start", "start"],
    "window_end": ["window_end", "end_time", "t_end", "end"],
}


@dataclass
class LatentConfig:
    dataset: Path
    output_dir: Path
    n_components: int
    sample_size: int | None
    random_state: int
    min_valid_pm: int
    make_plots: bool


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("pm_latent_states")


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

        # Fallback: looser matching for slightly different column names.
        short_lower = short_name.lower()
        candidates = [
            c
            for c in columns
            if short_lower in c.lower()
            and "pm" in c.lower()
            and ("mean" in c.lower() or "scaled" in c.lower())
        ]
        if candidates:
            found[short_name] = candidates[0]

    return found


def load_dataset(config: LatentConfig, id_cols: dict[str, str], pm_cols: dict[str, str]) -> pd.DataFrame:
    usecols = list(dict.fromkeys(list(id_cols.values()) + list(pm_cols.values())))

    if config.dataset.suffix.lower() == ".parquet":
        df = pd.read_parquet(config.dataset, columns=usecols)
    else:
        df = pd.read_csv(config.dataset, usecols=usecols)

    if config.sample_size is not None and len(df) > config.sample_size:
        df = df.sample(n=config.sample_size, random_state=config.random_state).reset_index(drop=True)

    return df


def clean_pm_matrix(
    df: pd.DataFrame,
    pm_cols: dict[str, str],
    min_valid_pm: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pm_df = df[list(pm_cols.values())].copy()
    rename_map = {v: k for k, v in pm_cols.items()}
    pm_df = pm_df.rename(columns=rename_map)

    for col in pm_df.columns:
        pm_df[col] = pd.to_numeric(pm_df[col], errors="coerce")

    valid_count = pm_df.notna().sum(axis=1)
    mask = valid_count >= min_valid_pm

    pm_clean = pm_df.loc[mask].copy()

    # Fill remaining missing values by column median.
    medians = pm_clean.median(axis=0, numeric_only=True)
    pm_clean = pm_clean.fillna(medians)

    meta_clean = df.loc[mask].copy().reset_index(drop=True)
    pm_clean = pm_clean.reset_index(drop=True)

    return meta_clean, pm_clean


def build_pca(pm_scaled: np.ndarray, pm_names: list[str], n_components: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PCA]:
    pca = PCA(n_components=n_components, random_state=0)
    coords = pca.fit_transform(pm_scaled)

    coord_cols = [f"pca_{i + 1}" for i in range(n_components)]
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


def build_factor_analysis(pm_scaled: np.ndarray, pm_names: list[str], n_components: int) -> tuple[pd.DataFrame, pd.DataFrame, FactorAnalysis]:
    fa = FactorAnalysis(n_components=n_components, random_state=0)
    coords = fa.fit_transform(pm_scaled)

    coord_cols = [f"factor_{i + 1}" for i in range(n_components)]
    coords_df = pd.DataFrame(coords, columns=coord_cols)

    loadings_df = pd.DataFrame(
        fa.components_.T,
        index=pm_names,
        columns=coord_cols,
    ).reset_index(names="pm_metric")

    return coords_df, loadings_df, fa


def compute_correlations(pm_df: pd.DataFrame, coords_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    for pm_col in pm_df.columns:
        for latent_col in coords_df.columns:
            pearson = pm_df[pm_col].corr(coords_df[latent_col], method="pearson")
            spearman = pm_df[pm_col].corr(coords_df[latent_col], method="spearman")
            rows.append(
                {
                    "method": prefix,
                    "pm_metric": pm_col,
                    "latent_component": latent_col,
                    "pearson": pearson,
                    "spearman": spearman,
                    "abs_pearson": abs(pearson) if pd.notna(pearson) else np.nan,
                    "abs_spearman": abs(spearman) if pd.notna(spearman) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def infer_latent_axis_names(loadings_df: pd.DataFrame, component_prefix: str) -> pd.DataFrame:
    rows = []
    component_cols = [c for c in loadings_df.columns if c.startswith(component_prefix)]

    for component in component_cols:
        sub = loadings_df[["pm_metric", component]].copy()
        sub["abs_loading"] = sub[component].abs()
        sub = sub.sort_values("abs_loading", ascending=False)

        top = sub.head(3)
        top_metrics = ", ".join(top["pm_metric"].tolist())

        suggested_name = "Unlabeled"
        metrics = set(top["pm_metric"].tolist())

        if {"stress", "excitement"} & metrics:
            suggested_name = "Stress / Arousal"
        if {"focus", "attention"} & metrics:
            suggested_name = "Workload / Attention"
        if "relaxation" in metrics:
            suggested_name = "Recovery / Fatigue"
        if {"engagement", "interest"} & metrics:
            suggested_name = "Engagement / Involvement"

        rows.append(
            {
                "component": component,
                "suggested_latent_state": suggested_name,
                "top_metrics": top_metrics,
                "top_abs_loading_sum": top["abs_loading"].sum(),
            }
        )

    return pd.DataFrame(rows)


def plot_explained_variance(explained_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(explained_df["component"], explained_df["explained_variance_ratio"])
    ax.plot(
        explained_df["component"],
        explained_df["cumulative_explained_variance_ratio"],
        marker="o",
    )
    ax.set_title("PCA explained variance")
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance ratio")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_loadings_heatmap(loadings_df: pd.DataFrame, output_path: Path, title: str) -> None:
    component_cols = [c for c in loadings_df.columns if c != "pm_metric"]
    matrix = loadings_df.set_index("pm_metric")[component_cols]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix.values, aspect="auto")
    ax.set_xticks(np.arange(len(component_cols)))
    ax.set_xticklabels(component_cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_latent_scatter(coords_df: pd.DataFrame, meta_df: pd.DataFrame, output_path: Path, title: str) -> None:
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
        unique_sources = sorted(sources.unique().tolist())
        for source in unique_sources:
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

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(
    output_dir: Path,
    config: LatentConfig,
    dataset_info: dict,
    pm_cols: dict[str, str],
    id_cols: dict[str, str],
    pca_explained: pd.DataFrame,
    pca_axis_summary: pd.DataFrame,
    fa_axis_summary: pd.DataFrame,
    correlation_df: pd.DataFrame,
) -> None:
    top_corr = (
        correlation_df.sort_values("abs_spearman", ascending=False)
        .head(20)
        .loc[:, ["method", "pm_metric", "latent_component", "pearson", "spearman"]]
    )

    lines = []
    lines.append("# PM latent states report")
    lines.append("")
    lines.append("## Input")
    lines.append("")
    lines.append(f"- Dataset: `{config.dataset}`")
    lines.append(f"- Output dir: `{config.output_dir}`")
    lines.append(f"- Rows loaded: `{dataset_info['rows_loaded']}`")
    lines.append(f"- Rows used after PM filtering: `{dataset_info['rows_used']}`")
    lines.append(f"- PM metrics used: `{dataset_info['pm_metrics_used']}`")
    lines.append(f"- PCA / Factor components: `{config.n_components}`")
    lines.append("")
    lines.append("## Detected ID columns")
    lines.append("")
    if id_cols:
        for logical, col in id_cols.items():
            lines.append(f"- `{logical}` → `{col}`")
    else:
        lines.append("- No ID columns detected.")
    lines.append("")
    lines.append("## Detected PM columns")
    lines.append("")
    for short_name, col in pm_cols.items():
        lines.append(f"- `{short_name}` → `{col}`")
    lines.append("")
    lines.append("## PCA explained variance")
    lines.append("")
    lines.append(pca_explained.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Suggested PCA latent axes")
    lines.append("")
    lines.append(pca_axis_summary.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Suggested Factor Analysis latent axes")
    lines.append("")
    lines.append(fa_axis_summary.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Strongest PM ↔ latent component correlations")
    lines.append("")
    lines.append(top_corr.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Interpretation template")
    lines.append("")
    lines.append("Use the loadings table to manually validate the semantic interpretation of each axis:")
    lines.append("")
    lines.append("- `Stress / Arousal`: high loading from Stress and/or Excitement.")
    lines.append("- `Workload / Attention`: high loading from Focus and/or Attention.")
    lines.append("- `Recovery / Fatigue`: high loading from Relaxation, often with opposite sign to Stress/Arousal.")
    lines.append("- `Engagement / Involvement`: high loading from Engagement and/or Interest.")
    lines.append("")
    lines.append("Important: axis signs in PCA/FA are arbitrary. Interpret absolute loadings and relative direction, not the sign alone.")
    lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> LatentConfig:
    parser = argparse.ArgumentParser(
        description="Build latent state space from PM metrics in windowed EEG/PM dataset."
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
        default=Path("reports/latent_states"),
        help="Output directory.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=4,
        help="Number of PCA / Factor Analysis components.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row sample size for faster runs.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state.",
    )
    parser.add_argument(
        "--min-valid-pm",
        type=int,
        default=4,
        help="Minimum number of non-missing PM metrics required per row.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plots.",
    )
    args = parser.parse_args()

    return LatentConfig(
        dataset=args.dataset,
        output_dir=args.output_dir,
        n_components=args.n_components,
        sample_size=args.sample_size,
        random_state=args.random_state,
        min_valid_pm=args.min_valid_pm,
        make_plots=not args.no_plots,
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
    logger.info("Build PM latent states")
    logger.info("=" * 80)
    logger.info("Dataset: %s", config.dataset)
    logger.info("Output dir: %s", config.output_dir)

    if not config.dataset.exists():
        raise FileNotFoundError(f"Dataset was not found: {config.dataset}")

    columns = read_columns(config.dataset)
    id_cols = detect_id_columns(columns)
    pm_cols = detect_pm_columns(columns)

    if len(pm_cols) < 3:
        raise ValueError(
            "Too few PM columns detected. "
            f"Detected: {pm_cols}. Available columns example: {columns[:50]}"
        )

    n_components = min(config.n_components, len(pm_cols))
    config = LatentConfig(
        dataset=config.dataset,
        output_dir=config.output_dir,
        n_components=n_components,
        sample_size=config.sample_size,
        random_state=config.random_state,
        min_valid_pm=min(config.min_valid_pm, len(pm_cols)),
        make_plots=config.make_plots,
    )

    logger.info("Detected ID columns: %s", id_cols)
    logger.info("Detected PM columns: %s", pm_cols)

    df = load_dataset(config, id_cols=id_cols, pm_cols=pm_cols)
    logger.info("Loaded rows: %d", len(df))

    meta_df, pm_df = clean_pm_matrix(df, pm_cols=pm_cols, min_valid_pm=config.min_valid_pm)
    logger.info("Rows after PM filtering: %d", len(pm_df))

    scaler = StandardScaler()
    pm_scaled = scaler.fit_transform(pm_df.values)

    pm_names = list(pm_df.columns)

    pca_coords, pca_loadings, pca_explained, _ = build_pca(
        pm_scaled=pm_scaled,
        pm_names=pm_names,
        n_components=config.n_components,
    )

    fa_coords, fa_loadings, _ = build_factor_analysis(
        pm_scaled=pm_scaled,
        pm_names=pm_names,
        n_components=config.n_components,
    )

    pca_axis_summary = infer_latent_axis_names(pca_loadings, "pca_")
    fa_axis_summary = infer_latent_axis_names(fa_loadings, "factor_")

    pca_corr = compute_correlations(pm_df, pca_coords, "pca")
    fa_corr = compute_correlations(pm_df, fa_coords, "factor_analysis")
    correlation_df = pd.concat([pca_corr, fa_corr], ignore_index=True)

    # Save main outputs.
    output_meta = meta_df[list(id_cols.values())].copy() if id_cols else pd.DataFrame(index=meta_df.index)
    output_meta = output_meta.reset_index(drop=True)
    output_meta.insert(0, "row_index_filtered", np.arange(len(output_meta)))

    latent_coords = pd.concat(
        [
            output_meta,
            pca_coords.add_prefix("latent_"),
            fa_coords.add_prefix("latent_"),
        ],
        axis=1,
    )

    pm_clean = pd.concat([output_meta, pm_df], axis=1)

    latent_coords.to_csv(config.output_dir / "pm_latent_coordinates.csv", index=False)
    pm_clean.to_csv(config.output_dir / "pm_metrics_clean.csv", index=False)
    pca_loadings.to_csv(config.output_dir / "pca_loadings.csv", index=False)
    fa_loadings.to_csv(config.output_dir / "factor_analysis_loadings.csv", index=False)
    pca_explained.to_csv(config.output_dir / "pca_explained_variance.csv", index=False)
    pca_axis_summary.to_csv(config.output_dir / "pca_axis_summary.csv", index=False)
    fa_axis_summary.to_csv(config.output_dir / "factor_analysis_axis_summary.csv", index=False)
    correlation_df.to_csv(config.output_dir / "pm_latent_correlations.csv", index=False)

    summary = {
        "dataset": str(config.dataset),
        "output_dir": str(config.output_dir),
        "rows_loaded": int(len(df)),
        "rows_used": int(len(pm_df)),
        "id_columns": id_cols,
        "pm_columns": pm_cols,
        "pm_metrics_used": pm_names,
        "n_components": int(config.n_components),
        "sample_size": config.sample_size,
        "min_valid_pm": int(config.min_valid_pm),
        "pca_cumulative_explained_variance_ratio": pca_explained[
            "cumulative_explained_variance_ratio"
        ].tolist(),
    }
    save_json(config.output_dir / "summary.json", summary)

    if config.make_plots:
        plot_explained_variance(
            pca_explained,
            figures_dir / "pca_explained_variance.png",
        )
        plot_loadings_heatmap(
            pca_loadings,
            figures_dir / "pca_loadings_heatmap.png",
            title="PCA PM loadings",
        )
        plot_loadings_heatmap(
            fa_loadings,
            figures_dir / "factor_analysis_loadings_heatmap.png",
            title="Factor Analysis PM loadings",
        )
        plot_latent_scatter(
            pca_coords,
            meta_df,
            figures_dir / "pca_latent_scatter.png",
            title="PCA latent space",
        )
        plot_latent_scatter(
            fa_coords,
            meta_df,
            figures_dir / "factor_analysis_latent_scatter.png",
            title="Factor Analysis latent space",
        )

    write_report(
        output_dir=config.output_dir,
        config=config,
        dataset_info=summary,
        pm_cols=pm_cols,
        id_cols=id_cols,
        pca_explained=pca_explained,
        pca_axis_summary=pca_axis_summary,
        fa_axis_summary=fa_axis_summary,
        correlation_df=correlation_df,
    )

    logger.info("=" * 80)
    logger.info("Saved PM latent state outputs")
    logger.info("=" * 80)
    logger.info("Coordinates: %s", config.output_dir / "pm_latent_coordinates.csv")
    logger.info("PCA loadings: %s", config.output_dir / "pca_loadings.csv")
    logger.info("FA loadings: %s", config.output_dir / "factor_analysis_loadings.csv")
    logger.info("Report: %s", config.output_dir / "report.md")
    logger.info("Done.")


if __name__ == "__main__":
    main()