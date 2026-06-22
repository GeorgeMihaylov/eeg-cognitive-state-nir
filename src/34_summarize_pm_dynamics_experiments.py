from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class Config:
    input_dirs: list[Path]
    output_dir: Path
    run_name: str
    no_plots: bool


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("summarize_pm_dynamics")


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File was not found: {path}")
    return pd.read_csv(path)


def infer_experiment_name(input_dir: Path) -> str:
    name = input_dir.name

    if "cross_source" in name:
        return "cross_source"

    if "fast_slow" in name:
        return "fast_slow"

    if "test" in name:
        return "test_fixed"

    if "fixed" in name:
        return "fixed"

    return name


def add_experiment_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def map_group(name: str) -> str:
        name = str(name)

        if name in {"fixed", "fast_slow"}:
            return "fixed_extended"

        if name == "cross_source":
            return "cross_source"

        if name == "test_fixed":
            return "test_fixed"

        return name

    out["experiment_group"] = out["experiment"].map(map_group)
    return out


def load_experiment(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    summary_path = input_dir / "summary.csv"
    fold_path = input_dir / "fold_metrics.csv"
    json_path = input_dir / "summary.json"

    summary = safe_read_csv(summary_path)
    folds = pd.read_csv(fold_path) if fold_path.exists() else None

    meta = {}
    if json_path.exists():
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    experiment_name = infer_experiment_name(input_dir)

    summary.insert(0, "experiment", experiment_name)
    summary.insert(1, "input_dir", str(input_dir))

    if folds is not None:
        folds.insert(0, "experiment", experiment_name)
        folds.insert(1, "input_dir", str(input_dir))

    return summary, folds, meta


def combine_experiments(input_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    summaries = []
    folds = []
    metas = []

    for input_dir in input_dirs:
        summary, fold_df, meta = load_experiment(input_dir)
        summaries.append(summary)

        if fold_df is not None:
            folds.append(fold_df)

        metas.append({"input_dir": str(input_dir), "meta": meta})

    summary_all = pd.concat(summaries, ignore_index=True)
    folds_all = pd.concat(folds, ignore_index=True) if folds else pd.DataFrame()

    return summary_all, folds_all, metas


def add_quality_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()

    out["main_metric"] = np.nan
    out["main_metric_name"] = ""

    is_reg = out["task_type"] == "regression"
    is_clf = out["task_type"] == "classification"

    if "r2_mean" in out.columns:
        out.loc[is_reg, "main_metric"] = out.loc[is_reg, "r2_mean"]
        out.loc[is_reg, "main_metric_name"] = "r2_mean"

    if "balanced_accuracy_mean" in out.columns:
        out.loc[is_clf, "main_metric"] = out.loc[is_clf, "balanced_accuracy_mean"]
        out.loc[is_clf, "main_metric_name"] = "balanced_accuracy_mean"

    return out


def build_best_by_target_mode(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = [
        "experiment_group",
        "target",
        "target_mode",
        "task_type",
        "validation",
    ]

    for keys, sub in summary.groupby(group_cols, dropna=False):
        sub = sub.copy()
        sub = sub.sort_values("main_metric", ascending=False, na_position="last")
        best = sub.iloc[0].to_dict()
        rows.append(best)

    return pd.DataFrame(rows)


def build_random_vs_subject_drop(best: pd.DataFrame) -> pd.DataFrame:
    fixed = best[best["experiment_group"] == "fixed_extended"].copy()

    if fixed.empty:
        return pd.DataFrame()

    key_cols = ["experiment_group", "target", "target_mode", "task_type", "model"]

    random_df = fixed[fixed["validation"] == "random_split"][
        key_cols + ["main_metric", "main_metric_name"]
    ].rename(columns={"main_metric": "random_main_metric"})

    subject_df = fixed[fixed["validation"] == "groupkfold_subject"][
        key_cols + ["main_metric"]
    ].rename(columns={"main_metric": "subject_main_metric"})

    merged = pd.merge(
        random_df,
        subject_df,
        on=key_cols,
        how="inner",
    )

    merged["drop_random_to_subject"] = (
        merged["random_main_metric"] - merged["subject_main_metric"]
    )

    denom = merged["random_main_metric"].replace(0, np.nan)
    merged["relative_drop_random_to_subject"] = (
        merged["drop_random_to_subject"] / denom
    )

    return merged.sort_values("drop_random_to_subject", ascending=False).reset_index(drop=True)


def build_cross_source_matrix(best: pd.DataFrame) -> pd.DataFrame:
    cross = best[best["experiment_group"] == "cross_source"].copy()

    if cross.empty:
        return pd.DataFrame()

    rows = []

    for _, row in cross.iterrows():
        validation = str(row["validation"])

        direction = validation
        if "cross_source_train_" in validation:
            direction = validation.replace("cross_source_train_", "")
            direction = direction.replace("_test_", " -> ")

        rows.append(
            {
                "target": row["target"],
                "target_mode": row["target_mode"],
                "task_type": row["task_type"],
                "direction": direction,
                "model": row["model"],
                "main_metric_name": row["main_metric_name"],
                "main_metric": row["main_metric"],
                "r2_mean": row.get("r2_mean", np.nan),
                "spearman_mean": row.get("spearman_mean", np.nan),
                "balanced_accuracy_mean": row.get("balanced_accuracy_mean", np.nan),
                "macro_f1_mean": row.get("macro_f1_mean", np.nan),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["target", "target_mode", "main_metric"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def infer_regression_interpretation(target: str, best_subject_mode: str | None) -> str:
    if best_subject_mode == "slow":
        base = "slow/background regression target"
    elif best_subject_mode == "absolute":
        base = "absolute-level regression target"
    elif best_subject_mode == "fast":
        base = "fast/local-deviation regression target"
    elif best_subject_mode == "delta":
        base = "delta/change regression target"
    else:
        base = "regression target requires manual interpretation"

    if target in {"stress", "excitement"}:
        return f"{base}; arousal-related PM"
    if target in {"attention", "focus"}:
        return f"{base}; cognitive-control PM"
    if target == "relaxation":
        return f"{base}; recovery/fatigue-related PM"
    if target in {"engagement", "interest"}:
        return f"{base}; involvement-related PM"

    return base


def infer_classification_interpretation(
    target: str,
    best_subject_mode: str | None,
    best_subject_balanced_accuracy: float | None,
) -> str:
    if best_subject_mode == "trend":
        base = "direction-of-change classification target"
    else:
        base = "classification target requires manual interpretation"

    if best_subject_balanced_accuracy is not None and pd.notna(best_subject_balanced_accuracy):
        if best_subject_balanced_accuracy >= 0.45:
            strength = "relatively stronger subject-wise trend signal"
        elif best_subject_balanced_accuracy >= 0.40:
            strength = "weak but usable subject-wise trend signal"
        else:
            strength = "weak subject-wise trend signal"
    else:
        strength = "trend strength unavailable"

    if target in {"stress", "excitement"}:
        return f"{base}; {strength}; arousal-related PM"
    if target in {"attention", "focus"}:
        return f"{base}; {strength}; cognitive-control PM"
    if target == "relaxation":
        return f"{base}; {strength}; recovery/fatigue-related PM"
    if target in {"engagement", "interest"}:
        return f"{base}; {strength}; involvement-related PM"

    return f"{base}; {strength}"


def build_regression_recommendation_table(best: pd.DataFrame) -> pd.DataFrame:
    """
    Regression recommendations only:
    absolute / delta / fast / slow are compared by R².
    Classification metrics are not mixed with regression metrics.
    """
    reg = best[
        (best["experiment_group"] == "fixed_extended")
        & (best["task_type"] == "regression")
    ].copy()

    if reg.empty:
        return pd.DataFrame()

    rows = []

    for target, sub in reg.groupby("target", dropna=False):
        random_sub = sub[sub["validation"] == "random_split"].copy()
        subject_sub = sub[sub["validation"] == "groupkfold_subject"].copy()

        def best_row(df: pd.DataFrame) -> dict:
            if df.empty:
                return {}
            return df.sort_values("r2_mean", ascending=False, na_position="last").iloc[0].to_dict()

        br_random = best_row(random_sub)
        br_subject = best_row(subject_sub)

        rows.append(
            {
                "target": target,
                "best_random_regression_mode": br_random.get("target_mode"),
                "best_random_r2": br_random.get("r2_mean"),
                "best_random_spearman": br_random.get("spearman_mean"),
                "best_subject_regression_mode": br_subject.get("target_mode"),
                "best_subject_r2": br_subject.get("r2_mean"),
                "best_subject_spearman": br_subject.get("spearman_mean"),
                "interpretation": infer_regression_interpretation(
                    target=target,
                    best_subject_mode=br_subject.get("target_mode"),
                ),
            }
        )

    return pd.DataFrame(rows)


def build_classification_recommendation_table(best: pd.DataFrame) -> pd.DataFrame:
    """
    Classification recommendations only:
    trend is evaluated by balanced accuracy / macro F1.
    Regression metrics are not mixed with classification metrics.
    """
    clf = best[
        (best["experiment_group"] == "fixed_extended")
        & (best["task_type"] == "classification")
    ].copy()

    if clf.empty:
        return pd.DataFrame()

    rows = []

    for target, sub in clf.groupby("target", dropna=False):
        random_sub = sub[sub["validation"] == "random_split"].copy()
        subject_sub = sub[sub["validation"] == "groupkfold_subject"].copy()

        def best_row(df: pd.DataFrame) -> dict:
            if df.empty:
                return {}
            return df.sort_values(
                "balanced_accuracy_mean",
                ascending=False,
                na_position="last",
            ).iloc[0].to_dict()

        br_random = best_row(random_sub)
        br_subject = best_row(subject_sub)

        rows.append(
            {
                "target": target,
                "best_random_classification_mode": br_random.get("target_mode"),
                "best_random_balanced_accuracy": br_random.get("balanced_accuracy_mean"),
                "best_random_macro_f1": br_random.get("macro_f1_mean"),
                "best_subject_classification_mode": br_subject.get("target_mode"),
                "best_subject_balanced_accuracy": br_subject.get("balanced_accuracy_mean"),
                "best_subject_macro_f1": br_subject.get("macro_f1_mean"),
                "interpretation": infer_classification_interpretation(
                    target=target,
                    best_subject_mode=br_subject.get("target_mode"),
                    best_subject_balanced_accuracy=br_subject.get("balanced_accuracy_mean"),
                ),
            }
        )

    return pd.DataFrame(rows)


def build_cross_source_recommendation_table(cross: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-source recommendations, keeping regression and classification metrics separate.
    """
    if cross.empty:
        return pd.DataFrame()

    rows = []

    for target, sub in cross.groupby("target", dropna=False):
        reg = sub[sub["task_type"] == "regression"].copy()
        clf = sub[sub["task_type"] == "classification"].copy()

        best_reg = {}
        if not reg.empty:
            best_reg = reg.sort_values("r2_mean", ascending=False, na_position="last").iloc[0].to_dict()

        best_clf = {}
        if not clf.empty:
            best_clf = clf.sort_values(
                "balanced_accuracy_mean",
                ascending=False,
                na_position="last",
            ).iloc[0].to_dict()

        rows.append(
            {
                "target": target,
                "best_cross_regression_mode": best_reg.get("target_mode"),
                "best_cross_regression_direction": best_reg.get("direction"),
                "best_cross_r2": best_reg.get("r2_mean"),
                "best_cross_spearman": best_reg.get("spearman_mean"),
                "best_cross_classification_mode": best_clf.get("target_mode"),
                "best_cross_classification_direction": best_clf.get("direction"),
                "best_cross_balanced_accuracy": best_clf.get("balanced_accuracy_mean"),
                "best_cross_macro_f1": best_clf.get("macro_f1_mean"),
            }
        )

    return pd.DataFrame(rows)


def plot_metric_by_mode(
    summary: pd.DataFrame,
    experiment_group: str,
    validation: str,
    metric_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    sub = summary[
        (summary["experiment_group"] == experiment_group)
        & (summary["validation"] == validation)
    ].copy()

    if sub.empty or metric_col not in sub.columns:
        return

    sub = sub[np.isfinite(sub[metric_col])].copy()
    if sub.empty:
        return

    pivot = sub.pivot_table(
        index="target",
        columns="target_mode",
        values=metric_col,
        aggfunc="max",
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("PM target")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_cross_source(cross: pd.DataFrame, output_dir: Path) -> None:
    if cross.empty:
        return

    for mode, metric_col, title, ylabel in [
        ("absolute", "r2_mean", "Cross-source absolute PM prediction", "R²"),
        ("delta", "r2_mean", "Cross-source delta PM prediction", "R²"),
        ("trend", "balanced_accuracy_mean", "Cross-source trend classification", "Balanced accuracy"),
    ]:
        sub = cross[cross["target_mode"] == mode].copy()

        if sub.empty or metric_col not in sub.columns:
            continue

        sub = sub[np.isfinite(sub[metric_col])].copy()
        if sub.empty:
            continue

        pivot = sub.pivot_table(
            index="target",
            columns="direction",
            values=metric_col,
            aggfunc="max",
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("PM target")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"cross_source_{mode}.png", dpi=160)
        plt.close(fig)


def top_rows_for_report(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if df.empty:
        return df

    cols = [
        c
        for c in [
            "experiment_group",
            "experiment",
            "target",
            "target_mode",
            "task_type",
            "validation",
            "direction",
            "model",
            "main_metric_name",
            "main_metric",
            "r2_mean",
            "spearman_mean",
            "balanced_accuracy_mean",
            "macro_f1_mean",
        ]
        if c in df.columns
    ]

    return df.sort_values("main_metric", ascending=False, na_position="last")[cols].head(n)


def write_report(
    config: Config,
    summary: pd.DataFrame,
    best: pd.DataFrame,
    drops: pd.DataFrame,
    cross: pd.DataFrame,
    regression_recommendations: pd.DataFrame,
    classification_recommendations: pd.DataFrame,
    cross_source_recommendations: pd.DataFrame,
    metas: list[dict],
) -> None:
    lines = []

    lines.append("# PM dynamics experiments summary")
    lines.append("")
    lines.append("## Input experiments")
    lines.append("")

    for d in config.input_dirs:
        lines.append(f"- `{d}`")

    lines.append("")
    lines.append("## Combined data")
    lines.append("")
    lines.append(f"- Summary rows: `{len(summary)}`")
    lines.append(f"- Best rows: `{len(best)}`")
    lines.append("")
    lines.append("## Experiment groups")
    lines.append("")

    group_counts = (
        summary.groupby(["experiment_group", "experiment"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["experiment_group", "experiment"])
    )
    lines.append(group_counts.to_markdown(index=False))
    lines.append("")

    lines.append("## Best fixed-extended experiment rows")
    lines.append("")
    fixed_top = top_rows_for_report(best[best["experiment_group"] == "fixed_extended"], n=40)
    lines.append(
        fixed_top.to_markdown(index=False, floatfmt=".5f")
        if not fixed_top.empty
        else "No fixed-extended rows."
    )
    lines.append("")

    lines.append("## Random split to subject-wise drop")
    lines.append("")
    if not drops.empty:
        display_cols = [
            "target",
            "target_mode",
            "task_type",
            "model",
            "random_main_metric",
            "subject_main_metric",
            "drop_random_to_subject",
            "relative_drop_random_to_subject",
        ]
        display_cols = [c for c in display_cols if c in drops.columns]
        lines.append(drops[display_cols].head(50).to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No random-vs-subject comparison rows.")

    lines.append("")
    lines.append("## Cross-source results")
    lines.append("")
    if not cross.empty:
        lines.append(top_rows_for_report(cross, n=50).to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No cross-source rows.")

    lines.append("")
    lines.append("## Regression recommendations by PM target")
    lines.append("")
    if not regression_recommendations.empty:
        lines.append(regression_recommendations.to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No regression recommendation rows.")

    lines.append("")
    lines.append("## Classification recommendations by PM target")
    lines.append("")
    if not classification_recommendations.empty:
        lines.append(classification_recommendations.to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No classification recommendation rows.")

    lines.append("")
    lines.append("## Cross-source recommendations by PM target")
    lines.append("")
    if not cross_source_recommendations.empty:
        lines.append(cross_source_recommendations.to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No cross-source recommendation rows.")

    lines.append("")
    lines.append("## Working interpretation")
    lines.append("")
    lines.append("1. Regression and classification metrics are not compared directly.")
    lines.append("2. Slow PM components should be interpreted as background-state regression targets.")
    lines.append("3. Trend PM targets should be interpreted as direction-of-change classification targets.")
    lines.append("4. Absolute PM targets remain useful, but they are more sensitive to subject-wise validation.")
    lines.append("5. Delta regression is not uniformly better than absolute or slow regression.")
    lines.append("6. The results support analyzing PM metrics as heterogeneous state indicators rather than as one uniform target family.")
    lines.append("")
    lines.append("## Suggested next steps")
    lines.append("")
    lines.append("- Compare slow targets against latent PM coordinates.")
    lines.append("- Add subject-calibration experiments: zero-shot vs 5%, 10%, 20% calibration.")
    lines.append("- Validate whether cross-source trend quality is affected by source/protocol artifacts.")
    lines.append("- Add motion/artifact reliability analysis where movement or quality indicators are available.")
    lines.append("")

    (config.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Summarize PM dynamics baseline experiments."
    )
    parser.add_argument(
        "--input-dirs",
        type=str,
        default=(
            "reports/state_dynamics/pm_w10_baselines_fixed,"
            "reports/state_dynamics/pm_w10_baselines_cross_source_fixed,"
            "reports/state_dynamics/pm_w10_baselines_test_fixed,"
            "reports/state_dynamics/pm_w10_baselines_fast_slow"
        ),
        help="Comma-separated experiment directories containing summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/state_dynamics/pm_w10_experiment_summary_v4"),
        help="Output directory.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="pm_w10_experiment_summary_v4",
        help="Run name.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plots.",
    )

    args = parser.parse_args()

    input_dirs = [Path(x.strip()) for x in args.input_dirs.split(",") if x.strip()]

    return Config(
        input_dirs=input_dirs,
        output_dir=args.output_dir,
        run_name=args.run_name,
        no_plots=args.no_plots,
    )


def main() -> None:
    logger = setup_logging()
    config = parse_args()

    config.input_dirs = [p.resolve() for p in config.input_dirs]
    config.output_dir = config.output_dir.resolve()
    figures_dir = config.output_dir / "figures"

    config.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Summarize PM dynamics experiments")
    logger.info("=" * 80)
    logger.info("Input dirs:")
    for d in config.input_dirs:
        logger.info("  %s", d)
    logger.info("Output dir: %s", config.output_dir)

    summary, folds, metas = combine_experiments(config.input_dirs)
    summary = add_quality_columns(summary)
    summary = add_experiment_group(summary)

    best = build_best_by_target_mode(summary)
    drops = build_random_vs_subject_drop(best)
    cross = build_cross_source_matrix(best)

    regression_recommendations = build_regression_recommendation_table(best)
    classification_recommendations = build_classification_recommendation_table(best)
    cross_source_recommendations = build_cross_source_recommendation_table(cross)

    summary.to_csv(config.output_dir / "combined_summary.csv", index=False)

    if not folds.empty:
        folds = add_experiment_group(folds)
        folds.to_csv(config.output_dir / "combined_fold_metrics.csv", index=False)

    best.to_csv(config.output_dir / "best_by_target_mode_validation.csv", index=False)
    drops.to_csv(config.output_dir / "random_to_subject_drop.csv", index=False)
    cross.to_csv(config.output_dir / "cross_source_matrix.csv", index=False)

    regression_recommendations.to_csv(
        config.output_dir / "target_recommendations_regression.csv",
        index=False,
    )
    classification_recommendations.to_csv(
        config.output_dir / "target_recommendations_classification.csv",
        index=False,
    )
    cross_source_recommendations.to_csv(
        config.output_dir / "target_recommendations_cross_source.csv",
        index=False,
    )

    legacy_recommendations = pd.concat(
        [
            regression_recommendations.assign(recommendation_type="regression"),
            classification_recommendations.assign(recommendation_type="classification"),
        ],
        ignore_index=True,
        sort=False,
    )
    legacy_recommendations.to_csv(
        config.output_dir / "target_recommendations.csv",
        index=False,
    )

    (config.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_name": config.run_name,
                "input_dirs": [str(p) for p in config.input_dirs],
                "output_dir": str(config.output_dir),
                "n_combined_summary_rows": int(len(summary)),
                "n_combined_fold_rows": int(len(folds)),
                "n_best_rows": int(len(best)),
                "n_drop_rows": int(len(drops)),
                "n_cross_rows": int(len(cross)),
                "n_regression_recommendation_rows": int(len(regression_recommendations)),
                "n_classification_recommendation_rows": int(len(classification_recommendations)),
                "n_cross_source_recommendation_rows": int(len(cross_source_recommendations)),
                "metas": metas,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not config.no_plots:
        plot_metric_by_mode(
            summary=summary,
            experiment_group="fixed_extended",
            validation="random_split",
            metric_col="r2_mean",
            title="Regression targets: random split",
            ylabel="R²",
            output_path=figures_dir / "regression_r2_random_split.png",
        )
        plot_metric_by_mode(
            summary=summary,
            experiment_group="fixed_extended",
            validation="groupkfold_subject",
            metric_col="r2_mean",
            title="Regression targets: subject-wise validation",
            ylabel="R²",
            output_path=figures_dir / "regression_r2_groupkfold_subject.png",
        )
        plot_metric_by_mode(
            summary=summary,
            experiment_group="fixed_extended",
            validation="random_split",
            metric_col="balanced_accuracy_mean",
            title="Classification targets: random split",
            ylabel="Balanced accuracy",
            output_path=figures_dir / "classification_balanced_accuracy_random_split.png",
        )
        plot_metric_by_mode(
            summary=summary,
            experiment_group="fixed_extended",
            validation="groupkfold_subject",
            metric_col="balanced_accuracy_mean",
            title="Classification targets: subject-wise validation",
            ylabel="Balanced accuracy",
            output_path=figures_dir / "classification_balanced_accuracy_groupkfold_subject.png",
        )
        plot_cross_source(cross, figures_dir)

    write_report(
        config=config,
        summary=summary,
        best=best,
        drops=drops,
        cross=cross,
        regression_recommendations=regression_recommendations,
        classification_recommendations=classification_recommendations,
        cross_source_recommendations=cross_source_recommendations,
        metas=metas,
    )

    logger.info("=" * 80)
    logger.info("Saved PM dynamics experiment summary")
    logger.info("=" * 80)
    logger.info("Combined summary: %s", config.output_dir / "combined_summary.csv")
    logger.info("Best rows: %s", config.output_dir / "best_by_target_mode_validation.csv")
    logger.info("Drop table: %s", config.output_dir / "random_to_subject_drop.csv")
    logger.info("Cross-source matrix: %s", config.output_dir / "cross_source_matrix.csv")
    logger.info(
        "Regression recommendations: %s",
        config.output_dir / "target_recommendations_regression.csv",
    )
    logger.info(
        "Classification recommendations: %s",
        config.output_dir / "target_recommendations_classification.csv",
    )
    logger.info(
        "Cross-source recommendations: %s",
        config.output_dir / "target_recommendations_cross_source.csv",
    )
    logger.info(
        "Legacy recommendations: %s",
        config.output_dir / "target_recommendations.csv",
    )
    logger.info("Report: %s", config.output_dir / "report.md")
    logger.info("Done.")


if __name__ == "__main__":
    main()