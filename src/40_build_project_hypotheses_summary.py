from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Config:
    output_dir: Path
    run_name: str

    latent_dir: Path
    slow_latent_dir: Path
    slow_latent_cross_source_dir: Path
    dynamics_summary_dir: Path
    dynamics_sensitivity_dir: Path
    user_calibration_dir: Path
    device_alignment_dir: Path
    wesad_dir: Path


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("project_hypotheses_summary")


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def fmt_float(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"

    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def make_output_dirs(config: Config) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)


def load_inputs(config: Config) -> dict[str, pd.DataFrame | dict]:
    return {
        "pm_pca_loadings": safe_read_csv(config.latent_dir / "pca_loadings.csv"),
        "pm_pca_axis_summary": safe_read_csv(config.latent_dir / "pca_axis_summary.csv"),
        "slow_pca_loadings": safe_read_csv(config.slow_latent_dir / "slow_pm_pca_loadings.csv"),
        "slow_axis_summary": safe_read_csv(config.slow_latent_dir / "slow_pm_axis_summary.csv"),
        "slow_latent_regression": safe_read_csv(config.slow_latent_dir / "latent_regression_summary.csv"),
        "slow_cross_summary": safe_read_csv(config.slow_latent_cross_source_dir / "summary.csv"),
        "slow_cross_r2": safe_read_csv(config.slow_latent_cross_source_dir / "r2_pivot.csv"),
        "slow_cross_spearman": safe_read_csv(config.slow_latent_cross_source_dir / "spearman_pivot.csv"),
        "dynamics_regression_recs": safe_read_csv(config.dynamics_summary_dir / "target_recommendations_regression.csv"),
        "dynamics_classification_recs": safe_read_csv(config.dynamics_summary_dir / "target_recommendations_classification.csv"),
        "dynamics_cross_source_recs": safe_read_csv(config.dynamics_summary_dir / "target_recommendations_cross_source.csv"),
        "calibration_regression": safe_read_csv(config.user_calibration_dir / "calibration_summary_regression.csv"),
        "calibration_classification": safe_read_csv(config.user_calibration_dir / "calibration_summary_classification.csv"),
        "calibration_gain": safe_read_csv(config.user_calibration_dir / "calibration_gain_vs_zero_shot.csv"),
        "sensitivity_best": safe_read_csv(config.dynamics_sensitivity_dir / "sensitivity_best_params.csv"),
        "sensitivity_stability": safe_read_csv(config.dynamics_sensitivity_dir / "sensitivity_stability.csv"),
        "trend_label_distribution": safe_read_csv(config.dynamics_sensitivity_dir / "trend_label_distribution.csv"),
        "device_mapping": safe_read_csv(config.device_alignment_dir / "device_metric_mapping.csv"),
        "evidence_matrix": safe_read_csv(config.device_alignment_dir / "latent_state_evidence_matrix.csv"),
        "helmet_vs_bracelet": safe_read_csv(config.device_alignment_dir / "helmet_vs_bracelet_comparison.csv"),
        "device_summary_json": safe_read_json(config.device_alignment_dir / "summary.json"),
        "sensitivity_summary_json": safe_read_json(config.dynamics_sensitivity_dir / "summary.json"),
        "cross_source_summary_json": safe_read_json(config.slow_latent_cross_source_dir / "summary.json"),
    }


def manual_latent_axis_table() -> pd.DataFrame:
    rows = [
        {
            "axis": "slow_pca_1",
            "interpretation": "Slow Stress / Arousal / General activation",
            "main_pm_metrics": "Excitement, Stress, Interest, Focus, Relaxation",
            "status": "interpretable",
        },
        {
            "axis": "slow_pca_2",
            "interpretation": "Slow Recovery vs Focus",
            "main_pm_metrics": "Relaxation, Focus, Interest, Engagement",
            "status": "interpretable; most transferable",
        },
        {
            "axis": "slow_pca_3",
            "interpretation": "Slow Attention vs Engagement",
            "main_pm_metrics": "Attention, Engagement, Interest, Focus",
            "status": "interpretable; weaker transfer",
        },
        {
            "axis": "slow_pca_4",
            "interpretation": "Slow Cognitive involvement",
            "main_pm_metrics": "Attention, Engagement, Interest",
            "status": "interpretable",
        },
    ]
    return pd.DataFrame(rows)


def build_hypothesis_table(data: dict[str, pd.DataFrame | dict]) -> pd.DataFrame:
    rows = [
        {
            "hypothesis_id": "H1",
            "hypothesis": "PM-метрики образуют интерпретируемое latent-state пространство.",
            "main_experiments": "PM PCA/FA; slow PM PCA; EEG/POW → slow latent regression; cross-source transfer",
            "key_result": "Выделены slow_pca_1..4; slow_pca_1/2/4 имеют содержательную интерпретацию и переносятся между Old_EEG и gpn_data.",
            "status": "supported",
            "main_limitation": "Нужна ручная интерпретация осей; требуется контроль возможной структуры источников.",
        },
        {
            "hypothesis_id": "H2",
            "hypothesis": "PM-состояния делятся на slow/background и trend/change-direction типы.",
            "main_experiments": "absolute/delta/fast/slow/trend baselines; regression/classification separation; sensitivity analysis",
            "key_result": "Slow лучше подходит для regression; trend лучше подходит для classification; sensitivity показал устойчивость вывода.",
            "status": "supported",
            "main_limitation": "Оптимальное rolling-window зависит от конкретной PM-метрики.",
        },
        {
            "hypothesis_id": "H3",
            "hypothesis": "Перенос на нового пользователя требует калибровки.",
            "main_experiments": "zero-shot vs 5%/10%/20% calibration vs subject-dependent",
            "key_result": "5–20% данных нового пользователя стабильно улучшают качество для slow latent regression и trend classification.",
            "status": "supported",
            "main_limitation": "Subject-dependent качество заметно выше, значит остается большой индивидуальный компонент.",
        },
        {
            "hypothesis_id": "H4",
            "hypothesis": "Шлемы и браслеты нужно сопоставлять через промежуточные состояния, а не напрямую по метрикам.",
            "main_experiments": "WESAD baseline/ablation; device metric alignment; evidence matrix",
            "key_result": "EEG/PM дает latent-state axes; WESAD дает stress/arousal proxy; ACC выступает как movement/context/reliability confounder.",
            "status": "partially supported",
            "main_limitation": "Smart-watch датасеты пока не подключены экспериментально; EEG reliability proxy еще не построен.",
        },
    ]

    return pd.DataFrame(rows)


def build_cross_source_highlight(data: dict[str, pd.DataFrame | dict]) -> pd.DataFrame:
    summary = data["slow_cross_summary"]

    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return pd.DataFrame()

    cols = [
        c
        for c in [
            "target",
            "direction",
            "model",
            "n_train",
            "n_test",
            "r2_mean",
            "spearman_mean",
            "mae_mean",
            "rmse_mean",
        ]
        if c in summary.columns
    ]

    out = summary[cols].copy()

    if "r2_mean" in out.columns:
        out = out.sort_values("r2_mean", ascending=False, na_position="last")

    return out


def build_best_state_dynamics_table(data: dict[str, pd.DataFrame | dict]) -> pd.DataFrame:
    reg = data["dynamics_regression_recs"]
    clf = data["dynamics_classification_recs"]

    rows = []

    if isinstance(reg, pd.DataFrame) and not reg.empty:
        for _, row in reg.iterrows():
            rows.append(
                {
                    "task_family": "slow/background regression",
                    "target": row.get("target"),
                    "best_mode": row.get("best_subject_regression_mode"),
                    "metric": "subject-wise R²",
                    "score": row.get("best_subject_r2"),
                    "interpretation": row.get("interpretation"),
                }
            )

    if isinstance(clf, pd.DataFrame) and not clf.empty:
        for _, row in clf.iterrows():
            rows.append(
                {
                    "task_family": "trend/change-direction classification",
                    "target": row.get("target"),
                    "best_mode": row.get("best_subject_classification_mode"),
                    "metric": "subject-wise balanced accuracy",
                    "score": row.get("best_subject_balanced_accuracy"),
                    "interpretation": row.get("interpretation"),
                }
            )

    out = pd.DataFrame(rows)

    if not out.empty:
        out["score"] = pd.to_numeric(out["score"], errors="coerce")
        out = out.sort_values(["task_family", "score"], ascending=[True, False])

    return out


def build_calibration_highlight(data: dict[str, pd.DataFrame | dict]) -> pd.DataFrame:
    gain = data["calibration_gain"]

    if not isinstance(gain, pd.DataFrame) or gain.empty:
        return pd.DataFrame()

    keep_modes = ["calibration_20pct", "subject_dependent"]

    out = gain[gain["mode"].isin(keep_modes)].copy()

    cols = [
        c
        for c in [
            "task_type",
            "target",
            "mode",
            "calibration_frac",
            "metric_name",
            "zero_shot_metric",
            "current_metric",
            "absolute_gain_vs_zero_shot",
        ]
        if c in out.columns
    ]

    out = out[cols].copy()

    if "absolute_gain_vs_zero_shot" in out.columns:
        out["absolute_gain_vs_zero_shot"] = pd.to_numeric(
            out["absolute_gain_vs_zero_shot"], errors="coerce"
        )
        out = out.sort_values("absolute_gain_vs_zero_shot", ascending=False)

    return out


def build_sensitivity_highlight(data: dict[str, pd.DataFrame | dict]) -> pd.DataFrame:
    stability = data["sensitivity_stability"]

    if not isinstance(stability, pd.DataFrame) or stability.empty:
        return pd.DataFrame()

    cols = [
        c
        for c in [
            "task_type",
            "target",
            "validation",
            "metric",
            "metric_mean_across_params",
            "metric_std_across_params",
            "metric_min_across_params",
            "metric_max_across_params",
            "metric_range_across_params",
        ]
        if c in stability.columns
    ]

    out = stability[cols].copy()

    if "metric_range_across_params" in out.columns:
        out["metric_range_across_params"] = pd.to_numeric(
            out["metric_range_across_params"], errors="coerce"
        )
        out = out.sort_values("metric_range_across_params", ascending=True)

    return out


def build_best_sensitivity_params(data: dict[str, pd.DataFrame | dict]) -> pd.DataFrame:
    best = data["sensitivity_best"]

    if not isinstance(best, pd.DataFrame) or best.empty:
        return pd.DataFrame()

    cols = [
        c
        for c in [
            "task_type",
            "target",
            "validation",
            "rolling_window",
            "trend_threshold",
            "selection_metric",
            "selection_value",
        ]
        if c in best.columns
    ]

    return best[cols].copy()


def build_device_alignment_highlight(data: dict[str, pd.DataFrame | dict]) -> pd.DataFrame:
    evidence = data["evidence_matrix"]

    if not isinstance(evidence, pd.DataFrame) or evidence.empty:
        return pd.DataFrame()

    cols = [
        c
        for c in [
            "latent_state",
            "helmet_evidence",
            "helmet_trend_evidence",
            "bracelet_evidence",
            "movement_or_context_evidence",
            "status",
        ]
        if c in evidence.columns
    ]

    return evidence[cols].copy()


def save_tables(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(output_dir / f"{name}.csv", index=False)


def append_table(lines: list[str], title: str, df: pd.DataFrame, max_rows: int | None = None) -> None:
    lines.append(f"## {title}")
    lines.append("")

    if df.empty:
        lines.append("No data available.")
        lines.append("")
        return

    show_df = df.copy()

    if max_rows is not None:
        show_df = show_df.head(max_rows)

    lines.append(show_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")


def write_report(
    output_dir: Path,
    config: Config,
    tables: dict[str, pd.DataFrame],
    data: dict[str, pd.DataFrame | dict],
) -> None:
    lines = []

    lines.append("# Project hypotheses summary")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This report combines the current results for the main EEG/PM project hypotheses into one document."
    )
    lines.append("")
    lines.append("The laboratory work is handled in a separate project. This report only summarizes the main research branch.")
    lines.append("")

    lines.append("## Input result directories")
    lines.append("")
    lines.append(f"- Latent PM: `{config.latent_dir}`")
    lines.append(f"- Slow latent PM: `{config.slow_latent_dir}`")
    lines.append(f"- Slow latent cross-source: `{config.slow_latent_cross_source_dir}`")
    lines.append(f"- PM dynamics summary: `{config.dynamics_summary_dir}`")
    lines.append(f"- PM dynamics sensitivity: `{config.dynamics_sensitivity_dir}`")
    lines.append(f"- User calibration: `{config.user_calibration_dir}`")
    lines.append(f"- Device alignment: `{config.device_alignment_dir}`")
    lines.append(f"- WESAD: `{config.wesad_dir}`")
    lines.append("")

    append_table(lines, "Hypotheses status", tables["hypotheses"])
    append_table(lines, "Manual slow latent axis interpretation", tables["manual_latent_axes"])
    append_table(lines, "Slow latent cross-source transfer", tables["cross_source_highlight"])
    append_table(lines, "State dynamics best targets", tables["state_dynamics_best"])
    append_table(lines, "User calibration highlights", tables["calibration_highlight"], max_rows=30)
    append_table(lines, "Sensitivity: best parameter settings", tables["sensitivity_best_params"])
    append_table(lines, "Sensitivity: stability across parameters", tables["sensitivity_highlight"])
    append_table(lines, "Device alignment evidence", tables["device_alignment_highlight"])

    lines.append("## Main conclusions")
    lines.append("")
    lines.append("1. PM metrics should not be treated as independent absolute scales.")
    lines.append("2. Slow PM components form interpretable latent-state axes.")
    lines.append("3. Slow latent states show positive cross-source transfer between `Old_EEG` and `gpn_data`.")
    lines.append("4. PM dynamics should be separated into slow/background regression targets and trend/change-direction classification targets.")
    lines.append("5. User calibration improves both slow latent regression and trend classification.")
    lines.append("6. EEG helmets and wearable bracelets should be aligned through intermediate latent states, not by direct metric-name matching.")
    lines.append("7. Movement/ACC should be treated as a confounder and reliability/context marker.")
    lines.append("")

    lines.append("## Current limitations")
    lines.append("")
    lines.append("- Slow latent axes still require manual semantic interpretation.")
    lines.append("- Cross-source results should be checked for possible dataset/protocol structure effects.")
    lines.append("- Smart-watch datasets have not yet been integrated experimentally.")
    lines.append("- EEG-side reliability/artifact proxy has not yet been implemented.")
    lines.append("- Current models are mostly classical ML baselines; stronger DL models can be added later.")
    lines.append("")

    lines.append("## Recommended next steps")
    lines.append("")
    lines.append("1. Add EEG reliability/artifact proxy and evaluate whether unreliable windows degrade PM/latent predictions.")
    lines.append("2. Test reliability-aware training: filtering, sample weighting, or reliability as an additional feature.")
    lines.append("3. Prepare a short presentation for the supervisor using the three main hypotheses.")
    lines.append("4. If needed for publication direction, add domain adaptation for cross-source latent-state transfer.")
    lines.append("5. Keep laboratory EEG generation/DL tasks in the separate lab project.")
    lines.append("")

    (output_dir / "project_hypotheses_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # Short version for chat/supervisor.
    short_lines = []
    short_lines.append("# Short project summary")
    short_lines.append("")
    short_lines.append("## Main result")
    short_lines.append("")
    short_lines.append(
        "We confirmed a working framework for analyzing EEG/PM cognitive states through slow latent states, trend transitions, and user calibration."
    )
    short_lines.append("")
    short_lines.append("## Key points")
    short_lines.append("")
    short_lines.append("- PM metrics form interpretable latent axes: arousal, recovery-vs-focus, attention-vs-engagement, involvement.")
    short_lines.append("- Slow latent states transfer between `Old_EEG` and `gpn_data` with positive R² in both directions.")
    short_lines.append("- Slow/background states are better treated as regression targets.")
    short_lines.append("- Trend/change-direction states are better treated as classification targets.")
    short_lines.append("- 5–20% user calibration consistently improves quality.")
    short_lines.append("- Helmet and bracelet metrics should be aligned through latent states; ACC/movement is a confounder/reliability marker.")
    short_lines.append("")
    short_lines.append("## Next step")
    short_lines.append("")
    short_lines.append("Build EEG reliability/artifact proxy and test reliability-aware filtering or weighting.")
    short_lines.append("")

    (output_dir / "project_hypotheses_summary_short.md").write_text(
        "\n".join(short_lines),
        encoding="utf-8",
    )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Build a unified summary report for EEG/PM project hypotheses."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/project_summary"),
        help="Output directory.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="project_hypotheses_summary",
        help="Run name.",
    )
    parser.add_argument(
        "--latent-dir",
        type=Path,
        default=Path("reports/latent_states/pm_w10"),
        help="Directory with PM latent-state outputs.",
    )
    parser.add_argument(
        "--slow-latent-dir",
        type=Path,
        default=Path("reports/slow_latent_states/pm_w10"),
        help="Directory with slow latent-state outputs.",
    )
    parser.add_argument(
        "--slow-latent-cross-source-dir",
        type=Path,
        default=Path("reports/slow_latent_states/cross_source"),
        help="Directory with slow latent cross-source outputs.",
    )
    parser.add_argument(
        "--dynamics-summary-dir",
        type=Path,
        default=Path("reports/state_dynamics/pm_w10_experiment_summary_v4"),
        help="Directory with state dynamics experiment summary.",
    )
    parser.add_argument(
        "--dynamics-sensitivity-dir",
        type=Path,
        default=Path("reports/state_dynamics/sensitivity"),
        help="Directory with dynamics sensitivity outputs.",
    )
    parser.add_argument(
        "--user-calibration-dir",
        type=Path,
        default=Path("reports/user_calibration/pm_w10"),
        help="Directory with user calibration outputs.",
    )
    parser.add_argument(
        "--device-alignment-dir",
        type=Path,
        default=Path("reports/device_metric_alignment"),
        help="Directory with device alignment outputs.",
    )
    parser.add_argument(
        "--wesad-dir",
        type=Path,
        default=Path("reports/wearable_pm_alignment"),
        help="Directory with WESAD / wearable outputs.",
    )

    args = parser.parse_args()

    return Config(
        output_dir=args.output_dir,
        run_name=args.run_name,
        latent_dir=args.latent_dir,
        slow_latent_dir=args.slow_latent_dir,
        slow_latent_cross_source_dir=args.slow_latent_cross_source_dir,
        dynamics_summary_dir=args.dynamics_summary_dir,
        dynamics_sensitivity_dir=args.dynamics_sensitivity_dir,
        user_calibration_dir=args.user_calibration_dir,
        device_alignment_dir=args.device_alignment_dir,
        wesad_dir=args.wesad_dir,
    )


def main() -> None:
    logger = setup_logging()
    config = parse_args()

    config.output_dir = config.output_dir.resolve()
    config.latent_dir = config.latent_dir.resolve()
    config.slow_latent_dir = config.slow_latent_dir.resolve()
    config.slow_latent_cross_source_dir = config.slow_latent_cross_source_dir.resolve()
    config.dynamics_summary_dir = config.dynamics_summary_dir.resolve()
    config.dynamics_sensitivity_dir = config.dynamics_sensitivity_dir.resolve()
    config.user_calibration_dir = config.user_calibration_dir.resolve()
    config.device_alignment_dir = config.device_alignment_dir.resolve()
    config.wesad_dir = config.wesad_dir.resolve()

    make_output_dirs(config)

    logger.info("=" * 80)
    logger.info("Build project hypotheses summary")
    logger.info("=" * 80)
    logger.info("Output dir: %s", config.output_dir)

    data = load_inputs(config)

    tables = {
        "hypotheses": build_hypothesis_table(data),
        "manual_latent_axes": manual_latent_axis_table(),
        "cross_source_highlight": build_cross_source_highlight(data),
        "state_dynamics_best": build_best_state_dynamics_table(data),
        "calibration_highlight": build_calibration_highlight(data),
        "sensitivity_best_params": build_best_sensitivity_params(data),
        "sensitivity_highlight": build_sensitivity_highlight(data),
        "device_alignment_highlight": build_device_alignment_highlight(data),
    }

    save_tables(config.output_dir, tables)

    metadata = {
        "run_name": config.run_name,
        "output_dir": str(config.output_dir),
        "input_dirs": {
            "latent_dir": str(config.latent_dir),
            "slow_latent_dir": str(config.slow_latent_dir),
            "slow_latent_cross_source_dir": str(config.slow_latent_cross_source_dir),
            "dynamics_summary_dir": str(config.dynamics_summary_dir),
            "dynamics_sensitivity_dir": str(config.dynamics_sensitivity_dir),
            "user_calibration_dir": str(config.user_calibration_dir),
            "device_alignment_dir": str(config.device_alignment_dir),
            "wesad_dir": str(config.wesad_dir),
        },
        "tables": {
            name: int(len(df)) if isinstance(df, pd.DataFrame) else 0
            for name, df in tables.items()
        },
        "inputs_found": {
            name: (not value.empty if isinstance(value, pd.DataFrame) else bool(value))
            for name, value in data.items()
        },
    }

    (config.output_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_report(
        output_dir=config.output_dir,
        config=config,
        tables=tables,
        data=data,
    )

    logger.info("=" * 80)
    logger.info("Saved project hypotheses summary")
    logger.info("=" * 80)
    logger.info("Full report: %s", config.output_dir / "project_hypotheses_summary.md")
    logger.info("Short report: %s", config.output_dir / "project_hypotheses_summary_short.md")
    logger.info("Metadata: %s", config.output_dir / "summary.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()