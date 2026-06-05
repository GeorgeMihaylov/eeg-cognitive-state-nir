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

    latent_report_dir: Path
    slow_latent_report_dir: Path
    dynamics_summary_dir: Path
    user_calibration_dir: Path
    wesad_summary_dir: Path

    run_name: str


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("device_metric_alignment")


def safe_read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_best_regression_rows(path: Path) -> pd.DataFrame:
    df = safe_read_csv(path)
    if df.empty:
        return df

    cols = [
        c
        for c in [
            "target",
            "best_random_regression_mode",
            "best_random_r2",
            "best_random_spearman",
            "best_subject_regression_mode",
            "best_subject_r2",
            "best_subject_spearman",
            "interpretation",
        ]
        if c in df.columns
    ]

    return df[cols].copy()


def get_best_classification_rows(path: Path) -> pd.DataFrame:
    df = safe_read_csv(path)
    if df.empty:
        return df

    cols = [
        c
        for c in [
            "target",
            "best_random_classification_mode",
            "best_random_balanced_accuracy",
            "best_random_macro_f1",
            "best_subject_classification_mode",
            "best_subject_balanced_accuracy",
            "best_subject_macro_f1",
            "interpretation",
        ]
        if c in df.columns
    ]

    return df[cols].copy()


def get_calibration_gain(path: Path) -> pd.DataFrame:
    df = safe_read_csv(path)
    if df.empty:
        return df

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
        if c in df.columns
    ]

    return df[cols].copy()


def build_device_metric_mapping() -> pd.DataFrame:
    rows = [
        {
            "device_family": "EEG helmet",
            "dataset": "gpn_data + Old_EEG",
            "signals": "EEG / EEG-derived POW features",
            "raw_metrics": "PM.Stress, PM.Excitement",
            "derived_targets": "stress_slow, excitement_slow, slow_pca_1",
            "mapped_latent_state": "Stress / Arousal / General activation",
            "task_type": "regression / classification",
            "current_evidence": "slow/background regression and trend classification are supported",
            "main_limitation": "subject variability; user calibration improves quality",
        },
        {
            "device_family": "EEG helmet",
            "dataset": "gpn_data + Old_EEG",
            "signals": "EEG / EEG-derived POW features",
            "raw_metrics": "PM.Focus, PM.Attention",
            "derived_targets": "focus_slow, attention_slow, slow_pca_2, slow_pca_3",
            "mapped_latent_state": "Workload / Attention / Cognitive control",
            "task_type": "regression / classification",
            "current_evidence": "focus_trend and attention_trend are usable; slow_pca_2 is one of the most stable latent targets",
            "main_limitation": "attention/focus differ across users; calibration is important",
        },
        {
            "device_family": "EEG helmet",
            "dataset": "gpn_data + Old_EEG",
            "signals": "EEG / EEG-derived POW features",
            "raw_metrics": "PM.Relaxation",
            "derived_targets": "relaxation_slow, slow_pca_2",
            "mapped_latent_state": "Recovery / Fatigue / Relaxation",
            "task_type": "regression",
            "current_evidence": "slow_pm_relaxation is one of the best zero-shot individual slow PM targets",
            "main_limitation": "semantic interpretation depends on PM metric definition",
        },
        {
            "device_family": "EEG helmet",
            "dataset": "gpn_data + Old_EEG",
            "signals": "EEG / EEG-derived POW features",
            "raw_metrics": "PM.Engagement, PM.Interest",
            "derived_targets": "engagement_slow, interest_trend, slow_pca_4",
            "mapped_latent_state": "Engagement / Involvement",
            "task_type": "regression / classification",
            "current_evidence": "interest_trend is one of the stronger trend targets",
            "main_limitation": "attention and engagement are not equivalent and should not be merged directly",
        },
        {
            "device_family": "Wearable bracelet",
            "dataset": "WESAD",
            "signals": "BVP/PPG, EDA, TEMP",
            "raw_metrics": "stress / non-stress label",
            "derived_targets": "physiology-based stress proxy",
            "mapped_latent_state": "Stress / Arousal",
            "task_type": "classification",
            "current_evidence": "WESAD baseline gives high stress/non-stress performance",
            "main_limitation": "stress proxy is not equivalent to PM.Stress",
        },
        {
            "device_family": "Wearable bracelet",
            "dataset": "WESAD",
            "signals": "ACC",
            "raw_metrics": "movement signal",
            "derived_targets": "ACC-only baseline, movement context",
            "mapped_latent_state": "Movement / Context / Reliability",
            "task_type": "classification / confounder analysis",
            "current_evidence": "ACC-only is strong, indicating motion/protocol confounding",
            "main_limitation": "movement can inflate stress classification quality",
        },
        {
            "device_family": "Smart-watch-like wearables",
            "dataset": "Not processed yet",
            "signals": "HR, HRV, PPG, ACC, sleep, activity",
            "raw_metrics": "device-dependent",
            "derived_targets": "future wearable state proxies",
            "mapped_latent_state": "Stress / Recovery / Activity context",
            "task_type": "not implemented",
            "current_evidence": "conceptual mapping only",
            "main_limitation": "no experimental results in current baseline",
        },
    ]

    return pd.DataFrame(rows)


def build_latent_state_evidence_matrix(
    regression_recs: pd.DataFrame,
    classification_recs: pd.DataFrame,
    calibration_gain: pd.DataFrame,
) -> pd.DataFrame:
    def get_regression_evidence(targets: list[str]) -> str:
        if regression_recs.empty:
            return "not available"

        sub = regression_recs[regression_recs["target"].isin(targets)].copy()
        if sub.empty:
            return "not available"

        parts = []
        for _, row in sub.iterrows():
            target = row.get("target", "")
            mode = row.get("best_subject_regression_mode", "")
            r2 = row.get("best_subject_r2", np.nan)
            if pd.notna(r2):
                parts.append(f"{target}: {mode}, subject R²={float(r2):.3f}")
            else:
                parts.append(f"{target}: {mode}")
        return "; ".join(parts)

    def get_classification_evidence(targets: list[str]) -> str:
        if classification_recs.empty:
            return "not available"

        normalized_targets = set()
        for t in targets:
            normalized_targets.add(t)
            if t.endswith("_trend"):
                normalized_targets.add(t.replace("_trend", ""))
            else:
                normalized_targets.add(f"{t}_trend")

        sub = classification_recs[
            classification_recs["target"].astype(str).isin(normalized_targets)
        ].copy()

        if sub.empty:
            return "not available"

        parts = []
        for _, row in sub.iterrows():
            target = row.get("target", "")
            mode = row.get("best_subject_classification_mode", "")
            ba = row.get("best_subject_balanced_accuracy", np.nan)
            if pd.notna(ba):
                parts.append(f"{target}: {mode}, subject BA={float(ba):.3f}")
            else:
                parts.append(f"{target}: {mode}")

        return "; ".join(parts)

    def get_calibration_evidence(targets: list[str]) -> str:
        if calibration_gain.empty:
            return "not available"

        sub = calibration_gain[
            calibration_gain["target"].isin(targets)
            & calibration_gain["mode"].isin(["calibration_20pct", "subject_dependent"])
        ].copy()

        if sub.empty:
            return "not available"

        parts = []
        for _, row in sub.iterrows():
            target = row.get("target", "")
            mode = row.get("mode", "")
            gain = row.get("absolute_gain_vs_zero_shot", np.nan)
            if pd.notna(gain):
                parts.append(f"{target} {mode}: gain={float(gain):.3f}")
        return "; ".join(parts)

    rows = [
        {
            "latent_state": "Stress / Arousal / General activation",
            "helmet_evidence": get_regression_evidence(["stress", "excitement", "slow_pca_1", "slow_pm_stress", "slow_pm_excitement"]),
            "helmet_trend_evidence": get_classification_evidence(["stress_trend"]),
            "bracelet_evidence": "WESAD BVP/EDA/TEMP stress/non-stress baseline; high performance reported in wearable baseline",
            "movement_or_context_evidence": "WESAD ACC-only is strong, so movement/protocol can confound stress recognition",
            "calibration_evidence": get_calibration_evidence(["slow_pca_1", "slow_pm_stress", "slow_pm_excitement", "stress_trend"]),
            "status": "supported, but motion and user effects must be controlled",
        },
        {
            "latent_state": "Workload / Attention / Cognitive control",
            "helmet_evidence": get_regression_evidence(["attention", "focus", "slow_pca_2", "slow_pm_focus"]),
            "helmet_trend_evidence": get_classification_evidence(["focus_trend", "attention_trend"]),
            "bracelet_evidence": "no direct WESAD analogue; possible indirect wearable proxy through arousal/activity",
            "movement_or_context_evidence": "movement can alter apparent cognitive state but does not directly define workload",
            "calibration_evidence": get_calibration_evidence(["slow_pca_2", "slow_pm_focus", "focus_trend", "attention_trend"]),
            "status": "supported mainly by EEG/PM; wearable mapping is indirect",
        },
        {
            "latent_state": "Recovery / Fatigue / Relaxation",
            "helmet_evidence": get_regression_evidence(["relaxation", "slow_pm_relaxation", "slow_pca_2"]),
            "helmet_trend_evidence": get_classification_evidence(["relaxation_trend"]),
            "bracelet_evidence": "future smartwatch/HRV/sleep datasets could support this axis",
            "movement_or_context_evidence": "low activity may correlate with recovery but can also indicate protocol state",
            "calibration_evidence": get_calibration_evidence(["slow_pm_relaxation", "slow_pca_2"]),
            "status": "supported by EEG/PM; wearable evidence not processed yet",
        },
        {
            "latent_state": "Engagement / Involvement",
            "helmet_evidence": get_regression_evidence(["engagement", "interest", "slow_pca_4"]),
            "helmet_trend_evidence": get_classification_evidence(["interest_trend"]),
            "bracelet_evidence": "no direct WESAD analogue",
            "movement_or_context_evidence": "activity/protocol can influence involvement proxies",
            "calibration_evidence": get_calibration_evidence(["slow_pca_4", "interest_trend"]),
            "status": "supported by EEG/PM; wearable mapping is weak",
        },
        {
            "latent_state": "Movement / Context / Reliability",
            "helmet_evidence": "EEG artifact/reliability proxy not implemented yet",
            "helmet_trend_evidence": "not applicable",
            "bracelet_evidence": "WESAD ACC-only baseline is strong",
            "movement_or_context_evidence": "movement should be treated as context and reliability marker, not only as a predictive feature",
            "calibration_evidence": "not applicable",
            "status": "partially supported by WESAD; EEG-side reliability analysis is still missing",
        },
    ]

    return pd.DataFrame(rows)


def build_comparison_table(
    regression_recs: pd.DataFrame,
    classification_recs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    if not regression_recs.empty:
        for _, row in regression_recs.iterrows():
            rows.append(
                {
                    "state_source": "EEG helmet",
                    "dataset": "gpn_data + Old_EEG",
                    "target": row.get("target"),
                    "target_type": "slow/background regression",
                    "best_mode": row.get("best_subject_regression_mode"),
                    "metric": "subject-wise R²",
                    "score": row.get("best_subject_r2"),
                    "interpretation": row.get("interpretation"),
                }
            )

    if not classification_recs.empty:
        for _, row in classification_recs.iterrows():
            rows.append(
                {
                    "state_source": "EEG helmet",
                    "dataset": "gpn_data + Old_EEG",
                    "target": row.get("target"),
                    "target_type": "trend/change-direction classification",
                    "best_mode": row.get("best_subject_classification_mode"),
                    "metric": "subject-wise balanced accuracy",
                    "score": row.get("best_subject_balanced_accuracy"),
                    "interpretation": row.get("interpretation"),
                }
            )

    rows.extend(
        [
            {
                "state_source": "Wearable bracelet",
                "dataset": "WESAD",
                "target": "stress / non-stress",
                "target_type": "wearable physiology classification",
                "best_mode": "BVP/EDA/TEMP/ACC baseline",
                "metric": "balanced accuracy",
                "score": np.nan,
                "interpretation": "wearable physiology provides stress/arousal proxy; exact score should be read from WESAD report",
            },
            {
                "state_source": "Wearable bracelet",
                "dataset": "WESAD",
                "target": "ACC-only",
                "target_type": "movement/protocol confounding check",
                "best_mode": "ACC-only baseline",
                "metric": "balanced accuracy",
                "score": np.nan,
                "interpretation": "high ACC-only quality indicates movement/protocol confounding risk",
            },
        ]
    )

    return pd.DataFrame(rows)


def write_markdown_table(lines: list[str], title: str, df: pd.DataFrame) -> None:
    lines.append(f"## {title}")
    lines.append("")

    if df.empty:
        lines.append("No data available.")
    else:
        lines.append(df.to_markdown(index=False, floatfmt=".4f"))

    lines.append("")


def write_report(
    output_dir: Path,
    config: Config,
    device_mapping: pd.DataFrame,
    evidence_matrix: pd.DataFrame,
    comparison_table: pd.DataFrame,
    regression_recs: pd.DataFrame,
    classification_recs: pd.DataFrame,
    calibration_gain: pd.DataFrame,
    metadata: dict,
) -> None:
    lines: list[str] = []

    lines.append("# Device metric alignment report")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append(
        "This report summarizes how EEG helmet metrics and wearable bracelet metrics can be aligned through intermediate latent states rather than direct label-level matching."
    )
    lines.append("")
    lines.append("Main principle:")
    lines.append("")
    lines.append("```text")
    lines.append("device → signal → features → target representation → latent state → interpretation limits")
    lines.append("```")
    lines.append("")

    lines.append("## Input result directories")
    lines.append("")
    lines.append(f"- Latent PM: `{config.latent_report_dir}`")
    lines.append(f"- Slow latent PM: `{config.slow_latent_report_dir}`")
    lines.append(f"- PM dynamics summary: `{config.dynamics_summary_dir}`")
    lines.append(f"- User calibration: `{config.user_calibration_dir}`")
    lines.append(f"- WESAD summary: `{config.wesad_summary_dir}`")
    lines.append("")

    write_markdown_table(lines, "Device metric mapping", device_mapping)
    write_markdown_table(lines, "Latent state evidence matrix", evidence_matrix)
    write_markdown_table(lines, "Helmet vs bracelet comparison table", comparison_table)

    lines.append("## Main conclusions")
    lines.append("")
    lines.append("1. EEG helmet and bracelet metrics should not be matched directly by metric names.")
    lines.append("2. EEG/PM data provide richer evidence for cognitive state axes such as arousal, recovery, attention/control, and involvement.")
    lines.append("3. WESAD-like bracelet data provide a useful stress/arousal proxy, but motion/protocol confounding must be controlled.")
    lines.append("4. Slow PM components are better interpreted as background-state regression targets.")
    lines.append("5. Trend PM targets are better interpreted as direction-of-change classification targets.")
    lines.append("6. User calibration improves both slow latent regression and trend classification.")
    lines.append("")

    lines.append("## Current limitations")
    lines.append("")
    lines.append("- Smart-watch datasets are not yet experimentally integrated.")
    lines.append("- EEG-side artifact/reliability score is not yet implemented.")
    lines.append("- WESAD stress is not equivalent to PM.Stress; it should be treated as a stress/arousal proxy.")
    lines.append("- ACC/movement must be treated as both feature and confounder.")
    lines.append("")

    lines.append("## Recommended next steps")
    lines.append("")
    lines.append("1. Add EEG artifact/reliability proxy for helmet data.")
    lines.append("2. Add real vs synthetic EEG generation block for the laboratory requirements.")
    lines.append("3. Add at least one deep learning model, for example CNN/CNN-LSTM for trend classification.")
    lines.append("4. Extend wearable alignment with smartwatch-like datasets if time permits.")
    lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    comparison_path = output_dir / "helmet_vs_bracelet_comparison.md"
    comparison_lines: list[str] = []
    comparison_lines.append("# Helmet vs bracelet comparison")
    comparison_lines.append("")
    comparison_lines.append("## Short summary")
    comparison_lines.append("")
    comparison_lines.append(
        "EEG helmet data and bracelet data are comparable only through intermediate states, not through direct metric equality."
    )
    comparison_lines.append("")
    write_markdown_table(comparison_lines, "Comparison table", comparison_table)
    write_markdown_table(comparison_lines, "Evidence matrix", evidence_matrix)
    comparison_path.write_text("\n".join(comparison_lines), encoding="utf-8")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Build device metric alignment report for EEG helmets and wearable bracelets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/device_metric_alignment"),
        help="Output directory.",
    )
    parser.add_argument(
        "--latent-report-dir",
        type=Path,
        default=Path("reports/latent_states/pm_w10"),
        help="Directory with PM latent state outputs.",
    )
    parser.add_argument(
        "--slow-latent-report-dir",
        type=Path,
        default=Path("reports/slow_latent_states/pm_w10"),
        help="Directory with slow latent state outputs.",
    )
    parser.add_argument(
        "--dynamics-summary-dir",
        type=Path,
        default=Path("reports/state_dynamics/pm_w10_experiment_summary_v4"),
        help="Directory with PM dynamics summary outputs.",
    )
    parser.add_argument(
        "--user-calibration-dir",
        type=Path,
        default=Path("reports/user_calibration/pm_w10"),
        help="Directory with user calibration outputs.",
    )
    parser.add_argument(
        "--wesad-summary-dir",
        type=Path,
        default=Path("reports/wearable_pm_alignment"),
        help="Directory with WESAD outputs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="device_metric_alignment",
        help="Run name.",
    )

    args = parser.parse_args()

    return Config(
        output_dir=args.output_dir,
        latent_report_dir=args.latent_report_dir,
        slow_latent_report_dir=args.slow_latent_report_dir,
        dynamics_summary_dir=args.dynamics_summary_dir,
        user_calibration_dir=args.user_calibration_dir,
        wesad_summary_dir=args.wesad_summary_dir,
        run_name=args.run_name,
    )


def main() -> None:
    logger = setup_logging()
    config = parse_args()

    config.output_dir = config.output_dir.resolve()
    config.latent_report_dir = config.latent_report_dir.resolve()
    config.slow_latent_report_dir = config.slow_latent_report_dir.resolve()
    config.dynamics_summary_dir = config.dynamics_summary_dir.resolve()
    config.user_calibration_dir = config.user_calibration_dir.resolve()
    config.wesad_summary_dir = config.wesad_summary_dir.resolve()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Build device metric alignment report")
    logger.info("=" * 80)
    logger.info("Output dir: %s", config.output_dir)

    regression_recs = get_best_regression_rows(
        config.dynamics_summary_dir / "target_recommendations_regression.csv"
    )
    classification_recs = get_best_classification_rows(
        config.dynamics_summary_dir / "target_recommendations_classification.csv"
    )
    calibration_gain = get_calibration_gain(
        config.user_calibration_dir / "calibration_gain_vs_zero_shot.csv"
    )

    device_mapping = build_device_metric_mapping()
    evidence_matrix = build_latent_state_evidence_matrix(
        regression_recs=regression_recs,
        classification_recs=classification_recs,
        calibration_gain=calibration_gain,
    )
    comparison_table = build_comparison_table(
        regression_recs=regression_recs,
        classification_recs=classification_recs,
    )

    device_mapping.to_csv(config.output_dir / "device_metric_mapping.csv", index=False)
    evidence_matrix.to_csv(config.output_dir / "latent_state_evidence_matrix.csv", index=False)
    comparison_table.to_csv(config.output_dir / "helmet_vs_bracelet_comparison.csv", index=False)

    metadata = {
        "run_name": config.run_name,
        "output_dir": str(config.output_dir),
        "latent_report_dir": str(config.latent_report_dir),
        "slow_latent_report_dir": str(config.slow_latent_report_dir),
        "dynamics_summary_dir": str(config.dynamics_summary_dir),
        "user_calibration_dir": str(config.user_calibration_dir),
        "wesad_summary_dir": str(config.wesad_summary_dir),
        "n_device_mapping_rows": int(len(device_mapping)),
        "n_evidence_matrix_rows": int(len(evidence_matrix)),
        "n_comparison_rows": int(len(comparison_table)),
        "inputs_found": {
            "regression_recommendations": not regression_recs.empty,
            "classification_recommendations": not classification_recs.empty,
            "calibration_gain": not calibration_gain.empty,
        },
    }

    (config.output_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_report(
        output_dir=config.output_dir,
        config=config,
        device_mapping=device_mapping,
        evidence_matrix=evidence_matrix,
        comparison_table=comparison_table,
        regression_recs=regression_recs,
        classification_recs=classification_recs,
        calibration_gain=calibration_gain,
        metadata=metadata,
    )

    logger.info("=" * 80)
    logger.info("Saved device metric alignment outputs")
    logger.info("=" * 80)
    logger.info("Device mapping: %s", config.output_dir / "device_metric_mapping.csv")
    logger.info("Evidence matrix: %s", config.output_dir / "latent_state_evidence_matrix.csv")
    logger.info("Comparison CSV: %s", config.output_dir / "helmet_vs_bracelet_comparison.csv")
    logger.info("Report: %s", config.output_dir / "report.md")
    logger.info("Markdown comparison: %s", config.output_dir / "helmet_vs_bracelet_comparison.md")
    logger.info("Done.")


if __name__ == "__main__":
    main()