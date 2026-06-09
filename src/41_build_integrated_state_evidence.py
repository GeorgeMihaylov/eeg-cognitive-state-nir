from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Config:
    output_dir: Path
    run_name: str

    project_summary_dir: Path
    slow_latent_cross_source_dir: Path
    dynamics_summary_dir: Path
    dynamics_sensitivity_dir: Path
    user_calibration_dir: Path
    device_alignment_dir: Path
    wesad_dir: Path

    transformer_summary_dir: Path
    transformer_cross_source_dir: Path

    temporal_modeling_note: str | None


STATE_REGISTRY = {
    "Stress / Arousal / General activation": {
        "latent_axis": "slow_pca_1",
        "latent_interpretation": "Slow Stress / Arousal / General activation",
        "slow_targets": ["excitement", "stress"],
        "trend_targets": ["stress", "excitement"],
        "calibration_targets": ["slow_pca_1", "slow_pm_stress", "slow_pm_excitement", "stress_trend"],
        "external_proxy": "WESAD BVP/EDA/TEMP stress proxy; ACC as movement confounder",
        "temporal_relevance": "high",
        "temporal_comment": "DL lab indicates Excitement is one of the best PM targets; TransformerEncoder is the preferred model family.",
        "recommended_next_step": "Train Transformer-based temporal model for arousal trajectory: EEG/POW sequence -> slow_pca_1 / excitement_slow / stress_trend.",
    },
    "Recovery / Fatigue / Relaxation": {
        "latent_axis": "slow_pca_2",
        "latent_interpretation": "Slow Recovery vs Focus",
        "slow_targets": ["relaxation", "focus"],
        "trend_targets": ["relaxation", "focus"],
        "calibration_targets": ["slow_pca_2", "slow_pm_relaxation", "slow_pm_focus", "focus_trend"],
        "external_proxy": "No direct WESAD analogue; possible future proxy via HRV/sleep/recovery datasets.",
        "temporal_relevance": "high",
        "temporal_comment": "DL lab indicates Relaxation is one of the best PM targets; this supports temporal modeling of recovery trajectories.",
        "recommended_next_step": "Train Transformer-based temporal model for recovery/focus trajectory: EEG/POW sequence -> slow_pca_2 / relaxation_slow.",
    },
    "Workload / Attention / Cognitive control": {
        "latent_axis": "slow_pca_3",
        "latent_interpretation": "Slow Attention vs Engagement",
        "slow_targets": ["attention", "focus"],
        "trend_targets": ["attention", "focus"],
        "calibration_targets": ["attention_trend", "focus_trend"],
        "external_proxy": "No direct WESAD analogue; wearable mapping is indirect through arousal/activity.",
        "temporal_relevance": "medium",
        "temporal_comment": "Temporal modeling may help, but evidence is weaker than for Excitement and Relaxation.",
        "recommended_next_step": "Use this state as secondary target in latent trajectory model; evaluate cross-subject stability.",
    },
    "Engagement / Involvement": {
        "latent_axis": "slow_pca_4",
        "latent_interpretation": "Slow Cognitive involvement",
        "slow_targets": ["engagement", "interest"],
        "trend_targets": ["interest", "engagement"],
        "calibration_targets": ["slow_pca_4", "interest_trend"],
        "external_proxy": "No direct WESAD analogue.",
        "temporal_relevance": "medium",
        "temporal_comment": "Transformer-based temporal modeling is relevant, but external validation is weak.",
        "recommended_next_step": "Use slow_pca_4 and interest_trend as involvement trajectory targets.",
    },
    "Movement / Context / Reliability": {
        "latent_axis": None,
        "latent_interpretation": "No PM latent axis; this is a reliability/context dimension.",
        "slow_targets": [],
        "trend_targets": [],
        "calibration_targets": [],
        "external_proxy": "WESAD ACC; anomaly detection and EEG artifact proxy.",
        "temporal_relevance": "supporting",
        "temporal_comment": "DL lab suggests anomaly detection is useful as quality control, not as primary target.",
        "recommended_next_step": "Build EEG reliability/artifact proxy and use it for filtering, sample weighting, or additional model input.",
    },
}


TEMPORAL_MODELING_EVIDENCE = [
    {
        "model_family": "MLP",
        "evidence": "Weak baseline",
        "interpretation": "Single-window modeling is insufficient for the main NIR direction.",
        "priority": "low",
    },
    {
        "model_family": "LSTM",
        "evidence": "Partial improvement",
        "interpretation": "Sequential modeling helps, but results are not consistently best.",
        "priority": "medium",
    },
    {
        "model_family": "TransformerEncoder",
        "evidence": "Best overall model in the DL lab summary and strong cross-source latent trajectory transfer in the main project",
        "interpretation": "The main direction should be Transformer-based temporal modeling of EEG/POW sequences.",
        "priority": "high",
    },
    {
        "model_family": "Autoencoder / synthetic augmentation",
        "evidence": "Feature-space generation works but gives no stable gain yet",
        "interpretation": "Keep as auxiliary component, not as the main research line.",
        "priority": "supporting",
    },
    {
        "model_family": "Anomaly detection",
        "evidence": "Useful for unstable window filtering",
        "interpretation": "Use as data-quality and reliability-control block.",
        "priority": "supporting",
    },
]


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("integrated_state_evidence")


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def fmt_float(value: Any, digits: int = 3) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def to_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def normalize_target_name(value: str | None) -> str:
    if value is None:
        return ""

    value = str(value)
    value = value.replace("pm_", "")
    value = value.replace("_slow", "")
    value = value.replace("_trend_next", "")
    value = value.replace("_trend", "")
    value = value.replace("slow_pm_", "")
    return value.strip().lower()


def load_inputs(config: Config) -> dict[str, Any]:
    return {
        "hypotheses": safe_read_csv(config.project_summary_dir / "hypotheses.csv"),
        "manual_latent_axes": safe_read_csv(config.project_summary_dir / "manual_latent_axes.csv"),
        "cross_source_highlight": safe_read_csv(config.slow_latent_cross_source_dir / "summary.csv"),
        "cross_source_r2_pivot": safe_read_csv(config.slow_latent_cross_source_dir / "r2_pivot.csv"),
        "cross_source_spearman_pivot": safe_read_csv(config.slow_latent_cross_source_dir / "spearman_pivot.csv"),
        "state_dynamics_best": safe_read_csv(config.project_summary_dir / "state_dynamics_best.csv"),
        "dynamics_regression_recs": safe_read_csv(config.dynamics_summary_dir / "target_recommendations_regression.csv"),
        "dynamics_classification_recs": safe_read_csv(config.dynamics_summary_dir / "target_recommendations_classification.csv"),
        "calibration_highlight": safe_read_csv(config.project_summary_dir / "calibration_highlight.csv"),
        "calibration_gain": safe_read_csv(config.user_calibration_dir / "calibration_gain_vs_zero_shot.csv"),
        "sensitivity_best_params": safe_read_csv(config.dynamics_sensitivity_dir / "sensitivity_best_params.csv"),
        "sensitivity_stability": safe_read_csv(config.dynamics_sensitivity_dir / "sensitivity_stability.csv"),
        "device_alignment": safe_read_csv(config.device_alignment_dir / "latent_state_evidence_matrix.csv"),
        "helmet_vs_bracelet": safe_read_csv(config.device_alignment_dir / "helmet_vs_bracelet_comparison.csv"),
        "device_mapping": safe_read_csv(config.device_alignment_dir / "device_metric_mapping.csv"),
        "device_summary": safe_read_json(config.device_alignment_dir / "summary.json"),
        "wesad_summary": safe_read_json(config.wesad_dir / "summary.json"),
        "transformer_summary": safe_read_csv(config.transformer_summary_dir / "summary.csv"),
        "transformer_cross_source_summary": safe_read_csv(config.transformer_cross_source_dir / "summary.csv"),
    }


def extract_cross_source_for_axis(axis: str | None, data: dict[str, Any]) -> dict[str, Any]:
    if axis is None:
        return {
            "cross_source_available": False,
            "cross_source_best_r2": np.nan,
            "cross_source_mean_r2": np.nan,
            "cross_source_best_spearman": np.nan,
            "cross_source_mean_spearman": np.nan,
            "cross_source_summary": "not applicable",
        }

    df = data.get("cross_source_highlight", pd.DataFrame())

    if not isinstance(df, pd.DataFrame) or df.empty or "target" not in df.columns:
        return {
            "cross_source_available": False,
            "cross_source_best_r2": np.nan,
            "cross_source_mean_r2": np.nan,
            "cross_source_best_spearman": np.nan,
            "cross_source_mean_spearman": np.nan,
            "cross_source_summary": "missing",
        }

    sub = df[df["target"].astype(str) == axis].copy()

    if sub.empty:
        return {
            "cross_source_available": False,
            "cross_source_best_r2": np.nan,
            "cross_source_mean_r2": np.nan,
            "cross_source_best_spearman": np.nan,
            "cross_source_mean_spearman": np.nan,
            "cross_source_summary": "missing",
        }

    r2 = pd.to_numeric(sub.get("r2_mean"), errors="coerce")
    sp = pd.to_numeric(sub.get("spearman_mean"), errors="coerce")

    directions = []
    if "direction" in sub.columns:
        for _, row in sub.iterrows():
            directions.append(
                f"{row.get('direction')}: R2={fmt_float(row.get('r2_mean'))}, "
                f"Spearman={fmt_float(row.get('spearman_mean'))}"
            )

    return {
        "cross_source_available": True,
        "cross_source_best_r2": float(r2.max()),
        "cross_source_mean_r2": float(r2.mean()),
        "cross_source_best_spearman": float(sp.max()),
        "cross_source_mean_spearman": float(sp.mean()),
        "cross_source_summary": "; ".join(directions),
    }


def extract_transformer_evidence(axis: str | None, data: dict[str, Any]) -> dict[str, Any]:
    if axis is None:
        return {
            "transformer_groupkfold_available": False,
            "transformer_groupkfold_r2": np.nan,
            "transformer_groupkfold_spearman": np.nan,
            "transformer_cross_source_available": False,
            "transformer_cross_source_r2": np.nan,
            "transformer_cross_source_spearman": np.nan,
            "transformer_cross_source_summary": "not applicable",
        }

    group_df = data.get("transformer_summary", pd.DataFrame())
    cross_df = data.get("transformer_cross_source_summary", pd.DataFrame())

    group_r2 = np.nan
    group_spearman = np.nan
    group_available = False

    if isinstance(group_df, pd.DataFrame) and not group_df.empty:
        required_cols = {"model", "validation", "target", "r2_mean", "spearman_mean"}
        if required_cols.issubset(set(group_df.columns)):
            sub = group_df[
                (group_df["model"].astype(str) == "transformer")
                & (group_df["validation"].astype(str) == "groupkfold_subject")
                & (group_df["target"].astype(str) == axis)
            ].copy()

            if not sub.empty:
                group_available = True
                group_r2 = pd.to_numeric(sub["r2_mean"], errors="coerce").mean()
                group_spearman = pd.to_numeric(sub["spearman_mean"], errors="coerce").mean()

    cross_r2 = np.nan
    cross_spearman = np.nan
    cross_available = False
    cross_summary = "missing"

    if isinstance(cross_df, pd.DataFrame) and not cross_df.empty:
        required_cols = {"model", "validation", "target", "r2_mean", "spearman_mean"}
        if required_cols.issubset(set(cross_df.columns)):
            sub = cross_df[
                (cross_df["model"].astype(str) == "transformer")
                & (cross_df["validation"].astype(str) == "cross_source")
                & (cross_df["target"].astype(str) == axis)
            ].copy()

            if not sub.empty:
                cross_available = True
                cross_r2 = pd.to_numeric(sub["r2_mean"], errors="coerce").mean()
                cross_spearman = pd.to_numeric(sub["spearman_mean"], errors="coerce").mean()
                cross_summary = (
                    f"Transformer cross-source: R2={fmt_float(cross_r2)}, "
                    f"Spearman={fmt_float(cross_spearman)}"
                )

    return {
        "transformer_groupkfold_available": group_available,
        "transformer_groupkfold_r2": float(group_r2) if not pd.isna(group_r2) else np.nan,
        "transformer_groupkfold_spearman": float(group_spearman) if not pd.isna(group_spearman) else np.nan,
        "transformer_cross_source_available": cross_available,
        "transformer_cross_source_r2": float(cross_r2) if not pd.isna(cross_r2) else np.nan,
        "transformer_cross_source_spearman": float(cross_spearman) if not pd.isna(cross_spearman) else np.nan,
        "transformer_cross_source_summary": cross_summary,
    }


def extract_slow_regression_evidence(state_cfg: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    df = data.get("state_dynamics_best", pd.DataFrame())

    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "slow_regression_available": False,
            "slow_regression_best_target": "missing",
            "slow_regression_best_score": np.nan,
            "slow_regression_summary": "missing",
        }

    targets = {normalize_target_name(t) for t in state_cfg.get("slow_targets", [])}

    if "target" not in df.columns or "task_family" not in df.columns:
        return {
            "slow_regression_available": False,
            "slow_regression_best_target": "missing",
            "slow_regression_best_score": np.nan,
            "slow_regression_summary": "missing",
        }

    sub = df[df["task_family"].astype(str).str.contains("slow/background", case=False, na=False)].copy()
    sub["target_norm"] = sub["target"].map(normalize_target_name)
    sub = sub[sub["target_norm"].isin(targets)]

    if sub.empty:
        return {
            "slow_regression_available": False,
            "slow_regression_best_target": "missing",
            "slow_regression_best_score": np.nan,
            "slow_regression_summary": "missing",
        }

    sub["score_float"] = pd.to_numeric(sub["score"], errors="coerce")
    best = sub.sort_values("score_float", ascending=False).iloc[0]

    summary = []
    for _, row in sub.sort_values("score_float", ascending=False).iterrows():
        summary.append(f"{row.get('target')}: R2={fmt_float(row.get('score'))}")

    return {
        "slow_regression_available": True,
        "slow_regression_best_target": best.get("target"),
        "slow_regression_best_score": to_float(best.get("score")),
        "slow_regression_summary": "; ".join(summary),
    }


def extract_trend_evidence(state_cfg: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    df = data.get("state_dynamics_best", pd.DataFrame())

    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "trend_available": False,
            "trend_best_target": "missing",
            "trend_best_score": np.nan,
            "trend_summary": "missing",
        }

    targets = {normalize_target_name(t) for t in state_cfg.get("trend_targets", [])}

    if "target" not in df.columns or "task_family" not in df.columns:
        return {
            "trend_available": False,
            "trend_best_target": "missing",
            "trend_best_score": np.nan,
            "trend_summary": "missing",
        }

    sub = df[df["task_family"].astype(str).str.contains("trend/change", case=False, na=False)].copy()
    sub["target_norm"] = sub["target"].map(normalize_target_name)
    sub = sub[sub["target_norm"].isin(targets)]

    if sub.empty:
        return {
            "trend_available": False,
            "trend_best_target": "missing",
            "trend_best_score": np.nan,
            "trend_summary": "missing",
        }

    sub["score_float"] = pd.to_numeric(sub["score"], errors="coerce")
    best = sub.sort_values("score_float", ascending=False).iloc[0]

    summary = []
    for _, row in sub.sort_values("score_float", ascending=False).iterrows():
        summary.append(f"{row.get('target')}: BA={fmt_float(row.get('score'))}")

    return {
        "trend_available": True,
        "trend_best_target": best.get("target"),
        "trend_best_score": to_float(best.get("score")),
        "trend_summary": "; ".join(summary),
    }


def extract_calibration_evidence(state_cfg: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    df = data.get("calibration_highlight", pd.DataFrame())

    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "calibration_available": False,
            "calibration_20pct_best_gain": np.nan,
            "subject_dependent_best_gain": np.nan,
            "calibration_summary": "missing",
        }

    targets = {normalize_target_name(t) for t in state_cfg.get("calibration_targets", [])}

    if "target" not in df.columns:
        return {
            "calibration_available": False,
            "calibration_20pct_best_gain": np.nan,
            "subject_dependent_best_gain": np.nan,
            "calibration_summary": "missing",
        }

    sub = df.copy()
    sub["target_norm"] = sub["target"].map(normalize_target_name)
    sub = sub[sub["target_norm"].isin(targets)]

    if sub.empty:
        return {
            "calibration_available": False,
            "calibration_20pct_best_gain": np.nan,
            "subject_dependent_best_gain": np.nan,
            "calibration_summary": "missing",
        }

    sub["gain_float"] = pd.to_numeric(sub["absolute_gain_vs_zero_shot"], errors="coerce")

    cal20 = sub[sub["mode"].astype(str) == "calibration_20pct"]
    subject = sub[sub["mode"].astype(str) == "subject_dependent"]

    cal20_gain = cal20["gain_float"].max() if not cal20.empty else np.nan
    subject_gain = subject["gain_float"].max() if not subject.empty else np.nan

    summary = []
    for _, row in sub.sort_values("gain_float", ascending=False).head(6).iterrows():
        summary.append(
            f"{row.get('target')} {row.get('mode')}: "
            f"gain={fmt_float(row.get('absolute_gain_vs_zero_shot'))}"
        )

    return {
        "calibration_available": True,
        "calibration_20pct_best_gain": float(cal20_gain) if not pd.isna(cal20_gain) else np.nan,
        "subject_dependent_best_gain": float(subject_gain) if not pd.isna(subject_gain) else np.nan,
        "calibration_summary": "; ".join(summary),
    }


def extract_sensitivity_evidence(state_cfg: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    df = data.get("sensitivity_stability", pd.DataFrame())

    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "sensitivity_available": False,
            "sensitivity_best_stability_range": np.nan,
            "sensitivity_summary": "missing",
            "sensitivity_level": "unknown",
        }

    targets = {
        f"pm_{normalize_target_name(t)}_slow" for t in state_cfg.get("slow_targets", [])
    } | {
        f"pm_{normalize_target_name(t)}_trend_next" for t in state_cfg.get("trend_targets", [])
    }

    if "target" not in df.columns:
        return {
            "sensitivity_available": False,
            "sensitivity_best_stability_range": np.nan,
            "sensitivity_summary": "missing",
            "sensitivity_level": "unknown",
        }

    sub = df[df["target"].astype(str).isin(targets)].copy()

    if sub.empty:
        return {
            "sensitivity_available": False,
            "sensitivity_best_stability_range": np.nan,
            "sensitivity_summary": "missing",
            "sensitivity_level": "unknown",
        }

    sub["range_float"] = pd.to_numeric(sub["metric_range_across_params"], errors="coerce")
    best_range = sub["range_float"].min()

    summary = []
    for _, row in sub.sort_values("range_float", ascending=True).iterrows():
        summary.append(
            f"{row.get('target')}: range={fmt_float(row.get('metric_range_across_params'))}, "
            f"mean={fmt_float(row.get('metric_mean_across_params'))}"
        )

    if pd.isna(best_range):
        level = "unknown"
    elif best_range <= 0.03:
        level = "high"
    elif best_range <= 0.08:
        level = "medium"
    else:
        level = "low"

    return {
        "sensitivity_available": True,
        "sensitivity_best_stability_range": float(best_range),
        "sensitivity_summary": "; ".join(summary),
        "sensitivity_level": level,
    }


def extract_device_evidence(state_name: str, data: dict[str, Any]) -> dict[str, Any]:
    df = data.get("device_alignment", pd.DataFrame())

    if not isinstance(df, pd.DataFrame) or df.empty or "latent_state" not in df.columns:
        return {
            "device_alignment_available": False,
            "helmet_evidence": "missing",
            "helmet_trend_evidence": "missing",
            "bracelet_evidence": "missing",
            "movement_or_context_evidence": "missing",
            "device_status": "missing",
        }

    state_tokens = state_name.lower().replace("/", " ").split()

    def rough_match(x: str) -> bool:
        low = str(x).lower()
        return any(tok in low for tok in state_tokens if len(tok) > 4)

    sub = df[df["latent_state"].map(rough_match)].copy()

    if sub.empty and state_name == "Movement / Context / Reliability":
        sub = df[df["latent_state"].astype(str).str.contains("Movement|Reliability", case=False, regex=True, na=False)]

    if sub.empty:
        return {
            "device_alignment_available": False,
            "helmet_evidence": "missing",
            "helmet_trend_evidence": "missing",
            "bracelet_evidence": "missing",
            "movement_or_context_evidence": "missing",
            "device_status": "missing",
        }

    row = sub.iloc[0]

    return {
        "device_alignment_available": True,
        "helmet_evidence": row.get("helmet_evidence", "missing"),
        "helmet_trend_evidence": row.get("helmet_trend_evidence", "missing"),
        "bracelet_evidence": row.get("bracelet_evidence", "missing"),
        "movement_or_context_evidence": row.get("movement_or_context_evidence", "missing"),
        "device_status": row.get("status", "missing"),
    }


def score_regression(r2: float) -> int:
    if pd.isna(r2):
        return 0
    if r2 >= 0.35:
        return 3
    if r2 >= 0.20:
        return 2
    if r2 > 0.05:
        return 1
    return 0


def score_trend(ba: float) -> int:
    if pd.isna(ba):
        return 0
    if ba >= 0.46:
        return 3
    if ba >= 0.42:
        return 2
    if ba >= 0.38:
        return 1
    return 0


def score_cross_source(r2: float) -> int:
    if pd.isna(r2):
        return 0
    if r2 >= 0.50:
        return 3
    if r2 >= 0.35:
        return 2
    if r2 > 0.10:
        return 1
    return 0


def score_transformer_cross_source(r2: float) -> int:
    if pd.isna(r2):
        return 0
    if r2 >= 0.70:
        return 3
    if r2 >= 0.55:
        return 2
    if r2 >= 0.30:
        return 1
    return 0


def score_transformer_groupkfold(r2: float) -> int:
    if pd.isna(r2):
        return 0
    if r2 >= 0.30:
        return 2
    if r2 >= 0.10:
        return 1
    return 0


def score_calibration(gain: float) -> int:
    if pd.isna(gain):
        return 0
    if gain >= 0.20:
        return 2
    if gain >= 0.05:
        return 1
    return 0


def score_sensitivity(level: str) -> int:
    if level == "high":
        return 2
    if level == "medium":
        return 1
    return 0


def score_external(proxy: str) -> int:
    low = str(proxy).lower()
    if "wesad" in low and ("bvp" in low or "eda" in low or "temp" in low):
        return 2
    if "wesad" in low or "acc" in low or "proxy" in low:
        return 1
    return 0


def score_temporal(relevance: str) -> int:
    if relevance == "high":
        return 2
    if relevance == "medium":
        return 1
    return 0


def evidence_level(total: int) -> str:
    if total >= 16:
        return "strong"
    if total >= 10:
        return "moderate"
    if total >= 5:
        return "weak"
    return "insufficient"


def build_state_evidence_matrix(data: dict[str, Any]) -> pd.DataFrame:
    rows = []

    for state_name, state_cfg in STATE_REGISTRY.items():
        latent_axis = state_cfg["latent_axis"]

        cross = extract_cross_source_for_axis(latent_axis, data)
        transformer = extract_transformer_evidence(latent_axis, data)
        slow = extract_slow_regression_evidence(state_cfg, data)
        trend = extract_trend_evidence(state_cfg, data)
        calibration = extract_calibration_evidence(state_cfg, data)
        sensitivity = extract_sensitivity_evidence(state_cfg, data)
        device = extract_device_evidence(state_name, data)

        latent_score = 1 if latent_axis else 0
        slow_score = score_regression(slow["slow_regression_best_score"])
        trend_score_value = score_trend(trend["trend_best_score"])
        cross_score = score_cross_source(cross["cross_source_mean_r2"])
        transformer_cross_score = score_transformer_cross_source(transformer["transformer_cross_source_r2"])
        transformer_group_score = score_transformer_groupkfold(transformer["transformer_groupkfold_r2"])
        calibration_score = score_calibration(calibration["calibration_20pct_best_gain"])
        sensitivity_score = score_sensitivity(sensitivity["sensitivity_level"])
        external_score = score_external(state_cfg["external_proxy"])
        temporal_score = score_temporal(state_cfg["temporal_relevance"])

        total = (
            latent_score
            + slow_score
            + trend_score_value
            + cross_score
            + transformer_cross_score
            + transformer_group_score
            + calibration_score
            + sensitivity_score
            + external_score
            + temporal_score
        )

        if state_name == "Movement / Context / Reliability":
            main_conclusion = (
                "Supporting reliability dimension. Use as quality-control and confounder-control block."
            )
        elif transformer_cross_score >= 2 and total >= 16:
            main_conclusion = "Strong candidate for Transformer-based latent trajectory modeling."
        elif total >= 10:
            main_conclusion = "Moderate candidate; keep as trajectory target or validation axis."
        else:
            main_conclusion = "Weak current evidence; use cautiously or as auxiliary analysis."

        rows.append(
            {
                "state": state_name,
                "latent_axis": latent_axis or "none",
                "latent_interpretation": state_cfg["latent_interpretation"],
                "slow_regression_best_target": slow["slow_regression_best_target"],
                "slow_regression_best_r2": slow["slow_regression_best_score"],
                "slow_regression_summary": slow["slow_regression_summary"],
                "trend_best_target": trend["trend_best_target"],
                "trend_best_balanced_accuracy": trend["trend_best_score"],
                "trend_summary": trend["trend_summary"],
                "cross_source_mean_r2": cross["cross_source_mean_r2"],
                "cross_source_best_r2": cross["cross_source_best_r2"],
                "cross_source_mean_spearman": cross["cross_source_mean_spearman"],
                "cross_source_summary": cross["cross_source_summary"],
                "transformer_groupkfold_r2": transformer["transformer_groupkfold_r2"],
                "transformer_groupkfold_spearman": transformer["transformer_groupkfold_spearman"],
                "transformer_cross_source_r2": transformer["transformer_cross_source_r2"],
                "transformer_cross_source_spearman": transformer["transformer_cross_source_spearman"],
                "transformer_cross_source_summary": transformer["transformer_cross_source_summary"],
                "calibration_20pct_best_gain": calibration["calibration_20pct_best_gain"],
                "subject_dependent_best_gain": calibration["subject_dependent_best_gain"],
                "calibration_summary": calibration["calibration_summary"],
                "sensitivity_level": sensitivity["sensitivity_level"],
                "sensitivity_best_range": sensitivity["sensitivity_best_stability_range"],
                "sensitivity_summary": sensitivity["sensitivity_summary"],
                "temporal_modeling_relevance": state_cfg["temporal_relevance"],
                "temporal_modeling_comment": state_cfg["temporal_comment"],
                "external_dataset_evidence": state_cfg["external_proxy"],
                "helmet_evidence": device["helmet_evidence"],
                "helmet_trend_evidence": device["helmet_trend_evidence"],
                "bracelet_evidence": device["bracelet_evidence"],
                "movement_or_context_evidence": device["movement_or_context_evidence"],
                "device_status": device["device_status"],
                "latent_score": latent_score,
                "slow_regression_score": slow_score,
                "trend_score": trend_score_value,
                "cross_source_score": cross_score,
                "transformer_cross_source_score": transformer_cross_score,
                "transformer_groupkfold_score": transformer_group_score,
                "calibration_score": calibration_score,
                "sensitivity_score": sensitivity_score,
                "external_score": external_score,
                "temporal_score": temporal_score,
                "total_evidence_score": total,
                "evidence_level": evidence_level(total),
                "main_conclusion": main_conclusion,
                "recommended_next_step": state_cfg["recommended_next_step"],
            }
        )

    return pd.DataFrame(rows)


def build_state_scores(evidence: pd.DataFrame) -> pd.DataFrame:
    score_cols = [
        "latent_score",
        "slow_regression_score",
        "trend_score",
        "cross_source_score",
        "transformer_cross_source_score",
        "transformer_groupkfold_score",
        "calibration_score",
        "sensitivity_score",
        "external_score",
        "temporal_score",
        "total_evidence_score",
    ]

    keep_cols = ["state", "latent_axis", "evidence_level"] + score_cols

    out = evidence[keep_cols].copy()
    out = out.sort_values("total_evidence_score", ascending=False).reset_index(drop=True)
    return out


def build_external_dataset_mapping(evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in evidence.iterrows():
        rows.append(
            {
                "state": row["state"],
                "external_dataset": "WESAD",
                "external_signal_or_proxy": row["external_dataset_evidence"],
                "use_in_project": (
                    "main external proxy"
                    if "WESAD" in str(row["external_dataset_evidence"])
                    else "no direct analogue"
                ),
                "interpretation_limit": (
                    "Not equivalent to EEG/PM target; use only as physiological proxy."
                    if "WESAD" in str(row["external_dataset_evidence"])
                    else "External validation still missing."
                ),
            }
        )

    return pd.DataFrame(rows)


def build_state_recommendations(evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in evidence.iterrows():
        if row["state"] == "Movement / Context / Reliability":
            priority = "high-supporting"
            model_target = "EEG reliability / artifact score"
            modeling = "anomaly detection, filtering, sample weighting"
        elif row["evidence_level"] == "strong":
            priority = "high"
            model_target = row["latent_axis"]
            modeling = "TransformerEncoder temporal trajectory model"
        elif row["evidence_level"] == "moderate":
            priority = "medium"
            model_target = row["latent_axis"]
            modeling = "secondary target in temporal model"
        else:
            priority = "low"
            model_target = row["latent_axis"]
            modeling = "auxiliary analysis only"

        rows.append(
            {
                "state": row["state"],
                "priority": priority,
                "recommended_model_target": model_target,
                "recommended_modeling_strategy": modeling,
                "recommended_next_step": row["recommended_next_step"],
            }
        )

    return pd.DataFrame(rows)


def build_state_limitations(evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in evidence.iterrows():
        limitations = []

        if row["latent_axis"] == "none":
            limitations.append("no latent PM axis")
        if row["cross_source_score"] == 0:
            limitations.append("no classical cross-source evidence")
        if row["transformer_cross_source_score"] == 0 and row["latent_axis"] != "none":
            limitations.append("no Transformer cross-source evidence")
        if row["external_score"] == 0:
            limitations.append("no direct external wearable validation")
        if row["calibration_score"] == 0:
            limitations.append("calibration evidence missing or weak")
        if row["sensitivity_score"] == 0:
            limitations.append("sensitivity stability missing or weak")
        if "ACC" in str(row["external_dataset_evidence"]) or "movement" in str(row["movement_or_context_evidence"]).lower():
            limitations.append("movement/context confounding must be controlled")
        if row["transformer_groupkfold_score"] == 0 and row["latent_axis"] != "none":
            limitations.append("cross-subject Transformer generalization is weak or missing")

        rows.append(
            {
                "state": row["state"],
                "limitations": "; ".join(limitations) if limitations else "no major limitation beyond general dataset constraints",
            }
        )

    return pd.DataFrame(rows)


def build_temporal_modeling_evidence(note: str | None) -> pd.DataFrame:
    df = pd.DataFrame(TEMPORAL_MODELING_EVIDENCE)

    if note:
        df["project_note"] = note
    else:
        df["project_note"] = (
            "DL lab and project experiments support Transformer-based temporal modeling; "
            "cross-source Transformer transfer is strong for slow latent axes."
        )

    return df


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
    evidence: pd.DataFrame,
    scores: pd.DataFrame,
    temporal: pd.DataFrame,
    external: pd.DataFrame,
    recommendations: pd.DataFrame,
    limitations: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# Integrated cognitive-affective state evidence report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This report integrates the confirmed hypotheses of the EEG/PM project into a single evidence matrix."
    )
    lines.append("")
    lines.append("It combines:")
    lines.append("")
    lines.append("- latent cognitive-affective states;")
    lines.append("- slow/background and trend/change-direction state types;")
    lines.append("- classical cross-source transfer between `Old_EEG` and `gpn_data`;")
    lines.append("- Transformer-based latent trajectory transfer;")
    lines.append("- user calibration evidence;")
    lines.append("- dynamics sensitivity analysis;")
    lines.append("- helmet vs bracelet alignment;")
    lines.append("- external wearable proxy evidence from WESAD;")
    lines.append("- DL-lab conclusion about temporal modeling.")
    lines.append("")

    lines.append("## Central interpretation")
    lines.append("")
    lines.append(
        "The main research direction should shift from isolated EEG/POW window prediction "
        "to temporal modeling of latent cognitive-affective state trajectories."
    )
    lines.append("")
    lines.append("The preferred target form is:")
    lines.append("")
    lines.append("```text")
    lines.append("sequence of EEG/POW windows -> trajectory of slow latent states")
    lines.append("```")
    lines.append("")
    lines.append("The preferred model family is Transformer-based temporal modeling.")
    lines.append("")

    compact_cols = [
        "state",
        "latent_axis",
        "slow_regression_best_target",
        "slow_regression_best_r2",
        "trend_best_target",
        "trend_best_balanced_accuracy",
        "cross_source_mean_r2",
        "transformer_groupkfold_r2",
        "transformer_cross_source_r2",
        "transformer_cross_source_spearman",
        "calibration_20pct_best_gain",
        "sensitivity_level",
        "temporal_modeling_relevance",
        "external_dataset_evidence",
        "total_evidence_score",
        "evidence_level",
        "main_conclusion",
    ]

    append_table(lines, "Integrated state evidence matrix", evidence[compact_cols])
    append_table(lines, "Evidence scores", scores)
    append_table(lines, "Temporal modeling evidence", temporal)
    append_table(lines, "External dataset mapping", external)
    append_table(lines, "State recommendations", recommendations)
    append_table(lines, "State limitations", limitations)

    lines.append("## Main conclusion")
    lines.append("")
    lines.append(
        "The strongest integrated direction is Transformer-based modeling of latent cognitive-affective trajectories. "
        "The cross-source Transformer experiment strongly supports this direction: all four slow latent axes show positive transfer between `Old_EEG` and `gpn_data`, "
        "with the strongest evidence for arousal/stress and recovery/focus trajectories."
    )
    lines.append("")
    lines.append(
        "Autoencoder-based generation and synthetic augmentation should remain auxiliary, because feature-space generation works but has not yet shown stable performance gains."
    )
    lines.append("")
    lines.append(
        "Anomaly detection and movement/reliability analysis should be used as quality-control components, especially because WESAD evidence shows that movement can act as a strong confounder."
    )
    lines.append("")
    lines.append("## Recommended next technical step")
    lines.append("")
    lines.append("```text")
    lines.append("src/43_train_transformer_latent_trajectory_with_calibration.py")
    lines.append("```")
    lines.append("")
    lines.append("Target:")
    lines.append("")
    lines.append("```text")
    lines.append("EEG/POW sequence -> slow_pca_1..4 trajectory")
    lines.append("```")
    lines.append("")
    lines.append("Validation:")
    lines.append("")
    lines.append("```text")
    lines.append("GroupKFold / cross-subject")
    lines.append("cross-source Old_EEG <-> gpn_data")
    lines.append("user calibration mode")
    lines.append("```")
    lines.append("")

    (output_dir / "integrated_state_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    short_lines = []
    short_lines.append("# Integrated state evidence short summary")
    short_lines.append("")
    short_lines.append("Main result:")
    short_lines.append("")
    short_lines.append(
        "The project evidence supports moving from isolated EEG/POW windows to Transformer-based modeling of latent cognitive-affective trajectories. "
        "The cross-source Transformer experiment gave strong positive transfer for all four slow latent axes."
    )
    short_lines.append("")
    short_lines.append("Most promising directions:")
    short_lines.append("")
    short_lines.append("- Stress / Arousal / General activation: strong EEG/PM, Transformer, and WESAD-proxy evidence.")
    short_lines.append("- Recovery / Fatigue / Relaxation: strong EEG/PM evidence and strong Transformer cross-source transfer.")
    short_lines.append("- Engagement / Involvement and Attention / Control: useful trajectory targets, but cross-subject stability needs additional work.")
    short_lines.append("- Movement / Context / Reliability: not a primary cognitive state, but required as quality-control/confounder-control block.")
    short_lines.append("")
    short_lines.append("Next script:")
    short_lines.append("")
    short_lines.append("```text")
    short_lines.append("src/43_train_transformer_latent_trajectory_with_calibration.py")
    short_lines.append("```")
    short_lines.append("")

    (output_dir / "integrated_state_report_short.md").write_text(
        "\n".join(short_lines),
        encoding="utf-8",
    )


def save_outputs(
    output_dir: Path,
    evidence: pd.DataFrame,
    scores: pd.DataFrame,
    temporal: pd.DataFrame,
    external: pd.DataFrame,
    recommendations: pd.DataFrame,
    limitations: pd.DataFrame,
    config: Config,
    data: dict[str, Any],
) -> None:
    evidence.to_csv(output_dir / "state_evidence_matrix.csv", index=False)
    scores.to_csv(output_dir / "state_evidence_scores.csv", index=False)
    temporal.to_csv(output_dir / "temporal_modeling_evidence.csv", index=False)
    external.to_csv(output_dir / "external_dataset_mapping.csv", index=False)
    recommendations.to_csv(output_dir / "state_recommendations.csv", index=False)
    limitations.to_csv(output_dir / "state_limitations.csv", index=False)

    metadata = {
        "run_name": config.run_name,
        "output_dir": str(output_dir),
        "input_dirs": {
            "project_summary_dir": str(config.project_summary_dir),
            "slow_latent_cross_source_dir": str(config.slow_latent_cross_source_dir),
            "dynamics_summary_dir": str(config.dynamics_summary_dir),
            "dynamics_sensitivity_dir": str(config.dynamics_sensitivity_dir),
            "user_calibration_dir": str(config.user_calibration_dir),
            "device_alignment_dir": str(config.device_alignment_dir),
            "wesad_dir": str(config.wesad_dir),
            "transformer_summary_dir": str(config.transformer_summary_dir),
            "transformer_cross_source_dir": str(config.transformer_cross_source_dir),
        },
        "inputs_found": {
            name: (not value.empty if isinstance(value, pd.DataFrame) else bool(value))
            for name, value in data.items()
        },
        "n_states": int(len(evidence)),
        "n_strong_states": int((evidence["evidence_level"] == "strong").sum()),
        "n_moderate_states": int((evidence["evidence_level"] == "moderate").sum()),
        "recommended_next_script": "src/43_train_transformer_latent_trajectory_with_calibration.py",
    }

    (output_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Build integrated evidence matrix for EEG/PM cognitive-affective states."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/integrated_state_evidence_v2"),
        help="Output directory.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="integrated_state_evidence_v2",
        help="Run name.",
    )
    parser.add_argument(
        "--project-summary-dir",
        type=Path,
        default=Path("reports/project_summary"),
        help="Directory with project summary outputs from script 40.",
    )
    parser.add_argument(
        "--slow-latent-cross-source-dir",
        type=Path,
        default=Path("reports/slow_latent_states/cross_source"),
        help="Directory with slow latent classical cross-source outputs.",
    )
    parser.add_argument(
        "--dynamics-summary-dir",
        type=Path,
        default=Path("reports/state_dynamics/pm_w10_experiment_summary_v4"),
        help="Directory with state dynamics summary outputs.",
    )
    parser.add_argument(
        "--dynamics-sensitivity-dir",
        type=Path,
        default=Path("reports/state_dynamics/sensitivity"),
        help="Directory with state dynamics sensitivity outputs.",
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
        help="Directory with device metric alignment outputs.",
    )
    parser.add_argument(
        "--wesad-dir",
        type=Path,
        default=Path("reports/wearable_pm_alignment"),
        help="Directory with WESAD/wearable outputs.",
    )
    parser.add_argument(
        "--transformer-summary-dir",
        type=Path,
        default=Path("reports/latent_trajectory_transformer"),
        help="Directory with Transformer trajectory outputs for random/groupkfold validation.",
    )
    parser.add_argument(
        "--transformer-cross-source-dir",
        type=Path,
        default=Path("reports/latent_trajectory_transformer_cross_source"),
        help="Directory with Transformer trajectory cross-source outputs.",
    )
    parser.add_argument(
        "--temporal-modeling-note",
        type=str,
        default=None,
        help="Optional note from DL lab about temporal modeling.",
    )

    args = parser.parse_args()

    return Config(
        output_dir=args.output_dir,
        run_name=args.run_name,
        project_summary_dir=args.project_summary_dir,
        slow_latent_cross_source_dir=args.slow_latent_cross_source_dir,
        dynamics_summary_dir=args.dynamics_summary_dir,
        dynamics_sensitivity_dir=args.dynamics_sensitivity_dir,
        user_calibration_dir=args.user_calibration_dir,
        device_alignment_dir=args.device_alignment_dir,
        wesad_dir=args.wesad_dir,
        transformer_summary_dir=args.transformer_summary_dir,
        transformer_cross_source_dir=args.transformer_cross_source_dir,
        temporal_modeling_note=args.temporal_modeling_note,
    )


def main() -> None:
    logger = setup_logging()
    config = parse_args()

    config.output_dir = config.output_dir.resolve()
    config.project_summary_dir = config.project_summary_dir.resolve()
    config.slow_latent_cross_source_dir = config.slow_latent_cross_source_dir.resolve()
    config.dynamics_summary_dir = config.dynamics_summary_dir.resolve()
    config.dynamics_sensitivity_dir = config.dynamics_sensitivity_dir.resolve()
    config.user_calibration_dir = config.user_calibration_dir.resolve()
    config.device_alignment_dir = config.device_alignment_dir.resolve()
    config.wesad_dir = config.wesad_dir.resolve()
    config.transformer_summary_dir = config.transformer_summary_dir.resolve()
    config.transformer_cross_source_dir = config.transformer_cross_source_dir.resolve()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Build integrated state evidence v2")
    logger.info("=" * 80)
    logger.info("Output dir: %s", config.output_dir)
    logger.info("Transformer summary dir: %s", config.transformer_summary_dir)
    logger.info("Transformer cross-source dir: %s", config.transformer_cross_source_dir)

    data = load_inputs(config)

    evidence = build_state_evidence_matrix(data)
    scores = build_state_scores(evidence)
    temporal = build_temporal_modeling_evidence(config.temporal_modeling_note)
    external = build_external_dataset_mapping(evidence)
    recommendations = build_state_recommendations(evidence)
    limitations = build_state_limitations(evidence)

    save_outputs(
        output_dir=config.output_dir,
        evidence=evidence,
        scores=scores,
        temporal=temporal,
        external=external,
        recommendations=recommendations,
        limitations=limitations,
        config=config,
        data=data,
    )

    write_report(
        output_dir=config.output_dir,
        config=config,
        evidence=evidence,
        scores=scores,
        temporal=temporal,
        external=external,
        recommendations=recommendations,
        limitations=limitations,
    )

    logger.info("=" * 80)
    logger.info("Saved integrated state evidence v2")
    logger.info("=" * 80)
    logger.info("Evidence matrix: %s", config.output_dir / "state_evidence_matrix.csv")
    logger.info("Evidence scores: %s", config.output_dir / "state_evidence_scores.csv")
    logger.info("Report: %s", config.output_dir / "integrated_state_report.md")
    logger.info("Short report: %s", config.output_dir / "integrated_state_report_short.md")
    logger.info("Summary: %s", config.output_dir / "summary.json")

    with pd.option_context("display.max_rows", 20, "display.max_columns", 12, "display.width", 180):
        logger.info(
            "Scores:\n%s",
            scores[
                [
                    "state",
                    "latent_axis",
                    "total_evidence_score",
                    "evidence_level",
                    "transformer_cross_source_score",
                    "transformer_groupkfold_score",
                    "temporal_score",
                    "external_score",
                ]
            ].to_string(index=False),
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()