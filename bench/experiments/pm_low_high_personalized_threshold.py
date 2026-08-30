"""Leakage-safe personalized decision-threshold experiment for seven PM LOW/HIGH tasks.

The experiment reuses already stored outer-test probabilities from the completed
XGBoost and LightGBM LOW/HIGH experiments. No base model is retrained and no
base-model inference is repeated.

For each outer-test participant and PM:
- calibration labels come only from the chronological prefix of the earliest
  logical recording;
- the evaluation set is fixed strictly after +300 s for every budget;
- cross-record UTC overlap is removed by the completed feasibility protocol;
- calibration budgets are 60, 120, and 300 s; 30 s is not executed because the
  feasibility audit found zero participant-PM cells with >=2 LOW and >=2 HIGH;
- supervised threshold adaptation requires at least two LOW and two HIGH
  calibration examples and a fully available budget;
- otherwise the operational policy falls back to the zero-shot threshold 0.5;
- evaluation labels never influence threshold fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from bench.experiments.pm_low_high_personalization_feasibility import (
    load_config as load_feasibility_config,
    prepare_protocol as prepare_feasibility_protocol,
    _subject_pm_timeline,
)
from bench.experiments.pm_low_high_q3_extremes_confirmatory import (
    PM_NAMES,
    _sample_hash,
    participant_binary_metrics,
    stable_hash,
)

SCHEMA_VERSION = "pm-low-high-personalized-threshold-v1"
MODELS = ("xgboost", "lightgbm")
BUDGETS = (60, 120, 300)
STRATEGIES = ("median_midpoint", "empirical_balanced_accuracy")
METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "low_recall",
    "high_recall",
    "precision",
    "accuracy",
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                   allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def _safe_hash_frame(frame: pd.DataFrame) -> str:
    return stable_hash(
        frame.to_csv(index=False, lineterminator="\n", na_rep="<NA>")
    )


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if cfg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    refs = cfg.get("references", {})
    expected = {
        "feasibility": "94c568d7e41344478c0550f573b0abf8893783831f6c7241b92c8e4fdd25c9cd",
        "xgboost": "ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431",
        "lightgbm": "a2c51deb4d94e33d71863f1f1c7927470266b7352647664f3a411fb5db819b4e",
    }
    for key, phash in expected.items():
        if refs.get(key, {}).get("protocol_hash") != phash:
            raise ValueError(f"{key} reference protocol changed")

    c = cfg.get("scientific_contract", {})
    expected_contract = {
        "pm_names": list(PM_NAMES),
        "models": list(MODELS),
        "alignment": "EEG(t-10s) -> PM(t)",
        "lag_seconds": -10,
        "target_transform": "outer_train_q33_q67_extremes",
        "threshold_fit_scope": "outer_train_continuous_complete_cases",
        "middle_policy": "exclude",
        "calibration_record_policy":
            "earliest_logical_record_by_selected_record_start_utc",
        "cross_record_overlap_policy":
            "earlier_record_precedence_trim_later_overlapping_prefix_by_feature_grid_utc",
        "fixed_evaluation_boundary_seconds": 300,
        "fixed_evaluation_policy":
            "strictly_after_earliest_record_start_plus_300s",
        "budgets_seconds": list(BUDGETS),
        "budget_roles": {"60":"exploratory","120":"secondary","300":"primary"},
        "excluded_budget_30s_reason": "feasibility_min2_each_zero_of_378",
        "minimum_calibration_per_class": 2,
        "minimum_fixed_evaluation_extremes": 20,
        "require_both_evaluation_classes": True,
        "ineligible_policy": "zero_shot_fallback_no_budget_extension",
        "probability_source": "stored_predict_proba_high",
        "base_decision_threshold": 0.5,
    }
    if c != expected_contract:
        raise ValueError("Scientific contract changed")

    s = cfg.get("strategies", {})
    if tuple(s) != STRATEGIES:
        raise ValueError("Strategies/order changed")
    if s["median_midpoint"] != {
        "role": "primary",
        "rule": "midpoint_between_median_LOW_and_median_HIGH_probability",
        "nonseparated_medians_policy": "zero_shot_fallback",
    }:
        raise ValueError("Primary threshold strategy changed")
    if s["empirical_balanced_accuracy"] != {
        "role": "sensitivity",
        "rule": "maximize_calibration_balanced_accuracy_over_probability_midpoints",
        "tie_break": ["closest_to_0.5","lower_threshold"],
    }:
        raise ValueError("Sensitivity threshold strategy changed")

    ev = cfg.get("evaluation", {})
    if ev != {
        "primary_metric": "balanced_accuracy",
        "secondary_metrics": ["macro_f1","low_recall","high_recall","precision","accuracy"],
        "unchanged_ranking_metrics": ["roc_auc","pr_auc"],
        "aggregation": "mean_pm_within_participant_then_mean_participants",
        "primary_estimand":
            "operational_all_fixed_evaluation_ready_with_zero_shot_fallback",
        "secondary_estimand": "adaptation_applied_only",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 42,
        "bootstrap_unit": "subject_id",
    }:
        raise ValueError("Evaluation contract changed")
    forbidden = cfg.get("forbidden", {})
    if not forbidden or not all(value is True for value in forbidden.values()):
        raise ValueError("All forbidden switches must remain true")
    return cfg


def _read_completed_protocol(root: Path, ref: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = root / ref["output_dir"] / "protocol.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol_hash") != ref["protocol_hash"]:
        raise RuntimeError(f"{label} stored protocol hash changed")
    if value.get("result_status") not in {
        "confirmatory_complete", "feasibility_audit_complete"
    }:
        raise RuntimeError(f"{label} is not complete")
    return value


def _prediction_source_matrix(
    root: Path, cfg: Mapping[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # XGBoost: original confirmatory run matrix.
    xref = cfg["references"]["xgboost"]
    xmatrix = pd.read_csv(root / xref["output_dir"] / "run_matrix.csv")
    if len(xmatrix) != 35:
        raise RuntimeError("XGBoost source run matrix must contain 35 rows")
    for row in xmatrix.to_dict("records"):
        rows.append({
            "model": "xgboost",
            "outer_fold": int(row["outer_fold"]),
            "pm": str(row["pm"]),
            "run_id": str(row["run_id"]),
            "threshold_hash": str(row["threshold_hash"]),
            "test_sample_hash": str(row["test_sample_hash"]),
            "source_output_dir": xref["output_dir"],
        })

    # LightGBM: candidate run matrix from model robustness.
    lref = cfg["references"]["lightgbm"]
    lmatrix = pd.read_csv(root / lref["output_dir"] / "run_matrix.csv")
    lmatrix = lmatrix.loc[lmatrix["model"].astype(str).eq("lightgbm")].copy()
    if len(lmatrix) != 35:
        raise RuntimeError("LightGBM source run matrix must contain 35 rows")
    for row in lmatrix.to_dict("records"):
        rows.append({
            "model": "lightgbm",
            "outer_fold": int(row["outer_fold"]),
            "pm": str(row["pm"]),
            "run_id": str(row["run_id"]),
            "threshold_hash": str(row["threshold_hash"]),
            "test_sample_hash": str(row["test_sample_hash"]),
            "source_output_dir": lref["output_dir"],
        })

    frame = pd.DataFrame(rows).sort_values(
        ["model", "outer_fold", "pm"], kind="stable"
    ).reset_index(drop=True)
    if len(frame) != 70 or frame.duplicated(["model","outer_fold","pm"]).any():
        raise RuntimeError("Exactly 70 unique stored prediction sources are required")
    return frame


def _audit_prediction_sources(
    root: Path, source_matrix: pd.DataFrame
) -> pd.DataFrame:
    """Strictly verify stored predictions against their frozen run identity."""
    audits = []
    required = {
        "target_sample_id", "subject_id", "record_id", "outer_fold", "pm",
        "y_true", "probability_high",
    }

    for src in source_matrix.to_dict("records"):
        run_dir = (
            root / src["source_output_dir"] / "runs" / src["run_id"]
        )
        prediction_path = run_dir / "predictions.parquet"
        summary_path = run_dir / "run_summary.json"

        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)

        pred = pd.read_parquet(prediction_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        missing = sorted(required - set(pred.columns))
        if missing:
            raise RuntimeError(
                f"{prediction_path}: missing columns {missing}"
            )

        if pred["target_sample_id"].astype(str).duplicated().any():
            raise RuntimeError(
                f"{prediction_path}: duplicate target_sample_id"
            )

        if set(pred["pm"].astype(str)) != {src["pm"]}:
            raise RuntimeError(f"{prediction_path}: PM mismatch")

        if set(pred["outer_fold"].astype(int)) != {
            int(src["outer_fold"])
        }:
            raise RuntimeError(f"{prediction_path}: fold mismatch")

        y = pred["y_true"].to_numpy(dtype=int)
        if not set(np.unique(y)).issubset({0, 1}):
            raise RuntimeError(f"{prediction_path}: invalid y_true")

        probability = pred["probability_high"].to_numpy(dtype=float)
        if (
            not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))
        ):
            raise RuntimeError(
                f"{prediction_path}: invalid probabilities"
            )

        expected_protocol_hash = (
            "ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431"
            if src["model"] == "xgboost"
            else
            "a2c51deb4d94e33d71863f1f1c7927470266b7352647664f3a411fb5db819b4e"
        )

        if summary.get("status") != "complete":
            raise RuntimeError(
                f"{summary_path}: source run is not complete"
            )

        if summary.get("protocol_hash") != expected_protocol_hash:
            raise RuntimeError(
                f"{summary_path}: protocol hash mismatch"
            )

        if str(summary.get("run_id")) != str(src["run_id"]):
            raise RuntimeError(
                f"{summary_path}: run_id mismatch"
            )

        if int(summary.get("outer_fold", -1)) != int(
            src["outer_fold"]
        ):
            raise RuntimeError(
                f"{summary_path}: outer_fold mismatch"
            )

        if str(summary.get("pm")) != str(src["pm"]):
            raise RuntimeError(
                f"{summary_path}: PM mismatch"
            )

        if str(summary.get("threshold_hash")) != str(
            src["threshold_hash"]
        ):
            raise RuntimeError(
                f"{summary_path}: threshold hash mismatch"
            )

        if int(summary.get("n_test", -1)) != len(pred):
            raise RuntimeError(
                f"{summary_path}: n_test differs from predictions"
            )

        actual_sample_hash = _sample_hash(
            pred["target_sample_id"].astype(str).tolist()
        )

        if actual_sample_hash != str(src["test_sample_hash"]):
            raise RuntimeError(
                f"{prediction_path}: test_sample_hash mismatch"
            )

        audits.append({
            **src,
            "prediction_rows": int(len(pred)),
            "prediction_subjects": int(
                pred["subject_id"].astype(str).nunique()
            ),
            "prediction_sample_hash": actual_sample_hash,
            "sample_hash_matches": True,
            "run_summary_matches": True,
            "prediction_file": str(prediction_path),
            "run_summary_file": str(summary_path),
            "valid": True,
        })

    return pd.DataFrame(audits)


@dataclass
class ThresholdContext:
    root: Path
    output_dir: Path
    config: dict[str, Any]
    feasibility: Any
    feasibility_detail: pd.DataFrame
    source_matrix: pd.DataFrame
    source_audit: pd.DataFrame
    protocol: dict[str, Any]


def prepare_protocol(
    config: Mapping[str, Any],
    *,
    root: str | Path,
    feature_cache_dir: str | Path,
    output_dir: str | Path | None = None,
) -> ThresholdContext:
    root = Path(root).resolve()
    output = Path(output_dir or config["output_dir"])
    if not output.is_absolute():
        output = root / output

    fref = config["references"]["feasibility"]
    _read_completed_protocol(root, fref, "feasibility")
    _read_completed_protocol(root, config["references"]["xgboost"], "xgboost")
    _read_completed_protocol(root, config["references"]["lightgbm"], "lightgbm")

    fcfg = load_feasibility_config(root / fref["config"])
    feasibility = prepare_feasibility_protocol(
        fcfg,
        root=root,
        feature_cache_dir=feature_cache_dir,
        output_dir=root / fref["output_dir"],
    )
    if feasibility.protocol["protocol_hash"] != fref["protocol_hash"]:
        raise RuntimeError("Recomputed feasibility protocol hash changed")

    detail_path = root / fref["output_dir"] / "participant_pm_budget_feasibility.csv"
    detail = pd.read_csv(detail_path)
    if len(detail) != 1890:
        raise RuntimeError("Feasibility detail must contain 1890 rows")

    source_matrix = _prediction_source_matrix(root, config)
    source_audit = _audit_prediction_sources(root, source_matrix)

    scientific_payload = {
        "schema_version": SCHEMA_VERSION,
        "references": config["references"],
        "scientific_contract": config["scientific_contract"],
        "strategies": config["strategies"],
        "evaluation": config["evaluation"],
        "forbidden": config["forbidden"],
        "feasibility_detail_hash": _safe_hash_frame(detail),
        "prediction_source_matrix_hash": _safe_hash_frame(
            source_matrix.drop(columns=["source_output_dir"])
        ),
        "prediction_source_audit_hash": _safe_hash_frame(
            source_audit.drop(
                columns=["prediction_file", "run_summary_file"]
            )
        ),
        "fixed_fold_hash": feasibility.low_high.protocol["fixed_fold_hash"],
        "threshold_hashes": feasibility.low_high.protocol["threshold_hashes"],
    }
    phash = stable_hash(scientific_payload)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "result_status": "preregistered_candidate",
        "threshold_calibration_executed": False,
        "base_model_training_executed": False,
        "base_model_inference_executed": False,
        "git_commit": _git_head(root),
        "protocol_hash": phash,
        "references": config["references"],
        "scientific_contract": config["scientific_contract"],
        "strategies": config["strategies"],
        "evaluation": config["evaluation"],
        "feasibility_detail_hash": scientific_payload["feasibility_detail_hash"],
        "prediction_source_rows": int(len(source_matrix)),
        "prediction_source_audit_mismatches": int((~source_audit["valid"]).sum()),
        "fixed_fold_hash": feasibility.low_high.protocol["fixed_fold_hash"],
        "threshold_hashes": feasibility.low_high.protocol["threshold_hashes"],
    }
    return ThresholdContext(
        root=root, output_dir=output, config=dict(config),
        feasibility=feasibility, feasibility_detail=detail,
        source_matrix=source_matrix, source_audit=source_audit,
        protocol=protocol,
    )


def write_dry_run(context: ThresholdContext) -> dict[str, Any]:
    context.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(context.output_dir / "protocol.json", context.protocol)
    _write_csv(context.output_dir / "prediction_source_matrix.csv", context.source_matrix)
    _write_csv(context.output_dir / "prediction_source_audit.csv", context.source_audit)
    summary = {
        "experiment_id": context.config["experiment_id"],
        "protocol_hash": context.protocol["protocol_hash"],
        "models": list(MODELS),
        "budgets_seconds": list(BUDGETS),
        "strategies": list(STRATEGIES),
        "prediction_source_rows": int(len(context.source_matrix)),
        "prediction_source_audit_mismatches": 0,
        "feasibility_rows": int(len(context.feasibility_detail)),
        "base_model_training_executed": False,
        "base_model_inference_executed": False,
        "threshold_calibration_executed": False,
    }
    _atomic_json(context.output_dir / "dry_run_summary.json", summary)
    readme = f"""# PM LOW/HIGH personalized threshold v1

This experiment reuses completed XGBoost and LightGBM outer-test probabilities.
It does not retrain either base model and does not repeat model inference.

- PM: 7
- models: XGBoost, LightGBM
- budgets: 60 s exploratory, 120 s secondary, 300 s primary
- 30 s omitted from adaptation because feasibility found 0/378 cells with >=2 LOW and >=2 HIGH
- calibration eligibility: fully available budget and >=2 LOW + >=2 HIGH
- fixed evaluation: strictly after +300 s, >=20 extreme samples and both classes
- ineligible calibration: zero-shot threshold 0.5 fallback
- primary strategy: midpoint between calibration LOW/HIGH median probabilities
- sensitivity strategy: calibration balanced-accuracy maximizing threshold
- primary metric: balanced accuracy
- protocol hash: `{context.protocol['protocol_hash']}`
- base training by dry-run: false
- base inference by dry-run: false
- threshold calibration by dry-run: false
"""
    (context.output_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary


def _load_prediction_lookup(context: ThresholdContext) -> dict[tuple[str,int,str], pd.DataFrame]:
    lookup = {}
    for src in context.source_matrix.to_dict("records"):
        key = (src["model"], int(src["outer_fold"]), src["pm"])
        path = (
            context.root / src["source_output_dir"] / "runs"
            / src["run_id"] / "predictions.parquet"
        )
        pred = pd.read_parquet(path).copy()
        pred["target_sample_id"] = pred["target_sample_id"].astype(str)
        lookup[key] = pred.set_index("target_sample_id", drop=False)
    return lookup


def _state_to_y(state: Sequence[str]) -> np.ndarray:
    mapping = {"low": 0, "high": 1}
    values = []
    for item in state:
        if str(item) not in mapping:
            raise ValueError("Only LOW/HIGH states can enter binary threshold calibration")
        values.append(mapping[str(item)])
    return np.asarray(values, dtype=np.int64)


def _median_midpoint(probability: np.ndarray, y: np.ndarray) -> tuple[float, bool, str, dict[str,float]]:
    low = probability[y == 0]
    high = probability[y == 1]
    med_low = float(np.median(low))
    med_high = float(np.median(high))
    extra = {"median_probability_low": med_low, "median_probability_high": med_high}
    if not med_high > med_low:
        return 0.5, False, "nonseparated_medians", extra
    return float((med_low + med_high) / 2.0), True, "adapted", extra


def _balanced_accuracy_binary(y: np.ndarray, pred: np.ndarray) -> float:
    low = float(np.mean(pred[y == 0] == 0))
    high = float(np.mean(pred[y == 1] == 1))
    return (low + high) / 2.0


def _empirical_ba_threshold(probability: np.ndarray, y: np.ndarray) -> tuple[float, bool, str, dict[str,float]]:
    unique = np.unique(probability)
    candidates = [0.0, 0.5, 1.0]
    if len(unique) > 1:
        candidates.extend(((unique[:-1] + unique[1:]) / 2.0).tolist())
    candidates = sorted(set(float(np.clip(value, 0.0, 1.0)) for value in candidates))
    scored = []
    for threshold in candidates:
        pred = (probability >= threshold).astype(np.int64)
        ba = _balanced_accuracy_binary(y, pred)
        scored.append((ba, abs(threshold - 0.5), threshold))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    best_ba, _, threshold = scored[0]
    return float(threshold), True, "adapted", {
        "calibration_balanced_accuracy_at_threshold": float(best_ba),
        "threshold_candidate_count": float(len(candidates)),
    }


def _metric_row(y: np.ndarray, probability: np.ndarray, threshold: float, subject_id: str) -> dict[str,float]:
    pred = (probability >= threshold).astype(np.int64)
    participants, _ = participant_binary_metrics(
        y, pred, probability, np.repeat(str(subject_id), len(y))
    )
    if len(participants) != 1:
        raise RuntimeError("Expected one participant metric row")
    row = participants.iloc[0]
    return {
        "balanced_accuracy": float(row["balanced_accuracy"]),
        "macro_f1": float(row["macro_f1"]),
        "low_recall": float(row["low_recall"]),
        "high_recall": float(row["high_recall"]),
        "precision": float(row["precision"]),
        "accuracy": float(row["accuracy"]),
        "roc_auc": float(row["roc_auc"]),
        "pr_auc": float(row["pr_auc"]),
    }


def _timeline_extremes(context: ThresholdContext, subject_id: str, pm: str) -> tuple[pd.DataFrame, pd.Series]:
    timeline = _subject_pm_timeline(
        context.feasibility, subject_id=str(subject_id), pm=str(pm)
    ).copy()
    timeline["target_sample_id"] = timeline["target_sample_id"].astype(str)
    subject = context.feasibility.subject_chronology.loc[
        context.feasibility.subject_chronology["subject_id"].astype(str).eq(str(subject_id))
    ]
    if len(subject) != 1:
        raise RuntimeError("Expected one subject chronology row")
    return timeline, subject.iloc[0]


def _aggregate_applied_only(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate adaptation-applied rows PM-within-participant first."""
    applied = results.loc[results["adaptation_applied"]].copy()

    participant_columns = [
        "model", "budget_seconds", "budget_role",
        "strategy", "strategy_role", "subject_id",
    ]
    summary_columns = [
        "model", "budget_seconds", "budget_role",
        "strategy", "strategy_role",
    ]

    participant_rows: list[dict[str, Any]] = []
    for keys, group in applied.groupby(participant_columns, sort=True):
        row = dict(zip(participant_columns, keys))
        row["n_adapted_pm"] = int(group["pm"].nunique())
        for metric in METRICS:
            row[f"delta_{metric}"] = float(
                group[f"delta_{metric}"].mean()
            )
        participant_rows.append(row)

    participant = pd.DataFrame(participant_rows)

    summary_rows: list[dict[str, Any]] = []
    if not participant.empty:
        for keys, group in participant.groupby(
            summary_columns, sort=True
        ):
            row = dict(zip(summary_columns, keys))
            raw_mask = (
                applied["model"].eq(keys[0])
                & applied["budget_seconds"].eq(keys[1])
                & applied["strategy"].eq(keys[3])
            )
            row["participant_pm_rows"] = int(raw_mask.sum())
            row["participants"] = int(
                group["subject_id"].nunique()
            )
            row["adapted_pm_per_participant_mean"] = float(
                group["n_adapted_pm"].mean()
            )
            row["adapted_pm_per_participant_min"] = int(
                group["n_adapted_pm"].min()
            )
            row["adapted_pm_per_participant_max"] = int(
                group["n_adapted_pm"].max()
            )
            for metric in METRICS:
                values = group[f"delta_{metric}"]
                row[f"delta_{metric}_mean"] = float(values.mean())
                row[f"delta_{metric}_median"] = float(values.median())
            summary_rows.append(row)

    return participant, pd.DataFrame(summary_rows)


def run_experiment(context: ThresholdContext) -> dict[str, Any]:
    predictions = _load_prediction_lookup(context)
    detail = context.feasibility_detail.copy()
    detail["subject_id"] = detail["subject_id"].astype(str)

    # Fixed-evaluation readiness is budget-invariant. Use the 0-s feasibility row.
    readiness = detail.loc[detail["budget_seconds"].astype(int).eq(0)].set_index(
        ["subject_id","pm"]
    )
    ready_keys = [
        key for key, row in readiness.iterrows()
        if str(row["fixed_evaluation_ready_min20_both_classes"]).lower() == "true"
    ]
    if len(ready_keys) != 368:
        raise RuntimeError(f"Expected 368 fixed-evaluation-ready participant-PM cells, got {len(ready_keys)}")

    result_rows = []
    for subject_id, pm in ready_keys:
        fold = int(readiness.loc[(subject_id, pm), "outer_fold"])
        timeline, subject = _timeline_extremes(context, subject_id, pm)
        extreme = timeline.loc[timeline["state"].isin(["low","high"])].copy()
        boundary = float(pd.Timestamp(subject["calibration_record_start_utc"]).timestamp()) + 300.0
        evaluation = extreme.loc[
            extreme["absolute_target_epoch_seconds"].to_numpy(dtype=float) > boundary
        ].copy()
        y_eval = _state_to_y(evaluation["state"])
        if len(evaluation) < 20 or set(np.unique(y_eval)) != {0,1}:
            raise RuntimeError("Runtime evaluation differs from feasibility readiness")

        for model in MODELS:
            pred_source = predictions[(model, fold, pm)]
            missing_eval = sorted(set(evaluation["target_sample_id"]) - set(pred_source.index))
            if missing_eval:
                raise RuntimeError(
                    f"{model}/{fold}/{pm}/{subject_id}: missing evaluation predictions"
                )
            p_eval = pred_source.loc[
                evaluation["target_sample_id"], "probability_high"
            ].to_numpy(dtype=float)
            source_y_eval = pred_source.loc[
                evaluation["target_sample_id"], "y_true"
            ].to_numpy(dtype=int)
            if not np.array_equal(source_y_eval, y_eval):
                raise RuntimeError("Stored prediction y_true differs from feasibility labels")
            zero_metrics = _metric_row(y_eval, p_eval, 0.5, subject_id)

            for budget in BUDGETS:
                frow = detail.loc[
                    detail["subject_id"].eq(subject_id)
                    & detail["pm"].astype(str).eq(pm)
                    & detail["budget_seconds"].astype(int).eq(budget)
                ]
                if len(frow) != 1:
                    raise RuntimeError("Missing feasibility participant-PM-budget row")
                frow = frow.iloc[0]
                fully_available = str(frow["budget_fully_available"]).lower() == "true"
                n_low = int(frow["calibration_low"])
                n_high = int(frow["calibration_high"])
                class_eligible = fully_available and n_low >= 2 and n_high >= 2

                cal_group = str(subject["calibration_record_group_id"])
                if budget:
                    calibration = extreme.loc[
                        extreme["record_group_id"].astype(str).eq(cal_group)
                        & (extreme["target_relative_seconds"].to_numpy(dtype=float) > 0.0)
                        & (extreme["target_relative_seconds"].to_numpy(dtype=float) <= float(budget))
                    ].copy()
                else:
                    calibration = extreme.iloc[0:0].copy()

                if int(np.sum(calibration["state"].eq("low"))) != n_low or int(np.sum(calibration["state"].eq("high"))) != n_high:
                    raise RuntimeError("Runtime calibration counts differ from feasibility audit")

                if len(calibration):
                    missing_cal = sorted(set(calibration["target_sample_id"]) - set(pred_source.index))
                    if missing_cal:
                        raise RuntimeError(
                            f"{model}/{fold}/{pm}/{subject_id}: missing calibration predictions"
                        )
                    p_cal = pred_source.loc[
                        calibration["target_sample_id"], "probability_high"
                    ].to_numpy(dtype=float)
                    y_cal = _state_to_y(calibration["state"])
                else:
                    p_cal = np.asarray([], dtype=float)
                    y_cal = np.asarray([], dtype=np.int64)

                for strategy in STRATEGIES:
                    threshold = 0.5
                    applied = False
                    reason = "ineligible_zero_shot_fallback"
                    extras: dict[str, float] = {}
                    if class_eligible:
                        if strategy == "median_midpoint":
                            threshold, applied, reason, extras = _median_midpoint(p_cal, y_cal)
                        else:
                            threshold, applied, reason, extras = _empirical_ba_threshold(p_cal, y_cal)

                    personal = _metric_row(y_eval, p_eval, threshold, subject_id)
                    row = {
                        "model": model,
                        "outer_fold": fold,
                        "pm": pm,
                        "subject_id": subject_id,
                        "budget_seconds": budget,
                        "budget_role": context.config["scientific_contract"]["budget_roles"][str(budget)],
                        "strategy": strategy,
                        "strategy_role": context.config["strategies"][strategy]["role"],
                        "budget_fully_available": fully_available,
                        "calibration_low": n_low,
                        "calibration_high": n_high,
                        "calibration_extreme": n_low + n_high,
                        "calibration_class_eligible": class_eligible,
                        "adaptation_applied": applied,
                        "adaptation_reason": reason,
                        "personalized_threshold": float(threshold),
                        "evaluation_low": int(np.sum(y_eval == 0)),
                        "evaluation_high": int(np.sum(y_eval == 1)),
                        "evaluation_extreme": int(len(y_eval)),
                        "evaluation_sample_hash": stable_hash(
                            sorted(evaluation["target_sample_id"].astype(str).tolist())
                        ),
                        **extras,
                    }
                    for metric in METRICS:
                        row[f"zero_shot_{metric}"] = zero_metrics[metric]
                        row[f"personalized_{metric}"] = personal[metric]
                        row[f"delta_{metric}"] = personal[metric] - zero_metrics[metric]
                    row["zero_shot_roc_auc"] = zero_metrics["roc_auc"]
                    row["zero_shot_pr_auc"] = zero_metrics["pr_auc"]
                    result_rows.append(row)

    results = pd.DataFrame(result_rows)
    expected = 368 * len(MODELS) * len(BUDGETS) * len(STRATEGIES)
    if len(results) != expected:
        raise RuntimeError(f"Expected {expected} result rows, got {len(results)}")
    _write_csv(context.output_dir / "participant_pm_results.csv", results)

    # Participant-first aggregation.
    participant_rows = []
    group_cols = ["model","budget_seconds","budget_role","strategy","strategy_role","subject_id"]
    for keys, group in results.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        row["n_pm"] = int(group["pm"].nunique())
        row["n_class_eligible_pm"] = int(group["calibration_class_eligible"].sum())
        row["n_adaptation_applied_pm"] = int(group["adaptation_applied"].sum())
        for metric in METRICS:
            row[f"zero_shot_{metric}"] = float(group[f"zero_shot_{metric}"].mean())
            row[f"personalized_{metric}"] = float(group[f"personalized_{metric}"].mean())
            row[f"delta_{metric}"] = float(group[f"delta_{metric}"].mean())
        participant_rows.append(row)
    participant = pd.DataFrame(participant_rows)
    _write_csv(context.output_dir / "participant_aggregate.csv", participant)

    summaries = []
    for keys, group in participant.groupby(
        ["model","budget_seconds","budget_role","strategy","strategy_role"], sort=True
    ):
        row = dict(zip(
            ["model","budget_seconds","budget_role","strategy","strategy_role"], keys
        ))
        row["participants"] = int(group["subject_id"].nunique())
        row["participant_pm_rows"] = int(
            len(results.loc[
                results["model"].eq(keys[0])
                & results["budget_seconds"].eq(keys[1])
                & results["strategy"].eq(keys[3])
            ])
        )
        row["class_eligible_participant_pm"] = int(
            results.loc[
                results["model"].eq(keys[0])
                & results["budget_seconds"].eq(keys[1])
                & results["strategy"].eq(keys[3]),
                "calibration_class_eligible",
            ].sum()
        )
        row["adaptation_applied_participant_pm"] = int(
            results.loc[
                results["model"].eq(keys[0])
                & results["budget_seconds"].eq(keys[1])
                & results["strategy"].eq(keys[3]),
                "adaptation_applied",
            ].sum()
        )
        for metric in METRICS:
            row[f"zero_shot_{metric}_mean"] = float(group[f"zero_shot_{metric}"].mean())
            row[f"personalized_{metric}_mean"] = float(group[f"personalized_{metric}"].mean())
            row[f"delta_{metric}_mean"] = float(group[f"delta_{metric}"].mean())
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    _write_csv(context.output_dir / "summary_operational.csv", summary)

    # Applied-only secondary estimand follows the same participant-first
    # aggregation contract as the primary operational estimand.
    applied_participant, applied_summary = _aggregate_applied_only(results)
    _write_csv(
        context.output_dir / "participant_aggregate_applied_only.csv",
        applied_participant,
    )
    _write_csv(
        context.output_dir / "summary_adaptation_applied_only.csv",
        applied_summary,
    )

    # PM-specific operational summary.
    pm_rows = []
    for keys, group in results.groupby(
        ["model","budget_seconds","strategy","pm"], sort=True
    ):
        row = dict(zip(["model","budget_seconds","strategy","pm"], keys))
        row["participant_rows"] = int(len(group))
        row["class_eligible"] = int(group["calibration_class_eligible"].sum())
        row["adaptation_applied"] = int(group["adaptation_applied"].sum())
        for metric in METRICS:
            row[f"delta_{metric}_mean"] = float(group[f"delta_{metric}"].mean())
        pm_rows.append(row)
    _write_csv(context.output_dir / "summary_by_pm.csv", pd.DataFrame(pm_rows))

    # Cluster bootstrap of participant-level operational deltas.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_rows = []
    for keys, group in participant.groupby(
        ["model","budget_seconds","strategy"], sort=True
    ):
        model, budget, strategy = keys
        subjects = group["subject_id"].astype(str).to_numpy()
        if len(subjects) == 0:
            continue
        for metric in ("balanced_accuracy","macro_f1"):
            values = group[f"delta_{metric}"].to_numpy(dtype=float)
            observed = float(np.mean(values))
            samples = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
            n = len(values)
            for i in range(BOOTSTRAP_REPLICATES):
                idx = rng.integers(0, n, size=n)
                samples[i] = float(np.mean(values[idx]))
            boot_rows.append({
                "model": model,
                "budget_seconds": int(budget),
                "strategy": strategy,
                "metric": metric,
                "observed_mean_delta": observed,
                "bootstrap_ci_low": float(np.quantile(samples, 0.025)),
                "bootstrap_ci_high": float(np.quantile(samples, 0.975)),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "resampling_unit": "subject_id",
                "n_subjects": int(n),
            })
    bootstrap = pd.DataFrame(boot_rows)
    _write_csv(context.output_dir / "paired_delta_bootstrap_ci.csv", bootstrap)

    protocol = dict(context.protocol)
    protocol.update({
        "result_status": "confirmatory_complete",
        "threshold_calibration_executed": True,
        "base_model_training_executed": False,
        "base_model_inference_executed": False,
        "fixed_evaluation_ready_participant_pm": 368,
        "participant_pm_result_rows": int(len(results)),
        "participant_aggregate_rows": int(len(participant)),
        "result_hash": _safe_hash_frame(results),
    })
    _atomic_json(context.output_dir / "protocol.json", protocol)

    pooled = {
        "experiment_id": context.config["experiment_id"],
        "protocol_hash": context.protocol["protocol_hash"],
        "result_status": "confirmatory_complete",
        "base_model_training_executed": False,
        "base_model_inference_executed": False,
        "threshold_calibration_executed": True,
        "models": list(MODELS),
        "budgets_seconds": list(BUDGETS),
        "strategies": list(STRATEGIES),
        "fixed_evaluation_ready_participant_pm": 368,
        "participant_pm_result_rows": int(len(results)),
        "primary_budget_seconds": 300,
        "primary_strategy": "median_midpoint",
    }
    _atomic_json(context.output_dir / "pooled_summary.json", pooled)
    return pooled


__all__ = [
    "BUDGETS","MODELS","STRATEGIES","ThresholdContext",
    "load_config","prepare_protocol","write_dry_run","run_experiment",
    "_median_midpoint","_empirical_ba_threshold",
]
