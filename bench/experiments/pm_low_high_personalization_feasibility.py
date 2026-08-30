"""Feasibility audit for chronological PM LOW/HIGH personalization.

No model is trained or evaluated here. The audit freezes chronological
calibration semantics before any personalization experiment:

- calibration starts at the earliest logical recording of each outer-test
  participant, ordered by the selected source-record UTC start;
- a calibration budget is elapsed recording time, never "scan forward until
  enough LOW/HIGH labels";
- calibration never stitches multiple logical recordings;
- every budget is compared against one fixed evaluation suffix strictly after
  the maximum 300-second boundary;
- outer-train Q33/Q67 thresholds from the completed LOW/HIGH protocol are
  applied unchanged to each outer-test participant.

The audit reports whether 30 s / 1 min / 2 min / 5 min prefixes contain
LOW/HIGH labels in quantities sufficient to support later supervised
calibration strategies. Middle and missing PM windows are counted explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from bench.experiments.pm_low_high_q3_extremes_confirmatory import (
    FIXED_LAG_SECONDS,
    PM_NAMES,
    ProtocolContext,
    load_config as load_low_high_config,
    prepare_protocol as prepare_low_high_protocol,
    stable_hash,
)


SCHEMA_VERSION = "pm-low-high-personalization-feasibility-v1"
BUDGETS_SECONDS = (0, 30, 60, 120, 300)
MAX_BUDGET_SECONDS = 300
ADVANCED_MODELS = ("xgboost", "lightgbm")


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _sample_hash(values: Sequence[Any]) -> str:
    return stable_hash([str(value) for value in values])


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    if pd.isna(value):
        return []
    return [value]


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    refs = config.get("references", {})
    if refs.get("low_high", {}).get("protocol_hash") != (
        "ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431"
    ):
        raise ValueError("LOW/HIGH reference protocol changed")
    matched = refs.get("matched_model_selection", {})
    if matched.get("protocol_hash") != (
        "e09f28dab2b37321dd665cc55653cfc08a5a29afc38927ee26bc2d2c6cc988e7"
    ):
        raise ValueError("Matched model-selection reference protocol changed")
    if tuple(matched.get("advanced_models", ())) != ADVANCED_MODELS:
        raise ValueError("Personalization candidate models changed")

    contract = config.get("scientific_contract", {})
    if tuple(contract.get("pm_names", ())) != PM_NAMES:
        raise ValueError("All seven PM in canonical order are required")
    if int(contract.get("lag_seconds", 999)) != FIXED_LAG_SECONDS:
        raise ValueError("Feasibility audit is frozen at lag -10 s")
    if tuple(contract.get("calibration_budgets_seconds", ())) != BUDGETS_SECONDS:
        raise ValueError("Calibration budgets must be exactly 0,30,60,120,300 s")
    if int(contract.get("maximum_calibration_budget_seconds", -1)) != 300:
        raise ValueError("Maximum calibration budget must be 300 seconds")
    expected_contract = {
        "calibration_record_policy":
            "earliest_logical_record_by_selected_record_start_utc",
        "calibration_cross_record_policy": "forbidden",
        "calibration_time_origin": "start_of_earliest_logical_record",
        "calibration_interval_rule":
            "0 < target_relative_seconds <= budget_seconds",
        "budget_measurement":
            "elapsed_recording_time_not_extreme_sample_count",
        "fixed_evaluation_policy":
            "all_exact_lag_targets_strictly_after_max_budget_utc_boundary",
        "fixed_evaluation_boundary_rule":
            "absolute_target_utc > earliest_record_start_utc + 300s",
        "reserved_interval_policy":
            "targets_after_current_budget_and_not_after_300s_are_unused",
        "target_transform": "outer_train_q33_q67_extremes",
        "threshold_fit_scope": "outer_train_continuous_complete_cases",
        "middle_policy":
            "exclude_from_binary_training_and_evaluation_but_count_in_feasibility",
        "missing_pm_policy": "count_as_missing_not_middle",
        "outer_group": "subject_id",
        "folds": [1, 2, 3, 4, 5],
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"Scientific contract changed at {key!r}")

    criteria = config.get("feasibility_criteria", {})
    if criteria != {
        "report_any_extreme": True,
        "report_both_extreme_classes": True,
        "report_minimum_per_class": [1, 2, 3],
        "minimum_fixed_evaluation_extremes_descriptive": 20,
        "minimum_fixed_evaluation_both_classes": True,
        "criteria_role":
            "descriptive_only_final_personalization_rules_frozen_after_feasibility_audit",
    }:
        raise ValueError("Feasibility criteria changed")
    forbidden = config.get("forbidden", {})
    if not forbidden or not all(value is True for value in forbidden.values()):
        raise ValueError("All forbidden switches must remain true")
    return config


def _selected_record_start(
    logical_row: pd.Series,
    actual_record_id: str,
) -> pd.Timestamp:
    record_ids = [str(value) for value in _as_list(logical_row["source_record_ids"])]
    starts = _as_list(logical_row["start_datetimes"])
    if len(record_ids) != len(starts):
        raise ValueError(
            f"logical record {logical_row['record_group_id']}: "
            "source_record_ids/start_datetimes length mismatch"
        )
    matches = [
        starts[index]
        for index, record_id in enumerate(record_ids)
        if record_id == str(actual_record_id)
    ]
    if len(matches) != 1 or matches[0] is None:
        raise ValueError(
            f"logical record {logical_row['record_group_id']}: "
            f"cannot resolve UTC start for actual record {actual_record_id}"
        )
    timestamp = pd.to_datetime(matches[0], utc=True, errors="raise")
    if pd.isna(timestamp):
        raise ValueError("Resolved logical-record UTC start is NaT")
    return timestamp


def _selected_record_duration(
    logical_row: pd.Series,
    actual_record_id: str,
) -> float:
    record_ids = [str(value) for value in _as_list(logical_row["source_record_ids"])]
    durations = _as_list(logical_row["duration_seconds"])
    if len(record_ids) != len(durations):
        raise ValueError(
            f"logical record {logical_row['record_group_id']}: "
            "source_record_ids/duration_seconds length mismatch"
        )
    matches = [
        durations[index]
        for index, record_id in enumerate(record_ids)
        if record_id == str(actual_record_id)
    ]
    if len(matches) != 1 or matches[0] is None:
        raise ValueError(
            f"logical record {logical_row['record_group_id']}: "
            f"cannot resolve duration for actual record {actual_record_id}"
        )
    duration = float(matches[0])
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Logical-record duration must be finite and positive")
    return duration


def build_record_chronology(
    feature_index: pd.DataFrame,
    logical_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_index = {
        "sample_id", "source", "subject_id", "record_id",
        "record_group_id", "outer_fold", "t_start",
    }
    missing = sorted(required_index - set(feature_index.columns))
    if missing:
        raise ValueError(f"Feature index missing chronology columns: {missing}")
    required_map = {
        "record_group_id", "subject_id", "source_record_ids",
        "start_datetimes", "duration_seconds", "selected_record_id",
    }
    missing = sorted(required_map - set(logical_map.columns))
    if missing:
        raise ValueError(f"Logical map missing chronology columns: {missing}")
    if logical_map["record_group_id"].astype(str).duplicated().any():
        raise ValueError("Logical map contains duplicate record_group_id")

    map_lookup = logical_map.copy()
    map_lookup["record_group_id"] = map_lookup["record_group_id"].astype(str)
    map_lookup = map_lookup.set_index("record_group_id")

    records = []
    grouped = feature_index.groupby("record_group_id", sort=True, dropna=False)
    for group_id, group in grouped:
        group_id = str(group_id)
        if group_id not in map_lookup.index:
            raise ValueError(f"Feature record_group_id absent from logical map: {group_id}")
        logical = map_lookup.loc[group_id]
        record_ids = group["record_id"].astype(str).unique()
        sources = group["source"].astype(str).unique()
        subjects = group["subject_id"].astype(str).unique()
        folds = group["outer_fold"].astype(int).unique()
        if any(len(values) != 1 for values in (record_ids, sources, subjects, folds)):
            raise ValueError(f"Inconsistent feature metadata in logical record {group_id}")
        record_id = str(record_ids[0])
        subject_id = str(subjects[0])
        if str(logical["subject_id"]) != subject_id:
            raise ValueError(f"Logical-map subject mismatch for {group_id}")
        start_utc = _selected_record_start(logical, record_id)
        source_duration = _selected_record_duration(logical, record_id)
        times = pd.to_numeric(group["t_start"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(times).all():
            raise ValueError(f"Non-finite t_start in {group_id}")
        local_origin = float(np.min(times))
        feature_grid_duration = float(np.max(times) - local_origin + 10.0)
        records.append({
            "record_group_id": group_id,
            "record_id": record_id,
            "source": str(sources[0]),
            "subject_id": subject_id,
            "outer_fold": int(folds[0]),
            "record_start_utc": start_utc.isoformat(),
            "record_start_epoch_seconds": float(start_utc.timestamp()),
            "source_duration_seconds": source_duration,
            "feature_grid_duration_seconds": feature_grid_duration,
            "feature_t_start_origin": local_origin,
            "feature_window_count": int(len(group)),
            "actual_record_is_selected_record": (
                record_id == str(logical["selected_record_id"])
            ),
        })
    chronology = pd.DataFrame(records).sort_values(
        ["subject_id", "record_start_epoch_seconds", "record_group_id"],
        kind="stable",
    ).reset_index(drop=True)
    if chronology.empty:
        raise ValueError("Record chronology is empty")
    chronology["subject_record_order"] = (
        chronology.groupby("subject_id", sort=False).cumcount() + 1
    )
    chronology["is_calibration_record"] = chronology["subject_record_order"].eq(1)

    subject_rows = []
    for subject, group in chronology.groupby("subject_id", sort=True):
        ordered = group.sort_values(
            ["record_start_epoch_seconds", "record_group_id"], kind="stable"
        )
        first = ordered.iloc[0]
        starts = ordered["record_start_epoch_seconds"].to_numpy(dtype=float)
        durations = ordered["source_duration_seconds"].to_numpy(dtype=float)
        ends = starts + durations
        overlaps = 0
        for index in range(1, len(ordered)):
            overlaps += int(starts[index] < ends[index - 1])
        subject_rows.append({
            "subject_id": str(subject),
            "outer_fold": int(first["outer_fold"]),
            "n_logical_records": int(len(ordered)),
            "calibration_record_group_id": str(first["record_group_id"]),
            "calibration_record_id": str(first["record_id"]),
            "calibration_record_start_utc": str(first["record_start_utc"]),
            "calibration_record_duration_seconds": float(
                first["source_duration_seconds"]
            ),
            "calibration_record_feature_grid_duration_seconds": float(
                first["feature_grid_duration_seconds"]
            ),
            "full_30s_available": bool(first["source_duration_seconds"] >= 30),
            "full_60s_available": bool(first["source_duration_seconds"] >= 60),
            "full_120s_available": bool(first["source_duration_seconds"] >= 120),
            "full_300s_available": bool(first["source_duration_seconds"] >= 300),
            "later_record_count": int(max(0, len(ordered) - 1)),
            "chronological_record_overlap_count": int(overlaps),
            "record_order_hash": _sample_hash(
                ordered["record_group_id"].astype(str).tolist()
            ),
        })
    subjects = pd.DataFrame(subject_rows)
    return chronology, subjects


@dataclass
class FeasibilityContext:
    root: Path
    output_dir: Path
    config: dict[str, Any]
    low_high: ProtocolContext
    record_chronology: pd.DataFrame
    subject_chronology: pd.DataFrame
    paired_timeline: pd.DataFrame
    protocol: dict[str, Any]


def _validate_completed_references(
    root: Path,
    config: Mapping[str, Any],
) -> None:
    low = config["references"]["low_high"]
    low_protocol = json.loads(
        (root / low["output_dir"] / "protocol.json").read_text(encoding="utf-8")
    )
    if low_protocol.get("protocol_hash") != low["protocol_hash"]:
        raise RuntimeError("Stored LOW/HIGH reference protocol hash changed")
    if low_protocol.get("result_status") != "confirmatory_complete":
        raise RuntimeError("LOW/HIGH reference is not confirmatory_complete")

    matched = config["references"]["matched_model_selection"]
    matched_protocol = json.loads(
        (root / matched["output_dir"] / "protocol.json").read_text(
            encoding="utf-8"
        )
    )
    if matched_protocol.get("protocol_hash") != matched["protocol_hash"]:
        raise RuntimeError("Stored matched-comparison protocol hash changed")
    if matched_protocol.get("result_status") != "confirmatory_complete":
        raise RuntimeError("Matched-comparison reference is not confirmatory_complete")
    selection = matched_protocol.get(
        "model_selection_for_personalization_result", {}
    )
    if tuple(selection.get("advanced_models", ())) != ADVANCED_MODELS:
        raise RuntimeError(
            "Completed matched comparison no longer selects XGBoost + LightGBM"
        )


def _build_paired_timeline(
    low_high: ProtocolContext,
    chronology: pd.DataFrame,
) -> pd.DataFrame:
    record_lookup = chronology.set_index("record_group_id")
    timeline = low_high.temporal_pairing.copy()
    timeline["record_group_id"] = timeline["record_group_id"].astype(str)
    unknown = sorted(
        set(timeline["record_group_id"]) - set(record_lookup.index.astype(str))
    )
    if unknown:
        raise RuntimeError(f"Temporal pairing has unknown logical records: {unknown}")

    starts = []
    origins = []
    for group_id in timeline["record_group_id"]:
        record = record_lookup.loc[str(group_id)]
        starts.append(float(record["record_start_epoch_seconds"]))
        origins.append(float(record["feature_t_start_origin"]))
    target_times = pd.to_numeric(
        timeline["target_time"], errors="raise"
    ).to_numpy(dtype=float)
    origins = np.asarray(origins, dtype=float)
    relative = target_times - origins
    if not np.isfinite(relative).all() or np.any(relative <= 0):
        raise RuntimeError(
            "Exact-lag PM targets must occur strictly after recording origin"
        )
    absolute_epoch = np.asarray(starts, dtype=float) + relative
    timeline["target_relative_seconds"] = relative
    timeline["absolute_target_epoch_seconds"] = absolute_epoch
    timeline["absolute_target_utc"] = pd.to_datetime(
        absolute_epoch, unit="s", utc=True
    ).astype(str)
    if not np.allclose(
        timeline["target_time"].to_numpy(dtype=float)
        - timeline["feature_time_lag_minus_10s"].to_numpy(dtype=float),
        10.0,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError("Feasibility timeline violated lag -10 pairing")
    return timeline


def prepare_protocol(
    config: Mapping[str, Any],
    *,
    root: str | Path,
    feature_cache_dir: str | Path,
    output_dir: str | Path | None = None,
) -> FeasibilityContext:
    root_path = Path(root).resolve()
    output = Path(output_dir or config["output_dir"])
    if not output.is_absolute():
        output = root_path / output

    _validate_completed_references(root_path, config)
    low = config["references"]["low_high"]
    low_config = load_low_high_config(root_path / low["config"])
    low_high = prepare_low_high_protocol(
        low_config,
        root=root_path,
        feature_cache_dir=feature_cache_dir,
        output_dir=root_path / low["output_dir"],
    )
    if low_high.protocol["protocol_hash"] != low["protocol_hash"]:
        raise RuntimeError("Recomputed LOW/HIGH protocol hash changed")

    logical_map = pd.read_parquet(
        root_path / config["data"]["logical_recording_map"]
    )
    record_chronology, subject_chronology = build_record_chronology(
        low_high.feature_index, logical_map
    )
    if int(subject_chronology["outer_fold"].nunique()) != 5:
        raise RuntimeError("Subject chronology does not span five fixed folds")
    if subject_chronology["subject_id"].duplicated().any():
        raise RuntimeError("Subject chronology contains duplicate subjects")
    paired_timeline = _build_paired_timeline(low_high, record_chronology)

    scientific_payload = {
        "schema_version": SCHEMA_VERSION,
        "references": config["references"],
        "scientific_contract": config["scientific_contract"],
        "feasibility_criteria": config["feasibility_criteria"],
        "forbidden": config["forbidden"],
        "feature_cache_identity": low_high.cache_identity,
        "fixed_fold_hash": low_high.protocol["fixed_fold_hash"],
        "temporal_pairing_hash": low_high.protocol["temporal_pairing_hash"],
        "threshold_hashes": low_high.protocol["threshold_hashes"],
        "record_chronology_hash": stable_hash(
            record_chronology[
                [
                    "record_group_id", "record_id", "subject_id", "outer_fold",
                    "record_start_utc", "source_duration_seconds",
                    "feature_t_start_origin",
                ]
            ].astype(str).to_dict("records")
        ),
        "subject_chronology_hash": stable_hash(
            subject_chronology.astype(str).to_dict("records")
        ),
    }
    protocol_hash = stable_hash(scientific_payload)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "result_status": "preregistered_candidate",
        "audit_executed": False,
        "model_training_executed": False,
        "model_inference_executed": False,
        "git_commit": _git_head(root_path),
        "low_high_reference_protocol_hash": low["protocol_hash"],
        "matched_model_selection_reference_protocol_hash": (
            config["references"]["matched_model_selection"]["protocol_hash"]
        ),
        "personalization_candidate_models": list(ADVANCED_MODELS),
        "feature_cache_identity": low_high.cache_identity,
        "fixed_fold_hash": low_high.protocol["fixed_fold_hash"],
        "temporal_pairing_hash": low_high.protocol["temporal_pairing_hash"],
        "threshold_hashes": low_high.protocol["threshold_hashes"],
        "budgets_seconds": list(BUDGETS_SECONDS),
        "max_budget_seconds": MAX_BUDGET_SECONDS,
        "n_subjects": int(subject_chronology["subject_id"].nunique()),
        "n_logical_records": int(len(record_chronology)),
        "chronology_overlap_subjects": int(
            subject_chronology["chronological_record_overlap_count"].gt(0).sum()
        ),
        "record_chronology_hash": scientific_payload["record_chronology_hash"],
        "subject_chronology_hash": scientific_payload["subject_chronology_hash"],
        "protocol_hash": protocol_hash,
    }
    return FeasibilityContext(
        root=root_path,
        output_dir=output,
        config=dict(config),
        low_high=low_high,
        record_chronology=record_chronology,
        subject_chronology=subject_chronology,
        paired_timeline=paired_timeline,
        protocol=protocol,
    )


def write_dry_run(context: FeasibilityContext) -> dict[str, Any]:
    context.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(context.output_dir / "protocol.json", context.protocol)
    _write_csv(
        context.output_dir / "record_chronology.csv",
        context.record_chronology,
    )
    _write_csv(
        context.output_dir / "subject_chronology.csv",
        context.subject_chronology,
    )
    summary = {
        "experiment_id": context.config["experiment_id"],
        "protocol_hash": context.protocol["protocol_hash"],
        "audit_executed": False,
        "model_training_executed": False,
        "model_inference_executed": False,
        "candidate_models": list(ADVANCED_MODELS),
        "budgets_seconds": list(BUDGETS_SECONDS),
        "max_budget_seconds": MAX_BUDGET_SECONDS,
        "subjects": int(context.protocol["n_subjects"]),
        "logical_records": int(context.protocol["n_logical_records"]),
        "chronology_overlap_subjects": int(
            context.protocol["chronology_overlap_subjects"]
        ),
        "subjects_with_full_300s_calibration_record": int(
            context.subject_chronology["full_300s_available"].sum()
        ),
        "paired_target_rows": int(len(context.paired_timeline)),
        "fixed_lag_seconds": FIXED_LAG_SECONDS,
    }
    _atomic_json(context.output_dir / "dry_run_summary.json", summary)
    readme = f"""# PM LOW/HIGH personalization feasibility v1

No model training or inference is performed.

Chronology:
- calibration record: earliest logical recording by actual selected-record UTC start
- calibration never crosses a logical-record boundary
- budgets: 0, 30, 60, 120, 300 seconds of elapsed recording time
- no scanning forward until LOW/HIGH labels appear
- fixed evaluation: exact-lag targets strictly after +300 s UTC boundary
- middle PM values are counted but excluded from the binary task
- missing PM values are counted separately

References:
- LOW/HIGH protocol: `{context.protocol['low_high_reference_protocol_hash']}`
- matched model-selection protocol: `{context.protocol['matched_model_selection_reference_protocol_hash']}`
- future personalization candidates: `xgboost`, `lightgbm`

Protocol hash: `{context.protocol['protocol_hash']}`
Audit executed by dry-run: `false`
"""
    (context.output_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary


def _categorize(values: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    result = np.full(len(values), "missing", dtype=object)
    finite = np.isfinite(values)
    result[finite & (values <= q_low)] = "low"
    result[finite & (values >= q_high)] = "high"
    result[finite & (values > q_low) & (values < q_high)] = "middle"
    return result


def _subject_pm_timeline(
    context: FeasibilityContext,
    *,
    subject_id: str,
    pm: str,
) -> pd.DataFrame:
    subject = context.subject_chronology.loc[
        context.subject_chronology["subject_id"].astype(str).eq(str(subject_id))
    ]
    if len(subject) != 1:
        raise RuntimeError(f"Expected one chronology row for subject {subject_id}")
    fold = int(subject.iloc[0]["outer_fold"])
    thresholds = context.low_high.transforms[(fold, pm)]

    rows = context.paired_timeline.loc[
        context.paired_timeline["subject_id"].astype(str).eq(str(subject_id))
    ].copy()
    target_lookup = context.low_high.full.set_index("sample_id")
    target_column = f"target_{pm}"
    values = pd.to_numeric(
        target_lookup.loc[rows["target_sample_id"].to_numpy(), target_column],
        errors="coerce",
    ).to_numpy(dtype=float)
    rows["continuous_target"] = values
    rows["state"] = _categorize(values, thresholds.q_low, thresholds.q_high)
    rows["pm"] = pm
    rows["outer_fold"] = fold
    rows["q_low"] = thresholds.q_low
    rows["q_high"] = thresholds.q_high
    return rows


def _state_counts(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    counts = frame["state"].value_counts()
    low = int(counts.get("low", 0))
    high = int(counts.get("high", 0))
    middle = int(counts.get("middle", 0))
    missing = int(counts.get("missing", 0))
    available = low + high + middle
    extreme = low + high
    return {
        f"{prefix}_exact_lag_slots": int(len(frame)),
        f"{prefix}_pm_available": available,
        f"{prefix}_missing_pm": missing,
        f"{prefix}_low": low,
        f"{prefix}_high": high,
        f"{prefix}_middle": middle,
        f"{prefix}_extreme": extreme,
        f"{prefix}_has_any_extreme": bool(extreme >= 1),
        f"{prefix}_has_both_classes": bool(low >= 1 and high >= 1),
        f"{prefix}_min1_each": bool(low >= 1 and high >= 1),
        f"{prefix}_min2_each": bool(low >= 2 and high >= 2),
        f"{prefix}_min3_each": bool(low >= 3 and high >= 3),
        f"{prefix}_extreme_fraction_of_available": (
            float(extreme / available) if available else float("nan")
        ),
    }


def run_audit(context: FeasibilityContext) -> dict[str, Any]:
    subject_lookup = context.subject_chronology.set_index("subject_id")
    detail_rows = []
    timeline_hash_rows = []

    for subject_id in sorted(subject_lookup.index.astype(str)):
        subject = subject_lookup.loc[subject_id]
        calibration_group = str(subject["calibration_record_group_id"])
        calibration_start = float(
            pd.Timestamp(subject["calibration_record_start_utc"]).timestamp()
        )
        calibration_duration = float(
            subject["calibration_record_duration_seconds"]
        )
        fixed_eval_boundary = calibration_start + MAX_BUDGET_SECONDS

        for pm in PM_NAMES:
            timeline = _subject_pm_timeline(
                context, subject_id=subject_id, pm=pm
            ).sort_values(
                ["absolute_target_epoch_seconds", "target_sample_id"],
                kind="stable",
            )
            calibration_record = timeline.loc[
                timeline["record_group_id"].astype(str).eq(calibration_group)
            ].copy()
            fixed_eval = timeline.loc[
                timeline["absolute_target_epoch_seconds"].to_numpy(dtype=float)
                > fixed_eval_boundary
            ].copy()
            evaluation_counts = _state_counts(fixed_eval, "evaluation")
            evaluation_extreme = fixed_eval.loc[
                fixed_eval["state"].isin(["low", "high"])
            ]
            eval_sample_hash = _sample_hash(
                evaluation_extreme["target_sample_id"].tolist()
            )
            evaluation_ready_20 = bool(
                evaluation_counts["evaluation_extreme"] >= 20
                and evaluation_counts["evaluation_has_both_classes"]
            )

            for budget in BUDGETS_SECONDS:
                if budget == 0:
                    calibration = calibration_record.iloc[0:0].copy()
                else:
                    relative = calibration_record[
                        "target_relative_seconds"
                    ].to_numpy(dtype=float)
                    calibration = calibration_record.loc[
                        (relative > 0.0) & (relative <= float(budget))
                    ].copy()

                reserved = timeline.loc[
                    (timeline["absolute_target_epoch_seconds"].to_numpy(dtype=float)
                     > calibration_start + float(budget))
                    & (timeline["absolute_target_epoch_seconds"].to_numpy(dtype=float)
                       <= fixed_eval_boundary)
                ].copy()
                cal_counts = _state_counts(calibration, "calibration")
                reserved_counts = _state_counts(reserved, "reserved")
                budget_fully_available = bool(
                    budget == 0 or calibration_duration >= float(budget)
                )
                actual_elapsed = float(
                    0.0 if budget == 0
                    else min(float(budget), calibration_duration)
                )
                cal_extreme = calibration.loc[
                    calibration["state"].isin(["low", "high"])
                ]

                detail_rows.append({
                    "subject_id": subject_id,
                    "outer_fold": int(subject["outer_fold"]),
                    "pm": pm,
                    "budget_seconds": int(budget),
                    "budget_minutes": float(budget / 60.0),
                    "calibration_record_group_id": calibration_group,
                    "calibration_record_id": str(
                        subject["calibration_record_id"]
                    ),
                    "calibration_record_start_utc": str(
                        subject["calibration_record_start_utc"]
                    ),
                    "calibration_record_duration_seconds": calibration_duration,
                    "budget_fully_available": budget_fully_available,
                    "actual_elapsed_calibration_seconds": actual_elapsed,
                    "fixed_evaluation_boundary_utc": pd.to_datetime(
                        fixed_eval_boundary, unit="s", utc=True
                    ).isoformat(),
                    "fixed_evaluation_ready_min20_both_classes": (
                        evaluation_ready_20
                    ),
                    "calibration_extreme_sample_hash": _sample_hash(
                        cal_extreme["target_sample_id"].tolist()
                    ),
                    "evaluation_extreme_sample_hash": eval_sample_hash,
                    **cal_counts,
                    **reserved_counts,
                    **evaluation_counts,
                })
            timeline_hash_rows.append({
                "subject_id": subject_id,
                "pm": pm,
                "fold": int(subject["outer_fold"]),
                "full_subject_pm_timeline_hash": _sample_hash(
                    timeline["target_sample_id"].tolist()
                ),
                "fixed_evaluation_extreme_sample_hash": eval_sample_hash,
            })

    detail = pd.DataFrame(detail_rows)
    expected = (
        context.subject_chronology["subject_id"].nunique()
        * len(PM_NAMES)
        * len(BUDGETS_SECONDS)
    )
    if len(detail) != expected:
        raise RuntimeError(
            f"Expected {expected} participant-PM-budget rows, got {len(detail)}"
        )
    if detail.duplicated(["subject_id", "pm", "budget_seconds"]).any():
        raise RuntimeError("Duplicate participant-PM-budget feasibility rows")

    summary_rows = []
    for budget, group in detail.groupby("budget_seconds", sort=True):
        summary_rows.append({
            "budget_seconds": int(budget),
            "budget_minutes": float(budget / 60.0),
            "participant_pm_rows": int(len(group)),
            "subjects": int(group["subject_id"].nunique()),
            "fully_available_rows": int(group["budget_fully_available"].sum()),
            "fully_available_subjects": int(
                group.loc[group["budget_fully_available"], "subject_id"].nunique()
            ),
            "has_any_extreme_rows": int(
                group["calibration_has_any_extreme"].sum()
            ),
            "has_both_classes_rows": int(
                group["calibration_has_both_classes"].sum()
            ),
            "min2_each_rows": int(group["calibration_min2_each"].sum()),
            "min3_each_rows": int(group["calibration_min3_each"].sum()),
            "fixed_evaluation_ready_min20_both_classes_rows": int(
                group["fixed_evaluation_ready_min20_both_classes"].sum()
            ),
            "median_calibration_extremes": float(
                group["calibration_extreme"].median()
            ),
            "median_calibration_low": float(group["calibration_low"].median()),
            "median_calibration_high": float(group["calibration_high"].median()),
            "median_calibration_middle": float(
                group["calibration_middle"].median()
            ),
            "median_calibration_missing_pm": float(
                group["calibration_missing_pm"].median()
            ),
            "median_evaluation_extremes": float(
                group["evaluation_extreme"].median()
            ),
        })
    summary = pd.DataFrame(summary_rows)

    pm_summary_rows = []
    for (pm, budget), group in detail.groupby(
        ["pm", "budget_seconds"], sort=True
    ):
        pm_summary_rows.append({
            "pm": str(pm),
            "budget_seconds": int(budget),
            "participant_rows": int(len(group)),
            "fully_available": int(group["budget_fully_available"].sum()),
            "has_any_extreme": int(
                group["calibration_has_any_extreme"].sum()
            ),
            "has_both_classes": int(
                group["calibration_has_both_classes"].sum()
            ),
            "min2_each": int(group["calibration_min2_each"].sum()),
            "min3_each": int(group["calibration_min3_each"].sum()),
            "fixed_eval_ready_min20_both": int(
                group["fixed_evaluation_ready_min20_both_classes"].sum()
            ),
            "median_extreme_count": float(
                group["calibration_extreme"].median()
            ),
            "median_low_count": float(group["calibration_low"].median()),
            "median_high_count": float(group["calibration_high"].median()),
            "median_middle_count": float(
                group["calibration_middle"].median()
            ),
        })
    pm_summary = pd.DataFrame(pm_summary_rows)

    context.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        context.output_dir / "participant_pm_budget_feasibility.csv", detail
    )
    _write_csv(context.output_dir / "summary_by_budget.csv", summary)
    _write_csv(context.output_dir / "summary_by_pm_budget.csv", pm_summary)
    _write_csv(
        context.output_dir / "timeline_hash_audit.csv",
        pd.DataFrame(timeline_hash_rows),
    )

    protocol = dict(context.protocol)
    protocol.update({
        "result_status": "feasibility_audit_complete",
        "audit_executed": True,
        "model_training_executed": False,
        "model_inference_executed": False,
        "participant_pm_budget_rows": int(len(detail)),
        "summary_rows": int(len(summary)),
        "pm_summary_rows": int(len(pm_summary)),
        "detail_hash": stable_hash(
            detail.astype(str).to_dict("records")
        ),
    })
    _atomic_json(context.output_dir / "protocol.json", protocol)

    pooled = {
        "experiment_id": context.config["experiment_id"],
        "protocol_hash": context.protocol["protocol_hash"],
        "result_status": "feasibility_audit_complete",
        "model_training_executed": False,
        "model_inference_executed": False,
        "subjects": int(context.subject_chronology["subject_id"].nunique()),
        "participant_pm_budget_rows": int(len(detail)),
        "budgets_seconds": list(BUDGETS_SECONDS),
        "candidate_models": list(ADVANCED_MODELS),
        "summary_by_budget": summary.to_dict("records"),
    }
    _atomic_json(context.output_dir / "pooled_summary.json", pooled)
    return pooled


__all__ = [
    "BUDGETS_SECONDS",
    "FeasibilityContext",
    "build_record_chronology",
    "load_config",
    "prepare_protocol",
    "run_audit",
    "write_dry_run",
]
