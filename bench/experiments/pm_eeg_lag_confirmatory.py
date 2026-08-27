"""Confirmatory previous-window EEG alignment on canonical 371 features.

The experiment compares two fixed conditions on one matched target cohort:
``X(t) -> y(t)`` and ``X(t-10s) -> y(t)``.  The target timing, fixed outer
folds and fold-local Q3 target transformations are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from bench.features.cogstate_feature_cache import (
    load_feature_cache,
    target_columns,
)
from bench.tasks.target_transforms import FoldLocalQuantileTargetTransform
from model_zoo import build_model


SCHEMA_VERSION = "pm-eeg-lag-confirmatory-v1"
PM_NAMES = (
    "attention",
    "engagement",
    "excitement",
    "stress",
    "relaxation",
    "interest",
    "focus",
)
CONDITIONS = (
    ("lag_0", 0),
    ("lag_minus_10s", -10),
)
FIXED_LABELS = (0, 1, 2)


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_hash(values: Sequence[Any]) -> str:
    return stable_hash([str(value) for value in values])


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


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    if tuple(config.get("pm_names", ())) != PM_NAMES:
        raise ValueError("Confirmatory protocol must contain all seven PM in canonical order")
    condition_ids = tuple(
        str(item["condition_id"]) for item in config.get("conditions", ())
    )
    if condition_ids != tuple(name for name, _ in CONDITIONS):
        raise ValueError("Confirmatory condition IDs are frozen")
    lags = tuple(int(item["lag_seconds"]) for item in config.get("conditions", ()))
    if lags != (0, -10):
        raise ValueError("Confirmatory conditions are frozen at lag 0 and lag -10 seconds")
    if any(int(item["lag_seconds"]) == -20 for item in config["conditions"]):
        raise ValueError("Per-target exploratory lag -20 is forbidden")
    if config.get("evaluation", {}).get("folds") != [1, 2, 3, 4, 5]:
        raise ValueError("Fixed outer folds must be [1, 2, 3, 4, 5]")
    if config["evaluation"].get("group_column") != "subject_id":
        raise ValueError("Outer grouping must use subject_id")
    if config["evaluation"].get("precomputed_fold_column") != "outer_fold":
        raise ValueError("The canonical precomputed outer_fold column is required")
    if config.get("target_transform") != {
        "name": "fold_local_quantile",
        "q": 3,
        "fit_scope": "outer_train_only",
        "duplicates": "drop",
    }:
        raise ValueError("The target transform must remain fold-local outer-train Q3")
    if config.get("model", {}).get("name") != "xgboost":
        raise ValueError("The confirmatory model is frozen at xgboost")
    if int(config["model"].get("seed", -1)) != 42:
        raise ValueError("The confirmatory seed is frozen at 42")
    if int(config["feature_cache_identity"].get("n_features", -1)) != 371:
        raise ValueError("Canonical feature count must be 371")
    return config


def _target_table(path: Path) -> pd.DataFrame:
    columns = ["subject_id", "record_id", *(f"target_{pm}" for pm in PM_NAMES)]
    frame = pd.read_parquet(path, columns=columns)
    if "sample_id" not in frame.columns:
        frame = frame.copy()
        frame.insert(0, "sample_id", frame.index.to_numpy())
    frame = frame.reset_index(drop=True)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Processed target table contains duplicate sample_id")
    return frame


def validate_cache_contract(
    matrix: np.ndarray,
    index: pd.DataFrame,
    feature_names: Sequence[str],
    manifest: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    identity = dict(manifest.get("identity", {}))
    checked = (
        "cache_schema_version",
        "cache_identity_hash",
        "feature_hash",
        "sample_id_universe_hash",
        "raw_preprocessing_hash",
        "rows",
        "n_features",
        "dtype",
    )
    mismatches = {
        key: {"expected": expected.get(key), "actual": identity.get(key)}
        for key in checked
        if identity.get(key) != expected.get(key)
    }
    if mismatches:
        raise ValueError(f"Canonical feature cache identity mismatch: {mismatches}")
    expected_shape = (int(expected["rows"]), int(expected["n_features"]))
    if tuple(matrix.shape) != expected_shape:
        raise ValueError(f"Feature matrix shape {matrix.shape} != {expected_shape}")
    if str(matrix.dtype) != str(expected["dtype"]):
        raise ValueError(f"Feature dtype {matrix.dtype} != {expected['dtype']}")
    if len(index) != expected_shape[0] or len(feature_names) != expected_shape[1]:
        raise ValueError("Feature matrix/index/name counts are inconsistent")
    if index["sample_id"].duplicated().any():
        raise ValueError("Feature index contains duplicate sample_id")
    forbidden = target_columns([*index.columns, *feature_names])
    if forbidden:
        raise ValueError(f"Target/label columns entered feature representation: {forbidden}")
    if not np.isfinite(matrix).all():
        raise ValueError("Canonical feature matrix contains NaN or Inf")
    return identity


def build_previous_window_pairing(
    index: pd.DataFrame,
    *,
    step_seconds: float = 10.0,
    time_column: str = "t_start",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pair each target row with the exact immediately preceding record window."""
    required = {"sample_id", "record_id", "subject_id", "outer_fold", time_column}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"Feature index lacks pairing columns: {missing}")
    if index["sample_id"].duplicated().any():
        raise ValueError("Feature index contains duplicate sample_id")
    times = pd.to_numeric(index[time_column], errors="coerce")
    if not np.isfinite(times.to_numpy(dtype=float)).all():
        raise ValueError(f"{time_column} contains non-finite values")
    records = index["record_id"].astype(str)
    origins = times.groupby(records, sort=False).transform("min")
    units = (times - origins) / float(step_seconds)
    coordinates = np.rint(units.to_numpy(dtype=float)).astype(np.int64)
    residual = np.abs(units.to_numpy(dtype=float) - coordinates)
    if len(residual) and float(residual.max()) > 1e-6:
        raise ValueError(
            f"{time_column} is not on the {step_seconds:g}-second grid; "
            f"max residual={float(residual.max()):.6g}"
        )
    keys = list(zip(records.tolist(), coordinates.tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate (record_id, window_coordinate) keys")
    lookup = {key: position for position, key in enumerate(keys)}
    previous = np.fromiter(
        (lookup.get((record, int(coord - 1)), -1) for record, coord in keys),
        dtype=np.int64,
        count=len(index),
    )
    target_positions = np.flatnonzero(previous >= 0)
    previous_positions = previous[target_positions]
    target = index.iloc[target_positions].reset_index(drop=True)
    lagged = index.iloc[previous_positions].reset_index(drop=True)
    if not target["record_id"].astype(str).eq(lagged["record_id"].astype(str)).all():
        raise RuntimeError("Lag pairing crossed a record_id boundary")
    if not target["subject_id"].astype(str).eq(lagged["subject_id"].astype(str)).all():
        raise RuntimeError("Lag pairing crossed a subject_id boundary")
    if not target["outer_fold"].astype(int).eq(lagged["outer_fold"].astype(int)).all():
        raise RuntimeError("Lag pairing crossed an outer-fold boundary")
    deltas = (
        pd.to_numeric(target[time_column], errors="raise").to_numpy(dtype=float)
        - pd.to_numeric(lagged[time_column], errors="raise").to_numpy(dtype=float)
    )
    if not np.allclose(deltas, float(step_seconds), atol=1e-6, rtol=0.0):
        raise RuntimeError("Lag pairing did not preserve an exact 10-second step")
    pairing = pd.DataFrame({
        "target_sample_id": target["sample_id"].to_numpy(),
        "lag_0_feature_sample_id": target["sample_id"].to_numpy(),
        "lag_minus_10s_feature_sample_id": lagged["sample_id"].to_numpy(),
        "lag_0_feature_position": target_positions,
        "lag_minus_10s_feature_position": previous_positions,
        "source": target.get("source", pd.Series("", index=target.index)).astype(str),
        "subject_id": target["subject_id"].astype(str),
        "record_id": target["record_id"].astype(str),
        "record_group_id": target.get(
            "record_group_id", pd.Series("", index=target.index)
        ).astype(str),
        "outer_fold": target["outer_fold"].astype(int),
        "target_time": pd.to_numeric(target[time_column], errors="raise"),
        "feature_time_lag_minus_10s": pd.to_numeric(
            lagged[time_column], errors="raise"
        ),
    })
    if pairing["target_sample_id"].duplicated().any():
        raise RuntimeError("Matched target sample IDs are not unique")
    first_window_losses = int(index["record_id"].nunique())
    summary = {
        "canonical_feature_rows": int(len(index)),
        "matched_target_rows": int(len(pairing)),
        "lost_without_exact_previous_window": int(len(index) - len(pairing)),
        "first_window_losses": first_window_losses,
        "additional_gap_losses": int(len(index) - len(pairing) - first_window_losses),
        "records": int(pairing["record_id"].nunique()),
        "subjects": int(pairing["subject_id"].nunique()),
        "time_column": time_column,
        "step_seconds": float(step_seconds),
        "cross_record_pairs": 0,
        "cross_subject_pairs": 0,
        "cross_fold_pairs": 0,
        "matched_target_sample_hash": _sample_hash(pairing["target_sample_id"]),
    }
    return pairing, summary


def condition_target_ids(pairing: pd.DataFrame) -> dict[str, np.ndarray]:
    result = {
        name: pairing["target_sample_id"].to_numpy(copy=True)
        for name, _ in CONDITIONS
    }
    reference = result[CONDITIONS[0][0]]
    for values in result.values():
        if not np.array_equal(reference, values):
            raise RuntimeError("Conditions do not use identical target sample IDs")
    return result


def _target_transform_payload(
    transform: FoldLocalQuantileTargetTransform,
    *,
    fold: int,
    pm: str,
    train_sample_ids: Sequence[Any],
) -> dict[str, Any]:
    payload = {
        "outer_fold": int(fold),
        "pm": pm,
        "target_id": f"label_{pm}_q3_candidate",
        "source_continuous_target": f"target_{pm}",
        "fit_scope": "outer_train_only",
        "outer_train_sample_hash": _sample_hash(train_sample_ids),
        **transform.manifest(),
    }
    payload["transform_hash"] = stable_hash(payload)
    return payload


def build_fold_transforms(
    full: pd.DataFrame,
    folds: Sequence[int],
    *,
    transform_factory: Callable[[], FoldLocalQuantileTargetTransform] | None = None,
) -> tuple[dict[tuple[int, str], FoldLocalQuantileTargetTransform], dict[str, Any]]:
    factory = transform_factory or (
        lambda: FoldLocalQuantileTargetTransform(q=3, duplicates="drop")
    )
    transforms: dict[tuple[int, str], FoldLocalQuantileTargetTransform] = {}
    manifests: dict[str, Any] = {}
    for fold in folds:
        train = full["outer_fold"].astype(int).ne(int(fold))
        for pm in PM_NAMES:
            column = f"target_{pm}"
            values = pd.to_numeric(full.loc[train, column], errors="coerce").to_numpy(
                dtype=float
            )
            transform = factory().fit(values)
            if transform.actual_class_count != 3:
                raise ValueError(f"fold {fold} {pm}: Q3 produced fewer than three classes")
            valid = np.isfinite(values)
            train_ids = full.loc[train, "sample_id"].to_numpy()[valid]
            transforms[(int(fold), pm)] = transform
            manifests[f"fold_{int(fold):02d}__{pm}"] = _target_transform_payload(
                transform,
                fold=int(fold),
                pm=pm,
                train_sample_ids=train_ids,
            )
    return transforms, manifests


def build_fold_audit(
    full: pd.DataFrame,
    pairing: pd.DataFrame,
    folds: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        train_subjects = sorted(
            full.loc[full["outer_fold"].astype(int).ne(int(fold)), "subject_id"]
            .astype(str).unique().tolist()
        )
        test_subjects = sorted(
            full.loc[full["outer_fold"].astype(int).eq(int(fold)), "subject_id"]
            .astype(str).unique().tolist()
        )
        overlap = sorted(set(train_subjects) & set(test_subjects))
        matched_train = pairing["outer_fold"].ne(int(fold))
        matched_test = pairing["outer_fold"].eq(int(fold))
        rows.append({
            "outer_fold": int(fold),
            "canonical_train_rows": int(full["outer_fold"].ne(int(fold)).sum()),
            "canonical_test_rows": int(full["outer_fold"].eq(int(fold)).sum()),
            "matched_train_rows": int(matched_train.sum()),
            "matched_test_rows": int(matched_test.sum()),
            "train_subject_count": len(train_subjects),
            "test_subject_count": len(test_subjects),
            "train_subjects": "|".join(train_subjects),
            "test_subjects": "|".join(test_subjects),
            "subject_overlap_count": len(overlap),
            "subject_overlap": "|".join(overlap),
            "matched_test_sample_hash": _sample_hash(
                pairing.loc[matched_test, "target_sample_id"]
            ),
        })
    result = pd.DataFrame(rows)
    if not result["subject_overlap_count"].eq(0).all():
        raise RuntimeError("Outer train/test subject leakage detected")
    return result


@dataclass
class ProtocolContext:
    root: Path
    output_dir: Path
    config: dict[str, Any]
    matrix: np.ndarray
    feature_index: pd.DataFrame
    feature_names: list[str]
    cache_manifest: dict[str, Any]
    cache_identity: dict[str, Any]
    full: pd.DataFrame
    pairing: pd.DataFrame
    pairing_summary: dict[str, Any]
    fold_audit: pd.DataFrame
    transforms: dict[tuple[int, str], FoldLocalQuantileTargetTransform]
    transform_manifests: dict[str, Any]
    protocol: dict[str, Any]
    run_matrix: pd.DataFrame


def prepare_protocol(
    config: Mapping[str, Any],
    *,
    root: str | Path,
    feature_cache_dir: str | Path,
    output_dir: str | Path | None = None,
) -> ProtocolContext:
    root_path = Path(root).resolve()
    cache_path = Path(feature_cache_dir).resolve()
    output = Path(output_dir or config["output_dir"])
    if not output.is_absolute():
        output = root_path / output
    matrix, feature_index, feature_names, cache_manifest = load_feature_cache(cache_path)
    identity = validate_cache_contract(
        matrix,
        feature_index,
        feature_names,
        cache_manifest,
        config["feature_cache_identity"],
    )
    targets = _target_table(root_path / config["data"]["processed_targets"])
    full = feature_index.merge(
        targets,
        on="sample_id",
        how="left",
        suffixes=("", "_target"),
        validate="one_to_one",
    )
    if len(full) != len(feature_index):
        raise RuntimeError("Feature/target join changed the canonical row count")
    for column in ("subject_id", "record_id"):
        target_column = f"{column}_target"
        if not full[column].astype(str).eq(full[target_column].astype(str)).all():
            raise RuntimeError(f"Feature/target {column} identity mismatch")
    folds = [int(value) for value in config["evaluation"]["folds"]]
    if sorted(full["outer_fold"].astype(int).unique().tolist()) != folds:
        raise ValueError("Feature cache outer folds differ from configured folds")
    pairing, pairing_summary = build_previous_window_pairing(
        feature_index,
        step_seconds=float(config["alignment"]["step_seconds"]),
        time_column=str(config["alignment"]["time_column"]),
    )
    condition_target_ids(pairing)
    transforms, transform_manifests = build_fold_transforms(full, folds)
    fold_audit = build_fold_audit(full, pairing, folds)
    scientific_config = {
        key: value for key, value in config.items() if key != "output_dir"
    }
    fixed_fold_hash = stable_hash(
        feature_index[["sample_id", "subject_id", "outer_fold"]]
        .sort_values("sample_id", kind="stable").astype(str).to_dict("records")
    )
    protocol_hash = stable_hash({
        "schema_version": SCHEMA_VERSION,
        "scientific_config": scientific_config,
        "feature_cache_identity": identity,
        "matched_target_sample_hash": pairing_summary["matched_target_sample_hash"],
        "fixed_fold_hash": fixed_fold_hash,
        "target_transform_hashes": {
            key: value["transform_hash"] for key, value in transform_manifests.items()
        },
    })
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "result_status": "confirmatory_preregistered_candidate",
        "training_executed": False,
        "candidate_lags_seconds": [0, -10],
        "candidate_statement": config["preregistration_statement"],
        "interpretation": "fixed causal temporal alignment candidate; not a proven physiological delay",
        "feature_cache_identity": identity,
        "feature_count": len(feature_names),
        "target_list": list(PM_NAMES),
        "fold_ids": folds,
        "model": config["model"],
        "seed": int(config["model"]["seed"]),
        "matched_cohort_count": int(len(pairing)),
        "matched_target_sample_hash": pairing_summary["matched_target_sample_hash"],
        "fixed_fold_hash": fixed_fold_hash,
        "target_transform_policy": config["target_transform"],
        "target_transform_hashes": {
            key: value["transform_hash"] for key, value in transform_manifests.items()
        },
        "git_commit": _git_head(root_path),
        "protocol_hash": protocol_hash,
    }
    specs: list[dict[str, Any]] = []
    for fold in folds:
        for pm in PM_NAMES:
            for condition, lag_seconds in CONDITIONS:
                spec = {
                    "outer_fold": fold,
                    "pm": pm,
                    "target_id": f"label_{pm}_q3_candidate",
                    "condition": condition,
                    "lag_seconds": lag_seconds,
                    "model": "xgboost",
                    "seed": int(config["model"]["seed"]),
                    "q3_transform_hash": transform_manifests[
                        f"fold_{fold:02d}__{pm}"
                    ]["transform_hash"],
                }
                spec_hash = stable_hash({"protocol_hash": protocol_hash, "run_spec": spec})
                spec["specification_hash"] = spec_hash
                spec["run_id"] = (
                    f"fold_{fold:02d}__{pm}__{condition}__{spec_hash[:12]}"
                )
                specs.append(spec)
    run_matrix = pd.DataFrame(specs)
    if len(run_matrix) != 70:
        raise RuntimeError(f"Expected 70 fixed runs, got {len(run_matrix)}")
    return ProtocolContext(
        root=root_path,
        output_dir=output,
        config=dict(config),
        matrix=matrix,
        feature_index=feature_index,
        feature_names=list(feature_names),
        cache_manifest=cache_manifest,
        cache_identity=identity,
        full=full,
        pairing=pairing,
        pairing_summary=pairing_summary,
        fold_audit=fold_audit,
        transforms=transforms,
        transform_manifests=transform_manifests,
        protocol=protocol,
        run_matrix=run_matrix,
    )


def write_dry_run(context: ProtocolContext) -> dict[str, Any]:
    context.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(context.output_dir / "protocol.json", context.protocol)
    _atomic_json(
        context.output_dir / "matched_cohort_summary.json",
        context.pairing_summary,
    )
    _write_csv(context.output_dir / "matched_cohort_by_fold.csv", context.fold_audit)
    _write_csv(context.output_dir / "run_matrix.csv", context.run_matrix)
    _atomic_json(
        context.output_dir / "q3_target_transforms.json",
        context.transform_manifests,
    )
    readme = f"""# PM EEG lag confirmatory v1

This protocol compares `X(t) -> y(t)` with `X(t-10s) -> y(t)` on an exact
matched target cohort. The `-10 s` candidate was fixed before this confirmatory
run from a separate exploratory lag screen. It is a previous-window alignment
candidate, not evidence of a physiological delay.

- protocol hash: `{context.protocol['protocol_hash']}`
- canonical features: `{len(context.feature_names)}`
- canonical rows: `{len(context.feature_index)}`
- matched target rows: `{len(context.pairing)}`
- subjects: `{context.pairing['subject_id'].nunique()}`
- records: `{context.pairing['record_id'].nunique()}`
- planned fits: `{len(context.run_matrix)}`
- training executed by dry-run: `false`
"""
    (context.output_dir / "README.md").write_text(readme, encoding="utf-8")
    try:
        output_reference = context.output_dir.relative_to(context.root).as_posix()
    except ValueError:
        output_reference = str(context.output_dir)
    summary = {
        **context.pairing_summary,
        "feature_cache_rows": int(context.matrix.shape[0]),
        "feature_count": int(context.matrix.shape[1]),
        "feature_dtype": str(context.matrix.dtype),
        "feature_cache_identity_hash": context.cache_identity["cache_identity_hash"],
        "feature_hash": context.cache_identity["feature_hash"],
        "folds": context.fold_audit.to_dict("records"),
        "target_valid_matched_rows": {
            pm: int(
                context.full.set_index("sample_id")
                .loc[context.pairing["target_sample_id"], f"target_{pm}"]
                .notna().sum()
            )
            for pm in PM_NAMES
        },
        "identical_target_ids_between_conditions": True,
        "identical_subject_ids_between_conditions": True,
        "identical_fold_membership_between_conditions": True,
        "target_transform_fit_scope": "outer_train_only",
        "planned_fits": int(len(context.run_matrix)),
        "training_executed": False,
        "protocol_hash": context.protocol["protocol_hash"],
        "output_dir": output_reference,
    }
    _atomic_json(context.output_dir / "dry_run_summary.json", summary)
    return summary


def _participant_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    labels = list(FIXED_LABELS)
    for subject in sorted(np.unique(subject_ids).tolist()):
        mask = subject_ids == subject
        truth = y_true[mask]
        prediction = y_pred[mask]
        rows.append({
            "subject_id": str(subject),
            "n_samples": int(mask.sum()),
            "accuracy": float(accuracy_score(truth, prediction)),
            "macro_f1": float(
                f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)
            ),
            "balanced_accuracy": float(
                recall_score(
                    truth, prediction, labels=labels, average="macro", zero_division=0
                )
            ),
        })
    frame = pd.DataFrame(rows)
    macro = {
        "participant_macro_f1": float(frame["macro_f1"].mean()),
        "participant_macro_balanced_accuracy": float(
            frame["balanced_accuracy"].mean()
        ),
        "participant_macro_accuracy": float(frame["accuracy"].mean()),
    }
    return frame, macro


def _run_directory(context: ProtocolContext, spec: Mapping[str, Any]) -> Path:
    return context.output_dir / "runs" / str(spec["run_id"])


def execute_run(
    context: ProtocolContext,
    spec: Mapping[str, Any],
    *,
    model_builder: Callable[..., Any] = build_model,
) -> dict[str, Any]:
    fold = int(spec["outer_fold"])
    pm = str(spec["pm"])
    condition = str(spec["condition"])
    position_column = (
        "lag_0_feature_position"
        if condition == "lag_0"
        else "lag_minus_10s_feature_position"
    )
    target_lookup = context.full.set_index("sample_id")
    target_ids = context.pairing["target_sample_id"].to_numpy()
    continuous = pd.to_numeric(
        target_lookup.loc[target_ids, f"target_{pm}"], errors="coerce"
    ).to_numpy(dtype=float)
    labels = context.transforms[(fold, pm)].transform(continuous)
    train_mask = context.pairing["outer_fold"].ne(fold).to_numpy() & np.isfinite(labels)
    test_mask = context.pairing["outer_fold"].eq(fold).to_numpy() & np.isfinite(labels)
    train_subjects = set(context.pairing.loc[train_mask, "subject_id"].astype(str))
    test_subjects = set(context.pairing.loc[test_mask, "subject_id"].astype(str))
    if train_subjects & test_subjects:
        raise RuntimeError("Outer subject leakage before model fit")
    x_positions = context.pairing[position_column].to_numpy(dtype=np.int64)
    x_train = np.asarray(context.matrix[x_positions[train_mask]], dtype=np.float32)
    x_test = np.asarray(context.matrix[x_positions[test_mask]], dtype=np.float32)
    y_train = labels[train_mask].astype(np.int64)
    y_test = labels[test_mask].astype(np.int64)
    if sorted(np.unique(y_train).tolist()) != list(FIXED_LABELS):
        raise RuntimeError(f"fold {fold} {pm}: outer train is not class-complete")
    started = time.perf_counter()
    model = model_builder(
        "xgboost",
        "classification",
        (len(context.feature_names),),
        3,
        context.config["model"]["params"],
    )
    model.fit(x_train, y_train)
    prediction = np.asarray(model.predict(x_test), dtype=np.int64)
    elapsed = time.perf_counter() - started
    test_pairing = context.pairing.loc[test_mask].reset_index(drop=True)
    participants, macro = _participant_metrics(
        y_test,
        prediction,
        test_pairing["subject_id"].astype(str).to_numpy(),
    )
    run_dir = _run_directory(context, spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = test_pairing[[
        "target_sample_id", "subject_id", "record_id", "outer_fold"
    ]].copy()
    predictions["feature_sample_id"] = context.pairing.loc[
        test_mask,
        "lag_0_feature_sample_id"
        if condition == "lag_0"
        else "lag_minus_10s_feature_sample_id",
    ].to_numpy()
    predictions["pm"] = pm
    predictions["condition"] = condition
    predictions["lag_seconds"] = int(spec["lag_seconds"])
    predictions["y_true"] = y_test
    predictions["y_pred"] = prediction
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    participants.insert(0, "condition", condition)
    participants.insert(0, "pm", pm)
    participants.insert(0, "outer_fold", fold)
    _write_csv(run_dir / "participant_metrics.csv", participants)
    summary = {
        "status": "complete",
        "result_status": "confirmatory",
        "protocol_hash": context.protocol["protocol_hash"],
        "specification_hash": spec["specification_hash"],
        "run_id": spec["run_id"],
        "outer_fold": fold,
        "pm": pm,
        "condition": condition,
        "lag_seconds": int(spec["lag_seconds"]),
        "target_id": spec["target_id"],
        "q3_transform_hash": spec["q3_transform_hash"],
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_test_participants": int(participants["subject_id"].nunique()),
        "training_time_seconds": float(elapsed),
        **macro,
    }
    _atomic_json(run_dir / "run_summary.json", summary)
    return summary


def _load_resumable_summary(
    context: ProtocolContext,
    spec: Mapping[str, Any],
) -> dict[str, Any] | None:
    run_dir = _run_directory(context, spec)
    path = run_dir / "run_summary.json"
    predictions = run_dir / "predictions.parquet"
    if not path.is_file() or not predictions.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        return None
    if payload.get("protocol_hash") != context.protocol["protocol_hash"]:
        return None
    if payload.get("specification_hash") != spec["specification_hash"]:
        return None
    return payload


def aggregate_results(context: ProtocolContext, summaries: Sequence[Mapping[str, Any]]) -> None:
    results = pd.DataFrame(summaries).sort_values(
        ["outer_fold", "pm", "lag_seconds"], kind="stable"
    )
    _write_csv(context.output_dir / "results_by_fold.csv", results)
    baseline = results.loc[results["condition"].eq("lag_0")].copy()
    lagged = results.loc[results["condition"].eq("lag_minus_10s")].copy()
    paired = baseline.merge(
        lagged,
        on=["outer_fold", "pm", "target_id"],
        suffixes=("_lag0", "_lag_minus_10s"),
        validate="one_to_one",
    )
    for metric in (
        "participant_macro_f1",
        "participant_macro_balanced_accuracy",
        "participant_macro_accuracy",
    ):
        paired[f"delta_{metric}"] = (
            paired[f"{metric}_lag_minus_10s"] - paired[f"{metric}_lag0"]
        )
    _write_csv(context.output_dir / "paired_delta_by_fold.csv", paired)
    pm_rows: list[dict[str, Any]] = []
    for pm, group in paired.groupby("pm", sort=False):
        row: dict[str, Any] = {"pm": pm, "n_folds": int(len(group))}
        for metric in ("participant_macro_f1", "participant_macro_balanced_accuracy"):
            row[f"lag0_{metric}_mean"] = float(group[f"{metric}_lag0"].mean())
            row[f"lag0_{metric}_std"] = float(group[f"{metric}_lag0"].std(ddof=1))
            row[f"lag_minus_10s_{metric}_mean"] = float(
                group[f"{metric}_lag_minus_10s"].mean()
            )
            row[f"lag_minus_10s_{metric}_std"] = float(
                group[f"{metric}_lag_minus_10s"].std(ddof=1)
            )
            row[f"paired_delta_{metric}_mean"] = float(
                group[f"delta_{metric}"].mean()
            )
            row[f"paired_delta_{metric}_std"] = float(
                group[f"delta_{metric}"].std(ddof=1)
            )
        pm_rows.append(row)
    summary_by_pm = pd.DataFrame(pm_rows)
    _write_csv(context.output_dir / "summary_by_pm.csv", summary_by_pm)
    pooled = {
        "n_fold_pm_pairs": int(len(paired)),
        "mean_delta_macro_f1": float(paired["delta_participant_macro_f1"].mean()),
        "std_delta_macro_f1": float(paired["delta_participant_macro_f1"].std(ddof=1)),
        "median_delta_macro_f1": float(paired["delta_participant_macro_f1"].median()),
        "positive_fold_pm_macro_f1": int((paired["delta_participant_macro_f1"] > 0).sum()),
        "mean_delta_balanced_accuracy": float(
            paired["delta_participant_macro_balanced_accuracy"].mean()
        ),
        "std_delta_balanced_accuracy": float(
            paired["delta_participant_macro_balanced_accuracy"].std(ddof=1)
        ),
        "median_delta_balanced_accuracy": float(
            paired["delta_participant_macro_balanced_accuracy"].median()
        ),
        "positive_fold_pm_balanced_accuracy": int(
            (paired["delta_participant_macro_balanced_accuracy"] > 0).sum()
        ),
        "positive_pm_mean_macro_f1": int(
            (summary_by_pm["paired_delta_participant_macro_f1_mean"] > 0).sum()
        ),
        "positive_pm_mean_balanced_accuracy": int(
            (
                summary_by_pm[
                    "paired_delta_participant_macro_balanced_accuracy_mean"
                ]
                > 0
            ).sum()
        ),
    }
    _write_csv(context.output_dir / "pooled_summary.csv", pd.DataFrame([pooled]))
    protocol = dict(context.protocol)
    protocol["training_executed"] = True
    protocol["result_status"] = "confirmatory_complete"
    _atomic_json(context.output_dir / "protocol.json", protocol)


def run_experiment(context: ProtocolContext, *, resume: bool) -> dict[str, int]:
    summaries: list[dict[str, Any]] = []
    reused = 0
    trained = 0
    for spec in context.run_matrix.to_dict("records"):
        existing = _load_resumable_summary(context, spec) if resume else None
        if existing is not None:
            summaries.append(existing)
            reused += 1
            continue
        run_dir = _run_directory(context, spec)
        if run_dir.exists() and not resume:
            raise FileExistsError(
                f"Run directory exists; use --resume after auditing it: {run_dir}"
            )
        summaries.append(execute_run(context, spec))
        trained += 1
    if len(summaries) != 70:
        raise RuntimeError("Full aggregation requires all 70 fixed runs")
    aggregate_results(context, summaries)
    return {"complete": len(summaries), "trained": trained, "reused": reused}


__all__ = [
    "CONDITIONS",
    "PM_NAMES",
    "ProtocolContext",
    "aggregate_results",
    "build_fold_audit",
    "build_fold_transforms",
    "build_previous_window_pairing",
    "condition_target_ids",
    "execute_run",
    "load_config",
    "prepare_protocol",
    "run_experiment",
    "stable_hash",
    "validate_cache_contract",
    "write_dry_run",
]
