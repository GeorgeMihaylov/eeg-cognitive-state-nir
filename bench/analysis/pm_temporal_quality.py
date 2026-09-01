"""Causal temporal-quality diagnostics for canonical Performance Metrics.

This module intentionally lives in the analysis layer.  It never mutates the
canonical target registry or the processed Parquet and it does not train an
EEG model.  Every temporal transform is record-local, causal, deterministic,
and preserves missing values.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

from bench.datasets.logical_recordings import ensure_record_group_ids
from bench.tasks.target_registry import PM_METRICS, get_target_spec
from bench.tasks.target_transforms import (
    build_fold_local_target_transform,
    build_target_transform_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_SCHEMA_VERSION = "pm-temporal-quality-v1"
VARIANT_ORDER = (
    "baseline_raw",
    "causal_median_w3",
    "causal_ema_a05",
    "causal_hampel_w5_k3",
)
TARGET_COLUMNS = tuple(f"target_{metric}" for metric in PM_METRICS)
IDENTITY_COLUMNS = (
    "source",
    "subject_id",
    "record_id",
    "record_group_id",
    "sample_id",
    "t_start",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else REPO_ROOT / resolved


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = _repo_path(path)
    document = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("PM temporal-quality config must be a JSON object")
    required = {
        "schema_version",
        "experiment_id",
        "result_status",
        "data",
        "pm_targets",
        "temporal",
        "variants",
        "folds",
        "behavioral_audit",
        "downstream",
        "output_dir",
    }
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"PM temporal-quality config is missing: {missing}")
    if document["schema_version"] != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("Unsupported PM temporal-quality schema_version")
    if document["result_status"] not in {
        "final", "baseline", "smoke", "diagnostic", "confirmatory"
    }:
        raise ValueError("Unsupported PM temporal-quality result_status")
    if tuple(document["pm_targets"].keys()) != PM_METRICS:
        raise ValueError(f"pm_targets must follow canonical order {PM_METRICS}")
    if tuple(document["variants"].keys()) != VARIANT_ORDER:
        raise ValueError(f"variants must follow fixed order {VARIANT_ORDER}")
    expected_variants = {
        "baseline_raw": {"method": "identity"},
        "causal_median_w3": {"method": "trailing_median", "window": 3},
        "causal_ema_a05": {"method": "causal_ema", "alpha": 0.5},
        "causal_hampel_w5_k3": {
            "method": "causal_hampel",
            "window": 5,
            "threshold": 3.0,
            "mad_scale": 1.4826,
        },
    }
    if document["variants"] != expected_variants:
        raise ValueError("The four preregistered PM variants must not be changed")
    temporal = document["temporal"]
    if temporal.get("missing_policy") != "preserve_nan_and_reset_state":
        raise ValueError("Missing PM must remain NaN and reset causal state")
    if float(temporal.get("expected_step_seconds", -1)) != 10.0:
        raise ValueError("Canonical PM step must be 10 seconds")
    if float(temporal.get("max_gap_seconds", -1)) != 10.01:
        raise ValueError("Configured temporal gap limit must be 10.01 seconds")
    return document


def prepare_pm_frame(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    required = {
        "source",
        "subject_id",
        "record_id",
        "t_start",
        *config["pm_targets"].values(),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Processed PM table is missing columns: {missing}")
    result = ensure_record_group_ids(frame)
    if "sample_id" not in result:
        result.insert(0, "sample_id", result.index.to_numpy(dtype=np.int64))
    if result["sample_id"].isna().any() or result["sample_id"].duplicated().any():
        raise ValueError("sample_id must be complete and unique")
    for column in config["pm_targets"].values():
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["t_start"] = pd.to_numeric(result["t_start"], errors="coerce")
    if result["t_start"].isna().any():
        raise ValueError("t_start must be finite")
    group_keys = list(config["temporal"]["group_keys"])
    expected = {"source", "subject_id", "record_group_id"}
    if not expected.issubset(group_keys):
        raise ValueError(f"Temporal groups must include {sorted(expected)}")
    missing_groups = sorted(set(group_keys) - set(result.columns))
    if missing_groups:
        raise ValueError(f"Temporal group columns are missing: {missing_groups}")
    return result.sort_values(
        [*group_keys, "t_start", "sample_id"], kind="stable"
    ).reset_index(drop=True)


def _split_group_indices(
    frame: pd.DataFrame,
    *,
    group_keys: Sequence[str],
    max_gap_seconds: float,
) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for _, group in frame.groupby(list(group_keys), sort=True, observed=True):
        indices = group.index.to_numpy(dtype=np.int64)
        times = group["t_start"].to_numpy(dtype=float)
        if not len(indices):
            continue
        starts = [0]
        deltas = np.diff(times)
        starts.extend((np.flatnonzero((deltas <= 0) | (deltas > max_gap_seconds)) + 1).tolist())
        starts.append(len(indices))
        for start, end in zip(starts[:-1], starts[1:]):
            segments.append(indices[start:end])
    return segments


def causal_transform_1d(
    values: np.ndarray,
    *,
    method: str,
    window: int | None = None,
    alpha: float | None = None,
    threshold: float | None = None,
    mad_scale: float = 1.4826,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Transform one already contiguous record segment without future values."""
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1:
        raise ValueError(f"Expected one-dimensional PM series, got {raw.shape}")
    output = np.full(raw.shape, np.nan, dtype=float)
    local_median = np.full(raw.shape, np.nan, dtype=float)
    local_mad = np.full(raw.shape, np.nan, dtype=float)
    deviation = np.full(raw.shape, np.nan, dtype=float)
    outlier = np.zeros(raw.shape, dtype=bool)
    history: list[float] = []
    ema_state: float | None = None
    for index, value in enumerate(raw):
        if not np.isfinite(value):
            history.clear()
            ema_state = None
            continue
        if method == "identity":
            output[index] = value
            continue
        if method == "trailing_median":
            if window is None or window < 1:
                raise ValueError("Trailing median requires a positive window")
            history.append(float(value))
            history = history[-window:]
            output[index] = float(np.median(history))
            continue
        if method == "causal_ema":
            if alpha is None or not 0 < alpha <= 1:
                raise ValueError("Causal EMA alpha must be in (0, 1]")
            ema_state = (
                float(value)
                if ema_state is None
                else float(alpha * value + (1.0 - alpha) * ema_state)
            )
            output[index] = ema_state
            continue
        if method == "causal_hampel":
            if window is None or window < 1 or threshold is None or threshold <= 0:
                raise ValueError("Causal Hampel requires positive window and threshold")
            history.append(float(value))
            history = history[-window:]
            median = float(np.median(history))
            mad = float(np.median(np.abs(np.asarray(history) - median)))
            delta = abs(float(value) - median)
            # With zero MAD there is no estimable robust scale.  The conservative
            # policy is not to flag a point rather than labeling every change after
            # a constant run as an outlier.
            flagged = bool(mad > 0 and delta > threshold * mad_scale * mad)
            local_median[index] = median
            local_mad[index] = mad
            deviation[index] = delta
            outlier[index] = flagged
            output[index] = median if flagged else float(value)
            continue
        raise ValueError(f"Unknown causal PM method {method!r}")
    return output, {
        "local_median": local_median,
        "local_mad": local_mad,
        "deviation": deviation,
        "outlier_flag": outlier,
    }


@dataclass(frozen=True)
class VariantResult:
    values: dict[str, dict[str, np.ndarray]]
    outlier_points: pd.DataFrame
    segments: tuple[np.ndarray, ...]


def build_variants(frame: pd.DataFrame, config: Mapping[str, Any]) -> VariantResult:
    group_keys = list(config["temporal"]["group_keys"])
    segments = tuple(
        _split_group_indices(
            frame,
            group_keys=group_keys,
            max_gap_seconds=float(config["temporal"]["max_gap_seconds"]),
        )
    )
    result: dict[str, dict[str, np.ndarray]] = {
        metric: {} for metric in PM_METRICS
    }
    outlier_frames: list[pd.DataFrame] = []
    for metric, column in config["pm_targets"].items():
        raw = frame[column].to_numpy(dtype=float)
        for variant, definition in config["variants"].items():
            transformed = np.full(raw.shape, np.nan, dtype=float)
            diagnostics = {
                "local_median": np.full(raw.shape, np.nan),
                "local_mad": np.full(raw.shape, np.nan),
                "deviation": np.full(raw.shape, np.nan),
                "outlier_flag": np.zeros(raw.shape, dtype=bool),
            }
            for segment in segments:
                values, local = causal_transform_1d(
                    raw[segment],
                    method=str(definition["method"]),
                    window=definition.get("window"),
                    alpha=definition.get("alpha"),
                    threshold=definition.get("threshold"),
                    mad_scale=float(definition.get("mad_scale", 1.4826)),
                )
                transformed[segment] = values
                for key in diagnostics:
                    diagnostics[key][segment] = local[key]
            if not np.array_equal(np.isnan(raw), np.isnan(transformed)):
                raise RuntimeError(f"Variant {variant} changed missingness for {metric}")
            result[metric][variant] = transformed
            if variant == "causal_hampel_w5_k3":
                valid = np.isfinite(raw)
                points = frame.loc[valid, list(IDENTITY_COLUMNS)].copy()
                points.insert(0, "pm", metric)
                points["raw_value"] = raw[valid]
                points["local_median"] = diagnostics["local_median"][valid]
                points["local_MAD"] = diagnostics["local_mad"][valid]
                points["deviation"] = diagnostics["deviation"][valid]
                points["outlier_flag"] = diagnostics["outlier_flag"][valid]
                points["replacement_value"] = transformed[valid]
                outlier_frames.append(points)
    return VariantResult(
        values=result,
        outlier_points=pd.concat(outlier_frames, ignore_index=True),
        segments=segments,
    )


def _finite_correlation(x: np.ndarray, y: np.ndarray, *, rank: bool = False) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    left, right = x[valid], y[valid]
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    result = spearmanr(left, right).statistic if rank else pearsonr(left, right).statistic
    return float(result)


def _adjacent_values(values: np.ndarray, segments: Iterable[np.ndarray]) -> np.ndarray:
    differences: list[np.ndarray] = []
    for segment in segments:
        current = values[segment]
        valid = np.isfinite(current[1:]) & np.isfinite(current[:-1])
        if valid.any():
            differences.append(np.abs(np.diff(current)[valid]))
    return np.concatenate(differences) if differences else np.asarray([], dtype=float)


def _lag_one_pairs(values: np.ndarray, segments: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    previous: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for segment in segments:
        sequence = values[segment]
        valid = np.isfinite(sequence[1:]) & np.isfinite(sequence[:-1])
        if valid.any():
            previous.append(sequence[:-1][valid])
            current.append(sequence[1:][valid])
    if not previous:
        empty = np.asarray([], dtype=float)
        return empty, empty
    return np.concatenate(previous), np.concatenate(current)


def _series_statistics(
    values: np.ndarray,
    baseline: np.ndarray,
    segments: Iterable[np.ndarray],
) -> dict[str, Any]:
    valid = np.isfinite(values)
    available = values[valid]
    differences = _adjacent_values(values, segments)
    previous, current = _lag_one_pairs(values, segments)
    paired = valid & np.isfinite(baseline)
    error = values[paired] - baseline[paired]
    return {
        "n_available": int(valid.sum()),
        "n_missing": int((~valid).sum()),
        "mean": float(np.mean(available)) if len(available) else float("nan"),
        "std": float(np.std(available, ddof=1)) if len(available) > 1 else float("nan"),
        "median": float(np.median(available)) if len(available) else float("nan"),
        "iqr": float(np.quantile(available, 0.75) - np.quantile(available, 0.25)) if len(available) else float("nan"),
        "mean_absolute_first_difference": float(np.mean(differences)) if len(differences) else float("nan"),
        "median_absolute_first_difference": float(np.median(differences)) if len(differences) else float("nan"),
        "p95_absolute_first_difference": float(np.quantile(differences, 0.95)) if len(differences) else float("nan"),
        "lag1_autocorrelation": _finite_correlation(previous, current),
        "mae_vs_baseline": float(np.mean(np.abs(error))) if len(error) else float("nan"),
        "rmse_vs_baseline": float(np.sqrt(np.mean(np.square(error)))) if len(error) else float("nan"),
        "pearson_vs_baseline": _finite_correlation(values, baseline),
        "spearman_vs_baseline": _finite_correlation(values, baseline, rank=True),
        "changed_value_fraction": float(np.mean(~np.isclose(values[paired], baseline[paired], rtol=0.0, atol=1e-12))) if paired.any() else float("nan"),
        "adjacent_pair_count": int(len(differences)),
    }


def summarize_variants(
    frame: pd.DataFrame,
    variants: VariantResult,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []
    for metric in PM_METRICS:
        baseline = variants.values[metric]["baseline_raw"]
        for variant in VARIANT_ORDER:
            stats = _series_statistics(
                variants.values[metric][variant], baseline, variants.segments
            )
            rows.append({"pm": metric, "variant": variant, **stats})
            subject_stats: list[dict[str, Any]] = []
            for subject in sorted(frame["subject_id"].astype(str).unique()):
                subject_indices = frame.index[
                    frame["subject_id"].astype(str).eq(subject)
                ].to_numpy(dtype=np.int64)
                subject_segments = tuple(
                    segment
                    for segment in variants.segments
                    if len(segment)
                    and str(frame.at[int(segment[0]), "subject_id"]) == subject
                )
                subject_values = np.full_like(variants.values[metric][variant], np.nan)
                subject_baseline = np.full_like(baseline, np.nan)
                subject_values[subject_indices] = variants.values[metric][variant][subject_indices]
                subject_baseline[subject_indices] = baseline[subject_indices]
                subject_stats.append(
                    _series_statistics(
                        subject_values, subject_baseline, subject_segments
                    )
                    | {"subject_id": subject}
                )
            numeric_keys = [
                key for key in subject_stats[0]
                if key not in {"subject_id", "n_available", "n_missing", "adjacent_pair_count"}
            ]
            macro = {
                key: float(np.nanmean([row[key] for row in subject_stats]))
                for key in numeric_keys
            }
            macro_rows.append(
                {
                    "pm": metric,
                    "variant": variant,
                    "participants_total": int(len(subject_stats)),
                    "participants_available": int(sum(row["n_available"] > 0 for row in subject_stats)),
                    **macro,
                }
            )
        points = variants.outlier_points.loc[variants.outlier_points["pm"].eq(metric)]
        flags = points["outlier_flag"].astype(bool)
        outlier_rows.append(
            {
                "pm": metric,
                "n_available": int(len(points)),
                "outlier_count": int(flags.sum()),
                "outlier_fraction": float(flags.mean()) if len(points) else float("nan"),
                "replacement_count": int(flags.sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(macro_rows), pd.DataFrame(outlier_rows)


def _normalise_fold(value: Any) -> int:
    text = str(value).strip().lower().replace("fold_", "")
    return int(text)


def _agreement_kappa(left: np.ndarray, right: np.ndarray) -> float:
    labels = np.unique(np.concatenate([left, right]))
    if len(labels) < 2:
        return 1.0 if np.array_equal(left, right) else float("nan")
    return float(cohen_kappa_score(left, right, labels=labels))


def load_fixed_subject_folds(reference_path: Path, frame: pd.DataFrame) -> dict[str, int]:
    reference = pd.read_parquet(reference_path, columns=["subject_id", "fold"])
    reference["subject_id"] = reference["subject_id"].astype(str)
    counts = reference.groupby("subject_id", sort=True)["fold"].nunique()
    if not counts.eq(1).all():
        raise ValueError("Canonical reference assigns a subject to multiple folds")
    mapping = {
        str(row.subject_id): _normalise_fold(row.fold)
        for row in reference.drop_duplicates("subject_id").itertuples()
    }
    available_subjects = set(
        frame.loc[frame[list(TARGET_COLUMNS)].notna().any(axis=1), "subject_id"].astype(str)
    )
    if available_subjects != set(mapping):
        raise ValueError("Fixed-fold subject universe differs from PM target universe")
    if set(mapping.values()) != {1, 2, 3, 4, 5}:
        raise ValueError("Fixed folds must be exactly 1..5")
    return mapping


def calculate_q3_stability(
    frame: pd.DataFrame,
    variants: VariantResult,
    fold_by_subject: Mapping[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assigned = frame["subject_id"].astype(str).map(fold_by_subject)
    if frame[list(TARGET_COLUMNS)].notna().any(axis=1).to_numpy()[assigned.isna()].any():
        raise ValueError("At least one available PM row has no fixed outer fold")
    threshold_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for metric in PM_METRICS:
        target_spec = get_target_spec(f"pm_{metric}_q3_fold_local")
        baseline = variants.values[metric]["baseline_raw"]
        for fold in range(1, 6):
            train_subject = assigned.to_numpy(dtype=float) != fold
            test_subject = assigned.to_numpy(dtype=float) == fold
            baseline_train = train_subject & np.isfinite(baseline)
            baseline_test = test_subject & np.isfinite(baseline)
            baseline_transform = build_fold_local_target_transform(target_spec).fit(
                baseline[baseline_train]
            )
            baseline_test_classes = baseline_transform.transform(baseline[baseline_test]).astype(int)
            baseline_test_ids = frame.loc[baseline_test, "sample_id"].to_numpy()
            for variant in VARIANT_ORDER:
                values = variants.values[metric][variant]
                if not np.array_equal(np.isfinite(values), np.isfinite(baseline)):
                    raise RuntimeError("PM variants must preserve the sample universe")
                train = train_subject & np.isfinite(values)
                test = test_subject & np.isfinite(values)
                transform = build_fold_local_target_transform(target_spec).fit(values[train])
                train_classes = transform.transform(values[train]).astype(int)
                test_classes = transform.transform(values[test]).astype(int)
                manifest = build_target_transform_manifest(
                    target_spec,
                    transform,
                    outer_fold=fold,
                    outer_train_sample_ids=frame.loc[train, "sample_id"].to_numpy(),
                    outer_train_targets=values[train],
                )
                boundaries = manifest["boundaries"]
                threshold_rows.append(
                    {
                        "pm": metric,
                        "variant": variant,
                        "outer_fold": fold,
                        "q1": float(boundaries[1]),
                        "q2": float(boundaries[2]),
                        "actual_class_count": int(manifest["actual_class_count"]),
                        "transform_hash": manifest["transform_hash"],
                        "train_class_counts": json.dumps(
                            {str(i): int(np.sum(train_classes == i)) for i in range(3)},
                            sort_keys=True,
                        ),
                        "test_class_counts": json.dumps(
                            {str(i): int(np.sum(test_classes == i)) for i in range(3)},
                            sort_keys=True,
                        ),
                    }
                )
                test_ids = frame.loc[test, "sample_id"].to_numpy()
                if not np.array_equal(test_ids, baseline_test_ids):
                    raise RuntimeError("Q3 variants changed outer-test sample IDs")
                transitions = np.zeros((3, 3), dtype=np.int64)
                np.add.at(transitions, (baseline_test_classes, test_classes), 1)
                changed = baseline_test_classes != test_classes
                stability_rows.append(
                    {
                        "pm": metric,
                        "variant": variant,
                        "outer_fold": fold,
                        "sample_count": int(len(test_classes)),
                        "class_change_fraction": float(np.mean(changed)),
                        "agreement_accuracy": float(np.mean(~changed)),
                        "cohen_kappa": _agreement_kappa(
                            baseline_test_classes, test_classes
                        ),
                        "low_to_medium": int(transitions[0, 1]),
                        "low_to_high": int(transitions[0, 2]),
                        "medium_to_low": int(transitions[1, 0]),
                        "medium_to_high": int(transitions[1, 2]),
                        "high_to_medium": int(transitions[2, 1]),
                        "high_to_low": int(transitions[2, 0]),
                    }
                )
    return pd.DataFrame(threshold_rows), pd.DataFrame(stability_rows)


def calculate_lag_diagnostics(
    variants: VariantResult,
    *,
    expected_step_seconds: float,
    max_lag_windows: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in PM_METRICS:
        raw = variants.values[metric]["baseline_raw"]
        for variant in VARIANT_ORDER[1:]:
            transformed = variants.values[metric][variant]
            correlations: dict[int, float] = {}
            pair_counts: dict[int, int] = {}
            for lag in range(-max_lag_windows, max_lag_windows + 1):
                left_parts: list[np.ndarray] = []
                right_parts: list[np.ndarray] = []
                for segment in variants.segments:
                    x, y = raw[segment], transformed[segment]
                    if lag > 0:
                        x, y = x[:-lag], y[lag:]
                    elif lag < 0:
                        offset = -lag
                        x, y = x[offset:], y[:-offset]
                    valid = np.isfinite(x) & np.isfinite(y)
                    if valid.any():
                        left_parts.append(x[valid])
                        right_parts.append(y[valid])
                left = np.concatenate(left_parts) if left_parts else np.asarray([])
                right = np.concatenate(right_parts) if right_parts else np.asarray([])
                correlations[lag] = _finite_correlation(left, right)
                pair_counts[lag] = int(len(left))
            finite_lags = [lag for lag, value in correlations.items() if np.isfinite(value)]
            best_lag = max(finite_lags, key=lambda lag: (correlations[lag], -abs(lag), -lag))
            theoretical = (
                (1.0 - 0.5) / 0.5
                if variant == "causal_ema_a05"
                else float("nan")
            )
            rows.append(
                {
                    "pm": metric,
                    "variant": variant,
                    "lag_convention": "positive means transformed series is delayed vs raw",
                    "best_lag_windows": int(best_lag),
                    "best_lag_seconds": float(best_lag * expected_step_seconds),
                    "correlation_at_best_lag": correlations[best_lag],
                    "correlation_at_zero_lag": correlations[0],
                    "theoretical_low_frequency_delay_windows": theoretical,
                    "theoretical_low_frequency_delay_seconds": theoretical * expected_step_seconds if np.isfinite(theoretical) else float("nan"),
                    "correlations_by_lag": json.dumps(correlations, sort_keys=True),
                    "pairs_by_lag": json.dumps(pair_counts, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def _safe_examples(values: pd.Series, limit: int = 3) -> str:
    examples = [str(value) for value in values.dropna().astype(str).unique()[:limit]]
    return json.dumps(examples, ensure_ascii=False)


def audit_behavioral_sources(
    raw_root: Path,
    raw_index_path: Path,
    pm_frame: pd.DataFrame,
    *,
    smoke_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    inventory_rows: list[dict[str, Any]] = []
    feasibility_rows: list[dict[str, Any]] = []
    gpn_root = raw_root / "gpn_data"
    json_paths = sorted(gpn_root.rglob("*.json"))
    json_keys: dict[str, list[Any]] = {}
    marker_items = 0
    for path in json_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, value in payload.items():
            json_keys.setdefault(key, []).append(value)
        marker_items += len(payload.get("Markers") or [])
    for key in sorted(json_keys):
        values = pd.Series([value for value in json_keys[key] if not isinstance(value, (dict, list))], dtype=object)
        inventory_rows.append(
            {
                "source": "gpn_data",
                "file_type": "emotiv_json_metadata",
                "file_count": len(json_paths),
                "row_count": len(json_paths),
                "field": key,
                "example_values": _safe_examples(values),
                "semantic_class": "event_context" if key == "Markers" else "metadata",
                "is_behavioral_measurement": False,
            }
        )
    marker_paths = sorted(gpn_root.rglob("*intervalMarker*"))
    marker_frames: list[pd.DataFrame] = []
    for path in marker_paths:
        compression = "bz2" if path.suffix.lower() == ".bz2" else None
        frame = pd.read_csv(path, compression=compression)
        frame["_path"] = path.as_posix()
        marker_frames.append(frame)
    markers = pd.concat(marker_frames, ignore_index=True) if marker_frames else pd.DataFrame()
    for field in markers.columns.drop("_path", errors="ignore"):
        inventory_rows.append(
            {
                "source": "gpn_data",
                "file_type": "interval_marker",
                "file_count": len(marker_paths),
                "row_count": int(len(markers)),
                "field": field,
                "example_values": _safe_examples(markers[field]),
                "semantic_class": "timestamp" if field in {"timestamp", "latency", "duration"} else "event_context",
                "is_behavioral_measurement": False,
            }
        )
    feasibility_rows.extend(
        [
            {
                "source": "gpn_data",
                "field": "intervalMarker.*",
                "has_timestamp": True,
                "timestamp_semantics": "absolute timestamp plus record-relative latency",
                "measurement_level": "event_or_interval",
                "record_link": "filename/JSON record",
                "window_alignment": "technically possible for two non-empty files",
                "actual_behavioral_metric": False,
                "reason": "marker type/value semantics are not externally validated as outcomes",
            },
            {
                "source": "gpn_data",
                "field": "JSON.Markers",
                "has_timestamp": True,
                "timestamp_semantics": "marker objects when present",
                "measurement_level": "event_or_interval",
                "record_link": "JSON sidecar",
                "window_alignment": "technically possible",
                "actual_behavioral_metric": False,
                "reason": "event/context annotations are not automatically behavioral outcomes",
            },
        ]
    )

    annotation_root = raw_root / "Old_EEG" / "experiment-1" / "annotations"
    annotation_paths = sorted(annotation_root.rglob("*.csv"))
    annotation_frames: list[pd.DataFrame] = []
    for path in annotation_paths:
        frame = pd.read_csv(path)
        frame["_path"] = path.as_posix()
        annotation_frames.append(frame)
    annotations = pd.concat(annotation_frames, ignore_index=True, sort=False) if annotation_frames else pd.DataFrame()
    for field in annotations.columns.drop("_path", errors="ignore"):
        is_metric = field in {"Time Spent (Seconds)", "Correct Answer"}
        semantic = (
            "behavioral_measurement"
            if is_metric
            else "timestamp"
            if "Answered At" in field
            else "event_context"
            if field in {"Lesson Title", "Slide Type", "Lesson Number", "Slide Number"}
            else "identifier"
        )
        inventory_rows.append(
            {
                "source": "Old_EEG",
                "file_type": "course_annotation",
                "file_count": len(annotation_paths),
                "row_count": int(len(annotations)),
                "field": field,
                "example_values": _safe_examples(annotations[field]),
                "semantic_class": semantic,
                "is_behavioral_measurement": is_metric,
            }
        )

    aligned_rows: list[dict[str, Any]] = []
    match_summary = {"events": 0, "unique_matches": 0, "ambiguous_matches": 0, "unmatched": 0}
    if len(annotations) and raw_index_path.is_file():
        dedup_columns = [
            column for column in (
                "User Id", "User Name", "Course Id", "Lesson Id", "Slide Id",
                "First Attempt Answered At", "Time Spent (Seconds)", "Correct Answer",
            ) if column in annotations
        ]
        events = annotations.drop_duplicates(dedup_columns).copy()
        events["subject_id"] = events["User Name"].astype(str).str.strip()
        events["event_time"] = pd.to_datetime(
            events["First Attempt Answered At"], errors="coerce", utc=True
        )
        events = events.loc[events["event_time"].notna()].sort_values(
            ["subject_id", "event_time", "Slide Id"], kind="stable"
        )
        index = pd.read_parquet(
            raw_index_path,
            columns=[
                "sample_id", "source", "subject_id", "record_id",
                "absolute_t_start", "absolute_t_end", "status",
            ],
        )
        index = index.loc[index["source"].eq("Old_EEG") & index["status"].eq("ok")].copy()
        index["subject_id"] = index["subject_id"].astype(str)
        pm_by_id = pm_frame.set_index("sample_id")
        for _, event in events.iterrows():
            timestamp = event["event_time"].timestamp()
            candidates = index.loc[
                index["subject_id"].eq(str(event["subject_id"]))
                & index["absolute_t_start"].le(timestamp)
                & index["absolute_t_end"].gt(timestamp)
            ]
            match_summary["events"] += 1
            if len(candidates) == 0:
                match_summary["unmatched"] += 1
                continue
            if len(candidates) > 1:
                match_summary["ambiguous_matches"] += 1
                continue
            match_summary["unique_matches"] += 1
            window = candidates.iloc[0]
            sample_id = window["sample_id"]
            if sample_id not in pm_by_id.index:
                continue
            pm_row = pm_by_id.loc[sample_id]
            for field in ("Time Spent (Seconds)", "Correct Answer"):
                value = event.get(field, np.nan)
                numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.isna(numeric):
                    continue
                aligned_rows.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": str(event["subject_id"]),
                        "record_id": str(window["record_id"]),
                        "event_timestamp_utc": event["event_time"].isoformat(),
                        "window_absolute_t_start": float(window["absolute_t_start"]),
                        "window_absolute_t_end": float(window["absolute_t_end"]),
                        "time_difference_to_window_center_seconds": float(
                            timestamp - (float(window["absolute_t_start"]) + float(window["absolute_t_end"])) / 2.0
                        ),
                        "overlap": True,
                        "behavioral_variable": field,
                        "behavioral_value": float(numeric),
                        **{column: pm_row[column] for column in TARGET_COLUMNS},
                    }
                )
                if len(aligned_rows) >= smoke_limit:
                    break
            if len(aligned_rows) >= smoke_limit:
                break
    aligned = pd.DataFrame(aligned_rows)
    exact_rate = (
        match_summary["unique_matches"] / match_summary["events"]
        if match_summary["events"] else 0.0
    )
    for field, level, note in (
        ("Time Spent (Seconds)", "event_level", "slide dwell/completion duration; not a reaction-time measure"),
        ("Correct Answer", "event_level", "binary correctness for question rows"),
    ):
        feasibility_rows.append(
            {
                "source": "Old_EEG",
                "field": field,
                "has_timestamp": True,
                "timestamp_semantics": "naive export timestamp; UTC interpretation supported by EEG overlap",
                "measurement_level": level,
                "record_link": "subject_id plus unique physical-record time containment",
                "window_alignment": "feasible only for unique half-open interval matches",
                "actual_behavioral_metric": True,
                "reason": note,
                "events_checked": match_summary["events"],
                "unique_match_fraction": exact_rate,
                "ambiguous_matches": match_summary["ambiguous_matches"],
                "nearest_tolerance_used": False,
            }
        )
    audit = {
        "gpn_json_files": len(json_paths),
        "gpn_json_marker_items": marker_items,
        "gpn_interval_marker_files": len(marker_paths),
        "gpn_interval_marker_rows": int(len(markers)),
        "gpn_nonempty_interval_marker_files": int(sum(len(frame) > 0 for frame in marker_frames)),
        "old_annotation_files": len(annotation_paths),
        "old_annotation_rows_raw": int(len(annotations)),
        "alignment": match_summary,
        "alignment_smoke_rows": int(len(aligned)),
        "correlation_analysis_executed": False,
    }
    return (
        pd.DataFrame(inventory_rows),
        pd.DataFrame(feasibility_rows),
        aligned,
        audit,
    )


def _save_figure(fig: Any, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=160, bbox_inches="tight")


def create_figures(
    output_dir: Path,
    frame: pd.DataFrame,
    variants: VariantResult,
    summary: pd.DataFrame,
    outliers: pd.DataFrame,
    stability: pd.DataFrame,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    group_keys = ["source", "subject_id", "record_group_id", "record_id"]
    representative: np.ndarray | None = None
    representative_name = ""
    focus = variants.values["focus"]["baseline_raw"]
    for keys, group in frame.groupby(group_keys, sort=True, observed=True):
        indices = group.index.to_numpy(dtype=np.int64)
        if np.isfinite(focus[indices]).sum() >= 30:
            representative = indices
            representative_name = " / ".join(map(str, keys))
            break
    if representative is not None:
        fig, axis = plt.subplots(figsize=(11, 4.5))
        x = frame.loc[representative, "t_start"].to_numpy(dtype=float) / 60.0
        for variant in VARIANT_ORDER:
            axis.plot(x, variants.values["focus"][variant][representative], label=variant, linewidth=1.25)
        axis.set(title=f"Representative focus series: {representative_name}", xlabel="t_start, min", ylabel="PM focus")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
        base = figures_dir / "representative_focus_time_series"
        _save_figure(fig, base)
        plt.close(fig)
        created.extend([
            "figures/representative_focus_time_series.svg",
            "figures/representative_focus_time_series.png",
        ])

    fig, axis = plt.subplots(figsize=(9, 4.5))
    data = []
    labels = []
    for variant in VARIANT_ORDER:
        values = summary.loc[summary["variant"].eq(variant), "mean_absolute_first_difference"].to_numpy(dtype=float)
        data.append(values)
        labels.append(variant.replace("causal_", ""))
    axis.boxplot(data, tick_labels=labels, showmeans=True)
    axis.set(title="Absolute first-difference summary across seven PM", ylabel="Mean |Δ PM| per PM")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    base = figures_dir / "absolute_first_difference_by_variant"
    _save_figure(fig, base)
    plt.close(fig)
    created.extend([
        "figures/absolute_first_difference_by_variant.svg",
        "figures/absolute_first_difference_by_variant.png",
    ])

    class_plot = stability.loc[~stability["variant"].eq("baseline_raw")].groupby(
        ["pm", "variant"], sort=True
    )["class_change_fraction"].mean().unstack("variant").reindex(PM_METRICS)
    fig, axis = plt.subplots(figsize=(10, 4.5))
    class_plot.plot(kind="bar", ax=axis)
    axis.set(title="Fold-local Q3 class changes", ylabel="Outer-test changed fraction", xlabel="PM")
    axis.tick_params(axis="x", rotation=30)
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    base = figures_dir / "q3_class_change_fraction"
    _save_figure(fig, base)
    plt.close(fig)
    created.extend([
        "figures/q3_class_change_fraction.svg",
        "figures/q3_class_change_fraction.png",
    ])

    fig, axis = plt.subplots(figsize=(9, 4.5))
    ordered = outliers.set_index("pm").reindex(PM_METRICS)
    axis.bar(ordered.index, ordered["outlier_fraction"])
    axis.set(title="Causal Hampel statistical anomaly fraction", ylabel="Fraction", xlabel="PM")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    base = figures_dir / "hampel_outlier_fraction"
    _save_figure(fig, base)
    plt.close(fig)
    created.extend([
        "figures/hampel_outlier_fraction.svg",
        "figures/hampel_outlier_fraction.png",
    ])
    return created


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def plan_analysis(config_path: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    return {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "pm_targets": list(config["pm_targets"]),
        "variants": list(config["variants"]),
        "group_keys": config["temporal"]["group_keys"],
        "expected_step_seconds": config["temporal"]["expected_step_seconds"],
        "max_gap_seconds": config["temporal"]["max_gap_seconds"],
        "missing_policy": config["temporal"]["missing_policy"],
        "folds": config["folds"]["fold_ids"],
        "q3_policy": config["folds"]["q3_transform_policy"],
        "output_dir": str(output_dir or config["output_dir"]),
        "models_trained": 0,
        "writes_performed": False,
        "config_hash": stable_hash(config),
    }


def _render_readme(
    manifest: Mapping[str, Any],
    summary: pd.DataFrame,
    outliers: pd.DataFrame,
    stability: pd.DataFrame,
    lags: pd.DataFrame,
) -> str:
    baseline = summary.loc[summary["variant"].eq("baseline_raw")]
    changes = summary.loc[~summary["variant"].eq("baseline_raw")]
    class_changes = stability.loc[~stability["variant"].eq("baseline_raw")]
    lines = [
        "# PM temporal-quality ablation v1",
        "",
        "Статус: **diagnostic**. Модели EEG→PM не обучались.",
        "",
        "## Канонический исходный контракт",
        "",
        "`bench/datasets/emotiv_pm_window_builder.py::read_and_aggregate_record` группирует "
        "каждую physical recording по `floor(Timestamp / 10 s)` и формирует "
        "`target_*` из `PM.*.Scaled__mean`. В этом пути нет PM-интерполяции, "
        "дополнительного smoothing или outlier removal.",
        "",
        "## Основные числа",
        "",
        f"- Окон: {manifest['dataset']['rows']}; участников: {manifest['dataset']['subjects']}; source-records: {manifest['dataset']['records']}.",
        f"- Baseline mean |first difference| по семи PM: {baseline['mean_absolute_first_difference'].mean():.6f}.",
        f"- Максимальная доля изменённых continuous PM: {changes['changed_value_fraction'].max():.4%}.",
        f"- Hampel отметил как статистически аномальные {outliers['outlier_count'].sum()} из {outliers['n_available'].sum()} доступных точек ({outliers['outlier_count'].sum()/outliers['n_available'].sum():.4%}).",
        f"- Максимальная средняя по folds доля изменённых Q3 labels: {class_changes.groupby(['pm','variant'])['class_change_fraction'].mean().max():.4%}.",
        f"- Наблюдаемые best-lag значения: {sorted(lags['best_lag_windows'].unique().tolist())} окон; для EMA теоретическая низкочастотная задержка равна 1 окну (10 s).",
        "",
        "Median/EMA снижают кратковременную вариативность, но это не является "
        "доказательством улучшения target quality. Hampel flags обозначают только "
        "статистически аномальные точки, а не подтверждённые ошибки разметки. "
        "Отсутствие заполнения NaN и record-local causal state гарантируют, что "
        "эксперимент не создаёт скрытой интерполяции и не использует будущие PM.",
        "",
        "## Behavioral audit",
        "",
        f"Найдены {manifest['behavioral_audit']['old_annotation_files']} Old_EEG annotation CSV с `Time Spent (Seconds)`, `Correct Answer` и timestamp полями. "
        "Это реальные event-level behavioural measurements, но `Time Spent` — "
        "длительность/время на слайде, не подтверждённое reaction time. GPN JSON и "
        "intervalMarker содержат в основном event/context markers; их нельзя автоматически "
        "считать поведенческим outcome. Выполнен только небольшой точный interval-containment "
        "alignment smoke; сложная корреляционная статистика не запускалась.",
        "",
        "## Научный вывод",
        "",
        "На этом этапе можно количественно сообщать об изменении smoothness, доле "
        "Hampel-анomalies, изменении continuous PM и fold-local Q3 labels, а также "
        "temporal lag. Нельзя утверждать, что smoothing улучшил качество или "
        "predictability EEG→PM, пока не выполнен заранее подготовленный downstream-runner. "
        "Canonical target pipeline менять по результатам только этой диагностики не следует.",
        "",
        f"Protocol hash: `{manifest['protocol_hash']}`.",
        "",
    ]
    return "\n".join(lines)


def run_analysis(
    config_path: str | Path,
    *,
    data_path: str | Path | None = None,
    reference_predictions: str | Path | None = None,
    raw_index_path: str | Path | None = None,
    raw_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    effective_data = _repo_path(data_path or config["data"]["processed_pm"])
    effective_reference = _repo_path(reference_predictions or config["data"]["reference_predictions"])
    effective_raw_index = _repo_path(raw_index_path or config["data"]["raw_window_index"])
    effective_raw_root = _repo_path(raw_root or config["data"]["raw_root"])
    effective_output = _repo_path(output_dir or config["output_dir"])
    for path, label in (
        (effective_data, "processed PM"),
        (effective_reference, "reference predictions"),
        (effective_raw_index, "raw window index"),
        (effective_raw_root, "raw metadata root"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    before_hash = file_sha256(effective_data)
    before_size = effective_data.stat().st_size
    columns = ["source", "subject_id", "record_id", "t_start", *config["pm_targets"].values()]
    frame = prepare_pm_frame(pd.read_parquet(effective_data, columns=columns), config)
    variants = build_variants(frame, config)
    summary, participant_macro, outlier_summary = summarize_variants(frame, variants)
    fold_by_subject = load_fixed_subject_folds(effective_reference, frame)
    thresholds, stability = calculate_q3_stability(frame, variants, fold_by_subject)
    lags = calculate_lag_diagnostics(
        variants,
        expected_step_seconds=float(config["temporal"]["expected_step_seconds"]),
        max_lag_windows=int(config["temporal"].get("lag_diagnostic_max_windows", 3)),
    )
    inventory, feasibility, behavioral_smoke, behavioral_audit = audit_behavioral_sources(
        effective_raw_root,
        effective_raw_index,
        frame,
        smoke_limit=int(config["behavioral_audit"]["alignment_smoke_limit"]),
    )
    effective_output.mkdir(parents=True, exist_ok=True)
    figures = create_figures(effective_output, frame, variants, summary, outlier_summary, stability)
    _write_csv(effective_output / "pm_variant_summary.csv", summary)
    _write_csv(effective_output / "pm_variant_participant_macro.csv", participant_macro)
    _write_csv(effective_output / "pm_outlier_summary.csv", outlier_summary)
    variants.outlier_points.to_parquet(effective_output / "pm_outlier_points.parquet", index=False)
    _write_csv(effective_output / "q3_thresholds_by_fold.csv", thresholds)
    _write_csv(effective_output / "q3_class_stability.csv", stability)
    _write_csv(effective_output / "temporal_lag_diagnostics.csv", lags)
    _write_csv(effective_output / "behavioral_inventory.csv", inventory)
    _write_csv(effective_output / "behavioral_alignment_feasibility.csv", feasibility)
    behavioral_smoke.to_parquet(effective_output / "behavioral_alignment_smoke.parquet", index=False)
    after_hash = file_sha256(effective_data)
    if before_hash != after_hash or before_size != effective_data.stat().st_size:
        raise RuntimeError("Canonical processed PM Parquet changed during analysis")
    availability = {
        metric: {
            "windows": int(frame[column].notna().sum()),
            "missing": int(frame[column].isna().sum()),
            "subjects": int(frame.loc[frame[column].notna(), "subject_id"].nunique()),
            "records": int(frame.loc[frame[column].notna(), "record_id"].nunique()),
        }
        for metric, column in config["pm_targets"].items()
    }
    fold_documents = {
        str(fold): {
            "test_subjects": sorted(subject for subject, assigned in fold_by_subject.items() if assigned == fold),
            "train_subjects": sorted(subject for subject, assigned in fold_by_subject.items() if assigned != fold),
        }
        for fold in range(1, 6)
    }
    manifest: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "result_status": config["result_status"],
        "analysis_only": True,
        "models_trained": 0,
        "git_commit": _git_head(),
        "config_hash": stable_hash(config),
        "dataset": {
            "logical_path": config["data"]["processed_pm"],
            "sha256": before_hash,
            "sha256_after": after_hash,
            "input_modified": False,
            "rows": int(len(frame)),
            "subjects": int(frame["subject_id"].nunique()),
            "records": int(frame["record_id"].nunique()),
            "sources": sorted(frame["source"].astype(str).unique().tolist()),
            "availability": availability,
        },
        "canonical_contract": {
            "builder": "bench/datasets/emotiv_pm_window_builder.py::read_and_aggregate_record",
            "physical_record_local": True,
            "window_seconds": 10.0,
            "stride_seconds": 10.0,
            "overlap_seconds": 0.0,
            "target_aggregation": "mean PM.*.Scaled inside absolute temporal bin",
            "pm_interpolation": False,
            "pm_smoothing": False,
            "pm_outlier_removal": False,
        },
        "temporal_policy": config["temporal"],
        "variants": config["variants"],
        "variant_sample_universe_identical": True,
        "folds": fold_documents,
        "reference_fold_sha256": file_sha256(effective_reference),
        "q3_transform": "bench.tasks.target_transforms.FoldLocalQuantileTargetTransform",
        "behavioral_audit": behavioral_audit,
        "figures": figures,
        "artifacts": sorted(path.name for path in effective_output.iterdir() if path.is_file()),
    }
    manifest["protocol_hash"] = stable_hash(manifest)
    _write_json(effective_output / "manifest.json", manifest)
    (effective_output / "README.md").write_text(
        _render_readme(manifest, summary, outlier_summary, stability, lags),
        encoding="utf-8",
    )
    return _jsonable(manifest)


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "PM_METRICS",
    "TARGET_COLUMNS",
    "VARIANT_ORDER",
    "VariantResult",
    "audit_behavioral_sources",
    "build_variants",
    "calculate_lag_diagnostics",
    "calculate_q3_stability",
    "causal_transform_1d",
    "load_config",
    "load_fixed_subject_folds",
    "plan_analysis",
    "prepare_pm_frame",
    "run_analysis",
    "stable_hash",
    "summarize_variants",
]
