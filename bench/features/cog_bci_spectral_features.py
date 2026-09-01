"""Deterministic window- and record-level spectral features for COG-BCI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import signal


COG_BCI_SPECTRAL_SCHEMA_VERSION = "cog-bci-spectral-features-v1"

FREQUENCY_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 45.0),
}

SPECTRAL_FEATURE_TYPES = (
    *(f"log_power_{name}" for name in FREQUENCY_BANDS),
    *(f"relative_power_{name}" for name in FREQUENCY_BANDS),
    "theta_alpha",
    "theta_beta",
    "log_variance",
)
NUISANCE_FEATURE_TYPES = (
    "dc_magnitude",
    "power_49_51",
    "line_to_1_45_ratio",
)
CHANNEL_SUMMARY_STATISTICS = ("mean", "std", "median", "min", "max")
RECORD_AGGREGATIONS = ("mean", "median", "std", "iqr")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


@dataclass(frozen=True)
class SpectralFeatureSpec:
    """Resolved Welch and feature-engineering contract."""

    sampling_rate_hz: float = 500.0
    nperseg: int = 512
    noverlap: int = 256
    detrend: str = "constant"
    scaling: str = "density"

    def __post_init__(self) -> None:
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if self.nperseg < 8:
            raise ValueError("nperseg must be at least 8")
        if self.noverlap < 0 or self.noverlap >= self.nperseg:
            raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")
        if self.detrend != "constant":
            raise ValueError("Only the audited detrend='constant' contract is supported")
        if self.scaling != "density":
            raise ValueError("Only Welch scaling='density' is supported")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COG_BCI_SPECTRAL_SCHEMA_VERSION,
            "sampling_rate_hz": float(self.sampling_rate_hz),
            "welch": {
                "window": "hann_periodic_scipy_default",
                "nperseg": int(self.nperseg),
                "noverlap": int(self.noverlap),
                "detrend": self.detrend,
                "scaling": self.scaling,
                "average": "mean",
                "axis": "time",
            },
            "frequency_bands_hz": {
                name: [float(low), float(high)]
                for name, (low, high) in FREQUENCY_BANDS.items()
            },
            "relative_power_denominator_hz": [1.0, 45.0],
            "nuisance_frequency_band_hz": [49.0, 51.0],
            "channel_summary_statistics": list(CHANNEL_SUMMARY_STATISTICS),
            "record_aggregations": list(RECORD_AGGREGATIONS),
            "output_dtype": "float32",
        }

    def stable_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SpectralFeatureBundle:
    """Both predefined representations for one batch of EEG windows."""

    channel_wise: np.ndarray
    channel_wise_columns: tuple[str, ...]
    channel_wise_spectral_columns: tuple[str, ...]
    channel_wise_nuisance_columns: tuple[str, ...]
    global_summary: np.ndarray
    global_summary_columns: tuple[str, ...]
    global_summary_spectral_columns: tuple[str, ...]
    global_summary_nuisance_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        arrays_and_names = (
            (self.channel_wise, self.channel_wise_columns),
            (self.global_summary, self.global_summary_columns),
        )
        for array, names in arrays_and_names:
            if array.ndim != 2:
                raise ValueError("Spectral feature arrays must be two-dimensional")
            if array.shape[1] != len(names):
                raise ValueError("Feature-name count does not match feature array")
            if not np.isfinite(array).all():
                raise ValueError("Spectral features contain NaN or Inf")


def _integrated_power(
    frequencies: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
    *,
    include_high: bool = False,
) -> np.ndarray:
    upper = frequencies <= high_hz if include_high else frequencies < high_hz
    mask = (frequencies >= low_hz) & upper
    bins = int(mask.sum())
    if bins == 0:
        raise ValueError(
            f"Insufficient Welch bins for frequency range {low_hz}-{high_hz} Hz"
        )
    if bins == 1:
        if len(frequencies) < 2:
            raise ValueError("Welch spectrum has no resolvable frequency spacing")
        return psd[..., mask][..., 0] * float(frequencies[1] - frequencies[0])
    return np.trapezoid(psd[..., mask], frequencies[mask], axis=-1)


def _channel_feature_values(
    windows: np.ndarray,
    spec: SpectralFeatureSpec,
) -> dict[str, np.ndarray]:
    nperseg = min(spec.nperseg, windows.shape[-1])
    noverlap = min(spec.noverlap, max(0, nperseg - 1))
    frequencies, psd = signal.welch(
        windows,
        fs=spec.sampling_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=spec.detrend,
        scaling=spec.scaling,
        axis=-1,
    )
    epsilon = np.finfo(np.float64).tiny
    powers = {
        name: _integrated_power(frequencies, psd, low, high)
        for name, (low, high) in FREQUENCY_BANDS.items()
    }
    total_power = _integrated_power(
        frequencies,
        psd,
        1.0,
        45.0,
        include_high=True,
    )
    line_power = _integrated_power(
        frequencies,
        psd,
        49.0,
        51.0,
        include_high=True,
    )
    values: dict[str, np.ndarray] = {}
    for name in FREQUENCY_BANDS:
        values[f"log_power_{name}"] = np.log10(np.maximum(powers[name], epsilon))
    for name in FREQUENCY_BANDS:
        values[f"relative_power_{name}"] = powers[name] / np.maximum(
            total_power, epsilon
        )
    values["theta_alpha"] = powers["theta"] / np.maximum(
        powers["alpha"], epsilon
    )
    values["theta_beta"] = powers["theta"] / np.maximum(powers["beta"], epsilon)
    variance = np.var(windows, axis=-1, dtype=np.float64)
    values["log_variance"] = np.log10(np.maximum(variance, epsilon))
    values["dc_magnitude"] = np.abs(np.mean(windows, axis=-1, dtype=np.float64))
    values["power_49_51"] = line_power
    values["line_to_1_45_ratio"] = line_power / np.maximum(total_power, epsilon)
    return values


def extract_spectral_feature_bundle(
    windows: np.ndarray,
    *,
    channel_names: Sequence[str],
    spec: SpectralFeatureSpec,
) -> SpectralFeatureBundle:
    """Extract audited channel-wise and global-summary window features."""

    array = np.asarray(windows, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("Spectral input must have shape [windows, channels, time]")
    if array.shape[0] == 0 or array.shape[-1] < 8:
        raise ValueError("Spectral input must contain non-empty windows")
    if array.shape[1] != len(channel_names):
        raise ValueError("Channel-name count does not match spectral input")
    normalized_names = tuple(str(name) for name in channel_names)
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("Channel names must be unique and ordered")
    if not np.isfinite(array).all():
        raise ValueError("Spectral input contains NaN or Inf")

    feature_values = _channel_feature_values(array, spec)
    all_types = (*SPECTRAL_FEATURE_TYPES, *NUISANCE_FEATURE_TYPES)

    channel_parts: list[np.ndarray] = []
    channel_columns: list[str] = []
    channel_spectral: list[str] = []
    channel_nuisance: list[str] = []
    for channel_index, channel_name in enumerate(normalized_names):
        for feature_type in all_types:
            column = f"cw__{channel_name}__{feature_type}"
            channel_parts.append(feature_values[feature_type][:, channel_index])
            channel_columns.append(column)
            if feature_type in NUISANCE_FEATURE_TYPES:
                channel_nuisance.append(column)
            else:
                channel_spectral.append(column)

    summary_parts: list[np.ndarray] = []
    summary_columns: list[str] = []
    summary_spectral: list[str] = []
    summary_nuisance: list[str] = []
    reducers = {
        "mean": lambda value: np.mean(value, axis=1),
        "std": lambda value: np.std(value, axis=1),
        "median": lambda value: np.median(value, axis=1),
        "min": lambda value: np.min(value, axis=1),
        "max": lambda value: np.max(value, axis=1),
    }
    for feature_type in all_types:
        values = feature_values[feature_type]
        for statistic in CHANNEL_SUMMARY_STATISTICS:
            column = f"gs__{feature_type}__{statistic}"
            summary_parts.append(reducers[statistic](values))
            summary_columns.append(column)
            if feature_type in NUISANCE_FEATURE_TYPES:
                summary_nuisance.append(column)
            else:
                summary_spectral.append(column)

    channel_array = np.column_stack(channel_parts).astype(np.float32)
    summary_array = np.column_stack(summary_parts).astype(np.float32)
    return SpectralFeatureBundle(
        channel_wise=channel_array,
        channel_wise_columns=tuple(channel_columns),
        channel_wise_spectral_columns=tuple(channel_spectral),
        channel_wise_nuisance_columns=tuple(channel_nuisance),
        global_summary=summary_array,
        global_summary_columns=tuple(summary_columns),
        global_summary_spectral_columns=tuple(summary_spectral),
        global_summary_nuisance_columns=tuple(summary_nuisance),
    )


def aggregate_record_features(
    window_features: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Aggregate windows within one record using mean/median/std/IQR."""

    required = {
        "record_id",
        "subject_id",
        "session_id",
        "target",
        "class_name",
        "outer_fold",
    }
    missing = sorted(required - set(window_features.columns))
    if missing:
        raise ValueError(f"Window feature table is missing metadata: {missing}")
    if not feature_columns:
        raise ValueError("At least one feature column is required")
    unknown = sorted(set(feature_columns) - set(window_features.columns))
    if unknown:
        raise ValueError(f"Unknown feature columns: {unknown[:5]}")

    rows: list[dict[str, Any]] = []
    for record_id, group in window_features.groupby("record_id", sort=True):
        identity = group[
            ["subject_id", "session_id", "target", "class_name", "outer_fold"]
        ].drop_duplicates()
        if len(identity) != 1:
            raise ValueError(f"Record identity changed within {record_id}")
        values = group[list(feature_columns)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Record {record_id} contains non-finite features")
        q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
        aggregated = {
            "mean": np.mean(values, axis=0),
            "median": np.median(values, axis=0),
            "std": np.std(values, axis=0),
            "iqr": q75 - q25,
        }
        row: dict[str, Any] = {
            "record_id": str(record_id),
            **identity.iloc[0].to_dict(),
            "window_count": int(len(group)),
        }
        for aggregation in RECORD_AGGREGATIONS:
            for column, value in zip(
                feature_columns, aggregated[aggregation], strict=True
            ):
                row[f"record_{aggregation}__{column}"] = float(value)
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty or result["record_id"].duplicated().any():
        raise ValueError("Record aggregation produced an empty or duplicate index")
    return result


def feature_columns_for(
    record_features: pd.DataFrame,
    *,
    representation: str,
    feature_set: str,
) -> list[str]:
    """Resolve a predefined model feature set without data-dependent selection."""

    representation_prefix = {
        "channel_wise": "cw__",
        "global_summary": "gs__",
    }.get(str(representation))
    if representation_prefix is None:
        raise ValueError(
            "representation must be one of ['channel_wise', 'global_summary']"
        )
    if feature_set not in {"spectral_only", "spectral_plus_nuisance"}:
        raise ValueError(
            "feature_set must be one of "
            "['spectral_only', 'spectral_plus_nuisance']"
        )
    columns = []
    for column in record_features.columns:
        if not column.startswith("record_"):
            continue
        marker = column.find(representation_prefix)
        if marker < 0:
            continue
        is_nuisance = any(
            f"__{name}" in column for name in NUISANCE_FEATURE_TYPES
        )
        if feature_set == "spectral_only" and is_nuisance:
            continue
        columns.append(column)
    if not columns:
        raise ValueError(
            f"No columns for representation={representation!r}, "
            f"feature_set={feature_set!r}"
        )
    return columns
