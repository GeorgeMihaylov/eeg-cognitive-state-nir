"""Opt-in montage-independent regional EEG feature representation.

Version 1 aggregates existing target-free per-channel spectral, statistical,
and entropy features. Existing global connectivity summaries are deliberately
excluded: averaging different physical channel-pair sets is not a
device-invariant connectivity measurement.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from . import entropy, spectral, statistical
from ._validation import json_safe, validate_sample_rate, validate_window
from .montage import (
    CANONICAL_REGIONS,
    build_montage_manifest,
    montage_hash as hash_montage_manifest,
    normalize_custom_mapping,
)


REGIONAL_FEATURE_SCHEMA_VERSION = "cogstate-regional-features-v1"
REGIONAL_FEATURE_GROUP_ORDER = ("spectral", "statistical", "entropy")
REGIONAL_AGGREGATIONS = ("median", "iqr")


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RegionalFeatureConfig:
    """Configuration for the opt-in regional feature schema."""

    sample_rate: float
    include_spectral: bool = True
    include_statistical: bool = True
    include_entropy: bool = True
    spectral_config: spectral.SpectralConfig | None = None
    statistical_config: statistical.StatisticalConfig = field(
        default_factory=statistical.StatisticalConfig
    )
    entropy_config: entropy.EntropyConfig | None = None
    aggregations: tuple[str, ...] = REGIONAL_AGGREGATIONS
    missing_fill_value: float = 0.0
    custom_channel_mapping: (
        Mapping[str, str] | Sequence[tuple[str, str]] | None
    ) = None

    def __post_init__(self) -> None:
        rate = validate_sample_rate(self.sample_rate)
        if not any(
            (self.include_spectral, self.include_statistical, self.include_entropy)
        ):
            raise ValueError("at least one regional feature group must be enabled")
        requested = tuple(str(name).strip().lower() for name in self.aggregations)
        unknown = set(requested) - set(REGIONAL_AGGREGATIONS)
        if unknown or not requested or len(set(requested)) != len(requested):
            raise ValueError(
                f"aggregations must be unique values from {REGIONAL_AGGREGATIONS}"
            )
        ordered = tuple(name for name in REGIONAL_AGGREGATIONS if name in requested)
        fill = float(self.missing_fill_value)
        if not np.isfinite(fill):
            raise ValueError("missing_fill_value must be finite")
        spectral_config = self.spectral_config or spectral.SpectralConfig(rate)
        entropy_config = self.entropy_config or entropy.EntropyConfig(rate)
        for name, nested_rate in (
            ("spectral_config", spectral_config.sample_rate),
            ("entropy_config", entropy_config.sample_rate),
        ):
            if float(nested_rate) != rate:
                raise ValueError(f"{name}.sample_rate must equal sample_rate")
        object.__setattr__(self, "sample_rate", rate)
        object.__setattr__(self, "aggregations", ordered)
        object.__setattr__(self, "missing_fill_value", fill)
        object.__setattr__(self, "spectral_config", spectral_config)
        object.__setattr__(self, "entropy_config", entropy_config)
        object.__setattr__(
            self,
            "custom_channel_mapping",
            normalize_custom_mapping(self.custom_channel_mapping),
        )


class RegionalFeaturePipeline:
    """Aggregate channel-level extractors into a fixed regional vector."""

    def __init__(self, config: RegionalFeatureConfig):
        self.config = config

    def _group_definitions(self) -> tuple[tuple[str, bool, Any, Any, Any], ...]:
        return (
            (
                "spectral",
                self.config.include_spectral,
                spectral.extract_spectral_features,
                spectral.feature_names,
                self.config.spectral_config,
            ),
            (
                "statistical",
                self.config.include_statistical,
                statistical.extract_statistical_features,
                statistical.feature_names,
                self.config.statistical_config,
            ),
            (
                "entropy",
                self.config.include_entropy,
                entropy.extract_entropy_features,
                entropy.feature_names,
                self.config.entropy_config,
            ),
        )

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for group, enabled, _, name_builder, group_config in self._group_definitions():
            if not enabled:
                continue
            for base_name in name_builder(group_config):
                for region in CANONICAL_REGIONS:
                    names.extend(
                        f"{group}__{base_name}__{region}__{aggregation}"
                        for aggregation in self.config.aggregations
                    )
        for region in CANONICAL_REGIONS:
            names.extend(
                (
                    f"coverage__{region}__present",
                    f"coverage__{region}__channel_count",
                )
            )
        return names

    def feature_specification(self) -> dict[str, Any]:
        """Return a device-independent semantic schema specification."""
        payload = {
            "schema_version": REGIONAL_FEATURE_SCHEMA_VERSION,
            "input_layout": "samples,channels",
            "sample_rate": float(self.config.sample_rate),
            "canonical_regions": list(CANONICAL_REGIONS),
            "enabled_groups": [
                group
                for group, enabled, _, _, _ in self._group_definitions()
                if enabled
            ],
            "spectral": asdict(self.config.spectral_config),
            "statistical": asdict(self.config.statistical_config),
            "entropy": asdict(self.config.entropy_config),
            "aggregation_policy": list(self.config.aggregations),
            "missing_policy": {
                "fill_value": float(self.config.missing_fill_value),
                "coverage_features": ["present", "channel_count"],
            },
            "connectivity": {
                "included": False,
                "reason": "region_to_region_connectivity_deferred_after_v1",
            },
            "feature_names": self.feature_names(),
        }
        payload["n_features"] = len(payload["feature_names"])
        return json_safe(payload)

    def schema_hash(self) -> str:
        """Hash only semantic schema inputs, never device montage details."""
        return _stable_hash(self.feature_specification())

    def montage_manifest(self, channel_names: Sequence[str]) -> dict[str, Any]:
        return build_montage_manifest(
            channel_names,
            custom_mapping=self.config.custom_channel_mapping,
        )

    def montage_hash(self, channel_names: Sequence[str]) -> str:
        return hash_montage_manifest(self.montage_manifest(channel_names))

    @staticmethod
    def _aggregate(values: np.ndarray, aggregation: str) -> float:
        if aggregation == "median":
            return float(np.median(values))
        if aggregation == "iqr":
            lower, upper = np.percentile(values, (25.0, 75.0))
            return float(upper - lower)
        raise RuntimeError(f"unsupported regional aggregation {aggregation!r}")

    def transform_window(
        self,
        window: np.ndarray,
        *,
        channel_names: Sequence[str],
    ) -> np.ndarray:
        signal = validate_window(window)
        manifest = self.montage_manifest(channel_names)
        if len(manifest["channels"]) != signal.shape[1]:
            raise ValueError(
                "channel_names length must match the EEG window channel dimension"
            )
        # Canonicalize extractor input order as well as the later aggregation.
        # Vectorized FFT/statistical implementations may otherwise differ by a
        # few floating-point ULPs when their channel columns are permuted.
        ordered_rows = sorted(
            manifest["channels"], key=lambda row: str(row["normalized_name"])
        )
        ordered_indices = [int(row["input_index"]) for row in ordered_rows]
        ordered_signal = signal[:, ordered_indices]
        region_indices = {region: [] for region in CANONICAL_REGIONS}
        for canonical_index, row in enumerate(ordered_rows):
            region_indices[str(row["region"])].append(canonical_index)

        output: list[float] = []
        for _, enabled, extractor, name_builder, group_config in self._group_definitions():
            if not enabled:
                continue
            features = extractor(ordered_signal, group_config)
            expected_names = name_builder(group_config)
            if tuple(features) != tuple(expected_names):
                raise RuntimeError("extractor feature order differs from its schema")
            for base_name in expected_names:
                channel_values = np.asarray(features[base_name], dtype=float).reshape(-1)
                if len(channel_values) != ordered_signal.shape[1]:
                    raise RuntimeError("extractor did not return one value per channel")
                for region in CANONICAL_REGIONS:
                    indices = region_indices[region]
                    if not indices:
                        output.extend(
                            self.config.missing_fill_value
                            for _ in self.config.aggregations
                        )
                        continue
                    selected = channel_values[np.asarray(indices, dtype=int)]
                    output.extend(
                        self._aggregate(selected, aggregation)
                        for aggregation in self.config.aggregations
                    )
        for region in CANONICAL_REGIONS:
            count = len(region_indices[region])
            output.extend((float(count > 0), float(count)))

        result = np.ascontiguousarray(output, dtype=np.float64)
        if len(result) != len(self.feature_names()):
            raise RuntimeError("regional feature vector width differs from schema")
        if not np.isfinite(result).all():
            raise RuntimeError("regional feature pipeline produced NaN or Inf")
        return result

    def transform_batch(
        self,
        windows: np.ndarray,
        *,
        channel_names: Sequence[str],
        chunk_size: int | None = None,
    ) -> np.ndarray:
        if not isinstance(windows, np.ndarray):
            raise TypeError("EEG batch must be a numpy.ndarray")
        if windows.ndim != 3:
            raise ValueError(
                "EEG batch must have shape [batch, samples, channels], "
                f"got {windows.shape}"
            )
        if len(windows) == 0:
            raise ValueError("EEG batch cannot be empty")
        if not np.isfinite(windows).all():
            raise ValueError("EEG batch contains NaN or Inf")
        size = len(windows) if chunk_size is None else int(chunk_size)
        if size <= 0:
            raise ValueError("chunk_size must be positive or None")
        rows: list[np.ndarray] = []
        for start in range(0, len(windows), size):
            rows.extend(
                self.transform_window(windows[index], channel_names=channel_names)
                for index in range(start, min(start + size, len(windows)))
            )
        return np.ascontiguousarray(np.stack(rows), dtype=np.float64)
