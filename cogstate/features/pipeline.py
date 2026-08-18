"""Canonical target-free EEG feature pipeline.

Input windows use the explicit layout ``[samples, channels]``. Feature groups
are independently configurable, names and vector order are stable, and batch
transformation is equivalent to stacking single-window transformations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from . import connectivity, entropy, spectral, statistical
from ._validation import json_safe, validate_sample_rate, validate_window


FEATURE_SCHEMA_VERSION = "cogstate-features-v1"
FEATURE_GROUP_ORDER = ("spectral", "statistical", "entropy", "connectivity")


@dataclass(frozen=True)
class FeaturePipelineConfig:
    sample_rate: float
    channel_names: tuple[str, ...] | None = None
    include_spectral: bool = True
    include_statistical: bool = True
    include_entropy: bool = True
    include_connectivity: bool = True
    spectral_config: spectral.SpectralConfig | None = None
    statistical_config: statistical.StatisticalConfig = field(
        default_factory=statistical.StatisticalConfig
    )
    entropy_config: entropy.EntropyConfig | None = None
    connectivity_config: connectivity.ConnectivityConfig | None = None

    def __post_init__(self) -> None:
        rate = validate_sample_rate(self.sample_rate)
        if not any(
            (
                self.include_spectral,
                self.include_statistical,
                self.include_entropy,
                self.include_connectivity,
            )
        ):
            raise ValueError("at least one feature group must be enabled")
        if self.channel_names is not None:
            names = tuple(str(name).strip() for name in self.channel_names)
            if not names or any(not name for name in names) or len(set(names)) != len(names):
                raise ValueError("channel_names must be non-empty and unique")
            object.__setattr__(self, "channel_names", names)
        spectral_config = self.spectral_config or spectral.SpectralConfig(rate)
        entropy_config = self.entropy_config or entropy.EntropyConfig(rate)
        connectivity_config = self.connectivity_config or connectivity.ConnectivityConfig(rate)
        for name, nested_rate in (
            ("spectral_config", spectral_config.sample_rate),
            ("entropy_config", entropy_config.sample_rate),
            ("connectivity_config", connectivity_config.sample_rate),
        ):
            if float(nested_rate) != rate:
                raise ValueError(f"{name}.sample_rate must equal pipeline sample_rate")
        object.__setattr__(self, "spectral_config", spectral_config)
        object.__setattr__(self, "entropy_config", entropy_config)
        object.__setattr__(self, "connectivity_config", connectivity_config)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FeaturePipelineConfig":
        """Build a config from a YAML/JSON-compatible feature profile."""
        values = dict(payload.get("feature_pipeline", payload))
        groups = dict(values.get("groups", {}))
        sample_rate = float(values["sample_rate"])
        spectral_values = dict(values.get("spectral", {}))
        statistical_values = dict(values.get("statistical", {}))
        entropy_values = dict(values.get("entropy", {}))
        connectivity_values = dict(values.get("connectivity", {}))
        for nested in (spectral_values, entropy_values, connectivity_values):
            nested.setdefault("sample_rate", sample_rate)
        if "bands" in spectral_values:
            spectral_values["bands"] = {
                str(name): tuple(limits)
                for name, limits in spectral_values["bands"].items()
            }
        if "bands" in connectivity_values:
            connectivity_values["bands"] = {
                str(name): tuple(limits)
                for name, limits in connectivity_values["bands"].items()
            }
        for key in ("metrics", "summary_statistics"):
            if key in connectivity_values:
                connectivity_values[key] = tuple(connectivity_values[key])
        channel_names = values.get("channel_names")
        return cls(
            sample_rate=sample_rate,
            channel_names=(
                None if channel_names is None else tuple(str(name) for name in channel_names)
            ),
            include_spectral=bool(groups.get("spectral", True)),
            include_statistical=bool(groups.get("statistical", True)),
            include_entropy=bool(groups.get("entropy", True)),
            include_connectivity=bool(groups.get("connectivity", True)),
            spectral_config=spectral.SpectralConfig(**spectral_values),
            statistical_config=statistical.StatisticalConfig(**statistical_values),
            entropy_config=entropy.EntropyConfig(**entropy_values),
            connectivity_config=connectivity.ConnectivityConfig(**connectivity_values),
        )


def _flatten_group(
    features: Mapping[str, np.ndarray],
    expected_names: Sequence[str],
    *,
    expected_width: int,
) -> np.ndarray:
    if tuple(features) != tuple(expected_names):
        raise RuntimeError(
            f"feature implementation order {tuple(features)} does not match "
            f"schema order {tuple(expected_names)}"
        )
    arrays = [np.asarray(features[name], dtype=float).reshape(-1) for name in expected_names]
    if any(len(value) != expected_width for value in arrays):
        raise RuntimeError("feature block width does not match the declared schema")
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=float)


class FeaturePipeline:
    """Extract deterministic target-free EEG feature vectors."""

    def __init__(self, config: FeaturePipelineConfig):
        self.config = config

    def _channel_names(
        self,
        n_channels: int,
        channel_names: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        supplied = self.config.channel_names if channel_names is None else tuple(channel_names)
        if supplied is None:
            return tuple(f"channel_{index:02d}" for index in range(n_channels))
        names = tuple(str(name) for name in supplied)
        if len(names) != n_channels or len(set(names)) != len(names):
            raise ValueError(
                f"channel_names must contain {n_channels} unique values"
            )
        return names

    def transform_window(self, window: np.ndarray) -> np.ndarray:
        signal = validate_window(window)
        self._channel_names(signal.shape[1])
        blocks: list[np.ndarray] = []
        if self.config.include_spectral:
            names = spectral.feature_names(self.config.spectral_config)
            blocks.append(
                _flatten_group(
                    spectral.extract_spectral_features(signal, self.config.spectral_config),
                    names,
                    expected_width=signal.shape[1],
                )
            )
        if self.config.include_statistical:
            names = statistical.feature_names(self.config.statistical_config)
            blocks.append(
                _flatten_group(
                    statistical.extract_statistical_features(
                        signal, self.config.statistical_config
                    ),
                    names,
                    expected_width=signal.shape[1],
                )
            )
        if self.config.include_entropy:
            names = entropy.feature_names(self.config.entropy_config)
            blocks.append(
                _flatten_group(
                    entropy.extract_entropy_features(signal, self.config.entropy_config),
                    names,
                    expected_width=signal.shape[1],
                )
            )
        if self.config.include_connectivity:
            names = connectivity.feature_names(self.config.connectivity_config)
            blocks.append(
                _flatten_group(
                    connectivity.extract_connectivity_features(
                        signal, self.config.connectivity_config
                    ),
                    names,
                    expected_width=1,
                )
            )
        result = np.ascontiguousarray(np.concatenate(blocks), dtype=np.float64)
        if len(result) != len(self.feature_names(signal.shape[1])):
            raise RuntimeError("feature vector and feature names have different lengths")
        if not np.isfinite(result).all():
            raise RuntimeError("feature pipeline produced NaN or Inf")
        return result

    def __call__(self, clean_signal: np.ndarray, window: object | None = None) -> np.ndarray:
        """Compatibility call for future ``StreamProcessor`` use.

        The optional metadata window is intentionally ignored; feature extraction
        depends only on the target-free EEG signal.
        """
        del window
        return self.transform_window(clean_signal)

    def transform_batch(
        self,
        windows: np.ndarray,
        *,
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
                self.transform_window(windows[index])
                for index in range(start, min(start + size, len(windows)))
            )
        return np.ascontiguousarray(np.stack(rows), dtype=np.float64)

    def feature_names(
        self,
        n_channels: int | None = None,
        channel_names: Sequence[str] | None = None,
    ) -> list[str]:
        if n_channels is None:
            if channel_names is not None:
                n_channels = len(channel_names)
            elif self.config.channel_names is not None:
                n_channels = len(self.config.channel_names)
            else:
                raise ValueError("n_channels or configured channel_names is required")
        channels = self._channel_names(int(n_channels), channel_names)
        names: list[str] = []
        group_configs = (
            ("spectral", self.config.include_spectral, spectral.feature_names(self.config.spectral_config)),
            ("statistical", self.config.include_statistical, statistical.feature_names(self.config.statistical_config)),
            ("entropy", self.config.include_entropy, entropy.feature_names(self.config.entropy_config)),
        )
        for group, enabled, base_names in group_configs:
            if enabled:
                for base_name in base_names:
                    names.extend(f"{group}__{base_name}__{channel}" for channel in channels)
        if self.config.include_connectivity:
            names.extend(
                f"connectivity__{name}"
                for name in connectivity.feature_names(self.config.connectivity_config)
            )
        return names

    def feature_specification(
        self,
        n_channels: int | None = None,
        channel_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        if n_channels is None:
            n_channels = (
                len(channel_names)
                if channel_names is not None
                else len(self.config.channel_names or ())
            )
        if not n_channels:
            raise ValueError("n_channels or configured channel_names is required")
        channels = self._channel_names(int(n_channels), channel_names)
        pairs = connectivity.channel_pairs(
            len(channels), self.config.connectivity_config.max_channel_pairs
        )
        spectral_spec = asdict(self.config.spectral_config)
        if spectral_spec.get("spectral_edge_band_hz") is None:
            spectral_spec.pop("spectral_edge_band_hz", None)
        if spectral_spec.get("include_engagement_index") is False:
            spectral_spec.pop("include_engagement_index", None)
        connectivity_spec = asdict(self.config.connectivity_config)
        if connectivity_spec.get("plv_mode") == "broadband":
            connectivity_spec.pop("plv_mode", None)
            connectivity_spec.pop("phase_filter_order", None)
        payload = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "input_layout": "samples,channels",
            "sample_rate": float(self.config.sample_rate),
            "channel_names": list(channels),
            "enabled_groups": [
                name
                for name, enabled in (
                    ("spectral", self.config.include_spectral),
                    ("statistical", self.config.include_statistical),
                    ("entropy", self.config.include_entropy),
                    ("connectivity", self.config.include_connectivity),
                )
                if enabled
            ],
            "spectral": spectral_spec,
            "statistical": asdict(self.config.statistical_config),
            "entropy": asdict(self.config.entropy_config),
            "connectivity": {
                **connectivity_spec,
                "pair_policy": (
                    "all_unique_unordered"
                    if self.config.connectivity_config.max_channel_pairs is None
                    else "deterministic_lexicographic_prefix"
                ),
                "channel_pairs": [
                    {
                        "indices": [first, second],
                        "channels": [channels[first], channels[second]],
                    }
                    for first, second in pairs
                ],
            },
            "feature_names": self.feature_names(len(channels), channels),
        }
        payload["n_features"] = len(payload["feature_names"])
        return json_safe(payload)

    def feature_hash(
        self,
        n_channels: int | None = None,
        channel_names: Sequence[str] | None = None,
    ) -> str:
        payload = self.feature_specification(n_channels, channel_names)
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def build_default_pipeline(
    sample_rate: float,
    channel_names: Sequence[str] | None = None,
) -> FeaturePipeline:
    return FeaturePipeline(
        FeaturePipelineConfig(
            sample_rate=sample_rate,
            channel_names=(None if channel_names is None else tuple(channel_names)),
        )
    )
