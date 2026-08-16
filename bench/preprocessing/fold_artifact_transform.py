"""Fold-scoped FASTER-like and ICA transforms for cached raw EEG windows."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from cogstate.preprocessing.artifact_removal import (
    ArtifactICA,
    FasterConfig,
    IcaConfig,
    apply_faster,
)


ARTIFACT_VARIANTS = ("raw", "faster", "ica", "faster_ica")


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_calibration_indices(
    manifest: pd.DataFrame,
    *,
    max_windows: int,
) -> np.ndarray:
    """Select a deterministic participant-balanced subset from outer train."""
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")
    required = {"sample_id", "subject_id", "record_id", "t_start"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Calibration manifest is missing columns: {missing}")
    frame = manifest.reset_index(drop=True).copy()
    frame["_position"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.sort_values(
        ["subject_id", "record_id", "t_start", "sample_id"],
        kind="mergesort",
    )
    participants = max(1, frame["subject_id"].astype(str).nunique())
    per_participant = max(1, int(np.ceil(max_windows / participants)))
    balanced = frame.groupby("subject_id", sort=True).head(per_participant)
    selected = balanced.head(max_windows)
    if len(selected) < min(max_windows, len(frame)):
        remaining = frame.loc[~frame["_position"].isin(selected["_position"])]
        selected = pd.concat(
            [selected, remaining.head(max_windows - len(selected))],
            ignore_index=True,
        )
    return selected["_position"].to_numpy(dtype=np.int64)


@dataclass(frozen=True)
class FoldArtifactConfig:
    variant: str
    sample_rate: float = 256.0
    calibration_max_windows: int = 64
    faster: FasterConfig = field(default_factory=FasterConfig)
    ica: IcaConfig = field(default_factory=IcaConfig)

    def __post_init__(self) -> None:
        normalized = str(self.variant).strip().lower()
        if normalized not in ARTIFACT_VARIANTS:
            raise ValueError(
                f"Unknown artifact variant {self.variant!r}; available={ARTIFACT_VARIANTS}"
            )
        object.__setattr__(self, "variant", normalized)
        if self.sample_rate <= 0 or self.calibration_max_windows <= 0:
            raise ValueError("sample_rate and calibration_max_windows must be positive")


class FoldArtifactTransform:
    """Fit ICA once on outer-train calibration data and reuse it unchanged."""

    def __init__(self, config: FoldArtifactConfig) -> None:
        self.config = config
        self.ica_: ArtifactICA | None = None
        self.manifest_: dict[str, Any] | None = None
        self.fit_count_ = 0
        self.transform_calls_ = 0
        self.transform_seconds_ = 0.0
        self.changed_windows_ = 0
        self.max_abs_delta_ = 0.0
        self.mean_abs_delta_sum_ = 0.0

    @property
    def uses_faster(self) -> bool:
        return self.config.variant in {"faster", "faster_ica"}

    @property
    def uses_ica(self) -> bool:
        return self.config.variant in {"ica", "faster_ica"}

    def fit(self, outer_train_view: Any, *, fold: int) -> "FoldArtifactTransform":
        if self.fit_count_:
            raise RuntimeError("FoldArtifactTransform cannot be fitted more than once")
        manifest = outer_train_view.manifest.reset_index(drop=True)
        indices = (
            select_calibration_indices(
                manifest, max_windows=self.config.calibration_max_windows
            )
            if self.uses_ica else np.asarray([], dtype=np.int64)
        )
        selected = manifest.iloc[indices].copy()
        sample_ids = selected["sample_id"].astype(str).tolist()
        state_hash: str | None = None
        n_artifact_components = 0
        n_iter: int | None = None
        converged: bool | None = None
        if self.uses_ica:
            calibration_parts = []
            for index in indices:
                signal = np.asarray(outer_train_view[int(index)][0], dtype=np.float32).T
                if self.uses_faster:
                    signal = apply_faster(signal, self.config.faster)
                calibration_parts.append(np.asarray(signal, dtype=np.float32))
            calibration = np.concatenate(calibration_parts, axis=0)
            if not np.isfinite(calibration).all():
                raise ValueError("ICA calibration data contain NaN or infinite values")
            self.ica_ = ArtifactICA(self.config.ica).fit(
                calibration, sample_rate=self.config.sample_rate
            )
            fitted = self.ica_._ica
            if fitted is None:
                raise RuntimeError("ArtifactICA did not retain its fitted state")
            n_artifact_components = self.ica_.n_artifact_components
            n_iter = int(getattr(fitted, "n_iter_", 0))
            converged = n_iter < int(self.config.ica.max_iter)
            digest = hashlib.sha256()
            for name in ("components_", "mixing_", "mean_", "whitening_"):
                values = np.ascontiguousarray(getattr(fitted, name), dtype=np.float64)
                digest.update(name.encode("ascii"))
                digest.update(str(values.shape).encode("ascii"))
                digest.update(values.view(np.uint8))
            digest.update(
                json.dumps(self.ica_._artifact_components, separators=(",", ":")).encode(
                    "ascii"
                )
            )
            state_hash = digest.hexdigest()
        selection_payload = {
            "fold": int(fold),
            "sample_ids": sample_ids,
            "variant": self.config.variant,
        }
        self.fit_count_ = 1
        self.manifest_ = {
            "fold": int(fold),
            "variant": self.config.variant,
            "implementation": (
                "per_window_faster_like_mean_channel_interpolation"
                if self.uses_faster else "none"
            ),
            "faster_semantics": (
                "apply_faster_per_window_no_epoch_dropping"
                if self.uses_faster else "not_applied"
            ),
            "ica_fit_scope": "outer_train_only" if self.uses_ica else "not_applicable",
            "ica_refit_on_transform": False,
            "outer_train_windows": int(len(manifest)),
            "outer_train_participants": sorted(
                manifest["subject_id"].astype(str).unique().tolist()
            ),
            "calibration_windows": int(len(selected)),
            "calibration_samples": int(
                len(selected) * int(outer_train_view.shape[-1])
            ),
            "calibration_participants": sorted(
                selected["subject_id"].astype(str).unique().tolist()
            ),
            "calibration_sample_ids_hash": stable_hash(sample_ids),
            "calibration_selection_hash": stable_hash(selection_payload),
            "faster_config": asdict(self.config.faster),
            "ica_config": {
                "n_components": self.config.ica.n_components,
                "max_iter": self.config.ica.max_iter,
                "random_state": self.config.ica.random_state,
            },
            "ica_state_hash": state_hash,
            "ica_n_iter": n_iter,
            "ica_converged": converged,
            "ica_artifact_components": int(n_artifact_components),
            "estimated_peak_calibration_bytes": int(
                len(selected) * int(outer_train_view.shape[-1])
                * int(outer_train_view.shape[-2]) * 8 * 3
            ),
            "operation_order": [
                "canonical_raw_cache",
                *(("apply_faster_per_window",) if self.uses_faster else ()),
                *(("fold_fitted_ica_transform",) if self.uses_ica else ()),
                "train_only_channel_normalization_in_torch_adapter",
            ],
        }
        return self

    def transform_window(self, window: np.ndarray) -> np.ndarray:
        if self.manifest_ is None:
            raise RuntimeError("FoldArtifactTransform must be fitted before transform")
        started = time.perf_counter()
        values = np.asarray(window, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != 1:
            raise ValueError(f"Expected [1, channels, time], got {values.shape}")
        signal = values[0].T
        if self.uses_faster:
            signal = apply_faster(signal, self.config.faster)
        if self.uses_ica:
            if self.ica_ is None:
                raise RuntimeError("ICA variant has no fitted ArtifactICA state")
            signal = self.ica_.transform(signal)
        result = np.asarray(signal.T[None, :, :], dtype=np.float32)
        if result.shape != values.shape or not np.isfinite(result).all():
            raise ValueError(f"Artifact transform produced invalid window {result.shape}")
        absolute_delta = np.abs(result.astype(np.float64) - values.astype(np.float64))
        max_delta = float(np.max(absolute_delta))
        if max_delta > 1e-7:
            self.changed_windows_ += 1
        self.max_abs_delta_ = max(self.max_abs_delta_, max_delta)
        self.mean_abs_delta_sum_ += float(np.mean(absolute_delta))
        self.transform_calls_ += 1
        self.transform_seconds_ += time.perf_counter() - started
        return np.ascontiguousarray(result)

    def runtime_diagnostics(self) -> dict[str, Any]:
        """Return transformation counters without changing fitted state."""
        return {
            "transform_calls": int(self.transform_calls_),
            "changed_windows": int(self.changed_windows_),
            "changed_window_fraction": (
                float(self.changed_windows_ / self.transform_calls_)
                if self.transform_calls_ else 0.0
            ),
            "transform_seconds": float(self.transform_seconds_),
            "max_abs_delta": float(self.max_abs_delta_),
            "mean_abs_delta": (
                float(self.mean_abs_delta_sum_ / self.transform_calls_)
                if self.transform_calls_ else 0.0
            ),
        }


class ArtifactTransformedRawView:
    """Lazy raw-window view applying one immutable fold transform."""

    is_lazy_raw_eeg = True

    def __init__(
        self,
        base: Any,
        transform: FoldArtifactTransform,
        *,
        channel_mean: np.ndarray | None = None,
        channel_scale: np.ndarray | None = None,
        cache_transformed_windows: bool = False,
        _shared_cache: dict[str, np.ndarray] | None = None,
        _shared_cache_stats: dict[str, int] | None = None,
    ) -> None:
        if transform.manifest_ is None:
            raise RuntimeError("Artifact transform must be fitted before wrapping a view")
        self.base = base
        self.transform = transform
        self.manifest = base.manifest.reset_index(drop=True).copy()
        self.shape = tuple(base.shape)
        self.ndim = 4
        self.dtype = np.dtype(np.float32)
        self.channel_mean = (
            None if channel_mean is None else np.asarray(channel_mean, dtype=np.float32)
        )
        self.channel_scale = (
            None if channel_scale is None else np.asarray(channel_scale, dtype=np.float32)
        )
        self.cache_transformed_windows = bool(cache_transformed_windows)
        self._cache = _shared_cache if _shared_cache is not None else {}
        self._cache_stats = (
            _shared_cache_stats
            if _shared_cache_stats is not None
            else {"hits": 0, "misses": 0}
        )
        if (self.channel_mean is None) != (self.channel_scale is None):
            raise ValueError("channel_mean and channel_scale must be set together")

    def __len__(self) -> int:
        return self.shape[0]

    def _read_scalar(self, index: int) -> np.ndarray:
        sample_id = str(self.manifest.iloc[int(index)]["sample_id"])
        if self.cache_transformed_windows and sample_id in self._cache:
            self._cache_stats["hits"] += 1
            result = self._cache[sample_id]
        else:
            self._cache_stats["misses"] += 1
            result = self.transform.transform_window(self.base[int(index)])
            if self.cache_transformed_windows:
                self._cache[sample_id] = result
        if self.channel_mean is not None and self.channel_scale is not None:
            result = (
                result - self.channel_mean[None, :, None]
            ) / self.channel_scale[None, :, None]
        if not np.isfinite(result).all():
            raise ValueError("Normalized artifact-transformed window is non-finite")
        return np.ascontiguousarray(result, dtype=np.float32)

    def __getitem__(self, index: Any) -> Any:
        if np.isscalar(index):
            scalar = int(index)
            if scalar < 0:
                scalar += len(self)
            if scalar < 0 or scalar >= len(self):
                raise IndexError(scalar)
            return self._read_scalar(scalar)
        indices = np.arange(len(self))[index]
        return ArtifactTransformedRawView(
            self.base[np.asarray(indices, dtype=np.int64)], self.transform,
            channel_mean=self.channel_mean, channel_scale=self.channel_scale,
            cache_transformed_windows=self.cache_transformed_windows,
            _shared_cache=self._cache,
            _shared_cache_stats=self._cache_stats,
        )

    def with_channel_normalization(
        self, mean: np.ndarray, scale: np.ndarray
    ) -> "ArtifactTransformedRawView":
        return ArtifactTransformedRawView(
            self.base, self.transform, channel_mean=mean, channel_scale=scale,
            cache_transformed_windows=self.cache_transformed_windows,
            _shared_cache=self._cache,
            _shared_cache_stats=self._cache_stats,
        )

    def cache_diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self.cache_transformed_windows,
            "entries": len(self._cache),
            "hits": int(self._cache_stats["hits"]),
            "misses": int(self._cache_stats["misses"]),
            "estimated_bytes": int(
                len(self._cache) * np.prod(self.shape[1:]) * self.dtype.itemsize
            ),
        }

    def compute_channel_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        total = np.zeros(self.shape[2], dtype=np.float64)
        total_squares = np.zeros(self.shape[2], dtype=np.float64)
        count = 0
        for index in range(len(self)):
            window = self._read_scalar(index)[0].astype(np.float64, copy=False)
            total += window.sum(axis=1)
            total_squares += np.square(window).sum(axis=1)
            count += window.shape[1]
        mean = total / count
        variance = np.maximum(total_squares / count - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale[scale < 1e-8] = 1.0
        return mean.astype(np.float32), scale.astype(np.float32)
