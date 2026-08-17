"""Composable preprocessing pipeline for offline EEG model training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .artifact_removal import (
    ArtifactICA,
    FasterConfig,
    detect_bad_channels,
    interpolate_channels,
)
from .denoising import (
    WaveletDenoisingConfig,
    WaveletDenoisingReport,
    detrend_signal,
    wavelet_denoise,
)
from .eog import EOGRegression, EOGRegressionReport, _mean_absolute_correlation
from .filtering import FilterConfig, apply_causal, apply_offline
from .referencing import (
    ReferenceMethod,
    ReferenceReport,
    common_average_reference,
    rereference,
)


@dataclass
class OfflinePreprocessingConfig:
    """Configuration whose conservative operations are enabled by default.

    Wavelet shrinkage and EOG regression are opt-in because their benefit is
    data-dependent and they can suppress useful EEG components.
    """

    sample_rate: float
    detrend_order: int | None = 1
    apply_filter: bool = True
    filter_mode: Literal["zero_phase", "causal"] = "zero_phase"
    filter_config: FilterConfig | None = None
    reference_method: ReferenceMethod = "robust_average"
    robust_reference_z_threshold: float = 3.0
    robust_reference_max_iterations: int = 5
    detect_and_interpolate_bad_channels: bool = True
    faster_config: FasterConfig = field(default_factory=FasterConfig)
    wavelet_config: WaveletDenoisingConfig | None = None
    use_eog_regression: bool = False
    eog_ridge_alpha: float = 1e-6

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.filter_mode not in {"zero_phase", "causal"}:
            raise ValueError("filter_mode must be 'zero_phase' or 'causal'")
        if self.filter_config is None:
            self.filter_config = FilterConfig(sample_rate=self.sample_rate)
        elif self.filter_config.sample_rate != self.sample_rate:
            raise ValueError("filter_config.sample_rate must match sample_rate")
        if self.eog_ridge_alpha < 0:
            raise ValueError("eog_ridge_alpha cannot be negative")


@dataclass(frozen=True)
class OfflinePreprocessingReport:
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    reference: ReferenceReport
    bad_channels: tuple[int, ...]
    filter_applied: bool
    filter_mode: str
    detrend_order: int | None
    eog_regression: EOGRegressionReport | None
    ica_applied: bool
    wavelet: WaveletDenoisingReport | None


@dataclass(frozen=True)
class OfflinePreprocessingResult:
    values: np.ndarray
    report: OfflinePreprocessingReport


def _as_matrix(signal: object, *, name: str) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError(f"{name} must be a non-empty [samples, channels] matrix")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


class OfflinePreprocessingPipeline:
    """Apply offline-only, zero-phase preprocessing without changing row count.

    The EOG model has an explicit ``fit`` step so cross-validation code cannot
    silently estimate regression coefficients from validation/test records.
    ICA is likewise accepted only as an already fitted transform.
    """

    def __init__(
        self,
        config: OfflinePreprocessingConfig,
        *,
        ica: ArtifactICA | None = None,
        eog_regressor: EOGRegression | None = None,
    ) -> None:
        self.config = config
        self.ica = ica
        self.eog_regressor = eog_regressor

    def _temporal_steps(self, signal: np.ndarray) -> np.ndarray:
        result = signal
        if self.config.detrend_order is not None:
            result = detrend_signal(result, order=self.config.detrend_order)
        if self.config.apply_filter:
            assert self.config.filter_config is not None
            result = (
                apply_causal(result, self.config.filter_config)
                if self.config.filter_mode == "causal"
                else apply_offline(result, self.config.filter_config)
            )
        return result

    def _prepare_eog(self, eog: object, expected_samples: int) -> np.ndarray:
        values = _as_matrix(eog, name="EOG")
        if len(values) != expected_samples:
            raise ValueError("EOG and EEG must contain the same number of samples")
        return self._temporal_steps(values)

    def _prepare_eeg(
        self, eeg: object
    ) -> tuple[np.ndarray, ReferenceReport, tuple[int, ...]]:
        values = _as_matrix(eeg, name="EEG")
        values = self._temporal_steps(values)
        bad_channels: set[int] = set()
        if self.config.detect_and_interpolate_bad_channels:
            bad_channels.update(detect_bad_channels(values, self.config.faster_config))

        if self.config.reference_method == "common_average" and bad_channels:
            values = common_average_reference(values, exclude=bad_channels)
            reference_report = ReferenceReport(
                method="common_average",
                excluded_channels=tuple(sorted(bad_channels)),
            )
        else:
            values, reference_report = rereference(
                values,
                method=self.config.reference_method,
                robust_z_threshold=self.config.robust_reference_z_threshold,
                robust_max_iterations=self.config.robust_reference_max_iterations,
            )
        bad_channels.update(reference_report.excluded_channels)

        if self.config.detect_and_interpolate_bad_channels:
            values = interpolate_channels(values, sorted(bad_channels))
        return values, reference_report, tuple(sorted(bad_channels))

    def fit(self, eeg: object, *, eog: object | None = None) -> "OfflinePreprocessingPipeline":
        """Fit only data-dependent cleanup stages on training/calibration data."""
        if self.config.use_eog_regression:
            if eog is None:
                raise ValueError("EOG data is required when use_eog_regression=True")
            prepared_eeg, _, _ = self._prepare_eeg(eeg)
            prepared_eog = self._prepare_eog(eog, len(prepared_eeg))
            self.eog_regressor = EOGRegression(
                ridge_alpha=self.config.eog_ridge_alpha
            ).fit(prepared_eeg, prepared_eog)
        return self

    def transform(self, eeg: object, *, eog: object | None = None) -> OfflinePreprocessingResult:
        if self.config.use_eog_regression and self.eog_regressor is None:
            raise RuntimeError("Call fit on training/calibration data before transform")
        raw = _as_matrix(eeg, name="EEG")
        values, reference_report, bad_channels = self._prepare_eeg(raw)
        eog_report: EOGRegressionReport | None = None

        if self.config.use_eog_regression:
            if eog is None:
                raise ValueError("EOG data is required when use_eog_regression=True")
            assert self.eog_regressor is not None
            prepared_eog = self._prepare_eog(eog, len(values))
            before = _mean_absolute_correlation(values, prepared_eog)
            values = self.eog_regressor.transform(values, prepared_eog)
            eog_report = EOGRegressionReport(
                eog_channels=prepared_eog.shape[1],
                mean_absolute_correlation_before=before,
                mean_absolute_correlation_after=_mean_absolute_correlation(
                    values, prepared_eog
                ),
            )

        if self.ica is not None:
            values = self.ica.transform(values)

        wavelet_report: WaveletDenoisingReport | None = None
        if self.config.wavelet_config is not None:
            values, wavelet_report = wavelet_denoise(
                values, self.config.wavelet_config
            )

        return OfflinePreprocessingResult(
            values=values,
            report=OfflinePreprocessingReport(
                input_shape=raw.shape,
                output_shape=values.shape,
                reference=reference_report,
                bad_channels=bad_channels,
                filter_applied=self.config.apply_filter,
                filter_mode=self.config.filter_mode,
                detrend_order=self.config.detrend_order,
                eog_regression=eog_report,
                ica_applied=self.ica is not None,
                wavelet=wavelet_report,
            ),
        )

    def fit_transform(
        self, eeg: object, *, eog: object | None = None
    ) -> OfflinePreprocessingResult:
        return self.fit(eeg, eog=eog).transform(eeg, eog=eog)
