"""Resolved whole-record preprocessing contracts for COG-BCI diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


SPEC_VERSION = "cog-bci-whole-record-preprocessing-v1"
OPERATION_ORDER = ("demean", "bandpass", "notch")
VARIANT_ORDER = (
    "A_raw",
    "B_record_demean",
    "C_notch",
    "D_bandpass",
    "E_demean_notch",
    "F_demean_bandpass",
    "G_bandpass_notch",
    "H_demean_bandpass_notch",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sos_default_padlen(sos: np.ndarray) -> int:
    """Resolve SciPy's SOS pad length instead of leaving it implicit."""

    zeros_at_origin = int((sos[:, 2] == 0).sum())
    poles_at_origin = int((sos[:, 5] == 0).sum())
    return 3 * (2 * len(sos) + 1 - min(zeros_at_origin, poles_at_origin))


@dataclass(frozen=True)
class COGBCIWholeRecordPreprocessing:
    """Stateless processing applied to ``[channel, time]`` before windowing."""

    variant_id: str
    name: str
    demean: bool
    bandpass_enabled: bool
    notch_enabled: bool
    sampling_rate_hz: float = 500.0
    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 45.0
    bandpass_order: int = 4
    notch_frequency_hz: float = 50.0
    notch_q: float = 30.0
    padtype: str = "odd"

    def __post_init__(self) -> None:
        if self.variant_id not in VARIANT_ORDER:
            raise ValueError(f"Unknown preprocessing variant {self.variant_id!r}")
        if not np.isclose(self.sampling_rate_hz, 500.0):
            raise ValueError("COG-BCI preprocessing audit requires 500 Hz")
        if self.bandpass_order != 4:
            raise ValueError("The registered band-pass contract requires order 4")
        if not (
            0
            < self.bandpass_low_hz
            < self.bandpass_high_hz
            < self.sampling_rate_hz / 2
        ):
            raise ValueError("Invalid band-pass frequencies")
        if not 0 < self.notch_frequency_hz < self.sampling_rate_hz / 2:
            raise ValueError("Invalid notch frequency")
        if self.notch_q <= 0:
            raise ValueError("notch_q must be positive")
        if self.padtype != "odd":
            raise ValueError("Only explicit odd-reflection padding is supported")

    @property
    def is_identity(self) -> bool:
        return not (self.demean or self.bandpass_enabled or self.notch_enabled)

    def _bandpass_sos(self) -> np.ndarray:
        return butter(
            self.bandpass_order,
            [self.bandpass_low_hz, self.bandpass_high_hz],
            btype="bandpass",
            fs=self.sampling_rate_hz,
            output="sos",
        )

    def _notch_coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        return iirnotch(
            self.notch_frequency_hz,
            self.notch_q,
            fs=self.sampling_rate_hz,
        )

    def to_dict(self) -> dict[str, Any]:
        sos = self._bandpass_sos()
        notch_b, notch_a = self._notch_coefficients()
        operations = [
            name
            for name, enabled in (
                ("demean", self.demean),
                ("bandpass", self.bandpass_enabled),
                ("notch", self.notch_enabled),
            )
            if enabled
        ]
        return {
            "schema_version": SPEC_VERSION,
            "variant_id": self.variant_id,
            "name": self.name,
            "scope": "whole_continuous_physical_record_before_windowing",
            "operation_order": operations or ["identity"],
            "output_dtype": "float32",
            "sampling_rate_hz": float(self.sampling_rate_hz),
            "filter_library": "scipy.signal",
            "library_version": scipy.__version__,
            "demean": {
                "enabled": bool(self.demean),
                "axis": "full_record_time_per_channel",
            },
            "bandpass": {
                "enabled": bool(self.bandpass_enabled),
                "type": "Butterworth IIR band-pass",
                "low_hz": float(self.bandpass_low_hz),
                "high_hz": float(self.bandpass_high_hz),
                "prototype_order": int(self.bandpass_order),
                "realized_sos_sections": int(len(sos)),
                "phase": "zero_phase_forward_backward",
                "transition_bandwidth_hz": "not_applicable_iir",
                "filter_length": "infinite_impulse_response",
                "padding": {
                    "type": self.padtype,
                    "padlen_samples": _sos_default_padlen(sos),
                },
                "sos_coefficients": sos.tolist(),
            },
            "notch": {
                "enabled": bool(self.notch_enabled),
                "method": "second_order_IIR_notch",
                "frequency_hz": float(self.notch_frequency_hz),
                "quality_factor": float(self.notch_q),
                "width_hz": float(self.notch_frequency_hz / self.notch_q),
                "phase": "zero_phase_forward_backward",
                "transition_bandwidth_hz": "not_applicable_iir",
                "filter_length": "infinite_impulse_response",
                "padding": {
                    "type": self.padtype,
                    "padlen_samples": 3 * max(len(notch_a), len(notch_b)),
                },
                "numerator": notch_b.tolist(),
                "denominator": notch_a.tolist(),
            },
        }

    def stable_hash(
        self,
        *,
        channels: Sequence[str] = (),
        loader_schema_version: str = "",
    ) -> str:
        payload = {
            "preprocessing": self.to_dict(),
            "channels": [str(channel) for channel in channels],
            "loader_schema_version": str(loader_schema_version),
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def apply(
        self,
        signals: np.ndarray,
        *,
        sampling_rate: float,
    ) -> np.ndarray:
        if not np.isclose(float(sampling_rate), self.sampling_rate_hz):
            raise ValueError(
                f"sampling_rate={sampling_rate} does not match "
                f"{self.sampling_rate_hz}"
            )
        array = np.asarray(signals)
        if array.ndim != 2:
            raise ValueError(
                f"Whole-record preprocessing expects [channels, time], got "
                f"{array.shape}"
            )
        if array.shape[0] < 1 or array.shape[1] < 2:
            raise ValueError(f"Whole-record input is too short: {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("Whole-record input contains NaN or Inf")
        result = np.asarray(array, dtype=np.float64).copy()
        if self.demean:
            result -= result.mean(axis=1, keepdims=True)
        if self.bandpass_enabled:
            sos = self._bandpass_sos()
            result = sosfiltfilt(
                sos,
                result,
                axis=1,
                padtype=self.padtype,
                padlen=_sos_default_padlen(sos),
            )
        if self.notch_enabled:
            numerator, denominator = self._notch_coefficients()
            result = filtfilt(
                numerator,
                denominator,
                result,
                axis=1,
                padtype=self.padtype,
                padlen=3 * max(len(numerator), len(denominator)),
            )
        output = np.ascontiguousarray(result, dtype=np.float32)
        if output.shape != array.shape:
            raise RuntimeError("Whole-record preprocessing changed signal shape")
        if not np.isfinite(output).all():
            raise ValueError("Whole-record preprocessing produced NaN or Inf")
        return output


def build_preprocessing_variants(
    documents: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[COGBCIWholeRecordPreprocessing, ...]:
    """Build the fixed A-H matrix, optionally validating config documents."""

    defaults = (
        ("A_raw", "raw", False, False, False),
        ("B_record_demean", "record_demean", True, False, False),
        ("C_notch", "notch", False, False, True),
        ("D_bandpass", "bandpass", False, True, False),
        ("E_demean_notch", "demean_notch", True, False, True),
        ("F_demean_bandpass", "demean_bandpass", True, True, False),
        ("G_bandpass_notch", "bandpass_notch", False, True, True),
        (
            "H_demean_bandpass_notch",
            "demean_bandpass_notch",
            True,
            True,
            True,
        ),
    )
    configured = {
        str(item["variant_id"]): dict(item)
        for item in (documents or ())
    }
    if configured and set(configured) != set(VARIANT_ORDER):
        raise ValueError("Configured preprocessing matrix must contain exactly A-H")
    variants = []
    for variant_id, name, demean, bandpass, notch in defaults:
        item = configured.get(variant_id, {})
        expected = {
            "name": name,
            "demean": demean,
            "bandpass_enabled": bandpass,
            "notch_enabled": notch,
        }
        for key, value in expected.items():
            if key in item and item[key] != value:
                raise ValueError(
                    f"{variant_id}.{key} must be {value!r}, got {item[key]!r}"
                )
        variants.append(
            COGBCIWholeRecordPreprocessing(
                variant_id=variant_id,
                name=name,
                demean=demean,
                bandpass_enabled=bandpass,
                notch_enabled=notch,
                sampling_rate_hz=float(item.get("sampling_rate_hz", 500.0)),
                bandpass_low_hz=float(item.get("bandpass_low_hz", 1.0)),
                bandpass_high_hz=float(item.get("bandpass_high_hz", 45.0)),
                bandpass_order=int(item.get("bandpass_order", 4)),
                notch_frequency_hz=float(
                    item.get("notch_frequency_hz", 50.0)
                ),
                notch_q=float(item.get("notch_q", 30.0)),
            )
        )
    return tuple(variants)
