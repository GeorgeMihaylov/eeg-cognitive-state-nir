"""Validated, reproducible preprocessing for timestamp-aligned raw EEG."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


RAW_PREPROCESSING_VERSION = "raw-preprocessing-v1"
PREPROCESSING_SPEC_VERSION = "preprocessing-spec-v1"
DEFAULT_BANDPASS_ORDER = 4
DEFAULT_FILTER_PADDING_SECONDS = 2.0
DEFAULT_WINDOW_SECONDS = 10.0

DEFAULT_RAW_PREPROCESSING: dict[str, Any] = {
    "resample_hz": 256.0,
    "bandpass": {
        "enabled": False,
        "low_hz": 1.0,
        "high_hz": 45.0,
    },
    "notch": {
        "enabled": False,
        "frequency_hz": 50.0,
        "quality_factor": 30.0,
    },
    "rereference": {"mode": "none"},
    "artifact_rejection": {
        "enabled": False,
        "max_abs_amplitude": None,
        "max_flat_fraction": None,
    },
}


def normalize_raw_preprocessing(
    config: Mapping[str, Any] | None,
    *,
    default_resample_hz: float = 256.0,
) -> dict[str, Any]:
    """Merge and validate the public raw-preprocessing schema."""
    normalized = deepcopy(DEFAULT_RAW_PREPROCESSING)
    normalized["resample_hz"] = float(default_resample_hz)
    if config is not None:
        unknown = sorted(set(config) - set(normalized))
        if unknown:
            raise ValueError(f"Unknown raw_preprocessing keys: {unknown}")
        for key, value in config.items():
            if isinstance(normalized[key], dict):
                if not isinstance(value, Mapping):
                    raise ValueError(f"raw_preprocessing.{key} must be a mapping")
                unknown_nested = sorted(set(value) - set(normalized[key]))
                if unknown_nested:
                    raise ValueError(
                        f"Unknown raw_preprocessing.{key} keys: {unknown_nested}"
                    )
                normalized[key].update(value)
            else:
                normalized[key] = value

    resample_hz = float(normalized["resample_hz"])
    if not np.isfinite(resample_hz) or resample_hz <= 0:
        raise ValueError("raw_preprocessing.resample_hz must be positive and finite")
    normalized["resample_hz"] = resample_hz
    nyquist = resample_hz / 2.0

    bandpass = normalized["bandpass"]
    bandpass["enabled"] = bool(bandpass["enabled"])
    bandpass["low_hz"] = float(bandpass["low_hz"])
    bandpass["high_hz"] = float(bandpass["high_hz"])
    if bandpass["enabled"] and not (
        0 < bandpass["low_hz"] < bandpass["high_hz"] < nyquist
    ):
        raise ValueError(
            "Enabled bandpass requires 0 < low_hz < high_hz < Nyquist"
        )

    notch = normalized["notch"]
    notch["enabled"] = bool(notch["enabled"])
    notch["frequency_hz"] = float(notch["frequency_hz"])
    notch["quality_factor"] = float(notch["quality_factor"])
    if notch["enabled"] and not 0 < notch["frequency_hz"] < nyquist:
        raise ValueError("Enabled notch frequency_hz must be below Nyquist")
    if notch["enabled"] and notch["quality_factor"] <= 0:
        raise ValueError("Enabled notch quality_factor must be positive")

    rereference = normalized["rereference"]
    mode = str(rereference["mode"]).strip().lower()
    if mode == "car":
        mode = "common_average"
    if mode not in {"none", "common_average"}:
        raise ValueError(
            "raw_preprocessing.rereference.mode must be 'none' or "
            "'common_average'"
        )
    rereference["mode"] = mode

    rejection = normalized["artifact_rejection"]
    rejection["enabled"] = bool(rejection["enabled"])
    for key in ("max_abs_amplitude", "max_flat_fraction"):
        value = rejection[key]
        if value is not None:
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"raw_preprocessing.artifact_rejection.{key} must be "
                    "non-negative and finite"
                )
        rejection[key] = value
    flat_limit = rejection["max_flat_fraction"]
    if flat_limit is not None and flat_limit > 1:
        raise ValueError("max_flat_fraction must be in [0, 1]")
    return normalized


def raw_preprocessing_hash(
    config: Mapping[str, Any] | None,
    *,
    channels: Sequence[str],
    default_resample_hz: float = 256.0,
) -> str:
    """Hash preprocessing plus channel order for cache namespace isolation."""
    payload = {
        "version": RAW_PREPROCESSING_VERSION,
        "channels": [str(channel) for channel in channels],
        "raw_preprocessing": normalize_raw_preprocessing(
            config, default_resample_hz=default_resample_hz
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def preprocessing_variant_name(config: Mapping[str, Any]) -> str:
    """Return a short human-readable cache namespace prefix."""
    parts = ["raw"]
    if config["bandpass"]["enabled"]:
        parts.append("bp")
    if config["notch"]["enabled"]:
        parts.append("notch")
    if config["rereference"]["mode"] == "common_average":
        parts.append("car")
    if config["artifact_rejection"]["enabled"]:
        parts.append("artifact-qc")
    return "-".join(parts)


def apply_raw_preprocessing(
    signals: np.ndarray,
    *,
    sampling_rate: float,
    config: Mapping[str, Any] | None,
) -> np.ndarray:
    """Apply zero-phase filters and rereferencing to ``[channel, time]``."""
    normalized = normalize_raw_preprocessing(
        config, default_resample_hz=sampling_rate
    )
    array = np.asarray(signals, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(
            f"Raw preprocessing expects [channels, time], got {array.shape}"
        )
    if array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError(f"Raw preprocessing input is too short: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Raw preprocessing input contains NaN or Inf")

    result = array.astype(np.float64, copy=True)
    bandpass = normalized["bandpass"]
    if bandpass["enabled"]:
        sos = butter(
            4,
            [bandpass["low_hz"], bandpass["high_hz"]],
            btype="bandpass",
            fs=float(sampling_rate),
            output="sos",
        )
        try:
            result = sosfiltfilt(sos, result, axis=1)
        except ValueError as exc:
            raise ValueError(
                "Raw EEG interval is too short for zero-phase bandpass filtering"
            ) from exc

    notch = normalized["notch"]
    if notch["enabled"]:
        numerator, denominator = iirnotch(
            notch["frequency_hz"],
            notch["quality_factor"],
            fs=float(sampling_rate),
        )
        try:
            result = filtfilt(numerator, denominator, result, axis=1)
        except ValueError as exc:
            raise ValueError(
                "Raw EEG interval is too short for zero-phase notch filtering"
            ) from exc

    if normalized["rereference"]["mode"] == "common_average":
        result = result - result.mean(axis=0, keepdims=True)
    result = np.ascontiguousarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Raw preprocessing produced NaN or Inf")
    return result


def raw_window_artifact_metrics(signals: np.ndarray) -> dict[str, Any]:
    """Compute threshold-independent per-window diagnostics by channel."""
    array = np.asarray(signals, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected [channels, time], got {array.shape}")
    finite = np.isfinite(array)
    differences = np.diff(array, axis=1)
    return {
        "max_abs_amplitude": float(np.nanmax(np.abs(array))),
        "max_flat_fraction": float(
            np.max(np.mean(np.abs(differences) <= 1e-6, axis=1))
        ),
        "channel_peak_to_peak": np.ptp(array, axis=1).astype(float).tolist(),
        "channel_variance": np.var(array, axis=1).astype(float).tolist(),
        "channel_flat_fraction": np.mean(
            np.abs(differences) <= 1e-6, axis=1
        ).astype(float).tolist(),
        "non_finite_fraction": float(1.0 - finite.mean()),
    }


@dataclass(frozen=True)
class PreprocessingComponentDefinition:
    """Registry metadata for one preprocessing component."""

    name: str
    execution_order: int
    parameter_defaults: tuple[tuple[str, Any], ...] = ()
    fit_scope: str = "stateless"
    cacheable: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Preprocessing component name must not be empty")
        if self.fit_scope not in {"stateless", "outer_train_only"}:
            raise ValueError(
                "fit_scope must be 'stateless' or 'outer_train_only', got "
                f"{self.fit_scope!r}"
            )
        names = [name for name, _ in self.parameter_defaults]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate parameter names for {self.name!r}")

    @property
    def parameters(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.parameter_defaults))

    @property
    def stateful(self) -> bool:
        return self.fit_scope != "stateless"


@dataclass(frozen=True)
class PreprocessingStepSpec:
    """Immutable, serializable configuration of one registered step."""

    name: str
    enabled: bool
    parameters: tuple[tuple[str, Any], ...]
    execution_order: int
    fit_scope: str = "stateless"
    cacheable: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Preprocessing step name must not be empty")
        if self.fit_scope not in {"stateless", "outer_train_only"}:
            raise ValueError(
                "fit_scope must be 'stateless' or 'outer_train_only', got "
                f"{self.fit_scope!r}"
            )
        parameter_names = [name for name, _ in self.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError(f"Duplicate parameters for step {self.name!r}")

    @property
    def stateful(self) -> bool:
        return self.fit_scope != "stateless"

    def parameter_dict(self) -> dict[str, Any]:
        return {name: value for name, value in self.parameters}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": bool(self.enabled),
            "parameters": self.parameter_dict(),
            "execution_order": int(self.execution_order),
            "fit_scope": self.fit_scope,
            "stateful": self.stateful,
            "cacheable": bool(self.cacheable),
        }


_PREPROCESSING_REGISTRY: dict[str, PreprocessingComponentDefinition] = {}


def register_preprocessing_component(
    definition: PreprocessingComponentDefinition,
) -> None:
    """Register a component once under a stable public identifier."""
    if definition.name in _PREPROCESSING_REGISTRY:
        raise ValueError(
            f"Preprocessing component {definition.name!r} is already registered"
        )
    if any(
        existing.execution_order == definition.execution_order
        for existing in _PREPROCESSING_REGISTRY.values()
    ):
        raise ValueError(
            f"Execution order {definition.execution_order} is already registered"
        )
    _PREPROCESSING_REGISTRY[definition.name] = definition


def get_preprocessing_component(
    name: str,
) -> PreprocessingComponentDefinition:
    try:
        return _PREPROCESSING_REGISTRY[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown preprocessing component {name!r}; available: "
            f"{sorted(_PREPROCESSING_REGISTRY)}"
        ) from exc


def get_preprocessing_registry(
) -> Mapping[str, PreprocessingComponentDefinition]:
    return MappingProxyType(dict(_PREPROCESSING_REGISTRY))


register_preprocessing_component(
    PreprocessingComponentDefinition("identity", execution_order=0)
)
register_preprocessing_component(
    PreprocessingComponentDefinition(
        "bandpass",
        execution_order=10,
        parameter_defaults=(
            ("low_hz", 1.0),
            ("high_hz", 45.0),
            ("order", DEFAULT_BANDPASS_ORDER),
        ),
    )
)
register_preprocessing_component(
    PreprocessingComponentDefinition(
        "notch",
        execution_order=20,
        parameter_defaults=(("frequency_hz", 50.0), ("q", 30.0)),
    )
)
register_preprocessing_component(
    PreprocessingComponentDefinition("car", execution_order=30)
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _build_step(
    definition: PreprocessingComponentDefinition,
    value: Mapping[str, Any] | None,
    *,
    enabled_default: bool = False,
) -> PreprocessingStepSpec:
    configured = {} if value is None else dict(value)
    allowed = {"enabled", *definition.parameters.keys()}
    unknown = sorted(set(configured) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown preprocessing.{definition.name} parameters: {unknown}"
        )
    enabled = bool(configured.pop("enabled", enabled_default))
    parameters = dict(definition.parameters)
    parameters.update(configured)
    return PreprocessingStepSpec(
        name=definition.name,
        enabled=enabled,
        parameters=tuple(sorted(parameters.items())),
        execution_order=definition.execution_order,
        fit_scope=definition.fit_scope,
        cacheable=definition.cacheable,
    )


@dataclass(frozen=True)
class PreprocessingSpec:
    """Immutable semantic specification independent of YAML file names."""

    steps: tuple[PreprocessingStepSpec, ...]
    target_sampling_rate: float = 256.0
    padding_seconds: float = DEFAULT_FILTER_PADDING_SECONDS
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    output_dtype: str = "float32"
    schema_version: str = PREPROCESSING_SPEC_VERSION

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.steps, key=lambda step: step.execution_order))
        if ordered != self.steps:
            object.__setattr__(self, "steps", ordered)
        names = [step.name for step in self.steps]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate preprocessing steps: {names}")
        orders = [step.execution_order for step in self.steps]
        if len(orders) != len(set(orders)):
            raise ValueError(f"Duplicate preprocessing execution orders: {orders}")
        if not np.isfinite(self.target_sampling_rate) or self.target_sampling_rate <= 0:
            raise ValueError("target_sampling_rate must be positive and finite")
        if not np.isfinite(self.padding_seconds) or self.padding_seconds < 0:
            raise ValueError("padding_seconds must be non-negative and finite")
        if not np.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive and finite")
        if self.output_dtype != "float32":
            raise ValueError("The current raw EEG pipeline only supports float32")
        self._validate_component_values()

    def _validate_component_values(self) -> None:
        bandpass = self.step("bandpass")
        bandpass_parameters = bandpass.parameter_dict()
        low_hz = float(bandpass_parameters["low_hz"])
        high_hz = float(bandpass_parameters["high_hz"])
        order = int(bandpass_parameters["order"])
        if bandpass.enabled and not (
            0 < low_hz < high_hz < self.target_sampling_rate / 2.0
        ):
            raise ValueError(
                "Enabled bandpass requires 0 < low_hz < high_hz < Nyquist"
            )
        if order != DEFAULT_BANDPASS_ORDER:
            raise ValueError(
                "The current raw EEG implementation requires bandpass order 4"
            )

        notch = self.step("notch")
        notch_parameters = notch.parameter_dict()
        frequency_hz = float(notch_parameters["frequency_hz"])
        quality_factor = float(notch_parameters["q"])
        if notch.enabled and not 0 < frequency_hz < self.target_sampling_rate / 2.0:
            raise ValueError("Enabled notch frequency_hz must be below Nyquist")
        if notch.enabled and quality_factor <= 0:
            raise ValueError("Enabled notch q must be positive")

        car = self.step("car")
        if car.parameters:
            raise ValueError("CAR does not accept parameters")
        identity = self.step("identity")
        if identity.parameters:
            raise ValueError("identity does not accept parameters")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PreprocessingSpec":
        document = dict(value)
        if "preprocessing" in document:
            if len(document) != 1:
                unknown_document = sorted(set(document) - {"preprocessing"})
                raise ValueError(
                    "When 'preprocessing' is present it must be the only root key; "
                    f"got additional keys {unknown_document}"
                )
            nested = document["preprocessing"]
            if not isinstance(nested, Mapping):
                raise ValueError("preprocessing must be a mapping")
            document = dict(nested)

        scalar_keys = {
            "schema_version",
            "target_sampling_rate",
            "padding_seconds",
            "window_seconds",
            "output_dtype",
        }
        component_names = set(_PREPROCESSING_REGISTRY)
        unknown = sorted(set(document) - scalar_keys - component_names)
        if unknown:
            raise ValueError(f"Unknown preprocessing parameters: {unknown}")

        steps: list[PreprocessingStepSpec] = []
        for definition in sorted(
            _PREPROCESSING_REGISTRY.values(),
            key=lambda component: component.execution_order,
        ):
            configured = document.get(definition.name)
            if configured is not None and not isinstance(configured, Mapping):
                raise ValueError(
                    f"preprocessing.{definition.name} must be a mapping"
                )
            steps.append(
                _build_step(
                    definition,
                    configured,
                    enabled_default=(definition.name == "identity"),
                )
            )

        transform_enabled = any(
            step.enabled for step in steps if step.name != "identity"
        )
        identity_index = next(
            index for index, step in enumerate(steps) if step.name == "identity"
        )
        steps[identity_index] = replace(
            steps[identity_index], enabled=not transform_enabled
        )
        return cls(
            steps=tuple(steps),
            target_sampling_rate=float(document.get("target_sampling_rate", 256.0)),
            padding_seconds=float(
                document.get("padding_seconds", DEFAULT_FILTER_PADDING_SECONDS)
            ),
            window_seconds=float(
                document.get("window_seconds", DEFAULT_WINDOW_SECONDS)
            ),
            output_dtype=str(document.get("output_dtype", "float32")),
            schema_version=str(
                document.get("schema_version", PREPROCESSING_SPEC_VERSION)
            ),
        )

    @classmethod
    def from_legacy_raw_preprocessing(
        cls,
        value: Mapping[str, Any] | None,
        *,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        padding_seconds: float = DEFAULT_FILTER_PADDING_SECONDS,
    ) -> "PreprocessingSpec":
        normalized = normalize_raw_preprocessing(value)
        rejection = normalized["artifact_rejection"]
        if rejection["enabled"]:
            raise ValueError(
                "Artifact rejection is outside the preprocessing ablation spec"
            )
        return cls.from_dict(
            {
                "target_sampling_rate": normalized["resample_hz"],
                "padding_seconds": padding_seconds,
                "window_seconds": window_seconds,
                "output_dtype": "float32",
                "bandpass": {
                    "enabled": normalized["bandpass"]["enabled"],
                    "low_hz": normalized["bandpass"]["low_hz"],
                    "high_hz": normalized["bandpass"]["high_hz"],
                    "order": DEFAULT_BANDPASS_ORDER,
                },
                "notch": {
                    "enabled": normalized["notch"]["enabled"],
                    "frequency_hz": normalized["notch"]["frequency_hz"],
                    "q": normalized["notch"]["quality_factor"],
                },
                "car": {
                    "enabled": normalized["rereference"]["mode"]
                    == "common_average"
                },
            }
        )

    def step(self, name: str) -> PreprocessingStepSpec:
        for step in self.steps:
            if step.name == name:
                return step
        raise ValueError(
            f"Preprocessing spec has no {name!r} step; available: "
            f"{[step.name for step in self.steps]}"
        )

    @property
    def effective_padding_seconds(self) -> float:
        return (
            float(self.padding_seconds)
            if self.step("bandpass").enabled or self.step("notch").enabled
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_sampling_rate": float(self.target_sampling_rate),
            "padding_seconds": float(self.padding_seconds),
            "effective_padding_seconds": self.effective_padding_seconds,
            "window_seconds": float(self.window_seconds),
            "output_dtype": self.output_dtype,
            "steps": [step.to_dict() for step in self.steps],
        }

    def stable_serialization(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(
        self,
        *,
        channels: Sequence[str] | None = None,
        loader_schema_version: str | None = None,
        source_identity: Mapping[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {"preprocessing": self.to_dict()}
        if channels is not None:
            payload["channels"] = [str(channel) for channel in channels]
        if loader_schema_version is not None:
            payload["loader_schema_version"] = str(loader_schema_version)
        if source_identity is not None:
            payload["source_identity"] = dict(source_identity)
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def to_legacy_raw_preprocessing(self) -> dict[str, Any]:
        bandpass = self.step("bandpass")
        bandpass_parameters = bandpass.parameter_dict()
        notch = self.step("notch")
        notch_parameters = notch.parameter_dict()
        return normalize_raw_preprocessing(
            {
                "resample_hz": float(self.target_sampling_rate),
                "bandpass": {
                    "enabled": bandpass.enabled,
                    "low_hz": float(bandpass_parameters["low_hz"]),
                    "high_hz": float(bandpass_parameters["high_hz"]),
                },
                "notch": {
                    "enabled": notch.enabled,
                    "frequency_hz": float(notch_parameters["frequency_hz"]),
                    "quality_factor": float(notch_parameters["q"]),
                },
                "rereference": {
                    "mode": (
                        "common_average" if self.step("car").enabled else "none"
                    )
                },
                "artifact_rejection": {
                    "enabled": False,
                    "max_abs_amplitude": None,
                    "max_flat_fraction": None,
                },
            },
            default_resample_hz=self.target_sampling_rate,
        )

    def assert_global_cacheable(self) -> None:
        unsafe = [
            step.name
            for step in self.steps
            if step.enabled and (step.fit_scope != "stateless" or not step.cacheable)
        ]
        if unsafe:
            raise ValueError(
                "Global raw cache requires enabled stateless, cacheable steps; "
                f"unsafe steps: {unsafe}"
            )


def apply_preprocessing_spec(
    signals: np.ndarray,
    *,
    sampling_rate: float,
    spec: PreprocessingSpec,
) -> np.ndarray:
    """Execute a typed spec through the established numerical pipeline."""
    if not np.isclose(float(sampling_rate), spec.target_sampling_rate):
        raise ValueError(
            f"sampling_rate={sampling_rate} does not match spec target "
            f"{spec.target_sampling_rate}"
        )
    spec.assert_global_cacheable()
    return apply_raw_preprocessing(
        signals,
        sampling_rate=sampling_rate,
        config=spec.to_legacy_raw_preprocessing(),
    )
