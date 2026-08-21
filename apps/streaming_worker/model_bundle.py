"""Load validated raw-EEG ShallowConvNet or handcrafted-feature bundles."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from cogstate.features import FEATURE_SCHEMA_VERSION
from cogstate.model_zoo.factory import build_model
from cogstate.model_zoo.multitask import PMMultiTaskClassifier
from cogstate.model_zoo.weights import load_torch_weights
from cogstate.protocol import PM_METRICS


DEFAULT_CLASS_NAMES = ("low", "medium", "high")
RAW_EEG_INPUT_MODE = "raw_eeg"
SHALLOW_CONVNET_MODEL_TYPE = "torch_shallow_convnet_multitask"


@dataclass(frozen=True)
class FeatureModelManifest:
    version: str
    model_type: str
    n_features: int
    sample_rate: float
    channels: tuple[str, ...]
    window_seconds: float
    feature_profile: str
    diagnostic_only: bool = False
    model_file: str = "model.joblib"
    scaler_file: str | None = "scaler.joblib"
    selector_file: str | None = None
    feature_schema_version: str = "legacy"
    input_mode: str = "features"
    bootstrap: bool = False
    description: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "FeatureModelManifest":
        values = dict(payload)
        values["channels"] = tuple(values.get("channels", ()))
        return cls(**values)

    @classmethod
    def from_path(cls, path: Path) -> "FeatureModelManifest":
        return cls.from_payload(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class RawEEGModelManifest:
    version: str
    model_type: str
    sample_rate: float
    channels: tuple[str, ...]
    window_seconds: float
    n_times: int
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES
    target_names: tuple[str, ...] = PM_METRICS
    input_mode: str = RAW_EEG_INPUT_MODE
    input_layout: str = "batch,1,channels,time"
    diagnostic_only: bool = False
    model_file: str = "model.pt"
    bootstrap: bool = False
    preprocessing: dict[str, Any] = field(default_factory=dict)
    description: str | None = None

    @classmethod
    def from_path(cls, path: Path) -> "RawEEGModelManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["channels"] = tuple(payload.get("channels", ()))
        payload["class_names"] = tuple(
            payload.get("class_names", DEFAULT_CLASS_NAMES)
        )
        payload["target_names"] = tuple(payload.get("target_names", PM_METRICS))
        return cls(**payload)


class FeatureModelBundle:
    """Apply optional feature transforms and expose PM probabilities."""

    def __init__(
        self,
        estimator: PMMultiTaskClassifier,
        *,
        manifest: FeatureModelManifest,
        scaler: Any | None = None,
        selector: Any | None = None,
    ) -> None:
        self.estimator = estimator
        self.manifest = manifest
        self.scaler = scaler
        self.selector = selector
        self.version = manifest.version

    def predict_pm_proba(self, features: np.ndarray) -> dict[str, dict[str, float]]:
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.manifest.n_features:
            raise ValueError(
                f"Expected {self.manifest.n_features} features, got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Model features contain non-finite values")
        if self.scaler is not None:
            values = self.scaler.transform(values)
        if self.selector is not None:
            values = self.selector.transform(values)

        raw = self.estimator.predict_proba(values)
        result: dict[str, dict[str, float]] = {}
        for metric, probabilities in raw.items():
            model = self.estimator.models_[metric]
            result[metric] = {
                DEFAULT_CLASS_NAMES[int(label)]: float(probability)
                for label, probability in zip(model.classes_, probabilities[0])
            }
        return result


# Backward-compatible names for callers that use the original feature bundle.
ModelManifest = FeatureModelManifest
BundlePMModel = FeatureModelBundle


class ShallowConvNetBundle:
    """Expose seven raw-EEG PM classification heads to ``InferenceService``."""

    def __init__(self, estimator: Any, *, manifest: RawEEGModelManifest) -> None:
        self.estimator = estimator
        self.manifest = manifest
        self.version = manifest.version

    def predict_pm_proba(
        self, eeg_window: np.ndarray
    ) -> dict[str, dict[str, float]]:
        values = np.asarray(eeg_window, dtype=np.float32)
        expected = (1, len(self.manifest.channels), self.manifest.n_times)
        if values.shape == expected:
            values = values[None, ...]
        if values.shape != (1, *expected):
            raise ValueError(
                "ShallowConvNet expects one EEG window with shape "
                f"{expected} (or batched {(1, *expected)}), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("EEG model input contains non-finite values")
        raw = self.estimator.predict_proba(values)
        if set(raw) != set(self.manifest.target_names):
            raise ValueError("Model targets do not match manifest.target_names")
        result: dict[str, dict[str, float]] = {}
        for metric in self.manifest.target_names:
            probabilities = np.asarray(raw[metric])[0]
            if len(probabilities) != len(self.manifest.class_names):
                raise ValueError(
                    f"Model output count for {metric!r} does not match class_names"
                )
            result[metric] = dict(
                zip(self.manifest.class_names, map(float, probabilities))
            )
        return result


def _validate_raw_manifest(
    manifest: RawEEGModelManifest,
    *,
    sample_rate: float,
    channels: tuple[str, ...],
    window_seconds: float,
    preprocessing: dict[str, Any],
) -> None:
    errors: list[str] = []
    expected_n_times = int(round(sample_rate * window_seconds))
    if manifest.model_type != SHALLOW_CONVNET_MODEL_TYPE:
        errors.append(
            f"model_type {manifest.model_type!r} != {SHALLOW_CONVNET_MODEL_TYPE!r}"
        )
    if manifest.input_mode != RAW_EEG_INPUT_MODE:
        errors.append(f"input_mode {manifest.input_mode!r} != 'raw_eeg'")
    if manifest.input_layout != "batch,1,channels,time":
        errors.append(f"unsupported input_layout {manifest.input_layout!r}")
    if not np.isclose(manifest.sample_rate, sample_rate):
        errors.append(f"sample_rate {manifest.sample_rate} != {sample_rate}")
    if manifest.channels != channels:
        errors.append("channel order differs")
    if not np.isclose(manifest.window_seconds, window_seconds):
        errors.append(
            f"window_seconds {manifest.window_seconds} != {window_seconds}"
        )
    if manifest.n_times != expected_n_times:
        errors.append(f"n_times {manifest.n_times} != {expected_n_times}")
    if len(manifest.class_names) < 2 or len(set(manifest.class_names)) != len(
        manifest.class_names
    ):
        errors.append("class_names must contain at least two unique names")
    if manifest.target_names != PM_METRICS:
        errors.append(
            "target_names must contain all seven PM metrics in protocol order"
        )
    for name, expected_value in preprocessing.items():
        if name not in manifest.preprocessing:
            errors.append(f"preprocessing.{name} is missing")
            continue
        actual_value = manifest.preprocessing[name]
        if isinstance(expected_value, (int, float)) and not isinstance(
            expected_value, bool
        ):
            if not np.isclose(float(actual_value), float(expected_value)):
                errors.append(
                    f"preprocessing.{name} {actual_value} != {expected_value}"
                )
        elif actual_value != expected_value:
            errors.append(
                f"preprocessing.{name} {actual_value!r} != {expected_value!r}"
            )
    if errors:
        raise ValueError("Incompatible ShallowConvNet bundle: " + "; ".join(errors))


def _validate_estimator(estimator: Any, manifest: RawEEGModelManifest) -> None:
    expected_shape = (1, len(manifest.channels), manifest.n_times)
    actual_shape = tuple(getattr(estimator, "input_shape", ()))
    if actual_shape != expected_shape:
        raise ValueError(
            f"Model input_shape {actual_shape} does not match manifest {expected_shape}"
        )
    num_classes = int(getattr(estimator, "num_classes", 0))
    if num_classes != len(manifest.class_names):
        raise ValueError(
            f"Model has {num_classes} outputs, manifest has "
            f"{len(manifest.class_names)} class names"
        )
    metadata = dict(getattr(estimator, "model_metadata", {}))
    if metadata.get("model_type") != manifest.model_type:
        raise ValueError(
            "Saved model type does not match manifest: "
            f"{metadata.get('model_type')!r} != {manifest.model_type!r}"
        )
    if not np.isclose(float(metadata.get("sampling_rate", np.nan)), manifest.sample_rate):
        raise ValueError("Saved model sampling_rate does not match manifest")
    if tuple(metadata.get("channel_names", ())) != manifest.channels:
        raise ValueError("Saved model channel order does not match manifest")
    metric_names = tuple(getattr(estimator, "metric_names", ()))
    if metric_names != manifest.target_names:
        raise ValueError(
            f"Saved model targets {metric_names} do not match manifest "
            f"{manifest.target_names}"
        )


def _bootstrap_raw_bundle(
    manifest: RawEEGModelManifest, *, device: str
) -> ShallowConvNetBundle:
    """Build an untrained diagnostic network for end-to-end smoke tests only."""
    estimator = build_model(
        SHALLOW_CONVNET_MODEL_TYPE,
        "classification",
        (1, len(manifest.channels), manifest.n_times),
        len(manifest.class_names),
        {
            "sampling_rate": manifest.sample_rate,
            "channel_names": list(manifest.channels),
            "metric_names": list(manifest.target_names),
            "standardize": False,
            "device": device,
            "random_state": 42,
        },
    )
    estimator.is_fitted_ = True
    estimator.model.eval()
    return ShallowConvNetBundle(estimator, manifest=manifest)


def _validate_feature_manifest(
    manifest: FeatureModelManifest,
    *,
    n_features: int,
    sample_rate: float,
    channels: tuple[str, ...],
    window_seconds: float,
    feature_profile: str,
    feature_schema_version: str,
) -> None:
    errors: list[str] = []
    if manifest.input_mode != "features":
        errors.append(f"input_mode {manifest.input_mode!r} != 'features'")
    if manifest.n_features != n_features:
        errors.append(f"features {manifest.n_features} != {n_features}")
    if not np.isclose(manifest.sample_rate, sample_rate):
        errors.append(f"sample_rate {manifest.sample_rate} != {sample_rate}")
    if manifest.channels and manifest.channels != channels:
        errors.append("channel order differs")
    if not np.isclose(manifest.window_seconds, window_seconds):
        errors.append(f"window_seconds {manifest.window_seconds} != {window_seconds}")
    if manifest.feature_profile != feature_profile:
        errors.append(
            f"feature_profile {manifest.feature_profile!r} != {feature_profile!r}"
        )
    if manifest.feature_schema_version != feature_schema_version:
        errors.append(
            "feature_schema_version "
            f"{manifest.feature_schema_version!r} != {feature_schema_version!r}"
        )
    if errors:
        raise ValueError("Incompatible feature model bundle: " + "; ".join(errors))


def _bootstrap_feature_bundle(
    *,
    n_features: int,
    sample_rate: float,
    channels: tuple[str, ...],
    window_seconds: float,
    feature_profile: str,
    feature_schema_version: str,
    version: str = "pm-logreg-bootstrap-v1",
) -> FeatureModelBundle:
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(3, dtype=np.int8), 8)
    centers = labels.astype(float)[:, None] - 1.0
    features = centers + rng.normal(0.0, 0.15, size=(len(labels), n_features))
    scaler = StandardScaler().fit(features)
    standardized = scaler.transform(features)
    targets = np.column_stack(
        [np.roll(labels, shift=index % 3) for index in range(len(PM_METRICS))]
    )
    estimator = PMMultiTaskClassifier(
        "logistic_regression",
        params={"max_iter": 200, "random_state": 42},
    ).fit(standardized, targets)
    manifest = FeatureModelManifest(
        version=version,
        model_type="logistic_regression",
        n_features=n_features,
        sample_rate=sample_rate,
        channels=channels,
        window_seconds=window_seconds,
        feature_profile=feature_profile,
        feature_schema_version=feature_schema_version,
        diagnostic_only=True,
        model_file="",
        scaler_file=None,
        bootstrap=True,
    )
    return FeatureModelBundle(estimator, manifest=manifest, scaler=scaler)


def load_model_bundle(
    artifact_dir: str | Path,
    *,
    sample_rate: float,
    channels: tuple[str, ...],
    window_seconds: float,
    allow_bootstrap: bool,
    preprocessing: dict[str, Any] | None = None,
    device: str = "auto",
    n_features: int | None = None,
    feature_profile: str | None = None,
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> ShallowConvNetBundle | FeatureModelBundle:
    directory = Path(artifact_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        if not allow_bootstrap:
            raise FileNotFoundError(f"Model manifest not found: {manifest_path}")
        if n_features is None or feature_profile is None:
            raise ValueError("Feature bootstrap requires n_features and feature_profile")
        return _bootstrap_feature_bundle(
            n_features=n_features,
            sample_rate=sample_rate,
            channels=channels,
            window_seconds=window_seconds,
            feature_profile=feature_profile,
            feature_schema_version=feature_schema_version,
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_mode = payload.get("input_mode", "features")
    if input_mode == "features":
        if n_features is None or feature_profile is None:
            raise ValueError("Feature bundle requires n_features and feature_profile")
        if payload.get("bootstrap", False):
            if not allow_bootstrap:
                raise RuntimeError("Feature bundle requests bootstrap but it is disabled")
            return _bootstrap_feature_bundle(
                n_features=n_features,
                sample_rate=sample_rate,
                channels=channels,
                window_seconds=window_seconds,
                feature_profile=feature_profile,
                feature_schema_version=feature_schema_version,
                version=payload.get("version", "pm-logreg-bootstrap-v1"),
            )
        feature_manifest = FeatureModelManifest.from_payload(payload)
        _validate_feature_manifest(
            feature_manifest,
            n_features=n_features,
            sample_rate=sample_rate,
            channels=channels,
            window_seconds=window_seconds,
            feature_profile=feature_profile,
            feature_schema_version=feature_schema_version,
        )
        estimator = joblib.load(directory / feature_manifest.model_file)
        scaler = (
            joblib.load(directory / feature_manifest.scaler_file)
            if feature_manifest.scaler_file
            else None
        )
        selector = (
            joblib.load(directory / feature_manifest.selector_file)
            if feature_manifest.selector_file
            else None
        )
        return FeatureModelBundle(
            estimator,
            manifest=feature_manifest,
            scaler=scaler,
            selector=selector,
        )
    if input_mode != RAW_EEG_INPUT_MODE:
        raise ValueError(f"Unsupported model input_mode: {input_mode!r}")

    manifest = RawEEGModelManifest.from_path(manifest_path)
    _validate_raw_manifest(
        manifest,
        sample_rate=sample_rate,
        channels=channels,
        window_seconds=window_seconds,
        preprocessing=preprocessing or {},
    )
    if manifest.bootstrap:
        if not allow_bootstrap:
            raise RuntimeError(
                "Bundle requests diagnostic ShallowConvNet bootstrap but it is disabled"
            )
        if not manifest.diagnostic_only:
            raise ValueError("A bootstrap bundle must set diagnostic_only=true")
        return _bootstrap_raw_bundle(manifest, device=device)

    model_path = directory / manifest.model_file
    if not model_path.exists():
        raise FileNotFoundError(f"ShallowConvNet weights not found: {model_path}")
    estimator = load_torch_weights(model_path, device=device)
    _validate_estimator(estimator, manifest)
    return ShallowConvNetBundle(estimator, manifest=manifest)
