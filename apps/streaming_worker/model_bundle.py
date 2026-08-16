"""Load a complete inference bundle or build a diagnostic model-zoo bundle."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from model_zoo.ML.multitask import PMMultiTaskClassifier
from cogstate.protocol import PM_METRICS


CLASS_NAMES = ("low", "medium", "high")


@dataclass(frozen=True)
class ModelManifest:
    version: str
    model_type: str
    n_features: int
    sample_rate: float
    channels: tuple[str, ...]
    window_seconds: float
    feature_profile: str
    feature_schema_hash: str = ""
    preprocessing_hash: str = ""
    signal_preprocessing_hash: str = ""
    diagnostic_only: bool = False
    model_file: str = "model.joblib"
    scaler_file: str | None = "scaler.joblib"
    selector_file: str | None = None
    imputer_file: str | None = None
    target_metrics: tuple[str, ...] = PM_METRICS
    q3_thresholds: dict[str, list[float]] | None = None
    q3_thresholds_hash: str | None = None
    target_contract: str | None = None
    training_fold: dict[str, Any] | None = None
    training_participant_ids_hash: str | None = None
    model_parameters: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    protocol_hash: str | None = None
    training_seconds: float | None = None

    @classmethod
    def from_path(cls, path: Path) -> "ModelManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["channels"] = tuple(payload.get("channels", ()))
        payload["target_metrics"] = tuple(payload.get("target_metrics", PM_METRICS))
        return cls(**payload)


class BundlePMModel:
    """Apply bundle transforms and expose named probabilities to inference."""

    def __init__(
        self,
        estimator: PMMultiTaskClassifier,
        *,
        manifest: ModelManifest,
        scaler: Any | None = None,
        selector: Any | None = None,
        imputer: Any | None = None,
    ) -> None:
        self.estimator = estimator
        self.manifest = manifest
        self.scaler = scaler
        self.selector = selector
        self.imputer = imputer
        self.version = manifest.version

    def predict_pm_proba(self, features: np.ndarray) -> dict[str, dict[str, float]]:
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.manifest.n_features:
            raise ValueError(
                f"Expected {self.manifest.n_features} features, got {values.shape}"
            )
        if self.imputer is not None:
            values = self.imputer.transform(values)
        if not np.isfinite(values).all():
            raise ValueError("Model features contain non-finite values after imputation")
        if self.scaler is not None:
            values = self.scaler.transform(values)
        if self.selector is not None:
            values = self.selector.transform(values)

        raw = self.estimator.predict_proba(values)
        result: dict[str, dict[str, float]] = {}
        if tuple(raw) != tuple(self.manifest.target_metrics):
            raise ValueError("Model does not predict exactly the manifest PM targets")
        for metric, probabilities in raw.items():
            model = self.estimator.models_[metric]
            classes = np.asarray(model.classes_, dtype=int)
            values_by_class = {name: 0.0 for name in CLASS_NAMES}
            for label, probability in zip(classes, probabilities[0]):
                values_by_class[CLASS_NAMES[int(label)]] = float(probability)
            total = sum(values_by_class.values())
            if not np.isfinite(list(values_by_class.values())).all() or total <= 0:
                raise ValueError(f"Invalid probabilities for PM target {metric!r}")
            result[metric] = {
                name: probability / total
                for name, probability in values_by_class.items()
            }
        return result


def _validate_manifest(
    manifest: ModelManifest,
    *,
    n_features: int,
    sample_rate: float,
    channels: tuple[str, ...],
    window_seconds: float,
    feature_profile: str,
    feature_schema_hash_value: str,
    preprocessing_hash_value: str,
) -> None:
    errors: list[str] = []
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
    if manifest.feature_schema_hash != feature_schema_hash_value:
        errors.append("feature schema hash differs")
    if manifest.preprocessing_hash != preprocessing_hash_value:
        errors.append("preprocessing hash differs")
    if (
        manifest.signal_preprocessing_hash
        and manifest.signal_preprocessing_hash != preprocessing_hash_value
    ):
        errors.append("signal preprocessing hash differs")
    if tuple(manifest.target_metrics) != tuple(PM_METRICS):
        errors.append("target metric order differs")
    if errors:
        raise ValueError("Incompatible model bundle: " + "; ".join(errors))


def _bootstrap_bundle(
    *,
    n_features: int,
    sample_rate: float,
    channels: tuple[str, ...],
    window_seconds: float,
    feature_profile: str,
    feature_schema_hash_value: str = "",
    preprocessing_hash_value: str = "",
    version: str = "pm-logreg-bootstrap-v1",
) -> BundlePMModel:
    """Fit a real model-zoo estimator on synthetic anchors for smoke tests."""
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
    manifest = ModelManifest(
        version=version,
        model_type="logistic_regression",
        n_features=n_features,
        sample_rate=sample_rate,
        channels=channels,
        window_seconds=window_seconds,
        feature_profile=feature_profile,
        feature_schema_hash=feature_schema_hash_value,
        preprocessing_hash=preprocessing_hash_value,
        diagnostic_only=True,
        model_file="",
        scaler_file=None,
    )
    return BundlePMModel(estimator, manifest=manifest, scaler=scaler)


def load_model_bundle(
    artifact_dir: str | Path,
    *,
    n_features: int,
    sample_rate: float,
    channels: tuple[str, ...],
    window_seconds: float,
    feature_profile: str,
    feature_schema_hash_value: str,
    preprocessing_hash_value: str,
    allow_bootstrap: bool,
) -> BundlePMModel:
    directory = Path(artifact_dir)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("bootstrap", False):
            if not allow_bootstrap:
                raise RuntimeError("Bundle requests diagnostic bootstrap but it is disabled")
            return _bootstrap_bundle(
                n_features=n_features,
                sample_rate=sample_rate,
                channels=channels,
                window_seconds=window_seconds,
                feature_profile=feature_profile,
                feature_schema_hash_value=feature_schema_hash_value,
                preprocessing_hash_value=preprocessing_hash_value,
                version=payload.get("version", "pm-logreg-bootstrap-v1"),
            )

        manifest = ModelManifest.from_path(manifest_path)
        _validate_manifest(
            manifest,
            n_features=n_features,
            sample_rate=sample_rate,
            channels=channels,
            window_seconds=window_seconds,
            feature_profile=feature_profile,
            feature_schema_hash_value=feature_schema_hash_value,
            preprocessing_hash_value=preprocessing_hash_value,
        )
        estimator = joblib.load(directory / manifest.model_file)
        scaler = (
            joblib.load(directory / manifest.scaler_file)
            if manifest.scaler_file
            else None
        )
        selector = (
            joblib.load(directory / manifest.selector_file)
            if manifest.selector_file
            else None
        )
        imputer = (
            joblib.load(directory / manifest.imputer_file)
            if manifest.imputer_file
            else None
        )
        return BundlePMModel(
            estimator,
            manifest=manifest,
            scaler=scaler,
            selector=selector,
            imputer=imputer,
        )

    if not allow_bootstrap:
        raise FileNotFoundError(f"Model manifest not found: {manifest_path}")
    return _bootstrap_bundle(
        n_features=n_features,
        sample_rate=sample_rate,
        channels=channels,
        window_seconds=window_seconds,
        feature_profile=feature_profile,
        feature_schema_hash_value=feature_schema_hash_value,
        preprocessing_hash_value=preprocessing_hash_value,
    )
