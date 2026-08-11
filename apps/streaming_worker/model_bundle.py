"""Load a complete inference bundle or build a diagnostic model-zoo bundle."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from cogstate.model_zoo.multitask import PMMultiTaskClassifier
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
    diagnostic_only: bool = False
    model_file: str = "model.joblib"
    scaler_file: str | None = "scaler.joblib"
    selector_file: str | None = None

    @classmethod
    def from_path(cls, path: Path) -> "ModelManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["channels"] = tuple(payload.get("channels", ()))
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
            classes = np.asarray(model.classes_, dtype=int)
            result[metric] = {
                CLASS_NAMES[int(label)]: float(probability)
                for label, probability in zip(classes, probabilities[0])
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
    if errors:
        raise ValueError("Incompatible model bundle: " + "; ".join(errors))


def _bootstrap_bundle(
    *,
    n_features: int,
    sample_rate: float,
    channels: tuple[str, ...],
    window_seconds: float,
    feature_profile: str,
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
        return BundlePMModel(
            estimator, manifest=manifest, scaler=scaler, selector=selector
        )

    if not allow_bootstrap:
        raise FileNotFoundError(f"Model manifest not found: {manifest_path}")
    return _bootstrap_bundle(
        n_features=n_features,
        sample_rate=sample_rate,
        channels=channels,
        window_seconds=window_seconds,
        feature_profile=feature_profile,
    )
