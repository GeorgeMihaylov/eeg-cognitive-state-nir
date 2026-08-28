"""Leakage-safe, serializable preprocessing for feature-based Torch models."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from sklearn.preprocessing import RobustScaler


SUPPORTED_FEATURE_SCALING_STRATEGIES = {
    "none",
    "standard",
    "robust",
    "standard_clip",
    "robust_clip",
    "pow_log_standard",
    "pow_log_robust",
}


def _finite_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim not in {2, 3}:
        raise ValueError(
            f"{name} must have shape [samples, features] or "
            f"[samples, timesteps, features], got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values")
    numeric = np.asarray(array, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return numeric


class FeaturePreprocessor:
    """Fit feature transforms on an inner-training partition only."""

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ) -> None:
        resolved = dict(config or {})
        self.strategy = str(resolved.get("strategy", "standard")).strip().lower()
        if self.strategy not in SUPPORTED_FEATURE_SCALING_STRATEGIES:
            raise ValueError(
                f"Unknown feature scaling strategy {self.strategy!r}. "
                f"Available: {sorted(SUPPORTED_FEATURE_SCALING_STRATEGIES)}"
            )
        quantile_range = resolved.get("quantile_range", [25.0, 75.0])
        clip_percentiles = resolved.get("clip_percentiles", [0.5, 99.5])
        if len(quantile_range) != 2:
            raise ValueError("quantile_range must contain two values")
        if len(clip_percentiles) != 2:
            raise ValueError("clip_percentiles must contain two values")
        self.quantile_range = tuple(float(value) for value in quantile_range)
        self.clip_percentiles = tuple(
            float(value) for value in clip_percentiles
        )
        if not 0 <= self.quantile_range[0] < self.quantile_range[1] <= 100:
            raise ValueError(
                "quantile_range must satisfy 0 <= low < high <= 100"
            )
        if not 0 <= self.clip_percentiles[0] < self.clip_percentiles[1] <= 100:
            raise ValueError(
                "clip_percentiles must satisfy 0 <= low < high <= 100"
            )
        self.scale_floor = float(resolved.get("scale_floor", 1e-8))
        if not np.isfinite(self.scale_floor) or self.scale_floor <= 0:
            raise ValueError("scale_floor must be finite and positive")
        self.feature_names = (
            None if feature_names is None else tuple(str(name) for name in feature_names)
        )
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.clip_lower_: Optional[np.ndarray] = None
        self.clip_upper_: Optional[np.ndarray] = None
        self.pow_mask_: Optional[np.ndarray] = None
        self.pow_log_rule_: Optional[str] = None
        self.n_features_in_: Optional[int] = None
        self.n_fit_samples_: Optional[int] = None
        self.fitted_: bool = False

    @property
    def uses_clipping(self) -> bool:
        return self.strategy in {"standard_clip", "robust_clip"}

    @property
    def uses_robust_scaling(self) -> bool:
        return self.strategy in {
            "robust",
            "robust_clip",
            "pow_log_robust",
        }

    @property
    def uses_pow_log(self) -> bool:
        return self.strategy in {"pow_log_standard", "pow_log_robust"}

    def _resolve_feature_names(self, n_features: int) -> tuple[str, ...]:
        if self.feature_names is None:
            names = tuple(f"feature_{index:03d}" for index in range(n_features))
        else:
            names = self.feature_names
        if len(names) != n_features:
            raise ValueError(
                f"Expected {n_features} feature names, got {len(names)}"
            )
        self.feature_names = names
        return names

    @staticmethod
    def _flatten(array: np.ndarray) -> np.ndarray:
        return (
            array.reshape(-1, array.shape[-1])
            if array.ndim == 3
            else array
        )

    def _apply_pow_log(self, values: np.ndarray) -> np.ndarray:
        if not self.uses_pow_log:
            return values
        if self.pow_mask_ is None or self.pow_log_rule_ is None:
            raise RuntimeError("POW transform has not been fitted")
        transformed = values.copy()
        selected = transformed[..., self.pow_mask_]
        if self.pow_log_rule_ == "log1p":
            if np.any(selected < 0):
                raise ValueError(
                    "POW features were non-negative in inner train but contain "
                    "negative values in the transformed partition"
                )
            transformed[..., self.pow_mask_] = np.log1p(selected)
        else:
            transformed[..., self.pow_mask_] = (
                np.sign(selected) * np.log1p(np.abs(selected))
            )
        return transformed

    def fit(self, X: Any) -> "FeaturePreprocessor":
        values = _finite_array(X, name="Inner-train features")
        flat = self._flatten(values)
        names = self._resolve_feature_names(flat.shape[1])
        self.n_features_in_ = int(flat.shape[1])
        self.n_fit_samples_ = int(flat.shape[0])
        self.pow_mask_ = np.asarray(
            [name.upper().startswith("POW.") for name in names],
            dtype=bool,
        )
        if self.uses_pow_log and not np.any(self.pow_mask_):
            raise ValueError(
                "POW log scaling requires feature names beginning with 'POW.'"
            )
        if self.uses_pow_log:
            self.pow_log_rule_ = (
                "log1p"
                if np.all(flat[:, self.pow_mask_] >= 0)
                else "signed_log1p"
            )
        else:
            self.pow_log_rule_ = None

        transformed = self._apply_pow_log(flat)
        if self.uses_clipping:
            percentiles = np.percentile(
                transformed,
                self.clip_percentiles,
                axis=0,
            )
            self.clip_lower_ = np.asarray(percentiles[0], dtype=np.float64)
            self.clip_upper_ = np.asarray(percentiles[1], dtype=np.float64)
            transformed = np.clip(
                transformed,
                self.clip_lower_,
                self.clip_upper_,
            )
        else:
            self.clip_lower_ = None
            self.clip_upper_ = None

        if self.strategy == "none":
            center = np.zeros(flat.shape[1], dtype=np.float64)
            scale = np.ones(flat.shape[1], dtype=np.float64)
        elif self.uses_robust_scaling:
            scaler = RobustScaler(
                with_centering=True,
                with_scaling=True,
                quantile_range=self.quantile_range,
            )
            scaler.fit(transformed)
            center = np.asarray(scaler.center_, dtype=np.float64)
            scale = np.asarray(scaler.scale_, dtype=np.float64)
        else:
            center = transformed.mean(axis=0, dtype=np.float64)
            scale = transformed.std(axis=0, dtype=np.float64)
        scale = np.where(np.abs(scale) < self.scale_floor, 1.0, scale)
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ValueError("Fitted feature preprocessing statistics are not finite")
        self.center_ = np.asarray(center, dtype=np.float32)
        self.scale_ = np.asarray(scale, dtype=np.float32)
        self.fitted_ = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not self.fitted_ or self.center_ is None or self.scale_ is None:
            raise RuntimeError("FeaturePreprocessor must be fitted before transform")
        values = _finite_array(X, name="Features")
        if values.shape[-1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {values.shape[-1]}"
            )
        transformed = self._apply_pow_log(values)
        if self.uses_clipping:
            transformed = np.clip(
                transformed,
                self.clip_lower_,
                self.clip_upper_,
            )
        transformed = (transformed - self.center_) / self.scale_
        if not np.isfinite(transformed).all():
            raise ValueError(
                "Feature preprocessing produced NaN or infinite values"
            )
        return np.ascontiguousarray(transformed, dtype=np.float32)

    def diagnostics(self, X: Any) -> Dict[str, Any]:
        transformed = self.transform(X)
        absolute = np.abs(np.asarray(transformed, dtype=np.float64))
        return {
            "samples": int(transformed.shape[0]),
            "values": int(transformed.size),
            "max_abs": float(np.max(absolute)),
            "p95_abs": float(np.percentile(absolute, 95)),
            "p99_abs": float(np.percentile(absolute, 99)),
            "values_abs_gt_5": int(np.sum(absolute > 5)),
            "values_abs_gt_10": int(np.sum(absolute > 10)),
            "values_abs_gt_100": int(np.sum(absolute > 100)),
            "values_abs_gt_1000": int(np.sum(absolute > 1000)),
            "nonfinite_values": int(np.sum(~np.isfinite(transformed))),
        }

    @property
    def feature_hash(self) -> str:
        names = self.feature_names or ()
        payload = "".join(f"{name}\n" for name in names).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_state(self) -> Dict[str, Any]:
        if not self.fitted_ or self.center_ is None or self.scale_ is None:
            raise RuntimeError("Cannot serialize an unfitted FeaturePreprocessor")
        return {
            "schema_version": 1,
            "strategy": self.strategy,
            "scope": "inner_train_only",
            "train_only": True,
            "quantile_range": list(self.quantile_range),
            "clip_percentiles": list(self.clip_percentiles),
            "scale_floor": self.scale_floor,
            "feature_names": list(self.feature_names or ()),
            "feature_hash": self.feature_hash,
            "n_features": self.n_features_in_,
            "n_fit_samples": self.n_fit_samples_,
            "center": self.center_.tolist(),
            "scale": self.scale_.tolist(),
            "clipping_enabled": self.uses_clipping,
            "clip_lower": (
                None if self.clip_lower_ is None else self.clip_lower_.tolist()
            ),
            "clip_upper": (
                None if self.clip_upper_ is None else self.clip_upper_.tolist()
            ),
            "pow_log_enabled": self.uses_pow_log,
            "pow_log_rule": self.pow_log_rule_,
            "pow_feature_indices": (
                []
                if self.pow_mask_ is None
                else np.flatnonzero(self.pow_mask_).astype(int).tolist()
            ),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "FeaturePreprocessor":
        preprocessor = cls(
            {
                "strategy": state["strategy"],
                "quantile_range": state.get("quantile_range", [25.0, 75.0]),
                "clip_percentiles": state.get(
                    "clip_percentiles", [0.5, 99.5]
                ),
                "scale_floor": state.get("scale_floor", 1e-8),
            },
            feature_names=state.get("feature_names"),
        )
        preprocessor.center_ = np.asarray(state["center"], dtype=np.float32)
        preprocessor.scale_ = np.asarray(state["scale"], dtype=np.float32)
        lower = state.get("clip_lower")
        upper = state.get("clip_upper")
        preprocessor.clip_lower_ = (
            None if lower is None else np.asarray(lower, dtype=np.float64)
        )
        preprocessor.clip_upper_ = (
            None if upper is None else np.asarray(upper, dtype=np.float64)
        )
        preprocessor.n_features_in_ = int(
            state.get("n_features", len(preprocessor.center_))
        )
        preprocessor.n_fit_samples_ = int(state.get("n_fit_samples", 0))
        pow_indices = np.asarray(
            state.get("pow_feature_indices", []), dtype=np.int64
        )
        preprocessor.pow_mask_ = np.zeros(
            preprocessor.n_features_in_, dtype=bool
        )
        preprocessor.pow_mask_[pow_indices] = True
        preprocessor.pow_log_rule_ = state.get("pow_log_rule")
        preprocessor.fitted_ = True
        return preprocessor
