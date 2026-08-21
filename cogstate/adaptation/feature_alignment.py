"""Leakage-safe participant feature-space alignment.

The aligner is intentionally target-free. Reference statistics are estimated
from the authorized outer-train feature matrix, while participant statistics
are estimated only from the chronological calibration prefix.

The already trained downstream estimator remains in the reference feature
coordinate system. Participant features are therefore mapped back into that
coordinate system instead of being independently standardized.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


SUPPORTED_ALIGNMENT_METHODS = frozenset(
    {
        "standard_location_scale",
        "robust_location_scale",
    }
)


@dataclass(frozen=True)
class FeatureAlignmentConfig:
    """Configuration for participant feature-space alignment."""

    method: str = "standard_location_scale"
    scale_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_ALIGNMENT_METHODS:
            raise ValueError(
                "method must be one of "
                f"{sorted(SUPPORTED_ALIGNMENT_METHODS)}, "
                f"got {self.method!r}"
            )
        if (
            not np.isfinite(self.scale_epsilon)
            or float(self.scale_epsilon) <= 0.0
        ):
            raise ValueError(
                "scale_epsilon must be finite and positive"
            )


@dataclass(frozen=True)
class FeatureAlignmentStats:
    """Location/scale statistics for one feature distribution."""

    center: np.ndarray
    scale: np.ndarray
    n_samples: int
    n_features: int

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64).copy()
        scale = np.asarray(self.scale, dtype=np.float64).copy()

        if center.ndim != 1 or scale.ndim != 1:
            raise ValueError(
                "center and scale must be one-dimensional"
            )
        if center.shape != scale.shape:
            raise ValueError(
                "center and scale must have identical shapes"
            )
        if len(center) != int(self.n_features):
            raise ValueError(
                "statistics width does not match n_features"
            )
        if int(self.n_samples) < 1:
            raise ValueError("n_samples must be positive")
        if not np.isfinite(center).all():
            raise ValueError("center contains NaN or Inf")
        if not np.isfinite(scale).all():
            raise ValueError("scale contains NaN or Inf")
        if np.any(scale < 0.0):
            raise ValueError("scale must be non-negative")

        center.setflags(write=False)
        scale.setflags(write=False)

        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)

    def digest(self) -> str:
        """Stable SHA-256 digest for audit/provenance."""

        hasher = hashlib.sha256()
        hasher.update(
            np.asarray(self.center, dtype="<f8").tobytes()
        )
        hasher.update(
            np.asarray(self.scale, dtype="<f8").tobytes()
        )
        hasher.update(str(int(self.n_samples)).encode("ascii"))
        hasher.update(str(int(self.n_features)).encode("ascii"))
        return hasher.hexdigest()


def _validate_matrix(
    X: np.ndarray,
    *,
    name: str,
    expected_features: int | None = None,
) -> np.ndarray:
    features = np.asarray(X, dtype=np.float64)

    if features.ndim != 2:
        raise ValueError(
            f"{name} must have shape [n_samples, n_features], "
            f"got {features.shape}"
        )
    if features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError(f"{name} cannot be empty")
    if expected_features is not None and (
        features.shape[1] != int(expected_features)
    ):
        raise ValueError(
            f"{name} has {features.shape[1]} features; "
            f"expected {expected_features}"
        )
    if not np.isfinite(features).all():
        raise ValueError(f"{name} contains NaN or Inf")

    return np.ascontiguousarray(
        features,
        dtype=np.float64,
    )


def apply_alignment_shrinkage(
    X_original: np.ndarray,
    X_aligned: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Interpolate deterministically between identity and full alignment.

    ``alpha=0`` is the legitimate no-adaptation endpoint and ``alpha=1`` is
    the complete fitted alignment.  This function performs no fitting and has
    no access to labels or participant metadata.
    """

    original = _validate_matrix(X_original, name="X_original")
    aligned = _validate_matrix(
        X_aligned,
        name="X_aligned",
        expected_features=original.shape[1],
    )
    if original.shape != aligned.shape:
        raise ValueError(
            "X_original and X_aligned must have identical shapes"
        )
    if not np.isfinite(alpha) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    transformed = original + float(alpha) * (aligned - original)
    if not np.isfinite(transformed).all():
        raise RuntimeError("alignment shrinkage produced NaN or Inf")
    return np.ascontiguousarray(transformed, dtype=np.float64)


def _estimate_stats(
    X: np.ndarray,
    config: FeatureAlignmentConfig,
) -> FeatureAlignmentStats:
    if config.method == "standard_location_scale":
        center = np.mean(X, axis=0)
        scale = np.std(X, axis=0, ddof=0)

    elif config.method == "robust_location_scale":
        center = np.median(X, axis=0)
        q25 = np.quantile(X, 0.25, axis=0)
        q75 = np.quantile(X, 0.75, axis=0)
        scale = q75 - q25

    else:  # protected by config validation
        raise RuntimeError(
            f"unsupported alignment method {config.method!r}"
        )

    return FeatureAlignmentStats(
        center=center,
        scale=scale,
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
    )


class FeatureAligner:
    """Map participant features into the outer-train coordinate system.

    Scientific contract
    -------------------
    1. ``fit_reference`` receives outer-train features only.
    2. ``fit_calibration`` receives the new participant's calibration
       prefix only.
    3. ``transform`` may then be applied to calibration or evaluation
       rows, but evaluation rows never influence fitted statistics.
    """

    def __init__(
        self,
        config: FeatureAlignmentConfig | None = None,
    ):
        self.config = (
            FeatureAlignmentConfig()
            if config is None
            else config
        )
        self.reference_stats_: FeatureAlignmentStats | None = None
        self.calibration_stats_: FeatureAlignmentStats | None = None

    @property
    def is_reference_fitted(self) -> bool:
        return self.reference_stats_ is not None

    @property
    def is_calibration_fitted(self) -> bool:
        return self.calibration_stats_ is not None

    def fit_reference(
        self,
        X_outer_train: np.ndarray,
    ) -> "FeatureAligner":
        """Estimate the global reference coordinate system."""

        X = _validate_matrix(
            X_outer_train,
            name="X_outer_train",
        )

        self.reference_stats_ = _estimate_stats(
            X,
            self.config,
        )

        # A new reference invalidates any participant calibration
        # statistics fitted against an older coordinate system.
        self.calibration_stats_ = None

        return self

    def fit_calibration(
        self,
        X_calibration: np.ndarray,
    ) -> "FeatureAligner":
        """Estimate participant statistics from calibration rows only."""

        if self.reference_stats_ is None:
            raise RuntimeError(
                "fit_reference must be called before fit_calibration"
            )

        X = _validate_matrix(
            X_calibration,
            name="X_calibration",
            expected_features=self.reference_stats_.n_features,
        )

        self.calibration_stats_ = _estimate_stats(
            X,
            self.config,
        )
        return self

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Map features into the outer-train reference space."""

        if self.reference_stats_ is None:
            raise RuntimeError(
                "fit_reference must be called before transform"
            )
        if self.calibration_stats_ is None:
            raise RuntimeError(
                "fit_calibration must be called before transform"
            )

        reference = self.reference_stats_
        calibration = self.calibration_stats_

        features = _validate_matrix(
            X,
            name="X",
            expected_features=reference.n_features,
        )

        eps = float(self.config.scale_epsilon)

        # If participant variance/IQR is effectively zero, the
        # standardized coordinate is undefined. Mapping that feature
        # to the reference centre is deterministic and avoids
        # arbitrarily large values.
        valid_calibration_scale = calibration.scale > eps

        standardized = np.zeros_like(
            features,
            dtype=np.float64,
        )

        standardized[:, valid_calibration_scale] = (
            features[:, valid_calibration_scale]
            - calibration.center[valid_calibration_scale]
        ) / calibration.scale[valid_calibration_scale]

        aligned = (
            reference.center[None, :]
            + standardized * reference.scale[None, :]
        )

        if not np.isfinite(aligned).all():
            raise RuntimeError(
                "feature alignment produced NaN or Inf"
            )

        return np.ascontiguousarray(
            aligned,
            dtype=np.float64,
        )

    def fit_transform_calibration(
        self,
        X_calibration: np.ndarray,
    ) -> np.ndarray:
        """Fit participant statistics and align calibration rows."""

        self.fit_calibration(X_calibration)
        return self.transform(X_calibration)

    def to_manifest(self) -> dict[str, object]:
        """Return audit metadata without exposing evaluation data."""

        reference = self.reference_stats_
        calibration = self.calibration_stats_

        return {
            "method": self.config.method,
            "scale_epsilon": float(
                self.config.scale_epsilon
            ),
            "reference_fitted": reference is not None,
            "calibration_fitted": calibration is not None,
            "n_features": (
                None
                if reference is None
                else reference.n_features
            ),
            "reference_n_samples": (
                None
                if reference is None
                else reference.n_samples
            ),
            "calibration_n_samples": (
                None
                if calibration is None
                else calibration.n_samples
            ),
            "reference_stats_hash": (
                None
                if reference is None
                else reference.digest()
            ),
            "calibration_stats_hash": (
                None
                if calibration is None
                else calibration.digest()
            ),
            "reference_degenerate_features": (
                None
                if reference is None
                else int(
                    np.sum(
                        reference.scale
                        <= self.config.scale_epsilon
                    )
                )
            ),
            "calibration_degenerate_features": (
                None
                if calibration is None
                else int(
                    np.sum(
                        calibration.scale
                        <= self.config.scale_epsilon
                    )
                )
            ),
        }
