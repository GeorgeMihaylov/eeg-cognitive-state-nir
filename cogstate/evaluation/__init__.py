"""Application-only latency and external-fold validation helpers.

Scientific metrics, GroupKFold and participant aggregation live exclusively in
``bench.validation``.
"""

from .metrics import latency_metrics
from .folds import ExternalFold, validate_external_folds

__all__ = ["latency_metrics", "ExternalFold", "validate_external_folds"]
