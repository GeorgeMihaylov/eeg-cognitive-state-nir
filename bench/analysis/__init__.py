"""Read-only statistical analysis for completed benchmark artifacts."""

from .alignment import AlignmentResult, check_alignment, require_alignment
from .error_analysis import calculate_error_analysis, summarize_by_source
from .paired_statistics import (
    holm_adjust,
    paired_subject_statistics,
    subject_bootstrap_interval,
)
from .run_inventory import (
    InventoryEntry,
    build_run_inventory,
    select_canonical_runs,
)
from .subject_metrics import calculate_subject_metrics

__all__ = [
    "AlignmentResult",
    "InventoryEntry",
    "build_run_inventory",
    "calculate_error_analysis",
    "calculate_subject_metrics",
    "check_alignment",
    "holm_adjust",
    "paired_subject_statistics",
    "require_alignment",
    "select_canonical_runs",
    "subject_bootstrap_interval",
    "summarize_by_source",
]
