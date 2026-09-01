"""Nested AutoML orchestration over the canonical benchmark pipeline."""

from .objective import AutoMLTrialResult, NestedBenchmarkObjective
from .search_space import (
    AutoMLStudySpec,
    SearchParameterSpec,
    SearchSpaceSpec,
)
__all__ = [
    "AutoMLStudyRunner",
    "AutoMLStudySpec",
    "AutoMLTrialResult",
    "NestedBenchmarkObjective",
    "SearchParameterSpec",
    "SearchSpaceSpec",
]


def __getattr__(name: str):
    if name == "AutoMLStudyRunner":
        from .study_runner import AutoMLStudyRunner

        return AutoMLStudyRunner
    raise AttributeError(name)
