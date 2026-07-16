"""Small study manifests that reference canonical benchmark artifacts."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from .objective import AutoMLTrialResult
from .search_space import AutoMLStudySpec


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )


def environment_manifest() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "optuna"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    except ImportError:
        cuda = {"available": False, "version": None, "device_count": 0}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
    }


def trial_frame(results: Sequence[AutoMLTrialResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        row = {
            "study_name": result.study_name,
            "trial_number": result.trial_number,
            "state": result.state,
            "outer_fold": result.outer_fold,
            "inner_split": result.inner_split,
            "objective_value": result.objective_value,
            "runtime_seconds": result.runtime_seconds,
            "resolved_config_hash": result.resolved_config_hash,
            "reused": result.reused,
            "failure_reason": result.failure_reason,
            "benchmark_run_reference": json.dumps(
                result.benchmark_run_reference, sort_keys=True, default=str
            ),
            "resolved_config": json.dumps(
                result.resolved_config, sort_keys=True, default=str
            ),
        }
        row.update({
            f"param::{path}": value
            for path, value in result.trial_parameters.items()
        })
        row.update({
            f"metric::{name}": value
            for name, value in result.secondary_metrics.items()
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "trial_number", kind="mergesort"
    ) if rows else pd.DataFrame()


def initialize_study_artifacts(
    study_dir: Path,
    spec: AutoMLStudySpec,
    outer_folds: Mapping[str, Any],
) -> None:
    study_dir.mkdir(parents=True, exist_ok=True)
    with open(study_dir / "study_spec.yaml", "w", encoding="utf-8") as output:
        yaml.safe_dump(spec.to_dict(), output, sort_keys=False)
    with open(study_dir / "search_space.yaml", "w", encoding="utf-8") as output:
        yaml.safe_dump(spec.search_space.to_dict(), output, sort_keys=False)
    _write_json(study_dir / "outer_folds.json", outer_folds)
    _write_json(study_dir / "environment.json", environment_manifest())


def update_study_artifacts(
    study_dir: Path,
    *,
    results: Sequence[AutoMLTrialResult],
    summary: Mapping[str, Any],
    best_trials: Mapping[str, Any],
) -> None:
    trials = trial_frame(results)
    trials.to_parquet(study_dir / "trials.parquet", index=False)
    _write_json(study_dir / "study_summary.json", dict(summary))
    _write_json(study_dir / "best_trials.json", dict(best_trials))
