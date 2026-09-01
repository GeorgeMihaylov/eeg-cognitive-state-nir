"""Benchmark dataset backed by a materialized ``cogstate.features`` cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bench.core.abstract_dataset import BaseDataset, EEGData
from bench.features.cogstate_feature_cache import load_feature_cache
from bench.tasks.target_registry import resolve_target_spec
from .target_view import attach_targets_by_sample_id, build_target_view


class CogstateFeatureDataset(BaseDataset):
    """Join a target-free feature cache to one explicit canonical target."""

    def load(self) -> EEGData:
        cache_dir = Path(self.config["data_path"])
        matrix, index, feature_names, manifest = load_feature_cache(cache_dir)
        target_spec = resolve_target_spec(self.config)
        target_path = Path(self.config["target_data_path"])
        if not target_path.is_file():
            raise FileNotFoundError(f"Canonical target table not found: {target_path}")
        target_columns = [
            "subject_id",
            "record_id",
            *target_spec.processed_columns,
        ]
        target_frame = pd.read_parquet(target_path, columns=target_columns)
        if "sample_id" not in target_frame.columns:
            target_frame = target_frame.copy()
            target_frame.insert(0, "sample_id", target_frame.index.to_numpy())
        joined = attach_targets_by_sample_id(
            index,
            target_frame,
            target_spec,
            validate_identifiers=True,
        )
        target_view = build_target_view(joined, target_spec)
        positions = target_view.cohort.selected_positions
        selected = joined.iloc[positions].copy()
        values = np.ascontiguousarray(matrix[positions], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(feature_names):
            raise RuntimeError("Feature cache produced an invalid benchmark matrix")
        if not np.isfinite(values).all():
            raise RuntimeError("Selected feature cohort contains NaN or Inf")
        row_metadata = {
            column: selected[column].to_numpy()
            for column in (
                "source",
                "record_group_id",
                "t_start",
                "t_end",
                "outer_fold",
                "preprocessing_hash",
                "preprocessing_variant",
            )
            if column in selected.columns
        }
        identity = manifest["identity"]
        self._data = EEGData(
            data=values,
            labels=target_view.targets,
            subject_ids=selected["subject_id"].astype(str).to_numpy(),
            feature_names=feature_names,
            sampling_rate=float(self.config.get("sampling_rate", 256.0)),
            sample_ids=selected["sample_id"].to_numpy(dtype=np.int64),
            record_ids=selected["record_id"].astype(str).to_numpy(),
            row_metadata=row_metadata,
            metadata={
                "observation_unit": "engineered_feature_window",
                "feature_cache_path": str(cache_dir),
                "feature_schema": identity["feature_schema"],
                "feature_hash": identity["feature_hash"],
                "feature_cache_identity_hash": identity["cache_identity_hash"],
                "raw_preprocessing_hash": identity["raw_preprocessing_hash"],
                "target_id": target_spec.target_id,
                "target_type": target_spec.target_type,
                "target_registry_status": target_spec.registry_status,
                "n_before_target_filter": int(len(index)),
                "n_after_target_filter": int(len(selected)),
            },
        )
        return self._data

    def get_description(self) -> dict[str, Any]:
        data = self.data
        return {
            "name": "Canonical cogstate feature cache",
            "n_samples": data.n_samples,
            "n_features": data.n_features,
            "n_subjects": data.n_subjects,
            "target_id": data.metadata["target_id"],
            "feature_schema": data.metadata["feature_schema"],
            "feature_hash": data.metadata["feature_hash"],
        }
