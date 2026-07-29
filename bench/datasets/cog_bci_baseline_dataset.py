"""Lazy supervised COG-BCI windows backed by the verified record cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from bench.core.abstract_dataset import BaseDataset, EEGData
from bench.datasets.raw_eeg_window_dataset import RawEEGWindowArrayView


EXPECTED_TASK_ID = "cog_bci_nback_3class"
EXPECTED_TARGET = "n_back_level"


def _shard_stem(record_id: str) -> str:
    return hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:24]


class PerWindowCenteredRawEEGWindowArrayView(RawEEGWindowArrayView):
    """Lazy view that removes each channel's temporal mean per window."""

    def _read_scalar(self, index: int) -> np.ndarray:
        window = super()._read_scalar(index)
        centered = window - window.mean(axis=-1, keepdims=True)
        return np.ascontiguousarray(centered, dtype=np.float32)

    def __getitem__(self, index: Any) -> Any:
        if np.isscalar(index):
            return super().__getitem__(index)
        indices = np.arange(len(self))[index]
        return PerWindowCenteredRawEEGWindowArrayView(
            self.manifest.iloc[np.asarray(indices, dtype=np.int64)],
            channel_mean=self.channel_mean,
            channel_scale=self.channel_scale,
        )

    def with_channel_normalization(
        self, mean: np.ndarray, scale: np.ndarray
    ) -> "PerWindowCenteredRawEEGWindowArrayView":
        return PerWindowCenteredRawEEGWindowArrayView(
            self.manifest, channel_mean=mean, channel_scale=scale
        )


class COGBCINBackWindowDataset(BaseDataset):
    """Load accepted N-Back windows and immutable split assignments."""

    def load(self) -> EEGData:
        cache_dir = Path(self.config["data_path"])
        protocol_dir = Path(self.config["task_protocol_path"])
        cache_manifest_path = cache_dir / "dataset_manifest.json"
        window_index_path = cache_dir / "window_index.parquet"
        protocol_summary_path = protocol_dir / "protocol_summary.json"
        target_index_path = protocol_dir / "target_index.parquet"
        outer_assignments_path = protocol_dir / "outer_assignments.parquet"
        inner_assignments_path = protocol_dir / "inner_assignments.parquet"
        required = [
            cache_manifest_path,
            window_index_path,
            protocol_summary_path,
            target_index_path,
            outer_assignments_path,
            inner_assignments_path,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"COG-BCI baseline inputs are incomplete: {missing}"
            )
        cache_manifest = json.loads(
            cache_manifest_path.read_text(encoding="utf-8")
        )
        protocol_summary = json.loads(
            protocol_summary_path.read_text(encoding="utf-8")
        )
        if protocol_summary.get("task_id") != EXPECTED_TASK_ID:
            raise ValueError(
                "COG-BCI N-Back dataset requires task_id "
                f"{EXPECTED_TASK_ID!r}"
            )
        if protocol_summary.get("target_name") != EXPECTED_TARGET:
            raise ValueError(
                f"COG-BCI N-Back target must be {EXPECTED_TARGET!r}"
            )
        expected_cache_hash = self.config.get("window_cache_config_hash")
        cache_hash = str(cache_manifest.get("config_hash", ""))
        if expected_cache_hash not in (None, cache_hash):
            raise ValueError("COG-BCI window cache config hash mismatch")
        expected_protocol_hash = self.config.get("task_protocol_hash")
        protocol_hash = str(protocol_summary.get("protocol_hash", ""))
        if expected_protocol_hash not in (None, protocol_hash):
            raise ValueError("COG-BCI task protocol hash mismatch")

        target = pd.read_parquet(target_index_path)
        target = target.loc[
            target["included_for_supervised"].astype(bool)
        ].copy()
        if target.empty or not target["status"].eq("accepted").all():
            raise ValueError("N-Back dataset must contain accepted windows only")
        if set(target["task_variant"]) != {
            "zero_back",
            "one_back",
            "two_back",
        }:
            raise ValueError("N-Back target contains unexpected task variants")
        if sorted(target["target"].unique().tolist()) != [0, 1, 2]:
            raise ValueError("N-Back target must contain classes 0, 1, 2")
        if target["sample_id"].duplicated().any():
            raise ValueError("N-Back target contains duplicate sample_id")
        sample_mapping_path = self.config.get("sample_id_mapping_path")
        sample_mapping: pd.DataFrame | None = None
        if sample_mapping_path is not None:
            sample_mapping = pd.read_parquet(Path(sample_mapping_path))
            required_mapping = {
                "raw_sample_id",
                "variant_sample_id",
                "record_id",
                "window_index",
            }
            if not required_mapping.issubset(sample_mapping):
                raise ValueError(
                    "COG-BCI sample mapping is missing columns "
                    f"{sorted(required_mapping - set(sample_mapping))}"
                )
            if (
                sample_mapping["raw_sample_id"].duplicated().any()
                or sample_mapping["variant_sample_id"].duplicated().any()
            ):
                raise ValueError("COG-BCI sample mapping must be one-to-one")
            target = target.rename(columns={"sample_id": "protocol_sample_id"})
            target = target.merge(
                sample_mapping[
                    [
                        "raw_sample_id",
                        "variant_sample_id",
                        "record_id",
                        "window_index",
                    ]
                ].rename(
                    columns={
                        "raw_sample_id": "protocol_sample_id",
                        "variant_sample_id": "sample_id",
                    }
                ),
                on=["protocol_sample_id", "record_id", "window_index"],
                how="left",
                validate="one_to_one",
            )
            if target["sample_id"].isna().any():
                raise ValueError("Sample mapping does not cover every target row")

        cache_rows = pd.read_parquet(
            window_index_path,
            columns=[
                "sample_id",
                "cache_offset",
                "channel_order",
                "sampling_rate_hz",
                "preprocessing_name",
            ],
        )
        frame = target.merge(
            cache_rows, on="sample_id", how="left", validate="one_to_one"
        )
        if frame["cache_offset"].isna().any() or frame["cache_offset"].lt(0).any():
            raise ValueError("Accepted target rows are absent from cache shards")
        outer = pd.read_parquet(outer_assignments_path)[["sample_id", "fold"]]
        outer_key = "sample_id"
        if sample_mapping is not None:
            outer = outer.rename(columns={"sample_id": "protocol_sample_id"})
            outer_key = "protocol_sample_id"
        outer = outer.rename(columns={"fold": "outer_fold"})
        frame = frame.merge(
            outer, on=outer_key, how="left", validate="one_to_one"
        )
        if frame["outer_fold"].isna().any():
            raise ValueError("Every N-Back sample requires an outer fold")
        if sorted(frame["outer_fold"].unique().tolist()) != [1, 2, 3, 4, 5]:
            raise ValueError("N-Back outer assignments must contain five folds")

        channel_order = tuple(cache_manifest.get("channel_order", []))
        if len(channel_order) != 14 or "ECG1" in channel_order:
            raise ValueError("Expected 14 EEG channels without ECG1")
        expected_shape = (
            int(cache_manifest.get("channel_count", 0)),
            int(cache_manifest.get("samples_per_window", 0)),
        )
        if expected_shape != (14, 2560):
            raise ValueError(
                f"Expected COG-BCI window shape (14, 2560), got {expected_shape}"
            )
        serialized_order = json.dumps(
            list(channel_order), separators=(",", ":")
        )
        if not frame["channel_order"].eq(serialized_order).all():
            raise ValueError("Window channel order differs from cache manifest")

        cache_files: list[str] = []
        shard_metadata: dict[str, dict[str, Any]] = {}
        for record_id in frame["record_id"].astype(str):
            stem = _shard_stem(record_id)
            array_path = cache_dir / "shards" / f"{stem}.npy"
            metadata_path = cache_dir / "shards" / f"{stem}.json"
            if not array_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(
                    f"Missing COG-BCI cache shard for {record_id}"
                )
            if stem not in shard_metadata:
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
                if (
                    metadata.get("record_id") != record_id
                    or metadata.get("config_hash") != cache_hash
                    or tuple(metadata.get("array_shape", [])[1:])
                    != expected_shape
                ):
                    raise ValueError(
                        f"Incompatible COG-BCI shard metadata for {record_id}"
                    )
                shard_metadata[stem] = metadata
            cache_files.append(str(array_path))
        frame["cache_file"] = cache_files
        frame["n_channels"] = expected_shape[0]
        frame["n_samples_expected"] = expected_shape[1]
        frame["status_for_view"] = "ok"
        view_manifest = frame.rename(
            columns={"status": "target_status", "status_for_view": "status"}
        )
        window_transform = str(
            self.config.get("window_transform", "none")
        )
        if window_transform == "none":
            view = RawEEGWindowArrayView(view_manifest)
        elif window_transform == "per_window_mean_removal":
            view = PerWindowCenteredRawEEGWindowArrayView(view_manifest)
        else:
            raise ValueError(
                f"Unknown COG-BCI window_transform {window_transform!r}"
            )
        labels = frame["target"].to_numpy(dtype=np.int64)
        inner_assignments = pd.read_parquet(inner_assignments_path)
        if sample_mapping is not None:
            inner_assignments = inner_assignments.rename(
                columns={"sample_id": "protocol_sample_id"}
            ).merge(
                sample_mapping[["raw_sample_id", "variant_sample_id"]].rename(
                    columns={
                        "raw_sample_id": "protocol_sample_id",
                        "variant_sample_id": "sample_id",
                    }
                ),
                on="protocol_sample_id",
                how="left",
                validate="many_to_one",
            )
            if inner_assignments["sample_id"].isna().any():
                raise ValueError("Sample mapping does not cover inner assignments")
        if len(inner_assignments) != len(frame) * 5:
            raise ValueError(
                "Inner assignments must contain one partition per sample/fold"
            )

        self._data = EEGData(
            data=view,  # type: ignore[arg-type]
            labels=labels,
            subject_ids=frame["subject_id"].astype(str).to_numpy(),
            feature_names=list(channel_order),
            sampling_rate=float(cache_manifest["sampling_rate_hz"]),
            sample_ids=frame["sample_id"].astype(str).to_numpy(),
            record_ids=frame["record_id"].astype(str).to_numpy(),
            row_metadata={
                column: frame[column].to_numpy()
                for column in (
                    "session_id",
                    "record_group_id",
                    "window_index",
                    "task_variant",
                    "outer_fold",
                    "cache_offset",
                )
            },
            metadata={
                "dataset": "cog_bci",
                "task_id": EXPECTED_TASK_ID,
                "target_name": EXPECTED_TARGET,
                "observation_unit": "raw_eeg_window",
                "input_shape": [1, *expected_shape],
                "channel_order": list(channel_order),
                "channel_mapping_hash": cache_manifest[
                    "channel_mapping_hash"
                ],
                "window_cache_config_hash": cache_hash,
                "task_protocol_hash": protocol_hash,
                "outer_split_hash": protocol_summary["outer_split_hash"],
                "inner_split_hash": protocol_summary["inner_split_hash"],
                "target_schema_hash": protocol_summary["target_schema_hash"],
                "target_index_hash": protocol_summary["target_index_hash"],
                "source_filter_status": cache_manifest[
                    "source_filter_status"
                ],
                "input_unit": "volt_after_mne_reader",
                "source_physical_unit_status": "not_exposed_by_reader",
                "window_transform": window_transform,
                "sample_id_mapping": sample_mapping_path,
                "inner_assignments": inner_assignments,
                "frame": frame,
            },
        )
        return self._data

    def get_description(self) -> Dict[str, Any]:
        data = self.data
        return {
            "name": "cog_bci_nback_raw",
            "task_id": EXPECTED_TASK_ID,
            "n_samples": data.n_samples,
            "n_subjects": data.n_subjects,
            "n_records": int(len(np.unique(data.record_ids))),
            "input_shape": list(data.data.shape[1:]),
            "sampling_rate": data.sampling_rate,
        }
