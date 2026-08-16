"""Lazy record-level loaders for the extracted CLARE and CL-Drive datasets."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import pandas as pd

from ..core.abstract_dataset import BaseRecordDataset


EEG_CHANNELS = ("TP9", "AF7", "AF8", "TP10")
PRIMARY_MODALITIES = ("EEG", "ECG", "EDA")
ALL_MODALITIES = (*PRIMARY_MODALITIES, "Gaze")


@dataclass(frozen=True)
class CognitiveLoadMultimodalRecord:
    dataset_id: str
    record_id: str
    participant_id: str
    source_participant_id: str
    task_id: str
    task_number: int
    label_path: str
    eeg_path: str | None
    ecg_path: str | None
    eda_path: str | None
    gaze_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CognitiveLoadMultimodalDataset(BaseRecordDataset):
    """Common lazy loader for label-addressable CLARE/CL-Drive task records."""

    dataset_id: str
    task_prefix: str
    task_pattern: re.Pattern[str]

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(dict(config))
        self.root = Path(self.config["data_path"])
        if not self.root.is_absolute():
            self.root = Path(__file__).resolve().parents[2] / self.root
        if not self.root.is_dir():
            raise FileNotFoundError(f"Extracted {self.dataset_id} root not found: {self.root}")

    @staticmethod
    def _participant_id(source_id: str) -> str:
        return f"sub-{source_id}"

    def _task_number_from_column(self, column: str) -> int:
        if column == "time":
            raise ValueError("The time column is not a task")
        match = re.fullmatch(r"(?:level|lvl)_(\d+)", column.strip().lower())
        if match is None:
            raise ValueError(f"Unknown {self.dataset_id} label column: {column!r}")
        return int(match.group(1))

    def _record_files(self, modality: str) -> dict[tuple[str, int], Path]:
        records: dict[tuple[str, int], Path] = {}
        for path in sorted((self.root / modality).glob("*/*.csv")):
            if "baseline" in path.stem.lower():
                continue
            match = self.task_pattern.fullmatch(path.stem.lower())
            if match is None:
                raise ValueError(f"Unknown {self.dataset_id} {modality} file: {path}")
            key = (path.parent.name, int(match.group(1)))
            if key in records:
                raise ValueError(f"Duplicate {self.dataset_id} record file for {key}")
            records[key] = path
        return records

    def iter_records(self) -> Iterator[CognitiveLoadMultimodalRecord]:
        paths = {modality: self._record_files(modality) for modality in ALL_MODALITIES}
        records: list[CognitiveLoadMultimodalRecord] = []
        label_root = self.root / "Labels"
        for label_path in sorted(label_root.glob("*.csv")):
            source_id = label_path.stem
            header = list(pd.read_csv(label_path, nrows=0).columns)
            for column in header:
                if column.strip().lower() == "time":
                    continue
                task_number = self._task_number_from_column(column)
                key = (source_id, task_number)
                participant_id = self._participant_id(source_id)
                task_id = f"task-{task_number:02d}"
                values = {
                    modality.lower() + "_path": (
                        None
                        if paths[modality].get(key) is None
                        else paths[modality][key].relative_to(self.root).as_posix()
                    )
                    for modality in ALL_MODALITIES
                }
                records.append(
                    CognitiveLoadMultimodalRecord(
                        dataset_id=self.dataset_id,
                        record_id=f"{self.dataset_id}__{participant_id}__{task_id}",
                        participant_id=participant_id,
                        source_participant_id=source_id,
                        task_id=task_id,
                        task_number=task_number,
                        label_path=label_path.relative_to(self.root).as_posix(),
                        **values,
                    )
                )
        yield from sorted(records, key=lambda item: (item.source_participant_id, item.task_number))

    def validate_record_files(self, record: CognitiveLoadMultimodalRecord) -> dict[str, Any]:
        paths = {
            modality: getattr(record, f"{modality.lower()}_path")
            for modality in ALL_MODALITIES
        }
        missing = sorted(modality for modality, value in paths.items() if value is None)
        empty = sorted(
            modality
            for modality, value in paths.items()
            if value is not None and (self.root / value).stat().st_size == 0
        )
        return {
            "missing_modalities": missing,
            "empty_modalities": empty,
            "complete_primary_modalities": not any(
                modality in missing or modality in empty
                for modality in PRIMARY_MODALITIES
            ),
            "complete_all_modalities": not missing and not empty,
        }

    def get_description(self) -> dict[str, Any]:
        records = list(self.iter_records())
        return {
            "name": self.dataset_id,
            "access": "lazy_record_level",
            "data_path": self.root.as_posix(),
            "n_participants": len({record.participant_id for record in records}),
            "n_label_addressable_records": len(records),
            "eeg_channels": list(EEG_CHANNELS),
            "eeg_sampling_rate_hz": 256.0,
            "target_source": "subjective cognitive-load score every 10 seconds",
        }


class CLAREDataset(CognitiveLoadMultimodalDataset):
    dataset_id = "clare"
    task_prefix = "level"
    task_pattern = re.compile(r"(?:eeg_data_exp|ecg_data_experiment|eda_data_experiment|gaze_data_experiment)_(\d+)")


class CLDriveDataset(CognitiveLoadMultimodalDataset):
    dataset_id = "cl_drive"
    task_prefix = "lvl"
    task_pattern = re.compile(r"(?:eeg_data_level|ecg_data_level|eda_data_level|gaze_data_level)_(\d+)")


__all__ = [
    "ALL_MODALITIES",
    "PRIMARY_MODALITIES",
    "EEG_CHANNELS",
    "CLAREDataset",
    "CLDriveDataset",
    "CognitiveLoadMultimodalDataset",
    "CognitiveLoadMultimodalRecord",
]
