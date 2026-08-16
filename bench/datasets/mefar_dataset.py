"""Lazy, record-level access to the extracted MEFAR multimodal dataset."""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from ..core.abstract_dataset import BaseRecordDataset


SUBJECT_RE = re.compile(r"^subject_(\d+)$")
SESSION_RE = re.compile(r"^(\d+)\.(morning|evening)$")
CORE_FILES = ("EEG.csv", "ACC.csv", "BVP.csv", "EDA.csv", "HR.csv", "TEMP.csv")
OPTIONAL_FILES = ("IBI.csv", "tags.csv")


@dataclass(frozen=True)
class MEFARRecord:
    record_id: str
    participant_id: str
    participant_number: int
    session_id: str
    session_order: int
    session_label: str
    relative_path: str
    available_files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MEFARDataset(BaseRecordDataset):
    """MEFAR records remain lazy until explicit feature materialization."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.data_path = Path(config["data_path"])
        if not self.data_path.is_absolute():
            self.data_path = Path(__file__).resolve().parents[2] / self.data_path
        self.raw_root = self.data_path / "MEFAR"

    def iter_records(self) -> Iterator[MEFARRecord]:
        if not self.raw_root.is_dir():
            raise FileNotFoundError(f"Extracted MEFAR raw root not found: {self.raw_root}")
        records: list[MEFARRecord] = []
        for participant_dir in self.raw_root.glob("subject_*"):
            subject_match = SUBJECT_RE.fullmatch(participant_dir.name)
            if subject_match is None:
                continue
            number = int(subject_match.group(1))
            participant_id = f"sub-{number:02d}"
            for session_dir in participant_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                session_match = SESSION_RE.fullmatch(session_dir.name)
                if session_match is None:
                    raise ValueError(f"Unknown MEFAR session directory: {session_dir}")
                order = int(session_match.group(1))
                label = session_match.group(2)
                available = tuple(sorted(path.name for path in session_dir.iterdir() if path.is_file()))
                relative = session_dir.relative_to(self.data_path).as_posix()
                records.append(
                    MEFARRecord(
                        record_id=f"{participant_id}__ses-{order:02d}-{label}",
                        participant_id=participant_id,
                        participant_number=number,
                        session_id=f"ses-{order:02d}-{label}",
                        session_order=order,
                        session_label=label,
                        relative_path=relative,
                        available_files=available,
                    )
                )
        yield from sorted(records, key=lambda item: (item.participant_number, item.session_order))

    def validate_record_files(self, record: MEFARRecord) -> dict[str, Any]:
        session_dir = self.data_path / record.relative_path
        missing_core = [name for name in CORE_FILES if not (session_dir / name).is_file()]
        empty_core = [
            name for name in CORE_FILES
            if (session_dir / name).is_file() and (session_dir / name).stat().st_size == 0
        ]
        return {
            "missing_core_files": missing_core,
            "empty_core_files": empty_core,
            "complete_core_modalities": not missing_core and not empty_core,
        }

    def get_description(self) -> dict[str, Any]:
        records = list(self.iter_records())
        participants = sorted({record.participant_id for record in records})
        return {
            "name": "MEFAR",
            "access": "lazy_record_level",
            "data_path": self.data_path.as_posix(),
            "n_participants": len(participants),
            "n_records": len(records),
            "session_labels": sorted({record.session_label for record in records}),
            "eeg_contract": "NeuroSky-derived band powers; no raw EEG channels",
            "wearable_contract": "Empatica E4 ACC/BVP/EDA/HR/IBI/TEMP",
        }


def count_csv_rows(path: Path) -> int:
    """Count physical CSV rows without materializing the file."""
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        return sum(1 for _ in csv.reader(stream))


__all__ = ["MEFARDataset", "MEFARRecord", "CORE_FILES", "OPTIONAL_FILES", "count_csv_rows"]
