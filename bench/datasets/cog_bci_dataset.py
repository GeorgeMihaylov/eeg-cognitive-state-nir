"""Lazy record-level access to the extracted COG-BCI EEGLAB dataset."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..core.abstract_dataset import BaseRecordDataset


INDEX_SCHEMA_VERSION = 1
DATASET_VERSION = "zenodo-7413650-v4"
DEFAULT_CACHE_PATH = Path("benchmark_results/cog_bci_record_index/index.json")
SUBJECT_RE = re.compile(r"(?i)^sub[-_]?(\d{1,2})$")
SESSION_RE = re.compile(r"(?i)^ses[-_]?s?(\d{1,2})$")

TASK_NORMALIZATION: dict[str, tuple[str, str, str | None]] = {
    "zeroback": ("n_back", "zero_back", "zero_back"),
    "oneback": ("n_back", "one_back", "one_back"),
    "twoback": ("n_back", "two_back", "two_back"),
    "matbeasy": ("matb", "matb_easy", "easy"),
    "matbmed": ("matb", "matb_medium", "medium"),
    "matbdiff": ("matb", "matb_difficult", "difficult"),
    "pvt": ("pvt", "pvt", None),
    "flanker": ("flanker", "flanker", None),
    "rs_beg_eo": ("resting_state", "rest_begin_eyes_open", "eyes_open"),
    "rs_beg_ec": ("resting_state", "rest_begin_eyes_closed", "eyes_closed"),
    "rs_end_eo": ("resting_state", "rest_end_eyes_open", "eyes_open"),
    "rs_end_ec": ("resting_state", "rest_end_eyes_closed", "eyes_closed"),
}

AUXILIARY_NAME_RE = re.compile(
    r"(?i)^(ecg|ekg|eog|emg|resp|stim|status|trigger|misc)"
)


class COGBCIIndexError(ValueError):
    """Raised when the extracted record tree violates the index contract."""


class COGBCICacheError(ValueError):
    """Raised when a metadata cache is stale or incompatible."""


@dataclass(frozen=True)
class COGBCIRecord:
    """Deterministic metadata for one paired EEGLAB recording."""

    record_id: str
    subject_id: str
    subject_number: int
    session_id: str
    session_label_raw: str
    task_family: str
    task_variant: str
    task_label_raw: str
    condition: str | None
    run_id: str | None
    set_relative_path: str
    fdt_relative_path: str
    set_size_bytes: int
    fdt_size_bytes: int
    sampling_rate_hz: float
    n_samples: int
    duration_seconds: float
    channel_count_total: int
    channel_count_eeg: int
    channel_count_auxiliary: int
    channel_names_total: tuple[str, ...]
    eeg_channel_names: tuple[str, ...]
    auxiliary_channel_names: tuple[str, ...]
    mne_channel_types: tuple[str, ...]
    has_cz: bool
    has_ecg1: bool
    event_count: int
    event_types: tuple[str, ...]
    channel_layout_id: str
    reader: str
    reference: str | None
    montage_status: str
    channels_without_scalp_position: tuple[str, ...]
    eeg_channels_without_scalp_position: tuple[str, ...]
    data_units: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "COGBCIRecord":
        document = dict(value)
        for field_name in ("set_relative_path", "fdt_relative_path"):
            path = str(document[field_name]).replace("\\", "/")
            if Path(path).is_absolute() or PureWindowsPath(path).is_absolute():
                raise COGBCICacheError(
                    f"{field_name} must be relative, got {document[field_name]!r}"
                )
            document[field_name] = path
        for field_name in (
            "channel_names_total",
            "eeg_channel_names",
            "auxiliary_channel_names",
            "mne_channel_types",
            "event_types",
            "channels_without_scalp_position",
            "eeg_channels_without_scalp_position",
            "data_units",
        ):
            document[field_name] = tuple(str(item) for item in document[field_name])
        return cls(**document)


def _require_mne() -> Any:
    if importlib.util.find_spec("mne") is None:
        raise ModuleNotFoundError(
            "COG-BCI lazy EEGLAB access requires MNE. "
            "Install it in the active environment with: python -m pip install "
            "'mne>=1.12,<2'"
        )
    import mne  # type: ignore

    return mne


def _normalize_subject(value: str | int) -> tuple[str, int]:
    text = str(value).strip()
    match = SUBJECT_RE.fullmatch(text)
    if match is None and text.isdigit():
        match = re.fullmatch(r"(\d{1,2})", text)
    if match is None:
        raise COGBCIIndexError(f"Unknown COG-BCI subject identifier: {value!r}")
    number = int(match.group(1))
    if not 1 <= number <= 99:
        raise COGBCIIndexError(f"Invalid COG-BCI subject number: {number}")
    return f"sub-{number:02d}", number


def _normalize_session(value: str | int) -> tuple[str, str]:
    text = str(value).strip()
    match = SESSION_RE.fullmatch(text)
    if match is None and text.isdigit():
        match = re.fullmatch(r"(\d{1,2})", text)
    if match is None:
        raise COGBCIIndexError(f"Unknown COG-BCI session identifier: {value!r}")
    number = int(match.group(1))
    return f"ses-{number:02d}", text


def _normalize_task(raw_label: str) -> tuple[str, str, str | None]:
    key = raw_label.casefold()
    try:
        return TASK_NORMALIZATION[key]
    except KeyError as error:
        raise COGBCIIndexError(
            f"Unknown COG-BCI task label {raw_label!r}; "
            f"known labels={sorted(TASK_NORMALIZATION)}"
        ) from error


def _identity_from_path(relative_path: Path) -> tuple[str, int, str, str, str]:
    subject_values: set[tuple[str, int]] = set()
    session_values: set[tuple[str, str]] = set()
    for part in relative_path.parts:
        if SUBJECT_RE.fullmatch(part):
            subject_values.add(_normalize_subject(part))
        if SESSION_RE.fullmatch(part):
            session_values.add(_normalize_session(part))
    if len(subject_values) != 1:
        raise COGBCIIndexError(
            f"Expected one subject identifier in {relative_path.as_posix()}, "
            f"found={sorted(subject_values)}"
        )
    if len(session_values) != 1:
        raise COGBCIIndexError(
            f"Expected one session identifier in {relative_path.as_posix()}, "
            f"found={sorted(session_values)}"
        )
    subject_id, subject_number = next(iter(subject_values))
    session_id, session_raw = next(iter(session_values))
    return subject_id, subject_number, session_id, session_raw, relative_path.stem


def _record_id(
    subject_id: str,
    session_id: str,
    task_variant: str,
    run_id: str | None,
) -> str:
    suffix = "run-na" if run_id is None else f"run-{run_id}"
    return f"cog_bci::{subject_id}::{session_id}::{task_variant}::{suffix}"


def _has_scalp_position(channel: Mapping[str, Any]) -> bool:
    location = channel.get("loc")
    if location is None:
        return False
    values = list(location)[:3]
    return any(
        math.isfinite(float(value)) and float(value) != 0.0 for value in values
    )


def _read_record_header(set_path: Path) -> dict[str, Any]:
    """Read one EEGLAB header without preloading its signal array."""

    mne = _require_mne()
    raw = mne.io.read_raw_eeglab(
        str(set_path),
        preload=False,
        verbose="ERROR",
    )
    try:
        channel_names = tuple(str(name) for name in raw.ch_names)
        channel_types = tuple(str(name) for name in raw.get_channel_types())
        auxiliary_mask = tuple(
            channel_type.casefold() != "eeg"
            or AUXILIARY_NAME_RE.match(channel_name) is not None
            for channel_name, channel_type in zip(channel_names, channel_types)
        )
        auxiliary = tuple(
            name for name, is_aux in zip(channel_names, auxiliary_mask) if is_aux
        )
        eeg = tuple(
            name for name, is_aux in zip(channel_names, auxiliary_mask) if not is_aux
        )
        channel_info = tuple(raw.info.get("chs", ()))
        missing_position = tuple(
            name
            for name, info in zip(channel_names, channel_info)
            if not _has_scalp_position(info)
        )
        missing_eeg_position = tuple(
            name for name in eeg if name in set(missing_position)
        )
        if missing_eeg_position:
            montage_status = "missing_eeg_positions"
        elif missing_position:
            montage_status = "auxiliary_missing_only"
        else:
            montage_status = "complete"
        annotations = getattr(raw, "annotations", None)
        descriptions = (
            ()
            if annotations is None
            else tuple(
                str(value)
                for value in getattr(annotations, "description", ())
            )
        )
        units_map = getattr(raw, "_orig_units", None) or {}
        units = tuple(
            sorted({str(value) for value in units_map.values() if value})
        )
        sampling_rate = float(raw.info["sfreq"])
        n_samples = int(raw.n_times)
        return {
            "sampling_rate_hz": sampling_rate,
            "n_samples": n_samples,
            "duration_seconds": n_samples / sampling_rate,
            "channel_names_total": channel_names,
            "eeg_channel_names": eeg,
            "auxiliary_channel_names": auxiliary,
            "mne_channel_types": channel_types,
            "event_count": len(descriptions),
            "event_types": tuple(sorted(set(descriptions))),
            "reference": str(raw.info.get("custom_ref_applied", "")) or None,
            "montage_status": montage_status,
            "channels_without_scalp_position": missing_position,
            "eeg_channels_without_scalp_position": missing_eeg_position,
            "data_units": units,
            "reader": "mne.io.read_raw_eeglab(preload=False)",
        }
    finally:
        close = getattr(raw, "close", None)
        if callable(close):
            close()


def source_root_fingerprint(root: Path) -> str:
    """Hash relative pair paths and file stats without reading signal payloads."""

    root = Path(root)
    entries: list[tuple[str, int, int]] = []
    for path in sorted(
        (
            item
            for item in root.rglob("*")
            if item.is_file() and item.suffix.casefold() in {".set", ".fdt"}
        ),
        key=lambda item: item.relative_to(root).as_posix().casefold(),
    ):
        stat = path.stat()
        entries.append(
            (path.relative_to(root).as_posix(), int(stat.st_size), int(stat.st_mtime_ns))
        )
    payload = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_inventory_timestamp(root: Path) -> str:
    """Return a deterministic timestamp derived from the newest source pair."""

    timestamps = [
        path.stat().st_mtime
        for path in Path(root).rglob("*")
        if path.is_file() and path.suffix.casefold() in {".set", ".fdt"}
    ]
    if not timestamps:
        return datetime.fromtimestamp(0, timezone.utc).isoformat()
    return datetime.fromtimestamp(max(timestamps), timezone.utc).isoformat()


@dataclass(frozen=True)
class COGBCIRecordIndex:
    """A deterministic, serializable COG-BCI record index."""

    records: tuple[COGBCIRecord, ...]
    source_root_fingerprint: str
    inventory_timestamp: str
    schema_version: int = INDEX_SCHEMA_VERSION
    dataset_version: str = DATASET_VERSION

    @classmethod
    def build(cls, root: Path) -> "COGBCIRecordIndex":
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"COG-BCI extracted root not found: {root}")
        files = sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.casefold() in {".set", ".fdt"}
            ),
            key=lambda path: path.relative_to(root).as_posix().casefold(),
        )
        grouped: dict[tuple[str, str], dict[str, list[Path]]] = {}
        for path in files:
            relative = path.relative_to(root)
            key = (relative.parent.as_posix().casefold(), path.stem.casefold())
            grouped.setdefault(key, {"set": [], "fdt": []})[
                path.suffix.casefold().lstrip(".")
            ].append(path)
        pair_errors: list[str] = []
        for key, value in sorted(grouped.items()):
            if len(value["set"]) != 1 or len(value["fdt"]) != 1:
                pair_errors.append(
                    f"{key[0]}/{key[1]}: set={len(value['set'])}, "
                    f"fdt={len(value['fdt'])}"
                )
        if pair_errors:
            raise COGBCIIndexError(
                "Incomplete or duplicate EEGLAB pairs: " + "; ".join(pair_errors[:20])
            )

        records: list[COGBCIRecord] = []
        seen_record_ids: set[str] = set()
        seen_path_pairs: set[tuple[str, str]] = set()
        for key in sorted(grouped):
            set_path = grouped[key]["set"][0]
            fdt_path = grouped[key]["fdt"][0]
            set_relative = set_path.relative_to(root)
            fdt_relative = fdt_path.relative_to(root)
            subject_id, subject_number, session_id, session_raw, task_raw = (
                _identity_from_path(set_relative)
            )
            task_family, task_variant, condition = _normalize_task(task_raw)
            run_id = None
            record_id = _record_id(
                subject_id, session_id, task_variant, run_id
            )
            if record_id in seen_record_ids:
                raise COGBCIIndexError(f"Duplicate record_id detected: {record_id}")
            pair = (set_relative.as_posix(), fdt_relative.as_posix())
            if pair in seen_path_pairs:
                raise COGBCIIndexError(f"Duplicate path pair detected: {pair}")
            try:
                header = _read_record_header(set_path)
            except Exception as error:
                raise COGBCIIndexError(
                    f"Failed to read {record_id} at {set_relative.as_posix()}: "
                    f"{type(error).__name__}: {error}"
                ) from error
            channel_names = tuple(header["channel_names_total"])
            eeg_names = tuple(header["eeg_channel_names"])
            auxiliary_names = tuple(header["auxiliary_channel_names"])
            layout_payload = "\n".join(channel_names).encode("utf-8")
            layout_id = "layout-" + hashlib.sha256(layout_payload).hexdigest()[:12]
            record = COGBCIRecord(
                record_id=record_id,
                subject_id=subject_id,
                subject_number=subject_number,
                session_id=session_id,
                session_label_raw=session_raw,
                task_family=task_family,
                task_variant=task_variant,
                task_label_raw=task_raw,
                condition=condition,
                run_id=run_id,
                set_relative_path=set_relative.as_posix(),
                fdt_relative_path=fdt_relative.as_posix(),
                set_size_bytes=set_path.stat().st_size,
                fdt_size_bytes=fdt_path.stat().st_size,
                sampling_rate_hz=float(header["sampling_rate_hz"]),
                n_samples=int(header["n_samples"]),
                duration_seconds=float(header["duration_seconds"]),
                channel_count_total=len(channel_names),
                channel_count_eeg=len(eeg_names),
                channel_count_auxiliary=len(auxiliary_names),
                channel_names_total=channel_names,
                eeg_channel_names=eeg_names,
                auxiliary_channel_names=auxiliary_names,
                mne_channel_types=tuple(header["mne_channel_types"]),
                has_cz=any(name.casefold() == "cz" for name in channel_names),
                has_ecg1=any(name.casefold() == "ecg1" for name in channel_names),
                event_count=int(header["event_count"]),
                event_types=tuple(header["event_types"]),
                channel_layout_id=layout_id,
                reader=str(header["reader"]),
                reference=header["reference"],
                montage_status=str(header["montage_status"]),
                channels_without_scalp_position=tuple(
                    header["channels_without_scalp_position"]
                ),
                eeg_channels_without_scalp_position=tuple(
                    header["eeg_channels_without_scalp_position"]
                ),
                data_units=tuple(header["data_units"]),
            )
            records.append(record)
            seen_record_ids.add(record_id)
            seen_path_pairs.add(pair)
        records.sort(key=lambda record: record.record_id)
        return cls(
            records=tuple(records),
            source_root_fingerprint=source_root_fingerprint(root),
            inventory_timestamp=source_inventory_timestamp(root),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "inventory_timestamp": self.inventory_timestamp,
            "source_root_fingerprint": self.source_root_fingerprint,
            "record_count": len(self.records),
            "records": [record.to_dict() for record in self.records],
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, root: Path) -> "COGBCIRecordIndex":
        path = Path(path)
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        schema_version = document.get("schema_version")
        if schema_version != INDEX_SCHEMA_VERSION:
            raise COGBCICacheError(
                f"Incompatible COG-BCI index schema {schema_version!r}; "
                f"expected {INDEX_SCHEMA_VERSION}"
            )
        dataset_version = document.get("dataset_version")
        if dataset_version != DATASET_VERSION:
            raise COGBCICacheError(
                f"Incompatible dataset version {dataset_version!r}; "
                f"expected {DATASET_VERSION}"
            )
        actual_fingerprint = source_root_fingerprint(Path(root))
        stored_fingerprint = str(document.get("source_root_fingerprint", ""))
        if stored_fingerprint != actual_fingerprint:
            raise COGBCICacheError(
                "Stale COG-BCI index: source root fingerprint changed"
            )
        records = tuple(
            COGBCIRecord.from_dict(value)
            for value in document.get("records", ())
        )
        if int(document.get("record_count", -1)) != len(records):
            raise COGBCICacheError(
                "COG-BCI cache record_count does not match serialized records"
            )
        for record in records:
            for relative in (
                record.set_relative_path,
                record.fdt_relative_path,
            ):
                if not (Path(root) / relative).is_file():
                    raise COGBCICacheError(
                        f"Cached COG-BCI file no longer exists: {relative}"
                    )
        return cls(
            records=records,
            source_root_fingerprint=stored_fingerprint,
            inventory_timestamp=str(document.get("inventory_timestamp", "")),
            schema_version=INDEX_SCHEMA_VERSION,
            dataset_version=DATASET_VERSION,
        )


class COGBCIDataset(BaseRecordDataset):
    """Record discovery, filtering and lazy single-record EEGLAB access."""

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(dict(config))
        path_value = self.config.get("data_path", self.config.get("root"))
        if path_value in (None, ""):
            raise ValueError("COG-BCI dataset requires data_path")
        self.root = Path(path_value)
        if not self.root.is_dir():
            raise FileNotFoundError(f"COG-BCI extracted root not found: {self.root}")
        self.cache_path = Path(
            self.config.get("index_cache_path", DEFAULT_CACHE_PATH)
        )
        self.use_cache = bool(self.config.get("use_index_cache", True))
        self.rebuild_index = bool(self.config.get("rebuild_index", False))
        self.include_auxiliary_channels = bool(
            self.config.get("include_auxiliary_channels", False)
        )
        self.require_canonical_complete = bool(
            self.config.get("require_canonical_complete", True)
        )
        self._index: COGBCIRecordIndex | None = None

    @property
    def index(self) -> COGBCIRecordIndex:
        if self._index is None:
            if self.use_cache and self.cache_path.is_file() and not self.rebuild_index:
                self._index = COGBCIRecordIndex.load(self.cache_path, self.root)
            else:
                self._index = COGBCIRecordIndex.build(self.root)
                if self.use_cache:
                    self._index.save(self.cache_path)
            if self.require_canonical_complete:
                self.validate_expected_structure()
        return self._index

    @property
    def records(self) -> tuple[COGBCIRecord, ...]:
        return self.index.records

    def validate_expected_structure(self) -> None:
        records = self._index.records if self._index is not None else self.index.records
        subjects = {record.subject_id for record in records}
        sessions = {record.session_id for record in records}
        subject_sessions = {
            (record.subject_id, record.session_id) for record in records
        }
        observed = {
            "records": len(records),
            "subjects": len(subjects),
            "sessions": len(sessions),
            "subject_sessions": len(subject_sessions),
        }
        expected = {
            "records": 1044,
            "subjects": 29,
            "sessions": 3,
            "subject_sessions": 87,
        }
        if observed != expected:
            raise COGBCIIndexError(
                f"COG-BCI canonical structure mismatch: observed={observed}, "
                f"expected={expected}"
            )

    def get_record(self, record_id: str) -> COGBCIRecord:
        matches = [record for record in self.records if record.record_id == record_id]
        if not matches:
            raise KeyError(f"Unknown COG-BCI record_id: {record_id}")
        return matches[0]

    @staticmethod
    def _validated_values(
        requested: Sequence[Any] | None,
        available: set[Any],
        *,
        label: str,
        normalizer: Any = str,
    ) -> set[Any] | None:
        if requested is None:
            return None
        if isinstance(requested, (str, bytes)):
            requested = [requested]
        normalized = {normalizer(value) for value in requested}
        unknown = sorted(normalized - available, key=str)
        if unknown:
            raise ValueError(
                f"Unknown COG-BCI {label} filter values: {unknown}; "
                f"available={sorted(available, key=str)}"
            )
        return normalized

    def query(
        self,
        *,
        subject_ids: Sequence[str] | None = None,
        session_ids: Sequence[str] | None = None,
        task_families: Sequence[str] | None = None,
        task_variants: Sequence[str] | None = None,
        has_cz: bool | None = None,
        channel_layout_ids: Sequence[str] | None = None,
    ) -> tuple[COGBCIRecord, ...]:
        records = self.records
        subjects = self._validated_values(
            subject_ids,
            {record.subject_id for record in records},
            label="subject_id",
            normalizer=lambda value: _normalize_subject(value)[0],
        )
        sessions = self._validated_values(
            session_ids,
            {record.session_id for record in records},
            label="session_id",
            normalizer=lambda value: _normalize_session(value)[0],
        )
        families = self._validated_values(
            task_families,
            {record.task_family for record in records},
            label="task_family",
            normalizer=lambda value: str(value).strip().casefold(),
        )
        variants = self._validated_values(
            task_variants,
            {record.task_variant for record in records},
            label="task_variant",
            normalizer=lambda value: str(value).strip().casefold(),
        )
        layouts = self._validated_values(
            channel_layout_ids,
            {record.channel_layout_id for record in records},
            label="channel_layout_id",
        )
        selected = [
            record
            for record in records
            if (subjects is None or record.subject_id in subjects)
            and (sessions is None or record.session_id in sessions)
            and (families is None or record.task_family in families)
            and (variants is None or record.task_variant in variants)
            and (has_cz is None or record.has_cz is bool(has_cz))
            and (layouts is None or record.channel_layout_id in layouts)
        ]
        return tuple(sorted(selected, key=lambda record: record.record_id))

    def iter_records(self, **filters: Any) -> Iterator[COGBCIRecord]:
        return iter(self.query(**filters))

    def open_raw(
        self,
        record_id: str,
        *,
        preload: bool = False,
        include_auxiliary_channels: bool | None = None,
    ) -> Any:
        record = self.get_record(record_id)
        set_path = self.root / record.set_relative_path
        include_auxiliary = (
            self.include_auxiliary_channels
            if include_auxiliary_channels is None
            else bool(include_auxiliary_channels)
        )
        mne = _require_mne()
        try:
            raw = mne.io.read_raw_eeglab(str(set_path), preload=preload)
            if not include_auxiliary:
                raw.pick(list(record.eeg_channel_names), ordered=True)
            return raw
        except Exception as error:
            raise RuntimeError(
                f"Failed to open COG-BCI record {record.record_id} at "
                f"{record.set_relative_path}: {type(error).__name__}: {error}"
            ) from error

    def get_description(self) -> dict[str, Any]:
        records = self.records
        return {
            "name": "cog_bci",
            "dataset_version": DATASET_VERSION,
            "access_level": "lazy_record",
            "record_count": len(records),
            "subject_count": len({record.subject_id for record in records}),
            "session_count": len({record.session_id for record in records}),
            "subject_session_count": len(
                {(record.subject_id, record.session_id) for record in records}
            ),
            "task_families": sorted({record.task_family for record in records}),
            "sampling_rates_hz": sorted(
                {record.sampling_rate_hz for record in records}
            ),
            "channel_layout_ids": sorted(
                {record.channel_layout_id for record in records}
            ),
            "preload_default": False,
            "window_materialization": "not_implemented",
            "scientific_task": "not_implemented",
            "training": "not_implemented",
        }
