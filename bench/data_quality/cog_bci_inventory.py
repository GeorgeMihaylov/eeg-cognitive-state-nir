#!/usr/bin/env python3
"""Verify, safely extract, and inventory local COG-BCI archives.

This is deliberately a data-management tool, not a dataset loader.  It never
preloads EEG arrays, creates targets, or modifies source archives.
"""

from __future__ import annotations

import argparse
import binascii
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
EXPECTED_SUBJECTS = tuple(f"sub-{number:02d}" for number in range(1, 30))
ARCHIVE_COLUMNS = [
    "filename",
    "subject_id",
    "size_bytes",
    "zip_readable",
    "zip_test_passed",
    "member_count",
    "compressed_size",
    "uncompressed_size",
    "checksum_expected",
    "checksum_actual",
    "checksum_match",
    "status",
    "error",
]
EXTRACTION_COLUMNS = [
    "filename",
    "subject_id",
    "member",
    "relative_path",
    "size_bytes",
    "crc32",
    "status",
    "error",
]
RECORD_COLUMNS = [
    "subject_id",
    "session_id",
    "task",
    "run",
    "file_type",
    "relative_path",
    "size_bytes",
    "paired_set_file",
    "paired_fdt_file",
]
CHANNEL_COLUMNS = [
    "subject_id",
    "session_id",
    "task",
    "run",
    "relative_path",
    "paired_fdt_file",
    "n_channels",
    "channel_names",
    "sampling_rate_hz",
    "n_samples",
    "duration_seconds",
    "event_count",
    "event_types",
    "annotation_count",
    "reference",
    "bad_channels",
    "data_units",
    "reader_used",
    "read_status",
    "error",
]
EVENT_COLUMNS = [
    "subject_id",
    "session_id",
    "task",
    "run",
    "relative_path",
    "event_source",
    "event_label",
    "trigger_code",
    "rating_name",
    "behavioural_outcome",
]
BEHAVIOURAL_COLUMNS = [
    "subject_id",
    "session_id",
    "task",
    "relative_path",
    "source_file",
    "variable_name",
    "metric_name",
    "data_type",
    "value_count",
    "missing_count",
    "minimum",
    "maximum",
    "mean",
    "source_level",
    "read_status",
    "error",
]
FILE_PAIR_COLUMNS = [
    "subject_id",
    "session_id",
    "task",
    "run",
    "set_path",
    "fdt_path",
    "set_size_bytes",
    "fdt_size_bytes",
    "expected_fdt_size_bytes",
    "size_match",
    "status",
    "error",
]
TASK_COLUMNS = ["task", "record_count", "subject_count", "session_count"]
SESSION_COLUMNS = ["subject_id", "session_id", "record_count", "task_count"]
SUBJECT_COLUMNS = ["subject_id", "session_count", "record_count", "task_count"]
ERROR_COLUMNS = [
    "stage",
    "subject_id",
    "relative_path",
    "error_type",
    "error",
]
KNOWN_FILE_TYPES = frozenset({".set", ".fdt", ".tsv", ".json", ".txt", ".mat"})
SERVICE_NAMES = frozenset(
    {
        ".ds_store",
        "thumbs.db",
        "desktop.ini",
        "__macosx",
        ".cog_bci_incomplete.json",
    }
)
SUBJECT_TOKEN_RE = re.compile(r"(?i)(?:^|[^a-z0-9])sub[-_]?(\d{1,3})(?:[^0-9]|$)")
CANONICAL_ARCHIVE_RE = re.compile(r"(?i)^sub-(\d{2})\.zip$")
SESSION_RE = re.compile(r"(?i)(?:^|[^a-z0-9])ses[-_]?([a-z0-9]+)")
TASK_RE = re.compile(r"(?i)(?:^|[_-])task[-_]?([a-z0-9]+)")
RUN_RE = re.compile(r"(?i)(?:^|[_-])run[-_]?([a-z0-9]+)")


class InventoryError(RuntimeError):
    """Base error for fatal inventory operations."""


class UnsafeArchiveError(InventoryError):
    """Raised before extraction when a ZIP member is unsafe."""


class ExtractionConflictError(InventoryError):
    """Raised when extraction would overwrite changed content."""


@dataclass
class ToolResult:
    """In-memory result plus the process exit code."""

    exit_code: int
    overall_status: str
    archive_rows: list[dict[str, Any]] = field(default_factory=list)
    extraction_rows: list[dict[str, Any]] = field(default_factory=list)
    record_rows: list[dict[str, Any]] = field(default_factory=list)
    channel_rows: list[dict[str, Any]] = field(default_factory=list)
    event_rows: list[dict[str, Any]] = field(default_factory=list)
    behavioural_rows: list[dict[str, Any]] = field(default_factory=list)
    file_pair_rows: list[dict[str, Any]] = field(default_factory=list)
    task_rows: list[dict[str, Any]] = field(default_factory=list)
    session_rows: list[dict[str, Any]] = field(default_factory=list)
    subject_rows: list[dict[str, Any]] = field(default_factory=list)
    error_rows: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _bool_cell(value: bool | None) -> str:
    if value is None:
        return "not_available"
    return "true" if value else "false"


def _clean_subject(value: str | int) -> str:
    text = str(value)
    match = re.search(r"(\d{1,3})", text)
    if not match:
        raise ValueError(f"Invalid subject identifier: {value!r}")
    return f"sub-{int(match.group(1)):02d}"


def _subject_from_name(name: str) -> str:
    match = SUBJECT_TOKEN_RE.search(name)
    return _clean_subject(match.group(1)) if match else ""


def _portable_path(path: Path, root: Path | None = None) -> str:
    path = Path(path)
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except (OSError, ValueError):
            pass
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        return f"<external>/{path.name}"


def _safe_error_text(error: BaseException | str, roots: Iterable[Path] = ()) -> str:
    text = str(error).replace("\r", " ").replace("\n", " ")
    for root in roots:
        try:
            variants = {str(root), str(root.resolve())}
        except OSError:
            variants = {str(root)}
        for variant in variants:
            text = text.replace(variant, "<root>")
            text = text.replace(variant.replace("\\", "/"), "<root>")
    return text


def _error_row(
    stage: str,
    error: BaseException | str,
    *,
    subject_id: str = "",
    relative_path: str = "",
    roots: Iterable[Path] = (),
) -> dict[str, Any]:
    return {
        "stage": stage,
        "subject_id": subject_id,
        "relative_path": relative_path,
        "error_type": type(error).__name__ if isinstance(error, BaseException) else "Error",
        "error": _safe_error_text(error, roots),
    }


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _read_checksum_manifest(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Checksum manifest not found: {path.name}")
    suffix = path.suffix.lower()
    result: dict[str, str] = {}
    if suffix == ".json":
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(document, Mapping):
            for name, value in document.items():
                if isinstance(value, Mapping):
                    digest = value.get("checksum") or value.get("sha256") or value.get("md5")
                else:
                    digest = value
                if digest:
                    result[Path(str(name)).name] = str(digest).strip().lower()
        elif isinstance(document, list):
            for item in document:
                if not isinstance(item, Mapping):
                    continue
                name = item.get("filename") or item.get("file")
                digest = item.get("checksum") or item.get("sha256") or item.get("md5")
                if name and digest:
                    result[Path(str(name)).name] = str(digest).strip().lower()
        return result
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for item in csv.DictReader(handle):
                name = item.get("filename") or item.get("file") or item.get("name")
                digest = item.get("checksum") or item.get("sha256") or item.get("md5")
                if name and digest:
                    result[Path(name).name] = digest.strip().lower()
        return result
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().replace(" *", "  ").split()
        if len(parts) >= 2:
            result[Path(parts[-1]).name] = parts[0].lower()
    return result


def discover_checksum_manifest(archives_dir: Path) -> Path | None:
    """Return a recognized checksum manifest without guessing a digest."""

    candidates = [
        "checksums.json",
        "checksums.csv",
        "checksums.txt",
        "sha256sums.txt",
        "md5sums.txt",
    ]
    local = next(
        (archives_dir / name for name in candidates if (archives_dir / name).is_file()),
        None,
    )
    if local is not None:
        return local
    zenodo_manifest = archives_dir.parent / "metadata" / "zenodo_archive_checksums.json"
    return zenodo_manifest if zenodo_manifest.is_file() else None


def _hash_file(path: Path, expected: str) -> str:
    algorithms = {32: "md5", 40: "sha1", 64: "sha256"}
    try:
        algorithm = algorithms[len(expected)]
    except KeyError as error:
        raise ValueError(
            "Expected checksum must be a 32-character MD5, "
            "40-character SHA-1, or 64-character SHA-256 digest"
        ) from error
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_partial_zip(path: Path) -> bool:
    if path.stat().st_size == 0:
        return True
    with path.open("rb") as handle:
        prefix = handle.read(4)
        try:
            handle.seek(max(0, path.stat().st_size - 65557))
            tail = handle.read()
        except OSError:
            return False
    return prefix.startswith(b"PK") and b"PK\x05\x06" not in tail


def _empty_archive_row(filename: str, subject_id: str, status: str) -> dict[str, Any]:
    return {
        "filename": filename,
        "subject_id": subject_id,
        "size_bytes": 0,
        "zip_readable": "false",
        "zip_test_passed": "false",
        "member_count": 0,
        "compressed_size": 0,
        "uncompressed_size": 0,
        "checksum_expected": "not_available",
        "checksum_actual": "not_computed",
        "checksum_match": "not_available",
        "status": status,
        "error": "",
    }


def verify_archive(
    path: Path,
    *,
    checksum_expected: str | None = None,
    test_contents: bool = True,
) -> dict[str, Any]:
    """Verify one ZIP while leaving the archive byte-for-byte unchanged."""

    subject_id = _subject_from_name(path.name)
    row = _empty_archive_row(path.name, subject_id, "corrupt")
    row["size_bytes"] = path.stat().st_size
    expected = checksum_expected.strip().lower() if checksum_expected else ""
    row["checksum_expected"] = expected or "not_available"
    if expected:
        actual = _hash_file(path, expected)
        row["checksum_actual"] = actual
        row["checksum_match"] = _bool_cell(actual == expected)
        if actual != expected:
            row["status"] = "checksum_mismatch"
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            row["zip_readable"] = "true"
            row["member_count"] = len(infos)
            row["compressed_size"] = sum(info.compress_size for info in infos)
            row["uncompressed_size"] = sum(info.file_size for info in infos)
            failing_member = archive.testzip() if test_contents else None
            if failing_member:
                row["zip_test_passed"] = "false"
                row["status"] = "corrupt"
                row["error"] = f"CRC failure in member: {failing_member}"
            else:
                row["zip_test_passed"] = "true" if test_contents else "not_run"
                if row["status"] != "checksum_mismatch":
                    row["status"] = "valid"
    except (zipfile.BadZipFile, EOFError, OSError, RuntimeError) as error:
        row["status"] = "partial" if _looks_partial_zip(path) else "corrupt"
        row["error"] = _safe_error_text(error, [path.parent])
    return row


def _archive_candidates(archives_dir: Path) -> list[Path]:
    if not archives_dir.is_dir():
        return []
    return sorted(
        (
            path
            for path in archives_dir.iterdir()
            if path.is_file()
            and (
                path.suffix.lower() == ".zip"
                or path.name.lower().endswith((".zip.part", ".zip.partial", ".crdownload"))
            )
        ),
        key=lambda item: item.name.lower(),
    )


def verify_archives(
    archives_dir: Path,
    *,
    expected_subjects: Sequence[str] = EXPECTED_SUBJECTS,
    checksum_manifest: Path | None = None,
    test_contents: bool = True,
    include_unselected_archives: bool = True,
) -> list[dict[str, Any]]:
    """Verify expected and unexpected local archives deterministically."""

    archives_dir = Path(archives_dir)
    expected = tuple(_clean_subject(item) for item in expected_subjects)
    if checksum_manifest is None:
        checksum_manifest = discover_checksum_manifest(archives_dir)
    checksums = _read_checksum_manifest(checksum_manifest)
    candidates = _archive_candidates(archives_dir)
    by_subject: dict[str, list[Path]] = defaultdict(list)
    no_subject: list[Path] = []
    for path in candidates:
        subject = _subject_from_name(path.name)
        if not include_unselected_archives and subject not in expected:
            continue
        (by_subject[subject] if subject else no_subject).append(path)

    rows: list[dict[str, Any]] = []
    consumed: set[Path] = set()
    for subject in expected:
        matches = by_subject.get(subject, [])
        canonical_name = f"{subject}.zip"
        canonical = next((path for path in matches if path.name.lower() == canonical_name), None)
        if canonical is None:
            rows.append(_empty_archive_row(canonical_name, subject, "missing"))
        else:
            row = verify_archive(
                canonical,
                checksum_expected=checksums.get(canonical.name),
                test_contents=test_contents,
            )
            rows.append(row)
            consumed.add(canonical)
        if len(matches) > 1:
            for row in rows:
                if row["subject_id"] == subject:
                    row["status"] = "duplicate_subject"
                    row["error"] = "Multiple archive files map to this subject"
            for duplicate in matches:
                if duplicate in consumed:
                    continue
                duplicate_row = verify_archive(
                    duplicate,
                    checksum_expected=checksums.get(duplicate.name),
                    test_contents=test_contents,
                )
                duplicate_row["status"] = "duplicate_subject"
                duplicate_row["error"] = "Multiple archive files map to this subject"
                rows.append(duplicate_row)
                consumed.add(duplicate)

    for path in candidates:
        if path in consumed:
            continue
        if (
            not include_unselected_archives
            and _subject_from_name(path.name) not in expected
        ):
            continue
        row = verify_archive(
            path,
            checksum_expected=checksums.get(path.name),
            test_contents=test_contents,
        )
        row["status"] = "unexpected_name"
        row["error"] = "Archive name is outside the expected subject archive set"
        rows.append(row)
    return sorted(rows, key=lambda row: (row["subject_id"], row["filename"].lower()))


def _safe_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    windows = PureWindowsPath(name)
    if (
        not normalized
        or pure.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise UnsafeArchiveError(f"Unsafe ZIP member path: {name!r}")
    return pure


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _destination_for_member(
    extract_dir: Path,
    subject_id: str,
    member_path: PurePosixPath,
) -> Path:
    parts = member_path.parts
    relative = Path(*parts) if parts and parts[0].lower() == subject_id.lower() else Path(subject_id, *parts)
    destination = extract_dir / relative
    try:
        destination.resolve().relative_to(extract_dir.resolve())
    except (OSError, ValueError) as error:
        raise UnsafeArchiveError(f"ZIP member escapes extraction root: {member_path}") from error
    return destination


def _crc32_file(path: Path) -> int:
    value = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value = binascii.crc32(chunk, value)
    return value & 0xFFFFFFFF


def safe_extract_archive(
    archive_path: Path,
    extract_dir: Path,
    *,
    subject_id: str | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Safely extract one verified archive using member-level atomic writes."""

    archive_path = Path(archive_path)
    extract_dir = Path(extract_dir)
    subject_id = subject_id or _subject_from_name(archive_path.name)
    if not subject_id:
        raise UnsafeArchiveError("Cannot extract archive without a subject identifier")
    marker = extract_dir / f".{subject_id}.extracting.json"
    if marker.exists() and not resume:
        raise ExtractionConflictError(
            f"Incomplete extraction marker exists for {subject_id}; use --resume"
        )

    with zipfile.ZipFile(archive_path, "r") as archive:
        planned: list[tuple[zipfile.ZipInfo, PurePosixPath, Path]] = []
        destinations: set[str] = set()
        for info in archive.infolist():
            member_path = _safe_member_path(info.filename)
            if info.flag_bits & 0x1:
                raise UnsafeArchiveError(f"Encrypted ZIP member is not supported: {info.filename}")
            if _is_symlink(info):
                raise UnsafeArchiveError(f"Symbolic link ZIP member is not allowed: {info.filename}")
            destination = _destination_for_member(extract_dir, subject_id, member_path)
            destination_key = str(destination.resolve()).lower()
            if destination_key in destinations:
                raise UnsafeArchiveError(f"Duplicate ZIP member destination: {info.filename}")
            destinations.add(destination_key)
            planned.append((info, member_path, destination))

        existing_correct: dict[Path, bool] = {}
        for info, _, destination in planned:
            if info.is_dir() or not destination.exists():
                continue
            correct = (
                destination.is_file()
                and destination.stat().st_size == info.file_size
                and _crc32_file(destination) == info.CRC
            )
            existing_correct[destination] = correct
            if not correct and not overwrite:
                relative = destination.relative_to(extract_dir).as_posix()
                raise ExtractionConflictError(
                    f"Existing file differs from archive member: {relative}"
                )

        extract_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "archive": archive_path.name,
                    "subject_id": subject_id,
                    "status": "incomplete",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        rows: list[dict[str, Any]] = []
        for info, member_path, destination in planned:
            relative = destination.relative_to(extract_dir).as_posix()
            base = {
                "filename": archive_path.name,
                "subject_id": subject_id,
                "member": member_path.as_posix(),
                "relative_path": relative,
                "size_bytes": info.file_size,
                "crc32": f"{info.CRC:08x}",
                "status": "",
                "error": "",
            }
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                rows.append({**base, "status": "directory"})
                continue
            existed_before = destination.exists()
            if existing_correct.get(destination, False):
                rows.append({**base, "status": "already_correct"})
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".cog_bci.part")
            with archive.open(info, "r") as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            if temporary.stat().st_size != info.file_size or _crc32_file(temporary) != info.CRC:
                raise InventoryError(f"Extracted member failed integrity check: {relative}")
            temporary.replace(destination)
            rows.append(
                {
                    **base,
                    "status": "overwritten" if existed_before else "extracted",
                }
            )
    marker.unlink(missing_ok=True)
    return rows


def extract_verified_archives(
    archives_dir: Path,
    extract_dir: Path,
    archive_rows: Sequence[Mapping[str, Any]],
    *,
    resume: bool = False,
    overwrite: bool = False,
    progress_callback: Callable[
        [Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]], None
    ]
    | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Extract only rows whose verification status is ``valid``."""

    extraction_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    fatal = False
    for row in archive_rows:
        if row.get("status") != "valid" or row.get("zip_test_passed") != "true":
            continue
        archive_path = Path(archives_dir) / str(row["filename"])
        before_size = archive_path.stat().st_size
        try:
            archive_member_rows = safe_extract_archive(
                archive_path,
                Path(extract_dir),
                subject_id=str(row["subject_id"]),
                resume=resume,
                overwrite=overwrite,
            )
            extraction_rows.extend(archive_member_rows)
            statuses = {str(item["status"]) for item in archive_member_rows}
            archive_status = (
                "extracted"
                if statuses & {"extracted", "overwritten"}
                else "already_complete"
            )
            extraction_rows.append(
                {
                    "filename": archive_path.name,
                    "subject_id": str(row["subject_id"]),
                    "member": "__archive__",
                    "relative_path": "",
                    "size_bytes": sum(
                        int(item.get("size_bytes", 0))
                        for item in archive_member_rows
                        if item.get("status") != "directory"
                    ),
                    "crc32": "",
                    "status": archive_status,
                    "error": "",
                }
            )
        except (InventoryError, OSError, zipfile.BadZipFile) as error:
            fatal = True
            status = "conflict" if isinstance(error, ExtractionConflictError) else "failed"
            extraction_rows.append(
                {
                    "filename": archive_path.name,
                    "subject_id": str(row.get("subject_id", "")),
                    "member": "__archive__",
                    "relative_path": "",
                    "size_bytes": 0,
                    "crc32": "",
                    "status": status,
                    "error": _safe_error_text(error),
                }
            )
            errors.append(
                _error_row(
                    "extract",
                    error,
                    subject_id=str(row.get("subject_id", "")),
                    relative_path=str(row.get("filename", "")),
                    roots=[Path(archives_dir), Path(extract_dir)],
                )
            )
        if not archive_path.exists() or archive_path.stat().st_size != before_size:
            raise InventoryError(f"Source archive was modified: {archive_path.name}")
        if progress_callback is not None:
            progress_callback(extraction_rows, errors)
    return (
        sorted(extraction_rows, key=lambda row: (row["subject_id"], row["relative_path"])),
        errors,
        fatal,
    )


def _parse_identity(relative_path: str) -> tuple[str, str, str, str]:
    pure = PurePosixPath(relative_path)
    joined = "/" + pure.as_posix()
    subject = _subject_from_name(joined)
    session_match = SESSION_RE.search(joined)
    session = f"ses-{session_match.group(1)}" if session_match else ""
    stem = pure.stem
    task_match = TASK_RE.search(stem)
    task = task_match.group(1) if task_match else stem
    run_match = RUN_RE.search(stem)
    run = run_match.group(1) if run_match else ""
    return subject, session, task, run


def _is_service_file(path: Path, relative: str) -> bool:
    return (
        path.name.lower() in SERVICE_NAMES
        or any(part.lower() in SERVICE_NAMES for part in PurePosixPath(relative).parts)
        or path.name.startswith("._")
        or path.name.endswith(".cog_bci.part")
    )


def _read_eeglab_metadata(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read EEGLAB headers without preloading signal samples."""

    if importlib.util.find_spec("mne") is None:
        raise ModuleNotFoundError(
            "Optional dependency 'mne' is unavailable; EEG header was not decoded"
        )
    import mne  # type: ignore

    raw = mne.io.read_raw_eeglab(str(path), preload=False, verbose="ERROR")
    try:
        annotations = getattr(raw, "annotations", None)
        descriptions = (
            []
            if annotations is None
            else [str(item) for item in getattr(annotations, "description", ())]
        )
        units_map = getattr(raw, "_orig_units", {}) or {}
        units = sorted({str(value) for value in units_map.values() if value})
        reference = raw.info.get("custom_ref_applied", "")
        channel_names = [str(item) for item in raw.ch_names]
        lowered_names = [name.casefold() for name in channel_names]
        duplicate_names = sorted(
            name for name, count in Counter(lowered_names).items() if count > 1
        )
        channels_info = raw.info.get("chs", ())
        missing_locations = 0
        for channel in channels_info:
            location = channel.get("loc")
            if location is None or not any(
                math.isfinite(float(value)) and float(value) != 0.0
                for value in list(location)[:3]
            ):
                missing_locations += 1
        warnings: list[str] = []
        if duplicate_names:
            warnings.append(f"duplicate_channel_names={','.join(duplicate_names)}")
        if missing_locations:
            warnings.append(f"missing_channel_locations={missing_locations}")
        metadata = {
            "n_channels": int(raw.info["nchan"]),
            "channel_count": int(raw.info["nchan"]),  # in-memory legacy alias
            "channel_names": "|".join(channel_names),
            "sampling_rate_hz": float(raw.info["sfreq"]),
            "n_samples": int(raw.n_times),
            "duration_seconds": float(raw.n_times / raw.info["sfreq"]),
            "data_units": "|".join(units),
            "event_count": len(descriptions),
            "event_types": "|".join(sorted(set(descriptions))),
            "annotation_count": 0 if annotations is None else len(annotations),
            "annotations_count": 0 if annotations is None else len(annotations),
            "reference": str(reference),
            "bad_channels": "|".join(map(str, raw.info.get("bads", []))),
            "reader_used": "mne.io.read_raw_eeglab(preload=False)",
            "metadata_reader": "mne.io.read_raw_eeglab(preload=False)",
            "read_status": "ok_with_warnings" if warnings else "ok",
            "error": "; ".join(warnings),
        }
        events = [
            {
                "event_source": "eeglab_annotation",
                "event_label": description,
                "trigger_code": "",
                "rating_name": "",
                "behavioural_outcome": "",
            }
            for description in descriptions
        ]
        return metadata, events
    finally:
        close = getattr(raw, "close", None)
        if callable(close):
            close()


def _read_eeglab_fallback_metadata(path: Path) -> dict[str, Any]:
    """Read only MATLAB variable headers as a safe fallback."""

    from scipy.io import whosmat

    inspected = whosmat(path)
    variables = (
        "|".join(sorted(name for name, _, _ in inspected))
        if inspected is not None
        else ""
    )
    return {
        "n_channels": "",
        "channel_count": "",
        "channel_names": "",
        "sampling_rate_hz": "",
        "n_samples": "",
        "duration_seconds": "",
        "data_units": "",
        "event_count": "",
        "event_types": "",
        "annotation_count": "",
        "annotations_count": "",
        "reference": "",
        "bad_channels": "",
        "reader_used": f"scipy.io.whosmat:variables={variables}",
        "metadata_reader": f"scipy.io.whosmat:variables={variables}",
        "read_status": "fallback_metadata_only",
        "error": "",
    }


def _tabular_events(path: Path) -> list[dict[str, str]]:
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Metadata table exceeds the 10 MiB inventory safety limit")
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for item in reader:
            event_label = next(
                (
                    str(item[key])
                    for key in ("trial_type", "event_type", "type", "event")
                    if key in item and item[key] not in (None, "")
                ),
                "",
            )
            trigger = next(
                (
                    str(item[key])
                    for key in ("value", "trigger", "code")
                    if key in item and item[key] not in (None, "")
                ),
                "",
            )
            rating_keys = [key for key in item if "rating" in key.lower()]
            outcome_keys = [
                key
                for key in item
                if any(token in key.lower() for token in ("outcome", "accuracy", "response", "score"))
            ]
            rows.append(
                {
                    "event_source": "tsv",
                    "event_label": event_label,
                    "trigger_code": trigger,
                    "rating_name": "|".join(sorted(rating_keys)),
                    "behavioural_outcome": "|".join(sorted(outcome_keys)),
                }
            )
    return rows


def _json_events(path: Path) -> list[dict[str, str]]:
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("JSON metadata exceeds the 10 MiB inventory safety limit")
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    rows: list[dict[str, str]] = []

    def visit(value: Any, prefix: str = "") -> None:
        if len(rows) >= 10_000:
            raise ValueError("JSON metadata exceeds the 10,000-item inventory limit")
        if isinstance(value, Mapping):
            for key in sorted(value, key=str):
                child = value[key]
                label = f"{prefix}.{key}" if prefix else str(key)
                lowered = str(key).lower()
                if child is not None and not isinstance(child, (Mapping, list)) and any(
                    token in lowered
                    for token in ("event", "annotation", "trigger", "rating", "score", "outcome", "response")
                ):
                    rows.append(
                        {
                            "event_source": "json_metadata",
                            "event_label": label if "event" in lowered or "annotation" in lowered else "",
                            "trigger_code": str(child) if "trigger" in lowered else "",
                            "rating_name": label if "rating" in lowered else "",
                            "behavioural_outcome": (
                                label
                                if any(token in lowered for token in ("score", "outcome", "response"))
                                else ""
                            ),
                        }
                    )
                visit(child, label)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{prefix}[{index}]")

    visit(document)
    return rows


def _mat_variables(path: Path) -> list[dict[str, str]]:
    """Inspect MATLAB variable headers without loading variable payloads."""

    from scipy.io import loadmat, whosmat

    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("MATLAB metadata exceeds the 10 MiB inventory safety limit")
    try:
        inspected = whosmat(path)
    except TypeError as error:
        # MATLAB table/opaque objects can expose a variable header whose
        # internal ``dims`` field is None. SciPy then raises while formatting
        # the header. Loading these small behavioural files is a bounded,
        # explicit fallback and does not touch the paired EEG payload.
        if "NoneType" not in str(error):
            raise
        document = loadmat(path, squeeze_me=False, struct_as_record=False)
        variables = [
            (str(name), getattr(value, "shape", ()), type(value).__name__)
            for name, value in document.items()
            if not str(name).startswith("__")
        ]
        return [
            {
                "event_source": "mat_opaque_fallback",
                "event_label": f"{name}:{kind}:{tuple(shape)}",
                "trigger_code": "",
                "rating_name": "",
                "behavioural_outcome": "",
            }
            for name, shape, kind in variables
        ]
    if inspected is None:
        return []
    rows: list[dict[str, str]] = []
    for name, shape, matlab_class in inspected:
        lowered = name.lower()
        rows.append(
            {
                "event_source": "mat_variable",
                "event_label": f"{name}:{matlab_class}:{tuple(shape)}",
                "trigger_code": "",
                "rating_name": name if "rating" in lowered else "",
                "behavioural_outcome": (
                    name
                    if any(token in lowered for token in ("score", "outcome", "response", "result"))
                    else ""
                ),
            }
        )
    return rows


def inventory_extracted(
    extract_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Inventory an extracted tree without loading raw EEG samples."""

    extract_dir = Path(extract_dir)
    files = (
        sorted((path for path in extract_dir.rglob("*") if path.is_file()), key=lambda p: p.as_posix().lower())
        if extract_dir.is_dir()
        else []
    )
    set_by_key: dict[tuple[str, str], str] = {}
    fdt_by_key: dict[tuple[str, str], str] = {}
    for path in files:
        relative = path.relative_to(extract_dir).as_posix()
        key = (str(PurePosixPath(relative).parent).lower(), path.stem.lower())
        if path.suffix.lower() == ".set":
            set_by_key[key] = relative
        elif path.suffix.lower() == ".fdt":
            fdt_by_key[key] = relative

    records: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(extract_dir).as_posix()
        if _is_service_file(path, relative):
            errors.append(
                _error_row(
                    "inventory",
                    "Service file ignored",
                    relative_path=relative,
                    roots=[extract_dir],
                )
            )
            continue
        subject, session, task, run = _parse_identity(relative)
        suffix = path.suffix.lower()
        key = (str(PurePosixPath(relative).parent).lower(), path.stem.lower())
        file_type = suffix.lstrip(".") if suffix in KNOWN_FILE_TYPES else "unknown"
        record = {
            "subject_id": subject,
            "session_id": session,
            "task": task,
            "run": run,
            "file_type": file_type,
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
            "paired_set_file": set_by_key.get(key, "") if suffix == ".fdt" else "",
            "paired_fdt_file": fdt_by_key.get(key, "") if suffix == ".set" else "",
        }
        records.append(record)
        if not subject:
            errors.append(
                _error_row(
                    "inventory",
                    "Participant identifier could not be derived from path",
                    relative_path=relative,
                    roots=[extract_dir],
                )
            )
        if file_type == "unknown":
            errors.append(
                _error_row(
                    "inventory",
                    "Unknown file type",
                    subject_id=subject,
                    relative_path=relative,
                    roots=[extract_dir],
                )
            )
        if suffix == ".set" and not record["paired_fdt_file"]:
            errors.append(
                _error_row(
                    "inventory",
                    "EEGLAB .set has no matching .fdt",
                    subject_id=subject,
                    relative_path=relative,
                    roots=[extract_dir],
                )
            )
        if suffix == ".fdt" and not record["paired_set_file"]:
            errors.append(
                _error_row(
                    "inventory",
                    "EEGLAB .fdt has no matching .set",
                    subject_id=subject,
                    relative_path=relative,
                    roots=[extract_dir],
                )
            )
        if suffix == ".set":
            identity = {
                "subject_id": subject,
                "session_id": session,
                "task": task,
                "run": run,
                "relative_path": relative,
                "paired_fdt_file": str(record["paired_fdt_file"]),
            }
            try:
                metadata, annotation_rows = _read_eeglab_metadata(path)
                channels.append({**identity, **metadata})
                events.extend({**identity, **item} for item in annotation_rows)
            except Exception as error:  # per-record failure must not abort inventory
                try:
                    fallback = _read_eeglab_fallback_metadata(path)
                except Exception:
                    fallback = {
                        "n_channels": "",
                        "channel_count": "",
                        "channel_names": "",
                        "sampling_rate_hz": "",
                        "n_samples": "",
                        "duration_seconds": "",
                        "data_units": "",
                        "event_count": "",
                        "event_types": "",
                        "annotation_count": "",
                        "annotations_count": "",
                        "reference": "",
                        "bad_channels": "",
                        "reader_used": "unavailable",
                        "metadata_reader": "unavailable",
                        "read_status": "error",
                        "error": _safe_error_text(error),
                    }
                else:
                    fallback["read_status"] = "fallback_metadata_only"
                    fallback["error"] = _safe_error_text(error)
                channels.append(
                    {
                        **identity,
                        **fallback,
                    }
                )
                errors.append(
                    _error_row(
                        "eeglab",
                        error,
                        subject_id=subject,
                        relative_path=relative,
                        roots=[extract_dir],
                    )
                )
        elif suffix in {".tsv", ".json", ".mat"}:
            try:
                metadata_events = (
                    _tabular_events(path)
                    if suffix == ".tsv"
                    else (_json_events(path) if suffix == ".json" else _mat_variables(path))
                )
                events.extend(
                    {
                        "subject_id": subject,
                        "session_id": session,
                        "task": task,
                        "run": run,
                        "relative_path": relative,
                        **item,
                    }
                    for item in metadata_events
                )
            except (
                OSError,
                UnicodeError,
                csv.Error,
                ValueError,
                json.JSONDecodeError,
                ImportError,
            ) as error:
                errors.append(
                    _error_row(
                        "metadata",
                        error,
                        subject_id=subject,
                        relative_path=relative,
                        roots=[extract_dir],
                    )
                )

    combinations: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for record in records:
        if record["file_type"] == "set":
            combinations[
                (
                    str(record["subject_id"]),
                    str(record["session_id"]),
                    str(record["task"]).lower(),
                    str(record["run"]),
                )
            ].append(str(record["relative_path"]))
    for combination, paths in sorted(combinations.items()):
        if len(paths) > 1:
            errors.append(
                _error_row(
                    "inventory",
                    f"Duplicate subject/session/task/run combination: {len(paths)} records",
                    subject_id=combination[0],
                    relative_path="|".join(sorted(paths)),
                    roots=[extract_dir],
                )
            )
    key = lambda row: (
        str(row.get("subject_id", "")),
        str(row.get("session_id", "")),
        str(row.get("task", "")).lower(),
        str(row.get("run", "")),
        str(row.get("relative_path", "")).lower(),
    )
    return (
        sorted(records, key=key),
        sorted(channels, key=key),
        sorted(events, key=key),
        sorted(errors, key=lambda row: tuple(str(row[column]) for column in ERROR_COLUMNS)),
    )


def build_file_pair_inventory(
    record_rows: Sequence[Mapping[str, Any]],
    channel_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build one explicit row per EEGLAB set/fdt stem."""

    grouped: dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: {"set": [], "fdt": []}
    )
    channel_by_path = {
        str(row.get("relative_path", "")): row for row in channel_rows
    }
    for row in record_rows:
        file_type = str(row.get("file_type", ""))
        if file_type not in {"set", "fdt"}:
            continue
        relative = str(row["relative_path"])
        pure = PurePosixPath(relative)
        grouped[(str(pure.parent).casefold(), pure.stem.casefold())][file_type].append(row)

    result: list[dict[str, Any]] = []
    for key in sorted(grouped):
        sets = grouped[key]["set"]
        fdts = grouped[key]["fdt"]
        exemplars = sets or fdts
        identity = exemplars[0]
        set_row = sets[0] if len(sets) == 1 else None
        fdt_row = fdts[0] if len(fdts) == 1 else None
        errors: list[str] = []
        if len(sets) > 1:
            errors.append(f"duplicate_set={len(sets)}")
        if len(fdts) > 1:
            errors.append(f"duplicate_fdt={len(fdts)}")
        if not sets:
            errors.append("missing_set")
        if not fdts:
            errors.append("missing_fdt")
        set_size = int(set_row["size_bytes"]) if set_row else 0
        fdt_size = int(fdt_row["size_bytes"]) if fdt_row else 0
        if set_row and set_size == 0:
            errors.append("empty_set")
        if fdt_row and fdt_size == 0:
            errors.append("empty_fdt")
        expected_size: int | str = ""
        size_match: str = "not_available"
        if set_row and fdt_row:
            channel = channel_by_path.get(str(set_row["relative_path"]), {})
            try:
                expected_size = int(channel["n_channels"]) * int(channel["n_samples"]) * 4
            except (KeyError, TypeError, ValueError):
                expected_size = ""
            if expected_size != "":
                size_match = _bool_cell(fdt_size == expected_size)
                if size_match == "false":
                    errors.append("unexpected_fdt_size")
        status = (
            "duplicate"
            if len(sets) > 1 or len(fdts) > 1
            else ("missing" if not sets or not fdts else ("invalid" if errors else "paired"))
        )
        result.append(
            {
                "subject_id": str(identity.get("subject_id", "")),
                "session_id": str(identity.get("session_id", "")),
                "task": str(identity.get("task", "")),
                "run": str(identity.get("run", "")),
                "set_path": str(set_row["relative_path"]) if set_row else "",
                "fdt_path": str(fdt_row["relative_path"]) if fdt_row else "",
                "set_size_bytes": set_size if set_row else "",
                "fdt_size_bytes": fdt_size if fdt_row else "",
                "expected_fdt_size_bytes": expected_size,
                "size_match": size_match,
                "status": status,
                "error": "; ".join(errors),
            }
        )
    return result


def _behavioural_row(
    *,
    relative_path: str,
    variable_name: str,
    values: Sequence[Any],
    data_type: str,
    read_status: str = "ok",
    error: str = "",
) -> dict[str, Any]:
    subject, session, task, _ = _parse_identity(relative_path)
    finite = _finite_numbers(values)
    return {
        "subject_id": subject,
        "session_id": session,
        "task": task,
        "relative_path": relative_path,
        "source_file": PurePosixPath(relative_path).name,
        "variable_name": variable_name,
        "metric_name": variable_name,
        "data_type": data_type,
        "value_count": len(finite),
        "missing_count": max(0, len(values) - len(finite)),
        "minimum": min(finite) if finite else "",
        "maximum": max(finite) if finite else "",
        "mean": sum(finite) / len(finite) if finite else "",
        "source_level": "task" if task else ("session" if session else "dataset"),
        "read_status": read_status,
        "error": error,
    }


def _mat_behavioural_rows(path: Path, relative_path: str) -> list[dict[str, Any]]:
    """Summarize small behavioural MAT payloads without touching EEG arrays."""

    import numpy as np
    from scipy.io import loadmat

    if path.stat().st_size > 10 * 1024 * 1024:
        return [
            _behavioural_row(
                relative_path=relative_path,
                variable_name="",
                values=[],
                data_type="not_loaded",
                read_status="not_available",
                error="MATLAB file exceeds the 10 MiB inventory safety limit",
            )
        ]
    document = loadmat(path, squeeze_me=True, struct_as_record=False)
    rows: list[dict[str, Any]] = []

    def visit(name: str, value: Any, depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.number):
                flattened = value.reshape(-1).tolist()
                rows.append(
                    _behavioural_row(
                        relative_path=relative_path,
                        variable_name=name,
                        values=flattened,
                        data_type=str(value.dtype),
                    )
                )
            elif value.dtype == object:
                for index, child in enumerate(value.reshape(-1)):
                    visit(f"{name}[{index}]", child, depth + 1)
            return
        field_names = getattr(value, "_fieldnames", None)
        if field_names is not None:
            for field_name in field_names:
                visit(
                    f"{name}.{field_name}",
                    getattr(value, field_name, None),
                    depth + 1,
                )
            return
        if isinstance(value, (int, float, np.integer, np.floating)):
            rows.append(
                _behavioural_row(
                    relative_path=relative_path,
                    variable_name=name,
                    values=[value],
                    data_type=type(value).__name__,
                )
            )
            return
        if type(value).__name__ == "MatlabOpaque":
            rows.append(
                _behavioural_row(
                    relative_path=relative_path,
                    variable_name=name,
                    values=[],
                    data_type="MatlabOpaque",
                    read_status="not_available",
                    error="MATLAB opaque/table payload is not semantically decodable by SciPy",
                )
            )

    for name, value in sorted(document.items()):
        if not str(name).startswith("__"):
            visit(str(name), value)
    if not rows:
        rows.append(
            _behavioural_row(
                relative_path=relative_path,
                variable_name="",
                values=[],
                data_type="empty",
                read_status="not_available",
                error="No supported behavioural variables found",
            )
        )
    return rows


def _text_behavioural_rows(path: Path, relative_path: str) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        reader = csv.DictReader(text.splitlines(), dialect=dialect)
        rows = list(reader)
    except csv.Error:
        rows = []
    result: list[dict[str, Any]] = []
    expected_value_count: int | None = None
    if rows:
        field_names = [str(column) for column in (reader.fieldnames or [])]
        subject_column = next(
            (column for column in field_names if column.casefold() in {"sbj", "subject"}),
            None,
        )
        session_column = next(
            (column for column in field_names if column.casefold() in {"sess", "session"}),
            None,
        )
        condition_columns = [
            column for column in field_names if column.casefold() == "condition"
        ]
        numeric_condition_column = next(
            (
                column
                for column in condition_columns
                if _finite_numbers(row.get(column) for row in rows)
            ),
            None,
        )
        if subject_column and session_column and numeric_condition_column:
            expected_value_count = (
                len({row.get(subject_column) for row in rows})
                * len({row.get(session_column) for row in rows})
                * len({row.get(numeric_condition_column) for row in rows})
            )
        for column in reader.fieldnames or []:
            values = [row.get(column) for row in rows]
            numeric = _finite_numbers(values)
            if numeric:
                item = _behavioural_row(
                    relative_path=relative_path,
                    variable_name=column,
                    values=values,
                    data_type="numeric_text",
                )
                if expected_value_count is not None:
                    item["missing_count"] = max(
                        0, expected_value_count - int(item["value_count"])
                    )
                item["source_level"] = (
                    "session_condition"
                    if path.name.casefold() == "kss.txt"
                    else "task"
                )
                result.append(item)
            else:
                present = [str(value) for value in values if value not in (None, "")]
                if present:
                    item = _behavioural_row(
                        relative_path=relative_path,
                        variable_name=column,
                        values=[],
                        data_type=(
                            "categorical_text:"
                            + "|".join(sorted(set(present), key=str.casefold))
                        ),
                    )
                    item["value_count"] = len(present)
                    item["missing_count"] = (
                        max(0, expected_value_count - len(present))
                        if expected_value_count is not None
                        else len(values) - len(present)
                    )
                    item["source_level"] = (
                        "session_condition"
                        if path.name.casefold() == "kss.txt"
                        else "task"
                    )
                    result.append(item)
    if not result:
        result.append(
            _behavioural_row(
                relative_path=relative_path,
                variable_name="",
                values=[],
                data_type="text",
                read_status="metadata_only",
                error="No numeric tabular fields decoded",
            )
        )
    return result


def inventory_behavioural(
    extract_dir: Path,
    metadata_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Inventory behavioural MAT and Zenodo metadata independently of EEG."""

    candidates = sorted(Path(extract_dir).rglob("*.mat"), key=lambda item: item.as_posix().lower())
    roots = [Path(extract_dir)]
    if metadata_dir is not None and Path(metadata_dir).is_dir():
        roots.append(Path(metadata_dir))
        candidates.extend(
            sorted(
                (
                    path
                    for path in Path(metadata_dir).iterdir()
                    if path.is_file() and path.suffix.lower() in {".mat", ".txt"}
                ),
                key=lambda item: item.name.lower(),
            )
        )
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for path in candidates:
        root = next((item for item in roots if path.is_relative_to(item)), roots[0])
        relative = (
            path.relative_to(root).as_posix()
            if root == Path(extract_dir)
            else f"metadata/{path.name}"
        )
        try:
            if path.suffix.lower() == ".mat":
                decoded = _mat_behavioural_rows(path, relative)
            elif path.name.casefold() in {"kss.txt", "rsme.txt"}:
                decoded = _text_behavioural_rows(path, relative)
            else:
                continue
            rows.extend(decoded)
            for item in decoded:
                if item["read_status"] == "not_available":
                    errors.append(
                        _error_row(
                            "behavioural",
                            str(item["error"]),
                            subject_id=str(item["subject_id"]),
                            relative_path=relative,
                            roots=roots,
                        )
                    )
        except (OSError, UnicodeError, ValueError, TypeError, ImportError) as error:
            errors.append(
                _error_row(
                    "behavioural",
                    error,
                    relative_path=relative,
                    roots=roots,
                )
            )
    key = lambda row: (
        str(row["subject_id"]),
        str(row["session_id"]),
        str(row["task"]).casefold(),
        str(row["relative_path"]).casefold(),
        str(row["variable_name"]).casefold(),
    )
    return sorted(rows, key=key), sorted(
        errors, key=lambda row: tuple(str(row[column]) for column in ERROR_COLUMNS)
    )


def build_structural_inventories(
    record_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sets = [row for row in record_rows if row.get("file_type") == "set"]
    tasks: list[dict[str, Any]] = []
    for task in sorted({str(row["task"]) for row in sets}, key=str.casefold):
        subset = [row for row in sets if str(row["task"]) == task]
        tasks.append(
            {
                "task": task,
                "record_count": len(subset),
                "subject_count": len({str(row["subject_id"]) for row in subset}),
                "session_count": len(
                    {(str(row["subject_id"]), str(row["session_id"])) for row in subset}
                ),
            }
        )
    sessions: list[dict[str, Any]] = []
    for subject, session in sorted(
        {(str(row["subject_id"]), str(row["session_id"])) for row in sets}
    ):
        subset = [
            row
            for row in sets
            if str(row["subject_id"]) == subject and str(row["session_id"]) == session
        ]
        sessions.append(
            {
                "subject_id": subject,
                "session_id": session,
                "record_count": len(subset),
                "task_count": len({str(row["task"]) for row in subset}),
            }
        )
    subjects: list[dict[str, Any]] = []
    for subject in sorted({str(row["subject_id"]) for row in sets}):
        subset = [row for row in sets if str(row["subject_id"]) == subject]
        subjects.append(
            {
                "subject_id": subject,
                "session_count": len({str(row["session_id"]) for row in subset}),
                "record_count": len(subset),
                "task_count": len({str(row["task"]) for row in subset}),
            }
        )
    return tasks, sessions, subjects


def _finite_numbers(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def build_summary(
    *,
    mode: str,
    archive_rows: Sequence[Mapping[str, Any]],
    extraction_rows: Sequence[Mapping[str, Any]],
    record_rows: Sequence[Mapping[str, Any]],
    channel_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    behavioural_rows: Sequence[Mapping[str, Any]],
    file_pair_rows: Sequence[Mapping[str, Any]],
    error_rows: Sequence[Mapping[str, Any]],
    expected_subjects: Sequence[str],
    checksum_manifest_available: bool,
    fatal: bool,
) -> dict[str, Any]:
    archive_counts = Counter(str(row["status"]) for row in archive_rows)
    expected = {_clean_subject(item) for item in expected_subjects}
    valid = {
        str(row["subject_id"])
        for row in archive_rows
        if row["status"] == "valid"
        and row.get("zip_test_passed") == "true"
        and row["subject_id"] in expected
    }
    complete = valid == expected and not fatal
    overall = "error" if fatal else ("complete" if complete else "incomplete")
    readable = sum(1 for row in archive_rows if row.get("zip_readable") == "true")
    layouts = sorted(
        {
            str(row["channel_names"])
            for row in channel_rows
            if str(row.get("channel_names", ""))
        }
    )
    sampling_rates = sorted(
        set(_finite_numbers(row.get("sampling_rate_hz") for row in channel_rows))
    )
    durations = _finite_numbers(row.get("duration_seconds") for row in channel_rows)
    set_records = [row for row in record_rows if row.get("file_type") == "set"]
    missing_pairs = sum(
        1 for row in file_pair_rows if row.get("status") != "paired"
    )
    extraction_archive_rows = [
        row for row in extraction_rows if row.get("member") == "__archive__"
    ]
    extraction_file_rows = [
        row
        for row in extraction_rows
        if row.get("member") != "__archive__" and row.get("status") != "directory"
    ]
    readable_channels = [
        row for row in channel_rows if str(row.get("read_status", "")).startswith("ok")
    ]
    channel_counts = sorted(
        set(_finite_numbers(row.get("n_channels") for row in readable_channels))
    )
    errors_by_stage = Counter(str(row.get("stage", "")) for row in error_rows)
    records_per_task = Counter(
        str(row.get("task", "")) for row in set_records if str(row.get("task", ""))
    )
    subjects = sorted(
        {str(row["subject_id"]) for row in record_rows if str(row.get("subject_id", ""))}
    )
    session_labels = sorted(
        {str(row["session_id"]) for row in record_rows if str(row.get("session_id", ""))}
    )
    tasks = sorted(
        {str(row["task"]) for row in set_records if str(row.get("task", ""))}
    )
    runs = sorted(
        {str(row["run"]) for row in record_rows if str(row.get("run", ""))}
    )
    participant_sessions = sorted(
        {
            (str(row["subject_id"]), str(row["session_id"]))
            for row in set_records
            if str(row.get("subject_id", "")) and str(row.get("session_id", ""))
        }
    )
    unique_record_keys = {
        (
            str(row.get("subject_id", "")),
            str(row.get("session_id", "")),
            str(row.get("task", "")),
            str(row.get("run", "")),
        )
        for row in set_records
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "result_status": "diagnostic",
        "overall_status": overall,
        "mode": mode,
        "expected_archive_count": len(expected),
        "readable_archive_count": readable,
        "valid_archive_count": len(valid),
        "crc_verified_archive_count": len(valid),
        "checksum_manifest": "available" if checksum_manifest_available else "not_available",
        "archive_status_counts": dict(sorted(archive_counts.items())),
        "extraction_status_counts": dict(
            sorted(Counter(str(row["status"]) for row in extraction_archive_rows).items())
        ),
        "extracted_archive_count": sum(
            1
            for row in extraction_archive_rows
            if row.get("status") in {"extracted", "already_complete"}
        ),
        "extracted_file_count": len(extraction_file_rows),
        "extracted_size_bytes": sum(int(row.get("size_bytes", 0)) for row in extraction_file_rows),
        "record_count": len(record_rows),
        "eeg_record_count": len(set_records),
        "readable_eeg_record_count": len(readable_channels),
        "event_count": len(event_rows),
        "behavioural_row_count": len(behavioural_rows),
        "error_count": len(error_rows),
        "errors_by_stage": dict(sorted(errors_by_stage.items())),
        "subject_count": len(subjects),
        "session_count": len(participant_sessions),
        "session_label_count": len(session_labels),
        "task_count": len(tasks),
        "run_count": len(runs),
        "subjects": subjects,
        "sessions": session_labels,
        "tasks": tasks,
        "runs": runs,
        "records_per_task": dict(sorted(records_per_task.items())),
        "unique_subject_session_task_run_count": len(unique_record_keys),
        "unique_channel_layouts": layouts,
        "unique_channel_counts": channel_counts,
        "unique_sampling_rates_hz": sampling_rates,
        "duration_seconds": {
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "median": median(durations) if durations else None,
        },
        "missing_set_fdt_pairs": missing_pairs,
        "file_pair_count": len(file_pair_rows),
        "complete_file_pair_count": sum(
            1 for row in file_pair_rows if row.get("status") == "paired"
        ),
    }


def render_runtime_report(summary: Mapping[str, Any]) -> str:
    """Render a deterministic, path-free diagnostic report."""

    lines = [
        "# COG-BCI structural inventory",
        "",
        f"- Result status: `{summary['result_status']}`",
        f"- Overall status: `{summary['overall_status']}`",
        f"- Mode: `{summary['mode']}`",
        f"- Expected archives: {summary['expected_archive_count']}",
        f"- ZIP-readable archives: {summary['readable_archive_count']}",
        f"- CRC-verified archives: {summary['crc_verified_archive_count']}",
        f"- Checksum manifest: `{summary['checksum_manifest']}`",
        f"- Records: {summary['record_count']}",
        f"- EEGLAB records: {summary['eeg_record_count']}",
        f"- Readable EEGLAB records: {summary['readable_eeg_record_count']}",
        f"- Complete `.set/.fdt` pairs: {summary['complete_file_pair_count']}",
        f"- Behavioural inventory rows: {summary['behavioural_row_count']}",
        f"- Inventory errors: {summary['error_count']}",
        "",
        "## Archive statuses",
        "",
    ]
    counts = summary.get("archive_status_counts", {})
    if counts:
        lines.extend(f"- `{name}`: {value}" for name, value in sorted(counts.items()))
    else:
        lines.append("- No archives were inspected.")
    lines.extend(
        [
            "",
            "## Structural summary",
            "",
            f"- Subjects: {len(summary.get('subjects', []))}",
            f"- Sessions: {len(summary.get('sessions', []))}",
            f"- Tasks: {len(summary.get('tasks', []))}",
            f"- Runs: {len(summary.get('runs', []))}",
            f"- Unique channel layouts: {len(summary.get('unique_channel_layouts', []))}",
            f"- Unique sampling rates: {summary.get('unique_sampling_rates_hz', [])}",
            f"- Missing `.set/.fdt` pairs: {summary.get('missing_set_fdt_pairs', 0)}",
            "",
            "This is a diagnostic inventory. It does not define targets, channel",
            "mapping, preprocessing, or a production dataset loader.",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(output_dir: Path, result: ToolResult) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "archive_inventory.csv", ARCHIVE_COLUMNS, result.archive_rows)
    _write_csv(output_dir / "extraction_manifest.csv", EXTRACTION_COLUMNS, result.extraction_rows)
    _write_csv(output_dir / "record_inventory.csv", RECORD_COLUMNS, result.record_rows)
    _write_csv(output_dir / "channel_inventory.csv", CHANNEL_COLUMNS, result.channel_rows)
    _write_csv(output_dir / "event_inventory.csv", EVENT_COLUMNS, result.event_rows)
    _write_csv(
        output_dir / "behavioural_inventory.csv",
        BEHAVIOURAL_COLUMNS,
        result.behavioural_rows,
    )
    _write_csv(
        output_dir / "file_pair_inventory.csv",
        FILE_PAIR_COLUMNS,
        result.file_pair_rows,
    )
    _write_csv(output_dir / "task_inventory.csv", TASK_COLUMNS, result.task_rows)
    _write_csv(output_dir / "session_inventory.csv", SESSION_COLUMNS, result.session_rows)
    _write_csv(output_dir / "subject_inventory.csv", SUBJECT_COLUMNS, result.subject_rows)
    _write_csv(output_dir / "errors.csv", ERROR_COLUMNS, result.error_rows)
    (output_dir / "inventory_summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "inventory_report.md").write_text(
        render_runtime_report(result.summary),
        encoding="utf-8",
    )


def run_tool(
    *,
    archives_dir: Path,
    extract_dir: Path,
    output_dir: Path,
    mode: str,
    resume: bool = False,
    overwrite: bool = False,
    verify_only: bool = False,
    require_complete: bool = False,
    checksum_manifest: Path | None = None,
    expected_subjects: Sequence[str] = EXPECTED_SUBJECTS,
    test_contents: bool = True,
    include_unselected_archives: bool = True,
) -> ToolResult:
    """Execute one CLI mode and write the standard artifact contract."""

    if mode not in {"verify-archives", "extract", "inventory", "all"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if not test_contents and mode in {"extract", "all"} and not verify_only:
        raise ValueError(
            "Content CRC testing cannot be skipped when extraction is requested"
        )
    archives_dir = Path(archives_dir)
    extract_dir = Path(extract_dir)
    output_dir = Path(output_dir)
    if checksum_manifest is None:
        checksum_manifest = discover_checksum_manifest(archives_dir)
    archive_rows: list[dict[str, Any]] = []
    extraction_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    behavioural_rows: list[dict[str, Any]] = []
    file_pair_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    fatal = False

    should_verify = (
        mode in {"verify-archives", "extract", "all"}
        or verify_only
        or require_complete
    )
    if should_verify:
        try:
            archive_rows = verify_archives(
                archives_dir,
                expected_subjects=expected_subjects,
                checksum_manifest=checksum_manifest,
                test_contents=test_contents,
                include_unselected_archives=include_unselected_archives,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            fatal = True
            error_rows.append(
                _error_row("verify", error, roots=[archives_dir, output_dir])
            )

    if not verify_only and mode in {"extract", "all"} and not fatal:
        def save_extraction_progress(
            rows: Sequence[Mapping[str, Any]],
            errors: Sequence[Mapping[str, Any]],
        ) -> None:
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_csv(output_dir / "extraction_manifest.csv", EXTRACTION_COLUMNS, rows)
            archive_progress = [
                {
                    "filename": row["filename"],
                    "subject_id": row["subject_id"],
                    "status": row["status"],
                    "error": row["error"],
                }
                for row in rows
                if row.get("member") == "__archive__"
            ]
            (output_dir / "extraction_progress.json").write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "completed_archive_count": len(archive_progress),
                        "archives": archive_progress,
                        "error_count": len(errors),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

        extracted, extraction_errors, extraction_fatal = extract_verified_archives(
            archives_dir,
            extract_dir,
            archive_rows,
            resume=resume,
            overwrite=overwrite,
            progress_callback=save_extraction_progress,
        )
        extraction_rows.extend(extracted)
        error_rows.extend(extraction_errors)
        fatal = fatal or extraction_fatal

    if not verify_only and mode in {"inventory", "all"}:
        records, channels, events, inventory_errors = inventory_extracted(extract_dir)
        record_rows.extend(records)
        channel_rows.extend(channels)
        event_rows.extend(events)
        error_rows.extend(inventory_errors)
        behavioural, behavioural_errors = inventory_behavioural(
            extract_dir,
            archives_dir.parent / "metadata",
        )
        behavioural_rows.extend(behavioural)
        error_rows.extend(behavioural_errors)
        file_pair_rows.extend(build_file_pair_inventory(record_rows, channel_rows))
        task_rows, session_rows, subject_rows = build_structural_inventories(record_rows)
        if mode == "inventory":
            manifest_path = output_dir / "extraction_manifest.csv"
            if manifest_path.is_file():
                with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
                    extraction_rows.extend(dict(row) for row in csv.DictReader(handle))

    summary = build_summary(
        mode=mode,
        archive_rows=archive_rows,
        extraction_rows=extraction_rows,
        record_rows=record_rows,
        channel_rows=channel_rows,
        event_rows=event_rows,
        behavioural_rows=behavioural_rows,
        file_pair_rows=file_pair_rows,
        error_rows=error_rows,
        expected_subjects=expected_subjects,
        checksum_manifest_available=checksum_manifest is not None,
        fatal=fatal,
    )
    incomplete = summary["overall_status"] != "complete"
    exit_code = 2 if fatal or (require_complete and incomplete) else 0
    result = ToolResult(
        exit_code=exit_code,
        overall_status=str(summary["overall_status"]),
        archive_rows=archive_rows,
        extraction_rows=extraction_rows,
        record_rows=record_rows,
        channel_rows=channel_rows,
        event_rows=event_rows,
        behavioural_rows=behavioural_rows,
        file_pair_rows=file_pair_rows,
        task_rows=task_rows,
        session_rows=session_rows,
        subject_rows=subject_rows,
        error_rows=sorted(
            error_rows,
            key=lambda row: tuple(str(row[column]) for column in ERROR_COLUMNS),
        ),
        summary=summary,
    )
    write_artifacts(output_dir, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives-dir", type=Path, required=True)
    parser.add_argument("--extract-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("verify-archives", "extract", "inventory", "all"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--checksum-manifest", type=Path)
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Optional diagnostic subset; normal complete verification expects sub-01..sub-29.",
    )
    parser.add_argument(
        "--skip-content-test",
        action="store_true",
        help="Inspect ZIP structure without reading every member (diagnostic only).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Re-raise fatal errors so Python prints the complete traceback.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected_subjects = (
        tuple(_clean_subject(item) for item in args.subjects)
        if args.subjects
        else EXPECTED_SUBJECTS
    )
    try:
        result = run_tool(
            archives_dir=args.archives_dir,
            extract_dir=args.extract_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            resume=args.resume,
            overwrite=args.overwrite,
            verify_only=args.verify_only,
            require_complete=args.require_complete,
            checksum_manifest=args.checksum_manifest,
            expected_subjects=expected_subjects,
            test_contents=not args.skip_content_test,
            include_unselected_archives=not bool(args.subjects),
        )
    except Exception as error:  # argparse entry point: turn fatal failures into code 2
        if args.debug:
            raise
        print(f"COG-BCI inventory failed: {_safe_error_text(error)}", file=sys.stderr)
        return 2
    print(
        f"COG-BCI inventory: status={result.overall_status}, "
        f"readable_archives={result.summary['readable_archive_count']}, "
        f"crc_verified_archives={result.summary['crc_verified_archive_count']}, "
        f"errors={result.summary['error_count']}"
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
