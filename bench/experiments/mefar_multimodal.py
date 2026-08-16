"""Leakage-safe MEFAR inventory, protocol planning, and deferred RF execution."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from bench.datasets.mefar_dataset import MEFARDataset
from bench.validation.cross_val import deterministic_group_kfold_indices
from bench.validation.metrics import MetricsCalculator
from model_zoo import build_model


SCHEMA_VERSION = "mefar-multimodal-v1"
EEG_PHYSIOLOGICAL_COLUMNS = (
    "Delta", "Theta", "Alpha1", "Alpha2", "Beta1", "Beta2", "Gamma1", "Gamma2"
)
EEG_DEVICE_COLUMNS = ("Attention", "Meditation")
WEARABLE_STREAMS = ("BVP", "EDA", "TEMP", "ACC_x", "ACC_y", "ACC_z", "HR", "IBI")
SUMMARY_STATISTICS = ("mean", "std", "min", "max", "median", "q25", "q75")
MODES = ("eeg_only", "wearable_only", "eeg_wearable")
EXPECTED_ARCHIVE_SHA256 = "c591ac136150032f58365248adbe52c68d063bc80a8846d22a32f29ad202048a"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "result_status", "dataset", "target",
        "synchronization", "features", "evaluation", "model", "output_dir",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"MEFAR config is missing: {missing}")
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported MEFAR config schema")
    if config["target"] != {
        "target_id": "mefar_cfs_fatigue_binary",
        "level": "session",
        "score": "Chalder Fatigue Scale Likert total",
        "threshold": {
            "operator": ">=",
            "value": 12,
            "class_0": "non_fatigue",
            "class_1": "fatigue",
        },
        "session_label_policy": "morning_evening_metadata_only",
        "diagnostic_target": "mefar_session_time_proxy",
    }:
        raise ValueError("MEFAR primary target contract must remain explicit and fixed")
    if config["evaluation"].get("protocol") != "group_kfold_participant":
        raise ValueError("MEFAR requires participant GroupKFold")
    if int(config["evaluation"].get("n_splits", 0)) != 5:
        raise ValueError("MEFAR requires five outer folds")
    if tuple(config["features"]["modes"]) != MODES:
        raise ValueError(f"MEFAR modes must be {MODES}")
    if config["model"].get("name") != "random_forest":
        raise ValueError("The preregistered MEFAR model is Random Forest")
    if config["model"].get("hyperparameter_search") is not False:
        raise ValueError("MEFAR hyperparameter search is forbidden")
    if config["model"].get("scaling") != "none":
        raise ValueError("Random Forest MEFAR protocol does not fit a scaler")
    return config


def safe_extract_nested(archive: Path, extracted_root: Path) -> dict[str, Any]:
    """Extract outer and nested ZIP members without path traversal."""
    before = file_sha256(archive)
    if before != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"Unexpected MEFAR archive SHA-256: {before}")
    extracted_root.mkdir(parents=True, exist_ok=True)
    archives = [archive]
    extracted_members = 0
    for current in archives:
        with ZipFile(current) as source:
            bad_member = source.testzip()
            if bad_member is not None:
                raise ValueError(f"ZIP CRC failed for {bad_member!r} in {current}")
            for member in source.infolist():
                destination = (extracted_root / member.filename).resolve()
                root = extracted_root.resolve()
                if destination != root and root not in destination.parents:
                    raise ValueError(f"Unsafe ZIP path: {member.filename!r}")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and destination.stat().st_size == member.file_size:
                    continue
                with source.open(member) as input_stream, destination.open("wb") as output_stream:
                    for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                        output_stream.write(chunk)
                extracted_members += 1
        if current == archive:
            archives.extend(sorted(extracted_root.rglob("*.zip")))
    after = file_sha256(archive)
    if after != before:
        raise RuntimeError("Source MEFAR archive changed during extraction")
    return {"archive_sha256_before": before, "archive_sha256_after": after, "new_files": extracted_members}


_XML_NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if letters is None:
        raise ValueError(f"Invalid spreadsheet reference {reference!r}")
    value = 0
    for letter in letters.group():
        value = value * 26 + ord(letter) - 64
    return value - 1


def read_xlsx_rows(path: Path) -> dict[str, list[list[str]]]:
    """Read values from the simple source XLSX without an optional Excel engine."""
    with ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.iter(f"{{{_XML_NS['m']}}}t"))
            for item in shared_root.findall("m:si", _XML_NS)
        ]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            item.attrib["Id"]: "xl/" + item.attrib["Target"].lstrip("/")
            for item in relations
        }
        result: dict[str, list[list[str]]] = {}
        for sheet in workbook.findall("m:sheets/m:sheet", _XML_NS):
            name = sheet.attrib["name"]
            relation_id = sheet.attrib[f"{{{_XML_NS['r']}}}id"]
            worksheet = ET.fromstring(archive.read(targets[relation_id]))
            rows: list[list[str]] = []
            for row in worksheet.findall("m:sheetData/m:row", _XML_NS):
                values: dict[int, str] = {}
                for cell in row.findall("m:c", _XML_NS):
                    node = cell.find("m:v", _XML_NS)
                    value = "" if node is None or node.text is None else node.text
                    if cell.attrib.get("t") == "s" and value:
                        value = shared[int(value)]
                    values[_column_index(cell.attrib["r"])] = value
                if values:
                    output = [""] * (max(values) + 1)
                    for column, value in values.items():
                        output[column] = value
                    rows.append(output)
            result[name] = rows
    return result


def _cfs_response_total(
    sheet_rows: list[list[str]], *, answer_column_offset: int, participant: str, session: str
) -> tuple[int, int]:
    points: list[int] = []
    for row in sheet_rows:
        padded = row + [""] * max(0, answer_column_offset + 5 - len(row))
        marked = [
            point
            for point, value in enumerate(
                padded[answer_column_offset + 1 : answer_column_offset + 5]
            )
            if value.strip().upper().startswith("X")
        ]
        if len(marked) > 1:
            raise ValueError(f"Multiple CFS responses for {participant} {session}")
        if marked:
            points.append(marked[0])
    if len(points) != 11:
        raise ValueError(
            f"Expected 11 CFS responses for {participant} {session}, found {len(points)}"
        )
    return sum(points), len(points)


def load_labels(workbook_path: Path) -> pd.DataFrame:
    workbook = read_xlsx_rows(workbook_path)
    rows = workbook["Subject List"]
    header = [item.strip() for item in rows[0]]
    output: list[dict[str, Any]] = []
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        source = dict(zip(header, padded))
        number = int(source["subjects"].lstrip("Ss"))
        subject_sheet = workbook.get(f"S{number}")
        if subject_sheet is None:
            raise ValueError(f"Missing CFS response sheet S{number}")
        sheet_text = "\n".join(value for row_values in subject_sheet for value in row_values)
        for session_label, score_key, order, answer_column_offset in (
            ("morning", "morning-mental fatigue state", 1, 0),
            ("evening", "evening-mental fatigue state", 2, 6),
        ):
            score = int(float(source[score_key]))
            score_match = re.search(
                rf"CFS {session_label.title()} Score\s*=\s*(\d+)", sheet_text
            )
            if score_match is None:
                raise ValueError(
                    f"Missing {session_label} CFS total in participant sheet S{number}"
                )
            sheet_score = int(score_match.group(1))
            response_score, response_count = _cfs_response_total(
                subject_sheet,
                answer_column_offset=answer_column_offset,
                participant=f"S{number}",
                session=session_label,
            )
            if sheet_score != score or response_score != score:
                raise ValueError(
                    f"CFS score mismatch for S{number} {session_label}: "
                    f"Subject List={score}, sheet total={sheet_score}, "
                    f"response sum={response_score}"
                )
            output.append(
                {
                    "participant_id": f"sub-{number:02d}",
                    "session_id": f"ses-{order:02d}-{session_label}",
                    "session_label": session_label,
                    "cfs_likert_score": score,
                    "target_id": "mefar_cfs_fatigue_binary",
                    "target": int(score >= 12),
                    "target_class_name": "fatigue" if score >= 12 else "non_fatigue",
                    "target_basis": "cfs_likert_score_greater_than_or_equal_to_12",
                    "cfs_threshold": 12,
                    "source_subject_list_score": score,
                    "source_reported_sheet_score": sheet_score,
                    "source_response_sheet_score": response_score,
                    "source_response_count": response_count,
                    "score_mapping_verified": True,
                    "session_time_proxy": 0 if session_label == "morning" else 1,
                    "session_time_proxy_role": "diagnostic_metadata_only",
                }
            )
    return pd.DataFrame(output).sort_values(["participant_id", "session_id"]).reset_index(drop=True)


def _line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: stream.read(1024 * 1024), b""))


def _empatica_header(path: Path, modality: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.reader(stream, skipinitialspace=True)
        rows = []
        for _ in range(2):
            try:
                rows.append(next(reader))
            except StopIteration:
                break
    row_count = _line_count(path)
    if modality == "IBI":
        start = float(rows[0][0]) if rows and rows[0] else None
        return {"start_timestamp": start, "sampling_rate_hz": None, "samples": max(0, row_count - 1)}
    if modality == "tags":
        return {"start_timestamp": None, "sampling_rate_hz": None, "samples": row_count}
    if len(rows) < 2:
        return {"start_timestamp": None, "sampling_rate_hz": None, "samples": 0}
    rate = float(rows[1][0])
    return {
        "start_timestamp": float(rows[0][0]),
        "sampling_rate_hz": rate,
        "samples": max(0, row_count - 2),
    }


def _eeg_metadata(path: Path) -> dict[str, Any]:
    count = 0
    times: list[float] = []
    class_values: Counter[str] = Counter()
    nonfinite = 0
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        columns = tuple((name or "").strip() for name in (reader.fieldnames or []))
        for raw in reader:
            row = {str(key).strip(): str(value).strip() for key, value in raw.items()}
            count += 1
            times.append(float(row["time"]))
            class_values[row["class"]] += 1
            for column in EEG_PHYSIOLOGICAL_COLUMNS + EEG_DEVICE_COLUMNS:
                try:
                    if not math.isfinite(float(row[column])):
                        nonfinite += 1
                except ValueError:
                    nonfinite += 1
    return {
        "columns": columns,
        "samples": count,
        "sampling_rate_hz": None,
        "observed_median_step_seconds": float(np.median(np.diff(times))) if len(times) > 1 else None,
        "relative_start_seconds": min(times) if times else None,
        "relative_end_seconds": max(times) if times else None,
        "duration_seconds": max(times) - min(times) if times else 0.0,
        "nonfinite_feature_values": nonfinite,
        "source_class_values": dict(class_values),
    }


def _processed_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "MEFAR_preprocessed").glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            header = [value.strip() for value in next(csv.reader(stream))]
        n_rows = max(0, _line_count(path) - 1)
        rows.append(
            {
                "file": path.relative_to(root).as_posix(),
                "rows": n_rows,
                "columns": len(header),
                "schema": "|".join(header),
                "normalized": True,
                "participant_id_retained": False,
                "session_id_retained": False,
                "safe_for_primary_benchmark": False,
                "reason": "subject/session IDs absent; apparent resampling and global normalization cannot be audited",
            }
        )
    return rows


def build_inventory(config: Mapping[str, Any]) -> dict[str, Any]:
    archive = Path(config["dataset"]["archive"])
    extracted = Path(config["dataset"]["extracted_root"])
    archive_before = file_sha256(archive)
    if archive_before != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("MEFAR source archive does not match the validated SHA-256")
    dataset = MEFARDataset({"data_path": str(extracted)})
    records = list(dataset.iter_records())
    labels = load_labels(extracted / "MEFAR" / "general_info.xlsx")
    label_by_session = labels.set_index(["participant_id", "session_id"]).to_dict("index")
    session_rows: list[dict[str, Any]] = []
    modality_rows: list[dict[str, Any]] = []
    for record in records:
        session_dir = extracted / record.relative_path
        validation = dataset.validate_record_files(record)
        eeg = _eeg_metadata(session_dir / "EEG.csv")
        modality_metadata: dict[str, dict[str, Any]] = {}
        for modality in ("ACC", "BVP", "EDA", "HR", "IBI", "TEMP", "tags"):
            path = session_dir / f"{modality}.csv"
            metadata = _empatica_header(path, modality)
            modality_metadata[modality] = metadata
            rate = metadata["sampling_rate_hz"]
            duration = (
                metadata["samples"] / rate if rate not in (None, 0) else None
            )
            modality_rows.append(
                {
                    "record_id": record.record_id,
                    "participant_id": record.participant_id,
                    "session_id": record.session_id,
                    "modality": modality,
                    "relative_path": (Path(record.relative_path) / f"{modality}.csv").as_posix(),
                    "size_bytes": path.stat().st_size,
                    "samples": metadata["samples"],
                    "sampling_rate_hz": rate,
                    "start_timestamp": metadata["start_timestamp"],
                    "duration_seconds": duration,
                    "empty": path.stat().st_size == 0,
                }
            )
        modality_rows.append(
            {
                "record_id": record.record_id,
                "participant_id": record.participant_id,
                "session_id": record.session_id,
                "modality": "EEG",
                "relative_path": (Path(record.relative_path) / "EEG.csv").as_posix(),
                "size_bytes": (session_dir / "EEG.csv").stat().st_size,
                "samples": eeg["samples"],
                "sampling_rate_hz": eeg["sampling_rate_hz"],
                "start_timestamp": None,
                "duration_seconds": eeg["duration_seconds"],
                "empty": False,
            }
        )
        wearable_starts = [
            value["start_timestamp"] for key, value in modality_metadata.items()
            if key not in {"tags"} and value["start_timestamp"] is not None
        ]
        label = label_by_session[(record.participant_id, record.session_id)]
        session_rows.append(
            {
                **record.to_dict(),
                **validation,
                **label,
                "eeg_samples": eeg["samples"],
                "eeg_duration_seconds": eeg["duration_seconds"],
                "eeg_relative_start_seconds": eeg["relative_start_seconds"],
                "eeg_median_step_seconds": eeg["observed_median_step_seconds"],
                "eeg_nonfinite_values": eeg["nonfinite_feature_values"],
                "eeg_source_class_values": json.dumps(eeg["source_class_values"], sort_keys=True),
                "wearable_start_spread_seconds": max(wearable_starts) - min(wearable_starts),
                "tags_count": modality_metadata["tags"]["samples"],
                "ibi_empty": (session_dir / "IBI.csv").stat().st_size == 0,
                "usable_multimodal": bool(validation["complete_core_modalities"]),
            }
        )
    sessions = pd.DataFrame(session_rows).sort_values(["participant_id", "session_id"]).reset_index(drop=True)
    modalities = pd.DataFrame(modality_rows).sort_values(
        ["participant_id", "session_id", "modality"]
    ).reset_index(drop=True)
    participant_rows = []
    for participant, group in sessions.groupby("participant_id", sort=True):
        participant_rows.append(
            {
                "participant_id": participant,
                "sessions": len(group),
                "session_labels": "|".join(sorted(group["session_label"].tolist())),
                "complete_multimodal_sessions": int(group["usable_multimodal"].sum()),
                "cfs_scores": "|".join(str(int(value)) for value in group["cfs_likert_score"]),
                "all_modalities_complete": bool(group["usable_multimodal"].all()),
            }
        )
    participants = pd.DataFrame(participant_rows)
    file_rows = []
    for path in sorted(item for item in extracted.rglob("*") if item.is_file()):
        file_rows.append(
            {
                "relative_path": path.relative_to(extracted).as_posix(),
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "empty": path.stat().st_size == 0,
                "sha256": file_sha256(path),
            }
        )
    files = pd.DataFrame(file_rows)
    processed = pd.DataFrame(_processed_inventory(extracted))
    archive_after = file_sha256(archive)
    if archive_before != archive_after:
        raise RuntimeError("MEFAR source archive was modified during inventory")
    return {
        "archive_before": archive_before,
        "archive_after": archive_after,
        "files": files,
        "participants": participants,
        "sessions": sessions,
        "modalities": modalities,
        "labels": labels,
        "processed": processed,
    }


def build_fold_manifest(sessions: pd.DataFrame, n_splits: int = 5) -> dict[str, Any]:
    cohort = sessions.loc[sessions["usable_multimodal"]].reset_index(drop=True)
    splits = deterministic_group_kfold_indices(
        cohort["participant_id"].to_numpy(), n_splits=n_splits
    )
    folds = []
    seen: set[str] = set()
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_subjects = sorted(cohort.iloc[train_idx]["participant_id"].unique().tolist())
        test_subjects = sorted(cohort.iloc[test_idx]["participant_id"].unique().tolist())
        if set(train_subjects) & set(test_subjects):
            raise RuntimeError(f"Participant leakage in fold {fold}")
        test_ids = sorted(cohort.iloc[test_idx]["record_id"].tolist())
        if seen & set(test_ids):
            raise RuntimeError("Duplicate evaluation sample IDs across folds")
        seen.update(test_ids)
        folds.append(
            {
                "fold": fold,
                "train_participants": train_subjects,
                "test_participants": test_subjects,
                "train_sample_ids": sorted(cohort.iloc[train_idx]["record_id"].tolist()),
                "test_sample_ids": test_ids,
                "train_class_counts": {
                    str(key): int(value)
                    for key, value in cohort.iloc[train_idx]["target"].value_counts().sort_index().items()
                },
                "test_class_counts": {
                    str(key): int(value)
                    for key, value in cohort.iloc[test_idx]["target"].value_counts().sort_index().items()
                },
                "participant_overlap": 0,
            }
        )
    if seen != set(cohort["record_id"]):
        raise RuntimeError("Outer folds do not cover every multimodal session exactly once")
    return {"protocol": "5-fold GroupKFold by participant_id", "folds": folds}


def feature_names(mode: str, *, include_device_metrics: bool = False) -> list[str]:
    if mode not in MODES:
        raise ValueError(f"Unknown MEFAR mode {mode!r}")
    eeg_columns = EEG_PHYSIOLOGICAL_COLUMNS + (EEG_DEVICE_COLUMNS if include_device_metrics else ())
    eeg = [f"eeg_{column}_{stat}" for column in eeg_columns for stat in SUMMARY_STATISTICS]
    wearable = [f"wearable_{column}_{stat}" for column in WEARABLE_STREAMS for stat in SUMMARY_STATISTICS]
    wearable.append("wearable_ibi_rmssd")
    if mode == "eeg_only":
        return eeg
    if mode == "wearable_only":
        return wearable
    return eeg + wearable


def build_run_matrix(config: Mapping[str, Any], fold_manifest: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for fold in fold_manifest["folds"]:
        sample_ids = fold["test_sample_ids"]
        for mode in MODES:
            payload = {
                "experiment_id": config["experiment_id"],
                "fold": fold["fold"],
                "mode": mode,
                "model": config["model"]["name"],
                "params": config["model"]["params"],
                "target_id": config["target"]["target_id"],
                "evaluation_sample_ids": sample_ids,
                "feature_names": feature_names(mode),
            }
            rows.append(
                {
                    "run_id": f"{mode}__fold{fold['fold']:02d}__{stable_hash(payload)[:10]}",
                    "fold": fold["fold"],
                    "mode": mode,
                    "model": config["model"]["name"],
                    "target_id": config["target"]["target_id"],
                    "n_train_participants": len(fold["train_participants"]),
                    "n_test_participants": len(fold["test_participants"]),
                    "n_train_samples": len(fold["train_sample_ids"]),
                    "n_test_samples": len(sample_ids),
                    "feature_count": len(feature_names(mode)),
                    "evaluation_sample_ids_hash": stable_hash(sample_ids),
                    "specification_hash": stable_hash(payload),
                }
            )
    return pd.DataFrame(rows)


def build_protocol(config: Mapping[str, Any], inventory: Mapping[str, Any]) -> dict[str, Any]:
    sessions = inventory["sessions"]
    folds = build_fold_manifest(sessions, n_splits=int(config["evaluation"]["n_splits"]))
    matrix = build_run_matrix(config, folds)
    cohort = sessions.loc[sessions["usable_multimodal"]]
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "result_status": config["result_status"],
        "archive_sha256": inventory["archive_before"],
        "participants": int(cohort["participant_id"].nunique()),
        "sessions": int(len(cohort)),
        "target": config["target"],
        "synchronization_level": "participant_session_summary",
        "windowing": "one full-session summary sample; no window-level clock merge",
        "modes": {
            mode: {"feature_count": len(feature_names(mode)), "sample_ids_hash": stable_hash(sorted(cohort["record_id"]))}
            for mode in MODES
        },
        "same_evaluation_sample_ids": True,
        "fold_manifest_hash": stable_hash(folds),
        "run_matrix_hash": stable_hash(matrix.to_dict(orient="records")),
        "run_count": int(len(matrix)),
        "model": config["model"],
        "primary_metrics": ["macro_f1", "balanced_accuracy"],
        "secondary_metrics": [
            "accuracy", "per_class_precision", "per_class_recall", "confusion_matrix"
        ],
        "primary_effect": "macro_f1(eeg_wearable) - macro_f1(eeg_only)",
        "secondary_effect": "macro_f1(wearable_only) - macro_f1(eeg_only)",
        "leakage_guards": {
            "participant_disjoint_outer_folds": True,
            "train_only_imputation": True,
            "global_scaler": False,
            "oversampling": False,
            "processed_down_mid_up_used": False,
            "label_columns_excluded_from_features": True,
            "device_attention_meditation_excluded_from_primary_eeg": True,
        },
    }
    protocol["protocol_hash"] = stable_hash(protocol)
    return {"folds": folds, "matrix": matrix, "protocol": protocol}


def build_plan_summary(
    inventory: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    sessions = inventory["sessions"]
    transitions = []
    for participant_id, group in sessions.groupby("participant_id", sort=True):
        ordered = group.sort_values("session_order")
        transitions.append(
            {
                "participant_id": participant_id,
                "morning_class": int(ordered.iloc[0]["target"]),
                "evening_class": int(ordered.iloc[1]["target"]),
            }
        )
    changed = sum(item["morning_class"] != item["evening_class"] for item in transitions)
    by_session = {
        session_label: {
            str(key): int(value)
            for key, value in group["target"].value_counts().sort_index().items()
        }
        for session_label, group in sessions.groupby("session_label", sort=True)
    }
    return {
        **plan["protocol"],
        "class_distribution": {
            str(key): int(value)
            for key, value in sessions["target"].value_counts().sort_index().items()
        },
        "class_distribution_by_session": by_session,
        "participants_changing_class": changed,
        "participants_same_class": len(transitions) - changed,
        "fold_class_distribution": [
            {
                "fold": fold["fold"],
                "train_class_counts": fold["train_class_counts"],
                "test_class_counts": fold["test_class_counts"],
                "both_classes_in_train": len(fold["train_class_counts"]) == 2,
                "both_classes_in_test": len(fold["test_class_counts"]) == 2,
            }
            for fold in plan["folds"]["folds"]
        ],
        "existing_group_kfold_usable": all(
            len(fold["train_class_counts"]) == 2 and len(fold["test_class_counts"]) == 2
            for fold in plan["folds"]["folds"]
        ),
        "missingness": {
            "empty_ibi_sessions": int(sessions["ibi_empty"].sum()),
            "empty_tag_sessions": int(sessions["tags_count"].eq(0).sum()),
            "missing_core_modality_sessions": int((~sessions["usable_multimodal"]).sum()),
        },
        "folds": plan["folds"]["folds"],
        "models_trained": 0,
        "writes_performed": False,
    }


def plan_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    inventory = build_inventory(config)
    plan = build_protocol(config, inventory)
    return build_plan_summary(inventory, plan)


def write_inventory_artifacts(config_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    output = Path(output_dir or config["output_dir"])
    inventory = build_inventory(config)
    plan = build_protocol(config, inventory)
    output.mkdir(parents=True, exist_ok=True)
    archive_manifest = {
        "logical_path": config["dataset"]["archive"],
        "size_bytes": Path(config["dataset"]["archive"]).stat().st_size,
        "sha256": inventory["archive_before"],
        "sha256_after": inventory["archive_after"],
        "immutable": inventory["archive_before"] == inventory["archive_after"],
        "outer_entries": 2,
        "extracted_file_count": int(len(inventory["files"])),
        "extracted_file_types": {
            str(key): int(value)
            for key, value in inventory["files"]["suffix"].value_counts().sort_index().items()
        },
        "directory_contract": "MEFAR/subject_<1..23>/<1.morning|2.evening>/<modality>.csv",
        "crc_validated": True,
        "nested_archives": [
            {
                "name": "MEFAR_raw_data.zip",
                "sha256": "2de6973d2c9e4595c670dd01f66d592efda1daaea7342eeba518a1913ab8afbc",
                "entries": 485,
                "files": 415,
            },
            {
                "name": "MEFAR_preprocessed.zip",
                "sha256": "e5ddd75d14ce35ff5a3e86b22a0917ea6cfb5089cf3a7875d0ca03ad158b359e",
                "entries": 4,
                "files": 3,
            },
        ],
    }
    synchronization = inventory["sessions"].loc[:, [
        "record_id", "participant_id", "session_id", "eeg_relative_start_seconds",
        "eeg_duration_seconds", "wearable_start_spread_seconds", "tags_count",
    ]].copy()
    synchronization["common_absolute_clock"] = False
    synchronization["explicit_sync_marker"] = synchronization["tags_count"].gt(0)
    synchronization["safe_fusion_level"] = "participant_session_summary"
    _write_json(output / "archive_manifest.json", archive_manifest)
    _write_csv(output / "file_inventory.csv", inventory["files"])
    _write_csv(output / "participant_inventory.csv", inventory["participants"])
    _write_csv(output / "session_inventory.csv", inventory["sessions"])
    _write_csv(output / "modality_inventory.csv", inventory["modalities"])
    _write_csv(output / "label_audit.csv", inventory["labels"])
    _write_csv(output / "synchronization_audit.csv", synchronization)
    _write_json(output / "fold_manifest.json", plan["folds"])
    _write_csv(output / "run_matrix.csv", plan["matrix"])
    _write_json(output / "protocol_manifest.json", plan["protocol"])
    plan_summary = build_plan_summary(inventory, plan)
    plan_summary["writes_performed"] = True
    _write_json(output / "plan_summary.json", plan_summary)
    _write_csv(output / "processed_dataset_audit.csv", inventory["processed"])
    return {**plan["protocol"], "output_dir": output.as_posix(), "writes_performed": True, "models_trained": 0}


def _summary(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return [float("nan")] * len(SUMMARY_STATISTICS)
    return [
        float(np.mean(finite)), float(np.std(finite)), float(np.min(finite)),
        float(np.max(finite)), float(np.median(finite)),
        float(np.quantile(finite, 0.25)), float(np.quantile(finite, 0.75)),
    ]


def _read_numeric_csv(path: Path, *, skip_rows: int, columns: int = 1) -> list[np.ndarray]:
    data: list[list[float]] = [[] for _ in range(columns)]
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.reader(stream, skipinitialspace=True)
        for _ in range(skip_rows):
            next(reader, None)
        for row in reader:
            for index in range(min(columns, len(row))):
                try:
                    data[index].append(float(row[index]))
                except ValueError:
                    data[index].append(float("nan"))
    return [np.asarray(values, dtype=float) for values in data]


def materialize_session_features(config: Mapping[str, Any], sessions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    extracted = Path(config["dataset"]["extracted_root"])
    rows_by_mode = {mode: [] for mode in MODES}
    for session in sessions.loc[sessions["usable_multimodal"]].to_dict(orient="records"):
        session_dir = extracted / session["relative_path"]
        eeg_values = {column: [] for column in EEG_PHYSIOLOGICAL_COLUMNS}
        with (session_dir / "EEG.csv").open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            for raw in csv.DictReader(stream, skipinitialspace=True):
                row = {str(key).strip(): str(value).strip() for key, value in raw.items()}
                for column in EEG_PHYSIOLOGICAL_COLUMNS:
                    try:
                        eeg_values[column].append(float(row[column]))
                    except ValueError:
                        eeg_values[column].append(float("nan"))
        eeg_features = [value for column in EEG_PHYSIOLOGICAL_COLUMNS for value in _summary(np.asarray(eeg_values[column]))]
        wearable_arrays = {
            "BVP": _read_numeric_csv(session_dir / "BVP.csv", skip_rows=2)[0],
            "EDA": _read_numeric_csv(session_dir / "EDA.csv", skip_rows=2)[0],
            "TEMP": _read_numeric_csv(session_dir / "TEMP.csv", skip_rows=2)[0],
            "HR": _read_numeric_csv(session_dir / "HR.csv", skip_rows=2)[0],
        }
        acc = _read_numeric_csv(session_dir / "ACC.csv", skip_rows=2, columns=3)
        wearable_arrays.update({"ACC_x": acc[0], "ACC_y": acc[1], "ACC_z": acc[2]})
        ibi = _read_numeric_csv(session_dir / "IBI.csv", skip_rows=1, columns=2)[1] if (session_dir / "IBI.csv").stat().st_size else np.asarray([])
        wearable_arrays["IBI"] = ibi
        wearable_features = [value for column in WEARABLE_STREAMS for value in _summary(wearable_arrays[column])]
        wearable_features.append(
            float(np.sqrt(np.mean(np.diff(ibi) ** 2))) if len(ibi) > 1 else float("nan")
        )
        metadata = {
            "sample_id": session["record_id"], "participant_id": session["participant_id"],
            "session_id": session["session_id"], "target": int(session["target"]),
        }
        for mode, values in (
            ("eeg_only", eeg_features), ("wearable_only", wearable_features),
            ("eeg_wearable", eeg_features + wearable_features),
        ):
            rows_by_mode[mode].append(metadata | dict(zip(feature_names(mode), values)))
    return {mode: pd.DataFrame(rows) for mode, rows in rows_by_mode.items()}


def run_experiment(config_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Execute the preregistered RF experiment; never called by plan-only."""
    config = load_config(config_path)
    output = Path(output_dir or config["output_dir"])
    inventory = build_inventory(config)
    plan = build_protocol(config, inventory)
    features = materialize_session_features(config, inventory["sessions"])
    summary_rows = []
    for run in plan["matrix"].to_dict(orient="records"):
        fold = plan["folds"]["folds"][int(run["fold"]) - 1]
        frame = features[str(run["mode"])]
        train = frame["sample_id"].isin(fold["train_sample_ids"])
        test = frame["sample_id"].isin(fold["test_sample_ids"])
        columns = feature_names(str(run["mode"]))
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(frame.loc[train, columns])
        x_test = imputer.transform(frame.loc[test, columns])
        y_train = frame.loc[train, "target"].to_numpy(dtype=int)
        y_test = frame.loc[test, "target"].to_numpy(dtype=int)
        model = build_model("random_forest", "classification", (len(columns),), 2, config["model"]["params"])
        model.fit(x_train, y_train)
        prediction = np.asarray(model.predict(x_test), dtype=int)
        probabilities = np.asarray(model.predict_proba(x_test), dtype=float)
        metrics = MetricsCalculator.calculate_all_metrics(y_test, prediction, y_proba=probabilities, labels=np.asarray([0, 1]))
        metrics["class_metrics"] = MetricsCalculator.calculate_class_metrics(
            y_test, prediction, labels=np.asarray([0, 1])
        )
        run_dir = output / str(run["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        predictions = frame.loc[test, ["sample_id", "participant_id", "session_id", "target"]].copy()
        predictions = predictions.rename(columns={"target": "y_true"})
        predictions["y_pred"] = prediction
        predictions["proba_0"] = probabilities[:, 0]
        predictions["proba_1"] = probabilities[:, 1]
        predictions.to_parquet(run_dir / "predictions.parquet", index=False)
        _write_json(run_dir / "metrics.json", metrics)
        _write_json(run_dir / "normalization_stats.json", {
            "imputation": "outer_train_median", "scaling": "none",
            "statistics_fit_sample_ids": sorted(frame.loc[train, "sample_id"].tolist()),
            "median": imputer.statistics_.tolist(),
        })
        summary_rows.append({**run, **{key: value for key, value in metrics.items() if np.isscalar(value)}})
    _write_csv(output / "summary.csv", pd.DataFrame(summary_rows))
    _write_json(output / "protocol_manifest.json", plan["protocol"])
    return plan["protocol"] | {"models_trained": len(summary_rows), "writes_performed": True}


__all__ = [
    "MODES", "SCHEMA_VERSION", "build_fold_manifest", "build_inventory",
    "build_plan_summary",
    "build_protocol", "build_run_matrix", "feature_names", "load_config",
    "load_labels", "materialize_session_features", "plan_experiment",
    "run_experiment", "safe_extract_nested", "stable_hash", "write_inventory_artifacts",
]
