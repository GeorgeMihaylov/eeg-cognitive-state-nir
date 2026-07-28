from __future__ import annotations

import csv
import hashlib
import json
import sys
from types import SimpleNamespace
import zipfile
from pathlib import Path

import pytest

from scripts.data import cog_bci_inventory as inventory


def make_zip(path: Path, members: dict[str, bytes] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in (members or {"sub-01/ses-S1/eeg/task.set": b"set"}).items():
            archive.writestr(name, content)
    return path


def standard_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return tmp_path / "archives", tmp_path / "extracted", tmp_path / "output"


def test_valid_archive_and_unavailable_checksum_are_explicit(tmp_path: Path) -> None:
    archives, _, _ = standard_paths(tmp_path)
    make_zip(archives / "sub-01.zip")
    row = inventory.verify_archives(archives, expected_subjects=["sub-01"])[0]
    assert row["status"] == "valid"
    assert row["zip_readable"] == "true"
    assert row["zip_test_passed"] == "true"
    assert row["checksum_expected"] == "not_available"
    assert row["checksum_match"] == "not_available"


def test_checksum_manifest_match_and_mismatch(tmp_path: Path) -> None:
    archives, _, _ = standard_paths(tmp_path)
    archive = make_zip(archives / "sub-01.zip")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = archives / "checksums.json"
    manifest.write_text(json.dumps({"sub-01.zip": digest}), encoding="utf-8")
    matched = inventory.verify_archives(
        archives, expected_subjects=["01"], checksum_manifest=manifest
    )[0]
    assert matched["checksum_match"] == "true"
    manifest.write_text(json.dumps({"sub-01.zip": "0" * 64}), encoding="utf-8")
    mismatched = inventory.verify_archives(
        archives, expected_subjects=["01"], checksum_manifest=manifest
    )[0]
    assert mismatched["status"] == "checksum_mismatch"
    assert mismatched["checksum_match"] == "false"


def test_corrupt_and_partial_archives_are_distinguished(tmp_path: Path) -> None:
    archives, _, _ = standard_paths(tmp_path)
    archives.mkdir()
    (archives / "sub-01.zip").write_bytes(b"not a zip")
    (archives / "sub-02.zip").write_bytes(b"PK\x03\x04incomplete")
    rows = inventory.verify_archives(archives, expected_subjects=["01", "02"])
    assert {row["subject_id"]: row["status"] for row in rows} == {
        "sub-01": "corrupt",
        "sub-02": "partial",
    }


def test_missing_archive_and_empty_directory_are_incomplete_not_fatal(
    tmp_path: Path,
) -> None:
    archives, extracted, output = standard_paths(tmp_path)
    result = inventory.run_tool(
        archives_dir=archives,
        extract_dir=extracted,
        output_dir=output,
        mode="verify-archives",
        expected_subjects=["01"],
    )
    assert result.exit_code == 0
    assert result.overall_status == "incomplete"
    assert result.archive_rows[0]["status"] == "missing"


def test_require_complete_returns_nonzero(tmp_path: Path) -> None:
    archives, extracted, output = standard_paths(tmp_path)
    result = inventory.run_tool(
        archives_dir=archives,
        extract_dir=extracted,
        output_dir=output,
        mode="verify-archives",
        require_complete=True,
        expected_subjects=["01"],
    )
    assert result.exit_code != 0


def test_duplicate_subject_and_unexpected_archive_name(tmp_path: Path) -> None:
    archives, _, _ = standard_paths(tmp_path)
    make_zip(archives / "sub-01.zip")
    make_zip(archives / "copy-sub-01.zip")
    make_zip(archives / "mystery.zip")
    rows = inventory.verify_archives(archives, expected_subjects=["01"])
    statuses = {row["filename"]: row["status"] for row in rows}
    assert statuses["sub-01.zip"] == "duplicate_subject"
    assert statuses["copy-sub-01.zip"] == "duplicate_subject"
    assert statuses["mystery.zip"] == "unexpected_name"


@pytest.mark.parametrize("member", ["../escape.txt", "/absolute.txt", "C:\\escape.txt"])
def test_zip_slip_and_absolute_members_are_rejected_before_writing(
    tmp_path: Path, member: str
) -> None:
    archives, extracted, _ = standard_paths(tmp_path)
    archive = make_zip(archives / "sub-01.zip", {member: b"bad"})
    with pytest.raises(inventory.UnsafeArchiveError):
        inventory.safe_extract_archive(archive, extracted)
    assert not (tmp_path / "escape.txt").exists()
    assert not extracted.exists()


def test_extract_reextract_and_source_archive_preserved(tmp_path: Path) -> None:
    archives, extracted, _ = standard_paths(tmp_path)
    archive = make_zip(
        archives / "sub-01.zip",
        {
            "sub-01/ses-S1/eeg/task.set": b"set",
            "sub-01/ses-S1/eeg/task.fdt": b"fdt",
        },
    )
    before = hashlib.sha256(archive.read_bytes()).hexdigest()
    first = inventory.safe_extract_archive(archive, extracted)
    second = inventory.safe_extract_archive(archive, extracted)
    assert {row["status"] for row in first if row["size_bytes"]} == {"extracted"}
    assert {row["status"] for row in second if row["size_bytes"]} == {"already_correct"}
    assert archive.exists()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == before


def test_resume_after_incomplete_marker(tmp_path: Path) -> None:
    archives, extracted, _ = standard_paths(tmp_path)
    archive = make_zip(archives / "sub-01.zip", {"task.set": b"original"})
    extracted.mkdir()
    (extracted / ".sub-01.extracting.json").write_text("{}", encoding="utf-8")
    with pytest.raises(inventory.ExtractionConflictError, match="--resume"):
        inventory.safe_extract_archive(archive, extracted)
    rows = inventory.safe_extract_archive(archive, extracted, resume=True)
    assert rows[0]["status"] == "extracted"
    assert not (extracted / ".sub-01.extracting.json").exists()


def test_changed_existing_file_requires_overwrite(tmp_path: Path) -> None:
    archives, extracted, _ = standard_paths(tmp_path)
    archive = make_zip(archives / "sub-01.zip", {"task.set": b"archive"})
    destination = extracted / "sub-01" / "task.set"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"changed")
    with pytest.raises(inventory.ExtractionConflictError):
        inventory.safe_extract_archive(archive, extracted)
    assert destination.read_bytes() == b"changed"
    inventory.safe_extract_archive(archive, extracted, overwrite=True)
    assert destination.read_bytes() == b"archive"


def test_pair_inventory_and_unpaired_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, extracted, _ = standard_paths(tmp_path)
    eeg = extracted / "sub-01" / "ses-S1" / "eeg"
    eeg.mkdir(parents=True)
    (eeg / "paired.set").write_bytes(b"set")
    (eeg / "paired.fdt").write_bytes(b"fdt")
    (eeg / "set_only.set").write_bytes(b"set")
    (eeg / "fdt_only.fdt").write_bytes(b"fdt")
    monkeypatch.setattr(
        inventory,
        "_read_eeglab_metadata",
        lambda path: (
            {
                "channel_count": 2,
                "n_channels": 2,
                "channel_names": "C3|C4",
                "sampling_rate_hz": 256.0,
                "n_samples": 256,
                "duration_seconds": 1.0,
                "data_units": "uV",
                "event_count": 0,
                "event_types": "",
                "annotations_count": 0,
                "annotation_count": 0,
                "reference": "",
                "bad_channels": "",
                "metadata_reader": "mock",
                "reader_used": "mock",
                "read_status": "ok",
                "error": "",
            },
            [],
        ),
    )
    records, channels, _, errors = inventory.inventory_extracted(extracted)
    by_name = {Path(row["relative_path"]).name: row for row in records}
    assert by_name["paired.set"]["paired_fdt_file"].endswith("paired.fdt")
    assert by_name["paired.fdt"]["paired_set_file"].endswith("paired.set")
    assert any("no matching .fdt" in row["error"] for row in errors)
    assert any("no matching .set" in row["error"] for row in errors)
    assert channels[0]["channel_names"] == "C3|C4"
    pairs = inventory.build_file_pair_inventory(records, channels)
    assert {row["status"] for row in pairs} == {"invalid", "missing"}


def test_unknown_and_service_files_are_reported(tmp_path: Path) -> None:
    _, extracted, _ = standard_paths(tmp_path)
    subject = extracted / "sub-01"
    subject.mkdir(parents=True)
    (subject / "unknown.bin").write_bytes(b"x")
    (subject / "Thumbs.db").write_bytes(b"x")
    records, _, _, errors = inventory.inventory_extracted(extracted)
    assert records[0]["file_type"] == "unknown"
    assert any("Unknown file type" in row["error"] for row in errors)
    assert any("Service file ignored" in row["error"] for row in errors)


def test_participant_naming_mismatch_is_reported(tmp_path: Path) -> None:
    _, extracted, _ = standard_paths(tmp_path)
    extracted.mkdir()
    (extracted / "record.txt").write_text("metadata", encoding="utf-8")
    records, _, _, errors = inventory.inventory_extracted(extracted)
    assert records[0]["subject_id"] == ""
    assert any("Participant identifier" in row["error"] for row in errors)


def test_tsv_events_are_inventoried_without_target_creation(tmp_path: Path) -> None:
    _, extracted, _ = standard_paths(tmp_path)
    path = extracted / "sub-01" / "ses-S1" / "task-test_events.tsv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "trial_type\tvalue\trating_workload\taccuracy\nstimulus\t7\t4\t1\n",
        encoding="utf-8",
    )
    _, _, events, _ = inventory.inventory_extracted(extracted)
    assert events[0]["event_label"] == "stimulus"
    assert events[0]["trigger_code"] == "7"
    assert events[0]["rating_name"] == "rating_workload"
    assert events[0]["behavioural_outcome"] == "accuracy"
    assert all("target" not in key for key in events[0])


def test_eeglab_failure_is_local_and_inventory_continues(tmp_path: Path) -> None:
    _, extracted, _ = standard_paths(tmp_path)
    path = extracted / "sub-01" / "ses-S1" / "eeg" / "task.set"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a set")
    records, channels, _, errors = inventory.inventory_extracted(extracted)
    assert len(records) == 1
    assert len(channels) == 1
    assert channels[0]["metadata_reader"] == "unavailable"
    assert any(row["stage"] == "eeglab" for row in errors)


def test_all_artifacts_are_written_with_relative_paths(tmp_path: Path) -> None:
    archives, extracted, output = standard_paths(tmp_path)
    make_zip(
        archives / "sub-01.zip",
        {
            "sub-01/ses-S1/eeg/task.set": b"set",
            "sub-01/ses-S1/eeg/task.fdt": b"fdt",
        },
    )
    result = inventory.run_tool(
        archives_dir=archives,
        extract_dir=extracted,
        output_dir=output,
        mode="all",
        expected_subjects=["01"],
    )
    assert result.exit_code == 0
    expected = {
        "archive_inventory.csv",
        "extraction_manifest.csv",
        "record_inventory.csv",
        "channel_inventory.csv",
        "event_inventory.csv",
        "behavioural_inventory.csv",
        "file_pair_inventory.csv",
        "task_inventory.csv",
        "session_inventory.csv",
        "subject_inventory.csv",
        "extraction_progress.json",
        "inventory_summary.json",
        "inventory_report.md",
        "errors.csv",
    }
    assert {path.name for path in output.iterdir()} == expected
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in output.iterdir() if path.suffix in {".csv", ".md", ".json"}
    )
    assert str(tmp_path) not in combined
    assert "F:\\" not in combined


def test_cli_modes_and_verify_only(tmp_path: Path) -> None:
    for mode in ("verify-archives", "extract", "inventory", "all"):
        root = tmp_path / mode
        archives, extracted, output = standard_paths(root)
        make_zip(archives / "sub-01.zip", {"sub-01/task.txt": b"ok"})
        args = [
            "--archives-dir",
            str(archives),
            "--extract-dir",
            str(extracted),
            "--output-dir",
            str(output),
            "--mode",
            mode,
            "--subjects",
            "01",
        ]
        assert inventory.main(args) == 0
        assert (output / "inventory_summary.json").is_file()
    verify_root = tmp_path / "verify-only"
    archives, extracted, output = standard_paths(verify_root)
    make_zip(archives / "sub-01.zip")
    assert (
        inventory.main(
            [
                "--archives-dir",
                str(archives),
                "--extract-dir",
                str(extracted),
                "--output-dir",
                str(output),
                "--mode",
                "all",
                "--verify-only",
                "--subjects",
                "01",
            ]
        )
        == 0
    )
    assert not extracted.exists()


def test_cli_require_complete_and_fatal_extract_codes(tmp_path: Path) -> None:
    archives, extracted, output = standard_paths(tmp_path / "missing")
    assert (
        inventory.main(
            [
                "--archives-dir",
                str(archives),
                "--extract-dir",
                str(extracted),
                "--output-dir",
                str(output),
                "--mode",
                "verify-archives",
                "--subjects",
                "01",
                "--require-complete",
            ]
        )
        != 0
    )
    archives, extracted, output = standard_paths(tmp_path / "unsafe")
    make_zip(archives / "sub-01.zip", {"../escape": b"x"})
    assert (
        inventory.main(
            [
                "--archives-dir",
                str(archives),
                "--extract-dir",
                str(extracted),
                "--output-dir",
                str(output),
                "--mode",
                "extract",
                "--subjects",
                "01",
            ]
        )
        != 0
    )


def test_outputs_are_deterministic_and_sorted(tmp_path: Path) -> None:
    archives, extracted, output1 = standard_paths(tmp_path)
    archive = make_zip(
        archives / "sub-01.zip",
        {"sub-01/z.txt": b"z", "sub-01/a.txt": b"a"},
    )
    inventory.safe_extract_archive(archive, extracted)
    output2 = tmp_path / "output2"
    for output in (output1, output2):
        inventory.run_tool(
            archives_dir=archives,
            extract_dir=extracted,
            output_dir=output,
            mode="inventory",
            expected_subjects=["01"],
        )
    for name in (
        "archive_inventory.csv",
        "record_inventory.csv",
        "inventory_summary.json",
        "inventory_report.md",
    ):
        assert (output1 / name).read_bytes() == (output2 / name).read_bytes()
    with (output1 / "record_inventory.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["relative_path"] for row in rows] == sorted(
        (row["relative_path"] for row in rows), key=str.lower
    )


def test_zero_byte_archive_is_partial(tmp_path: Path) -> None:
    archives, _, _ = standard_paths(tmp_path)
    archives.mkdir()
    (archives / "sub-01.zip").write_bytes(b"")
    row = inventory.verify_archives(archives, expected_subjects=["01"])[0]
    assert row["status"] == "partial"


def test_subject_subset_does_not_read_other_large_archives(tmp_path: Path) -> None:
    archives, _, _ = standard_paths(tmp_path)
    make_zip(archives / "sub-01.zip")
    (archives / "sub-02.zip").write_bytes(b"not inspected")
    rows = inventory.verify_archives(
        archives,
        expected_subjects=["01"],
        include_unselected_archives=False,
    )
    assert [row["filename"] for row in rows] == ["sub-01.zip"]


def test_none_annotations_and_missing_channel_locations_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "record.set"
    path.write_bytes(b"stub")

    class Raw:
        annotations = None
        ch_names = ["C3", "C4"]
        n_times = 256
        _orig_units = None
        info = {
            "nchan": 2,
            "sfreq": 256.0,
            "custom_ref_applied": "",
            "bads": [],
            "chs": [{"loc": None}, {"loc": [0.0, 0.0, 0.0]}],
        }

        def close(self) -> None:
            pass

    fake_mne = SimpleNamespace(
        io=SimpleNamespace(read_raw_eeglab=lambda *args, **kwargs: Raw())
    )
    monkeypatch.setattr(inventory.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "mne", fake_mne)
    metadata, events = inventory._read_eeglab_metadata(path)
    assert events == []
    assert metadata["annotation_count"] == 0
    assert metadata["read_status"] == "ok_with_warnings"
    assert "missing_channel_locations=2" in metadata["error"]


def test_json_events_none_is_an_empty_collection(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    path.write_text('{"events": null}', encoding="utf-8")
    assert inventory._json_events(path) == []


def test_mat_none_dims_uses_bounded_opaque_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np
    import scipy.io

    path = tmp_path / "0-Back.mat"
    path.write_bytes(b"small")

    class MatlabOpaque:
        shape = (1,)

    monkeypatch.setattr(
        scipy.io,
        "whosmat",
        lambda value: (_ for _ in ()).throw(TypeError("'NoneType' object is not iterable")),
    )
    monkeypatch.setattr(
        scipy.io,
        "loadmat",
        lambda *args, **kwargs: {"__header__": b"x", "None": MatlabOpaque()},
    )
    rows = inventory._mat_variables(path)
    assert rows == [
        {
            "event_source": "mat_opaque_fallback",
            "event_label": "None:MatlabOpaque:(1,)",
            "trigger_code": "",
            "rating_name": "",
            "behavioural_outcome": "",
        }
    ]
    assert np is not None


def test_mat_without_variables_and_empty_pair_collection_are_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scipy.io

    path = tmp_path / "empty.mat"
    path.write_bytes(b"small")
    monkeypatch.setattr(scipy.io, "whosmat", lambda value: None)
    assert inventory._mat_variables(path) == []
    assert inventory.build_file_pair_inventory([], []) == []


def test_multiple_sessions_are_counted_separately() -> None:
    records = [
        {
            "subject_id": "sub-01",
            "session_id": session,
            "task": "PVT",
            "file_type": "set",
        }
        for session in ("ses-S1", "ses-S2", "ses-S3")
    ]
    _, sessions, subjects = inventory.build_structural_inventories(records)
    assert len(sessions) == 3
    assert subjects[0]["session_count"] == 3


def test_resume_twice_is_idempotent(tmp_path: Path) -> None:
    archives, extracted, _ = standard_paths(tmp_path)
    archive = make_zip(archives / "sub-01.zip", {"task.set": b"original"})
    first = inventory.safe_extract_archive(archive, extracted, resume=True)
    second = inventory.safe_extract_archive(archive, extracted, resume=True)
    assert first[0]["status"] == "extracted"
    assert second[0]["status"] == "already_correct"


def test_debug_reraises_and_normal_cli_is_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(**kwargs: object) -> inventory.ToolResult:
        raise TypeError("synthetic fatal")

    monkeypatch.setattr(inventory, "run_tool", fail)
    base = [
        "--archives-dir",
        str(tmp_path),
        "--extract-dir",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    assert inventory.main(base) == 2
    captured = capsys.readouterr()
    assert "synthetic fatal" in captured.err
    assert "Traceback" not in captured.err
    with pytest.raises(TypeError, match="synthetic fatal"):
        inventory.main([*base, "--debug"])


def test_missing_mne_uses_explicit_fallback_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "record.set"
    path.write_bytes(b"stub")
    monkeypatch.setattr(inventory.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ModuleNotFoundError, match="mne"):
        inventory._read_eeglab_metadata(path)


def test_one_archive_conflict_does_not_prevent_later_archive(
    tmp_path: Path,
) -> None:
    archives, extracted, _ = standard_paths(tmp_path)
    first = make_zip(archives / "sub-01.zip", {"task.set": b"archive"})
    second = make_zip(archives / "sub-02.zip", {"task.set": b"second"})
    conflict = extracted / "sub-01" / "task.set"
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"changed")
    rows = [
        {
            "filename": first.name,
            "subject_id": "sub-01",
            "status": "valid",
            "zip_test_passed": "true",
        },
        {
            "filename": second.name,
            "subject_id": "sub-02",
            "status": "valid",
            "zip_test_passed": "true",
        },
    ]
    extracted_rows, errors, fatal = inventory.extract_verified_archives(
        archives, extracted, rows, resume=True
    )
    terminal = {
        row["subject_id"]: row["status"]
        for row in extracted_rows
        if row["member"] == "__archive__"
    }
    assert fatal
    assert terminal == {"sub-01": "conflict", "sub-02": "extracted"}
    assert len(errors) == 1
    assert (extracted / "sub-02" / "task.set").read_bytes() == b"second"


def test_text_behavioural_inventory_counts_missing_combinations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "KSS.txt"
    path.write_text(
        "sbj,sess,score,condition,Condition\n"
        "1,1,3,0,beginning\n"
        "1,1,4,1,end\n"
        "1,2,5,0,beginning\n",
        encoding="utf-8",
    )
    rows = inventory._text_behavioural_rows(path, "metadata/KSS.txt")
    by_metric = {row["metric_name"]: row for row in rows}
    assert by_metric["score"]["value_count"] == 3
    assert by_metric["score"]["missing_count"] == 1
    assert "beginning" in by_metric["Condition"]["data_type"]
