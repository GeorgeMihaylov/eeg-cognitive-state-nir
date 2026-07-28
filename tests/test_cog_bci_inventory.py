from __future__ import annotations

import csv
import hashlib
import json
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
                "channel_names": "C3|C4",
                "sampling_rate_hz": 256.0,
                "n_samples": 256,
                "duration_seconds": 1.0,
                "data_units": "uV",
                "event_count": 0,
                "event_types": "",
                "annotations_count": 0,
                "reference": "",
                "bad_channels": "",
                "metadata_reader": "mock",
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
