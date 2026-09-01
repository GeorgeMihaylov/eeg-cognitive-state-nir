from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

import cli
from bench.analysis.label_target_audit import LabelTargetAudit


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _synthetic_frame() -> pd.DataFrame:
    targets = np.linspace(0.05, 0.95, 20)
    sources = ["Old"] * 10 + ["gpn"] * 10
    subjects = []
    for index in range(20):
        if index in {0, 1, 10, 11}:
            subjects.append("shared")
        elif index < 10:
            subjects.append("old_only")
        else:
            subjects.append("gpn_only")
    frame = pd.DataFrame(
        {
            "source": sources,
            "subject_id": subjects,
            "record_id": [f"record_{source}_{index // 4}" for index, source in enumerate(sources)],
            "t_start": np.arange(20, dtype=float) * 10.0,
            "t_end": np.arange(1, 21, dtype=float) * 10.0,
            "PM.Focus.Scaled__mean": targets,
            "target_focus": targets,
            "target_main": targets,
        }
    )
    frame["label_q5"] = pd.qcut(
        frame["target_focus"], q=5, labels=False, duplicates="drop"
    ).astype(float)
    missing = pd.DataFrame(
        [{
            "source": "Old",
            "subject_id": "no_target",
            "record_id": "missing_record",
            "t_start": 0.0,
            "t_end": 10.0,
            "PM.Focus.Scaled__mean": np.nan,
            "target_focus": np.nan,
            "target_main": np.nan,
            "label_q5": np.nan,
        }]
    )
    return pd.concat([frame, missing], ignore_index=True)


def _write_case(tmp_path: Path, frame: pd.DataFrame | None = None) -> dict[str, Path]:
    frame = _synthetic_frame() if frame is None else frame
    data_path = tmp_path / "dataset.parquet"
    output_dir = tmp_path / "generated"
    report_path = tmp_path / "reports" / "audit.md"
    summary_path = tmp_path / "reports" / "audit.json"
    spec_path = tmp_path / "audit.yaml"
    frame.to_parquet(data_path, index=False)
    spec = {
        "audit": {
            "name": "synthetic_label_audit",
            "data_path": str(data_path),
            "output_dir": str(output_dir),
            "report_path": str(report_path),
            "summary_path": str(summary_path),
            "target_column": "target_focus",
            "label_column": "label_q5",
            "n_classes": 5,
            "expected_rows": len(frame),
            "expected_supervised_rows": int(frame["target_focus"].notna().sum()),
            "expected_subjects": int(frame["subject_id"].nunique()),
            "expected_records": int(
                frame[["source", "subject_id", "record_id"]].drop_duplicates().shape[0]
            ),
        }
    }
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return {
        "data": data_path,
        "output": output_dir,
        "report": report_path,
        "summary": summary_path,
        "spec": spec_path,
    }


def test_audit_reconstructs_labels_counts_and_class_ranges(tmp_path: Path) -> None:
    paths = _write_case(tmp_path)
    result = LabelTargetAudit(paths["spec"]).execute()

    assert result["rows"] == 21
    assert result["supervised_rows"] == 20
    assert result["target_focus_non_null"] == 20
    assert result["label_q5_non_null"] == 20
    assert result["label_q5"]["stored_labels_match_recomputed_global_qcut"] is True
    assert set(result["label_q5"]["counterfactual_source_specific_boundaries"]) == {
        "Old",
        "gpn",
    }
    assert result["models_trained"] == 0
    classes = pd.read_parquet(paths["output"] / "target_class_statistics.parquet")
    assert classes["class_id"].tolist() == [0, 1, 2, 3, 4]
    assert classes["windows"].tolist() == [4, 4, 4, 4, 4]
    boundaries = result["label_q5"]["global_quantile_boundaries"]
    for row in classes.itertuples(index=False):
        assert row.target_focus_min >= boundaries[row.class_id]
        assert row.target_focus_max <= boundaries[row.class_id + 1]


def test_missing_targets_and_cross_source_people_are_counted(tmp_path: Path) -> None:
    paths = _write_case(tmp_path)
    result = LabelTargetAudit(paths["spec"]).execute()

    assert result["label_q5"]["missing_target_rows"] == 1
    assert result["subjects"] == 4
    assert result["supervised_subjects"] == 3
    assert result["subjects_in_both_sources"] == 1
    assert result["subjects_in_one_source"] == 2
    subjects = pd.read_parquet(paths["output"] / "subject_target_statistics.parquet")
    shared = subjects.loc[subjects["subject_id"] == "shared"].iloc[0]
    assert json.loads(shared["source_membership"]) == ["Old", "gpn"]
    assert 0.0 < shared["majority_class_fraction"] <= 1.0
    assert set(json.loads(shared["class_counts"])) <= {"0", "1", "2", "3", "4"}


def test_unknown_or_non_global_labels_are_rejected(tmp_path: Path) -> None:
    frame = _synthetic_frame()
    frame.loc[0, "label_q5"] = 9
    paths = _write_case(tmp_path, frame)
    with pytest.raises(ValueError, match="classes are"):
        LabelTargetAudit(paths["spec"]).execute()


def test_audit_is_deterministic_and_does_not_modify_input(tmp_path: Path) -> None:
    paths = _write_case(tmp_path)
    before = _file_hash(paths["data"])
    audit = LabelTargetAudit(paths["spec"])
    first = audit.execute()
    first_report = paths["report"].read_bytes()
    first_summary = paths["summary"].read_bytes()
    first_subjects = pd.read_parquet(paths["output"] / "subject_target_statistics.parquet")

    second = audit.execute()
    assert first == second
    assert first_report == paths["report"].read_bytes()
    assert first_summary == paths["summary"].read_bytes()
    pdt.assert_frame_equal(
        first_subjects,
        pd.read_parquet(paths["output"] / "subject_target_statistics.parquet"),
    )
    assert before == _file_hash(paths["data"])
    assert first["input_sha256"] == first["input_sha256_after"] == before
    assert first["input_modified"] is False


def test_audit_never_constructs_a_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _write_case(tmp_path)
    import cogstate.model_zoo.factory

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("model construction is forbidden in label audit")

    monkeypatch.setattr(cogstate.model_zoo.factory, "build_model", forbidden)
    result = LabelTargetAudit(paths["spec"]).execute()
    assert result["analysis_only"] is True
    assert result["models_trained"] == 0


def test_plan_only_writes_nothing_and_preserves_gitignore(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_case(tmp_path)
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    gitignore_before = gitignore.read_bytes()
    report_override = tmp_path / "unused-output"

    cli.main(
        [
            "--label-target-audit",
            str(paths["spec"]),
            "--output-dir",
            str(report_override),
            "--plan-only",
        ]
    )
    output = capsys.readouterr().out
    assert "Models trained: 0" in output
    assert "Writes performed: no" in output
    assert not paths["output"].exists()
    assert not paths["report"].exists()
    assert not paths["summary"].exists()
    assert not report_override.exists()
    assert gitignore.read_bytes() == gitignore_before


def test_expected_supervised_count_is_enforced(tmp_path: Path) -> None:
    paths = _write_case(tmp_path)
    spec = yaml.safe_load(paths["spec"].read_text(encoding="utf-8"))
    spec["audit"]["expected_supervised_rows"] = 999
    paths["spec"].write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_supervised_rows"):
        LabelTargetAudit(paths["spec"]).execute()
