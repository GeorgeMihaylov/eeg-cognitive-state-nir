from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

import cli
import bench.analysis.label_definition_sensitivity as sensitivity_module
from bench.analysis.diagnostic_baselines import (
    align_with_canonical_predictions,
    assign_subject_folds,
)
from bench.analysis.label_definition_sensitivity import (
    GLOBAL_LABEL_COLUMN,
    SAFE_LABEL_COLUMN,
    LabelDefinitionSensitivity,
    apply_finite_thresholds,
    build_cross_fitted_labels,
    label_comparison_metrics,
)
from bench.analysis.temporal_target_structure import (
    make_lag_pairs,
    prepare_temporal_frame,
)


def _source_frame(n_subjects: int = 10, windows: int = 25) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_index in range(n_subjects):
        source = "gpn_data" if subject_index % 2 == 0 else "Old_EEG"
        for window in range(windows):
            rows.append(
                {
                    "source": source,
                    "subject_id": f"subject_{subject_index:02d}",
                    "record_id": f"record_{subject_index:02d}",
                    "t_start": float(window * 10),
                    "t_end": float((window + 1) * 10),
                    # Identical within-subject distributions make fold thresholds stable.
                    "target_focus": float(window / (windows - 1)),
                }
            )
    frame = pd.DataFrame(rows)
    labels, edges = pd.qcut(
        frame["target_focus"], q=5, labels=False, retbins=True, duplicates="drop"
    )
    assert len(edges) == 6
    frame["label_q5"] = labels.astype(int)
    return frame


def _folded() -> tuple[pd.DataFrame, list[float]]:
    source = _source_frame()
    prepared = prepare_temporal_frame(source)
    folded, _ = assign_subject_folds(prepared, n_splits=5)
    _, edges = pd.qcut(
        source["target_focus"], q=5, labels=False, retbins=True, duplicates="drop"
    )
    return folded, edges[1:-1].astype(float).tolist()


def test_outer_test_values_do_not_affect_their_fold_thresholds() -> None:
    folded, global_thresholds = _folded()
    _, first = build_cross_fitted_labels(
        folded, global_thresholds=global_thresholds
    )
    mutated = folded.copy()
    test_mask = mutated["outer_fold"] == 1
    mutated.loc[test_mask, "target_focus"] = np.linspace(100.0, 200.0, test_mask.sum())
    _, second = build_cross_fitted_labels(
        mutated, global_thresholds=global_thresholds
    )
    first_fold = first["folds"][0]
    second_fold = second["folds"][0]
    assert first_fold["test_subjects"] == second_fold["test_subjects"]
    for name in ("q20", "q40", "q60", "q80"):
        assert first_fold[name] == second_fold[name]


def test_thresholds_use_only_outer_train_subjects_and_cover_every_sample() -> None:
    folded, global_thresholds = _folded()
    legacy_before = folded["label_q5"].copy(deep=True)
    cross_fitted, thresholds = build_cross_fitted_labels(
        folded, global_thresholds=global_thresholds
    )
    assert thresholds["all_folds_valid"] is True
    assert len(cross_fitted) == len(folded)
    assert cross_fitted["sample_id"].is_unique
    assert set(cross_fitted["sample_id"]) == set(folded["sample_id"])
    assert sorted(cross_fitted[SAFE_LABEL_COLUMN].unique().tolist()) == [0, 1, 2, 3, 4]
    pdt.assert_series_equal(folded["label_q5"], legacy_before)

    for row in thresholds["folds"]:
        fold = int(row["fold"])
        train = folded.loc[folded["outer_fold"] != fold]
        test = folded.loc[folded["outer_fold"] == fold]
        assert row["subject_overlap"] == []
        assert set(row["train_subjects"]) == set(train["subject_id"])
        assert set(row["test_subjects"]) == set(test["subject_id"])
        expected = np.quantile(
            train["target_focus"].to_numpy(), [0.2, 0.4, 0.6, 0.8], method="linear"
        )
        np.testing.assert_allclose(
            [row["q20"], row["q40"], row["q60"], row["q80"]], expected
        )
        assert row["train_class_balance"]["all_classes_present"] is True
        assert row["test_class_balance"]["all_classes_present"] is True


def test_finite_extreme_values_receive_classes_and_duplicate_edges_are_invalid() -> None:
    labels = apply_finite_thresholds(
        [-1e12, 0.2, 0.3, 0.4, 0.5, 1e12], [0.2, 0.3, 0.4, 0.5]
    )
    assert labels.tolist() == [0, 0, 1, 2, 3, 4]

    folded, global_thresholds = _folded()
    constant = folded.copy()
    constant["target_focus"] = 0.5
    cross_fitted, thresholds = build_cross_fitted_labels(
        constant, global_thresholds=global_thresholds
    )
    assert cross_fitted.empty
    assert thresholds["all_folds_valid"] is False
    assert thresholds["invalid_folds"] == [1, 2, 3, 4, 5]
    assert all(
        row["status"] == "invalid_non_unique_thresholds"
        for row in thresholds["folds"]
    )


def test_comparison_is_deterministic_and_temporal_pairs_stay_in_records() -> None:
    folded, global_thresholds = _folded()
    first, _ = build_cross_fitted_labels(
        folded, global_thresholds=global_thresholds
    )
    second, _ = build_cross_fitted_labels(
        folded.sample(frac=1.0, random_state=11).reset_index(drop=True),
        global_thresholds=global_thresholds,
    )
    columns = [
        "sample_id",
        GLOBAL_LABEL_COLUMN,
        SAFE_LABEL_COLUMN,
        "label_changed",
        "absolute_label_shift",
    ]
    pdt.assert_frame_equal(first[columns], second[columns])
    assert label_comparison_metrics(first)["changed_fraction"] == 0.0

    pairs = make_lag_pairs(first, value_col=SAFE_LABEL_COLUMN, lag=1)
    previous = first.set_index("sample_id").loc[pairs["previous_sample_id"]]
    for column in ("source", "subject_id", "record_id"):
        assert previous[column].to_numpy().tolist() == pairs[column].tolist()


def test_canonical_folds_align_exactly(tmp_path: Path) -> None:
    folded, _ = _folded()
    reference = folded[
        ["sample_id", "outer_fold", "subject_id", "label_q5"]
    ].rename(columns={"outer_fold": "fold", "label_q5": "y_true"})
    path = tmp_path / "canonical.parquet"
    reference.to_parquet(path, index=False)
    alignment = align_with_canonical_predictions(folded, path)
    assert alignment["exact_match"] is True
    assert all(value == 0 for value in alignment["mismatches"].values())


def test_plan_only_writes_nothing_and_preserves_gitignore(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "generated"
    report = tmp_path / "reports" / "report.md"
    summary = tmp_path / "reports" / "summary.json"
    spec_path = tmp_path / "spec.yaml"
    spec = {
        "analysis": {
            "data_path": str(tmp_path / "missing.parquet"),
            "output_dir": str(output_dir),
            "report_path": str(report),
            "summary_path": str(summary),
            "canonical_reference_predictions": str(tmp_path / "reference.parquet"),
            "global_internal_thresholds": [0.2, 0.4, 0.6, 0.8],
        }
    }
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    gitignore_before = gitignore.read_bytes()
    cli.main(["--label-definition-sensitivity", str(spec_path), "--plan-only"])
    output = capsys.readouterr().out
    assert "Models trained: 0" in output
    assert "Writes performed: no" in output
    assert not output_dir.exists()
    assert not report.exists()
    assert not summary.exists()
    assert gitignore.read_bytes() == gitignore_before


def test_full_audit_does_not_train_models_or_modify_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_frame()
    data_path = tmp_path / "dataset.parquet"
    source.to_parquet(data_path, index=False)
    before = sha256(data_path.read_bytes()).hexdigest()
    prepared = prepare_temporal_frame(source)
    folded, _ = assign_subject_folds(prepared, n_splits=5)
    reference = folded[
        ["sample_id", "outer_fold", "subject_id", "label_q5"]
    ].rename(columns={"outer_fold": "fold", "label_q5": "y_true"})
    reference_path = tmp_path / "reference.parquet"
    reference.to_parquet(reference_path, index=False)
    _, edges = pd.qcut(
        source["target_focus"], q=5, labels=False, retbins=True, duplicates="drop"
    )
    output_dir = tmp_path / "generated"
    reports = tmp_path / "reports"
    spec_path = tmp_path / "spec.yaml"
    spec = {
        "analysis": {
            "name": "synthetic_sensitivity",
            "data_path": str(data_path),
            "output_dir": str(output_dir),
            "report_path": str(reports / "report.md"),
            "summary_path": str(reports / "summary.json"),
            "canonical_reference_predictions": str(reference_path),
            "global_internal_thresholds": edges[1:-1].astype(float).tolist(),
            "expected_supervised_rows": len(source),
            "repeat_diagnostics_if_changed_fraction_exceeds": 0.05,
        }
    }
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Diagnostic models must not run below the 5% condition")

    monkeypatch.setattr(sensitivity_module, "run_diagnostic_baselines", forbidden)
    result = LabelDefinitionSensitivity(spec_path).execute()
    assert result["models_trained"] == 0
    assert result["diagnostic_baselines"]["status"] == "not_repeated"
    assert result["legacy_label_modified"] is False
    assert result["canonical_alignment"]["exact_match"] is True
    assert result["input_sha256"] == result["input_sha256_after"] == before
    assert result["input_modified"] is False
    assert (output_dir / "fold_quantile_thresholds.json").is_file()
    comparison = pd.read_parquet(
        output_dir / "cross_fitted_label_comparison.parquet"
    )
    assert comparison["sample_id"].is_unique
    assert len(comparison) == len(source)
    assert (output_dir / "cross_fitted_temporal_statistics.json").is_file()
    assert sha256(data_path.read_bytes()).hexdigest() == before
