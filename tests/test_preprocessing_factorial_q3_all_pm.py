from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from bench.experiments import preprocessing_factorial_q3_all_pm as experiment
from bench.experiments.preprocessing_factorial_q3_all_pm import (
    FACTOR_CONTRASTS,
    FOLDS,
    PM_NAMES,
    VARIANTS,
    FactorialRunSpec,
    _factor_effects,
    _split_identity,
    _target_fold_plan,
    build_run_matrix,
    load_config,
    resumable_summary,
    run_specification_hash,
    smoke_run_matrix,
    target_id,
)


CONFIG = Path(
    "experiments/preprocessing/preprocessing_factorial_q3_all_pm_v1.json"
)


def test_config_and_exact_run_matrices() -> None:
    config = load_config(CONFIG)
    full = build_run_matrix(config)
    smoke = smoke_run_matrix(config)
    assert len(full) == 8 * 7 * 5 == 280
    assert len({spec.run_id for spec in full}) == 280
    assert len(smoke) == 2
    assert {(spec.outer_fold, spec.pm, spec.variant) for spec in smoke} == {
        (1, "attention", "A"),
        (1, "attention", "H"),
    }
    serialized = json.dumps(config).lower()
    assert "label_q5" not in serialized
    assert "regression" not in serialized
    assert config["validation"]["group_column"] == "record_group_id"


def test_factorial_mapping_and_target_contract() -> None:
    config = load_config(CONFIG)
    mappings = {row["variant"]: row for row in config["preprocessing_variants"]}
    assert {
        key: tuple(mappings[key][name] for name in ("bandpass", "notch", "car"))
        for key in VARIANTS
    } == {
        "A": (False, False, False),
        "B": (True, False, False),
        "C": (False, True, False),
        "D": (False, False, True),
        "E": (True, True, False),
        "F": (True, False, True),
        "G": (False, True, True),
        "H": (True, True, True),
    }
    assert set(FACTOR_CONTRASTS) == {"bandpass", "notch", "car"}
    for pm in PM_NAMES:
        spec = experiment.get_target_spec(target_id(pm))
        assert spec.task_type == "classification"
        assert spec.transform_policy == "fold_local_quantile_q3"


def test_fold_local_q3_is_fit_on_outer_train_only(
    monkeypatch,
) -> None:
    canonical_rows = []
    sample_id = 0
    for fold in FOLDS:
        for subject_offset in range(2):
            subject = f"s{fold}_{subject_offset}"
            for window in range(9):
                canonical_rows.append({
                    "sample_id": sample_id,
                    "subject_id": subject,
                    "record_id": f"r_{subject}",
                    "record_group_id": f"g_{subject}",
                    "outer_fold": fold,
                })
                sample_id += 1
    canonical = pd.DataFrame(canonical_rows)
    targets = canonical[["sample_id", "subject_id", "record_id"]].copy()
    for offset, pm in enumerate(PM_NAMES):
        targets[f"target_{pm}"] = (
            np.arange(len(targets), dtype=np.float32) + offset
        ) / float(len(targets))
    expected = {pm: len(targets) for pm in PM_NAMES}
    monkeypatch.setattr(experiment, "EXPECTED_COMPLETE_CASES", expected)
    monkeypatch.setattr(experiment, "_load_targets", lambda _: targets)
    audit, transforms, counts = _target_fold_plan({}, canonical)
    assert counts == expected
    assert len(audit) == 35
    assert len(transforms) == 35
    assert audit.q3_fit_scope.eq("outer_train_only").all()
    assert audit.all_variants_target_mask_identical.all()
    for fold in FOLDS:
        assert audit.loc[audit.outer_fold.eq(fold), "test_subjects"].eq(2).all()
    for manifest in transforms.values():
        assert manifest["fit_scope"] == "outer_train_only"
        assert manifest["actual_class_count"] == 3


def test_specification_hash_is_deterministic_and_protocol_bound() -> None:
    spec = FactorialRunSpec(1, "attention", "A", 42)
    assert run_specification_hash(spec, "p") == run_specification_hash(spec, "p")
    assert run_specification_hash(spec, "p") != run_specification_hash(spec, "q")
    assert run_specification_hash(spec, "p") != run_specification_hash(
        FactorialRunSpec(1, "attention", "H", 42), "p"
    )


def test_resume_requires_complete_current_identity_and_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "predictions.parquet"
    artifact.write_bytes(b"synthetic")
    path = tmp_path / "run_summary.json"
    current = {
        "status": "complete",
        "specification_hash": "current",
        "required_artifacts": [str(artifact)],
    }
    path.write_text(json.dumps(current), encoding="utf-8")
    assert resumable_summary(path, "current") == current
    assert resumable_summary(path, "stale") is None
    current["status"] = "failed"
    path.write_text(json.dumps(current), encoding="utf-8")
    assert resumable_summary(path, "current") is None


def test_runtime_split_identity_uses_current_task_split_fields() -> None:
    split = SimpleNamespace(
        sample_id_train=np.asarray([1, 2]),
        sample_id_test=np.asarray([3]),
        y_train=np.asarray([0, 1]),
        y_test=np.asarray([2]),
        subject_train=np.asarray(["train", "train"]),
        subject_test=np.asarray(["test"]),
    )
    identity = _split_identity(split)
    assert set(identity) == {
        "train_sample_id_hash", "test_sample_id_hash",
        "train_target_hash", "test_target_hash",
        "train_subject_hash", "test_subject_hash",
    }
    assert identity["train_subject_hash"] != identity["test_subject_hash"]


def test_factor_effects_pair_within_participant_pm_before_inference() -> None:
    rows = []
    factor_bonus = {
        "A": 0.00, "B": 0.01, "C": 0.02, "D": 0.03,
        "E": 0.03, "F": 0.04, "G": 0.05, "H": 0.06,
    }
    for subject_offset, subject in enumerate(("s1", "s2", "s3")):
        for pm in PM_NAMES:
            for variant in VARIANTS:
                score = 0.4 + factor_bonus[variant] + subject_offset * 0.001
                rows.append({
                    "outer_fold": subject_offset + 1,
                    "subject_id": subject,
                    "pm": pm,
                    "variant": variant,
                    "accuracy": score,
                    "balanced_accuracy": score,
                    "macro_f1": score,
                    "weighted_f1": score,
                })
    effects, per_pm = _factor_effects(pd.DataFrame(rows))
    assert len(effects) == 3 * 2
    assert len(per_pm) == 3 * 2 * 7
    assert effects.participants.eq(3).all()
    assert effects.holm_p.between(0, 1).all()
    expected = {"bandpass": 0.01, "notch": 0.02, "car": 0.03}
    for factor, delta in expected.items():
        actual = effects.loc[
            effects.factor.eq(factor) & effects.metric.eq("macro_f1"), "mean_delta"
        ].item()
        assert np.isclose(actual, delta)
