from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from bench.experiments.xgboost_robust_shrinkage_personalization import (
    ALPHA_CANDIDATES,
    RobustShrinkagePersonalizationExperiment,
    TrainBundle,
    aggregate_candidate_scores,
    build_full_plan,
    evaluate_alignment_candidates,
    protocol_hash,
    select_alpha,
    validate_config,
)
from model_zoo.ML.xgboost_personalization import xgboost_state_sha256


CONFIG_PATH = Path(
    "experiments/calibration/xgboost_robust_shrinkage_personalization_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_config_and_full_plan_are_locked_and_output_independent() -> None:
    config = validate_config(_config())
    plan, first_protocol, first_plan = build_full_plan(config)

    assert len(plan) == 175
    assert plan["phase"].eq("inner_model").sum() == 140
    assert plan["phase"].eq("outer_evaluation").sum() == 35
    assert not plan.duplicated("unit_id").any()

    relocated = deepcopy(config)
    relocated["experiment"]["output_dir"] = "some/other/runtime/path"
    second_plan, second_protocol, second_hash = build_full_plan(relocated)
    assert first_protocol == second_protocol == protocol_hash(config)
    assert first_plan == second_hash
    pd.testing.assert_frame_equal(plan, second_plan)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["alignment"].update({"alpha_candidates": [0, 1]}),
        lambda value: value["model"]["params"].update({"n_estimators": 201}),
        lambda value: value["scope"].update({"calibration_budget_fraction": 0.1}),
        lambda value: value["protocol"].update({"outer_test_labels_used_for_selection": True}),
    ],
)
def test_locked_scientific_contract_rejects_mutation(mutation) -> None:
    config = _config()
    mutation(config)
    with pytest.raises(ValueError):
        validate_config(config)


def test_aggregation_is_pm_then_participant_then_inner_fold() -> None:
    rows = []
    # Fold 2 has two participants; fold 3 has one. Equal inner-fold weighting
    # means the many-participant fold cannot dominate the final score.
    values = {
        (2, "a", "attention"): (0.9, 0.8),
        (2, "a", "focus"): (0.7, 0.6),
        (2, "b", "attention"): (0.2, 0.3),
        (2, "b", "focus"): (0.4, 0.5),
        (3, "c", "attention"): (0.6, 0.7),
        (3, "c", "focus"): (0.8, 0.9),
    }
    for (fold, subject, pm), (macro, balanced) in values.items():
        rows.append({
            "inner_pseudo_test_fold": fold, "subject_id": subject, "pm": pm,
            "alpha": 0.25, "macro_f1": macro,
            "balanced_accuracy": balanced, "accuracy": macro,
            "weighted_f1": macro,
        })
    inner, summary = aggregate_candidate_scores(pd.DataFrame(rows))

    assert len(inner) == 2
    # fold 2: mean([mean(.9,.7), mean(.2,.4)])=.55; fold 3=.70
    assert summary.loc[0, "macro_f1"] == pytest.approx((0.55 + 0.70) / 2)
    assert summary.loc[0, "inner_fold_count"] == 2


def test_selection_tie_breaks_balanced_accuracy_then_smaller_alpha() -> None:
    summary = pd.DataFrame({
        "alpha": [0.0, 0.1, 0.25],
        "macro_f1": [0.5, 0.5 + 5e-13, 0.49],
        "balanced_accuracy": [0.6, 0.6 + 5e-13, 0.9],
    })
    selected = select_alpha(summary, tolerance=1e-12)
    assert selected["selected_alpha"] == 0.0
    assert selected["tie_break_applied"] is True


def test_candidate_evaluation_reuses_one_frozen_booster_for_all_alpha() -> None:
    rng = np.random.default_rng(42)
    X_reference = rng.normal(size=(90, 5))
    y_reference = np.repeat(np.arange(3), 30)
    model = XGBClassifier(
        n_estimators=4, max_depth=2, n_jobs=1, random_state=42,
    ).fit(X_reference, y_reference)
    before = xgboost_state_sha256(model)
    X_calibration = rng.normal(loc=2.0, scale=1.5, size=(15, 5))
    X_evaluation = rng.normal(loc=2.0, scale=1.5, size=(18, 5))
    y_evaluation = np.tile(np.arange(3), 6)

    rows, audit = evaluate_alignment_candidates(
        model, X_reference=X_reference, X_calibration=X_calibration,
        X_evaluation=X_evaluation, y_evaluation=y_evaluation,
    )

    assert [row["alpha"] for row in rows] == list(ALPHA_CANDIDATES)
    assert audit["reference_n_samples"] == 90
    assert audit["calibration_n_samples"] == 15
    assert audit["booster_hash_before"] == audit["booster_hash_after"] == before
    assert xgboost_state_sha256(model) == before


def test_inner_transform_fits_only_non_outer_non_pseudo_rows(tmp_path: Path) -> None:
    experiment = RobustShrinkagePersonalizationExperiment(
        CONFIG_PATH, output_dir=tmp_path,
    )
    rows = []
    for fold in range(1, 6):
        for index in range(6):
            rows.append({
                "sample_id": f"f{fold}-{index}", "outer_fold": fold,
                "target_value": float(fold * 10 + index),
            })
    frame = pd.DataFrame(rows)
    mask = ~frame["outer_fold"].isin([1, 2])
    _, manifest = experiment._inner_transform(
        "focus", 1, 2, frame, mask.to_numpy(),
    )

    assert manifest["fit_scope"] == "inner_train_only"
    assert manifest["nested_real_outer_fold"] == 1
    assert manifest["inner_pseudo_test_fold"] == 2
    assert manifest["fit_sample_count"] == 18


def test_inner_model_resume_requires_current_specification(tmp_path: Path) -> None:
    experiment = RobustShrinkagePersonalizationExperiment(
        CONFIG_PATH, output_dir=tmp_path,
    )
    experiment.config["model"]["params"] = {
        "n_estimators": 2, "n_jobs": 1, "random_state": 42,
    }
    sample_ids = np.asarray([f"s{fold}-{index}" for fold in (2, 3, 4, 5) for index in range(9)])
    folds = np.repeat([2, 3, 4, 5], 9)
    subjects = np.asarray([f"p{fold}" for fold in (2, 3, 4, 5) for _ in range(9)])
    rng = np.random.default_rng(7)
    bundle = TrainBundle(
        X=rng.normal(size=(36, 4)), sample_ids=sample_ids,
        subjects=subjects, fixed_folds=folds,
        feature_names=("a", "b", "c", "d"),
    )
    frame = pd.DataFrame({
        "sample_id": sample_ids, "outer_fold": folds,
        "target_value": np.linspace(0.0, 1.0, 36),
    })
    fit_mask = frame["outer_fold"].isin([3, 4, 5])
    transform, manifest = experiment._inner_transform(
        "focus", 1, 2, frame, fit_mask.to_numpy(),
    )

    first, audit_first = experiment._inner_model(
        outer_fold=1, pseudo_fold=2, pm="focus", bundle=bundle,
        frame=frame, transform=transform, target_manifest=manifest,
        resume=False,
    )
    second, audit_second = experiment._inner_model(
        outer_fold=1, pseudo_fold=2, pm="focus", bundle=bundle,
        frame=frame, transform=transform, target_manifest=manifest,
        resume=True,
    )
    assert audit_first["resumed"] is False
    assert audit_second["resumed"] is True
    assert audit_first["specification_hash"] == audit_second["specification_hash"]
    assert xgboost_state_sha256(first) == xgboost_state_sha256(second)

    manifest_path = tmp_path / "fold_01/inner_models/pseudo_02/focus/manifest.json"
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved["specification_hash"] = "stale"
    manifest_path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Stale inner model cache"):
        experiment._inner_model(
            outer_fold=1, pseudo_fold=2, pm="focus", bundle=bundle,
            frame=frame, transform=transform, target_manifest=manifest,
            resume=True,
        )
