from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from bench.experiments import pm_eeg_absolute_lag as lag
from bench.experiments import pm_low_high_q3_extremes_confirmatory as low

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/pm_diagnostics/pm_eeg_absolute_lag_v1.json"


def index():
    rows = []
    for fold in range(1, 6):
        for w in range(14):
            rows.append({"sample_id": len(rows), "record_id": f"r{fold}",
                         "subject_id": f"s{fold}", "outer_fold": fold,
                         "t_start": -5 + w * 10., "t_end": 5 + w * 10.})
    return pd.DataFrame(rows)


def full_frame():
    frame = index()
    for pm in low.PM_NAMES:
        frame[f"target_{pm}"] = (frame.sample_id % 7) / 6
    return frame


@pytest.mark.parametrize("absolute_lag", lag.LAGS)
def test_actual_timestamp_sign_and_common_support(absolute_lag):
    frame = index()
    positions, common, audit = lag.build_alignment(frame)
    assert common.sum() == 50
    eeg = frame.iloc[positions[absolute_lag][common]]
    target = frame.loc[common]
    np.testing.assert_array_equal(eeg.t_start.to_numpy() - target.t_start.to_numpy(), absolute_lag)
    assert audit.loc[audit.absolute_lag_s.eq(absolute_lag), "actual_difference_s"].eq(absolute_lag).all()
    assert (lag.causal_status(absolute_lag) == "non_causal_diagnostic") == (absolute_lag == 10)
    if absolute_lag <= 0:
        assert (eeg.t_end.to_numpy() <= target.t_end.to_numpy()).all()


def test_gap_never_uses_previous_available_row():
    frame = index().loc[lambda f: f.sample_id.ne(5)].reset_index(drop=True)
    positions, common, _ = lag.build_alignment(frame)
    at_six = frame.index[frame.sample_id.eq(6)][0]
    assert positions[-10][at_six] == -1
    assert not common[at_six]


@pytest.mark.parametrize("corruption", ["duration", "subject", "fold", "duplicate", "off_grid"])
def test_reject_bad_alignment(corruption):
    frame = index()
    if corruption == "duration":
        frame.loc[3, "t_end"] += 1
    elif corruption == "subject":
        frame.loc[3, "subject_id"] = "other"
    elif corruption == "fold":
        frame.loc[3, "outer_fold"] = 2
    elif corruption == "duplicate":
        frame.loc[3, ["t_start", "t_end"]] = frame.loc[4, ["t_start", "t_end"]].to_numpy()
    else:
        frame.loc[3, ["t_start", "t_end"]] += .5
    with pytest.raises((ValueError, RuntimeError)):
        lag.build_alignment(frame)


def test_train_only_thresholds_and_same_evaluation_examples():
    full = full_frame()
    positions, common, _ = lag.build_alignment(full)
    cohorts, audit = lag.build_common_cohorts(full, positions, common)
    thresholds, _ = low.build_threshold_audit(full, cohorts, [1, 2, 3, 4, 5])
    changed = full.copy()
    changed.loc[changed.outer_fold.eq(1), "target_attention"] = 10000
    changed_cohorts, _ = lag.build_common_cohorts(changed, positions, common)
    other, _ = low.build_threshold_audit(changed, changed_cohorts, [1, 2, 3, 4, 5])
    assert thresholds[(1, "attention")] == other[(1, "attention")]
    q = np.quantile(full.loc[full.outer_fold.ne(1), "target_attention"], [1/3, 2/3])
    assert thresholds[(1, "attention")].q_low == q[0]
    assert thresholds[(1, "attention")].q_high == q[1]
    cohort = cohorts["attention"]
    labels = thresholds[(1, "attention")].transform(cohort.continuous_target)
    retained = np.isfinite(labels) & cohort.outer_fold.eq(1)
    hashes = []
    for absolute_lag in lag.LAGS:
        assert len(cohort.loc[retained, f"{lag.condition(absolute_lag)}_feature_position"]) == retained.sum()
        hashes.append(low._sample_hash(cohort.loc[retained, "target_sample_id"]))
    assert len(set(hashes)) == 1
    assert audit.groupby(["target", "fold", "partition"]).n_common_support.nunique().eq(1).all()


def test_attention_complete_cases_are_target_specific():
    full = full_frame()
    full.loc[5, "target_attention"] = np.nan
    positions, common, _ = lag.build_alignment(full)
    cohorts, _ = lag.build_common_cohorts(full, positions, common)
    assert len(cohorts["attention"]) == len(cohorts["focus"]) - 1


@pytest.mark.parametrize("field,value", [("reference_lag_s", 0), ("absolute_lags_s", [-10, 0]),
                                         ("threshold_fit_scope", "all_subjects")])
def test_frozen_contract_rejects_mutation(tmp_path, field, value):
    config = lag.load_config(CONFIG)
    config[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        lag.load_config(path)


def test_paired_delta_uses_minus10_and_self_delta_is_zero():
    frame = pd.DataFrame([{"target": "attention", "fold": 1, "absolute_lag_s": l,
                           **{m: .6 + l / 1000 for m in lag.METRICS}} for l in lag.LAGS])
    paired = lag.paired_deltas(frame, ["target", "fold"])
    assert paired.loc[paired.absolute_lag_s.eq(-10), "delta_balanced_accuracy_vs_minus10"].iloc[0] == 0
    assert paired.loc[paired.absolute_lag_s.eq(0), "delta_balanced_accuracy_vs_minus10"].iloc[0] == pytest.approx(.01)
    with pytest.raises(ValueError, match="Missing"):
        lag.paired_deltas(frame.loc[frame.absolute_lag_s.ne(-10)], ["target", "fold"])


def synthetic_base(tmp_path):
    full = full_frame()
    names = [f"f{i}" for i in range(371)]
    return low.ProtocolContext(
        root=ROOT, output_dir=tmp_path, config={},
        matrix=np.random.default_rng(42).normal(size=(len(full), 371)).astype("float32"),
        feature_index=index(), feature_names=names, cache_manifest={}, cache_identity={"n_features": 371},
        full=full, temporal_pairing=pd.DataFrame(), temporal_pairing_summary={"matched_target_rows": 30806},
        cohorts={}, cohort_summary={}, transforms={}, threshold_audit=pd.DataFrame(),
        protocol={"fixed_fold_hash": lag.FOLD_HASH}, run_matrix=pd.DataFrame())


def test_deterministic_preflight_execution_resume_and_artifacts(tmp_path, monkeypatch):
    base = synthetic_base(tmp_path)
    monkeypatch.setattr(low, "prepare_protocol", lambda *a, **k: base)
    config = lag.load_config(CONFIG)
    context = lag.prepare_protocol(config, root=ROOT, feature_cache_dir=tmp_path, smoke=True)
    again = lag.prepare_protocol(config, root=ROOT, feature_cache_dir=tmp_path, smoke=True)
    assert context.base.protocol == again.base.protocol
    pd.testing.assert_frame_equal(context.base.run_matrix, again.base.run_matrix)
    full = lag.prepare_protocol(config, root=ROOT, feature_cache_dir=tmp_path)
    assert len(full.base.run_matrix) == 175
    assert len(context.base.run_matrix) == 5
    assert lag.write_dry_run(context)["training_executed"] is False
    assert not (tmp_path / "runs").exists()
    result = lag.run_experiment(context)
    assert result["executed"] == 5
    with pytest.raises(ValueError, match="another protocol"):
        lag.write_dry_run(full)
    assert lag.run_experiment(context)["resumed"] == 5
    overall = pd.read_csv(tmp_path / "overall.csv")
    assert overall.loc[overall.absolute_lag_s.eq(-10), "delta_balanced_accuracy_vs_minus10"].iloc[0] == 0
    predictions = [pd.read_parquet(p) for p in (tmp_path / "runs").glob("*/predictions.parquet")]
    for p in predictions[1:]:
        pd.testing.assert_series_equal(p.target_sample_id, predictions[0].target_sample_id)
        pd.testing.assert_series_equal(p.y_true, predictions[0].y_true)
    path = next((tmp_path / "runs").glob("*/participant_metrics.csv"))
    path.write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupted"):
        lag.run_experiment(context)


def test_participant_bootstrap_deterministic():
    values = [.1, -.2, .3, .05]
    assert lag.subject_bootstrap_interval(values) == lag.subject_bootstrap_interval(values)
    assert lag.subject_bootstrap_interval(values)["n_subjects"] == 4
