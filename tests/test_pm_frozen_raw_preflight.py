from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.analysis.pm_frozen_raw_preflight import audit_loaded_inputs, main, sample_pairs
from bench.datasets.channel_contracts import PROJECT_EMOTIV_CHANNEL_ORDER
from bench.datasets.raw_preprocessing import DEFAULT_RAW_PREPROCESSING, raw_preprocessing_hash
from bench.experiments.pm_eeg_lag_confirmatory import build_previous_window_pairing, stable_hash
from bench.features.cogstate_feature_cache import load_feature_profile, sample_id_universe_hash


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(tmp_path):
    channels = list(PROJECT_EMOTIV_CHANNEL_ORDER)
    preprocessing_hash = raw_preprocessing_hash(DEFAULT_RAW_PREPROCESSING, channels=channels)
    rows = []
    for fold in range(1, 6):
        for person in range(2):
            subject = f"s{fold}-{person}"
            path = tmp_path / f"{subject}.npy"
            start = len(rows)
            # Distinct samples make an off-by-one raw address observable.
            np.save(path, np.stack([np.full((14, 2560), start + i, np.float32) for i in range(4)]))
            path.with_suffix(".json").write_text(json.dumps({
                "channels": channels, "record_id": subject, "sfreq_target": 256,
                "raw_preprocessing": DEFAULT_RAW_PREPROCESSING,
                "preprocessing_hash": preprocessing_hash, "accepted_windows": 4,
                "window_results": [{"sample_id": start + i, "cache_offset": i, "status": "ok"} for i in range(4)],
            }), encoding="utf-8")
            for i in range(4):
                rows.append({
                    "sample_id": start + i, "subject_id": subject, "record_id": subject,
                    "record_group_id": subject, "outer_fold": fold, "t_start": i * 10.0,
                    "t_end": (i + 1) * 10.0, "cache_file": path.name, "cache_offset": i,
                    "n_channels": 14, "n_samples_expected": 2560, "sfreq_target": 256,
                    "preprocessing_hash": preprocessing_hash, "status": "ok",
                })
    index = pd.DataFrame(rows)
    _, pipeline = load_feature_profile(ROOT / "experiments/features/preliminary_model_zoo_features_v1.json")
    names = pipeline.feature_names()
    identity = {
        "cache_schema_version": "cogstate-feature-cache-v1", "feature_hash": pipeline.feature_hash(),
        "sample_id_universe_hash": sample_id_universe_hash(index.sample_id),
        "raw_preprocessing_hash": preprocessing_hash, "rows": len(index), "n_features": len(names),
        "dtype": "float32", "channel_order": [c.removeprefix("EEG.") for c in channels],
        "raw_window_shape": [2560, 14], "target_columns_present": False,
    }
    identity["cache_identity_hash"] = stable_hash(identity)
    fold_hash = stable_hash(index[["sample_id", "subject_id", "outer_fold"]].astype(str).to_dict("records"))
    return {
        "matrix": np.zeros((len(index), len(names)), dtype=np.float32), "index": index,
        "names": names, "cache_manifest": {"identity": deepcopy(identity), "status": "complete"},
        "raw_manifest": index.copy(), "targets": index[["sample_id", "subject_id", "record_id"]].copy(),
        "expected_identity": identity, "expected_fold_hash": fold_hash, "folds": [1, 2, 3, 4, 5],
        "cache_path_root": tmp_path,
    }


def test_valid_read_only_chain_and_distinct_ids(inputs):
    root = inputs["cache_path_root"]
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    result = audit_loaded_inputs(**inputs)
    assert result["status"] == "PASS"
    assert result["sampled_pairs"] == 10
    assert result["temporal_pairing"]["matched_target_rows"] == 30
    for row in result["checked_pairs"]:
        assert row["feature_sample_id"] != row["target_sample_id"]
        assert row["target_time"] - row["feature_time"] == 10
        assert inputs["index"].iloc[row["feature_position"]].sample_id == row["feature_sample_id"]
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}


@pytest.mark.parametrize("shape,dtype", [((4, 14, 2559), "float32"), ((4, 14, 2560), "float64")])
def test_invalid_stored_shape_or_dtype(inputs, shape, dtype):
    np.save(inputs["cache_path_root"] / "s1-0.npy", np.zeros(shape, dtype=dtype))
    with pytest.raises(ValueError, match="Raw stored (shape|dtype)"):
        audit_loaded_inputs(**inputs)


@pytest.mark.parametrize("mutation,match", [
    ("reverse_channels", "channel order"), ("missing_channels", "channel order"),
    ("wrong_offset", "cache_offset"), ("missing_sample", "provenance"),
    ("wrong_record", "record_id"), ("preprocessing", "preprocessing hash"),
])
def test_sidecar_provenance_failures(inputs, mutation, match):
    path = inputs["cache_path_root"] / "s1-0.json"
    payload = json.loads(path.read_text())
    if mutation == "reverse_channels":
        payload["channels"].reverse()
    elif mutation == "missing_channels":
        del payload["channels"]
    elif mutation == "wrong_offset":
        payload["window_results"][0]["cache_offset"] = 1
    elif mutation == "missing_sample":
        payload["window_results"] = []
    elif mutation == "wrong_record":
        payload["record_id"] = "wrong"
    else:
        payload["raw_preprocessing"]["notch"]["enabled"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        audit_loaded_inputs(**inputs)


@pytest.mark.parametrize("column,value,match", [
    ("subject_id", "wrong-subject", "subject_id"),
    ("outer_fold", 2, "outer.fold"), ("record_id", "wrong-record", "record_id"),
    ("record_group_id", "wrong-group", "record_group_id"),
    ("sfreq_target", 128, "sampling rate"), ("cache_offset", -1, "cache_offset"),
])
def test_raw_metadata_mismatch(inputs, column, value, match):
    # Alter the whole subject's fold so mapping remains single-valued but differs.
    selection = inputs["raw_manifest"].subject_id.eq("s1-0") if column == "outer_fold" else [0]
    inputs["raw_manifest"].loc[selection, column] = value
    with pytest.raises(ValueError, match=match):
        audit_loaded_inputs(**inputs)


@pytest.mark.parametrize("table", ["index", "raw_manifest", "targets"])
def test_duplicate_sample_id(inputs, table):
    inputs[table].loc[1, "sample_id"] = inputs[table].loc[0, "sample_id"]
    with pytest.raises(ValueError, match="duplicate sample_id"):
        audit_loaded_inputs(**inputs)


def test_fold_hash_mismatch(inputs):
    inputs["expected_fold_hash"] = "wrong"
    with pytest.raises(ValueError, match="fold hash"):
        audit_loaded_inputs(**inputs)


@pytest.mark.parametrize("mutation,match", [
    ("row_count", "matrix shape"), ("width", "matrix shape"),
    ("cache_identity", "cache identity"), ("sample_universe", "universe hash"),
    ("feature_channels", "channel order"), ("fold_membership", "outer folds"),
    ("subject_multiple_folds", "multiple outer folds"),
])
def test_frozen_cohort_contract_failures(inputs, mutation, match):
    if mutation == "row_count":
        inputs["matrix"] = inputs["matrix"][:-1]
    elif mutation == "width":
        inputs["matrix"] = inputs["matrix"][:, :-1]
    elif mutation == "cache_identity":
        inputs["cache_manifest"]["identity"]["cache_identity_hash"] = "other-cache"
    elif mutation == "sample_universe":
        inputs["index"].loc[0, "sample_id"] = 99999
    elif mutation == "feature_channels":
        identity = inputs["cache_manifest"]["identity"]
        identity["channel_order"].reverse()
        identity["cache_identity_hash"] = stable_hash({k: v for k, v in identity.items() if k != "cache_identity_hash"})
        inputs["expected_identity"] = deepcopy(identity)
    elif mutation == "fold_membership":
        inputs["index"].loc[0, "outer_fold"] = 6
    else:
        inputs["index"].loc[0, "outer_fold"] = 2
    with pytest.raises(ValueError, match=match):
        audit_loaded_inputs(**inputs)


def test_target_subject_mismatch(inputs):
    inputs["targets"].loc[1, "subject_id"] = "wrong"
    with pytest.raises(ValueError, match="Target subject_id"):
        audit_loaded_inputs(**inputs)


def test_nan_raw_signal(inputs):
    path = inputs["cache_path_root"] / "s1-0.npy"
    values = np.load(path)
    values[0, 0, 0] = np.nan
    np.save(path, values)
    with pytest.raises(ValueError, match="invalid"):
        audit_loaded_inputs(**inputs)


def test_sampling_deterministic_and_subject_diverse(inputs):
    pairs, _ = build_previous_window_pairing(inputs["index"])
    first = sample_pairs(pairs, 2)
    pd.testing.assert_frame_equal(first, sample_pairs(pairs.sample(frac=1, random_state=8), 2))
    assert first.groupby("outer_fold").subject_id.nunique().eq(2).all()
    with pytest.raises(ValueError, match="positive"):
        sample_pairs(pairs, 0)


def test_gap_uses_no_previous_available_substitution(inputs):
    pairs, _ = build_previous_window_pairing(inputs["index"].drop(index=1))
    assert 2 not in pairs.target_sample_id.to_list()
    assert pairs.loc[pairs.target_sample_id.eq(3), "lag_minus_10s_feature_sample_id"].item() == 2


def test_cli_and_disk_integration(inputs, tmp_path, capsys):
    root = tmp_path
    config_path = Path("experiments/pm_diagnostics/pm_low_high_q3_extremes_confirmatory_v1.json")
    config = json.loads((ROOT / config_path).read_text())
    config["feature_cache_identity"] = inputs["expected_identity"]
    for rel, payload in [
        (config_path, config),
        (Path(config["output_dir"]) / "protocol.json", {
            "feature_cache_identity": inputs["expected_identity"], "fixed_fold_hash": inputs["expected_fold_hash"],
        }),
    ]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    profile = Path("experiments/features/preliminary_model_zoo_features_v1.json")
    (root / profile).parent.mkdir(parents=True)
    (root / profile).write_bytes((ROOT / profile).read_bytes())
    cache = root / "feature-cache"
    cache.mkdir()
    manifest = dict(inputs["cache_manifest"], schema_version="cogstate-feature-cache-v1")
    (cache / "feature_materialization_manifest.json").write_text(json.dumps(manifest))
    (cache / "feature_names.json").write_text(json.dumps({"feature_names": inputs["names"]}))
    np.save(cache / "features.npy", inputs["matrix"])
    inputs["index"].to_parquet(cache / "feature_index.parquet", index=False)
    raw_path = root / "raw.parquet"
    inputs["raw_manifest"].to_parquet(raw_path, index=False)
    targets_path = root / config["data"]["processed_targets"]
    targets_path.parent.mkdir(parents=True)
    inputs["targets"].set_index("sample_id").to_parquet(targets_path)
    args = ["--root", str(root), "--feature-cache-dir", "feature-cache", "--raw-manifest", "raw.parquet"]
    assert main(args) == 0
    assert "STATUS: PASS" in capsys.readouterr().out
    assert main(args + ["--output-json", "audit.json"]) == 0
    saved = (root / "audit.json").read_bytes()
    assert json.loads(saved)["sampled_pairs"] == 10
    assert main(args + ["--output-json", "audit.json"]) == 1
    assert (root / "audit.json").read_bytes() == saved
    assert main(args + ["--samples-per-fold", "0"]) == 1
    (root / "s1-0.json").unlink()
    assert main(args) == 1
