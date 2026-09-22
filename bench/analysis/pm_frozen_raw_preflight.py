"""Read-only audit of frozen PM feature rows and sampled raw EEG windows.

Only metadata pairing is constructed; no signal/target transform is fitted or
applied. Raw reads are limited to deterministic samples from the frozen index.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from bench.datasets.channel_contracts import PROJECT_EMOTIV_CHANNEL_ORDER
from bench.datasets.pm_union_raw_contract import fixed_outer_fold_mapping
from bench.datasets.raw_eeg_window_dataset import RawEEGWindowArrayView
from bench.datasets.raw_preprocessing import raw_preprocessing_hash
from bench.experiments.pm_eeg_lag_confirmatory import (
    build_previous_window_pairing, stable_hash, validate_cache_contract,
)
from bench.experiments.pm_low_high_q3_extremes_confirmatory import load_config
from bench.features.cogstate_feature_cache import (
    load_feature_cache, load_feature_profile, sample_id_universe_hash,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _required(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    _require(not missing, f"{name}: missing columns {missing}")
    _require(not frame[columns].isna().any().any(), f"{name}: missing metadata values")
    _require(not frame.sample_id.duplicated().any(), f"{name}: duplicate sample_id")


def sample_pairs(pairing: pd.DataFrame, samples_per_fold: int) -> pd.DataFrame:
    """Round-robin subjects, then records, then time, without random state."""
    _require(samples_per_fold > 0, "samples_per_fold must be positive")
    frame = pairing.sort_values(
        ["outer_fold", "subject_id", "record_id", "target_time", "target_sample_id"],
        kind="stable",
    ).copy()
    frame["_record_round"] = frame.groupby(["outer_fold", "subject_id", "record_id"]).cumcount()
    frame = frame.sort_values(
        ["outer_fold", "_record_round", "record_id", "target_time", "target_sample_id"],
        kind="stable",
    )
    frame["_subject_round"] = frame.groupby(["outer_fold", "subject_id"]).cumcount()
    return (
        frame.sort_values(
            ["outer_fold", "_subject_round", "subject_id", "_record_round", "record_id", "target_sample_id"],
            kind="stable",
        ).groupby("outer_fold", sort=True).head(samples_per_fold)
        .drop(columns=["_record_round", "_subject_round"]).reset_index(drop=True)
    )


def validate_frozen_raw_inputs(matrix, index, names, cache_manifest, raw_manifest, *,
                               expected_identity, expected_fold_hash, folds):
    """Validate frozen identity and raw metadata without reading PM targets."""
    identity = validate_cache_contract(matrix, index, names, cache_manifest, expected_identity)
    _require(cache_manifest.get("status") == "complete", "Feature cache is incomplete")
    _require(sample_id_universe_hash(index.sample_id) == identity["sample_id_universe_hash"],
             "Frozen sample_id universe hash mismatch")
    _require(
        stable_hash({k: v for k, v in identity.items() if k != "cache_identity_hash"})
        == identity["cache_identity_hash"], "Feature cache identity hash is inconsistent",
    )
    channels = list(PROJECT_EMOTIV_CHANNEL_ORDER)
    _require(identity.get("channel_order") == [c.removeprefix("EEG.") for c in channels],
             "Feature channel order differs from canonical channel contract")
    _require(identity.get("raw_window_shape") == [2560, len(channels)], "Feature raw shape contract mismatch")
    metadata = ["sample_id", "subject_id", "record_id", "record_group_id", "outer_fold", "t_start", "t_end"]
    _required(index, metadata, "Feature index")
    _required(raw_manifest, metadata + ["status", "cache_file", "cache_offset", "n_channels",
              "n_samples_expected", "sfreq_target", "preprocessing_hash"], "Raw manifest")
    _require(set(index.outer_fold) == set(folds), "Unexpected outer folds")
    feature_mapping = fixed_outer_fold_mapping(index)
    raw_mapping = fixed_outer_fold_mapping(raw_manifest)
    _require(all(raw_mapping.get(s) == f for s, f in feature_mapping.items()), "Raw/feature outer-fold mismatch")
    fold_hash = stable_hash(index[["sample_id", "subject_id", "outer_fold"]]
                            .sort_values("sample_id", kind="stable").astype(str).to_dict("records"))
    _require(fold_hash == expected_fold_hash, "Frozen fold hash mismatch")
    return identity, fold_hash


def load_verified_raw_window(feature: pd.Series, raw: pd.Series, *,
                             identity: Mapping[str, Any], cache_path_root: Path) -> np.ndarray:
    """Read one raw window only after checking provenance, channel order and storage."""
    channels = list(PROJECT_EMOTIV_CHANNEL_ORDER)
    eeg_id = feature.sample_id
    for column in ("sample_id", "subject_id", "record_id", "record_group_id", "outer_fold", "t_start", "t_end"):
        _require(raw[column] == feature[column], f"sample_id {eeg_id}: raw/feature {column} mismatch")
    _require(float(raw.t_end) - float(raw.t_start) == 10.0, "Raw duration mismatch")
    _require(raw.status == "ok", "Raw window is not QC accepted")
    _require(raw.n_channels == len(channels) and raw.n_samples_expected == 2560, "Raw shape metadata mismatch")
    _require(raw.sfreq_target == 256.0, "Raw sampling rate mismatch")
    _require(raw.preprocessing_hash == identity["raw_preprocessing_hash"], "Raw preprocessing mismatch")
    path = Path(str(raw.cache_file))
    if not path.is_absolute():
        path = cache_path_root / path
    # The sidecar was written by build_raw_eeg_cache, not inferred from shape.
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    _require(sidecar.get("channels") == channels, f"{path.name}: raw channel order unproven or incorrect")
    _require(sidecar.get("record_id") == raw.record_id, "Shard record_id mismatch")
    _require(sidecar.get("sfreq_target") == 256.0, "Shard sampling rate mismatch")
    _require("raw_preprocessing" in sidecar, "Shard preprocessing contract missing")
    preprocessing_hash = raw_preprocessing_hash(sidecar["raw_preprocessing"], channels=channels)
    _require(preprocessing_hash == sidecar.get("preprocessing_hash") == raw.preprocessing_hash,
             "Shard preprocessing hash mismatch")
    entries = [r for r in sidecar.get("window_results", []) if r.get("sample_id") == eeg_id]
    _require(len(entries) == 1, "Shard sample_id provenance missing or duplicated")
    entry = entries[0]
    _require(entry.get("status") == "ok" and entry.get("cache_offset") == raw.cache_offset,
             "Shard sample_id/cache_offset mismatch")
    # Inspect storage before RawEEGWindowArrayView can cast dtype to float32.
    stored = np.load(path, mmap_mode="r", allow_pickle=False)
    _require(stored.dtype == np.dtype("float32"), "Raw stored dtype must be float32")
    _require(stored.shape == (sidecar.get("accepted_windows"), len(channels), 2560), "Raw stored shape mismatch")
    _require(float(raw.cache_offset).is_integer() and 0 <= raw.cache_offset < len(stored), "Invalid raw cache_offset")
    del stored
    view = RawEEGWindowArrayView(raw.to_frame().T, cache_path_root=cache_path_root)
    window = view[0][0]
    _require(window.shape == (len(channels), 2560) and window.dtype == np.float32,
             "Loaded raw shape/dtype mismatch")
    _require(np.isfinite(window).all(), "Raw window contains non-finite values")
    _require(window.T.shape == (2560, len(channels)), "Feature input axes mismatch")
    return window


def audit_loaded_inputs(
    matrix: np.ndarray,
    index: pd.DataFrame,
    names: list[str],
    cache_manifest: Mapping[str, Any],
    raw_manifest: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    expected_identity: Mapping[str, Any],
    expected_fold_hash: str,
    folds: list[int],
    cache_path_root: Path,
    samples_per_fold: int = 2,
) -> dict[str, Any]:
    """Audit supplied inputs; expected identity is the frozen benchmark contract."""
    identity, fold_hash = validate_frozen_raw_inputs(
        matrix, index, names, cache_manifest, raw_manifest,
        expected_identity=expected_identity, expected_fold_hash=expected_fold_hash, folds=folds,
    )
    channels = list(PROJECT_EMOTIV_CHANNEL_ORDER)
    _required(targets, ["sample_id", "subject_id", "record_id"], "Target table")
    pairing, pairing_summary = build_previous_window_pairing(index)
    selected = sample_pairs(pairing, samples_per_fold)
    _require(set(selected.outer_fold) == set(folds), "No temporal pairs in one or more outer folds")
    raw_lookup = raw_manifest.set_index("sample_id", drop=False)
    feature_lookup = index.set_index("sample_id", drop=False)
    target_lookup = targets.set_index("sample_id", drop=False)
    checked = []
    for pair in selected.to_dict("records"):
        eeg_id = pair["lag_minus_10s_feature_sample_id"]
        target_id = pair["target_sample_id"]
        _require(eeg_id != target_id, "Previous-window EEG and target IDs must differ")
        _require(eeg_id in raw_lookup.index, f"Raw sample_id {eeg_id} missing")
        _require(target_id in target_lookup.index, f"Target sample_id {target_id} missing")
        feature, raw = feature_lookup.loc[eeg_id], raw_lookup.loc[eeg_id]
        target, target_feature = target_lookup.loc[target_id], feature_lookup.loc[target_id]
        for column in ("subject_id", "record_id", "record_group_id", "outer_fold"):
            _require(feature[column] == target_feature[column], f"Pair {column} mismatch")
            if column in target.index:
                _require(target[column] == target_feature[column], f"Target {column} mismatch")
        _require(float(target_feature.t_start) - float(feature.t_start) == 10.0,
                 "Temporal pair is not an exact previous 10-second window")
        load_verified_raw_window(feature, raw, identity=identity, cache_path_root=cache_path_root)
        checked.append({
            "feature_position": int(pair["lag_minus_10s_feature_position"]),
            "feature_sample_id": int(eeg_id), "target_sample_id": int(target_id),
            "subject_id": str(raw.subject_id), "record_id": str(raw.record_id),
            "record_group_id": str(raw.record_group_id), "outer_fold": int(raw.outer_fold),
            "feature_time": float(feature.t_start), "target_time": float(target_feature.t_start),
            "cache_offset": int(raw.cache_offset), "channel_order_verified": True,
        })
    return {
        "status": "PASS", "schema_version": "pm-frozen-raw-preflight-v1",
        "frozen_rows": len(index), "feature_width": matrix.shape[1],
        "cache_identity_hash": identity["cache_identity_hash"], "fixed_fold_hash": fold_hash,
        "outer_folds": folds, "channel_order": channels, "raw_shape": [len(channels), 2560],
        "raw_dtype": "float32", "sampling_rate": 256, "feature_input_shape": [2560, len(channels)],
        "temporal_pairing": pairing_summary, "sampled_pairs": len(checked),
        "samples_per_fold": samples_per_fold, "checked_pairs": checked,
        "training_executed": False, "features_recomputed": False,
    }


def run_preflight(*, root: Path, feature_cache_dir: Path, raw_manifest_path: Path,
                  cache_path_root: Path, samples_per_fold: int = 2) -> dict[str, Any]:
    """Use the existing frozen LOW/HIGH config and saved protocol as authority."""
    config = load_config(root / "experiments/pm_diagnostics/pm_low_high_q3_extremes_confirmatory_v1.json")
    protocol = json.loads((root / config["output_dir"] / "protocol.json").read_text(encoding="utf-8"))
    for key, value in config["feature_cache_identity"].items():
        _require(protocol["feature_cache_identity"].get(key) == value, f"Frozen protocol/config disagree: {key}")
    matrix, index, names, manifest = load_feature_cache(feature_cache_dir)
    _, pipeline = load_feature_profile(root / "experiments/features/preliminary_model_zoo_features_v1.json")
    _require(names == pipeline.feature_names(), "Frozen feature names/order mismatch")
    _require(pipeline.feature_hash() == config["feature_cache_identity"]["feature_hash"], "Feature profile hash mismatch")
    raw = pd.read_parquet(raw_manifest_path)
    targets = pd.read_parquet(root / config["data"]["processed_targets"])
    if "sample_id" not in targets:
        targets.insert(0, "sample_id", targets.index.to_numpy())
    return audit_loaded_inputs(
        matrix, index, names, manifest, raw, targets,
        expected_identity=config["feature_cache_identity"], expected_fold_hash=protocol["fixed_fold_hash"],
        folds=config["evaluation"]["folds"], cache_path_root=cache_path_root,
        samples_per_fold=samples_per_fold,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, default=Path("data/interim/raw_eeg_window_index_w10_pm_union_composite_v1.parquet"))
    parser.add_argument("--cache-path-root", type=Path, help="Root for relative shard paths; defaults to --root")
    parser.add_argument("--samples-per-fold", type=int, default=2)
    parser.add_argument("--output-json", type=Path, help="Optional new audit file; existing files are never overwritten")
    args = parser.parse_args(argv)
    root = args.root.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    try:
        _require(args.samples_per_fold > 0, "samples_per_fold must be positive")
        if args.output_json and resolve(args.output_json).exists():
            raise ValueError("Audit output already exists; refusing to overwrite")
        summary = run_preflight(
            root=root, feature_cache_dir=resolve(args.feature_cache_dir),
            raw_manifest_path=resolve(args.raw_manifest), cache_path_root=resolve(args.cache_path_root or root),
            samples_per_fold=args.samples_per_fold,
        )
        if args.output_json:
            with resolve(args.output_json).open("x", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2, allow_nan=False)
                stream.write("\n")
    except (ValueError, RuntimeError, OSError, KeyError, IndexError, TypeError) as exc:
        print(f"STATUS: FAIL\n{exc}")
        return 1
    print(f"Frozen rows: {summary['frozen_rows']} OK\nFeature width: {summary['feature_width']} OK\n"
          f"Outer folds: {len(summary['outer_folds'])} OK\nChannel contract: 14 OK\n"
          f"Raw shape: 14x2560 OK\nRaw dtype: float32 OK\n"
          f"Temporal pairs sampled: {summary['sampled_pairs']} OK\n"
          f"Subject/fold match: {summary['sampled_pairs']} OK\nSTATUS: PASS")
    return 0
