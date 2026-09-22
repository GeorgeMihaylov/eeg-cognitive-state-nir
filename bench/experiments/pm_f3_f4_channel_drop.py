"""Fixed F3/F4 channel-drop screening using the existing LOW/HIGH execution path."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import hashlib
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from bench.analysis.interchannel_lag_diagnostics import select_windows
from bench.analysis.paired_statistics import subject_bootstrap_interval
from bench.analysis.pm_frozen_raw_preflight import run_preflight, load_verified_raw_window
from bench.datasets.channel_contracts import PROJECT_EMOTIV_CHANNEL_ORDER
from bench.experiments.pm_eeg_lag_confirmatory import build_previous_window_pairing, stable_hash
from bench.experiments import pm_low_high_q3_extremes_confirmatory as low_high
from bench.experiments import pm_low_high_model_robustness as robustness
from bench.features.cogstate_feature_cache import (
    load_feature_cache, load_feature_profile, materialize_cogstate_features,
)
from cogstate.features import FeaturePipeline, FeaturePipelineConfig
from cogstate.model_zoo import build_model


VARIANTS = ("baseline_14ch", "drop_F3_13ch", "drop_F4_13ch")
METRICS = ("balanced_accuracy", "macro_f1", "roc_auc", "pr_auc", "low_recall", "high_recall")
QC_VARIANTS = ("baseline_14ch", "drop_F8_13ch", "drop_F7_13ch")
DROPPED_CHANNEL = {"baseline_14ch": None, **{f"drop_{c}_13ch": c for c in ("F3", "F4", "F8", "F7")}}
CONFIG_PATH = "experiments/pm_diagnostics/pm_f3_f4_channel_drop_v1.json"


def write_json(path: Path, value: dict) -> None:
    low_high._atomic_json(path, value)


def variant_profile(profile: dict, variant: str) -> tuple[dict, list[int]]:
    if variant not in DROPPED_CHANNEL:
        raise ValueError(f"Unsupported channel variant: {variant}")
    canonical = [c.removeprefix("EEG.") for c in PROJECT_EMOTIV_CHANNEL_ORDER]
    if profile["feature_pipeline"]["channel_names"] != canonical:
        raise ValueError("Profile channel order mismatch")
    drop = DROPPED_CHANNEL[variant]
    indices = [i for i, channel in enumerate(canonical) if channel != drop]
    result = deepcopy(profile)
    result["profile_id"] = variant
    result["feature_pipeline"]["channel_names"] = [canonical[i] for i in indices]
    pipeline = FeaturePipeline(FeaturePipelineConfig.from_mapping(result))
    expected = 371 if drop is None else 346
    if len(pipeline.feature_names()) != expected:
        raise ValueError("Unexpected feature_names width")
    if drop and any(name.endswith(f"__{drop}") for name in pipeline.feature_names()):
        raise ValueError("Dropped channel entered feature names")
    return result, indices


def assert_same_index(baseline: pd.DataFrame, variant: pd.DataFrame) -> None:
    columns = ["sample_id", "subject_id", "record_id", "record_group_id", "outer_fold", "t_start", "t_end"]
    try:
        pd.testing.assert_frame_equal(baseline[columns].reset_index(drop=True), variant[columns].reset_index(drop=True),
                                      check_dtype=False, check_exact=True)
        a, _ = build_previous_window_pairing(baseline)
        b, _ = build_previous_window_pairing(variant)
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError("Frozen index/fold/temporal pairing mismatch") from exc


def retained_rows(context, fold: int, pm: str, train: bool = False) -> pd.DataFrame:
    cohort = context.cohorts[pm]
    labels = context.transforms[(fold, pm)].transform(cohort.continuous_target.to_numpy())
    mask = cohort.outer_fold.ne(fold) if train else cohort.outer_fold.eq(fold)
    result = cohort.loc[mask & np.isfinite(labels)].copy()
    result["y_true"] = labels[mask & np.isfinite(labels)].astype(int)
    return result


def audit_predictions(context, spec: dict, path: Path) -> tuple[pd.DataFrame, dict]:
    """Recheck identities/thresholds and recompute participant metrics from saved predictions."""
    expected = retained_rows(context, int(spec["outer_fold"]), str(spec["pm"])).reset_index(drop=True)
    saved = pd.read_parquet(path).reset_index(drop=True)
    mapping = {"target_sample_id": "target_sample_id", "feature_sample_id": "lag_minus_10s_feature_sample_id",
               "subject_id": "subject_id", "record_id": "record_id", "outer_fold": "outer_fold", "y_true": "y_true"}
    for saved_column, expected_column in mapping.items():
        if saved[saved_column].astype(str).tolist() != expected[expected_column].astype(str).tolist():
            raise ValueError(f"Baseline prediction {saved_column} mismatch")
    for field in ("q_low", "q_high"):
        if not np.allclose(saved[field], spec[field], atol=1e-12, rtol=0):
            raise ValueError(f"Baseline {field} mismatch")
    participant, metrics = low_high.participant_binary_metrics(
        saved.y_true.to_numpy(), saved.y_pred.to_numpy(), saved.probability_high.to_numpy(),
        saved.subject_id.astype(str).to_numpy(),
    )
    participant["outer_fold"], participant["pm"] = spec["outer_fold"], spec["pm"]
    return participant, metrics


def reuse_baseline(root: Path, cache: Path, context, config: dict):
    """Return only audited LightGBM artifacts; no fitting or output writes."""
    ref_config = robustness.load_config(root / config["reference_config"])
    if ref_config["candidate_models"]["lightgbm"]["params"] != config["model_params"]:
        raise ValueError("LightGBM model params mismatch")
    candidate = robustness.prepare_protocol(ref_config, root=root, feature_cache_dir=cache)
    directory = root / ref_config["output_dir"]
    stored = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    for field in ("protocol_hash", "feature_cache_identity", "fixed_fold_hash", "temporal_pairing_hash", "threshold_hashes", "candidate_models"):
        if stored[field] != candidate.protocol[field]:
            raise ValueError(f"Baseline protocol {field} mismatch")
    participants, results, audit = [], [], []
    for spec in candidate.run_matrix.query("model == 'lightgbm'").to_dict("records"):
        directory_run = directory / "runs" / spec["run_id"]
        summary = json.loads((directory_run / "run_summary.json").read_text(encoding="utf-8"))
        for key in ("specification_hash", "threshold_hash", "model_params_hash", "model", "n_train", "n_test"):
            if summary[key] != spec[key]:
                raise ValueError(f"Baseline run {key} mismatch")
        if summary["status"] != "complete" or summary["protocol_hash"] != stored["protocol_hash"]:
            raise ValueError("Baseline run incomplete or wrong protocol")
        reference = context.run_matrix.set_index(["outer_fold", "pm"]).loc[(spec["outer_fold"], spec["pm"])]
        for key in ("train_sample_hash", "test_sample_hash", "matched_target_sample_hash", "threshold_hash"):
            if spec[key] != reference[key]:
                raise ValueError(f"Baseline cohort {key} mismatch")
        participant, metrics = audit_predictions(context, spec, directory_run / "predictions.parquet")
        for key, value in metrics.items():
            if not np.isclose(value, summary[key], rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError(f"Baseline metric {key} mismatch")
        participants.append(participant.assign(variant="baseline_14ch"))
        results.append(dict(summary, variant="baseline_14ch", **metrics))
        audit.append({"outer_fold": spec["outer_fold"], "pm": spec["pm"], "run_id": spec["run_id"],
                      "train_sample_hash": spec["train_sample_hash"], "test_sample_hash": spec["test_sample_hash"],
                      "threshold_hash": spec["threshold_hash"], "equivalent": True})
    if len(results) != 35:
        raise ValueError("Baseline must contain exactly 35 LightGBM runs")
    return pd.concat(participants, ignore_index=True), results, audit


def aggregate(participants: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["outer_fold", "pm", "subject_id"]
    baseline = participants.loc[participants.variant.eq("baseline_14ch")]
    summaries, deltas = [], []
    for variant in config.get("variants", VARIANTS):
        values = participants.loc[participants.variant.eq(variant)]
        for pm in [*config["targets"], "overall"]:
            group = (values.groupby("subject_id")[list(METRICS)].mean().reset_index()
                     if pm == "overall" else values.loc[values.pm.eq(pm)])
            row = {"variant": variant, "pm": pm, "n_subjects": len(group)}
            for metric in METRICS:
                row[metric] = group[metric].mean()
                row[f"median_{metric}"] = group[metric].median()
                row[f"valid_subjects_{metric}"] = int(group[metric].notna().sum())
            summaries.append(row)
        if variant == "baseline_14ch":
            continue
        joined = values.merge(baseline, on=keys, how="outer", suffixes=("_drop", "_baseline"),
                              validate="one_to_one", indicator=True)
        if not joined._merge.eq("both").all() or not joined.n_samples_drop.eq(joined.n_samples_baseline).all():
            raise ValueError("Participant cohort mismatch")
        delta = joined[keys].copy()
        delta["variant"] = variant
        for metric in METRICS:
            delta[f"baseline_{metric}"] = joined[f"{metric}_baseline"]
            delta[f"drop_{metric}"] = joined[f"{metric}_drop"]
            delta[f"delta_{metric}"] = joined[f"{metric}_drop"] - joined[f"{metric}_baseline"]
        overall = delta.groupby(["subject_id", "outer_fold", "variant"], as_index=False)[
            [f"{prefix}_{m}" for m in METRICS for prefix in ("baseline", "drop", "delta")]].mean()
        overall["pm"] = "overall"
        deltas.extend([delta, overall])
    delta_frame = pd.concat(deltas, ignore_index=True)
    bootstrap = []
    for (variant, pm), group in delta_frame.groupby(["variant", "pm"], sort=True):
        if group.subject_id.duplicated().any():
            raise ValueError("Bootstrap unit must be one paired value per subject")
        values = group.delta_balanced_accuracy.dropna().to_numpy()
        ci = subject_bootstrap_interval(values, **{k: v for k, v in config["bootstrap"].items() if k != "unit"})
        bootstrap.append({"variant": variant, "pm": pm, "n_subjects": len(values),
                          "mean_delta_ba": float(np.mean(values)), "median_delta_ba": float(np.median(values)),
                          "fraction_participants_improved": float(np.mean(values > 0)),
                          "fraction_participants_tied": float(np.mean(values == 0)),
                          "fraction_participants_worsened": float(np.mean(values < 0)),
                          "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]})
    return pd.DataFrame(summaries), delta_frame, pd.DataFrame(bootstrap)


def pretraining_contract(context, representations: dict) -> pd.DataFrame:
    """Audit every representation and all partition masks before any model fit."""
    rows = []
    for variant in representations:
        matrix, index, names, manifest = representations[variant]
        assert_same_index(context.feature_index, index)
        width = 371 if variant == "baseline_14ch" else 346
        if matrix.shape != (len(index), width) or len(names) != width or not np.isfinite(matrix).all():
            raise ValueError("Pre-training feature shape/finite contract mismatch")
        identity = manifest["identity"]
        expected_channels = [c.removeprefix("EEG.") for c in PROJECT_EMOTIV_CHANNEL_ORDER
                             if c.removeprefix("EEG.") != DROPPED_CHANNEL[variant]]
        if identity["channel_order"] != expected_channels:
            raise ValueError("Pre-training channel order mismatch")
        for spec in context.run_matrix.to_dict("records"):
            fold, pm = int(spec["outer_fold"]), spec["pm"]
            cohort = context.cohorts[pm]
            positions = cohort.lag_minus_10s_feature_position.to_numpy(dtype=int)
            if index.iloc[positions].sample_id.tolist() != cohort.lag_minus_10s_feature_sample_id.tolist():
                raise ValueError("Pre-training temporal feature IDs mismatch")
            labels = context.transforms[(fold, pm)].transform(cohort.continuous_target.to_numpy())
            if any(getattr(context.transforms[(fold, pm)], field) != spec[field] for field in ("q_low", "q_high")):
                raise ValueError("Pre-training threshold mismatch")
            row = {"variant": variant, "outer_fold": fold, "pm": pm,
                   "threshold_hash": spec["threshold_hash"], "q_low": spec["q_low"], "q_high": spec["q_high"],
                   "pairing_hash": stable_hash(cohort[["target_sample_id", "lag_minus_10s_feature_sample_id"]].to_dict("records"))}
            for part, partition in (("train", cohort.outer_fold.ne(fold)), ("test", cohort.outer_fold.eq(fold))):
                retained = partition & np.isfinite(labels)
                row[f"{part}_subjects_hash"] = stable_hash(sorted(index.loc[index.outer_fold.ne(fold) if part == "train" else index.outer_fold.eq(fold), "subject_id"].astype(str).unique()))
                for label_name, mask in (("all", partition), ("retained", retained),
                                         ("low", partition & (labels == 0)), ("high", partition & (labels == 1)),
                                         ("middle", partition & ~np.isfinite(labels))):
                    row[f"{part}_{label_name}_target_hash"] = low_high._sample_hash(cohort.loc[mask, "target_sample_id"])
                if row[f"{part}_retained_target_hash"] != spec[f"{part}_sample_hash"]:
                    raise ValueError("Pre-training retained target IDs mismatch")
                row[f"{part}_participants"] = cohort.loc[retained, "subject_id"].nunique()
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.groupby(["outer_fold", "pm"])[result.columns.difference(["variant", "outer_fold", "pm"])].nunique(dropna=False).ne(1).any().any():
        raise ValueError("Pre-training representation contract mismatch")
    identities = [representations[v][3]["identity"]["cache_identity_hash"] for v in representations]
    if len(set(identities)) != 3:
        raise ValueError("Representation cache identities must differ")
    return result


def write_report(output: Path, cache_root: Path, manifest: dict, summary: pd.DataFrame, ci: pd.DataFrame) -> None:
    """Render compact descriptive tables directly from the audited result tables."""
    def table(frame):
        def cell(value):
            return f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)
        return "\n".join(["| " + " | ".join(frame.columns) + " |",
                          "| " + " | ".join(["---"] * len(frame.columns)) + " |", *[
                              "| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]])
    lines = ["# F3/F4 channel-drop pilot", "", "Status: complete; diagnostic descriptive screening. No automatic winner or next-channel selection.", "",
             "## Protocol and validation", "",
             "Frozen 30,958 EEG windows / 54 participants / five fixed subject-disjoint folds / seven PM. "
             "Exact previous-window EEG(t-10s) -> PM(t); existing outer-train-only Q33/Q67 thresholds; MIDDLE excluded. "
             "Pre-training checks cover all 105 representation/fold/PM combinations, including LOW/HIGH/MIDDLE IDs, subjects, thresholds and pairing.", "",
             "35 baseline LightGBM runs reused after automatic artifact verification; 70 new LightGBM fits. "
             "n_estimators=200, n_jobs=4, random_state=42. Connectivity recomputed on the remaining 13 raw channels.", "",
             "Overall aggregation: mean across available PM within each participant, then equal participant weight. "
             "Undefined metrics remain undefined and are excluded only from that metric's mean; valid counts are in the CSVs. "
             "Attention and Engagement have 53 evaluated participants; the other PM and overall have 54.", "",
             "Paired bootstrap: 10,000 resamples of participants, seed 42, percentile 95% CI for mean BA difference. "
             "Overall bootstrap uses one available-PM-averaged paired difference per participant. Folds are not bootstrap units. "
             "Target-specific intervals are descriptive and not multiplicity-adjusted.", "",
             "## Descriptive result", "",
             table(summary.loc[summary.pm.eq("overall"), ["variant", "balanced_accuracy", "macro_f1", "roc_auc", "pr_auc"]]), "",
             "## Paired result", "",
             table(ci.loc[ci.pm.eq("overall")]), "", "## PM-specific results", "",
             table(summary.loc[summary.pm.ne("overall"), ["pm", "variant", "n_subjects", "balanced_accuracy", "macro_f1", "roc_auc", "pr_auc"]]), "",
             table(ci.loc[ci.pm.ne("overall")]), "", "## Interpretation", "",
             "Removing F3 reduced overall participant-first BA; its paired mean-difference CI is below zero. "
             "Effects vary across participants and PM; the mean is more negative than the median. "
             "Removing F4 was approximately neutral overall, with a CI including zero and mixed participant/target effects. "
             "These results do not establish equivalence or justify selecting additional channels. No post-hoc success threshold was introduced.", "",
             "## Feature caches and provenance", ""]
    for variant, identity in manifest["cache_identities"].items():
        duration = "existing baseline (not rematerialized)"
        if variant != "baseline_14ch":
            materialization = json.loads((cache_root / "materialize" / variant / "feature_materialization_manifest.json").read_text())
            duration = f"{materialization['elapsed_seconds']:.3f} seconds"
        lines.extend([f"- {variant}: [{identity['rows']}, {identity['n_features']}], float32; {duration}.",
                      f"  Cache identity: `{identity['cache_identity_hash']}`.",
                      f"  Feature hash: `{identity['feature_hash']}`."])
    lines.extend(["", f"Protocol hash: `{manifest['protocol_hash']}`.",
                  f"Branch: `{manifest['branch']}`; HEAD: `{manifest['git_head']}`.",
                  manifest["dirty_state_note"], "", "Code hashes, runtime versions, channel lists, cohort/fold/pairing identities and full configuration: `run_manifest.json`.", "",
                  "Artifacts: `summary_by_variant.csv`, `summary_by_target.csv`, `participant_metrics.csv`, `participant_deltas.csv`, "
                  "`fold_target_results.csv`, `bootstrap_summary.csv`, `pretraining_contract.csv`, `cohort_equivalence.csv`.", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(*, root: Path, feature_cache: Path, cache_root: Path, output: Path, phase: str, workers: int, config_path: str = CONFIG_PATH):
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    variants = tuple(config["variants"])
    if variants not in (VARIANTS, QC_VARIANTS) or config["model"] != "lightgbm":
        raise ValueError("Only the fixed F3/F4 or F8/F7 LightGBM protocols are supported")
    preflight = run_preflight(root=root, feature_cache_dir=feature_cache,
                             raw_manifest_path=root / config["raw_manifest"], cache_path_root=root)
    reference_config = robustness.load_config(root / config["reference_config"])
    if config["model_params"] != reference_config["candidate_models"]["lightgbm"]["params"]:
        raise ValueError("Reference LightGBM params changed")
    reference = low_high.load_config(root / reference_config["reference"]["config"])
    context = low_high.prepare_protocol(reference, root=root, feature_cache_dir=feature_cache)
    if config["targets"] != list(low_high.PM_NAMES) or config["folds"] != reference["evaluation"]["folds"]:
        raise ValueError("PM/fold contract mismatch")
    profile, _ = load_feature_profile(root / config["feature_profile"])
    profiles = {v: variant_profile(profile, v) for v in variants}
    protocol_hash = stable_hash({"config": config, "reference_protocol": context.protocol["protocol_hash"],
                                 "profiles": profiles, "baseline_identity": context.cache_identity})
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["protocol_hash"] != protocol_hash:
            raise ValueError("Existing experiment protocol differs")
    else:
        manifest = {"schema_version": config["schema_version"], "protocol_hash": protocol_hash,
                    "config": config, "result_status": config["result_status"], "git_commit": context.protocol["git_commit"],
                    "preflight": preflight, "reference_protocol_hash": context.protocol["protocol_hash"],
                    "cache_identities": {"baseline_14ch": context.cache_identity}, "status": "planned"}
        write_json(manifest_path, manifest)
    manifest["code_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("bench/experiments/pm_f3_f4_channel_drop.py",
                     "bench/experiments/pm_low_high_q3_extremes_confirmatory.py",
                     "bench/features/cogstate_feature_cache.py", "cogstate/features/pipeline.py",
                     "scripts/analysis/run_pm_f3_f4_channel_drop.py", "bench/analysis/pm_frozen_raw_preflight.py")
    }
    manifest["library_versions"] = {name: import_module(module).__version__ for name, module in
                                    (("numpy", "numpy"), ("scipy", "scipy"), ("scikit-learn", "sklearn"), ("lightgbm", "lightgbm"))}
    manifest["python_version"] = sys.version
    manifest["experiment_id"] = config["experiment_id"]
    manifest["branch"] = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    manifest["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    manifest["dirty_state_note"] = "Uncommitted channel-drop implementation; pre-existing unrelated modified/untracked artifacts retained. No commit created."
    manifest["frozen_cohort_hash"] = context.cache_identity["sample_id_universe_hash"]
    manifest["fold_hash"] = context.protocol["fixed_fold_hash"]
    manifest["temporal_pairing_identity"] = context.protocol["temporal_pairing_hash"]
    paths = {}
    for variant, (variant_config, _) in profiles.items():
        path = output / f"{variant}_profile.json"
        if path.exists() and json.loads(path.read_text()) != variant_config:
            raise ValueError("Existing profile differs")
        if not path.exists():
            write_json(path, variant_config)
        paths[variant] = path
    raw = pd.read_parquet(root / config["raw_manifest"])
    if raw.sample_id.duplicated().any():
        raise ValueError("Raw sample IDs duplicated")
    raw_lookup = raw.set_index("sample_id", drop=False)
    aligned = raw_lookup.loc[context.feature_index.sample_id].reset_index(drop=True)
    assert_same_index(context.feature_index, aligned)
    for _, row in aligned.groupby("record_id", sort=True).head(1).iterrows():
        # Verify every shard channel contract before materialization.
        feature = context.feature_index.set_index("sample_id", drop=False).loc[row.sample_id]
        load_verified_raw_window(feature, row, identity=context.cache_identity, cache_path_root=root)
    source = context.feature_index.copy()
    source["cache_file"] = aligned.cache_file.to_numpy()
    source["cache_offset"] = aligned.cache_offset.to_numpy()
    validation_path = output / "infrastructure_validation.json"
    if phase in ("materialize", "screen"):
        validation = json.loads(validation_path.read_text())
        if validation["status"] != "PASS" or validation["protocol_hash"] != protocol_hash:
            raise ValueError("Smoke validation is required before full materialization/training")
    if phase in ("smoke", "materialize"):
        subset = source
        if phase == "smoke":
            chosen = set(select_windows(source, config["smoke_windows_per_subject"]).sample_id)
            subset = source.loc[source.sample_id.isin(chosen)].reset_index(drop=True)
        for variant in variants[1:]:
            directory = cache_root / phase / variant
            print(f"{phase}: {variant}, {len(subset)} windows", flush=True)
            materialize_cogstate_features(
                manifest_path=root / config["raw_manifest"], logical_recording_map_path=root / "data/interim/logical_recording_map.parquet",
                cache_path_root=root, feature_profile_path=paths[variant], output_dir=directory,
                frozen_index=subset, raw_channel_indices=profiles[variant][1],
                source_cache_identity=context.cache_identity["cache_identity_hash"],
                chunk_size=128, workers=workers, resume=(directory / "feature_materialization_manifest.json").exists(),
            )
            matrix, index, names, cm = load_feature_cache(directory)
            assert_same_index(subset, index)
            if matrix.shape != (len(subset), 346) or not np.isfinite(matrix).all():
                raise ValueError("Invalid 13-channel matrix")
            if phase == "smoke":
                pipeline = FeaturePipeline(FeaturePipelineConfig.from_mapping(profiles[variant][0]))
                for i, row in subset.iterrows():
                    window = load_verified_raw_window(row, raw_lookup.loc[row.sample_id], identity=context.cache_identity, cache_path_root=root)
                    direct = pipeline.transform_window(window[profiles[variant][1]].T).astype(np.float32)
                    np.testing.assert_array_equal(matrix[i], direct)
                # Restore smoke rows into the unchanged full metadata index: pairing must stay identical.
                restored = context.feature_index.copy()
                for column in ("sample_id", "subject_id", "record_id", "record_group_id", "outer_fold", "t_start", "t_end"):
                    restored.loc[restored.sample_id.isin(index.sample_id), column] = index[column].to_numpy()
                assert_same_index(context.feature_index, restored)
            else:
                manifest["cache_identities"][variant] = cm["identity"]
        if phase == "smoke":
            write_json(validation_path, {"status": "PASS", "protocol_hash": protocol_hash, "rows": len(subset),
                        "sample_ids": subset.sample_id.astype(int).tolist(), "feature_width": 346,
                        "pairing_equal": True, "folds_equal": True, "direct_recompute_equal": True})
        manifest["status"] = f"{phase}_complete"
        write_json(manifest_path, manifest)
        return
    representations = {"baseline_14ch": (context.matrix, context.feature_index, context.feature_names, context.cache_manifest)}
    for variant in variants[1:]:
        representation = load_feature_cache(cache_root / "materialize" / variant)
        assert_same_index(context.feature_index, representation[1])
        if representation[3]["identity"] != manifest["cache_identities"][variant]:
            raise ValueError("Materialized cache identity mismatch")
        representations[variant] = representation
    pretraining_contract(context, representations).to_csv(output / "pretraining_contract.csv", index=False)
    baseline_participants, baseline_results, baseline_audit = reuse_baseline(root, feature_cache, context, config)
    baseline_ba = baseline_participants.groupby("subject_id").balanced_accuracy.mean().mean()
    if not np.isclose(baseline_ba, 0.7536625019623154, atol=1e-12, rtol=0):
        raise ValueError(f"Verified participant-first baseline BA changed: {baseline_ba}")
    baseline_reused = True
    manifest["baseline_reuse"] = {"status": "PASS", "audit": baseline_audit, "participant_first_ba": baseline_ba}
    manifest["pretraining_contract_status"] = "PASS"
    write_json(manifest_path, manifest)
    all_participants = [baseline_participants] if baseline_reused else []
    results = list(baseline_results) if baseline_reused else []
    cohort_audit = []
    for variant in variants:
        matrix, index, names, cm = representations[variant]
        execution = replace(context, output_dir=output / variant, matrix=matrix, feature_index=index,
                            feature_names=names, cache_manifest=cm, cache_identity=cm["identity"],
                            protocol={"protocol_hash": protocol_hash})
        for original in context.run_matrix.to_dict("records"):
            spec = dict(original, variant=variant, model="lightgbm", feature_identity=cm["identity"]["cache_identity_hash"])
            spec["specification_hash"] = stable_hash({"protocol_hash": protocol_hash, "run": spec})
            spec["run_id"] = f"fold_{spec['outer_fold']:02d}__{spec['pm']}__{spec['specification_hash'][:12]}"
            fold, pm = int(spec["outer_fold"]), str(spec["pm"])
            train, test = retained_rows(execution, fold, pm, True), retained_rows(execution, fold, pm)
            for part, values in (("train", train), ("test", test)):
                if low_high._sample_hash(values.target_sample_id) != spec[f"{part}_sample_hash"]:
                    raise ValueError("Retained cohort differs")
            cohort_audit.append({"variant": variant, "outer_fold": fold, "pm": pm,
                "train_sample_hash": spec["train_sample_hash"], "test_sample_hash": spec["test_sample_hash"],
                "train_subject_hash": stable_hash(sorted(train.subject_id.astype(str).unique())),
                "test_subject_hash": stable_hash(sorted(test.subject_id.astype(str).unique())),
                "threshold_hash": spec["threshold_hash"], "q_low": spec["q_low"], "q_high": spec["q_high"]})
            if variant == "baseline_14ch" and baseline_reused:
                continue
            summary = low_high.load_resumable_summary(execution, spec)
            if summary is None:
                print(f"fit: {variant} fold={fold} pm={pm}", flush=True)
                def builder(_name, task_type, input_shape, num_outputs, _params):
                    return build_model("lightgbm", task_type, input_shape, num_outputs, config["model_params"])
                summary = low_high.execute_run(execution, spec, model_builder=builder,
                          expected_feature_count=len(names), result_status=config["result_status"])
            participant, metrics = audit_predictions(execution, spec, execution.output_dir / "runs" / spec["run_id"] / "predictions.parquet")
            all_participants.append(participant.assign(variant=variant))
            results.append(dict(summary, variant=variant, model="lightgbm", **metrics))
    audit = pd.DataFrame(cohort_audit)
    if audit.groupby(["outer_fold", "pm"])[["train_sample_hash", "test_sample_hash", "train_subject_hash", "test_subject_hash", "threshold_hash", "q_low", "q_high"]].nunique().ne(1).any().any():
        raise ValueError("Variant cohort or thresholds differ")
    audit.to_csv(output / "cohort_equivalence.csv", index=False)
    participants = pd.concat(all_participants, ignore_index=True)
    participants.to_csv(output / "participant_metrics.csv", index=False)
    pd.DataFrame(results).to_csv(output / "fold_target_results.csv", index=False)
    summary, deltas, ci = aggregate(participants, config)
    summary.to_csv(output / "summary_by_variant.csv", index=False)
    summary.loc[summary.pm.ne("overall")].to_csv(output / "summary_by_target.csv", index=False)
    deltas.to_csv(output / "participant_deltas.csv", index=False)
    ci.to_csv(output / "participant_bootstrap.csv", index=False)
    ci.to_csv(output / "bootstrap_summary.csv", index=False)
    manifest.update(status="complete", baseline_reused=baseline_reused, completed_fold_target_rows=len(results),
                    new_fits=70 if baseline_reused else 105)
    write_json(manifest_path, manifest)
    if variants == VARIANTS:
        write_report(output, cache_root, manifest, summary, ci)
    else:
        from bench.experiments.pm_qc_channel_drop import write_qc_report
        write_qc_report(root, output, manifest, summary, ci)


def main(argv=None, *, config_path: str = CONFIG_PATH):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "materialize", "screen"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    def resolve(p):
        return p if p.is_absolute() else root / p
    with threadpool_limits(limits=1):
        run(root=root, feature_cache=resolve(args.feature_cache_dir), cache_root=resolve(args.cache_root),
            output=resolve(args.output_dir), phase=args.phase, workers=args.workers, config_path=config_path)
    print(f"{args.phase}: PASS", flush=True)
    return 0
