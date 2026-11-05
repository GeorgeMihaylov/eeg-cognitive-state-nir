"""Frozen history preflight and bounded smoke; the full benchmark is not enabled."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bench.analysis.pm_eeg_lag_sweep import _lag_positions
from bench.analysis.paired_statistics import subject_bootstrap_interval
from bench.experiments import pm_low_high_q3_extremes_confirmatory as low

HORIZONS = {10: (-1,), 20: (-2, -1), 30: (-3, -2, -1), 60: tuple(range(-6, 0)), 120: tuple(range(-12, 0))}
PRIMARY = "strict_common_support_h120"
SECONDARY = "horizon_specific_support"
PARAMS = {"n_estimators": 200, "n_jobs": 4, "random_state": 42}
FOLD_HASH = "2a176aad988fb814175c9edbf1e08266809f5c0a4ed4bf0dfd987ebf8c43c5dd"


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"horizons_s": list(HORIZONS), "model_params": PARAMS,
                "target_ids": [f"target_{pm}" for pm in low.PM_NAMES], "folds": [1, 2, 3, 4, 5],
                "primary_support": PRIMARY, "secondary_support": SECONDARY, "fixed_fold_hash": FOLD_HASH,
                "primary_metric": "participant_macro_balanced_accuracy", "history_order_control": "ordered",
                "threshold_fit_scope": "outer_train_continuous_complete_cases", "full_run_enabled": False}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Frozen history contract changed: {key}")
    return config


def history_positions(frame: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    """Exact historical lookup; all intervening blocks required for each horizon."""
    keys = ["sample_id", "record_id", "subject_id", "outer_fold", "t_start", "t_end"]
    if frame[keys].isna().any().any() or frame.sample_id.duplicated().any():
        raise ValueError("Missing or duplicate history identity")
    if frame.groupby("subject_id").outer_fold.nunique().max() != 1:
        raise ValueError("Subject crosses fold boundary")
    if (frame.groupby("record_id").subject_id.nunique() != 1).any():
        raise ValueError("Record crosses subject boundary")
    times = frame[["t_start", "t_end"]].to_numpy(dtype=float)
    if not np.isfinite(times).all() or not np.allclose(times[:, 1]-times[:, 0], 10, atol=1e-6, rtol=0):
        raise ValueError("Expected exact finite ten-second windows")
    positions, _, coordinate = _lag_positions(frame[["record_id", "t_start"]], HORIZONS[120])
    audit = []
    for lag, pos in positions.items():
        valid = pos >= 0
        target = frame.loc[valid].reset_index(drop=True)
        history = frame.iloc[pos[valid]].reset_index(drop=True)
        for key in ("record_id", "subject_id", "outer_fold"):
            if not target[key].equals(history[key]):
                raise ValueError(f"History crosses {key}")
        if not np.allclose(history.t_start-target.t_start, lag*10, atol=1e-6, rtol=0):
            raise ValueError("Actual history timestamps disagree with lag")
        if (history.t_end > target.t_start + 1e-6).any():
            raise ValueError("History overlaps PM/future")
        for i in range(min(3, len(target))):
            audit.append({"lag_s": lag*10, "target_sample_id": target.sample_id.iloc[i],
                          "history_sample_id": history.sample_id.iloc[i], "record_id": target.record_id.iloc[i],
                          "pm_start": target.t_start.iloc[i], "eeg_start": history.t_start.iloc[i],
                          "eeg_end": history.t_end.iloc[i], "coordinate": coordinate})
    masks = {h: np.logical_and.reduce([positions[lag] >= 0 for lag in offsets]) for h, offsets in HORIZONS.items()}
    for a, b in zip(list(HORIZONS)[:-1], list(HORIZONS)[1:]):
        if (masks[b] & ~masks[a]).any():
            raise ValueError("Non-nested history support")
    return positions, masks, pd.DataFrame(audit)


def feature_names(names: list[str], horizon: int) -> list[str]:
    return [f"lag_m{-10*lag}__{name}" for lag in HORIZONS[horizon] for name in names]


def history_matrix(matrix: np.ndarray, positions: dict, target_positions: np.ndarray,
                   horizon: int, *, shuffled: bool = False, sample_ids=None) -> np.ndarray:
    """Oldest first. Shuffle is independent per example, never a global column permutation."""
    indices = np.column_stack([positions[lag][target_positions] for lag in HORIZONS[horizon]])
    if (indices < 0).any():
        raise ValueError("Missing exact historical block; no substitution allowed")
    if shuffled:
        if sample_ids is None or len(sample_ids) != len(indices):
            raise ValueError("Stable target sample IDs required for shuffle")
        for i, sample_id in enumerate(sample_ids):
            digest = hashlib.sha256(f"42|{sample_id}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            indices[i, :-1] = rng.permutation(indices[i, :-1])
    return np.asarray(matrix[indices].reshape(len(indices), -1), dtype=np.float32)


def cohort_for(full: pd.DataFrame, mask: np.ndarray, pm: str) -> pd.DataFrame:
    valid = mask & np.isfinite(full[f"target_{pm}"].to_numpy(dtype=float))
    columns = ["sample_id", "subject_id", "record_id", "outer_fold"]
    columns += [c for c in ("source", "record_group_id", "session_id") if c in full]
    cohort = full.loc[valid, columns].rename(columns={"sample_id": "target_sample_id"}).reset_index(drop=True)
    cohort["continuous_target"] = full.loc[valid, f"target_{pm}"].to_numpy()
    cohort["target_position"] = np.flatnonzero(valid)
    return cohort


def support_identity(cohort: pd.DataFrame, labels: np.ndarray, keep: np.ndarray) -> dict:
    """Ordered hashes bind each label/group/fold to the same target row order."""
    return {"sample_hash": low._sample_hash(cohort.loc[keep, "target_sample_id"]),
            "ordered_sample_hash": low.stable_hash(cohort.loc[keep, "target_sample_id"].astype(str).tolist()),
            "label_hash": low.stable_hash(labels[keep].astype(int).tolist()),
            "participant_hash": low.stable_hash(cohort.loc[keep, "subject_id"].astype(str).tolist()),
            "fold_hash": low.stable_hash(cohort.loc[keep, "outer_fold"].astype(int).tolist())}


@dataclass
class HistoryContext:
    base: low.ProtocolContext
    config: dict
    positions: dict
    masks: dict
    protocol: dict
    run_matrix: pd.DataFrame
    support_audit: pd.DataFrame
    alignment_audit: pd.DataFrame
    output: Path


def prepare_protocol(config: dict, *, root: Path, feature_cache_dir: Path, output_dir=None, smoke=False) -> HistoryContext:
    root = Path(root).resolve()
    reference = low.load_config(root / config["reference_config"])
    base = low.prepare_protocol(reference, root=root, feature_cache_dir=feature_cache_dir,
                                output_dir=root / "benchmark_results/_temporal_prediction_reference")
    output = root / (output_dir or (str(config["output_dir"]) + ("/smoke" if smoke else "")))
    return prepare_from_base(base, config, output, smoke=smoke)


def prepare_from_base(base: low.ProtocolContext, config: dict, output: Path, *, smoke=False) -> HistoryContext:
    if base.protocol["fixed_fold_hash"] != config["fixed_fold_hash"]:
        raise ValueError("Fixed folds changed")
    positions, masks, alignment = history_positions(base.full)
    params = {**PARAMS, "n_estimators": 10 if smoke else 200}
    input_columns = ["sample_id", "subject_id", "record_id", "outer_fold", "t_start", "t_end", *config["target_ids"]]
    protocol = {"schema_version": "pm-temporal-predictability-v1", "experiment_type": "diagnostic" if smoke else "confirmatory_planned",
        "mode": "smoke" if smoke else "dry_run", "config": config, "params": params,
        "reference_protocol_hash": base.protocol["protocol_hash"], "cache_identity": base.cache_identity,
        "input_table_sha256": hashlib.sha256(pd.util.hash_pandas_object(base.full[input_columns], index=False).to_numpy().tobytes()).hexdigest(),
        "feature_values_sha256": hashlib.sha256(memoryview(base.matrix).cast("B")).hexdigest(),
        "feature_names_hash": low.stable_hash(base.feature_names),
        "package_versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm")},
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [Path(__file__), Path(low.__file__), base.root / "bench/analysis/pm_eeg_lag_sweep.py",
                           base.root / "bench/analysis/paired_statistics.py", base.root / "cogstate/model_zoo/ML/sklearn_models.py"]},
        "primary_planned_fits": 5 if smoke else 175, "secondary_planned_fits": 0 if smoke else 175,
        "secondary_execution": "sensitivity only; not included in primary 175; no execution enabled",
        "canonical_lag_s": -10, "block_order": "oldest to newest; canonical -10 last",
        "threshold_scope": "reuse full canonical outer-train complete-case thresholds before history filtering",
        "git_commit": low._git_head(base.root), "training_executed": False}
    # Git HEAD is provenance, not an integrity key that invalidates unchanged code.
    protocol["protocol_hash"] = low.stable_hash({k:v for k,v in protocol.items() if k != "git_commit"})
    specs, support = [], []
    threshold_rows = base.threshold_audit.set_index(["outer_fold", "pm"])
    for mode in ([PRIMARY] if smoke else [PRIMARY, SECONDARY]):
        for pm in low.PM_NAMES:
            for fold in config["folds"]:
                if smoke and (pm != "attention" or fold != 1):
                    continue
                threshold = base.transforms[(fold, pm)]
                threshold_hash = threshold_rows.loc[(fold, pm), "threshold_hash"]
                for horizon in HORIZONS:
                    cohort = cohort_for(base.full, masks[120 if mode == PRIMARY else horizon], pm)
                    labels = threshold.transform(cohort.continuous_target.to_numpy())
                    spec = {"pm": pm, "target_id": f"target_{pm}", "outer_fold": fold,
                            "horizon_s": horizon, "support_mode": mode, "condition": f"H{horizon}", "lag_seconds": -10,
                            "feature_count": len(base.feature_names)*len(HORIZONS[horizon]),
                            "q_low": threshold.q_low, "q_high": threshold.q_high, "threshold_hash": threshold_hash}
                    for partition in ("train", "test"):
                        in_partition = cohort.outer_fold.ne(fold).to_numpy() if partition == "train" else cohort.outer_fold.eq(fold).to_numpy()
                        keep = in_partition & np.isfinite(labels)
                        identities = support_identity(cohort, labels, keep)
                        spec.update({f"{partition}_{k}": v for k,v in identities.items()})
                        spec[f"n_{partition}"] = int(keep.sum())
                        if not keep.any():
                            raise ValueError("Empty retained history partition")
                        base_cohort = cohort_for(base.full, masks[10], pm)
                        base_labels = threshold.transform(base_cohort.continuous_target.to_numpy())
                        baseline_part = base_cohort.outer_fold.ne(fold) if partition == "train" else base_cohort.outer_fold.eq(fold)
                        denominator = int((baseline_part & np.isfinite(base_labels)).sum())
                        support.append({"support_mode": mode, "pm": pm, "outer_fold": fold, "horizon_s": horizon,
                            "partition": partition, "n_samples": int(keep.sum()), "n_subjects": cohort.loc[keep,"subject_id"].nunique(),
                            "n_records": cohort.loc[keep,"record_id"].nunique(), "retention_vs_h10": keep.sum()/denominator})
                    spec["specification_hash"] = low.stable_hash({"protocol_hash": protocol["protocol_hash"], **spec})
                    spec["run_id"] = f"{mode}__{pm}__fold{fold}__H{horizon}__{spec['specification_hash'][:12]}"
                    specs.append(spec)
    runs = pd.DataFrame(specs)
    primary = runs.loc[runs.support_mode.eq(PRIMARY)]
    hash_columns = [c for c in primary if c.endswith("_hash") and c != "specification_hash"]
    if (primary.groupby(["pm", "outer_fold"])[hash_columns].nunique() != 1).any().any():
        raise ValueError("Primary horizons differ in samples, labels, participants, folds or thresholds")
    if len(primary) != protocol["primary_planned_fits"]:
        raise ValueError("Wrong primary run count")
    return HistoryContext(base, config, positions, masks, protocol, runs, pd.DataFrame(support), alignment, Path(output))


def write_dry_run(context: HistoryContext) -> dict:
    output = context.output
    old = output / "protocol.json"
    if old.exists() and json.loads(old.read_text())["protocol_hash"] != context.protocol["protocol_hash"]:
        if (output / "runs").exists() or (output / "execution_status.json").exists():
            raise ValueError("Existing results have different protocol; use new namespace")
    low._atomic_json(old, context.protocol)
    for name, frame in [("run_matrix", context.run_matrix), ("support_audit", context.support_audit),
                        ("alignment_audit", context.alignment_audit), ("threshold_audit", context.base.threshold_audit)]:
        low._write_csv(output / f"{name}.csv", frame)
    for h in HORIZONS:
        low._atomic_json(output / f"H{h}_feature_names.json", {"names": feature_names(context.base.feature_names, h)})
    summary = {"protocol_hash": context.protocol["protocol_hash"], "training_executed": False,
               "primary_planned_fits": context.protocol["primary_planned_fits"],
               "secondary_planned_fits": context.protocol["secondary_planned_fits"],
               "canonical_rows": len(context.base.full), "history_support": {str(h): int(m.sum()) for h,m in context.masks.items()},
               "primary_identity_hashes_equal": True, "thresholds_horizon_invariant": True,
               "cross_subject_record_fold_pairs": 0, "full_run_enabled": False}
    low._atomic_json(output / "dry_run_summary.json", summary)
    return summary


def paired_history_results(participants: pd.DataFrame, *, bootstrap: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair within PM/participant/support mode; secondary remains sensitivity only."""
    keys = ["support_mode", "pm", "subject_id", "outer_fold"]
    if participants.duplicated([*keys, "horizon_s"]).any():
        raise ValueError("Duplicate participant history results")
    metrics = [m for m in ("balanced_accuracy", "macro_f1", "roc_auc", "pr_auc") if m in participants]
    reference = participants.loc[participants.horizon_s.eq(10), [*keys, *metrics]]
    paired = participants.merge(reference, on=keys, suffixes=("", "_h10"), how="left", validate="many_to_one", indicator=True)
    if not paired._merge.eq("both").all():
        raise ValueError("Missing H10 participant reference")
    paired = paired.drop(columns="_merge")
    for metric in metrics:
        paired[f"delta_{metric}"] = paired[metric] - paired[f"{metric}_h10"]
    intervals = []
    if bootstrap:
        for (mode, pm, horizon), group in paired.groupby(["support_mode", "pm", "horizon_s"]):
            if group.subject_id.duplicated().any():
                raise ValueError("Bootstrap requires one row per independent participant")
            for metric in metrics:
                intervals.append({"support_mode": mode, "pm": pm, "horizon_s": horizon, "metric": metric,
                    **subject_bootstrap_interval(group[f"delta_{metric}"], n_resamples=10000, random_state=42)})
    return paired, pd.DataFrame(intervals)


def run_smoke(context: HistoryContext) -> dict:
    """Hard limit: one Attention/fold-1, five-horizon, ten-tree smoke only."""
    runs = context.run_matrix
    if (context.protocol["mode"] != "smoke" or len(runs) != 5 or not runs.pm.eq("attention").all()
        or not runs.outer_fold.eq(1).all() or not runs.support_mode.eq(PRIMARY).all()
        or context.protocol["params"] != {**PARAMS, "n_estimators": 10}
        or runs.horizon_s.tolist() != list(HORIZONS)):
        raise ValueError("Only the five-fit Attention/fold1 smoke is enabled")
    summaries, participants = [], []
    executed = resumed = 0
    base = context.base
    if hashlib.sha256(memoryview(base.matrix).cast("B")).hexdigest() != context.protocol["feature_values_sha256"]:
        raise ValueError("Feature values changed since preflight")
    for spec in runs.to_dict("records"):
        horizon = spec["horizon_s"]
        cohort = cohort_for(base.full, context.masks[120], "attention")
        targets = cohort.target_position.to_numpy()
        labels = base.transforms[(1, "attention")].transform(cohort.continuous_target.to_numpy())
        for partition in ("train", "test"):
            mask = (cohort.outer_fold.ne(1) if partition == "train" else cohort.outer_fold.eq(1)).to_numpy() & np.isfinite(labels)
            for key, value in support_identity(cohort, labels, mask).items():
                if spec[f"{partition}_{key}"] != value:
                    raise ValueError("Runtime history identity mismatch")
        matrix = history_matrix(base.matrix, context.positions, targets, horizon)
        cohort["history_feature_position"] = np.arange(len(cohort))
        cohort["history_feature_sample_id"] = [
            "|".join(base.full.iloc[[context.positions[lag][i] for lag in HORIZONS[horizon]]].sample_id.astype(str)) for i in targets]
        view = replace(base, output_dir=context.output, matrix=matrix, feature_names=feature_names(base.feature_names, horizon),
                       cache_identity={**base.cache_identity, "n_features": matrix.shape[1], "derived_history_view": True},
                       cohorts={"attention": cohort}, protocol=context.protocol,
                       config={**base.config, "model": {"name": "lightgbm", "params": context.protocol["params"]}})
        run_dir = context.output / "runs" / spec["run_id"]
        if run_dir.exists():
            summary = low.load_resumable_summary(view, spec)
            if summary is None or len(summary.get("artifact_sha256", {})) != 3:
                raise ValueError("Incomplete or incompatible smoke artifact; no silent retraining")
            for name, digest in summary["artifact_sha256"].items():
                if hashlib.sha256((run_dir/name).read_bytes()).hexdigest() != digest:
                    raise ValueError("Corrupt smoke artifact")
            resumed += 1
        else:
            run_dir.mkdir(parents=True, exist_ok=False)
            provenance = cohort.copy()
            for lag in HORIZONS[horizon]:
                provenance[f"lag_m{-lag*10}_sample_id"] = base.full.iloc[context.positions[lag][targets]].sample_id.to_numpy()
            provenance.to_parquet(run_dir / "history_provenance.parquet", index=False)
            summary = low.execute_binary_run(view, spec, feature_prefix="history", model_name="lightgbm",
                                             expected_feature_count=matrix.shape[1], result_status="smoke_only")
            summary.update(horizon_s=horizon, support_mode=PRIMARY, artifact_sha256={
                name: hashlib.sha256((run_dir/name).read_bytes()).hexdigest() for name in ("predictions.parquet", "participant_metrics.csv", "history_provenance.parquet")})
            low._atomic_json(run_dir / "run_summary.json", summary)
            executed += 1
        summaries.append(summary)
        participants.append(pd.read_csv(run_dir / "participant_metrics.csv").assign(horizon_s=horizon, support_mode=PRIMARY))
        print(f"Smoke H{horizon}: complete ({executed} executed, {resumed} restored)", flush=True)
    low._write_csv(context.output / "fold_metrics.csv", pd.DataFrame(summaries))
    participant_frame = pd.concat(participants, ignore_index=True)
    low._write_csv(context.output / "participant_metrics.csv", participant_frame)
    paired, _ = paired_history_results(participant_frame)
    low._write_csv(context.output / "participant_paired_deltas.csv", paired)
    status = {"executed": executed, "resumed": resumed, "complete": len(summaries), "full_run_executed": False,
              "result_status": "smoke_only_not_scientific_evidence", "protocol_hash": context.protocol["protocol_hash"]}
    low._atomic_json(context.output / "execution_status.json", status)
    return status
