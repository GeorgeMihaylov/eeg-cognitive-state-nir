"""Fixed absolute-window lag confirmation on matched LOW/HIGH PM support.

Absolute lag = EEG.t_start - PM.t_start (also end-to-end for 10s windows).
Lag 0 is contemporaneous, available at PM-window end; +10 is non-causal.
No interpretation as a physiological delay or proprietary PM support is made.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bench.analysis.paired_statistics import subject_bootstrap_interval
from bench.analysis.pm_eeg_lag_sweep import _lag_positions
from bench.experiments import pm_low_high_q3_extremes_confirmatory as low_high
from cogstate.model_zoo import build_model

SCHEMA = "pm-eeg-absolute-lag-v1"
LAGS = (-30, -20, -10, 0, 10)
REFERENCE = -10
FOLD_HASH = "2a176aad988fb814175c9edbf1e08266809f5c0a4ed4bf0dfd987ebf8c43c5dd"
METRICS = ("balanced_accuracy", "macro_f1", "roc_auc", "pr_auc",
           "low_recall", "high_recall", "precision", "accuracy")
PARAMS = {"n_estimators": 200, "n_jobs": 4, "random_state": 42}


def condition(lag: int) -> str:
    if lag not in LAGS:
        raise ValueError("Unsupported absolute lag")
    return f"absolute_{lag:+d}s"


def causal_status(lag: int) -> str:
    condition(lag)
    return "non_causal_diagnostic" if lag > 0 else (
        "contemporaneous_at_pm_window_end" if lag == 0 else "causal_previous_window")


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA, "absolute_lags_s": list(LAGS),
        "reference_lag_s": REFERENCE,
        "target_ids": [f"target_{pm}" for pm in low_high.PM_NAMES],
        "folds": [1, 2, 3, 4, 5], "fixed_fold_hash": FOLD_HASH,
        "model": {"name": "lightgbm", "params": PARAMS},
        "primary_metric": "participant_macro_balanced_accuracy",
        "threshold_fit_scope": "outer_train_continuous_complete_cases",
        "common_support": "target_complete_and_exact_pairs_for_all_five_lags",
        "alignment": "EEG.t_start - PM.t_start; causal cutoff PM.t_end",
        "bootstrap": {"unit": "subject_id", "replicates": 10000, "seed": 42},
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Frozen absolute-lag contract changed: {key}")
    return config


def build_alignment(index: pd.DataFrame):
    """Reuse exact lag lookup, force real t_start rather than nominal window_id."""
    required = ["sample_id", "record_id", "subject_id", "outer_fold", "t_start", "t_end"]
    frame = index[required].reset_index(drop=True).copy()
    if frame.isna().any().any() or frame.sample_id.duplicated().any():
        raise ValueError("Missing alignment identity or duplicate sample_id")
    times = frame[["t_start", "t_end"]].to_numpy(dtype=float)
    if not np.isfinite(times).all() or not np.allclose(
        times[:, 1] - times[:, 0], 10, atol=1e-6, rtol=0
    ):
        raise ValueError("Expected finite exact ten-second windows")
    if frame.groupby("subject_id").outer_fold.nunique().max() != 1:
        raise ValueError("Subject spans outer folds")
    if (frame.groupby("record_id").subject_id.nunique() != 1).any():
        raise ValueError("Record spans subjects")
    positions, common, _ = _lag_positions(frame.drop(columns="t_end"), tuple(x // 10 for x in LAGS))
    positions = {lag: positions[lag // 10] for lag in LAGS}
    audits = []
    for lag, feature_positions in positions.items():
        target_positions = np.flatnonzero(feature_positions >= 0)
        target = frame.iloc[target_positions].reset_index(drop=True)
        eeg = frame.iloc[feature_positions[target_positions]].reset_index(drop=True)
        for key in ("record_id", "subject_id", "outer_fold"):
            if not target[key].equals(eeg[key]):
                raise ValueError(f"Alignment crosses {key}")
        delta = eeg.t_start.to_numpy() - target.t_start.to_numpy()
        if not np.allclose(delta, lag, atol=1e-6, rtol=0):
            raise ValueError("Actual timestamp difference disagrees with absolute lag")
        if lag <= 0 and (eeg.t_end.to_numpy() > target.t_end.to_numpy() + 1e-6).any():
            raise ValueError("EEG extends beyond causal PM-window endpoint")
        rows = pd.DataFrame({
            "target_sample_id": target.sample_id, "feature_sample_id": eeg.sample_id,
            "subject_id": target.subject_id, "record_id": target.record_id,
            "outer_fold": target.outer_fold, "pm_window_start": target.t_start,
            "pm_window_end": target.t_end, "eeg_window_start": eeg.t_start,
            "eeg_window_end": eeg.t_end, "absolute_lag_s": lag,
            "actual_difference_s": delta, "causal_status": causal_status(lag),
            "non_causal_diagnostic": lag > 0,
            "in_common_support": common[target_positions],
        })
        # Real examples from each fold, all chosen from the actual common support.
        audits.append(rows.loc[rows.in_common_support].groupby("outer_fold", sort=True).head(3))
    return positions, common, pd.concat(audits, ignore_index=True)


def build_common_cohorts(full: pd.DataFrame, positions: dict, common: np.ndarray):
    cohorts, audit = {}, []
    for pm in low_high.PM_NAMES:
        valid = np.isfinite(pd.to_numeric(full[f"target_{pm}"], errors="coerce"))
        keep = common & valid.to_numpy()
        cohort = full.loc[keep, ["sample_id", "subject_id", "record_id", "outer_fold"]].copy()
        cohort = cohort.rename(columns={"sample_id": "target_sample_id"}).reset_index(drop=True)
        cohort["continuous_target"] = full.loc[keep, f"target_{pm}"].to_numpy()
        for lag in LAGS:
            prefix = condition(lag)
            selected = positions[lag][keep]
            cohort[f"{prefix}_feature_position"] = selected
            cohort[f"{prefix}_feature_sample_id"] = full.iloc[selected].sample_id.to_numpy()
        cohorts[pm] = cohort
        for fold in sorted(full.outer_fold.unique()):
            for partition in ("train", "test"):
                partition_mask = full.outer_fold.ne(fold) if partition == "train" else full.outer_fold.eq(fold)
                n_complete = int((valid & partition_mask).sum())
                n_common = int((keep & partition_mask).sum())
                for lag in LAGS:
                    before = int((valid & partition_mask & (positions[lag] >= 0)).sum())
                    audit.append({"target": f"target_{pm}", "fold": int(fold),
                                  "partition": partition, "absolute_lag_s": lag,
                                  "n_complete_cases": n_complete, "n_before_intersection": before,
                                  "n_common_support": n_common,
                                  "lost_fraction_vs_lag": 1 - n_common / before if before else np.nan,
                                  "lost_fraction_vs_complete": 1 - n_common / n_complete if n_complete else np.nan})
    return cohorts, pd.DataFrame(audit)


@dataclass
class AbsoluteContext:
    base: low_high.ProtocolContext
    config: dict
    common_audit: pd.DataFrame
    alignment_audit: pd.DataFrame


def prepare_protocol(config, *, root, feature_cache_dir, output_dir=None, smoke=False):
    root = Path(root).resolve()
    reference = low_high.load_config(root / config["reference_config"])
    base = low_high.prepare_protocol(reference, root=root, feature_cache_dir=feature_cache_dir,
                                    output_dir=output_dir or config["output_dir"])
    if base.protocol["fixed_fold_hash"] != config["fixed_fold_hash"]:
        raise ValueError("Fixed-fold identity changed")
    if base.temporal_pairing_summary["matched_target_rows"] != 30806:
        raise ValueError("Canonical previous-window audit changed; investigate before training")
    positions, common, alignment = build_alignment(base.feature_index)
    cohorts, common_audit = build_common_cohorts(base.full, positions, common)
    # Preserve the established LOW/HIGH fit universe: all finite outer-train PM.
    # Intersection changes retained examples, never thresholds between lags.
    transforms, thresholds = low_high.build_threshold_audit(base.full, cohorts, config["folds"])
    params = dict(PARAMS)
    if smoke:
        params["n_estimators"] = 10
    scientific = {k: v for k, v in config.items() if k != "output_dir"}
    input_columns = ["sample_id", "subject_id", "record_id", "outer_fold", "t_start", "t_end",
                     *config["target_ids"]]
    input_hash = hashlib.sha256(pd.util.hash_pandas_object(
        base.full[input_columns], index=False).to_numpy().tobytes()).hexdigest()
    source_paths = [Path(__file__), Path(low_high.__file__),
                    root / "bench/analysis/pm_eeg_lag_sweep.py",
                    root / "bench/analysis/paired_statistics.py",
                    root / "cogstate/model_zoo/ML/sklearn_models.py"]
    source_hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source_paths}
    versions = {name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scikit-learn", "lightgbm")}
    model = build_model("lightgbm", "classification", (371,), 2, params)
    protocol = {"schema_version": SCHEMA, "experiment_id": config["experiment_id"],
                "scientific_config": scientific, "mode": "smoke" if smoke else "confirmatory",
                "model_params": params, "effective_model_params": model.get_params(),
                "reference_lag_s": REFERENCE, "input_table_hash": input_hash,
                "feature_values_sha256": hashlib.sha256(memoryview(base.matrix).cast("B")).hexdigest(),
                "cache_identity": base.cache_identity, "fixed_fold_hash": config["fixed_fold_hash"],
                "source_hashes": source_hashes, "package_versions": versions,
                "threshold_hashes": thresholds.threshold_hash.tolist(),
                "common_sample_hashes": {pm: low_high._sample_hash(c.target_sample_id) for pm, c in cohorts.items()},
                "pairing_hash": low_high.stable_hash({str(lag): positions[lag][common].tolist() for lag in LAGS}),
                "planned_fits": 5 if smoke else 175, "auxiliary_fits": 0,
                "canonical_lag_changed": False,
                "causal_cutoff": "PM-window end; lag 0 includes contemporaneous PM window",
                "preprocessing": "reuse unchanged canonical cached features; no new fit-like preprocessing"}
    protocol["protocol_hash"] = low_high.stable_hash(protocol)
    protocol["git_commit"] = low_high._git_head(root)
    specs = []
    for row in thresholds.to_dict("records"):
        fold, pm = int(row["outer_fold"]), row["pm"]
        if smoke and (fold != 1 or pm != "attention"):
            continue
        cohort = cohorts[pm]
        retained = np.isfinite(transforms[(fold, pm)].transform(cohort.continuous_target))
        n_train_subjects = cohort.loc[retained & cohort.outer_fold.ne(fold), "subject_id"].nunique()
        for lag in LAGS:
            spec = {"outer_fold": fold, "pm": pm, "target_id": f"target_{pm}",
                    "condition": condition(lag), "lag_seconds": lag, "absolute_lag_s": lag,
                    "reference_lag_s": REFERENCE, "model": "lightgbm", "seed": 42,
                    "non_causal_diagnostic": lag > 0, "causal_status": causal_status(lag),
                    "q_low": row["q_low"], "q_high": row["q_high"],
                    "threshold_hash": row["threshold_hash"],
                    "n_train": row["n_train_retained"], "n_test": row["n_test_retained"],
                    "n_train_subjects": int(n_train_subjects),
                    "n_test_subjects": row["n_test_subjects_retained"],
                    "train_sample_hash": row["train_retained_sample_hash"],
                    "test_sample_hash": row["test_retained_sample_hash"]}
            spec["specification_hash"] = low_high.stable_hash({"protocol": protocol["protocol_hash"], "spec": spec})
            spec["run_id"] = f"fold_{fold:02d}__{pm}__{condition(lag)}__{spec['specification_hash'][:12]}"
            specs.append(spec)
    run_matrix = pd.DataFrame(specs)
    if len(run_matrix) != protocol["planned_fits"] or run_matrix.run_id.duplicated().any():
        raise ValueError("Invalid scientific run matrix")
    for _, group in run_matrix.groupby(["pm", "outer_fold"]):
        if tuple(group.absolute_lag_s) != LAGS:
            raise ValueError("Each cell must include all five lags")
        if (group[["train_sample_hash", "test_sample_hash", "threshold_hash"]].nunique() != 1).any():
            raise ValueError("Lags differ in support or thresholds")
    base = replace(base, config={**base.config, "model": {"name": "lightgbm", "params": params}},
                   cohorts=cohorts, transforms=transforms, threshold_audit=thresholds,
                   protocol=protocol, run_matrix=run_matrix)
    return AbsoluteContext(base, dict(config), common_audit, alignment)


def write_dry_run(context):
    base = context.base
    output = base.output_dir
    old = output / "protocol.json"
    if old.exists() and json.loads(old.read_text(encoding="utf-8"))["protocol_hash"] != base.protocol["protocol_hash"]:
        # A preflight can be refreshed before training; never relabel existing results.
        if (output / "runs").exists() or (output / "execution_status.json").exists():
            raise ValueError("Output belongs to another protocol; use a fresh output directory")
    low_high._atomic_json(old, base.protocol)
    for name, frame in (("run_matrix", base.run_matrix), ("threshold_audit", base.threshold_audit),
                        ("common_support_audit", context.common_audit), ("alignment_audit", context.alignment_audit)):
        low_high._write_csv(output / f"{name}.csv", frame)
    summary = {"protocol_hash": base.protocol["protocol_hash"], "planned_fits": len(base.run_matrix),
               "training_executed": False, "reference_lag_s": REFERENCE,
               "canonical_rows": len(base.full),
               "common_complete_cases": {pm: len(c) for pm, c in base.cohorts.items()},
               "cross_record_pairs": 0, "cross_subject_pairs": 0, "cross_fold_pairs": 0,
               "identical_evaluation_examples_between_lags": True}
    low_high._atomic_json(output / "dry_run_summary.json", summary)
    return summary


def paired_deltas(frame, keys):
    if frame.duplicated([*keys, "absolute_lag_s"]).any():
        raise ValueError("Duplicate paired metric key")
    reference = frame.loc[frame.absolute_lag_s.eq(REFERENCE), [*keys, *METRICS]]
    result = frame.merge(reference, on=keys, how="left", suffixes=("", "_reference"), validate="many_to_one", indicator=True)
    if not result._merge.eq("both").all():
        raise ValueError("Missing -10 reference")
    result = result.drop(columns="_merge")
    for metric in METRICS:
        result[f"delta_{metric}_vs_minus10"] = result[metric] - result[f"{metric}_reference"]
        # Reference identity is exactly zero even when the metric is undefined.
        # Undefined reference metrics themselves remain NaN with valid counts 0.
        result.loc[result.absolute_lag_s.eq(REFERENCE), f"delta_{metric}_vs_minus10"] = 0.0
    return result


def aggregate_results(context):
    base = context.base
    participants, folds, evaluation_hashes = [], [], {}
    for spec in base.run_matrix.to_dict("records"):
        run_dir = base.output_dir / "runs" / spec["run_id"]
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        participant = pd.read_csv(run_dir / "participant_metrics.csv", dtype={"subject_id": str})
        participant["target"] = spec["target_id"]
        participant["absolute_lag_s"] = spec["lag_seconds"]
        participant["causal_status"] = spec["causal_status"]
        participants.append(participant)
        prediction = pd.read_parquet(run_dir / "predictions.parquet")
        pairing_key = (spec["target_id"], spec["outer_fold"])
        evaluation_hash = low_high.stable_hash(prediction[
            ["target_sample_id", "subject_id", "record_id", "outer_fold", "y_true"]
        ].astype(str).to_dict("records"))
        if evaluation_hashes.setdefault(pairing_key, evaluation_hash) != evaluation_hash:
            raise ValueError("Evaluation examples or labels differ between lags")
        folds.append({"target": spec["target_id"], "fold": spec["outer_fold"],
                      "absolute_lag_s": spec["lag_seconds"], "causal_status": spec["causal_status"],
                      "n_train_subjects": spec["n_train_subjects"], "n_test_subjects": len(participant),
                      "n_train_examples": spec["n_train"], "n_test_examples": spec["n_test"],
                      "n_records": prediction.record_id.nunique(),
                      **{m: summary[f"participant_macro_{'f1' if m == 'macro_f1' else m}"] for m in METRICS},
                      **{f"valid_{m}": int(participant[m].notna().sum()) for m in METRICS}})
    participants = pd.concat(participants, ignore_index=True)
    if participants.duplicated(["target", "absolute_lag_s", "subject_id"]).any():
        raise ValueError("Participants appear in multiple outer tests")
    folds = paired_deltas(pd.DataFrame(folds), ["target", "fold"])
    paired = paired_deltas(participants, ["target", "outer_fold", "subject_id"])
    totals, intervals = [], []
    for (target, lag), group in paired.groupby(["target", "absolute_lag_s"], sort=False):
        f = folds.loc[folds.target.eq(target) & folds.absolute_lag_s.eq(lag)]
        row = {"target": target, "absolute_lag_s": lag, "n_subjects": len(group),
               "n_records": int(f.n_records.sum()), "n_examples": int(f.n_test_examples.sum()),
               "causal_status": causal_status(lag), "non_causal_diagnostic": lag > 0,
               "reference_lag_s": REFERENCE, "result_status": base.protocol["mode"]}
        for metric in METRICS:
            delta = f"delta_{metric}_vs_minus10"
            row[metric] = group[metric].mean()
            row[f"valid_{metric}"] = int(group[metric].notna().sum())
            row[delta] = group[delta].mean()
            row[f"fold_mean_{delta}"] = f[delta].mean()
            row[f"fold_std_{delta}"] = f[delta].std(ddof=1)
            row[f"favorable_folds_{metric}"] = int(f[delta].gt(0).sum())
            valid_pair = group[metric].notna() & group[f"{metric}_reference"].notna()
            intervals.append({"target": target, "absolute_lag_s": lag, "metric": metric,
                              "resampling_unit": "subject_id", "replicates": 10000, "seed": 42,
                              **subject_bootstrap_interval(group.loc[valid_pair, delta])})
        totals.append(row)
    total = pd.DataFrame(totals)
    intervals = pd.DataFrame(intervals)
    for name, frame in (("overall", total), ("fold_metrics", folds), ("participant_metrics", participants),
                        ("participant_paired_deltas", paired), ("participant_bootstrap", intervals)):
        low_high._write_csv(base.output_dir / f"{name}.csv", frame)
    plot_results(total, base.output_dir, base.protocol["mode"])
    report_results(context, total, intervals)


def report_results(context, total, intervals):
    """Descriptive evidence for the four questions, without choosing a new lag."""
    base = context.base
    lines = ["# Absolute EEG-window lag / LOW-HIGH PM", "",
             f"Status: **{base.protocol['mode']}**. Reference: **-10 s, retrained on common support**.",
             f"Protocol: `{base.protocol['protocol_hash']}`.", "",
             "Primary endpoint is equal-weight participant balanced accuracy. Deltas below are",
             "alternative minus -10. Fold spread is sample SD across paired fold deltas.",
             "95% percentile CIs resample participants (10,000 draws, seed 42), never windows.",
             "CIs are descriptive and unadjusted for multiple comparisons; training folds overlap.", "",
             "| PM | Lag (s) | Participant delta BA | 95% CI | Fold mean / SD | Favorable folds | Status |",
             "|---|---:|---:|---|---|---:|---|"]
    for row in total.itertuples(index=False):
        if row.absolute_lag_s == REFERENCE:
            continue
        ci = intervals.loc[intervals.target.eq(row.target)
                           & intervals.absolute_lag_s.eq(row.absolute_lag_s)
                           & intervals.metric.eq("balanced_accuracy")].iloc[0]
        lines.append(f"| {row.target} | {row.absolute_lag_s:+d} | {row.delta_balanced_accuracy_vs_minus10:+.6f} | "
                     f"[{ci.ci_low:+.6f}, {ci.ci_high:+.6f}] | "
                     f"{row.fold_mean_delta_balanced_accuracy_vs_minus10:+.6f} / "
                     f"{row.fold_std_delta_balanced_accuracy_vs_minus10:.6f} | "
                     f"{row.favorable_folds_balanced_accuracy} | {row.causal_status} |")
    lines += ["", "## Interpretation rules", "",
              "A. Assess the canonical -10 reference using all five folds and paired participant intervals;",
              "this experiment never automatically changes the canonical project alignment.",
              "B. For -20/-30, examine both paired effect size and consistency across folds for each PM.",
              "C. The 0 column measures contemporaneous-window differences relative to -10.",
              "D. Positive +10 differences are non-causal diagnostic evidence only. A conspicuously high",
              "+10 result warrants a separate audit of smoothing, temporal overlap and timestamp semantics;",
              "these are hypotheses, not established mechanisms or a production recommendation.", "",
              "Absolute lag means EEG-window start minus PM-window start. Lag 0 is available only at",
              "PM-window end and is not a prediction at PM-window start. Proprietary PM internal support",
              "and physiological delays are not established by this comparison.", "",
              "LOW/HIGH are derived PM proxies. Thresholds use only outer-train continuous complete cases",
              "under the existing contract; train/test retained support and labels are identical across lags.",
              "Single-class participant AUC/AP remain undefined, with metric-specific valid counts."]
    if base.protocol["mode"] == "smoke":
        lines += ["", "**Smoke only: one PM, one fold, ten trees. No scientific lag conclusion is warranted.**"]
    (base.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(total, output, status):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    targets = [f"target_{pm}" for pm in low_high.PM_NAMES if f"target_{pm}" in set(total.target)]
    fig, axes = plt.subplots(len(targets), 1, figsize=(8, 2.4 * len(targets)), squeeze=False)
    for target, ax in zip(targets, axes.flat):
        rows = total.loc[total.target.eq(target)].set_index("absolute_lag_s").reindex(LAGS)
        ax.plot(LAGS, rows.balanced_accuracy, marker="o")
        ax.axvline(REFERENCE, color="green", linestyle="--", label="-10s PRIMARY REFERENCE")
        ax.axvspan(5, 15, color="orange", alpha=.2, label="+10s non-causal diagnostic")
        ax.set(title=f"{target} ({status})", ylabel="Participant-macro BA", xticks=LAGS,
               xlabel="Absolute EEG lag (s)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "balanced_accuracy_vs_lag.png", dpi=160)
    plt.close(fig)
    values = total.pivot(index="target", columns="absolute_lag_s", values="delta_balanced_accuracy_vs_minus10").reindex(index=targets, columns=LAGS)
    if not values[REFERENCE].eq(0).all():
        raise ValueError("Reference heatmap column must be exactly zero")
    fig, ax = plt.subplots(figsize=(9, max(2.8, .6 * len(targets))))
    limit = max(.001, float(np.nanmax(np.abs(values.to_numpy()))))
    shown = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set(xticks=range(5), xticklabels=["-30", "-20", "-10 REFERENCE", "0", "+10 NON-CAUSAL"],
           yticks=range(len(targets)), yticklabels=targets, title=f"Paired participant BA delta vs -10s ({status})")
    for i in range(len(targets)):
        for j in range(5):
            ax.text(j, i, f"{values.iloc[i, j]:+.4f}", ha="center", va="center")
    fig.colorbar(shown, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "delta_heatmap.png", dpi=160)
    plt.close(fig)


def run_experiment(context):
    base = context.base
    executed = resumed = 0
    for spec in base.run_matrix.to_dict("records"):
        run_dir = base.output_dir / "runs" / spec["run_id"]
        cached = low_high.load_resumable_summary(base, spec)
        if cached is not None:
            # Completed cells must not be silently retrained or accepted after corruption.
            for name, expected in cached.get("artifact_sha256", {}).items():
                if hashlib.sha256((run_dir / name).read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Corrupted completed artifact: {run_dir / name}")
            if len(cached.get("artifact_sha256", {})) != 2:
                raise ValueError("Completed run lacks artifact checksums")
            resumed += 1
            continue
        if (run_dir / "run_summary.json").exists():
            raise ValueError(f"Incompatible existing run: {run_dir}")
        summary = low_high.execute_binary_run(
            base, spec, feature_prefix=condition(spec["lag_seconds"]), model_name="lightgbm",
            result_status=base.protocol["mode"])
        summary.update({"absolute_lag_s": spec["lag_seconds"], "reference_lag_s": REFERENCE,
                        "non_causal_diagnostic": spec["non_causal_diagnostic"],
                        "causal_status": spec["causal_status"], "model": "lightgbm",
                        "artifact_sha256": {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
                                            for name in ("predictions.parquet", "participant_metrics.csv")}})
        low_high._atomic_json(run_dir / "run_summary.json", summary)
        executed += 1
        print(f"Complete {executed + resumed}/{len(base.run_matrix)}: {spec['run_id']}", flush=True)
    aggregate_results(context)
    status = {"executed": executed, "resumed": resumed, "complete": executed + resumed,
              "result_status": base.protocol["mode"], "protocol_hash": base.protocol["protocol_hash"]}
    low_high._atomic_json(base.output_dir / "execution_status.json", status)
    return status
