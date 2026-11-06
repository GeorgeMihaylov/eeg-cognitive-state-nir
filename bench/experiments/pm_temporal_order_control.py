"""Prespecified H120 order control conditional on the matched history results."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from bench.experiments import pm_temporal_predictability as history
from bench.experiments import pm_low_high_q3_extremes_confirmatory as low

PMS = ("excitement", "focus", "interest", "relaxation", "stress")
KEYS = ["pm", "subject_id", "outer_fold"]
METRICS = ("balanced_accuracy", "macro_f1", "roc_auc", "pr_auc")
REFERENCE = "reports/diagnostics/pm_temporal_predictability_v1/full_matched"
OUTPUT = "reports/diagnostics/pm_temporal_predictability_v1/order_control_h120"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_artifacts(path: Path) -> tuple[dict, pd.DataFrame, dict]:
    """Verify completion, selection and selected runtime artifacts without retraining."""
    protocol = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
    status = json.loads((path / "execution_status.json").read_text(encoding="utf-8"))
    unsigned = {k: v for k, v in protocol.items() if k not in ("protocol_hash", "git_commit")}
    if low.stable_hash(unsigned) != protocol["protocol_hash"]:
        raise ValueError("Ordered reference protocol hash mismatch")
    if (status.get("state") != "complete" or status.get("complete") != 175 or status.get("failed") != 0
        or status["protocol_hash"] != protocol["protocol_hash"] or protocol["params"] != history.PARAMS
        or protocol["mode"] != "full_matched" or protocol["config"]["history_order_control"] != "ordered"):
        raise ValueError("Ordered reference is not the completed frozen matched experiment")
    all_runs = pd.read_csv(path / "run_matrix.csv")
    if len(all_runs) != 175 or all_runs.run_id.duplicated().any() or not all_runs.support_mode.eq(history.PRIMARY).all():
        raise ValueError("Invalid ordered run matrix")
    selection = pd.read_csv(path / "paired_ba_vs_h10.csv")
    selected = selection.loc[selection.horizon_s.eq(120) & selection.bh_q_28.lt(.05), "pm"]
    if sorted(selected.tolist()) != list(PMS):
        raise ValueError("Conditional five-PM selection differs from frozen request")
    runs = all_runs.loc[all_runs.pm.isin(PMS) & all_runs.horizon_s.eq(120)].copy()
    if set(runs[["pm", "outer_fold"]].itertuples(index=False, name=None)) != {(p,f) for p in PMS for f in range(1,6)} or len(runs) != 25:
        raise ValueError("Ordered reference lacks exactly 25 H120 cells")
    files = ["protocol.json", "execution_status.json", "run_matrix.csv", "participant_metrics.csv", "fold_metrics.csv", "paired_ba_vs_h10.csv"]
    hashes = {name: file_hash(path / name) for name in files}
    combined = pd.read_csv(path / "participant_metrics.csv", dtype={"subject_id": str})
    for spec in runs.to_dict("records"):
        directory = path / "runs" / spec["run_id"]
        summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
        if summary["status"] != "complete" or summary["protocol_hash"] != protocol["protocol_hash"] or summary["specification_hash"] != spec["specification_hash"]:
            raise ValueError("Ordered cell integrity mismatch")
        artifacts = summary.get("artifact_sha256", {})
        if set(artifacts) != {"participant_metrics.csv", "predictions.parquet", "history_provenance.parquet"}:
            raise ValueError("Ordered cell lacks artifact hashes")
        for name, digest in artifacts.items():
            if file_hash(directory / name) != digest:
                raise ValueError("Corrupt ordered reference artifact")
        hashes[str((directory / "run_summary.json").relative_to(path))] = file_hash(directory / "run_summary.json")
        cell = pd.read_csv(directory / "participant_metrics.csv", dtype={"subject_id": str})
        aggregate = combined.loc[combined.pm.eq(spec["pm"]) & combined.outer_fold.eq(spec["outer_fold"]) & combined.horizon_s.eq(120)]
        pd.testing.assert_frame_equal(cell.sort_values("subject_id").reset_index(drop=True),
            aggregate[cell.columns].sort_values("subject_id").reset_index(drop=True), check_exact=True)
    return protocol, runs, hashes


def prepare_protocol(config: dict, *, root: Path, feature_cache_dir: Path, output_dir=None) -> history.HistoryContext:
    root = Path(root).resolve()
    output = root / (output_dir or OUTPUT)
    reference = root / REFERENCE
    if output.resolve() == reference.resolve() or reference.resolve() in output.resolve().parents:
        raise ValueError("Order control cannot write inside ordered reference")
    ordered_protocol, ordered_runs, reference_hashes = reference_artifacts(reference)
    # Reuse the established loader/threshold fitting and manifest builder; no training or writes.
    context = history.prepare_protocol(config, root=root, feature_cache_dir=feature_cache_dir,
                                       output_dir=output, full=True)
    for key in ("reference_protocol_hash", "cache_identity", "input_table_sha256", "feature_values_sha256", "feature_names_hash", "package_versions", "params"):
        if context.protocol[key] != ordered_protocol[key]:
            raise ValueError(f"Order-control input/model identity differs: {key}")
    runs = context.run_matrix.loc[context.run_matrix.pm.isin(PMS) & context.run_matrix.horizon_s.eq(120)].copy()
    compare_reference_identity(runs, ordered_runs)
    protocol = {**context.protocol, "mode": "order_control_h120", "experiment_type": "confirmatory_conditional",
        "primary_planned_fits": 25, "secondary_planned_fits": 0,
        "history_order_control": "shuffled_older_blocks_canonical_fixed", "target_subset": list(PMS), "horizons_s": [120],
        "ordered_reference": REFERENCE, "ordered_protocol_hash": ordered_protocol["protocol_hash"],
        "reference_file_hashes": reference_hashes,
        "selection_rule": "H120 vs H10 BH q<0.05 in completed matched experiment; fixed five PM",
        "contrast": "ordered_H120_minus_shuffled_H120", "bootstrap_replicates": 10000, "bootstrap_seed": 42,
        "multiplicity": "BH across exactly five primary BA contrasts",
        "shuffle_rule": "existing history_matrix; SHA256(42|sample_id) seed; per-sample permutation of 11 older blocks",
        "feature_name_semantics": "lag labels identify ordered slots; older slots receive shuffled blocks; final slot remains -10",
        "source_hashes": {**context.protocol["source_hashes"], Path(__file__).name: file_hash(Path(__file__))}}
    protocol.pop("protocol_hash")
    protocol["ordered_reference_config"] = protocol.pop("config")
    protocol["protocol_hash"] = low.stable_hash({k:v for k,v in protocol.items() if k != "git_commit"})
    ordered_map = ordered_runs.set_index(["pm", "outer_fold"])
    specifications = []
    for row in runs.to_dict("records"):
        row.pop("specification_hash"); row.pop("run_id")
        row.update(condition="H120_shuffled_older", history_order="shuffled",
                   ordered_run_id=ordered_map.loc[(row["pm"], row["outer_fold"]), "run_id"])
        row["specification_hash"] = low.stable_hash({"protocol_hash": protocol["protocol_hash"], **row})
        row["run_id"] = f"order_control__{row['pm']}__fold{row['outer_fold']}__{row['specification_hash'][:12]}"
        specifications.append(row)
    support = context.support_audit.loc[context.support_audit.pm.isin(PMS) & context.support_audit.horizon_s.eq(120)].copy()
    context = replace(context, protocol=protocol, run_matrix=pd.DataFrame(specifications), support_audit=support)
    validate(context)
    return context


def compare_reference_identity(runs: pd.DataFrame, ordered: pd.DataFrame) -> None:
    keys = ["pm", "outer_fold"]
    columns = [c for c in ordered if c.endswith("_hash") and c != "specification_hash"]
    columns += ["n_train", "n_test", "horizon_s", "support_mode", "feature_count"]
    left = runs.set_index(keys).sort_index()
    right = ordered.set_index(keys).sort_index()
    pd.testing.assert_frame_equal(left[columns], right[columns], check_exact=True)
    for key in ("q_low", "q_high"):
        if not np.allclose(left[key], right[key], atol=1e-15, rtol=0):
            raise ValueError("Ordered/shuffled thresholds differ")


def validate(context: history.HistoryContext) -> None:
    runs, protocol = context.run_matrix, context.protocol
    expected = {(p,f,120) for p in PMS for f in range(1,6)}
    actual = set(runs[["pm","outer_fold","horizon_s"]].itertuples(index=False, name=None))
    if (len(runs) != 25 or actual != expected or not runs.support_mode.eq(history.PRIMARY).all()
        or not runs.history_order.eq("shuffled").all() or protocol["params"] != history.PARAMS
        or protocol["mode"] != "order_control_h120" or protocol["primary_planned_fits"] != 25
        or protocol["history_order_control"] != "shuffled_older_blocks_canonical_fixed"):
        raise ValueError("Order control requires exactly 25 frozen shuffled H120 cells")
    columns = ["sample_id", "subject_id", "record_id", "outer_fold", "t_start", "t_end", *context.config["target_ids"]]
    digest = hashlib.sha256(pd.util.hash_pandas_object(context.base.full[columns], index=False).to_numpy().tobytes()).hexdigest()
    if digest != protocol["input_table_sha256"] or file_hash(Path(history.__file__)) != protocol["source_hashes"][Path(history.__file__).name]:
        raise ValueError("Inputs or history implementation changed after preflight")
    positions, masks, _ = history.history_positions(context.base.full)
    if any(not np.array_equal(positions[k], context.positions[k]) for k in positions) or not np.array_equal(masks[120], context.masks[120]):
        raise ValueError("History lookup changed after preflight")
    for spec in runs.to_dict("records"):
        unsigned = {k:v for k,v in spec.items() if k not in ("run_id", "specification_hash")}
        if low.stable_hash({"protocol_hash": protocol["protocol_hash"], **unsigned}) != spec["specification_hash"]:
            raise ValueError("Order-control specification changed")
    reference = context.base.root / protocol["ordered_reference"]
    for name, digest in protocol["reference_file_hashes"].items():
        if file_hash(reference / name) != digest:
            raise ValueError("Ordered reference changed after preflight")
    ordered = pd.read_csv(reference / "run_matrix.csv")
    ordered = ordered.loc[ordered.pm.isin(PMS) & ordered.horizon_s.eq(120)]
    compare_reference_identity(runs, ordered)
    for spec in runs.to_dict("records"):
        cohort = history.cohort_for(context.base.full, context.masks[120], spec["pm"])
        labels = context.base.transforms[(spec["outer_fold"], spec["pm"])].transform(cohort.continuous_target.to_numpy())
        validate_cell(context, spec, cohort, labels)


def validate_cell(context: history.HistoryContext, spec: dict, cohort: pd.DataFrame, labels: np.ndarray) -> None:
    for partition in ("train", "test"):
        inside = cohort.outer_fold.ne(spec["outer_fold"]) if partition == "train" else cohort.outer_fold.eq(spec["outer_fold"])
        keep = inside.to_numpy() & np.isfinite(labels)
        if int(keep.sum()) != spec[f"n_{partition}"]:
            raise ValueError("Ordered/shuffled retained counts differ")
        for name, digest in history.support_identity(cohort, labels, keep).items():
            if spec[f"{partition}_{name}"] != digest:
                raise ValueError("Ordered/shuffled sample/label/subject/fold hash mismatch")


def write_dry_run(context: history.HistoryContext) -> dict:
    validate(context)
    path = context.output / "protocol.json"
    if path.exists() and json.loads(path.read_text())["protocol_hash"] != context.protocol["protocol_hash"]:
        if (context.output / "runs").exists() or (context.output / "execution_status.json").exists():
            raise ValueError("Existing order-control results have another protocol")
    low._atomic_json(path, context.protocol)
    low._write_csv(context.output / "run_matrix.csv", context.run_matrix)
    low._write_csv(context.output / "support_audit.csv", context.support_audit)
    summary = {"planned_fits": 25, "training_executed": False, "ordered_reference_retrained": False,
        "protocol_hash": context.protocol["protocol_hash"], "target_subset": list(PMS), "horizons_s": [120],
        "support_mode": history.PRIMARY, "ordered_identity_hashes_equal": True, "canonical_block_fixed": True}
    low._atomic_json(context.output / "dry_run_summary.json", summary)
    return summary


def run(context: history.HistoryContext) -> dict:
    validate(context)
    dry = json.loads((context.output / "dry_run_summary.json").read_text())
    if dry["planned_fits"] != 25 or dry["protocol_hash"] != context.protocol["protocol_hash"]:
        raise ValueError("Matching 25-fit dry-run required")
    try:
        return history._execute_runs(context, full=False, order_control=True)
    except Exception as exc:
        low._atomic_json(context.output / "failure.json", {"exception": type(exc).__name__, "message": str(exc), "unix_time": time.time()})
        raise


def aggregate_order_control(context: history.HistoryContext, folds: pd.DataFrame, participants: pd.DataFrame) -> None:
    reference = context.base.root / context.protocol["ordered_reference"]
    ordered = pd.read_csv(reference / "participant_metrics.csv", dtype={"subject_id": str})
    ordered = ordered.loc[ordered.pm.isin(PMS) & ordered.horizon_s.eq(120)]
    shuffled = participants.copy()
    shuffled.subject_id = shuffled.subject_id.astype(str)
    for frame in (ordered, shuffled):
        if frame.duplicated(KEYS).any():
            raise ValueError("Duplicate participant identity in order comparison")
    paired = ordered.merge(shuffled, on=KEYS, how="outer", suffixes=("_ordered", "_shuffled"), indicator=True, validate="one_to_one")
    if not paired._merge.eq("both").all():
        raise ValueError("Ordered/shuffled participant support differs")
    for column in ("n_samples", "n_low", "n_high"):
        if not paired[f"{column}_ordered"].eq(paired[f"{column}_shuffled"]).all():
            raise ValueError("Ordered/shuffled participant label counts differ")
    paired = paired.drop(columns="_merge")
    for metric in METRICS:
        paired[f"delta_{metric}"] = paired[f"{metric}_ordered"] - paired[f"{metric}_shuffled"]
    ordered_folds = pd.read_csv(reference / "fold_metrics.csv")
    ordered_folds = ordered_folds.loc[ordered_folds.pm.isin(PMS) & ordered_folds.horizon_s.eq(120)]
    fold_pairs = ordered_folds.merge(folds, on=["pm", "outer_fold"], suffixes=("_ordered", "_shuffled"), validate="one_to_one")
    if len(folds) != 25 or len(fold_pairs) != 25:
        raise ValueError("Incomplete order-control folds")
    fold_pairs["delta_ba"] = fold_pairs.participant_macro_balanced_accuracy_ordered - fold_pairs.participant_macro_balanced_accuracy_shuffled
    rows, secondary = [], []
    for pm, group in paired.groupby("pm", sort=True):
        delta = group.delta_balanced_accuracy.to_numpy()
        ci = history.subject_bootstrap_interval(delta, n_resamples=10000, random_state=42)
        f = fold_pairs.loc[fold_pairs.pm.eq(pm), "delta_ba"]
        rows.append({"pm": pm, "ordered_ba": group.balanced_accuracy_ordered.mean(),
            "shuffled_ba": group.balanced_accuracy_shuffled.mean(), "delta_ba": ci["estimate"],
            "ci95_low": ci["ci_low"], "ci95_high": ci["ci_high"], "n_subjects": ci["n_subjects"],
            "bootstrap_two_sided_p": history.centered_bootstrap_p(delta),
            "participants_ordered_better": int((delta>0).sum()), "participants_shuffled_better": int((delta<0).sum()),
            "participants_tied": int((delta==0).sum()), "folds_ordered_better": int((f>0).sum()),
            "folds_shuffled_better": int((f<0).sum()), "fold_delta_mean": f.mean(), "fold_delta_sd": f.std(ddof=1),
            "fold_delta_min": f.min(), "fold_delta_max": f.max()})
        for metric in METRICS[1:]:
            valid = group[[f"{metric}_ordered", f"{metric}_shuffled"]].notna().all(axis=1)
            sub = group.loc[valid]
            ci_secondary = history.subject_bootstrap_interval(sub[f"delta_{metric}"], n_resamples=10000, random_state=42)
            secondary.append({"pm": pm, "metric": metric, "ordered": sub[f"{metric}_ordered"].mean(),
                "shuffled": sub[f"{metric}_shuffled"].mean(), **ci_secondary})
    results = pd.DataFrame(rows)
    if sorted(results.pm) != list(PMS):
        raise ValueError("BH must cover exactly the five prespecified BA contrasts")
    results["bh_q_5"] = history.benjamini_hochberg(results.bootstrap_two_sided_p.to_numpy())
    for name, frame in [("ordered_vs_shuffled_participant_deltas", paired), ("ordered_vs_shuffled_ba", results),
                        ("ordered_vs_shuffled_fold_deltas", fold_pairs), ("secondary_metrics", pd.DataFrame(secondary))]:
        low._write_csv(context.output / f"{name}.csv", frame)
    low._atomic_json(context.output / "inference_protocol.json", {"contrast": "ordered_H120 - shuffled_H120",
        "primary_metric": "participant_macro_balanced_accuracy", "unit": "participant", "replicates": 10000,
        "seed": 42, "ci": "percentile 95% paired differences", "p": "two-sided centered-null bootstrap; plus-one correction",
        "multiplicity": "BH across exactly 5 primary BA contrasts", "secondary": "descriptive only",
        "limitations": "conditional target selection on same cohort; overlapping training folds; one fixed shuffle realization; no physiological causality"})
