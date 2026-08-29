"""Model robustness for the frozen seven-PM LOW/HIGH task."""
from __future__ import annotations
from dataclasses import dataclass, replace
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from bench.experiments.pm_low_high_q3_extremes_confirmatory import (
    FIXED_LAG_SECONDS, PM_NAMES, ProtocolContext,
    execute_run as execute_reference_run,
    load_config as load_reference_config,
    load_resumable_summary as load_reference_resumable_summary,
    prepare_protocol as prepare_reference_protocol,
    stable_hash,
)
from cogstate.model_zoo import build_model

SCHEMA_VERSION = "pm-low-high-model-robustness-v1"
CANDIDATE_MODEL_ORDER = ("random_forest", "lightgbm")
ALL_MODEL_ORDER = ("xgboost", "lightgbm", "random_forest")
METRICS = ("balanced_accuracy", "f1", "roc_auc", "pr_auc", "low_recall", "high_recall", "precision", "accuracy")
BOOTSTRAP_METRICS = ("balanced_accuracy", "f1", "roc_auc")
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_MODELS = {
    "random_forest": {"task_type": "classification", "estimator": "RandomForestClassifier", "params": {"n_estimators": 200, "n_jobs": -1, "random_state": 42}},
    "lightgbm": {"task_type": "classification", "estimator": "LGBMClassifier", "params": {"n_estimators": 200, "n_jobs": 4, "random_state": 42}},
}


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    if config.get("planned_new_fits") != 70:
        raise ValueError("Exactly 70 new fits are required")
    ref = config.get("reference", {})
    expected_ref_hash = "ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431"
    if ref.get("model") != "xgboost" or ref.get("protocol_hash") != expected_ref_hash:
        raise ValueError("Unexpected XGBoost reference contract")
    contract = config.get("scientific_contract", {})
    expected_contract = {
        "pm_names": list(PM_NAMES), "alignment": "EEG(t-10s) -> PM(t)", "lag_seconds": -10,
        "feature_count": 371, "target_transform": "outer_train_q33_q67_extremes",
        "middle_policy": "exclude", "outer_group": "subject_id", "folds": [1, 2, 3, 4, 5], "seed": 42,
    }
    if contract != expected_contract:
        raise ValueError("Scientific LOW/HIGH contract changed")
    if config.get("candidate_models") != EXPECTED_MODELS:
        raise ValueError("Candidate models/hyperparameters changed")
    evaluation = config.get("evaluation", {})
    if evaluation.get("primary_metric") != "participant_macro_balanced_accuracy":
        raise ValueError("Primary metric changed")
    if evaluation.get("probability_source") != "predict_proba_high":
        raise ValueError("Probability source changed")
    if evaluation.get("single_class_auc_policy") != "undefined_exclude_metric_only":
        raise ValueError("AUC undefined policy changed")
    selection = config.get("model_selection_for_personalization", {})
    if float(selection.get("practical_equivalence_margin", -1)) != 0.01:
        raise ValueError("Practical equivalence margin must be 0.01")
    if int(selection.get("maximum_models_advanced", -1)) != 2:
        raise ValueError("At most two models may advance")
    if tuple(selection.get("fixed_model_order", ())) != ALL_MODEL_ORDER:
        raise ValueError("Fixed model order changed")
    forbidden = config.get("forbidden", {})
    if not forbidden or not all(v is True for v in forbidden.values()):
        raise ValueError("All forbidden switches must remain true")
    return config


@dataclass
class RobustnessContext:
    root: Path
    output_dir: Path
    config: dict[str, Any]
    reference_context: ProtocolContext
    execution_context: ProtocolContext
    reference_protocol: dict[str, Any]
    reference_run_matrix: pd.DataFrame
    reference_results: pd.DataFrame
    cohort_equivalence: pd.DataFrame
    protocol: dict[str, Any]
    run_matrix: pd.DataFrame


def _build_equivalence_audit(reference_context: ProtocolContext, completed_protocol: Mapping[str, Any], completed_runs: pd.DataFrame) -> pd.DataFrame:
    if completed_protocol.get("protocol_hash") != reference_context.protocol["protocol_hash"]:
        raise RuntimeError("Stored and recomputed reference protocol hashes differ")
    for field in ("fixed_fold_hash", "temporal_pairing_hash"):
        if completed_protocol.get(field) != reference_context.protocol.get(field):
            raise RuntimeError(f"Reference {field} changed")
    if completed_protocol.get("result_status") != "confirmatory_complete" or completed_protocol.get("training_executed") is not True:
        raise RuntimeError("Reference LOW/HIGH experiment is not complete")
    merged = reference_context.run_matrix.merge(completed_runs, on=["outer_fold", "pm"], suffixes=("_current", "_reference"), validate="one_to_one")
    if len(merged) != 35:
        raise RuntimeError("Expected 35 reference PM-fold rows")
    exact = ("target_id", "condition", "lag_seconds", "threshold_hash", "n_train", "n_test", "n_test_participants", "train_sample_hash", "test_sample_hash", "matched_target_sample_hash")
    rows = []
    for row in merged.to_dict("records"):
        for field in exact:
            if str(row[f"{field}_current"]) != str(row[f"{field}_reference"]):
                raise RuntimeError(f"Cohort mismatch fold={row['outer_fold']} pm={row['pm']} field={field}")
        for field in ("q_low", "q_high"):
            if not np.isclose(float(row[f"{field}_current"]), float(row[f"{field}_reference"]), rtol=0.0, atol=1e-12):
                raise RuntimeError(f"Threshold mismatch fold={row['outer_fold']} pm={row['pm']} field={field}")
        rows.append({
            "outer_fold": int(row["outer_fold"]), "pm": str(row["pm"]),
            "threshold_hash": str(row["threshold_hash_current"]), "n_train": int(row["n_train_current"]), "n_test": int(row["n_test_current"]),
            "train_sample_hash": str(row["train_sample_hash_current"]), "test_sample_hash": str(row["test_sample_hash_current"]),
            "matched_target_sample_hash": str(row["matched_target_sample_hash_current"]), "equivalent": True,
        })
    return pd.DataFrame(rows)


def _candidate_run_matrix(reference_context: ProtocolContext, protocol_hash: str, config: Mapping[str, Any]) -> pd.DataFrame:
    specs = []
    for model_name in CANDIDATE_MODEL_ORDER:
        model_cfg = config["candidate_models"][model_name]
        for ref in reference_context.run_matrix.to_dict("records"):
            spec = {
                "outer_fold": int(ref["outer_fold"]), "pm": str(ref["pm"]), "target_id": str(ref["target_id"]),
                "task": "binary_classification", "condition": "lag_minus_10s", "lag_seconds": FIXED_LAG_SECONDS,
                "model": model_name, "estimator": model_cfg["estimator"], "seed": 42,
                "model_params_hash": stable_hash(model_cfg["params"]), "q_low": float(ref["q_low"]), "q_high": float(ref["q_high"]),
                "threshold_hash": str(ref["threshold_hash"]), "n_train": int(ref["n_train"]), "n_test": int(ref["n_test"]),
                "n_test_participants": int(ref["n_test_participants"]), "train_sample_hash": str(ref["train_sample_hash"]),
                "test_sample_hash": str(ref["test_sample_hash"]), "matched_target_sample_hash": str(ref["matched_target_sample_hash"]),
            }
            sh = stable_hash({"protocol_hash": protocol_hash, "run_spec": spec})
            spec["specification_hash"] = sh
            spec["run_id"] = f"{model_name}__fold_{int(ref['outer_fold']):02d}__{ref['pm']}__low_high__{sh[:12]}"
            specs.append(spec)
    frame = pd.DataFrame(specs)
    if len(frame) != 70 or frame["run_id"].duplicated().any():
        raise RuntimeError("Exactly 70 unique candidate runs are required")
    return frame


def prepare_protocol(config: Mapping[str, Any], *, root: str | Path, feature_cache_dir: str | Path, output_dir: str | Path | None = None) -> RobustnessContext:
    root = Path(root).resolve()
    output = Path(output_dir or config["output_dir"])
    if not output.is_absolute():
        output = root / output
    ref_cfg = load_reference_config(root / config["reference"]["config"])
    ref_output = root / config["reference"]["output_dir"]
    ref_context = prepare_reference_protocol(ref_cfg, root=root, feature_cache_dir=feature_cache_dir, output_dir=ref_output)
    protocol_path, run_matrix_path, results_path = ref_output / "protocol.json", ref_output / "run_matrix.csv", ref_output / "results_by_fold.csv"
    for path in (protocol_path, run_matrix_path, results_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    completed_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    completed_runs = pd.read_csv(run_matrix_path)
    completed_results = pd.read_csv(results_path)
    expected_hash = config["reference"]["protocol_hash"]
    if ref_context.protocol["protocol_hash"] != expected_hash or completed_protocol.get("protocol_hash") != expected_hash:
        raise RuntimeError("Reference protocol hash changed")
    if len(completed_results) != 35:
        raise RuntimeError("Reference results must contain 35 rows")
    equivalence = _build_equivalence_audit(ref_context, completed_protocol, completed_runs)
    robustness_hash = stable_hash({
        "schema_version": SCHEMA_VERSION, "reference_protocol_hash": expected_hash,
        "scientific_contract": config["scientific_contract"], "candidate_models": config["candidate_models"],
        "evaluation": config["evaluation"], "selection": config["model_selection_for_personalization"], "forbidden": config["forbidden"],
        "feature_cache_identity": ref_context.cache_identity, "fixed_fold_hash": ref_context.protocol["fixed_fold_hash"],
        "temporal_pairing_hash": ref_context.protocol["temporal_pairing_hash"], "threshold_hashes": ref_context.protocol["threshold_hashes"],
    })
    protocol = {
        "schema_version": SCHEMA_VERSION, "experiment_id": config["experiment_id"], "result_status": "preregistered_candidate",
        "training_executed": False, "git_commit": _git_head(root), "reference_experiment_id": config["reference"]["experiment_id"],
        "reference_protocol_hash": expected_hash, "reference_model": "xgboost", "scientific_contract": config["scientific_contract"],
        "candidate_models": config["candidate_models"], "evaluation": config["evaluation"],
        "model_selection_for_personalization": config["model_selection_for_personalization"], "feature_cache_identity": ref_context.cache_identity,
        "fixed_fold_hash": ref_context.protocol["fixed_fold_hash"], "temporal_pairing_hash": ref_context.protocol["temporal_pairing_hash"],
        "threshold_hashes": ref_context.protocol["threshold_hashes"], "cohort_equivalence_rows": 35,
        "cohort_equivalence_mismatches": 0, "planned_new_fits": 70, "protocol_hash": robustness_hash,
    }
    run_matrix = _candidate_run_matrix(ref_context, robustness_hash, config)
    execution_context = replace(ref_context, output_dir=output, protocol=protocol, run_matrix=run_matrix)
    return RobustnessContext(root, output, dict(config), ref_context, execution_context, completed_protocol, completed_runs, completed_results, equivalence, protocol, run_matrix)


def _factory_audit(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name in CANDIDATE_MODEL_ORDER:
        cfg = config["candidate_models"][name]
        model = build_model(name, "classification", (371,), 2, cfg["params"])
        rows.append({"model": name, "expected_estimator": cfg["estimator"], "actual_estimator": type(model).__name__, "matches": type(model).__name__ == cfg["estimator"]})
    if not all(r["matches"] for r in rows):
        raise RuntimeError(f"Factory audit failed: {rows}")
    return rows


def write_dry_run(context: RobustnessContext) -> dict[str, Any]:
    context.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(context.output_dir / "protocol.json", context.protocol)
    _write_csv(context.output_dir / "cohort_equivalence_audit.csv", context.cohort_equivalence)
    _write_csv(context.output_dir / "run_matrix.csv", context.run_matrix)
    summary = {
        "experiment_id": context.config["experiment_id"], "protocol_hash": context.protocol["protocol_hash"],
        "reference_protocol_hash": context.config["reference"]["protocol_hash"], "reference_model": "xgboost",
        "candidate_models": list(CANDIDATE_MODEL_ORDER), "factory_audit": _factory_audit(context.config),
        "cohort_equivalence_rows": 35, "cohort_equivalence_mismatches": 0, "feature_count": 371,
        "fixed_lag_seconds": FIXED_LAG_SECONDS, "planned_new_fits": 70, "training_executed": False,
    }
    _atomic_json(context.output_dir / "dry_run_summary.json", summary)
    (context.output_dir / "README.md").write_text(
        f"# PM LOW/HIGH model robustness v1\n\nReference: completed XGBoost LOW/HIGH confirmatory run.\n\n"
        f"- candidates: Random Forest, LightGBM\n- new fits: 70\n- PM/folds: 7 / 5\n- features: 371\n"
        f"- alignment: EEG(t-10s) -> PM(t)\n- thresholds/cohorts: identical to reference\n"
        f"- primary metric: participant-macro balanced accuracy\n- protocol hash: `{context.protocol['protocol_hash']}`\n"
        f"- cohort-equivalence mismatches: 0\n- dry-run training executed: false\n",
        encoding="utf-8",
    )
    return summary


def _model_builder(context: RobustnessContext, model_name: str):
    cfg = context.config["candidate_models"][model_name]
    def builder(_name: str, task_type: str, input_shape: Sequence[int] | None, num_outputs: int | None, _params: Mapping[str, Any] | None = None):
        return build_model(model_name, task_type, input_shape, num_outputs, cfg["params"])
    return builder


def _patch_summary(context: RobustnessContext, spec: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(summary)
    payload.update({"model": spec["model"], "estimator": spec["estimator"], "model_params_hash": spec["model_params_hash"]})
    _atomic_json(context.output_dir / "runs" / str(spec["run_id"]) / "run_summary.json", payload)
    return payload


def execute_run(context: RobustnessContext, spec: Mapping[str, Any]) -> dict[str, Any]:
    name = str(spec["model"])
    if name not in CANDIDATE_MODEL_ORDER:
        raise ValueError(name)
    summary = execute_reference_run(context.execution_context, spec, model_builder=_model_builder(context, name))
    return _patch_summary(context, spec, summary)


def load_resumable_summary(context: RobustnessContext, spec: Mapping[str, Any]) -> dict[str, Any] | None:
    payload = load_reference_resumable_summary(context.execution_context, spec)
    if payload is None:
        return None
    if payload.get("model") not in (None, spec["model"]):
        return None
    return _patch_summary(context, spec, payload) if payload.get("model") is None else payload


def _summary_by_model(combined: pd.DataFrame, by_pm: bool) -> pd.DataFrame:
    rows = []
    groups = [(m, pm) for m in ALL_MODEL_ORDER for pm in PM_NAMES] if by_pm else [(m, None) for m in ALL_MODEL_ORDER]
    for model, pm in groups:
        group = combined[combined["model"].eq(model)]
        if pm is not None:
            group = group[group["pm"].eq(pm)]
        expected = 5 if pm is not None else 35
        if len(group) != expected:
            raise RuntimeError(f"{model}/{pm}: expected {expected}, got {len(group)}")
        row = {"model": model, "n_fold_pm_rows": len(group)}
        if pm is not None:
            row["pm"] = pm
        for metric in METRICS:
            col = f"participant_macro_{metric}"
            values = group[col].to_numpy(dtype=float); finite = values[np.isfinite(values)]
            row[f"{col}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
            row[f"{col}_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
            row[f"{col}_median"] = float(np.median(finite)) if len(finite) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def paired_delta_vs_xgboost(combined: pd.DataFrame) -> pd.DataFrame:
    ref = combined[combined["model"].eq("xgboost")].set_index(["outer_fold", "pm"])
    rows = []
    for model in CANDIDATE_MODEL_ORDER:
        cand = combined[combined["model"].eq(model)].set_index(["outer_fold", "pm"])
        if set(cand.index) != set(ref.index):
            raise RuntimeError(f"{model}: PM-fold keys differ from reference")
        for key in sorted(ref.index):
            row = {"model": model, "outer_fold": int(key[0]), "pm": str(key[1])}
            for metric in METRICS:
                col = f"participant_macro_{metric}"
                row[f"delta_{metric}_vs_xgboost"] = float(cand.loc[key, col]) - float(ref.loc[key, col])
            rows.append(row)
    return pd.DataFrame(rows)


def _participant_paired_delta(context: RobustnessContext) -> pd.DataFrame:
    ref_lookup = context.reference_run_matrix.set_index(["outer_fold", "pm"])
    rows = []
    for spec in context.run_matrix.to_dict("records"):
        key = (int(spec["outer_fold"]), str(spec["pm"])); ref_run_id = str(ref_lookup.loc[key, "run_id"])
        cand = pd.read_csv(context.output_dir / "runs" / str(spec["run_id"]) / "participant_metrics.csv")
        ref = pd.read_csv(context.root / context.config["reference"]["output_dir"] / "runs" / ref_run_id / "participant_metrics.csv")
        merged = cand.merge(ref, on="subject_id", suffixes=("_candidate", "_xgboost"), validate="one_to_one")
        for item in merged.to_dict("records"):
            row = {"model": spec["model"], "outer_fold": key[0], "pm": key[1], "subject_id": str(item["subject_id"])}
            for metric in METRICS:
                c = "macro_f1" if metric == "f1" else metric
                a, b = float(item[f"{c}_candidate"]), float(item[f"{c}_xgboost"])
                row[f"delta_{metric}_vs_xgboost"] = a - b if np.isfinite(a) and np.isfinite(b) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def _cluster_bootstrap(deltas: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    subjects = np.asarray(sorted(deltas["subject_id"].astype(str).unique()))
    rows = []
    for model in CANDIDATE_MODEL_ORDER:
        frame = deltas[deltas["model"].eq(model)]
        for metric in BOOTSTRAP_METRICS:
            column = f"delta_{metric}_vs_xgboost"; observed = float(np.nanmean(frame[column].to_numpy(dtype=float)))
            samples = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
            for i in range(BOOTSTRAP_REPLICATES):
                drawn = rng.choice(subjects, size=len(subjects), replace=True); parts = []
                for subject in drawn:
                    values = frame.loc[frame["subject_id"].astype(str).eq(str(subject)), column].to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                    if len(values): parts.append(values)
                samples[i] = float(np.mean(np.concatenate(parts))) if parts else np.nan
            samples = samples[np.isfinite(samples)]
            rows.append({"model": model, "metric": metric, "observed_mean_delta": observed,
                         "bootstrap_ci_low": float(np.quantile(samples, 0.025)), "bootstrap_ci_high": float(np.quantile(samples, 0.975)),
                         "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
                         "resampling_unit": "subject_id_cluster", "n_unique_subjects": len(subjects)})
    return pd.DataFrame(rows)


def select_models_for_personalization(summary: pd.DataFrame, *, margin: float = 0.01) -> dict[str, Any]:
    frame = summary.copy(); rank = {m: i for i, m in enumerate(ALL_MODEL_ORDER)}; frame["_order"] = frame["model"].map(rank)
    frame = frame.sort_values(["participant_macro_balanced_accuracy_mean", "participant_macro_roc_auc_mean", "participant_macro_balanced_accuracy_std", "_order"], ascending=[False, False, True, True], kind="stable").reset_index(drop=True)
    best, second = str(frame.loc[0, "model"]), str(frame.loc[1, "model"])
    best_ba, second_ba = float(frame.loc[0, "participant_macro_balanced_accuracy_mean"]), float(frame.loc[1, "participant_macro_balanced_accuracy_mean"])
    gap = best_ba - second_ba; advanced = [best] + ([second] if gap <= margin else [])
    return {"best_model": best, "second_model": second, "best_balanced_accuracy": best_ba, "second_balanced_accuracy": second_ba,
            "best_minus_second_balanced_accuracy": gap, "practical_equivalence_margin": margin, "advanced_models": advanced}


def aggregate_results(context: RobustnessContext, summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate = pd.DataFrame(summaries)
    if len(candidate) != 70 or candidate["run_id"].duplicated().any() or set(candidate["model"]) != set(CANDIDATE_MODEL_ORDER):
        raise RuntimeError("Aggregation requires 70 valid candidate runs")
    _write_csv(context.output_dir / "results_by_fold.csv", candidate)
    reference = context.reference_results.copy(); reference["model"] = "xgboost"
    combined = pd.concat([reference, candidate], ignore_index=True, sort=False)
    if len(combined) != 105: raise RuntimeError("Combined comparison must contain 105 rows")
    _write_csv(context.output_dir / "combined_results_by_fold.csv", combined)
    by_pm = _summary_by_model(combined, True); by_model = _summary_by_model(combined, False)
    deltas = paired_delta_vs_xgboost(combined); participant = _participant_paired_delta(context); bootstrap = _cluster_bootstrap(participant)
    _write_csv(context.output_dir / "summary_by_model_pm.csv", by_pm); _write_csv(context.output_dir / "summary_by_model.csv", by_model)
    _write_csv(context.output_dir / "paired_delta_vs_xgboost.csv", deltas); _write_csv(context.output_dir / "participant_paired_delta.csv", participant)
    _write_csv(context.output_dir / "paired_delta_bootstrap_ci.csv", bootstrap)
    selection = select_models_for_personalization(by_model, margin=float(context.config["model_selection_for_personalization"]["practical_equivalence_margin"]))
    _write_csv(context.output_dir / "pooled_summary.csv", pd.DataFrame([{ "n_models": 3, "n_pm": 7, "n_folds": 5, "n_combined_fold_pm_rows": 105, "reference_model": "xgboost", **selection }]))
    protocol = dict(context.protocol); protocol.update({"training_executed": True, "result_status": "confirmatory_complete", "completed_new_fits": 70, "model_selection_for_personalization_result": selection})
    _atomic_json(context.output_dir / "protocol.json", protocol)
    return selection


def run_experiment(context: RobustnessContext, *, resume: bool) -> dict[str, Any]:
    summaries = []; reused = trained = 0
    for spec in context.run_matrix.to_dict("records"):
        existing = load_resumable_summary(context, spec) if resume else None
        if existing is not None:
            summaries.append(existing); reused += 1; continue
        run_dir = context.output_dir / "runs" / str(spec["run_id"])
        if run_dir.exists() and not resume: raise FileExistsError(run_dir)
        summaries.append(execute_run(context, spec)); trained += 1
    if len(summaries) != 70: raise RuntimeError("Exactly 70 candidate runs must complete")
    selection = aggregate_results(context, summaries)
    return {"complete": 70, "trained": trained, "reused": reused, "advanced_models": selection["advanced_models"]}

__all__ = ["ALL_MODEL_ORDER", "CANDIDATE_MODEL_ORDER", "EXPECTED_MODELS", "RobustnessContext", "load_config", "prepare_protocol", "write_dry_run", "run_experiment", "select_models_for_personalization"]
