"""Build deterministic, analysis-only artifacts for the canonical target contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.datasets.base_eeg_data_loader import feature_list_sha256
from bench.datasets.raw_eeg_window_dataset import RawEEGWindowDataset
from bench.datasets.target_view import (
    build_feature_target_view,
    build_target_view,
    target_cohort_manifest,
)
from bench.tasks.target_registry import (
    LEGACY_TARGET_ALIASES,
    PM_METRICS,
    get_target_spec,
    list_target_specs,
)
from bench.tasks.target_transforms import FoldLocalQuantileTargetTransform


def build_contract(config_path: Path, *, plan_only: bool = False) -> dict[str, Any]:
    config = _read_yaml(config_path)
    root = REPO_ROOT
    paths = {
        key: _resolve(root, config[key])
        for key in (
            "registry_path",
            "feature_dataset_path",
            "raw_manifest_path",
            "logical_recording_map_path",
            "output_dir",
            "report_path",
        )
    }
    for key in (
        "registry_path",
        "feature_dataset_path",
        "raw_manifest_path",
        "logical_recording_map_path",
    ):
        if not paths[key].is_file():
            raise FileNotFoundError(f"{key} not found: {paths[key]}")

    registry_source = _read_yaml(paths["registry_path"])
    canonical_source_ids = {
        str(item["target_id"]) for item in registry_source.get("targets", [])
    }
    required_source_ids = {
        *(f"target_{metric}" for metric in PM_METRICS),
        "label_q5",
    }
    missing_registry_sources = sorted(required_source_ids - canonical_source_ids)
    if missing_registry_sources:
        raise ValueError(
            f"Audit registry lacks canonical source targets: {missing_registry_sources}"
        )

    feature_frame = pd.read_parquet(paths["feature_dataset_path"])
    if "sample_id" not in feature_frame.columns:
        feature_frame = feature_frame.copy()
        feature_frame.insert(0, "sample_id", feature_frame.index.to_numpy())
    raw_manifest = pd.read_parquet(paths["raw_manifest_path"])
    outer_fold_by_subject = _fixed_outer_folds(
        raw_manifest,
        fold_column=str(config["fixed_outer_fold_column"]),
        group_column=str(config["fixed_outer_group_column"]),
        expected_folds=int(config["expected_outer_folds"]),
    )

    executable_specs = list_target_specs(executable_only=True)
    candidate_specs = tuple(
        spec for spec in list_target_specs() if not spec.is_executable
    )
    executable_rows = [_spec_row(spec) for spec in executable_specs]
    candidate_rows = [_spec_row(spec) for spec in candidate_specs]
    feature_rows: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []

    for spec in executable_specs:
        for feature_input in spec.allowed_feature_inputs:
            view = build_feature_target_view(feature_frame, spec, feature_input)
            feature_rows.append(
                {
                    "target_id": spec.target_id,
                    "feature_input": feature_input,
                    "compatible": True,
                    "n_samples": view.features.shape[0],
                    "n_features": view.features.shape[1],
                    "feature_list_sha256": feature_list_sha256(
                        list(view.feature_names)
                    ),
                    "target_metadata_in_features": False,
                }
            )
            shape_rows.append(
                {
                    "target_id": spec.target_id,
                    "input_view": f"feature_{feature_input}",
                    "input_shape": f"[{view.features.shape[1]}]",
                    "target_shape": _target_shape(spec, len(view.features)),
                    "target_dtype": str(view.target_view.targets.dtype),
                    "n_samples": len(view.features),
                }
            )
        target_manifest = target_cohort_manifest(
            feature_frame, spec, outer_fold_by_subject
        )
        target_manifest.insert(1, "input_view", "feature")
        cohort_rows.extend(target_manifest.to_dict("records"))

    raw_rows: list[dict[str, Any]] = []
    raw_ready = True
    for spec in list_target_specs():
        row: dict[str, Any] = {
            "target_id": spec.target_id,
            "declared_raw_input_supported": spec.raw_input_supported,
            "execution_status": spec.execution_status,
            "compatible": False,
            "reason": "registered_but_disabled",
            "n_samples": None,
            "n_subjects": None,
            "input_shape": None,
            "target_shape": None,
            "target_dtype": None,
        }
        if spec.is_executable and spec.raw_input_supported:
            dataset_config = {
                "data_path": str(paths["raw_manifest_path"]),
                "target_id": spec.target_id,
                "target_data_path": str(paths["feature_dataset_path"]),
                "dataset_mode": config["raw_dataset_mode"],
                "logical_recording_map_path": str(paths["logical_recording_map_path"]),
                "raw_preprocessing": config["raw_preprocessing"],
            }
            data = RawEEGWindowDataset(dataset_config).load()
            row.update(
                {
                    "compatible": True,
                    "reason": "validated_without_reading_window_tensors",
                    "n_samples": data.n_samples,
                    "n_subjects": data.n_subjects,
                    "input_shape": json.dumps(list(data.data.shape[1:])),
                    "target_shape": json.dumps(list(data.labels.shape)),
                    "target_dtype": str(data.labels.dtype),
                }
            )
            shape_rows.append(
                {
                    "target_id": spec.target_id,
                    "input_view": "raw_eeg",
                    "input_shape": json.dumps(list(data.data.shape[1:])),
                    "target_shape": json.dumps(list(data.labels.shape)),
                    "target_dtype": str(data.labels.dtype),
                    "n_samples": data.n_samples,
                }
            )
            for fold in sorted(set(data.row_metadata["outer_fold"])):
                mask = data.row_metadata["outer_fold"] == fold
                cohort_rows.append(
                    {
                        "target_id": spec.target_id,
                        "input_view": "raw_eeg_deduplicated",
                        "outer_fold": int(fold),
                        "n_samples": int(mask.sum()),
                        "n_subjects": int(pd.Series(data.subject_ids[mask]).nunique()),
                        "n_records": int(pd.Series(data.record_ids[mask]).nunique()),
                    }
                )
        elif spec.is_executable:
            raw_ready = False
            row["reason"] = "executable_target_not_approved_for_raw_input"
        raw_rows.append(row)

    derived_contracts = _build_fold_local_contracts(
        feature_frame, outer_fold_by_subject
    )
    legacy_rows = [
        {
            "legacy_name": alias,
            "canonical_target_id": target_id,
            "warning_required": True,
            "implicit_fallback_allowed": False,
        }
        for alias, target_id in sorted(LEGACY_TARGET_ALIASES.items())
    ]

    contract_status = (
        "canonical_target_contract_ready" if raw_ready else "partially_ready"
    )
    artifacts: dict[str, Any] = {
        "executable_targets.csv": executable_rows,
        "candidate_targets.csv": candidate_rows,
        "target_output_shapes.csv": shape_rows,
        "target_feature_compatibility.csv": feature_rows,
        "target_raw_compatibility.csv": raw_rows,
        "target_cohort_manifest.csv": cohort_rows,
        "legacy_aliases.csv": legacy_rows,
        "derived_transform_contracts.json": derived_contracts,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "execution_mode": "analysis_only_no_model_training",
        "decision": contract_status,
        "registry_audit_status": registry_source.get("audit_status"),
        "feature_dataset_sha256": _sha256(paths["feature_dataset_path"]),
        "raw_manifest_sha256": _sha256(paths["raw_manifest_path"]),
        "fixed_outer_fold_count": len(set(outer_fold_by_subject.values())),
        "fixed_outer_subject_count": len(outer_fold_by_subject),
        "executable_target_count": len(executable_specs),
        "candidate_target_count": len(candidate_specs),
        "all_executable_pm_raw_views_validated": raw_ready,
        "fold_local_boundaries_fit_scope": "outer_train_only",
        "legacy_implicit_target_main_fallback": False,
        "git_commit": _git_commit(root),
        "artifacts": {},
    }

    if plan_only:
        manifest["planned_artifacts"] = sorted(
            [*artifacts, "target_contract_manifest.json"]
        )
        return manifest

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in artifacts.items():
        destination = output_dir / filename
        if filename.endswith(".csv"):
            pd.DataFrame(payload).to_csv(destination, index=False, lineterminator="\n")
        else:
            destination.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        manifest["artifacts"][filename] = {
            "sha256": _sha256(destination),
            "rows": len(payload),
        }
    manifest_path = output_dir / "target_contract_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(paths["report_path"], manifest, executable_rows, raw_rows)
    return manifest


def _build_fold_local_contracts(
    frame: pd.DataFrame, outer_fold_by_subject: dict[str, int]
) -> dict[str, Any]:
    subjects = frame["subject_id"].astype(str)
    result: dict[str, Any] = {
        "fit_scope": "outer_train_only",
        "materializes_dataset_columns": False,
        "transforms": {},
    }
    for metric in PM_METRICS:
        values = pd.to_numeric(frame[f"target_{metric}"], errors="coerce").to_numpy()
        for q in (3, 5):
            target_id = f"pm_{metric}_q{q}_fold_local"
            folds = []
            for fold in sorted(set(outer_fold_by_subject.values())):
                is_test = subjects.map(outer_fold_by_subject).to_numpy() == fold
                transform = FoldLocalQuantileTargetTransform(q=q, duplicates="drop")
                transform.fit(values[~is_test])
                transformed = transform.transform(values)
                fold_manifest = transform.manifest()
                fold_manifest.update(
                    {
                        "outer_fold": int(fold),
                        "finite_transformed_count": int(pd.notna(transformed).sum()),
                    }
                )
                folds.append(fold_manifest)
            result["transforms"][target_id] = folds
    return result


def _fixed_outer_folds(
    manifest: pd.DataFrame, *, fold_column: str, group_column: str, expected_folds: int
) -> dict[str, int]:
    accepted = manifest.loc[manifest["status"] == "ok", [group_column, fold_column]]
    fold_counts = accepted.groupby(group_column)[fold_column].nunique()
    if not (fold_counts == 1).all():
        raise ValueError("One subject is assigned to multiple fixed outer folds")
    mapping = (
        accepted.drop_duplicates(group_column)
        .set_index(group_column)[fold_column]
        .astype(int)
        .to_dict()
    )
    if len(set(mapping.values())) != expected_folds:
        raise ValueError("Fixed outer fold count does not match configuration")
    return {str(key): int(value) for key, value in mapping.items()}


def _spec_row(spec: Any) -> dict[str, Any]:
    row = spec.to_dict()
    for name, value in list(row.items()):
        if isinstance(value, list):
            row[name] = json.dumps(value, ensure_ascii=False)
    return row


def _target_shape(spec: Any, n_samples: int) -> str:
    shape = [n_samples] if spec.output_dim == 1 else [n_samples, spec.output_dim]
    return json.dumps(shape)


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    executable_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_validated = sum(bool(row["compatible"]) for row in raw_rows)
    content = f"""# Canonical target contract

## Decision

`{manifest['decision']}`. This is an analysis-only integration result; no model was trained and no benchmark metric was produced.

## Registry authority

The executable contract is derived from `reports/summary/target_registry.yaml` and its provenance audit. It defines {len(executable_rows)} executable targets and {manifest['candidate_target_count']} registered-but-disabled candidates.

## Executable targets

Seven scalar PM regressions, the fixed-order seven-output PM regression, and the legacy global `label_q5` benchmark label are executable. The physical `label_q5` column is exposed only as `label_focus_q5_legacy` with registry status `legacy_global_benchmark_label`.

## Disabled candidates

Activity proxies, the seven-output activity multilabel target, fold-local Q3/Q5 ordinal candidates, and long-term excitement remain registered but disabled until their scientific or materialization prerequisites are approved.

## Feature target view

The shared feature view accepts `eeg`, `pow`, and `eeg_pow`, preserves source row/sample order, returns a target-specific availability mask, and excludes all `PM.*`, `target_*`, `label_*`, and identifier columns from model inputs.

## Raw target view

{raw_validated} executable raw target views were validated against the existing manifest and deduplicated logical-record selection without reading or rebuilding raw window tensors. Inputs remain `[1, 14, 2560]`; scalar regression labels use `float32`, classification labels use integer dtype, and multi-output regression follows the canonical seven-target order.

## Cohort policy

Outer subject-to-fold assignments are immutable. Missing targets create target-specific complete-case cohorts inside those fixed folds; folds are never rebuilt after target filtering.

## Fold-local ordinal transforms

Q3/Q5 boundaries are fit only on finite outer-train values, then applied unchanged to train, validation, and outer-test partitions. Duplicate boundaries are reported through the actual class count; there is no global fallback and no derived column is materialized.

## Legacy aliases

`label_q5` maps to `label_focus_q5_legacy`; `target_focus` maps to `pm_focus_regression`; `target_main` is a warned legacy alias for focus and is never an implicit fallback.

## Task integration

The existing task registry now exposes explicit PM scalar, PM multi-output, and legacy Q5 task IDs while preserving `focus_regression`, `performance_metrics_regression`, and `cognitive_load_5class` compatibility.

## Metrics contract

Regression targets recommend MAE, RMSE, R², and Spearman correlation. Classification targets recommend accuracy, balanced accuracy, macro/weighted F1, kappa, and applicable ordinal/AUC metrics. These are contracts only, not new results.

## Leakage controls

Target values and metadata never enter features. Target transforms fit on outer-train only. Raw attachment validates sample, subject, and record identifiers. Target missingness is filtered rather than zero-filled.

## Artifacts

Deterministic machine-readable artifacts are under `reports/summary/target_contract/`. CSV files are ignored by the repository-wide `*.csv` rule and require explicit force-add only if a future commit is requested.

## Compatibility

Legacy feature/raw label-Q5 configurations, scalar focus regression, seven-output PM regression, and config-only FOMAML/DANN paths remain loadable. Compatibility aliases emit explicit warnings.

## Limitations

Fold-local ordinal candidates remain disabled as benchmark tasks, activity semantics remain unapproved, and long-term excitement is not materialized in the processed table. Full test-suite status is recorded separately after generation.
"""
    path.write_text(content, encoding="utf-8", newline="\n")


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def _resolve(root: Path, value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/targets/canonical_target_contract.yaml",
        type=Path,
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    result = build_contract(args.config, plan_only=args.plan_only)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
