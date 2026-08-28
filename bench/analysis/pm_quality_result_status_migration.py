"""Safely migrate PM temporal-quality runtime manifest result status.

The migration is deliberately metadata-only.  It validates every configured
fold against the final experiment config before writing anything, preserves the
historical protocol hash, and records original manifest SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.analysis.pm_quality_downstream import build_downstream_plan
from bench.analysis.pm_temporal_quality import load_config, stable_hash


MIGRATION_SCHEMA_VERSION = "pm-quality-result-status-migration-v1"
EXPECTED_RUN_COUNT_PER_FOLD = 56
REQUIRED_RUN_ARTIFACTS = (
    "metrics.json",
    "normalization_stats.json",
    "predictions.parquet",
    "split.json",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _serialized_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, value: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _validate_protocol_hash(manifest: Mapping[str, Any], path: Path) -> str:
    stored = manifest.get("protocol_hash")
    payload = dict(manifest)
    payload.pop("protocol_hash", None)
    if isinstance(stored, str) and stable_hash(payload) == stored:
        return "current_manifest"
    legacy_payload = dict(payload)
    legacy_payload["result_status"] = "diagnostic"
    if (
        manifest.get("result_status") != "diagnostic"
        and isinstance(stored, str)
        and stable_hash(legacy_payload) == stored
    ):
        return "pre_migration_manifest"
    raise ValueError(f"Legacy protocol_hash is invalid: {path}")


def _validate_no_failed_runs(summary: pd.DataFrame, fold_dir: Path) -> None:
    for column in ("status", "run_status"):
        if column in summary:
            failed = summary[column].astype(str).str.lower().isin(
                {"failed", "error", "invalid"}
            )
            if failed.any():
                raise ValueError(f"Failed runs are present in {fold_dir / 'summary.csv'}")
    for column in ("error", "error_type", "failure_reason"):
        if column in summary and summary[column].fillna("").astype(str).str.strip().ne("").any():
            raise ValueError(f"Run errors are present in {fold_dir / 'summary.csv'}")


def _validate_fold(
    *,
    config: Mapping[str, Any],
    fold: int,
    fold_dir: Path,
) -> dict[str, Any]:
    required = ("manifest.json", "run_matrix.csv", "summary.csv")
    missing = [name for name in required if not (fold_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete fold {fold}: missing {missing}")

    manifest_path = fold_dir / "manifest.json"
    original_bytes = manifest_path.read_bytes()
    manifest = _read_json(manifest_path)
    expected_experiment_id = str(config["downstream"]["experiment_id"])
    if manifest.get("experiment_id") != expected_experiment_id:
        raise ValueError(
            f"Fold {fold} experiment_id mismatch: "
            f"{manifest.get('experiment_id')!r} != {expected_experiment_id!r}"
        )
    if manifest.get("run_count") != EXPECTED_RUN_COUNT_PER_FOLD:
        raise ValueError(f"Fold {fold} run_count must be {EXPECTED_RUN_COUNT_PER_FOLD}")
    if manifest.get("fixed_outer_folds") != [fold]:
        raise ValueError(f"Fold {fold} fixed_outer_folds must be [{fold}]")
    if manifest.get("result_status") not in {"diagnostic", config["result_status"]}:
        raise ValueError(f"Fold {fold} has an unsupported source result_status")
    protocol_hash_validation = _validate_protocol_hash(manifest, manifest_path)

    expected_matrix = build_downstream_plan(config, outer_folds=[fold])
    if len(expected_matrix) != EXPECTED_RUN_COUNT_PER_FOLD:
        raise ValueError(
            "Final config does not define exactly "
            f"{EXPECTED_RUN_COUNT_PER_FOLD} runs for fold {fold}"
        )
    expected_hash = stable_hash(expected_matrix.to_dict(orient="records"))
    if manifest.get("run_matrix_hash") != expected_hash:
        raise ValueError(f"Fold {fold} does not match the final config run matrix")

    actual_matrix = pd.read_csv(fold_dir / "run_matrix.csv")
    summary = pd.read_csv(fold_dir / "summary.csv")
    if len(actual_matrix) != EXPECTED_RUN_COUNT_PER_FOLD:
        raise ValueError(f"Fold {fold} run_matrix.csv must contain 56 rows")
    if len(summary) != EXPECTED_RUN_COUNT_PER_FOLD:
        raise ValueError(f"Fold {fold} summary.csv must contain 56 completed rows")
    _validate_no_failed_runs(summary, fold_dir)

    identity_columns = ("run_id", "specification_hash")
    expected_identities = expected_matrix.loc[:, identity_columns].astype(str)
    for frame_name, frame in (("run_matrix.csv", actual_matrix), ("summary.csv", summary)):
        missing_columns = sorted(set(identity_columns) - set(frame.columns))
        if missing_columns:
            raise ValueError(f"Fold {fold} {frame_name} is missing {missing_columns}")
        identities = frame.loc[:, identity_columns].astype(str)
        if identities.duplicated().any():
            raise ValueError(f"Fold {fold} {frame_name} contains duplicate runs")
        if set(map(tuple, identities.to_numpy())) != set(
            map(tuple, expected_identities.to_numpy())
        ):
            raise ValueError(f"Fold {fold} {frame_name} does not match the final config")

    for row in expected_matrix.to_dict(orient="records"):
        run_dir = fold_dir / str(row["run_id"])
        missing_artifacts = [
            name for name in REQUIRED_RUN_ARTIFACTS if not (run_dir / name).is_file()
        ]
        if row["task_type"] == "classification" and not (
            run_dir / "target_transform.json"
        ).is_file():
            missing_artifacts.append("target_transform.json")
        if missing_artifacts:
            raise FileNotFoundError(
                f"Fold {fold} run {row['run_id']} is incomplete: {missing_artifacts}"
            )

    updated = dict(manifest)
    updated["result_status"] = config["result_status"]
    changed_keys = {
        key for key in set(manifest) | set(updated) if manifest.get(key) != updated.get(key)
    }
    if changed_keys not in (set(), {"result_status"}):
        raise AssertionError(f"Unsafe manifest mutation requested: {sorted(changed_keys)}")
    changed = manifest["result_status"] != config["result_status"]
    updated_bytes = _serialized_json(updated) if changed else original_bytes
    return {
        "fold": fold,
        "manifest_path": manifest_path,
        "original_bytes": original_bytes,
        "updated_bytes": updated_bytes,
        "original_sha256": _sha256_bytes(original_bytes),
        "updated_sha256": _sha256_bytes(updated_bytes),
        "result_status_before": manifest["result_status"],
        "result_status_after": updated["result_status"],
        "changed": changed,
        "protocol_hash": manifest["protocol_hash"],
        "protocol_hash_validation": protocol_hash_validation,
        "run_count": int(manifest["run_count"]),
        "run_matrix_hash": str(manifest["run_matrix_hash"]),
    }


def migrate_result_status(
    *,
    config_path: str | Path,
    results_root: str | Path,
    audit_path: str | Path | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Validate all folds and optionally migrate only ``result_status``.

    Validation is completed for every fold before the first write.  If a write
    fails, manifests already written in this invocation are restored byte for
    byte.  ``protocol_hash`` is intentionally retained as historical execution
    provenance; recomputing it would violate the metadata-only field contract.
    """
    config_file = Path(config_path)
    root = Path(results_root)
    config = load_config(config_file)
    desired_status = str(config["result_status"])
    if desired_status != "confirmatory":
        raise ValueError("Migration config result_status must be 'confirmatory'")
    folds = [int(value) for value in config["folds"]["fold_ids"]]
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError("Migration requires the fixed final five-fold config")

    expected_fold_names = {f"fold{fold:02d}" for fold in folds}
    actual_fold_names = {
        path.name for path in root.glob("fold[0-9][0-9]") if path.is_dir()
    }
    if actual_fold_names != expected_fold_names:
        raise ValueError(
            "Results root fold directories do not match final config: "
            f"{sorted(actual_fold_names)} != {sorted(expected_fold_names)}"
        )

    plans = [
        _validate_fold(config=config, fold=fold, fold_dir=root / f"fold{fold:02d}")
        for fold in folds
    ]
    audit = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "experiment_id": config["downstream"]["experiment_id"],
        "config_sha256": _sha256_bytes(config_file.read_bytes()),
        "desired_result_status": desired_status,
        "apply_requested": bool(apply),
        "all_invariants_valid": True,
        "expected_run_count_per_fold": EXPECTED_RUN_COUNT_PER_FOLD,
        "total_validated_runs": EXPECTED_RUN_COUNT_PER_FOLD * len(folds),
        "changed_field": "result_status",
        "protocol_hash_policy": "preserve_historical_execution_hash",
        "files": [
            {
                key: value
                for key, value in plan.items()
                if key not in {"manifest_path", "original_bytes", "updated_bytes"}
            }
            | {"manifest_path": plan["manifest_path"].relative_to(root).as_posix()}
            for plan in plans
        ],
    }
    if not apply:
        audit["writes_performed"] = False
        return audit
    if audit_path is None:
        raise ValueError("audit_path is required with apply=True")
    effective_audit_path = Path(audit_path)
    if effective_audit_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing migration audit: {effective_audit_path}"
        )

    written: list[dict[str, Any]] = []
    try:
        for plan in plans:
            if plan["changed"]:
                _atomic_write(plan["manifest_path"], plan["updated_bytes"])
                written.append(plan)
        audit["writes_performed"] = True
        audit["changed_manifest_count"] = len(written)
        _atomic_write(effective_audit_path, _serialized_json(audit))
    except Exception:
        for plan in reversed(written):
            _atomic_write(plan["manifest_path"], plan["original_bytes"])
        raise
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--audit-path")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write manifests and the audit JSON; without this flag, validate only.",
    )
    args = parser.parse_args(argv)
    if args.apply and not args.audit_path:
        parser.error("--audit-path is required with --apply")
    result = migrate_result_status(
        config_path=args.config,
        results_root=args.results_root,
        audit_path=args.audit_path,
        apply=args.apply,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
