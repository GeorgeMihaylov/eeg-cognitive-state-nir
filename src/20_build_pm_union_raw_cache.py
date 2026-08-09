"""Plan or materialize the canonical PM-union composite raw EEG manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.datasets.pm_union_raw_contract import (  # noqa: E402
    finalize_pm_union_composite,
    plan_pm_union_composite,
)
from bench.datasets.raw_eeg_window_dataset import (  # noqa: E402
    CANONICAL_EEG_CHANNELS,
    build_raw_eeg_cache,
)
from bench.datasets.raw_preprocessing import (  # noqa: E402
    normalize_raw_preprocessing,
    raw_preprocessing_hash,
)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _relative_cache_paths(frame: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    result = frame.copy()
    ok = result["status"].astype(str).eq("ok")
    for index in result.index[ok]:
        path = Path(str(result.at[index, "cache_file"]))
        if path.is_absolute():
            try:
                path = path.relative_to(data_root)
            except ValueError as exc:
                raise ValueError(
                    f"Delta cache shard is outside data root: {path}"
                ) from exc
        result.at[index, "cache_file"] = str(path)
    return result


def _assert_safe_outputs(
    *,
    historical_manifest: Path,
    historical_cache_root: Path,
    output_manifest: Path,
    delta_cache_root: Path,
) -> None:
    paths = {
        "historical_manifest": historical_manifest.resolve(),
        "historical_cache_root": historical_cache_root.resolve(),
        "output_manifest": output_manifest.resolve(),
        "delta_cache_root": delta_cache_root.resolve(),
    }
    if paths["output_manifest"] == paths["historical_manifest"]:
        raise ValueError("PM-union output must not overwrite historical raw-v3")
    if paths["delta_cache_root"] == paths["historical_cache_root"]:
        raise ValueError("Delta cache root must differ from historical raw-v3 root")
    if paths["historical_cache_root"] in paths["delta_cache_root"].parents:
        raise ValueError("Delta cache root must be outside historical raw-v3 root")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        document = yaml.safe_load(input_file) or {}
    if document.get("mode") != "pm_union_composite":
        raise ValueError("Config mode must be 'pm_union_composite'")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/targets/pm_union_raw_composite.yaml",
    )
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT),
        help="Runtime root used to resolve data paths; never stored as identity",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--materialize-delta", action="store_true")
    parser.add_argument("--record-limit", type=int)
    args = parser.parse_args()

    config = _load_config(_resolve(REPO_ROOT, args.config))
    data_root = Path(args.data_root).resolve()
    processed = _resolve(data_root, config["processed_path"])
    catalog = _resolve(data_root, config["catalog_path"])
    audit_schema = _resolve(data_root, config["audit_schema_path"])
    historical_manifest = _resolve(
        data_root, config["historical_manifest_path"]
    )
    logical_map = _resolve(data_root, config["logical_recording_map_path"])
    historical_cache_root = _resolve(
        data_root, config["historical_cache_root"]
    )
    delta_cache_root = _resolve(data_root, config["delta_cache_root"])
    output_manifest = _resolve(data_root, config["output_manifest_path"])
    output_summary = _resolve(data_root, config["output_summary_path"])
    _assert_safe_outputs(
        historical_manifest=historical_manifest,
        historical_cache_root=historical_cache_root,
        output_manifest=output_manifest,
        delta_cache_root=delta_cache_root,
    )

    preprocessing = normalize_raw_preprocessing(
        config.get("raw_preprocessing"),
        default_resample_hz=float(config.get("target_sfreq", 256.0)),
    )
    target_sfreq = float(preprocessing["resample_hz"])
    semantic_hash = raw_preprocessing_hash(
        preprocessing,
        channels=CANONICAL_EEG_CHANNELS,
        default_resample_hz=target_sfreq,
    )
    plan, delta, summary = plan_pm_union_composite(
        processed,
        catalog,
        historical_manifest,
        logical_map,
        audit_schema_path=audit_schema,
        target_sfreq=target_sfreq,
    )
    if summary["preprocessing_hash"] != semantic_hash:
        raise ValueError(
            "Configured preprocessing differs from historical raw-v3: "
            f"configured={semantic_hash}, historical={summary['preprocessing_hash']}"
        )
    summary.update(
        {
            "mode": "plan_only" if args.plan_only else "materialized",
            "output_manifest_path": config["output_manifest_path"],
            "delta_cache_root": config["delta_cache_root"],
            "historical_cache_reused": True,
            "historical_cache_rebuilt": False,
        }
    )
    if args.plan_only:
        print(json.dumps(summary, indent=2))
        return

    built_delta, cache_stats = build_raw_eeg_cache(
        delta,
        delta_cache_root,
        target_sfreq=target_sfreq,
        max_missing_fraction=float(config.get("max_missing_fraction", 0.02)),
        repo_root=data_root,
        record_limit=args.record_limit,
        raw_preprocessing=preprocessing,
    )
    built_delta = _relative_cache_paths(built_delta, data_root)
    historical = pd.read_parquet(historical_manifest)
    composite = finalize_pm_union_composite(
        historical,
        built_delta,
        expected_preprocessing_hash=semantic_hash,
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    composite.to_parquet(output_manifest, index=False)
    summary.update(
        {
            "cache": cache_stats,
            "final_status_counts": {
                str(key): int(value)
                for key, value in composite["status"].value_counts().items()
            },
            "delta_status_counts": {
                str(key): int(value)
                for key, value in built_delta["status"].value_counts().items()
            },
            "delta_rejection_reasons": {
                str(key): int(value)
                for key, value in built_delta.loc[
                    built_delta["status"] != "ok", "rejection_reason"
                ].value_counts().items()
            },
        }
    )
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with output_summary.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
