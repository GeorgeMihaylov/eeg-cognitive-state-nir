#!/usr/bin/env python3
"""Build or verify record-safe COG-BCI raw-window cache shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.datasets.cog_bci_dataset import COGBCIDataset  # noqa: E402
from bench.datasets.cog_bci_window_cache import (  # noqa: E402
    COGBCIWindowBuilder,
    RawWindowSpec,
    audit_window_index,
)


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8-sig")
    document = (
        json.loads(text)
        if config_path.suffix.casefold() == ".json"
        else yaml.safe_load(text)
    )
    if not isinstance(document, dict):
        raise ValueError("Window-cache config must contain a mapping")
    unknown = sorted(
        set(document)
        - {
            "dataset_root",
            "index_cache",
            "output_dir",
            "channel_policy",
            "window",
            "selection",
            "resume",
            "overwrite",
            "verify_only",
        }
    )
    if unknown:
        raise ValueError(f"Unknown window-cache config keys: {unknown}")
    return document


def _value(
    args: argparse.Namespace,
    config: dict[str, Any],
    name: str,
    *,
    section: str | None = None,
    default: Any = None,
) -> Any:
    argument = getattr(args, name)
    if argument is not None:
        return argument
    source = config if section is None else config.get(section, {})
    return source.get(name, default)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    dataset_root = _value(
        args,
        config,
        "dataset_root",
        default="data/raw/cog_bci/extracted",
    )
    index_cache = _value(
        args,
        config,
        "index_cache",
        default="benchmark_results/cog_bci_loader/record_index.json",
    )
    output_dir = Path(
        _value(
            args,
            config,
            "output_dir",
            default="benchmark_results/cog_bci_windows/emotiv_common",
        )
    )
    channel_policy = _value(
        args, config, "channel_policy", default="emotiv_common"
    )
    window_values = dict(config.get("window", {}))
    allowed_window = set(RawWindowSpec.__dataclass_fields__)
    unknown_window = sorted(set(window_values) - allowed_window)
    if unknown_window:
        raise ValueError(f"Unknown window config keys: {unknown_window}")
    for name in allowed_window:
        argument = getattr(args, name, None)
        if argument is not None:
            window_values[name] = argument
    spec = RawWindowSpec(**window_values)

    dataset = COGBCIDataset(
        {
            "data_path": dataset_root,
            "index_cache_path": index_cache,
            "use_index_cache": True,
            "require_canonical_complete": True,
        }
    )
    builder = COGBCIWindowBuilder(
        dataset,
        output_dir=output_dir,
        channel_policy_name=channel_policy,
        spec=spec,
    )
    selection = dict(config.get("selection", {}))
    records = builder.select_records(
        subjects=(
            args.subjects
            if args.subjects is not None
            else selection.get("subjects")
        ),
        sessions=(
            args.sessions
            if args.sessions is not None
            else selection.get("sessions")
        ),
        task_families=(
            args.task_families
            if args.task_families is not None
            else selection.get("task_families")
        ),
        task_variants=(
            args.task_variants
            if args.task_variants is not None
            else selection.get("task_variants")
        ),
        max_records=(
            args.max_records
            if args.max_records is not None
            else selection.get("max_records")
        ),
        one_per_subject_family=(
            args.one_per_subject_family
            if args.one_per_subject_family is not None
            else bool(selection.get("one_per_subject_family", False))
        ),
    )
    summary = builder.run(
        records,
        resume=(
            args.resume
            if args.resume is not None
            else bool(config.get("resume", False))
        ),
        overwrite=(
            args.overwrite
            if args.overwrite is not None
            else bool(config.get("overwrite", False))
        ),
        verify_only=(
            args.verify_only
            if args.verify_only is not None
            else bool(config.get("verify_only", False))
        ),
    )
    window_index_path = output_dir / "window_index.parquet"
    if window_index_path.is_file():
        audit = audit_window_index(pd.read_parquet(window_index_path))
        if not bool(audit["leakage_safe"]):
            raise RuntimeError(f"Window identity audit failed: {audit}")
        if not bool(
            args.verify_only
            if args.verify_only is not None
            else config.get("verify_only", False)
        ):
            _atomic_json(output_dir / "leakage_audit.json", audit)
        summary["leakage_audit"] = audit
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--dataset-root")
    parser.add_argument("--index-cache")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--channel-policy",
        choices=["cog_bci_common", "emotiv_common"],
    )
    parser.add_argument(
        "--window-duration-seconds", type=float, dest="window_duration_seconds"
    )
    parser.add_argument(
        "--window-stride-seconds", type=float, dest="window_stride_seconds"
    )
    parser.add_argument(
        "--drop-incomplete-window",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="drop_incomplete_window",
    )
    parser.add_argument(
        "--minimum-valid-fraction", type=float, dest="minimum_valid_fraction"
    )
    parser.add_argument(
        "--segmentation-mode",
        choices=["record_full", "task_interval", "event_interval"],
        dest="segmentation_mode",
    )
    parser.add_argument(
        "--preprocessing",
        choices=["none", "bandpass", "notch", "bandpass_notch"],
    )
    parser.add_argument("--bandpass-low-hz", type=float, dest="bandpass_low_hz")
    parser.add_argument("--bandpass-high-hz", type=float, dest="bandpass_high_hz")
    parser.add_argument("--bandpass-order", type=int, dest="bandpass_order")
    parser.add_argument(
        "--notch-frequency-hz", type=float, dest="notch_frequency_hz"
    )
    parser.add_argument("--notch-q", type=float, dest="notch_q")
    parser.add_argument(
        "--target-sampling-rate-hz", type=float, dest="target_sampling_rate_hz"
    )
    parser.add_argument(
        "--constant-variance-threshold",
        type=float,
        dest="constant_variance_threshold",
    )
    parser.add_argument(
        "--near-zero-variance-threshold",
        type=float,
        dest="near_zero_variance_threshold",
    )
    parser.add_argument(
        "--reject-nonfinite",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="reject_nonfinite",
    )
    parser.add_argument(
        "--reject-constant-channels",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="reject_constant_channels",
    )
    parser.add_argument(
        "--allow-filtering-when-source-status-unknown",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="allow_filtering_when_source_status_unknown",
    )
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--sessions", nargs="+")
    parser.add_argument("--task-families", nargs="+")
    parser.add_argument("--task-variants", nargs="+")
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--one-per-subject-family",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--verify-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
