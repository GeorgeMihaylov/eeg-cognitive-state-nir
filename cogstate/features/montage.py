"""Deterministic mapping from EEG channel names to canonical scalp regions."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


MONTAGE_SCHEMA_VERSION = "cogstate-regional-montage-v1"

# Midline regions are explicit: assigning Fz/Cz/Pz/Oz to either hemisphere
# would create an artificial lateralization. Central regions are retained so
# standard 10-20/10-10 C/FC/CP electrodes are not folded into frontal/parietal.
CANONICAL_REGIONS = (
    "frontal_left",
    "frontal_midline",
    "frontal_right",
    "central_left",
    "central_midline",
    "central_right",
    "temporal_left",
    "temporal_right",
    "parietal_left",
    "parietal_midline",
    "parietal_right",
    "occipital_left",
    "occipital_midline",
    "occipital_right",
)

EMOTIV_14_CHANNELS = (
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
)

_CHANNEL_PATTERN = re.compile(
    r"^(FP|AF|FT|FC|TP|CP|PO|F|C|T|P|O|I)(Z|[0-9]{1,2})$"
)
_PREFIX_POSITION = {
    "FP": "frontal",
    "AF": "frontal",
    "F": "frontal",
    "FT": "temporal",
    "T": "temporal",
    "TP": "temporal",
    "FC": "central",
    "C": "central",
    "CP": "central",
    "P": "parietal",
    "PO": "parietal",
    "O": "occipital",
    "I": "occipital",
}
_LEGACY_10_20_ALIASES = {
    "T3": "temporal_left",
    "T4": "temporal_right",
    "T5": "parietal_left",
    "T6": "parietal_right",
}


def normalize_channel_name(channel_name: str) -> str:
    """Normalize a channel label for matching, without guessing references."""
    name = str(channel_name).strip().upper()
    if name.startswith("EEG."):
        name = name[4:]
    if not name:
        raise ValueError("channel names must be non-empty")
    return name


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _standard_region(normalized_name: str) -> str | None:
    if normalized_name in _LEGACY_10_20_ALIASES:
        return _LEGACY_10_20_ALIASES[normalized_name]
    match = _CHANNEL_PATTERN.fullmatch(normalized_name)
    if match is None:
        return None
    prefix, suffix = match.groups()
    position = _PREFIX_POSITION[prefix]
    if suffix == "Z":
        if position == "temporal":
            return None
        return f"{position}_midline"
    side = "left" if int(suffix) % 2 else "right"
    return f"{position}_{side}"


def normalize_custom_mapping(
    custom_mapping: Mapping[str, str] | Sequence[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    """Validate and freeze a device-specific channel mapping."""
    if custom_mapping is None:
        return ()
    items = custom_mapping.items() if isinstance(custom_mapping, Mapping) else custom_mapping
    normalized: dict[str, str] = {}
    for raw_channel, raw_region in items:
        channel = normalize_channel_name(raw_channel)
        region = str(raw_region).strip()
        if region not in CANONICAL_REGIONS:
            raise ValueError(
                f"custom mapping for {raw_channel!r} uses unknown region {region!r}"
            )
        if channel in normalized:
            raise ValueError(f"duplicate custom mapping for channel {channel!r}")
        normalized[channel] = region
    return tuple(sorted(normalized.items()))


def build_montage_manifest(
    channel_names: Sequence[str],
    *,
    custom_mapping: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Resolve an ordered montage and return a human-readable manifest."""
    if isinstance(channel_names, (str, bytes)):
        raise TypeError("channel_names must be a sequence of channel labels")
    names = tuple(str(name).strip() for name in channel_names)
    if not names or any(not name for name in names):
        raise ValueError("channel_names must be non-empty")
    normalized_names = tuple(normalize_channel_name(name) for name in names)
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("channel_names must be unique after normalization")

    custom = dict(normalize_custom_mapping(custom_mapping))
    rows: list[dict[str, Any]] = []
    unknown: list[str] = []
    for index, (original, normalized) in enumerate(zip(names, normalized_names)):
        if normalized in custom:
            region = custom[normalized]
            source = "custom"
        else:
            region = _standard_region(normalized)
            source = "standard_10_20_10_10"
        if region is None:
            unknown.append(original)
            continue
        rows.append(
            {
                "input_index": index,
                "channel_name": original,
                "normalized_name": normalized,
                "region": region,
                "mapping_source": source,
            }
        )
    if unknown:
        raise ValueError(
            "unknown EEG channel names require an explicit custom mapping: "
            + ", ".join(repr(name) for name in unknown)
        )

    counts = {region: 0 for region in CANONICAL_REGIONS}
    for row in rows:
        counts[str(row["region"])] += 1
    return {
        "montage_schema_version": MONTAGE_SCHEMA_VERSION,
        "canonical_regions": list(CANONICAL_REGIONS),
        "input_channel_count": len(names),
        "channel_order": list(names),
        "channels": rows,
        "region_channel_counts": counts,
    }


def montage_hash(manifest: Mapping[str, Any]) -> str:
    """Hash a resolved device montage, including its input channel order."""
    return _stable_hash(manifest)

