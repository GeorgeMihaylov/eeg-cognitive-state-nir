"""Stable training/runtime contracts for deployable streaming bundles."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_schema_hash(feature_names: Iterable[str]) -> str:
    return stable_hash([str(name) for name in feature_names])


def preprocessing_contract(
    *,
    sample_rate: float,
    bandpass_enabled: bool,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    notch_enabled: bool,
    notch_hz: float,
    faster: bool,
) -> dict[str, Any]:
    return {
        "sample_rate_hz": float(sample_rate),
        "streaming_filter": {
            "bandpass_enabled": bool(bandpass_enabled),
            "bandpass_low_hz": float(bandpass_low_hz),
            "bandpass_high_hz": float(bandpass_high_hz),
            "notch_enabled": bool(notch_enabled),
            "notch_hz": float(notch_hz),
        },
        "faster_enabled": bool(faster),
        "rereference": "none",
        "input_dtype": "float32",
    }


def preprocessing_hash(contract: Mapping[str, Any]) -> str:
    return stable_hash(dict(contract))
