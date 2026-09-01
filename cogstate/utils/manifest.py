"""Reproducible experiment manifests, including PM-label treatment."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import json
import platform


def _serialise(value):
    if is_dataclass(value): return _serialise(asdict(value))
    if isinstance(value, dict): return {str(key): _serialise(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_serialise(item) for item in value]
    return value


def build_manifest(*, config, seed: int, fold_id: int, model_name: str):
    packages = {}
    for name in ("numpy", "scikit-learn", "torch", "scipy"):
        try: packages[name] = version(name)
        except PackageNotFoundError: pass
    return {"created_at": datetime.now(timezone.utc).isoformat(), "seed": seed, "external_fold": fold_id, "model": model_name, "config": _serialise(config), "python": platform.python_version(), "packages": packages}


def save_manifest(manifest, path: str | Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_serialise(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
