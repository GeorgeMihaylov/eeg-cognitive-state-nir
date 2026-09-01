"""Discovery and deterministic selection of completed benchmark runs.

The inventory deliberately validates artifact contents. Directory names are
used only to discover candidates and to break a complete semantic tie.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml

from bench.bench_runner import BenchmarkRunner


REPO_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_TRIALS = {
    (False, False, False): "A",
    (True, False, False): "B",
    (False, True, False): "C",
    (False, False, True): "D",
    (True, True, False): "E",
    (True, False, True): "F",
    (False, True, True): "G",
    (True, True, True): "H",
}


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as input_file:
        value = yaml.safe_load(input_file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _normalize_fold(value: Any) -> str:
    text = str(value)
    if text.startswith("fold_"):
        return text
    try:
        return f"fold_{int(float(text)):02d}"
    except ValueError:
        return text


def _standard_run_directory(prediction_file: Path) -> Path:
    # run/dataset/task/model/protocol/predictions.parquet
    if prediction_file.parent.name != "group_kfold_subject":
        raise ValueError(f"Not a standard unified prediction file: {prediction_file}")
    return prediction_file.parents[4]


def _unified_prediction_files(root: Path, *, kind: str) -> list[Path]:
    if kind == "calibration":
        return sorted(
            path
            for path in root.rglob("predictions.parquet")
            if (path.parent / "run_manifest.json").is_file()
        )
    return sorted(
        path
        for path in root.rglob("predictions.parquet")
        if path.parent.name == "group_kfold_subject"
    )


def _model_config(config: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    models = config.get("models", {})
    if not isinstance(models, Mapping) or not models:
        return "", {}
    name = str(next(iter(models)))
    value = models[name]
    return name, value if isinstance(value, Mapping) else {}


def _seed_from_config(config: Mapping[str, Any], default: int = 42) -> int:
    _, model = _model_config(config)
    params = model.get("params", {}) if isinstance(model, Mapping) else {}
    if isinstance(params, Mapping) and params.get("random_state") is not None:
        return int(params["random_state"])
    evaluation = config.get("evaluation", {})
    if isinstance(evaluation, Mapping) and evaluation.get("random_state") is not None:
        return int(evaluation["random_state"])
    return int(default)


def _preprocessing_identity(config: Mapping[str, Any]) -> tuple[str, str | None]:
    preprocessing = config.get("raw_preprocessing", {})
    if not isinstance(preprocessing, Mapping):
        return "not_applicable", None
    bandpass = bool(
        (preprocessing.get("bandpass", {}) or {}).get("enabled", False)
    )
    notch = bool((preprocessing.get("notch", {}) or {}).get("enabled", False))
    rereference = (preprocessing.get("rereference", {}) or {}).get("mode", "none")
    car = str(rereference) == "common_average"
    trial = PREPROCESSING_TRIALS.get((bandpass, notch, car))
    label = (
        f"bandpass={int(bandpass)};notch={int(notch)};car={int(car)}"
    )
    if not bandpass and not notch and not car:
        label = "raw"
    return label, trial


def _metrics_file(run_dir: Path, search_root: Path, timestamp: str) -> Path | None:
    direct = run_dir / "metrics.json"
    if direct.is_file():
        return direct
    candidates = list(search_root.glob(f"benchmark_results_{timestamp}.json"))
    return candidates[0] if candidates else None


def _manifest(run_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        return None, None
    try:
        return path, _load_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return path, None


def _prediction_identity(predictions: pd.DataFrame) -> str | None:
    if "sequence_id" in predictions.columns:
        return "sequence_id"
    if "sample_id" in predictions.columns:
        return "sample_id"
    return None


@dataclass(frozen=True)
class InventoryEntry:
    analysis_track: str
    model: str
    seed: int
    run_directory: str
    config_hash: str
    dataset: str
    representation: str
    preprocessing: str
    prediction_unit: str
    number_of_predictions: int
    subjects: int
    folds: int
    prediction_file: str
    metrics_file: str | None
    usable: bool
    reason: str
    canonical: bool = False
    manifest_status: str = "legacy_no_manifest"
    config_match: bool = False
    smoke_limited: bool = False
    identity_column: str | None = None
    fold_filter: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["fold_filter"] = list(self.fold_filter)
        return value


def _read_config(run_dir: Path, rule: Mapping[str, Any]) -> tuple[dict[str, Any], Path | None]:
    resolved = run_dir / "config.yaml"
    if resolved.is_file():
        return _load_yaml(resolved), resolved
    source = rule.get("config")
    if source:
        path = _repo_path(str(source))
        if path.is_file():
            return _load_yaml(path), path
    return {}, None


def _config_matches_rule(
    config: Mapping[str, Any],
    rule: Mapping[str, Any],
    predictions: pd.DataFrame,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    expected_config_path = rule.get("config")
    if expected_config_path and config:
        expected_config = _load_yaml(_repo_path(str(expected_config_path)))
        if (
            BenchmarkRunner.config_hash_for(config)
            != BenchmarkRunner.config_hash_for(expected_config)
        ):
            reasons.append("resolved config hash differs from configured reference")
    expected_type = rule.get("expected_model_type")
    model_name, model = _model_config(config)
    actual_type = model.get("type", model_name) if isinstance(model, Mapping) else model_name
    if expected_type and str(actual_type) != str(expected_type):
        reasons.append(f"model type {actual_type!r} != {expected_type!r}")
    expected_dataset = rule.get("dataset")
    datasets = config.get("datasets", {})
    if expected_dataset and config and expected_dataset not in datasets:
        reasons.append(f"dataset {expected_dataset!r} absent from resolved config")
    if expected_dataset and "dataset" in predictions and not predictions.empty:
        actual_datasets = set(predictions["dataset"].astype(str).unique())
        if actual_datasets != {str(expected_dataset)}:
            reasons.append(f"prediction datasets={sorted(actual_datasets)}")
    expected_length = rule.get("sequence_length")
    if expected_length is not None:
        if "sequence_length" not in predictions:
            reasons.append("sequence_length column missing")
        else:
            lengths = set(predictions["sequence_length"].dropna().astype(int))
            if lengths != {int(expected_length)}:
                reasons.append(
                    f"sequence lengths={sorted(lengths)}, expected={expected_length}"
                )
    expected_preprocessing = rule.get("preprocessing")
    actual_preprocessing, _ = _preprocessing_identity(config)
    if expected_preprocessing == "raw" and actual_preprocessing != "raw":
        reasons.append(f"preprocessing={actual_preprocessing}, expected raw")
    return not reasons, reasons


def _standard_entry(
    prediction_file: Path,
    rule: Mapping[str, Any],
    search_root: Path,
) -> InventoryEntry:
    run_dir = _standard_run_directory(prediction_file)
    predictions = pd.read_parquet(prediction_file)
    fold_filter = tuple(int(value) for value in rule.get("fold_filter", []))
    if fold_filter:
        fold_column = "fold" if "fold" in predictions else "outer_fold"
        normalized = predictions[fold_column].map(_normalize_fold)
        wanted = {_normalize_fold(value) for value in fold_filter}
        predictions = predictions.loc[normalized.isin(wanted)].copy()

    config, config_path = _read_config(run_dir, rule)
    manifest_path, manifest = _manifest(run_dir)
    manifest_status = (
        "invalid_manifest"
        if manifest_path is not None and manifest is None
        else str((manifest or {}).get("status", "legacy_no_manifest"))
    )
    config_hash = str((manifest or {}).get("config_hash", ""))
    computed_config_hash = BenchmarkRunner.config_hash_for(config) if config else ""
    if not config_hash and config:
        config_hash = computed_config_hash

    identity = _prediction_identity(predictions)
    fold_column = "fold" if "fold" in predictions else "outer_fold"
    folds = (
        len({_normalize_fold(value) for value in predictions[fold_column].dropna()})
        if fold_column in predictions
        else 0
    )
    subjects = (
        int(predictions["subject_id"].nunique())
        if "subject_id" in predictions
        else 0
    )
    dataset = str(rule.get("dataset", ""))
    if "dataset" in predictions and not predictions.empty:
        dataset = str(predictions["dataset"].iloc[0])
    preprocessing, trial = _preprocessing_identity(config)
    if rule.get("preprocessing") and not rule.get("dynamic_preprocessing"):
        preprocessing = str(rule["preprocessing"])

    model = str(rule["model"])
    if rule.get("dynamic_preprocessing"):
        model = f"shallowconvnet_trial_{trial or 'unknown'}"
    seed = _seed_from_config(config, int(rule.get("seed", 42)))
    expected_predictions = rule.get("expected_predictions")
    expected_folds = int(rule.get("expected_folds", 5))

    smoke_reasons: list[str] = []
    datasets = config.get("datasets", {})
    for value in datasets.values() if isinstance(datasets, Mapping) else []:
        if isinstance(value, Mapping) and value.get("max_windows") is not None:
            smoke_reasons.append("max_windows is set")
    evaluation = config.get("evaluation", {})
    if (
        isinstance(evaluation, Mapping)
        and evaluation.get("folds")
        and not rule.get("allow_fold_subset", False)
    ):
        selected = evaluation.get("folds") or []
        if len(selected) < int(evaluation.get("n_splits", expected_folds)):
            smoke_reasons.append("fold subset is set")
    if "smoke" in {part.lower() for part in run_dir.parts}:
        smoke_reasons.append("smoke path")

    config_match, config_reasons = _config_matches_rule(config, rule, predictions)
    reasons = list(config_reasons)
    if manifest is not None and computed_config_hash and config_hash != computed_config_hash:
        reasons.append("manifest config hash differs from resolved config")
    if manifest_status not in {"completed", "legacy_no_manifest"}:
        reasons.append(f"manifest status={manifest_status}")
    if folds != expected_folds:
        reasons.append(f"folds={folds}, expected={expected_folds}")
    if expected_predictions is not None and len(predictions) != int(expected_predictions):
        reasons.append(
            f"predictions={len(predictions)}, expected={int(expected_predictions)}"
        )
    if identity is None:
        reasons.append("prediction identity missing")
    elif predictions[identity].duplicated().any():
        reasons.append(f"duplicate {identity}")
    if "subject_id" not in predictions:
        reasons.append("subject_id missing")
    if "y_true" not in predictions:
        reasons.append("y_true missing")
    if smoke_reasons:
        reasons.append("smoke-limited: " + ", ".join(sorted(set(smoke_reasons))))

    metrics = _metrics_file(run_dir, search_root, run_dir.name)
    if metrics is None:
        reasons.append("metrics file missing")
    if config_path is None:
        reasons.append("resolved/source config missing")

    usable = not reasons
    if not usable:
        reason = "; ".join(reasons)
    elif manifest_status == "completed":
        reason = "completed manifest; complete folds and prediction identities"
    else:
        reason = "validated legacy run; complete folds and source config"

    return InventoryEntry(
        analysis_track=str(rule["analysis_track"]),
        model=model,
        seed=seed,
        run_directory=str(run_dir.resolve()),
        config_hash=config_hash,
        dataset=dataset,
        representation=str(rule["representation"]),
        preprocessing=preprocessing,
        prediction_unit=str(rule["prediction_unit"]),
        number_of_predictions=int(len(predictions)),
        subjects=subjects,
        folds=folds,
        prediction_file=str(prediction_file.resolve()),
        metrics_file=None if metrics is None else str(metrics.resolve()),
        usable=usable,
        reason=reason,
        manifest_status=manifest_status,
        config_match=config_match,
        smoke_limited=bool(smoke_reasons),
        identity_column=identity,
        fold_filter=fold_filter,
    )


def _calibration_entry(
    prediction_file: Path,
    rule: Mapping[str, Any],
    search_root: Path,
) -> InventoryEntry:
    run_dir = prediction_file.parent
    predictions = pd.read_parquet(prediction_file)
    _, manifest = _manifest(run_dir)
    status = str((manifest or {}).get("status", "missing_manifest"))
    methods = set(predictions.get("calibration_method", pd.Series(dtype=str)).astype(str))
    expected = set(map(str, rule.get("methods", [])))
    reasons: list[str] = []
    if status != "completed":
        reasons.append(f"manifest status={status}")
    if expected and not expected.issubset(methods):
        reasons.append(f"methods missing={sorted(expected - methods)}")
    if "smoke" in {part.lower() for part in run_dir.parts}:
        reasons.append("smoke-limited: smoke path")
    metrics_path = run_dir / "user_calibration_subjects.csv"
    if not metrics_path.is_file():
        reasons.append("calibration subject metrics file missing")
    identity = "sequence_id" if "sequence_id" in predictions else None
    if identity is None:
        reasons.append("sequence_id missing")
    folds = (
        predictions["outer_fold"].map(_normalize_fold).nunique()
        if "outer_fold" in predictions
        else 0
    )
    subjects = predictions["subject_id"].nunique() if "subject_id" in predictions else 0
    usable = not reasons
    return InventoryEntry(
        analysis_track=str(rule["analysis_track"]),
        model=str(rule["model"]),
        seed=int(rule.get("seed", 42)),
        run_directory=str(run_dir.resolve()),
        config_hash=str((manifest or {}).get("config_hash", "")),
        dataset=str(rule.get("dataset", "emotiv_cognitive")),
        representation=str(rule["representation"]),
        preprocessing=str(rule.get("preprocessing", "not_applicable")),
        prediction_unit=str(rule["prediction_unit"]),
        number_of_predictions=int(len(predictions)),
        subjects=int(subjects),
        folds=int(folds),
        prediction_file=str(prediction_file.resolve()),
        metrics_file=str(metrics_path.resolve()) if metrics_path.is_file() else None,
        usable=usable,
        reason=(
            "completed calibration manifest and matched prediction artifact"
            if usable
            else "; ".join(reasons)
        ),
        manifest_status=status,
        config_match=usable,
        smoke_limited=any("smoke" in reason for reason in reasons),
        identity_column=identity,
    )


def discover_rule_candidates(rule: Mapping[str, Any]) -> list[InventoryEntry]:
    """Discover every artifact candidate described by one analysis rule."""

    root = _repo_path(str(rule["search_root"]))
    if not root.exists():
        return []
    kind = str(rule.get("kind", "standard"))
    entries: list[InventoryEntry] = []
    for prediction_file in _unified_prediction_files(root, kind=kind):
        try:
            if kind == "calibration":
                entry = _calibration_entry(prediction_file, rule, root)
            else:
                entry = _standard_entry(prediction_file, rule, root)
        except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
            run_dir = prediction_file.parent
            entries.append(InventoryEntry(
                analysis_track=str(rule["analysis_track"]),
                model=str(rule["model"]),
                seed=int(rule.get("seed", 42)),
                run_directory=str(run_dir.resolve()),
                config_hash="",
                dataset=str(rule.get("dataset", "")),
                representation=str(rule["representation"]),
                preprocessing=str(rule.get("preprocessing", "unknown")),
                prediction_unit=str(rule["prediction_unit"]),
                number_of_predictions=0,
                subjects=0,
                folds=0,
                prediction_file=str(prediction_file.resolve()),
                metrics_file=None,
                usable=False,
                reason=f"artifact validation failed: {type(exc).__name__}: {exc}",
            ))
            continue
        expected_underlying = rule.get("expected_prediction_model")
        if expected_underlying and kind == "standard":
            frame = pd.read_parquet(prediction_file, columns=["model"])
            models = set(frame["model"].astype(str))
            if models != {str(expected_underlying)}:
                continue
        entries.append(entry)
    return entries


def _selection_key(entry: InventoryEntry) -> tuple[str, str, int]:
    return entry.analysis_track, entry.model, entry.seed


def _timestamp_key(entry: InventoryEntry) -> tuple[int, str]:
    name = Path(entry.run_directory).name
    return (int(name) if name.isdigit() else -1, entry.run_directory)


def select_canonical_runs(entries: Sequence[InventoryEntry]) -> list[InventoryEntry]:
    """Mark one deterministic canonical run per track/model/seed.

    A candidate must first be usable. Completed manifests outrank legacy
    outputs; complete folds, config equality, and absence of smoke limits are
    already hard validation requirements. Recency is only the final tie-break.
    """

    grouped: dict[tuple[str, str, int], list[InventoryEntry]] = {}
    for entry in entries:
        grouped.setdefault(_selection_key(entry), []).append(entry)
    selected: dict[tuple[str, str, int], str] = {}
    for key, values in grouped.items():
        usable = [entry for entry in values if entry.usable]
        if not usable:
            continue
        usable.sort(
            key=lambda entry: (
                entry.manifest_status == "completed",
                entry.folds,
                entry.config_match,
                not entry.smoke_limited,
                _timestamp_key(entry),
            ),
            reverse=True,
        )
        selected[key] = usable[0].run_directory
    return [
        replace(
            entry,
            canonical=(selected.get(_selection_key(entry)) == entry.run_directory),
        )
        for entry in entries
    ]


def build_run_inventory(spec: Mapping[str, Any]) -> list[InventoryEntry]:
    """Build and select an inventory from a loaded analysis specification."""

    entries: list[InventoryEntry] = []
    for rule in spec.get("run_rules", []):
        entries.extend(discover_rule_candidates(rule))
    selected = select_canonical_runs(entries)
    return sorted(
        selected,
        key=lambda entry: (
            entry.analysis_track,
            entry.model,
            entry.seed,
            entry.run_directory,
        ),
    )


def canonical_entries(
    entries: Iterable[InventoryEntry],
    *,
    tracks: Iterable[str] | None = None,
) -> list[InventoryEntry]:
    allowed = None if tracks is None else set(tracks)
    return [
        entry
        for entry in entries
        if entry.canonical and (allowed is None or entry.analysis_track in allowed)
    ]
