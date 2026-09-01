from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bench.bench_runner import (
    BenchmarkRunner,
    CompletedBenchmarkRun,
    benchmark_config_hash,
)
from bench.datasets.raw_eeg_window_dataset import (
    CANONICAL_EEG_CHANNELS,
    RAW_LOADER_VERSION,
    _cache_config_hash,
)
from bench.experiments.preprocessing_ablation import (
    FACTOR_PATHS,
    PreprocessingAblation,
    expand_factorial_trials,
    load_experiment_spec,
    resolve_trial_config,
    resolve_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SPEC = REPO_ROOT / "experiments" / "preprocessing_ablation_shallowconvnet.yaml"


def _isolated_spec(tmp_path: Path, *, candidate: Path | None = None) -> Path:
    document = deepcopy(load_experiment_spec(BASE_SPEC))
    for name in ("processed.bin", "catalog.bin", "schema.bin"):
        (tmp_path / name).write_bytes(name.encode())
    document["cache"].update(
        {
            "cache_dir": str(tmp_path / "cache"),
            "index_dir": str(tmp_path / "generated_indices"),
            "processed_path": str(tmp_path / "processed.bin"),
            "catalog_path": str(tmp_path / "catalog.bin"),
            "audit_schema_path": str(tmp_path / "schema.bin"),
            "candidate_index_paths": [] if candidate is None else [str(candidate)],
            "estimated_cache_size_bytes": 1000,
            "minimum_free_reserve_bytes": 0,
        }
    )
    document["experiment"]["output_dir"] = str(tmp_path / "results")
    document["dataset"]["logical_recording_map_path"] = str(
        tmp_path / "logical.parquet"
    )
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _write_synthetic_cache(
    experiment: PreprocessingAblation,
    tmp_path: Path,
    *,
    dtype: np.dtype = np.dtype(np.float32),
) -> Path:
    trial = next(item for item in experiment.trials if item.trial_id == "A")
    cache_root = tmp_path / "synthetic_cache"
    cache_root.mkdir()
    raw_path = tmp_path / "raw_source.bin"
    raw_path.write_bytes(b"raw-source-identity")
    array_path = cache_root / "record.npy"
    np.save(array_path, np.ones((2, 14, 2560), dtype=dtype), allow_pickle=False)
    windows = [
        {"sample_id": 1, "t_start": 5.0, "t_end": 15.0},
        {"sample_id": 2, "t_start": 15.0, "t_end": 25.0},
    ]
    legacy = trial.preprocessing.to_legacy_raw_preprocessing()
    config_hash = _cache_config_hash(
        {"record_id": "record", "windows": windows},
        raw_path,
        CANONICAL_EEG_CHANNELS,
        256.0,
        0.02,
        legacy,
    )
    metadata = {
        "config_hash": config_hash,
        "loader_version": RAW_LOADER_VERSION,
        "record_id": "record",
        "raw_file_path": str(raw_path),
        "channels": list(CANONICAL_EEG_CHANNELS),
        "sfreq_original": 256.0,
        "sfreq_target": 256.0,
        "raw_preprocessing": legacy,
        "preprocessing_hash": trial.legacy_preprocessing_hash,
        "accepted_windows": 2,
        "window_results": [
            {
                "sample_id": item["sample_id"],
                "status": "ok",
                "rejection_reason": "",
                "cache_offset": index,
                "missing_fraction": 0.0,
                "sfreq_original": 256.0,
            }
            for index, item in enumerate(windows)
        ],
    }
    array_path.with_suffix(".json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    index_path = tmp_path / "index.parquet"
    pd.DataFrame(
        {
            "record_id": ["record", "record"],
            "sample_id": [1, 2],
            "t_start": [5.0, 15.0],
            "t_end": [15.0, 25.0],
            "status": ["ok", "ok"],
            "cache_file": [str(array_path), str(array_path)],
            "cache_offset": [0, 1],
            "sfreq_target": [256.0, 256.0],
            "preprocessing_hash": [
                trial.legacy_preprocessing_hash,
                trial.legacy_preprocessing_hash,
            ],
        }
    ).to_parquet(index_path, index=False)
    return index_path


def _write_standard_run(plan, *, with_manifest: bool = True) -> CompletedBenchmarkRun:
    run_dir = plan.benchmark_output_dir / "20260101_000000"
    fold_dir = run_dir / "f"
    fold_dir.mkdir(parents=True)
    predictions_path = fold_dir / "predictions.parquet"
    pd.DataFrame(
        {"sample_id": [1, 2], "fold": [1, 1], "y_true": [0, 1], "y_pred": [0, 1]}
    ).to_parquet(predictions_path, index=False)
    metrics_path = fold_dir / "metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")
    unified_path = run_dir / "predictions.parquet"
    pd.read_parquet(predictions_path).to_parquet(unified_path, index=False)
    results = {
        "dataset": {
            "models": {
                "task": {
                    "model": {
                        "group_kfold_subject": {
                            "n_folds": 1,
                            "folds": {
                                "fold_01": {
                                    "n_test": 2,
                                    "artifacts": {
                                        "predictions": str(predictions_path),
                                        "metrics": str(metrics_path),
                                    },
                                }
                            },
                            "artifacts": {"predictions": str(unified_path)},
                        }
                    }
                }
            }
        }
    }
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(dict(plan.resolved_config), sort_keys=False), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(json.dumps(results), encoding="utf-8")
    result_file = run_dir.parent / f"benchmark_results_{run_dir.name}.json"
    result_file.write_text(json.dumps(results), encoding="utf-8")
    summary_file = run_dir.parent / f"summary_{run_dir.name}.csv"
    summary_file.write_text("dataset\ndataset\n", encoding="utf-8")
    manifest_path = run_dir / "run_manifest.json"
    if with_manifest:
        manifest_path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "config_hash": plan.config_hash,
                    "benchmark_result_file": str(result_file),
                }
            ),
            encoding="utf-8",
        )
    return BenchmarkRunner.validate_completed_run(
        run_dir,
        expected_config_hash=plan.config_hash,
        result_file=result_file,
        manifest_file=manifest_path if with_manifest else None,
        legacy=not with_manifest,
    )


def test_factorial_expansion_is_deterministic_and_unique():
    document = load_experiment_spec(BASE_SPEC)
    first = expand_factorial_trials(document)
    second = expand_factorial_trials(deepcopy(document))
    assert [trial.trial_id for trial in first] == list("ABCDEFGH")
    assert [trial.parameter_dict() for trial in first] == [
        trial.parameter_dict() for trial in second
    ]
    assert len({trial.preprocessing_hash for trial in first}) == 8
    assert len({trial.cache_key_hash for trial in first}) == 8
    assert set(first[0].parameter_dict()) == set(FACTOR_PATHS)


def test_plan_only_calculation_does_not_create_output_or_cache_files(tmp_path):
    spec_path = _isolated_spec(tmp_path)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    experiment = PreprocessingAblation(spec_path)
    plans = experiment.plan(seed=42, fold_limit=1, max_windows=1000, max_epochs=3)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert before == after
    assert len(plans) == 8
    assert all(plan.action == "build_cache_and_run" for plan in plans)
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "cache").exists()


def test_semantic_lookup_reuses_complete_cache_and_rejects_corruption(tmp_path):
    placeholder_spec = _isolated_spec(tmp_path)
    placeholder_experiment = PreprocessingAblation(placeholder_spec)
    index_path = _write_synthetic_cache(placeholder_experiment, tmp_path)
    spec_path = _isolated_spec(tmp_path, candidate=index_path)
    experiment = PreprocessingAblation(spec_path)
    trial = next(item for item in experiment.trials if item.trial_id == "A")
    resolution = resolve_cache(trial, experiment.document)
    assert resolution.exists
    assert resolution.complete
    assert resolution.reusable
    assert resolution.index_path == index_path

    array_path = tmp_path / "synthetic_cache" / "record.npy"
    np.save(array_path, np.ones((2, 14, 2560), dtype=np.float64), allow_pickle=False)
    corrupted = resolve_cache(trial, experiment.document)
    assert not corrupted.reusable


def test_parameter_change_changes_semantic_and_legacy_hashes():
    trials = expand_factorial_trials(load_experiment_spec(BASE_SPEC))
    raw = next(trial for trial in trials if trial.trial_id == "A")
    bandpass = next(trial for trial in trials if trial.trial_id == "B")
    assert raw.preprocessing_hash != bandpass.preprocessing_hash
    assert raw.legacy_preprocessing_hash != bandpass.legacy_preprocessing_hash


def test_resume_uses_validated_standard_benchmark_result(tmp_path):
    experiment = PreprocessingAblation(_isolated_spec(tmp_path))
    plan = experiment.plan(trial_ids=["A"], seed=42)[0]
    completed = _write_standard_run(plan)

    refreshed = experiment.plan(trial_ids=["A"], seed=42)[0]
    assert refreshed.action == "skip_completed"
    assert refreshed.completed_run == completed

    different_seed = experiment.plan(trial_ids=["A"], seed=7)[0]
    assert different_seed.action != "skip_completed"
    assert different_seed.config_hash != plan.config_hash


def test_completed_smoke_trial_is_not_a_completed_full_trial(tmp_path):
    experiment = PreprocessingAblation(_isolated_spec(tmp_path))
    smoke = experiment.plan(
        trial_ids=["A"],
        seed=42,
        fold_limit=1,
        max_windows=1000,
        max_epochs=3,
    )[0]
    full = experiment.plan(trial_ids=["A"], seed=42)[0]

    assert smoke.run_mode == "smoke"
    assert full.run_mode == "full"
    assert smoke.reference_path != full.reference_path
    assert smoke.config_hash != full.config_hash
    assert smoke.resolved_config["evaluation"]["folds"] == [1]
    assert smoke.resolved_config["datasets"]["emotiv_raw_eeg"]["max_windows"] == 1000
    assert smoke.resolved_config["models"]["torch_shallow_convnet"]["params"]["max_epochs"] == 3
    assert "folds" not in full.resolved_config["evaluation"]
    assert full.resolved_config["models"]["torch_shallow_convnet"]["params"]["max_epochs"] == 15

    _write_standard_run(smoke)
    refreshed_full = experiment.plan(trial_ids=["A"], seed=42)[0]
    assert refreshed_full.action != "skip_completed"


def test_benchmark_config_preserves_leakage_safe_protocol(tmp_path):
    placeholder = PreprocessingAblation(_isolated_spec(tmp_path))
    index_path = _write_synthetic_cache(placeholder, tmp_path)
    experiment = PreprocessingAblation(_isolated_spec(tmp_path, candidate=index_path))
    plan = experiment.plan(trial_ids=["A"], seed=7, fold_limit=1)[0]
    config = plan.resolved_config
    assert config["validation"]["group_column"] == "record_group_id"
    assert config["evaluation"]["group_column"] == "subject_id"
    assert config["evaluation"]["n_splits"] == 5
    assert config["datasets"]["emotiv_raw_eeg"]["dataset_mode"] == (
        "raw_deduplicated_logical_records"
    )
    assert config["models"]["torch_shallow_convnet"]["params"][
        "random_state"
    ] == 7


def test_resolver_is_deterministic_and_rejects_unknown_parameters(tmp_path):
    experiment = PreprocessingAblation(_isolated_spec(tmp_path))
    plan = experiment.plan(trial_ids=["E"], seed=7)[0]
    parameters = {
        "preprocessing.bandpass.enabled": True,
        "preprocessing.notch.enabled": True,
        "preprocessing.car.enabled": False,
        "training.random_state": 7,
    }
    base = experiment._base_benchmark_config(plan.cache)
    first = resolve_trial_config(base, parameters)
    second = resolve_trial_config(deepcopy(base), deepcopy(parameters))
    assert first == second
    assert benchmark_config_hash(first) == benchmark_config_hash(second)
    assert first["raw_preprocessing"]["rereference"]["mode"] == "none"
    assert first["models"]["torch_shallow_convnet"]["params"]["random_state"] == 7

    try:
        resolve_trial_config(base, {"optimizer.secret": 1})
    except ValueError as exc:
        assert "Unsupported trial parameters" in str(exc)
    else:
        raise AssertionError("Unknown AutoML-style parameters must be rejected")


def test_cache_resume_does_not_imply_benchmark_resume(tmp_path):
    placeholder = PreprocessingAblation(_isolated_spec(tmp_path))
    index_path = _write_synthetic_cache(placeholder, tmp_path)
    experiment = PreprocessingAblation(_isolated_spec(tmp_path, candidate=index_path))
    plan = experiment.plan(trial_ids=["A"], seed=123)[0]
    assert plan.cache.reusable
    assert plan.completed_run is None
    assert plan.action == "reuse_cache_and_run"


def test_matrix_invokes_runner_once_and_only_writes_reference_metadata(
    tmp_path, monkeypatch
):
    placeholder = PreprocessingAblation(_isolated_spec(tmp_path))
    index_path = _write_synthetic_cache(placeholder, tmp_path)
    experiment = PreprocessingAblation(_isolated_spec(tmp_path, candidate=index_path))
    plan = experiment.plan(
        trial_ids=["A"], seed=7, fold_limit=1, max_windows=1000, max_epochs=1
    )[0]
    standard_dir = tmp_path / "standard-run"
    standard_dir.mkdir()
    result_file = standard_dir / "benchmark_results.json"
    result_file.write_text("{}", encoding="utf-8")
    completed = CompletedBenchmarkRun(
        config_hash=plan.config_hash,
        run_directory=standard_dir,
        result_file=result_file,
        summary_file=None,
        manifest_file=standard_dir / "run_manifest.json",
    )

    class SpyRunner:
        calls = []

        @classmethod
        def find_completed_run(cls, config, *, search_directories=None):
            return None

        def __init__(self, config):
            self.config = config
            self.calls.append(config)

        def run(self):
            return pd.DataFrame()

        def completed_run(self):
            return completed

    monkeypatch.setattr(
        "bench.experiments.preprocessing_ablation.BenchmarkRunner", SpyRunner
    )
    result = experiment.run_trial(
        plan, build_missing_cache=False, resume=True
    )

    assert len(SpyRunner.calls) == 1
    assert SpyRunner.calls[0] == plan.resolved_config
    assert set(path.name for path in plan.reference_path.iterdir()) == {
        "resolved_trial.yaml",
        "trial_reference.json",
    }
    assert result["trial_reference"]["benchmark_run_directory"] == str(standard_dir)


def test_legacy_standard_run_is_found_by_semantic_config_match(tmp_path):
    experiment = PreprocessingAblation(_isolated_spec(tmp_path))
    plan = experiment.plan(trial_ids=["A"], seed=42)[0]
    legacy_root = plan.legacy_output_dir
    original_output = plan.benchmark_output_dir
    object.__setattr__(plan, "benchmark_output_dir", legacy_root)
    legacy = _write_standard_run(plan, with_manifest=False)
    object.__setattr__(plan, "benchmark_output_dir", original_output)

    refreshed = experiment.plan(trial_ids=["A"], seed=42)[0]
    assert refreshed.completed_run is not None
    assert refreshed.completed_run.legacy
    assert refreshed.completed_run.run_directory == legacy.run_directory


def test_cli_and_programmatic_matrix_use_identical_resolver(
    tmp_path, monkeypatch, capsys
):
    from cli import main as cli_main

    spec_path = _isolated_spec(tmp_path)
    expected = PreprocessingAblation(spec_path).plan(
        trial_ids=["E"], seed=7, fold_limit=1, max_windows=1000, max_epochs=3
    )[0]
    captured = {}

    def fake_execute(self, plans, **kwargs):
        captured["plan"] = plans[0]
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(PreprocessingAblation, "execute", fake_execute)
    cli_main([
        "--experiment-matrix", str(spec_path),
        "--trial-ids", "E",
        "--seed", "7",
        "--fold-limit", "1",
        "--max-windows", "1000",
        "--max-epochs", "3",
        "--resume",
    ])
    capsys.readouterr()

    assert captured["plan"].resolved_config == expected.resolved_config
    assert captured["plan"].config_hash == expected.config_hash
    assert captured["kwargs"]["resume"] is True
