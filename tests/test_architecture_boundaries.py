"""Regression guards for the post-unification package boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

from cogstate.model_zoo.DL.adapter import TorchClassificationAdapter as AppAdapter
from cogstate.model_zoo.DL.eegnet import TorchEEGNetClassifier as AppEEGNet
from cogstate.model_zoo.DL.lstm import TorchLSTMClassifier as AppLSTM
from cogstate.model_zoo.DL.mlp import TorchMLP as AppMLP
from cogstate.model_zoo.DL.shallow_convnet import (
    TorchShallowConvNetClassifier as AppShallowConvNet,
)
from cogstate.protocol import PM_METRICS
from model_zoo.DL.adapter import TorchClassificationAdapter
from model_zoo.DL.eegnet import TorchEEGNetClassifier
from model_zoo.DL.lstm import TorchLSTMClassifier
from model_zoo.DL.mlp import TorchMLP
from model_zoo.DL.shallow_convnet import TorchShallowConvNetClassifier


def test_application_model_zoo_reexports_canonical_models() -> None:
    assert AppAdapter is TorchClassificationAdapter
    assert AppEEGNet is TorchEEGNetClassifier
    assert AppLSTM is TorchLSTMClassifier
    assert AppMLP is TorchMLP
    assert AppShallowConvNet is TorchShallowConvNetClassifier


def test_pm_registry_uses_canonical_protocol_constant() -> None:
    from bench.tasks.target_registry import PM_METRICS as REGISTRY_PM_METRICS

    assert REGISTRY_PM_METRICS is PM_METRICS


def test_legacy_src_layer_is_absent() -> None:
    assert not Path("src").exists()


def test_project_python_does_not_import_src() -> None:
    roots = ("bench", "cogstate", "model_zoo", "automl", "apps", "scripts", "tests")
    for root in roots:
        for path in Path(root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots = {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots = {node.module.split(".")[0]}
                else:
                    continue
                assert "src" not in imported_roots, path


def test_canonical_replacement_clis_exist() -> None:
    cli_paths = (
        "scripts/data/inventory_data.py",
        "scripts/data/inspect_emotiv_files.py",
        "scripts/data/build_emotiv_catalog.py",
        "scripts/data/validate_emotiv_catalog.py",
        "scripts/data/build_emotiv_pm_windows.py",
        "scripts/data/build_legacy_emotiv_features.py",
        "scripts/data/audit_raw_eeg.py",
        "scripts/data/build_raw_eeg_window_cache.py",
        "scripts/data/audit_logical_recordings.py",
        "scripts/data/audit_raw_eeg_artifacts.py",
        "scripts/run_preprocessing_ablation.py",
        "scripts/analysis/audit_robust_feature_scaling.py",
        "scripts/analysis/build_experiment_summary.py",
        "scripts/analysis/audit_experiment_configs.py",
        "scripts/analysis/build_requirements_coverage.py",
        "scripts/analysis/build_colleague_metrics_package.py",
        "scripts/analysis/build_project_final_package.py",
        "scripts/data/build_pm_union_raw_cache.py",
        "scripts/run_preliminary_streaming_handoff.py",
    )
    assert not [path for path in cli_paths if not Path(path).is_file()]


def test_historical_prototypes_and_generic_preprocessing_duplicates_are_absent() -> None:
    obsolete = (
        "bench/tasks/mixin/contrastive_learning.py",
        "bench/tasks/mixin/domain_adaptation.py",
        "bench/tasks/mixin/metalearning.py",
        "bench/tasks/mixin/transfer_learning.py",
        "bench/preprocessing/artifacts.py",
        "bench/preprocessing/features.py",
        "bench/preprocessing/filters.py",
        "bench/preprocessing/preprocessing_pipeline.py",
        "cogstate/adaptation/domain_adaptation.py",
        "cogstate/adaptation/few_shot_calibration.py",
        "cogstate/adaptation/personalization.py",
        "cogstate/evaluation/cross_subject_eval.py",
        "cogstate/evaluation/multitask.py",
    )
    assert not [path for path in obsolete if Path(path).exists()]
