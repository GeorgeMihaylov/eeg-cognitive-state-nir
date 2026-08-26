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


def test_src_contains_only_thin_compatibility_entry_points() -> None:
    tracked = sorted(Path("src").glob("*.py"))
    assert len(tracked) == 19
    for path in tracked:
        source = path.read_text(encoding="utf-8")
        assert len(source.splitlines()) <= 25, path
        tree = ast.parse(source, filename=str(path))
        forbidden = (
            ast.ClassDef,
            ast.AsyncFunctionDef,
        )
        assert not any(isinstance(node, forbidden) for node in ast.walk(tree)), path


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
