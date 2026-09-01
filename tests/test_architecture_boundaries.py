"""Regression guards for the final package-unification boundaries."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

from cogstate.adaptation.meta_learning.buffers import (
    architecture_schema_signature,
    stable_model_class_path,
)
from cogstate.model_zoo.DL.mlp import TorchMLP
from cogstate.model_zoo.factory import (
    SKLEARN_MODEL_NAMES,
    TORCH_MODEL_NAMES,
    build_model,
)
from cogstate.protocol import PM_METRICS


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = ("apps", "bench", "cogstate", "scripts", "tests")


def _python_paths(root: str) -> list[Path]:
    return sorted((REPO_ROOT / root).rglob("*.py"))


def _absolute_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_legacy_root_python_packages_are_absent() -> None:
    for name in ("model_zoo", "automl", "src"):
        assert not (REPO_ROOT / name).exists(), name


def test_project_python_does_not_import_legacy_roots() -> None:
    for root in PYTHON_ROOTS:
        for path in _python_paths(root):
            assert not ({"model_zoo", "automl", "src"} & _absolute_import_roots(path)), path


def test_cogstate_never_imports_bench() -> None:
    for path in _python_paths("cogstate"):
        assert "bench" not in _absolute_import_roots(path), path


def test_apps_depend_on_cogstate_not_bench() -> None:
    for path in _python_paths("apps"):
        assert "bench" not in _absolute_import_roots(path), path


def test_scripts_delegate_to_bench_or_apps_not_cogstate() -> None:
    for path in _python_paths("scripts"):
        assert "cogstate" not in _absolute_import_roots(path), path


def test_python_dynamic_paths_do_not_reference_legacy_packages() -> None:
    patterns = (
        re.compile(r"(?<!cogstate\.)\bmodel_zoo\."),
        re.compile(r"(?<!bench\.)\bautoml\."),
    )
    for root in PYTHON_ROOTS:
        for path in _python_paths(root):
            text = path.read_text(encoding="utf-8-sig")
            assert not [pattern.pattern for pattern in patterns if pattern.search(text)], path


def test_active_configs_and_manifests_do_not_use_legacy_module_paths() -> None:
    roots = ("configs", "experiments", "artifacts")
    suffixes = {".json", ".yaml", ".yml", ".toml"}
    patterns = (
        re.compile(r"(?<!cogstate\.)\bmodel_zoo\."),
        re.compile(r"(?<!bench\.)\bautoml\."),
    )
    for root in roots:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.casefold() in suffixes:
                text = path.read_text(encoding="utf-8-sig")
                assert not [pattern.pattern for pattern in patterns if pattern.search(text)], path
    for relative in (
        "reports/summary/config_curation.yaml",
        "reports/summary/requirements_registry.yaml",
    ):
        path = REPO_ROOT / relative
        text = path.read_text(encoding="utf-8-sig")
        assert not [pattern.pattern for pattern in patterns if pattern.search(text)], path


def test_cogstate_model_zoo_has_the_only_model_factory() -> None:
    factories = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in REPO_ROOT.rglob("factory.py")
        if "model_zoo" in path.parts
    )
    assert factories == ["cogstate/model_zoo/factory.py"]
    assert build_model.__module__ == "cogstate.model_zoo.factory"


def test_canonical_model_names_are_preserved() -> None:
    required = {
        "torch_mlp",
        "torch_lstm",
        "torch_bilstm",
        "torch_eegnet",
        "torch_shallow_convnet",
        "torch_shallow_convnet_multitask",
        "torch_shallow_fusion",
        "torch_transformer",
    }
    assert required <= TORCH_MODEL_NAMES
    assert SKLEARN_MODEL_NAMES


def test_model_metadata_uses_the_historical_logical_class_identity() -> None:
    model = TorchMLP(input_dim=4, num_classes=3, hidden_dims=(8,))
    assert model.__class__.__module__ == "cogstate.model_zoo.DL.mlp"
    historical_path = "model" + "_zoo.DL.mlp.TorchMLP"
    assert stable_model_class_path(model) == historical_path
    legacy_payload = {
        "class": historical_path,
        "parameters": [
            [name, list(value.shape), str(value.dtype)]
            for name, value in model.named_parameters()
        ],
        "buffers": [
            [name, list(value.shape), str(value.dtype)]
            for name, value in model.named_buffers()
        ],
        "latent_dim": int(getattr(model, "latent_dim", 0)),
    }
    legacy_signature = hashlib.sha256(
        json.dumps(
            legacy_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert architecture_schema_signature(model) == legacy_signature


def test_pm_registry_uses_canonical_protocol_constant() -> None:
    from bench.tasks.target_registry import PM_METRICS as REGISTRY_PM_METRICS

    assert REGISTRY_PM_METRICS is PM_METRICS


def test_scripts_are_thin_cli_modules() -> None:
    for path in _python_paths("scripts"):
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        assert len(source.splitlines()) <= 100, path
        assert not any(isinstance(node, ast.ClassDef) for node in ast.walk(tree)), path


def test_streaming_worker_uses_canonical_model_zoo() -> None:
    paths = _python_paths("apps")
    source = "\n".join(path.read_text(encoding="utf-8-sig") for path in paths)
    assert "from cogstate.model_zoo" in source
    assert not re.search(r"(?<!cogstate\.)\bmodel_zoo\.", source)


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
