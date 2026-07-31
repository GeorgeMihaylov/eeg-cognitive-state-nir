from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from bench.meta import run_fomaml_synthetic_smoke


def _config() -> dict:
    return {
        "result_status": "diagnostic",
        "output_dir": "output",
        "seed": 42,
        "device": "cpu",
        "classes": 3,
        "support_per_class": 5,
        "query_per_class": 10,
        "meta_train_episodes": 4,
        "meta_validation_episodes": 2,
        "protected_manifests": ["protected/main.json", "protected/cog.json"],
        "fomaml": {
            "algorithm_id": "fomaml",
            "task_type": "classification",
            "inner_steps": 2,
            "inner_learning_rate": 0.1,
            "meta_learning_rate": 0.01,
            "episodes_per_meta_batch": 2,
            "maximum_meta_steps": 2,
            "loss_name": "cross_entropy",
            "gradient_clip_norm": 5.0,
            "buffer_policy": "frozen",
            "device": "cpu",
            "seed": 42,
            "finite_check": True,
        },
    }


def test_synthetic_smoke_writes_deterministic_safe_artifacts(tmp_path: Path) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    for name in ("main.json", "cog.json"):
        (protected / name).write_text('{"split":"unchanged"}\n', encoding="utf-8")
    before = {path.name: path.read_bytes() for path in protected.iterdir()}
    first = run_fomaml_synthetic_smoke(_config(), repository_root=tmp_path)
    second = run_fomaml_synthetic_smoke(_config(), repository_root=tmp_path)
    assert first == second
    assert first["decision_status"] == "infrastructure_only"
    assert first["deterministic"] and first["leakage_safe"]
    assert not first["real_data_training_performed"]
    assert first["support_loss_after_mean"] < first["support_loss_before_mean"]
    expected = {
        "resolved_config.json", "synthetic_episode_manifest.json",
        "initial_model_manifest.json", "meta_training_history.csv",
        "episode_metrics.parquet", "gradient_audit.csv",
        "parameter_update_audit.json", "leakage_audit.json",
        "determinism_audit.json", "final_model_state.pt",
        "final_model_manifest.json", "smoke_summary.json", "errors.csv",
        "smoke_report.md",
    }
    output = tmp_path / "output"
    assert expected == {path.name for path in output.iterdir()}
    assert before == {path.name: path.read_bytes() for path in protected.iterdir()}
    pattern = re.compile(r"[A-Za-z]:[\\/]")
    for path in output.iterdir():
        if path.suffix in {".json", ".csv", ".md"}:
            assert not pattern.search(path.read_text(encoding="utf-8"))
    leakage = json.loads((output / "leakage_audit.json").read_text())
    assert leakage["support_query_overlap"] == 0
    assert leakage["protected_manifests_unchanged"]


def test_historical_mixin_and_optional_packages_are_not_runtime_dependencies() -> None:
    import bench.meta.fomaml as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "learn2learn" not in source
    assert "higher" not in source
    assert "bench.tasks.mixin" not in source
    assert "learn2learn" not in sys.modules


def test_synthetic_orchestration_has_no_dataset_loader_import() -> None:
    import bench.meta.synthetic as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "bench.datasets" not in source
    assert "data/processed" not in source
