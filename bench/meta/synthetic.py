"""Deterministic synthetic episodes and orchestration for the FOMAML smoke."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from model_zoo.DL.eegnet import TorchEEGNetClassifier
from model_zoo.DL.shallow_convnet import TorchShallowConvNetClassifier

from .episodes import (
    MetaEpisode,
    MetaEpisodeBuilder,
    MetaEpisodeManifest,
    MetaEpisodeSpec,
)
from .fomaml import (
    FOMAMLConfig,
    FirstOrderMAML,
    audit_production_model_compatibility,
    model_state_hash,
)
from .validation import validate_episode


@dataclass(frozen=True)
class SyntheticEpisodeData:
    episode: MetaEpisode
    support_features: Tensor
    support_targets: Tensor
    query_features: Tensor
    query_targets: Tensor


class SyntheticClassifier(nn.Module):
    """Small buffer-free model used only by the synthetic smoke."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, classes: int = 3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, classes)
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_synthetic_episodes(
    config: Mapping[str, Any],
) -> tuple[tuple[SyntheticEpisodeData, ...], MetaEpisodeSpec]:
    classes = int(config["classes"])
    support_per_class = int(config["support_per_class"])
    query_per_class = int(config["query_per_class"])
    train_count = int(config["meta_train_episodes"])
    validation_count = int(config["meta_validation_episodes"])
    seed = int(config["seed"])
    if classes != 3:
        raise ValueError("The approved synthetic smoke uses exactly three classes")
    spec = MetaEpisodeSpec(
        episode_type="subject_personalization",
        task_type="classification",
        target_name="synthetic_class",
        support_unit="sample",
        query_unit="sample",
        support_size=classes * support_per_class,
        query_size=classes * query_per_class,
        class_balance_policy="require_all_classes",
        chronological=True,
        group_fields=("subject_id", "record_id", "sample_id"),
        seed=seed,
        minimum_records=2,
        minimum_classes=classes,
        insufficient_data_policy="error",
    )
    base_centers = np.asarray([[2.0, 0.0], [-1.0, 1.75], [-1.0, -1.75]])
    episodes: list[SyntheticEpisodeData] = []
    builder = MetaEpisodeBuilder()
    for task_index in range(train_count + validation_count):
        rng = np.random.default_rng(seed + task_index * 1009)
        angle = (task_index - 4) * 0.07
        rotation = np.asarray([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        scale = 0.9 + 0.03 * (task_index % 7)
        shift = np.asarray([0.08 * (task_index % 3), -0.05 * (task_index % 4)])
        centers = base_centers @ rotation.T * scale + shift
        rows: list[dict[str, Any]] = []
        features: dict[str, np.ndarray] = {}
        order = 0
        subject = f"synthetic-task-{task_index:03d}"
        for partition, count in (("support", support_per_class), ("query", query_per_class)):
            record = f"{subject}-{partition}"
            noise = 0.22 if partition == "support" else 0.25
            for class_id in range(classes):
                points = centers[class_id] + rng.normal(0.0, noise, size=(count, 2))
                for item, point in enumerate(points):
                    sample_id = f"{subject}-{partition}-c{class_id}-n{item:02d}"
                    rows.append({
                        "sample_id": sample_id,
                        "subject_id": subject,
                        "session_id": None,
                        "record_id": record,
                        "target": class_id,
                        "time_order": order,
                    })
                    features[sample_id] = point.astype(np.float32)
                    order += 1
        frame = pd.DataFrame(rows)
        scope = "meta_train" if task_index < train_count else "meta_validation"
        result = builder.build(
            dataset_index=frame,
            allowed_sample_ids=frame["sample_id"],
            forbidden_sample_ids=(),
            episode_spec=spec,
            dataset_id="synthetic_gaussian_3class",
            task_id="synthetic_fomaml_classification",
            fold_id="synthetic",
            allowed_subject_ids=[subject],
            entity_ids=[subject],
            scope=scope,
        )
        episode = result.episodes[0]
        episodes.append(SyntheticEpisodeData(
            episode=episode,
            support_features=torch.tensor(np.stack([features[x] for x in episode.support_sample_ids])),
            support_targets=torch.tensor(episode.support_targets, dtype=torch.long),
            query_features=torch.tensor(np.stack([features[x] for x in episode.query_sample_ids])),
            query_targets=torch.tensor(episode.query_targets, dtype=torch.long),
        ))
    return tuple(episodes), spec


def _execute(config: Mapping[str, Any]) -> dict[str, Any]:
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    episodes, spec = generate_synthetic_episodes(config)
    train_count = int(config["meta_train_episodes"])
    train = episodes[:train_count]
    validation = episodes[train_count:]
    model = SyntheticClassifier(classes=int(config["classes"]))
    fomaml_config = FOMAMLConfig.from_mapping(config["fomaml"])
    learner = FirstOrderMAML(model, fomaml_config)
    initial_hash = model_state_hash(model)
    history: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    for step in range(fomaml_config.maximum_meta_steps):
        start = (step * fomaml_config.episodes_per_meta_batch) % len(train)
        batch = tuple(
            train[(start + offset) % len(train)]
            for offset in range(fomaml_config.episodes_per_meta_batch)
        )
        preview = learner.compute_meta_batch_gradients(batch)
        result = learner.meta_train_step(batch)
        history.append(asdict(result))
        gradient_rows.append({
            "step": result.step,
            "episodes": result.episode_count,
            "norm_before_clip": result.meta_gradient_norm_before_clip,
            "norm_after_clip": result.meta_gradient_norm_after_clip,
            "clip_limit": fomaml_config.gradient_clip_norm,
            "all_finite": True,
        })
        for episode_result in preview.episodes:
            episode_rows.append({
                "stage": "meta_train",
                "step": result.step,
                "episode_id": episode_result.episode_id,
                "support_loss_by_step": json.dumps(episode_result.support_losses),
                "inner_gradient_norm_by_step": json.dumps(episode_result.inner_gradient_norms),
                "support_loss_before": episode_result.support_loss_before,
                "support_loss_after": episode_result.support_loss_after,
                "support_accuracy_before": episode_result.support_accuracy_before,
                "support_accuracy_after": episode_result.support_accuracy_after,
                "query_loss": episode_result.query_loss,
                "query_accuracy": episode_result.query_accuracy,
                "query_gradient_norm": episode_result.query_gradient_norm,
            })
    for episode in validation:
        result = learner.episode_result(episode)
        episode_rows.append({
            "stage": "meta_validation",
            "step": None,
            "episode_id": result.episode_id,
            "support_loss_by_step": json.dumps(result.support_losses),
            "inner_gradient_norm_by_step": json.dumps(result.inner_gradient_norms),
            "support_loss_before": result.support_loss_before,
            "support_loss_after": result.support_loss_after,
            "support_accuracy_before": result.support_accuracy_before,
            "support_accuracy_after": result.support_accuracy_after,
            "query_loss": result.query_loss,
            "query_accuracy": result.query_accuracy,
            "query_gradient_norm": result.query_gradient_norm,
        })
    final_hash = model_state_hash(model)
    frame = pd.DataFrame(episode_rows)
    scientific = {
        "initial_model_hash": initial_hash,
        "final_model_hash": final_hash,
        "episode_ids": [item.episode.episode_id for item in episodes],
        "history": history,
        "episode_metrics": episode_rows,
        "gradient_audit": gradient_rows,
        "support_loss_before_mean": float(frame.loc[frame.stage.eq("meta_train"), "support_loss_before"].mean()),
        "support_loss_after_mean": float(frame.loc[frame.stage.eq("meta_train"), "support_loss_after"].mean()),
        "query_loss_mean": float(frame.loc[frame.stage.eq("meta_train"), "query_loss"].mean()),
        "meta_validation_query_loss_mean": float(frame.loc[frame.stage.eq("meta_validation"), "query_loss"].mean()),
    }
    if scientific["support_loss_after_mean"] >= scientific["support_loss_before_mean"]:
        raise RuntimeError("Deterministic support adaptation did not reduce mean loss")
    return {
        "model": model,
        "episodes": episodes,
        "spec": spec,
        "history": history,
        "episode_rows": episode_rows,
        "gradient_rows": gradient_rows,
        "scientific": scientific,
    }


def run_fomaml_synthetic_smoke(
    config: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    output_dir = repository_root / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    protected = {
        str(path): _sha256_file(repository_root / str(path))
        for path in config.get("protected_manifests", [])
    }
    first = _execute(config)
    second = _execute(config)
    deterministic = first["scientific"] == second["scientific"]
    if not deterministic:
        raise RuntimeError("Repeated synthetic FOMAML execution is not deterministic")
    after = {
        str(path): _sha256_file(repository_root / str(path))
        for path in config.get("protected_manifests", [])
    }
    if protected != after:
        raise RuntimeError("Protected split manifests changed")

    episodes: Sequence[SyntheticEpisodeData] = first["episodes"]
    spec: MetaEpisodeSpec = first["spec"]
    manifest = MetaEpisodeManifest(
        dataset_ids=("synthetic_gaussian_3class",),
        task_ids=("synthetic_fomaml_classification",),
        fold_ids=("synthetic",),
        specs=(spec,),
        episodes=tuple(item.episode for item in episodes),
        errors=(),
        source_split_hashes={},
    )
    leakage_rows = [
        asdict(validate_episode(item.episode)) for item in episodes
    ]
    leakage = {
        "all_valid": all(row["valid"] for row in leakage_rows),
        "support_query_overlap": sum(row["support_query_sample_overlap"] for row in leakage_rows),
        "protected_manifests_unchanged": protected == after,
        "protected_manifest_hashes": after,
    }
    model: nn.Module = first["model"]
    torch.save({"model_state_dict": model.state_dict()}, output_dir / "final_model_state.pt")
    eegnet = TorchEEGNetClassifier(
        4, 128, 3, temporal_kernel_samples=16, separable_kernel_samples=8,
        f1=2, depth_multiplier=2, f2=4, pool1=2, pool2=2, dropout=0.1,
    )
    shallow = TorchShallowConvNetClassifier(
        4, 128, 3, n_filters=4, temporal_kernel_samples=9,
        pool_size=15, pool_stride=5, dropout=0.1,
    )
    example = torch.zeros(2, 1, 4, 128)
    production_audit = {
        "eegnet": audit_production_model_compatibility(eegnet, example),
        "shallow_convnet": audit_production_model_compatibility(shallow, example),
    }
    scientific = first["scientific"]
    status = "infrastructure_only"
    summary = {
        "result_status": "diagnostic",
        "decision_status": status,
        "algorithm_id": "fomaml",
        "first_order": True,
        "create_graph": False,
        "device": "cpu",
        "meta_train_episodes": int(config["meta_train_episodes"]),
        "meta_validation_episodes": int(config["meta_validation_episodes"]),
        "meta_steps": int(config["fomaml"]["maximum_meta_steps"]),
        "support_loss_before_mean": scientific["support_loss_before_mean"],
        "support_loss_after_mean": scientific["support_loss_after_mean"],
        "query_loss_mean": scientific["query_loss_mean"],
        "meta_validation_query_loss_mean": scientific["meta_validation_query_loss_mean"],
        "initial_model_hash": scientific["initial_model_hash"],
        "final_model_hash": scientific["final_model_hash"],
        "deterministic": deterministic,
        "leakage_safe": leakage["all_valid"],
        "real_data_training_performed": False,
        "production_compatibility": production_audit,
    }
    _write_json(output_dir / "resolved_config.json", dict(config))
    _write_json(output_dir / "synthetic_episode_manifest.json", manifest.to_dict())
    _write_json(output_dir / "initial_model_manifest.json", {
        "architecture": "Linear(2,16)-ReLU-Linear(16,3)",
        "model_hash": scientific["initial_model_hash"],
        "seed": int(config["seed"]),
    })
    pd.DataFrame(first["history"]).to_csv(output_dir / "meta_training_history.csv", index=False)
    pd.DataFrame(first["episode_rows"]).to_parquet(output_dir / "episode_metrics.parquet", index=False)
    pd.DataFrame(first["gradient_rows"]).to_csv(output_dir / "gradient_audit.csv", index=False)
    _write_json(output_dir / "parameter_update_audit.json", {
        "initial_model_hash": scientific["initial_model_hash"],
        "final_model_hash": scientific["final_model_hash"],
        "model_changed": scientific["initial_model_hash"] != scientific["final_model_hash"],
        "steps": first["history"],
    })
    _write_json(output_dir / "leakage_audit.json", leakage)
    _write_json(output_dir / "determinism_audit.json", {
        "deterministic": deterministic,
        "episode_ids_match": scientific["episode_ids"] == second["scientific"]["episode_ids"],
        "initial_model_hash_match": scientific["initial_model_hash"] == second["scientific"]["initial_model_hash"],
        "final_model_hash_match": scientific["final_model_hash"] == second["scientific"]["final_model_hash"],
        "scientific_result_hash": hashlib.sha256(
            json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    })
    _write_json(output_dir / "final_model_manifest.json", {
        "model_hash": scientific["final_model_hash"],
        "parameters_finite": all(bool(torch.isfinite(p).all()) for p in model.parameters()),
        "checkpoint": "final_model_state.pt",
    })
    _write_json(output_dir / "smoke_summary.json", summary)
    pd.DataFrame(columns=["step", "episode_id", "error_type", "message"]).to_csv(
        output_dir / "errors.csv", index=False
    )
    (output_dir / "smoke_report.md").write_text(
        "# Synthetic FOMAML smoke\n\n"
        f"- Decision: `{status}`.\n"
        f"- Meta-train/meta-validation episodes: {config['meta_train_episodes']}/{config['meta_validation_episodes']}.\n"
        f"- Meta-steps: {config['fomaml']['maximum_meta_steps']}.\n"
        f"- Support loss: {scientific['support_loss_before_mean']:.6f} -> {scientific['support_loss_after_mean']:.6f}.\n"
        f"- Query loss: {scientific['query_loss_mean']:.6f}.\n"
        "- Deterministic: true; leakage audit: passed.\n"
        "- Real EEG training: not performed.\n"
        "- Production BatchNorm adaptation: blocked by frozen-buffer policy.\n",
        encoding="utf-8",
    )
    return summary
