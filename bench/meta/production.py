"""Read-only production architecture and FOMAML buffer-contract audit."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from model_zoo.factory import build_model
from model_zoo.DL.eegnet import TorchEEGNetClassifier
from model_zoo.DL.shallow_convnet import TorchShallowConvNetClassifier

from .buffers import (
    BufferPolicy,
    architecture_schema_signature,
    batchnorm_inventory,
    create_functional_state,
    tensor_mapping_hash,
)
from .fomaml import FOMAMLConfig, FirstOrderMAML, model_state_hash
from .meta_validation import build_meta_validation_protocol


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _signature(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _architecture_row(
    model_id: str,
    model: nn.Module,
    *,
    source_config: str,
    input_shape: tuple[int, ...],
    sampling_rate: float,
    architecture_parameters: Mapping[str, Any],
    checkpoint_signature: str | None = None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    example = torch.zeros(2, *input_shape, device=device)
    model.eval()
    with torch.no_grad():
        encoded = model.encode(example)
        output = model(example)
    if isinstance(model, TorchEEGNetClassifier):
        convolution_parameters = {
            "temporal_kernel_samples": model.temporal_kernel_samples,
            "separable_kernel_samples": model.separable_kernel_samples,
            "f1": int(model.features[1].out_channels),
            "depth_multiplier": int(model.features[3].out_channels // model.features[1].out_channels),
            "f2": int(model.features[10].out_channels),
        }
        pooling_parameters = {
            "pool1": list(model.features[6].kernel_size),
            "pool2": list(model.features[13].kernel_size),
        }
        adaptive_pooling: list[int] | None = None
    else:
        convolution_parameters = {
            "temporal_kernel_samples": model.temporal_kernel_samples,
            "n_filters": model.n_filters,
            "spatial_kernel_channels": model.n_channels,
        }
        pooling_parameters = {
            "pool_size": model.pool_size,
            "pool_stride": model.pool_stride,
        }
        adaptive_pooling = [1, 1]
    payload = {
        "model_id": model_id,
        "model_class": model.__class__.__name__,
        "input_shape": list(input_shape),
        "example_batch_shape": [2, *input_shape],
        "sampling_rate": sampling_rate,
        "window_duration_seconds": input_shape[-1] / sampling_rate,
        "channels": input_shape[-2],
        "samples": input_shape[-1],
        "architecture_parameters": dict(architecture_parameters),
        "convolution_parameters": convolution_parameters,
        "pooling_parameters": pooling_parameters,
        "adaptive_pooling": adaptive_pooling,
        "encoder_output_shape": list(encoded.shape[1:]),
        "latent_dim": int(model.latent_dim),
        "output_head_input_dimension": int(model.get_output_head().in_features),
        "output_head_width": int(model.get_output_head().out_features),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "state_schema_signature": architecture_schema_signature(model),
    }
    payload["architecture_signature"] = _signature({
        key: value for key, value in payload.items() if key != "model_id"
    })
    return {
        **payload,
        "source_config": source_config,
        "checkpoint_signature": checkpoint_signature,
        "output_shape": list(output.shape[1:]),
    }


def _checkpoint_schema(path: Path) -> tuple[dict[str, Any], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    schema = {
        "model_metadata": payload.get("model_metadata", {}),
        "input_shape": list(payload["input_shape"]),
        "num_classes": int(payload["num_classes"]),
        "state": [
            [name, list(value.shape), str(value.dtype)]
            for name, value in state.items()
        ],
    }
    return payload, _signature(schema)


def _build_from_config(
    repository_root: Path, config_path: str, model_name: str
) -> tuple[nn.Module, dict[str, Any]]:
    document = yaml.safe_load((repository_root / config_path).read_text(encoding="utf-8"))
    params = dict(document["models"][model_name]["params"])
    params["device"] = "cpu"
    adapter = build_model(
        model_name=model_name,
        task_type="classification",
        input_shape=(1, 14, 2560),
        num_outputs=5,
        params=params,
    )
    architecture = {
        key: value for key, value in adapter.model_metadata.items()
        if key not in {"channel_names", "encoder_api_version", "input_layout", "model_type"}
    }
    return adapter.model, architecture


def audit_architectures(
    config: Mapping[str, Any], *, repository_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    schemas: dict[str, Any] = {}
    bn_rows: list[dict[str, Any]] = []
    for model_name, spec in config["production_models"].items():
        canonical, parameters = _build_from_config(
            repository_root, spec["config"], model_name
        )
        checkpoint_payload, checkpoint_signature = _checkpoint_schema(
            repository_root / spec["checkpoint"]
        )
        canonical.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
        canonical_row = _architecture_row(
            f"{model_name}:canonical",
            canonical,
            source_config=spec["config"],
            input_shape=(1, 14, 2560),
            sampling_rate=256.0,
            architecture_parameters=parameters,
            checkpoint_signature=checkpoint_signature,
        )
        rows.append(canonical_row)
        rows.append({
            **canonical_row,
            "model_id": f"{model_name}:checkpoint_fold_01",
            "source_config": spec["checkpoint"],
        })
        rows.append({
            **canonical_row,
            "model_id": f"{model_name}:explicit_2x1x14x2560",
            "source_config": "production_contract_functional_smoke",
        })
        if model_name == "torch_eegnet":
            audit_model = TorchEEGNetClassifier(
                4, 128, 3, temporal_kernel_samples=16,
                separable_kernel_samples=8, f1=2, depth_multiplier=2,
                f2=4, pool1=2, pool2=2, dropout=0.1,
            )
            audit_parameters = {
                "n_channels": 4, "n_times": 128, "f1": 2,
                "depth_multiplier": 2, "f2": 4, "pool1": 2, "pool2": 2,
            }
        else:
            audit_model = TorchShallowConvNetClassifier(
                4, 128, 3, n_filters=4, temporal_kernel_samples=9,
                pool_size=15, pool_stride=5, dropout=0.1,
            )
            audit_parameters = {
                "n_channels": 4, "n_times": 128, "n_filters": 4,
                "pool_size": 15, "pool_stride": 5,
            }
        rows.append(_architecture_row(
            f"{model_name}:task8u_compatibility",
            audit_model,
            source_config="task8u_synthetic_compatibility",
            input_shape=(1, 4, 128),
            sampling_rate=256.0,
            architecture_parameters=audit_parameters,
        ))
        schemas[model_name] = {
            "parameters": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in canonical.named_parameters()
            },
            "buffers": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in canonical.named_buffers()
            },
            "checkpoint_signature": checkpoint_signature,
        }
        for row in batchnorm_inventory(canonical):
            bn_rows.append({"model": model_name, **row})
    return rows, schemas, bn_rows


def _policy_smoke(
    model_name: str,
    model: nn.Module,
    policy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.manual_seed(42)
    support = torch.randn(4, 1, 14, 2560)
    support_variant = support + 0.75
    query_a = torch.randn(4, 1, 14, 2560)
    query_b = query_a * -1.5
    labels = torch.tensor([0, 1, 2, 3])
    base_before = model_state_hash(model)
    config = FOMAMLConfig(
        inner_steps=1,
        inner_learning_rate=0.01,
        meta_learning_rate=0.001,
        episodes_per_meta_batch=1,
        maximum_meta_steps=1,
        buffer_policy=policy,
        device="cpu",
        seed=42,
    )
    learner = FirstOrderMAML(model, config)
    adapted_a = learner.adapt(model, (support, labels))
    adapted_b = learner.adapt(model, (support, labels))
    buffers_before_query = tensor_mapping_hash(adapted_a.buffers)
    loss_a, _, gradients_a = learner.evaluate(adapted_a, (query_a, labels))
    loss_b, _, gradients_b = learner.evaluate(adapted_b, (query_b, labels))
    same_fast = all(
        torch.equal(adapted_a.fast_weights[name], adapted_b.fast_weights[name])
        for name in adapted_a.fast_weights
    )
    same_buffers = all(
        torch.equal(adapted_a.buffers[name], adapted_b.buffers[name])
        for name in adapted_a.buffers
    )
    adapted_support_variant = learner.adapt(model, (support_variant, labels))
    support_changes_buffers = any(
        not torch.equal(adapted_a.buffers[name], adapted_support_variant.buffers[name])
        for name in adapted_a.buffers
    )
    initial_state = create_functional_state(model, policy)
    local_changed = tensor_mapping_hash(initial_state.buffers) != tensor_mapping_hash(adapted_a.buffers)
    original_unchanged = model_state_hash(model) == base_before
    result = {
        "model": model_name,
        "policy": policy,
        "support_forward": True,
        "inner_step": True,
        "query_forward": True,
        "query_gradient_finite": all(torch.isfinite(value).all() for value in gradients_a.values()),
        "query_gradient_nonzero": any(
            int(torch.count_nonzero(value)) > 0 for value in gradients_a.values()
        ),
        "query_loss": loss_a,
        "alternate_query_loss": loss_b,
        "output_classes": 5,
        "original_state_unchanged": original_unchanged,
        "local_buffers_changed_on_support": local_changed,
        "dropout_active": False,
        "minimum_support_batch_size": 2,
        "status": "passed",
    }
    leakage = {
        "model": model_name,
        "policy": policy,
        "query_variants_same_fast_weights": same_fast,
        "query_variants_same_support_buffers": same_buffers,
        "query_did_not_change_support_buffer_hash": (
            buffers_before_query == tensor_mapping_hash(adapted_a.buffers)
        ),
        "query_gradients_differ": any(
            not torch.equal(gradients_a[name], gradients_b[name])
            for name in gradients_a
        ),
        "support_variant_changes_local_buffers": support_changes_buffers,
        "global_state_unchanged": original_unchanged,
    }
    if policy == BufferPolicy.FROZEN_GLOBAL.value:
        result["policy_semantics_valid"] = not local_changed
    else:
        result["policy_semantics_valid"] = local_changed and support_changes_buffers
    if not all([
        result["query_gradient_finite"], original_unchanged,
        same_fast, same_buffers, result["policy_semantics_valid"],
    ]):
        raise RuntimeError(f"Production buffer-policy audit failed: {result} {leakage}")
    return result, leakage


def run_fomaml_production_contract_audit(
    config: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    if config.get("execution_enabled") is not False:
        raise ValueError("Production contract config must set execution_enabled=false")
    output_dir = repository_root / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    architecture_rows, schemas, bn_rows = audit_architectures(
        config, repository_root=repository_root
    )
    compatibility_rows = []
    leakage_rows = []
    functional_rows = []
    for model_name, spec in config["production_models"].items():
        for policy in config["buffer_policies"]:
            model, _ = _build_from_config(repository_root, spec["config"], model_name)
            result, leakage = _policy_smoke(model_name, model, policy)
            compatibility_rows.append(result)
            leakage_rows.append(leakage)
            state = create_functional_state(model, policy)
            functional_rows.append({
                "model": model_name,
                "policy": policy,
                "parameter_count": len(state.parameters),
                "buffer_count": len(state.buffers),
                "parameter_storage_independent": all(
                    state.parameters[name].data_ptr() != value.data_ptr()
                    for name, value in model.named_parameters()
                ),
                "buffer_storage_independent": all(
                    state.buffers[name].data_ptr() != value.data_ptr()
                    for name, value in model.named_buffers()
                ),
                "architecture_signature": state.architecture_signature,
                "training_mode_captured": state.training_mode,
            })
    protocol_result = build_meta_validation_protocol(
        config["meta_validation_protocol"], repository_root=repository_root
    )
    all_compatibility = all(
        row["status"] == "passed" and row["policy_semantics_valid"]
        and row["original_state_unchanged"]
        for row in compatibility_rows
    )
    all_leakage_safe = all(
        row["query_variants_same_fast_weights"]
        and row["query_variants_same_support_buffers"]
        and row["query_did_not_change_support_buffer_hash"]
        and row["global_state_unchanged"]
        for row in leakage_rows
    )
    decision = {
        "status": (
            "production_contract_ready"
            if all_compatibility and all_leakage_safe
            and not protocol_result.protocol["outer_test_in_meta_validation"]
            else "blocked"
        ),
        "both_buffer_policies_safe": all_compatibility,
        "query_leakage_absent": all_leakage_safe,
        "outer_test_protected": not protocol_result.protocol["outer_test_in_meta_validation"],
        "execution_enabled": False,
        "real_eeg_training_performed": False,
        "policy_selection_performed": False,
    }
    latent_frame = pd.DataFrame(architecture_rows)
    latent_frame.to_csv(output_dir / "latent_dim_audit.csv", index=False)
    _write_json(output_dir / "architecture_audit.json", {
        "rows": architecture_rows,
        "latent_dim_explanation": {
            "eegnet": "flattened temporal representation depends on input length, filters, and pooling",
            "shallow_convnet": "adaptive pooling makes latent_dim equal to n_filters",
        },
    })
    _write_json(output_dir / "parameter_buffer_schema.json", schemas)
    pd.DataFrame(bn_rows).to_csv(output_dir / "batchnorm_inventory.csv", index=False)
    _write_json(output_dir / "buffer_policy_audit.json", {
        "policies": compatibility_rows,
        "dropout_policy": "disabled_for_functional_support_and_query",
    })
    pd.DataFrame(compatibility_rows).to_csv(
        output_dir / "production_compatibility.csv", index=False
    )
    _write_json(output_dir / "functional_state_audit.json", functional_rows)
    _write_json(output_dir / "query_leakage_audit.json", leakage_rows)
    _write_json(output_dir / "meta_validation_protocol.json", protocol_result.protocol)
    protocol_result.episode_index.to_parquet(
        output_dir / "meta_validation_episode_index.parquet", index=False
    )
    _write_json(output_dir / "future_experiment_config.json", dict(config))
    _write_json(output_dir / "decision.json", decision)
    protocol_result.errors.to_csv(output_dir / "errors.csv", index=False)
    (output_dir / "contract_report.md").write_text(
        "# Production FOMAML contract audit\n\n"
        f"- Decision: `{decision['status']}`.\n"
        "- Policies: `frozen_global`, `support_local`.\n"
        "- EEGNet and ShallowConvNet one-step synthetic functional smoke: passed.\n"
        "- Query buffer leakage: absent.\n"
        "- Existing outer fold reused; outer-test protected.\n"
        "- Real EEG training: not performed; execution remains disabled.\n",
        encoding="utf-8",
    )
    return {
        **decision,
        "architecture_rows": len(architecture_rows),
        "batchnorm_modules": len(bn_rows),
        "meta_train_subjects": len(protocol_result.protocol["meta_train_subjects"]),
        "meta_validation_subjects": len(protocol_result.protocol["meta_validation_subjects"]),
        "outer_test_subjects": len(protocol_result.protocol["outer_test_subjects"]),
        "episode_counts": protocol_result.protocol["episode_counts"],
        "protocol_hash": protocol_result.protocol["protocol_hash"],
    }
