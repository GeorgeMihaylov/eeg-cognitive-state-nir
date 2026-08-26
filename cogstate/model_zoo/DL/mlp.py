"""Compatibility facade for the canonical feature MLP."""

from model_zoo.DL.mlp import TorchMLP, build_torch_mlp

__all__ = ["TorchMLP", "build_torch_mlp"]
