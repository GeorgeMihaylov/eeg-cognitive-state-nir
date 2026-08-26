"""Compatibility facade for the canonical EEGNet implementation."""

from model_zoo.DL.eegnet import TorchEEGNetClassifier, build_torch_eegnet

__all__ = ["TorchEEGNetClassifier", "build_torch_eegnet"]
