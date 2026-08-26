"""Compatibility facade for canonical LSTM/BiLSTM models."""

from model_zoo.DL.lstm import TorchLSTMClassifier, build_torch_lstm

__all__ = ["TorchLSTMClassifier", "build_torch_lstm"]
