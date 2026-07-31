from __future__ import annotations

import torch

from bench.meta import clone_model_for_episode, validate_model_clone
from model_zoo.DL.eegnet import TorchEEGNetClassifier
from model_zoo.DL.shallow_convnet import TorchShallowConvNetClassifier


def _models():
    return [
        TorchEEGNetClassifier(
            4, 128, 5,
            temporal_kernel_samples=16,
            separable_kernel_samples=8,
            f1=2,
            depth_multiplier=2,
            f2=4,
            pool1=2,
            pool2=2,
            dropout=0.1,
        ),
        TorchShallowConvNetClassifier(
            4, 128, 5,
            n_filters=4,
            temporal_kernel_samples=9,
            pool_size=15,
            pool_stride=5,
            dropout=0.1,
        ),
    ]


def test_production_eeg_models_clone_independently_on_cpu() -> None:
    example = torch.randn(2, 1, 4, 128)
    for original in _models():
        original_state = {
            name: tensor.detach().clone()
            for name, tensor in original.state_dict().items()
        }
        clone = clone_model_for_episode(
            original, device="cpu", example_input=example
        )
        audit = validate_model_clone(
            original, clone, example_input=example
        )
        assert audit.valid
        assert audit.output_shape_matches
        assert clone(example).shape == (2, 5)
        with torch.no_grad():
            next(clone.parameters()).add_(1.0)
        for name, tensor in original.state_dict().items():
            assert torch.equal(tensor, original_state[name])
