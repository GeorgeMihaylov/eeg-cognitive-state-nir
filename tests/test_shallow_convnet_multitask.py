import numpy as np

from cogstate.model_zoo.factory import build_model
from cogstate.protocol import PM_METRICS


def test_multitask_shallow_convnet_has_shared_encoder_and_seven_heads(tmp_path):
    model = build_model(
        "torch_shallow_convnet_multitask",
        "classification",
        (1, 4, 128),
        3,
        {
            "sampling_rate": 128,
            "channel_names": ["C1", "C2", "C3", "C4"],
            "pool_size": 32,
            "standardize": False,
            "device": "cpu",
        },
    )
    model.is_fitted_ = True
    probabilities = model.predict_proba(
        np.zeros((2, 1, 4, 128), dtype=np.float32)
    )

    assert tuple(model.model.heads) == PM_METRICS
    assert set(probabilities) == set(PM_METRICS)
    assert all(values.shape == (2, 3) for values in probabilities.values())
    assert all(np.allclose(values.sum(axis=1), 1.0) for values in probabilities.values())

    path = tmp_path / "model.pt"
    model.save(path)
    assert path.is_file()
