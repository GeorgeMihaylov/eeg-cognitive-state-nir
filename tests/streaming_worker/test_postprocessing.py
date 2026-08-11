from apps.streaming_worker.config import PostprocessingConfig
from apps.streaming_worker.postprocessing import PredictionFilter
from cogstate.streaming.inference import PredictionResult


def prediction(low, medium, high):
    probabilities = {"low": low, "medium": medium, "high": high}
    return PredictionResult(
        label=max(probabilities, key=probabilities.get),
        probabilities=probabilities,
        model_version="test",
        is_calibrated=False,
        inference_time_ms=1.0,
        target_labels={"attention": max(probabilities, key=probabilities.get)},
        target_probabilities={"attention": probabilities},
    )


def test_class_switch_requires_confirmation():
    filter_ = PredictionFilter(
        PostprocessingConfig(
            probability_ema_alpha=1.0,
            minimum_confidence=0.5,
            confirmation_windows=2,
        )
    )

    first = filter_.apply(prediction(0.1, 0.1, 0.8))
    second = filter_.apply(prediction(0.1, 0.1, 0.8))

    assert first.target_labels["attention"] == "unknown"
    assert second.target_labels["attention"] == "high"


def test_low_confidence_becomes_unknown():
    filter_ = PredictionFilter(
        PostprocessingConfig(
            probability_ema_alpha=1.0,
            minimum_confidence=0.6,
            confirmation_windows=1,
        )
    )

    result = filter_.apply(prediction(0.34, 0.33, 0.33))

    assert result.target_labels["attention"] == "unknown"
