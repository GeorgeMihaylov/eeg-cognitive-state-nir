from io import StringIO

from rich.console import Console

from apps.streaming_worker.dashboard import StreamingDashboardSink
from apps.streaming_worker.demo import iter_demo_outputs
from apps.streaming_worker.quality import QualityReport
from apps.streaming_worker.runtime import StreamingOutput
from cogstate.protocol import PM_METRICS
from cogstate.streaming.inference import PredictionResult


def test_dashboard_renders_all_pm_metrics():
    stream = StringIO()
    dashboard = StreamingDashboardSink(
        source_name="replay",
        sample_rate=128,
        channels=4,
        window_seconds=2,
        console=Console(file=stream, force_terminal=False, width=120),
    )
    target_probabilities = {
        metric: {"low": 0.1, "medium": 0.2, "high": 0.7}
        for metric in PM_METRICS
    }
    output = StreamingOutput(
        window_start=0.0,
        window_end=2.0,
        quality=QualityReport("good", True, (), 256, 256, 1.0, 128.0, 0.0),
        prediction=PredictionResult(
            "high", target_probabilities["attention"], "v1", False, 2.5,
            target_labels={metric: "high" for metric in PM_METRICS},
            target_probabilities=target_probabilities,
        ),
        stage_latencies_ms={"inference": 2.5},
        model_version="v1",
        model_type="torch_shallow_convnet_multitask",
        input_mode="raw_eeg",
        class_names=("low", "medium", "high"),
        diagnostic_model=False,
    )

    dashboard.latest = output
    dashboard.history.append(dict(output.prediction.target_labels))
    dashboard.console.print(dashboard.render())
    rendered = stream.getvalue()

    assert "Seven PM predictions" in rendered
    assert all(metric in rendered for metric in PM_METRICS)
    assert "70.0%" in rendered


def test_demo_outputs_are_reproducible_and_cover_all_pm_metrics():
    first = next(iter_demo_outputs(seed=7, sample_rate=128, window_seconds=2))
    repeated = next(iter_demo_outputs(seed=7, sample_rate=128, window_seconds=2))

    assert first.as_dict() == repeated.as_dict()
    assert first.model_version == "demo-synthetic-v1"
    assert set(first.prediction.target_labels) == set(PM_METRICS)
    assert set(first.prediction.target_probabilities) == set(PM_METRICS)
    assert all(
        abs(sum(probabilities.values()) - 1.0) < 1e-12
        for probabilities in first.prediction.target_probabilities.values()
    )
