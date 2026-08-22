"""User-facing command line interface for the streaming pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cogstate.features import FEATURE_SCHEMA_VERSION
from cogstate.protocol import PM_METRICS

from .config import WorkerConfig
from .runtime import StreamingRuntime


CLI_VERSION = "1.0.0"
COMMANDS = {
    "run",
    "api",
    "validate",
    "show-config",
    "inspect-model",
    "list-models",
}


def _parser() -> argparse.ArgumentParser:
    examples = """examples:
  python -m apps.streaming_worker run --config configs/streaming.yaml
  python -m apps.streaming_worker validate --config configs/streaming.yaml
  python -m apps.streaming_worker inspect-model artifacts/eegmat_shallow_v1
  python -m apps.streaming_worker api --host 0.0.0.0 --port 8000
"""
    parser = argparse.ArgumentParser(
        prog="cogstate-stream",
        description="Run, inspect and validate the Cogstate EEG streaming pipeline.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {CLI_VERSION}")
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in CLI status output.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    run = subparsers.add_parser("run", help="Run the standalone streaming worker.")
    run.add_argument("--config", default="configs/streaming.yaml")
    run.add_argument("--verbose", action="store_true")
    run.add_argument(
        "--dashboard",
        action="store_true",
        help="Show a live terminal dashboard with all seven PM predictions.",
    )
    run.add_argument("--refresh-rate", type=float, default=5.0)
    run.add_argument("--history", type=int, default=60)
    run.add_argument(
        "--demo",
        action="store_true",
        help="Open the dashboard with synthetic PM predictions; no EEG or model required.",
    )
    run.add_argument(
        "--demo-speed",
        type=float,
        default=1.0,
        help="Synthetic update speed multiplier (default: 1.0).",
    )
    run.add_argument("--demo-seed", type=int, default=42)

    api = subparsers.add_parser("api", help="Run the FastAPI control plane and worker.")
    api.add_argument("--config", default="configs/streaming.yaml")
    api.add_argument("--host")
    api.add_argument("--port", type=int)
    api.add_argument("--reload", action="store_true")
    api.add_argument("--verbose", action="store_true")

    validate = subparsers.add_parser(
        "validate", help="Validate configuration, source and model bundle contracts."
    )
    validate.add_argument("--config", default="configs/streaming.yaml")
    validate.add_argument(
        "--skip-source",
        action="store_true",
        help="Do not require the configured replay file to exist.",
    )

    show_config = subparsers.add_parser(
        "show-config", help="Render the resolved streaming configuration."
    )
    show_config.add_argument("--config", default="configs/streaming.yaml")

    inspect_model = subparsers.add_parser(
        "inspect-model", help="Inspect one model bundle manifest."
    )
    inspect_model.add_argument("artifact_dir")

    list_models = subparsers.add_parser(
        "list-models", help="List model bundles below an artifact directory."
    )
    list_models.add_argument("--root", default="artifacts")
    return parser


def _normalize_legacy_args(argv: Sequence[str]) -> list[str]:
    """Keep ``python -m apps.streaming_worker --config ...`` working."""
    values = list(argv)
    global_flags: list[str] = []
    while values and values[0] == "--no-color":
        global_flags.append(values.pop(0))
    if values and values[0] not in COMMANDS and values[0] not in {"-h", "--help", "--version"}:
        values.insert(0, "run")
    return [*global_flags, *values]


def _console(no_color: bool = False) -> Console:
    # Status belongs on stderr so JSON prediction output on stdout remains clean.
    if os.name == "nt" and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    return Console(stderr=True, no_color=no_color, legacy_windows=False)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _flatten(prefix: str, value: Any) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(path, nested))
    elif isinstance(value, (list, tuple)):
        rows.append((prefix, ", ".join(map(str, value))))
    else:
        rows.append((prefix, "null" if value is None else str(value)))
    return rows


def _render_config(console: Console, config: WorkerConfig, path: str | Path) -> None:
    table = Table(title=f"Resolved configuration · {path}", show_lines=False)
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    for key, value in _flatten("", asdict(config)):
        table.add_row(key, value)
    console.print(table)


def _manifest_path(artifact_dir: str | Path) -> Path:
    path = Path(artifact_dir)
    return path if path.name == "manifest.json" else path / "manifest.json"


def _read_manifest(artifact_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    path = _manifest_path(artifact_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Model manifest must be a JSON object")
    return path, payload


def _model_rows(path: Path, manifest: dict[str, Any]) -> list[tuple[str, str]]:
    artifact_dir = path.parent
    model_file = manifest.get("model_file")
    bootstrap = bool(manifest.get("bootstrap", False))
    weights_state = "diagnostic bootstrap" if bootstrap else "not declared"
    if model_file and not bootstrap:
        weights_state = "present" if (artifact_dir / str(model_file)).is_file() else "missing"
    channels = manifest.get("channels", [])
    return [
        ("Artifact", str(artifact_dir)),
        ("Version", str(manifest.get("version", "unknown"))),
        ("Model type", str(manifest.get("model_type", "unknown"))),
        ("Input mode", str(manifest.get("input_mode", "features"))),
        ("Diagnostic", str(bool(manifest.get("diagnostic_only", False)))),
        ("Sample rate", str(manifest.get("sample_rate", "dynamic"))),
        ("Window", str(manifest.get("window_seconds", "dynamic"))),
        ("Channels", str(len(channels)) if channels else "dynamic"),
        ("Classes", ", ".join(map(str, manifest.get("class_names", []))) or "PM targets"),
        ("PM targets", ", ".join(map(str, manifest.get("target_names", []))) or "from model"),
        ("Weights", weights_state),
    ]


def _render_model(console: Console, path: Path, manifest: dict[str, Any]) -> None:
    table = Table(title="Model bundle", show_header=False)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")
    for key, value in _model_rows(path, manifest):
        style = "red" if key == "Weights" and value == "missing" else None
        table.add_row(key, Text(value, style=style))
    console.print(table)


def _preprocessing_contract(config: WorkerConfig) -> dict[str, Any]:
    bundle_version = None
    if config.preprocessing.mne_faster_enabled:
        from cogstate.preprocessing.mne_faster import MNEFasterBundle

        assert config.preprocessing.mne_faster_bundle_dir is not None
        bundle_version = MNEFasterBundle.load(
            config.preprocessing.mne_faster_bundle_dir
        ).version
    return {
        "bandpass_low_hz": config.preprocessing.bandpass_low_hz,
        "bandpass_high_hz": config.preprocessing.bandpass_high_hz,
        "notch_hz": config.preprocessing.notch_hz,
        "filter_mode": "causal",
        "artifact_removal": (
            "mne_faster" if config.preprocessing.mne_faster_enabled else "none"
        ),
        "artifact_bundle_version": bundle_version,
    }


def _validate_bundle(config: WorkerConfig, manifest_path: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    mode = manifest.get("input_mode", "features")
    if mode not in {"raw_eeg", "features"}:
        errors.append(f"unsupported input_mode {mode!r}")
        return errors
    if "sample_rate" in manifest and float(manifest["sample_rate"]) != config.signal.sample_rate:
        errors.append("manifest sample_rate differs from signal.sample_rate")
    channels = tuple(manifest.get("channels", ()))
    if channels and channels != config.signal.channels:
        errors.append("manifest channel order differs from signal.channels")
    if "window_seconds" in manifest and float(manifest["window_seconds"]) != config.windowing.window_seconds:
        errors.append("manifest window_seconds differs from windowing.window_seconds")

    if mode == "raw_eeg":
        expected_times = int(round(config.signal.sample_rate * config.windowing.window_seconds))
        if int(manifest.get("n_times", -1)) != expected_times:
            errors.append(f"manifest n_times differs from expected {expected_times}")
        if manifest.get("model_type") != "torch_shallow_convnet_multitask":
            errors.append("raw_eeg bundle must use torch_shallow_convnet_multitask")
        if tuple(manifest.get("target_names", ())) != tuple(PM_METRICS):
            errors.append("raw_eeg bundle must declare all seven PM target_names")
        declared = manifest.get("preprocessing", {})
        for key, value in _preprocessing_contract(config).items():
            if declared.get(key) != value:
                errors.append(f"manifest preprocessing.{key} differs from config")
    else:
        if manifest.get("feature_profile") not in {None, config.features.profile}:
            errors.append("manifest feature_profile differs from features.profile")
        schema = manifest.get("feature_schema_version")
        if not manifest.get("bootstrap", False) and schema != FEATURE_SCHEMA_VERSION:
            errors.append(
                f"feature_schema_version {schema!r} differs from {FEATURE_SCHEMA_VERSION!r}"
            )

    if not manifest.get("bootstrap", False):
        for key in ("model_file", "scaler_file", "selector_file"):
            filename = manifest.get(key)
            if filename and not (manifest_path.parent / str(filename)).is_file():
                errors.append(f"missing {key}: {filename}")
    return errors


def _validation_checks(config_path: str | Path, *, skip_source: bool) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    path = Path(config_path)
    config = WorkerConfig.from_yaml(path)
    checks.append(("Configuration", True, str(path)))

    if config.source.type == "replay":
        source_path = Path(config.source.path or "")
        source_ok = skip_source or source_path.is_file()
        detail = "skipped" if skip_source else str(source_path)
        checks.append(("Replay source", source_ok, detail))
    else:
        checks.append(("LSL source", True, f"runtime discovery: {config.source.stream_name}"))

    manifest_path = _manifest_path(config.model.artifact_dir)
    if not manifest_path.is_file():
        checks.append(("Model manifest", False, str(manifest_path)))
        return checks
    checks.append(("Model manifest", True, str(manifest_path)))
    _, manifest = _read_manifest(manifest_path)
    bundle_errors = _validate_bundle(config, manifest_path, manifest)
    checks.append(
        (
            "Bundle contract",
            not bundle_errors,
            "; ".join(bundle_errors) if bundle_errors else "compatible",
        )
    )
    return checks


def _render_checks(console: Console, checks: list[tuple[str, bool, str]]) -> bool:
    table = Table(title="Streaming pipeline validation")
    table.add_column("Status", justify="center", width=8)
    table.add_column("Check", style="cyan")
    table.add_column("Details")
    valid = True
    for name, passed, detail in checks:
        valid &= passed
        marker = Text("PASS" if passed else "FAIL", style="bold green" if passed else "bold red")
        table.add_row(marker, name, detail)
    console.print(table)
    console.print(
        Panel.fit(
            "Configuration is ready." if valid else "Fix failed checks before starting.",
            border_style="green" if valid else "red",
        )
    )
    return valid


def _run_worker(args: argparse.Namespace, console: Console) -> int:
    _configure_logging(args.verbose)
    config = WorkerConfig.from_yaml(args.config)
    use_dashboard = args.dashboard or args.demo
    if not use_dashboard:
        console.print(
            Panel.fit(
                f"[bold]Cogstate streaming worker[/bold]\n"
                f"source: [cyan]{config.source.type}[/cyan] · "
                f"model: [cyan]{config.model.artifact_dir}[/cyan]",
                border_style="blue",
            )
        )
    dashboard = None
    sink = None
    if use_dashboard:
        from .dashboard import StreamingDashboardSink
        from .sinks import CompositeSink, JsonlSink

        dashboard = StreamingDashboardSink(
            source_name=config.source.type,
            sample_rate=config.signal.sample_rate,
            channels=len(config.signal.channels),
            window_seconds=config.windowing.window_seconds,
            refresh_rate=args.refresh_rate,
            history_size=args.history,
            console=console,
        )
        sinks: list[object] = [dashboard]
        if config.output.jsonl_path and not args.demo:
            sinks.append(JsonlSink(config.output.jsonl_path))
        sink = CompositeSink(sinks)
    if args.demo:
        from .demo import run_dashboard_demo

        assert dashboard is not None
        dashboard.start()
        try:
            run_dashboard_demo(
                dashboard,
                seed=args.demo_seed,
                speed=args.demo_speed,
                sample_rate=config.signal.sample_rate,
                window_seconds=config.windowing.window_seconds,
                step_seconds=config.windowing.step_seconds,
            )
        except KeyboardInterrupt:
            pass
        finally:
            dashboard.close()
        return 0
    runtime = StreamingRuntime(config, sink=sink)
    try:
        if dashboard is not None:
            dashboard.start()
        runtime.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping worker…[/yellow]")
        runtime.stop()
    if not use_dashboard:
        console.print(
            f"[green]Finished[/green] · processed={runtime.processed_windows} "
            f"rejected={runtime.rejected_windows} samples_rejected={runtime.rejected_samples}"
        )
    return 0


def _run_api(args: argparse.Namespace, console: Console) -> int:
    import uvicorn

    from .api.app import create_app

    _configure_logging(args.verbose)
    config = WorkerConfig.from_yaml(args.config)
    host = args.host or config.api.host
    port = args.port or config.api.port
    console.print(
        Panel.fit(
            f"[bold]Cogstate streaming API[/bold]\n"
            f"http://{host}:{port} · docs: http://{host}:{port}/docs",
            border_style="magenta",
        )
    )
    if args.reload:
        os.environ["COGSTATE_STREAMING_CONFIG"] = args.config
        uvicorn.run(
            "apps.streaming_worker.api.app:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
        )
    else:
        uvicorn.run(create_app(config=config), host=host, port=port)
    return 0


def _list_models(args: argparse.Namespace, console: Console) -> int:
    root = Path(args.root)
    manifests = sorted(root.glob("**/manifest.json")) if root.is_dir() else []
    table = Table(title=f"Model bundles · {root}")
    table.add_column("Bundle", style="cyan")
    table.add_column("Mode")
    table.add_column("Model")
    table.add_column("Input")
    table.add_column("Weights")
    for path in manifests:
        try:
            _, manifest = _read_manifest(path)
            channels = manifest.get("channels", [])
            rate = manifest.get("sample_rate")
            input_summary = f"{len(channels)} ch · {rate:g} Hz" if channels and rate else "dynamic"
            model_file = manifest.get("model_file")
            if manifest.get("bootstrap", False):
                weights = "diagnostic"
            elif model_file and (path.parent / str(model_file)).is_file():
                weights = "ready"
            else:
                weights = "missing"
            table.add_row(
                str(path.parent.relative_to(root)),
                str(manifest.get("input_mode", "features")),
                str(manifest.get("model_type", "unknown")),
                input_summary,
                Text(weights, style="green" if weights == "ready" else "yellow"),
            )
        except Exception as exc:
            table.add_row(str(path.parent), "invalid", type(exc).__name__, "—", Text("error", style="red"))
    console.print(table)
    if not manifests:
        console.print(f"[yellow]No manifests found below {root}[/yellow]")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    values = _normalize_legacy_args(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(values)
    console = _console(getattr(args, "no_color", False))
    if args.command is None:
        console.print(
            Panel.fit(
                "[bold cyan]Cogstate Streaming[/bold cyan]\n"
                "Run EEG inference, validate bundles and inspect device models.",
                border_style="cyan",
            )
        )
        parser.print_help()
        return 0

    try:
        if args.command == "run":
            return _run_worker(args, console)
        if args.command == "api":
            return _run_api(args, console)
        if args.command == "validate":
            checks = _validation_checks(args.config, skip_source=args.skip_source)
            return 0 if _render_checks(console, checks) else 1
        if args.command == "show-config":
            _render_config(console, WorkerConfig.from_yaml(args.config), args.config)
            return 0
        if args.command == "inspect-model":
            path, manifest = _read_manifest(args.artifact_dir)
            _render_model(console, path, manifest)
            return 0
        if args.command == "list-models":
            return _list_models(args, console)
        parser.error(f"Unknown command: {args.command}")
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError, yaml.YAMLError) as exc:
        console.print(
            Panel.fit(
                f"[bold red]{type(exc).__name__}[/bold red]\n{exc}",
                title="Command failed",
                border_style="red",
            )
        )
        return 2
    return 0


__all__ = ["main"]
