"""End-to-end reporting over immutable completed benchmark artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from bench.validation.metrics import MetricsCalculator

from .alignment import AlignmentResult, check_alignment
from .error_analysis import calculate_error_analysis, summarize_by_source
from .paired_statistics import (
    apply_holm_by_family,
    bootstrap_spearman,
    paired_subject_statistics,
    subject_bootstrap_interval,
)
from .run_inventory import (
    InventoryEntry,
    REPO_ROOT,
    build_run_inventory,
    canonical_entries,
)
from .subject_metrics import calculate_subject_metrics, evaluation_predictions
from .svg import (
    write_bar,
    write_boxplot,
    write_heatmap,
    write_lines,
    write_placeholder,
    write_scatter,
)


PRIMARY_METRICS = ("balanced_accuracy", "macro_f1", "auc")
TRIAL_FACTORS = {
    "A": (0, 0, 0),
    "B": (1, 0, 0),
    "C": (0, 1, 0),
    "D": (0, 0, 1),
    "E": (1, 1, 0),
    "F": (1, 0, 1),
    "G": (0, 1, 1),
    "H": (1, 1, 1),
}


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    def sanitize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): sanitize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [sanitize(child) for child in item]
        if isinstance(item, np.ndarray):
            return sanitize(item.tolist())
        if isinstance(item, np.generic):
            return sanitize(item.item())
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, Path):
            return str(item)
        return item

    path.write_text(
        json.dumps(
            sanitize(value),
            indent=2,
            ensure_ascii=False,
            default=_json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _table(frame: pd.DataFrame, *, index: bool = False) -> str:
    if frame.empty:
        return "_No applicable rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.6g}"
        )
    return display.to_markdown(index=index)


def _normalise_fold(value: Any) -> str:
    text = str(value)
    if text.startswith("fold_"):
        return text
    try:
        return f"fold_{int(float(text)):02d}"
    except ValueError:
        return text


def load_statistical_analysis_spec(path: str | Path) -> dict[str, Any]:
    spec_path = _repo_path(path)
    with open(spec_path, encoding="utf-8") as input_file:
        document = yaml.safe_load(input_file) or {}
    if not isinstance(document, dict) or "analysis" not in document:
        raise ValueError("Statistical analysis spec requires an analysis mapping")
    if not document.get("run_rules"):
        raise ValueError("Statistical analysis spec requires run_rules")
    return document


def _load_predictions(entry: InventoryEntry) -> pd.DataFrame:
    frame = pd.read_parquet(entry.prediction_file)
    if entry.fold_filter:
        column = "fold" if "fold" in frame else "outer_fold"
        wanted = {_normalise_fold(value) for value in entry.fold_filter}
        frame = frame.loc[frame[column].map(_normalise_fold).isin(wanted)].copy()
    return frame


def _find_group_result(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        if "folds" in value and "n_folds" in value:
            return value
        for child in value.values():
            result = _find_group_result(child)
            if result is not None:
                return result
    return None


def _artifact_training_summary(entry: InventoryEntry) -> dict[str, Any]:
    if not entry.metrics_file or not Path(entry.metrics_file).suffix == ".json":
        return {}
    try:
        document = json.loads(Path(entry.metrics_file).read_text(encoding="utf-8"))
        group = _find_group_result(document)
    except (OSError, json.JSONDecodeError):
        return {}
    if group is None:
        return {}
    aggregated = group.get("aggregated", {})
    folds = group.get("folds", {})
    first = next(iter(folds.values()), {})
    training = first.get("training", {})
    return {
        "training_time_seconds": aggregated.get("training_time_total"),
        "parameter_count": training.get("trainable_parameter_count"),
        "epochs_mean": aggregated.get("epochs_trained_mean"),
        "best_validation_loss_mean": aggregated.get("best_validation_loss_mean"),
    }


def _fold_metrics(entry: InventoryEntry, predictions: pd.DataFrame) -> pd.DataFrame:
    fold_column = "fold" if "fold" in predictions else "outer_fold"
    probability_columns = sorted(
        (column for column in predictions if column.startswith("proba_")),
        key=lambda column: int(column.split("_", 1)[1]),
    )
    rows: list[dict[str, Any]] = []
    for fold, group in predictions.groupby(fold_column, sort=True):
        proba = group[probability_columns].to_numpy(dtype=float) if probability_columns else None
        metrics = MetricsCalculator.calculate_all_metrics(
            group["y_true"].to_numpy(dtype=int),
            group["y_pred"].to_numpy(dtype=int),
            proba,
        )
        rows.append({
            "track": entry.analysis_track,
            "model": entry.model,
            "seed": entry.seed,
            "outer_fold": _normalise_fold(fold),
            "n_samples": len(group),
            **{
                metric: float(metrics.get(metric, np.nan))
                for metric in (
                    "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                    "kappa", "auc", "ordinal_mae", "adjacent_accuracy",
                )
            },
        })
    return pd.DataFrame(rows)


def _entry_key(track: str, model: str, seed: int) -> tuple[str, str, int]:
    return track, model, int(seed)


def _calibration_frames(
    entries: Sequence[InventoryEntry],
) -> tuple[dict[tuple[str, float], pd.DataFrame], pd.DataFrame]:
    frames: dict[tuple[str, float], pd.DataFrame] = {}
    metadata: list[pd.DataFrame] = []
    for entry in entries:
        if entry.analysis_track != "calibration":
            continue
        predictions = evaluation_predictions(_load_predictions(entry))
        for (method, budget), group in predictions.groupby(
            ["calibration_method", "budget_seconds"], sort=True
        ):
            frames[(str(method), float(budget))] = group.copy()
        if entry.metrics_file and Path(entry.metrics_file).is_file():
            subject_rows = pd.read_csv(entry.metrics_file)
            subject_rows["run_model"] = entry.model
            metadata.append(subject_rows)
    return frames, pd.concat(metadata, ignore_index=True) if metadata else pd.DataFrame()


def _calibration_comparison_specs() -> list[dict[str, Any]]:
    pairs = [
        ("head_only", "zero_shot"),
        ("full_model", "zero_shot"),
        ("subject_normalization", "zero_shot"),
        ("head_only", "full_model"),
    ]
    return [
        {
            "name": f"{left}_minus_{right}_budget_{budget}",
            "family": "calibration",
            "track": "calibration",
            "left_model": left,
            "right_model": right,
            "seed": 42,
            "budget_seconds": float(budget),
            "inferential": True,
        }
        for budget in (180, 300, 600)
        for left, right in pairs
    ]


def _subject_rows_for_entry(
    entry: InventoryEntry, predictions: pd.DataFrame
) -> pd.DataFrame:
    return calculate_subject_metrics(
        predictions,
        track=entry.analysis_track,
        model=entry.model,
        seed=entry.seed,
    )


def _subject_model_summary(
    subject_metrics: pd.DataFrame,
    *,
    n_resamples: int,
    confidence_level: float,
    random_state: int,
) -> pd.DataFrame:
    group_columns = ["track", "model", "seed"]
    if "budget_seconds" in subject_metrics:
        group_columns.append("budget_seconds")
    rows: list[dict[str, Any]] = []
    for keys, group in subject_metrics.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["subjects"] = int(group["subject_id"].nunique())
        for metric in PRIMARY_METRICS:
            interval = subject_bootstrap_interval(
                group[metric],
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                random_state=random_state,
            )
            row[f"{metric}_mean"] = interval["estimate"]
            row[f"{metric}_ci_low"] = interval["ci_low"]
            row[f"{metric}_ci_high"] = interval["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_metric_rows(
    comparison: Mapping[str, Any],
    subject_metrics: pd.DataFrame,
    *,
    n_resamples: int,
    confidence_level: float,
    random_state: int,
) -> list[dict[str, Any]]:
    track = str(comparison["track"])
    seed = int(comparison.get("seed", 42))
    left_model = str(comparison["left_model"])
    right_model = str(comparison["right_model"])
    subset = subject_metrics.loc[
        (subject_metrics["track"] == track) & (subject_metrics["seed"] == seed)
    ]
    budget = comparison.get("budget_seconds")
    if budget is not None and "budget_seconds" in subset:
        subset = subset.loc[subset["budget_seconds"] == float(budget)]
    left = subset.loc[subset["model"] == left_model]
    right = subset.loc[subset["model"] == right_model]
    keys = ["subject_id"]
    merged = left.merge(right, on=keys, suffixes=("_left", "_right"), validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for metric in PRIMARY_METRICS:
        result = paired_subject_statistics(
            merged[f"{metric}_left"],
            merged[f"{metric}_right"],
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_state=random_state,
        )
        row = {
            "comparison": str(comparison["name"]),
            "family": str(comparison["family"]),
            "track": track,
            "left_model": left_model,
            "right_model": right_model,
            "seed": seed,
            "metric": metric,
            "budget_seconds": budget,
            "inferential": bool(comparison.get("inferential", True)),
            **result,
        }
        if not row["inferential"]:
            row["wilcoxon_status"] = "not_run_pilot_case_study"
            row["wilcoxon_p_value"] = np.nan
            row["sign_test_status"] = "not_run_pilot_case_study"
            row["sign_test_p_value"] = np.nan
        rows.append(row)
    return rows


def _trial_letter(model: str) -> str | None:
    marker = "shallowconvnet_trial_"
    return model[len(marker):] if model.startswith(marker) else None


def _preprocessing_summary(fold_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = fold_metrics.loc[fold_metrics["track"] == "preprocessing"].copy()
    subset["trial"] = subset["model"].map(_trial_letter)
    factors = subset["trial"].map(TRIAL_FACTORS)
    subset["bandpass"] = factors.map(lambda value: value[0] if value else np.nan)
    subset["notch"] = factors.map(lambda value: value[1] if value else np.nan)
    subset["car"] = factors.map(lambda value: value[2] if value else np.nan)
    trial_summary = subset.groupby(
        ["trial", "seed", "bandpass", "notch", "car"], as_index=False
    ).agg(
        folds=("outer_fold", "nunique"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        auc_mean=("auc", "mean"),
    )
    seed42 = trial_summary.loc[trial_summary["seed"] == 42]
    effects = []
    for factor in ("bandpass", "notch", "car"):
        on = seed42.loc[seed42[factor] == 1, "balanced_accuracy_mean"].mean()
        off = seed42.loc[seed42[factor] == 0, "balanced_accuracy_mean"].mean()
        effects.append({
            "factor": factor,
            "on_mean": on,
            "off_mean": off,
            "main_effect_on_minus_off": on - off,
        })
    return trial_summary, pd.DataFrame(effects)


def _preprocessing_multiseed_deltas(trial_summary: pd.DataFrame) -> pd.DataFrame:
    baseline = trial_summary.loc[
        trial_summary["trial"] == "A",
        ["seed", "balanced_accuracy_mean", "macro_f1_mean", "auc_mean"],
    ].rename(columns={
        "balanced_accuracy_mean": "baseline_balanced_accuracy",
        "macro_f1_mean": "baseline_macro_f1",
        "auc_mean": "baseline_auc",
    })
    compared = trial_summary.loc[trial_summary["trial"].isin(["B", "E"])].merge(
        baseline,
        on="seed",
        how="inner",
        validate="many_to_one",
    )
    compared["delta_balanced_accuracy_vs_A"] = (
        compared["balanced_accuracy_mean"] - compared["baseline_balanced_accuracy"]
    )
    compared["delta_macro_f1_vs_A"] = (
        compared["macro_f1_mean"] - compared["baseline_macro_f1"]
    )
    compared["delta_auc_vs_A"] = compared["auc_mean"] - compared["baseline_auc"]
    return compared[[
        "trial", "seed", "delta_balanced_accuracy_vs_A",
        "delta_macro_f1_vs_A", "delta_auc_vs_A",
    ]].sort_values(["trial", "seed"])


def _class_entropy(value: Any) -> float:
    try:
        counts = json.loads(value) if isinstance(value, str) else dict(value)
        array = np.asarray(list(counts.values()), dtype=float)
    except (TypeError, ValueError, json.JSONDecodeError):
        return np.nan
    if array.sum() <= 0:
        return np.nan
    probability = array[array > 0] / array.sum()
    return float(-(probability * np.log2(probability)).sum())


def _calibration_effects(
    calibration_metadata: pd.DataFrame,
    *,
    n_resamples: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if calibration_metadata.empty:
        return pd.DataFrame(), pd.DataFrame()
    data = calibration_metadata.copy()
    data = data.loc[
        data["budget_seconds"].isin([180.0, 300.0, 600.0])
        & (data["status"] == "valid")
    ]
    data["calibration_class_entropy"] = data["class_counts"].map(_class_entropy)
    zero = data.loc[data["calibration_method"] == "zero_shot"].copy()
    zero = zero[[
        "subject_id", "budget_seconds", "balanced_accuracy", "macro_f1", "auc"
    ]].rename(columns={
        "balanced_accuracy": "zero_shot_balanced_accuracy",
        "macro_f1": "zero_shot_macro_f1",
        "auc": "zero_shot_auc",
    })
    calibrated = data.loc[data["calibration_method"] != "zero_shot"].copy()
    effects = calibrated.merge(
        zero,
        on=["subject_id", "budget_seconds"],
        how="inner",
        validate="many_to_one",
    )
    for metric in PRIMARY_METRICS:
        effects[f"delta_{metric}"] = (
            effects[metric] - effects[f"zero_shot_{metric}"]
        )
    correlation_rows: list[dict[str, Any]] = []
    correlation_resamples = min(2_000, n_resamples)
    for (method, budget), group in effects.groupby(
        ["calibration_method", "budget_seconds"], sort=True
    ):
        for predictor in (
            "zero_shot_balanced_accuracy",
            "number_of_classes",
            "calibration_class_entropy",
            "majority_class_fraction",
            "evaluation_sequences",
        ):
            result = bootstrap_spearman(
                group[predictor],
                group["delta_balanced_accuracy"],
                n_resamples=correlation_resamples,
                random_state=random_state,
            )
            correlation_rows.append({
                "calibration_method": method,
                "budget_seconds": budget,
                "predictor": predictor,
                "outcome": "delta_balanced_accuracy",
                "bootstrap_samples": correlation_resamples,
                **result,
            })
    return effects, pd.DataFrame(correlation_rows)


def _difficult_subjects(
    subject_metrics: pd.DataFrame,
    calibration_effects: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    primary = subject_metrics.loc[
        subject_metrics["track"].isin(["feature_window", "feature_sequence", "raw_eeg"])
        & (subject_metrics["seed"] == 42)
    ].copy()
    pivot = primary.pivot_table(
        index="subject_id", columns="model", values="balanced_accuracy", aggfunc="first"
    )
    metadata = primary.groupby("subject_id", as_index=True).agg(
        source=("source", lambda values: "+".join(sorted(set(values)))),
        records=("records", "max"),
        sample_count=("n_samples", "max"),
        classes_present=("classes_present", "first"),
        class_distribution=("class_distribution", "first"),
    )
    difficult = metadata.join(
        pd.DataFrame({
            "model_mean_balanced_accuracy": pivot.mean(axis=1),
            "model_disagreement_sd": pivot.std(axis=1),
        })
    )
    difficult = difficult.join(pivot.add_prefix("balanced_accuracy_"))
    consistently = difficult.nsmallest(10, "model_mean_balanced_accuracy").reset_index()
    disagreement = difficult.nlargest(10, "model_disagreement_sd").reset_index()
    harmed = pd.DataFrame()
    improved = pd.DataFrame()
    if not calibration_effects.empty:
        budget600 = calibration_effects.loc[calibration_effects["budget_seconds"] == 600]
        columns = [
            "subject_id", "calibration_method", "source", "evaluation_sequences",
            "number_of_classes", "class_counts", "delta_balanced_accuracy",
            "delta_macro_f1",
        ]
        available = [column for column in columns if column in budget600]
        harmed = budget600.nsmallest(10, "delta_balanced_accuracy")[available]
        improved = budget600.nlargest(10, "delta_balanced_accuracy")[available]
    return {
        "consistently_difficult": consistently,
        "high_model_disagreement": disagreement,
        "harmed_by_calibration": harmed,
        "improved_by_calibration": improved,
    }


def _build_figures(
    figures_dir: Path,
    *,
    subject_metrics: pd.DataFrame,
    predictions: Mapping[tuple[str, str, int], pd.DataFrame],
    preprocessing_trials: pd.DataFrame,
    preprocessing_effects: pd.DataFrame,
    calibration_effects: pd.DataFrame,
    source_summary: pd.DataFrame,
) -> list[Path]:
    paths: list[Path] = []

    path = figures_dir / "model_subject_balanced_accuracy.svg"
    primary = subject_metrics.loc[
        subject_metrics["track"].isin(["feature_window", "feature_sequence", "raw_eeg"])
        & (subject_metrics["seed"] == 42)
    ]
    if primary.empty:
        write_placeholder(path, "Subject balanced accuracy", "No applicable models")
    else:
        labels, values = [], []
        for (track, model), group in primary.groupby(["track", "model"], sort=True):
            labels.append(f"{track}\n{model}\nn={group.subject_id.nunique()}")
            values.append(group["balanced_accuracy"].dropna().to_numpy())
        write_boxplot(
            path,
            title="Subject-level performance by compatible analysis track",
            ylabel="Subject balanced accuracy",
            labels=labels,
            values=values,
        )
    paths.append(path)

    path = figures_dir / "sequence_model_paired_deltas.svg"
    seq = subject_metrics.loc[
        (subject_metrics["track"] == "feature_sequence")
        & (subject_metrics["seed"] == 42)
    ]
    left = seq.loc[seq["model"] == "lstm", ["subject_id", "balanced_accuracy"]]
    right = seq.loc[seq["model"] == "bilstm", ["subject_id", "balanced_accuracy"]]
    merged = left.merge(right, on="subject_id", suffixes=("_lstm", "_bilstm"))
    if merged.empty:
        write_placeholder(path, "Sequence paired deltas", "No exactly aligned sequence pair")
    else:
        delta = np.sort(
            merged["balanced_accuracy_lstm"] - merged["balanced_accuracy_bilstm"]
        )
        write_scatter(
            path,
            title="Length-10 gap-aware recurrent models; Transformer excluded",
            xlabel=f"Subjects sorted by delta (n={len(delta)})",
            ylabel="LSTM − BiLSTM balanced accuracy",
            series=[("LSTM − BiLSTM", np.arange(len(delta)), delta)],
            horizontal_zero=True,
        )
    paths.append(path)

    path = figures_dir / "sequence_model_confusion_matrices.svg"
    sequence_keys = [
        key for key in predictions
        if key[0] == "feature_sequence" and key[2] == 42
    ]
    if not sequence_keys:
        write_placeholder(path, "Sequence confusion matrices", "No sequence predictions")
    else:
        matrices = []
        labels = []
        for key in sorted(sequence_keys):
            analysis = calculate_error_analysis(predictions[key])
            matrices.append(np.asarray(analysis["row_normalized_confusion_matrix"]))
            labels.append(f"{key[1]} n={analysis['n_samples']}")
        combined = np.concatenate(matrices, axis=1)
        column_labels = [
            f"{label} / pred {predicted}"
            for label in labels
            for predicted in range(5)
        ]
        write_heatmap(
            path,
            title="Row-normalized sequence-model confusion matrices",
            xlabel="Model and predicted class",
            ylabel="True class 0–4",
            matrix=combined,
            column_labels=column_labels,
            width=1250,
            height=520,
        )
    paths.append(path)

    path = figures_dir / "raw_model_paired_deltas.svg"
    raw = subject_metrics.loc[(subject_metrics.track == "raw_eeg") & (subject_metrics.seed == 42)]
    shallow = raw.loc[raw.model == "shallowconvnet", ["subject_id", "balanced_accuracy"]]
    eegnet = raw.loc[raw.model == "eegnet", ["subject_id", "balanced_accuracy"]]
    merged = shallow.merge(eegnet, on="subject_id", suffixes=("_shallow", "_eegnet"))
    if merged.empty:
        write_placeholder(path, "Raw model paired deltas", "No exactly aligned raw pair")
    else:
        delta = np.sort(
            merged.balanced_accuracy_shallow - merged.balanced_accuracy_eegnet
        )
        write_scatter(
            path,
            title="Raw deduplicated EEG, seed 42",
            xlabel=f"Subjects sorted by delta (n={len(delta)})",
            ylabel="ShallowConvNet − EEGNet balanced accuracy",
            series=[("ShallowConvNet − EEGNet", np.arange(len(delta)), delta)],
            horizontal_zero=True,
        )
    paths.append(path)

    path = figures_dir / "preprocessing_factor_effects.svg"
    if preprocessing_effects.empty:
        write_placeholder(path, "Preprocessing effects", "No full factorial data")
    else:
        write_bar(
            path,
            title="ShallowConvNet factorial main effects, seed 42",
            xlabel="Preprocessing factor",
            ylabel="Mean fold balanced accuracy: enabled − disabled",
            labels=preprocessing_effects["factor"].tolist(),
            values=preprocessing_effects["main_effect_on_minus_off"].tolist(),
        )
    paths.append(path)

    path = figures_dir / "calibration_budget_curve.svg"
    if calibration_effects.empty:
        write_placeholder(path, "Calibration budget curve", "No matched calibration effects")
    else:
        line_series = []
        for method, group in calibration_effects.groupby("calibration_method", sort=True):
            summary = group.groupby("budget_seconds")["balanced_accuracy"].mean()
            line_series.append((str(method), summary.index.to_numpy(), summary.to_numpy()))
        zero = calibration_effects.groupby("budget_seconds")["zero_shot_balanced_accuracy"].mean()
        line_series.append(("matched zero-shot", zero.index.to_numpy(), zero.to_numpy()))
        write_lines(
            path,
            title="Matched evaluation tails by calibration budget",
            xlabel="Calibration budget (seconds)",
            ylabel="Mean subject balanced accuracy",
            series=line_series,
        )
    paths.append(path)

    path = figures_dir / "calibration_subject_heatmap.svg"
    if calibration_effects.empty:
        write_placeholder(path, "Calibration subject heatmap", "No matched calibration effects")
    else:
        heatmap = calibration_effects.pivot_table(
            index="subject_id",
            columns=["calibration_method", "budget_seconds"],
            values="delta_balanced_accuracy",
        ).sort_index()
        write_heatmap(
            path,
            title="Calibration effect by anonymized subject",
            xlabel="Method and budget",
            ylabel=f"Anonymized subjects (n={len(heatmap)})",
            matrix=heatmap.to_numpy(),
            column_labels=[
                f"{method} {int(budget)}s" for method, budget in heatmap.columns
            ],
        )
    paths.append(path)

    for filename, predictor, xlabel in (
        (
            "calibration_delta_vs_zero_shot.svg",
            "zero_shot_balanced_accuracy",
            "Matched zero-shot balanced accuracy",
        ),
        (
            "calibration_delta_vs_class_coverage.svg",
            "number_of_classes",
            "Calibration classes represented",
        ),
        ):
        path = figures_dir / filename
        if calibration_effects.empty:
            write_placeholder(path, xlabel, "No matched calibration effects")
        else:
            scatter_series = []
            for method, group in calibration_effects.groupby("calibration_method", sort=True):
                scatter_series.append((
                    str(method),
                    group[predictor].to_numpy(),
                    group["delta_balanced_accuracy"].to_numpy(),
                ))
            write_scatter(
                path,
                title=f"Subject-level matched effects (n={calibration_effects.subject_id.nunique()})",
                xlabel=xlabel,
                ylabel="Calibrated − matched zero-shot balanced accuracy",
                series=scatter_series,
                horizontal_zero=True,
            )
        paths.append(path)

    path = figures_dir / "source_performance.svg"
    if source_summary.empty:
        write_placeholder(path, "Source performance", "Source metadata unavailable")
    else:
        labels = source_summary["model"] + "\n" + source_summary["source"]
        write_bar(
            path,
            title="Performance by acquisition source",
            xlabel="Model and source (subject counts are not additive)",
            ylabel="Descriptive balanced accuracy",
            labels=labels.tolist(),
            values=source_summary["balanced_accuracy"].tolist(),
            width=1200,
            height=600,
        )
    paths.append(path)
    return paths


@dataclass
class StatisticalAnalysis:
    spec_path: Path
    spec: dict[str, Any]
    tracks: set[str] | None
    bootstrap_samples: int
    random_state: int
    output_dir: Path
    reports_dir: Path
    figures_dir: Path
    confidence_level: float

    def __init__(
        self,
        spec_path: str | Path,
        *,
        tracks: Iterable[str] | None = None,
        bootstrap_samples: int | None = None,
        random_state: int | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.spec_path = _repo_path(spec_path)
        self.spec = load_statistical_analysis_spec(self.spec_path)
        analysis = self.spec["analysis"]
        self.tracks = None if tracks is None else set(tracks)
        self.bootstrap_samples = int(
            bootstrap_samples
            if bootstrap_samples is not None
            else analysis.get("bootstrap_samples", 10_000)
        )
        self.random_state = int(
            random_state if random_state is not None else analysis.get("random_state", 42)
        )
        self.output_dir = _repo_path(output_dir or analysis.get("output_dir", "benchmark_results/analysis"))
        self.reports_dir = _repo_path(analysis.get("reports_dir", "reports"))
        self.figures_dir = _repo_path(analysis.get("figures_dir", "reports/figures"))
        self.confidence_level = float(analysis.get("confidence_level", 0.95))
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        available = {str(rule["analysis_track"]) for rule in self.spec["run_rules"]}
        if self.tracks is not None:
            unknown = self.tracks - available
            if unknown:
                raise ValueError(f"Unknown analysis tracks: {sorted(unknown)}")

    def _inventory(self) -> list[InventoryEntry]:
        entries = build_run_inventory(self.spec)
        if self.tracks is None:
            return entries
        return [entry for entry in entries if entry.analysis_track in self.tracks]

    def plan(self) -> dict[str, Any]:
        """Return a read-only plan; no output directory is created."""

        entries = self._inventory()
        selected = canonical_entries(entries, tracks=self.tracks)
        return {
            "analysis_name": self.spec["analysis"]["name"],
            "spec_path": str(self.spec_path),
            "tracks": sorted({entry.analysis_track for entry in selected}),
            "candidate_runs": len(entries),
            "usable_runs": sum(entry.usable for entry in entries),
            "canonical_runs": [entry.to_dict() for entry in selected],
            "excluded_runs": [entry.to_dict() for entry in entries if not entry.usable],
            "bootstrap_samples": self.bootstrap_samples,
            "confidence_level": self.confidence_level,
            "random_state": self.random_state,
            "output_dir": str(self.output_dir),
            "reports_dir": str(self.reports_dir),
            "figures_dir": str(self.figures_dir),
            "writes_performed": False,
        }

    @staticmethod
    def render_plan(plan: Mapping[str, Any]) -> str:
        lines = [
            "# Statistical analysis plan",
            "",
            f"- Spec: `{plan['spec_path']}`",
            f"- Tracks: {', '.join(plan['tracks'])}",
            f"- Candidate runs: {plan['candidate_runs']}",
            f"- Usable runs: {plan['usable_runs']}",
            f"- Canonical runs: {len(plan['canonical_runs'])}",
            f"- Excluded runs: {len(plan['excluded_runs'])}",
            f"- Subject bootstrap samples: {plan['bootstrap_samples']}",
            f"- Confidence level: {plan['confidence_level']}",
            f"- Random state: {plan['random_state']}",
            "- Writes performed: no",
            "",
            "## Canonical selections",
            "",
        ]
        for entry in plan["canonical_runs"]:
            lines.append(
                f"- {entry['analysis_track']} / {entry['model']} / seed "
                f"{entry['seed']}: `{entry['run_directory']}` "
                f"({entry['number_of_predictions']} predictions, {entry['folds']} folds)"
            )
        return "\n".join(lines)

    def _write_inventory(self, entries: Sequence[InventoryEntry]) -> tuple[Path, Path]:
        json_path = self.reports_dir / "statistical_analysis_run_inventory.json"
        md_path = self.reports_dir / "statistical_analysis_run_inventory.md"
        _write_json(json_path, [entry.to_dict() for entry in entries])
        table = pd.DataFrame([entry.to_dict() for entry in entries])
        columns = [
            "analysis_track", "model", "seed", "canonical", "usable",
            "manifest_status", "number_of_predictions", "subjects", "folds",
            "representation", "preprocessing", "prediction_unit", "reason",
            "run_directory", "config_hash", "prediction_file", "metrics_file",
        ]
        md_path.write_text(
            "# Statistical analysis run inventory\n\n"
            "Selection is content-aware: completed status, complete folds, resolved "
            "configuration semantics, smoke limits, identities, and only then recency. "
            "Legacy pre-manifest runs are explicitly labelled and accepted only when "
            "their complete prediction and metrics artifacts validate.\n\n"
            + _table(table[[column for column in columns if column in table]]),
            encoding="utf-8",
        )
        return md_path, json_path

    def execute(self) -> dict[str, Any]:
        """Run read-only aggregation and write analysis artifacts only."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        entries = self._inventory()
        inventory_paths = self._write_inventory(entries)
        selected = canonical_entries(entries, tracks=self.tracks)
        selected_map = {
            _entry_key(entry.analysis_track, entry.model, entry.seed): entry
            for entry in selected
        }

        predictions: dict[tuple[str, str, int], pd.DataFrame] = {}
        subject_frames: list[pd.DataFrame] = []
        fold_frames: list[pd.DataFrame] = []
        for entry in selected:
            if entry.analysis_track == "calibration":
                continue
            frame = _load_predictions(entry)
            key = _entry_key(entry.analysis_track, entry.model, entry.seed)
            predictions[key] = frame
            subject_frames.append(_subject_rows_for_entry(entry, frame))
            fold_frames.append(_fold_metrics(entry, frame))

        calibration_frames, calibration_metadata = _calibration_frames(selected)
        for (method, budget), frame in calibration_frames.items():
            if budget not in {180.0, 300.0, 600.0}:
                continue
            subject_frames.append(calculate_subject_metrics(
                frame,
                track="calibration",
                model=method,
                seed=42,
                budget_seconds=budget,
            ))

        subject_metrics = pd.concat(subject_frames, ignore_index=True, sort=False)
        subject_metrics_path = self.output_dir / "subject_metrics.parquet"
        subject_metrics.to_parquet(subject_metrics_path, index=False)
        fold_metrics = pd.concat(fold_frames, ignore_index=True, sort=False)
        fold_metrics_path = self.output_dir / "fold_metrics.parquet"
        fold_metrics.to_parquet(fold_metrics_path, index=False)
        model_summary = _subject_model_summary(
            subject_metrics,
            n_resamples=self.bootstrap_samples,
            confidence_level=self.confidence_level,
            random_state=self.random_state,
        )
        model_summary_path = self.output_dir / "model_subject_summary.parquet"
        model_summary.to_parquet(model_summary_path, index=False)

        comparison_specs = [
            comparison
            for comparison in self.spec.get("comparisons", [])
            if self.tracks is None or comparison["track"] in self.tracks
        ]
        if self.tracks is None or "calibration" in self.tracks:
            comparison_specs.extend(_calibration_comparison_specs())
        alignments: list[AlignmentResult] = []
        aligned_comparisons: list[dict[str, Any]] = []
        for comparison in comparison_specs:
            track = str(comparison["track"])
            seed = int(comparison.get("seed", 42))
            left_model = str(comparison["left_model"])
            right_model = str(comparison["right_model"])
            if track == "calibration":
                budget = float(comparison["budget_seconds"])
                left_frame = calibration_frames.get((left_model, budget))
                right_frame = calibration_frames.get((right_model, budget))
                prediction_unit = "sequence"
                identity = "sequence_id"
            else:
                left_entry = selected_map.get(_entry_key(track, left_model, seed))
                right_entry = selected_map.get(_entry_key(track, right_model, seed))
                left_frame = predictions.get(_entry_key(track, left_model, seed))
                right_frame = predictions.get(_entry_key(track, right_model, seed))
                prediction_unit = left_entry.prediction_unit if left_entry else "unknown"
                identity = left_entry.identity_column if left_entry else None
            if left_frame is None or right_frame is None:
                result = AlignmentResult(
                    left_model, right_model, prediction_unit, identity or "",
                    False, "canonical prediction artifact missing", 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0,
                )
            else:
                result = check_alignment(
                    left_frame,
                    right_frame,
                    left_model=left_model,
                    right_model=right_model,
                    prediction_unit=prediction_unit,
                    identity_column=identity,
                )
            alignments.append(result)
            if result.aligned:
                aligned_comparisons.append(dict(comparison))

        alignment_rows = []
        for comparison, result in zip(comparison_specs, alignments):
            alignment_rows.append({"comparison": comparison["name"], **result.to_dict()})
        alignment_path = self.output_dir / "alignment.json"
        _write_json(alignment_path, alignment_rows)
        alignment_report = self.reports_dir / "statistical_alignment_report.md"
        alignment_report.write_text(
            "# Statistical alignment report\n\n"
            "Paired analysis requires exact identity, outer fold, subject, and target "
            "equality. No time-based or cross-representation approximation is used.\n\n"
            + _table(pd.DataFrame(alignment_rows)[[
                "comparison", "prediction_unit", "identity_column", "aligned", "reason",
                "left_predictions", "right_predictions", "matched_predictions",
                "left_duplicates", "right_duplicates", "fold_mismatches",
                "subject_mismatches", "target_mismatches",
            ]])
            + "\n\nDirect paired tests between feature windows, feature sequences, and raw "
            "EEG windows are intentionally absent. The completed recurrent runs use "
            "length 10 (43,828 units), while Transformer uses length 8 (44,142 units); "
            "only the exact LSTM/BiLSTM length-10 pair is inferentially comparable.",
            encoding="utf-8",
        )

        paired_rows: list[dict[str, Any]] = []
        aligned_names = {comparison["name"] for comparison in aligned_comparisons}
        for comparison in comparison_specs:
            if comparison["name"] not in aligned_names:
                continue
            paired_rows.extend(_paired_metric_rows(
                comparison,
                subject_metrics,
                n_resamples=self.bootstrap_samples,
                confidence_level=self.confidence_level,
                random_state=self.random_state,
            ))
        inferential = [row for row in paired_rows if row["inferential"]]
        descriptive = [row for row in paired_rows if not row["inferential"]]
        paired_rows = apply_holm_by_family(inferential) + [
            {**row, "holm_adjusted_p_value": np.nan} for row in descriptive
        ]
        paired_path = self.output_dir / "paired_statistics.json"
        _write_json(paired_path, paired_rows)

        preprocessing_trials, preprocessing_effects = _preprocessing_summary(fold_metrics)
        preprocessing_multiseed = _preprocessing_multiseed_deltas(
            preprocessing_trials
        )
        preprocessing_trials.to_parquet(
            self.output_dir / "preprocessing_trial_summary.parquet", index=False
        )
        preprocessing_multiseed.to_parquet(
            self.output_dir / "preprocessing_multiseed_deltas.parquet", index=False
        )
        calibration_effects, calibration_correlations = _calibration_effects(
            calibration_metadata,
            n_resamples=self.bootstrap_samples,
            random_state=self.random_state,
        )
        calibration_effects.to_parquet(
            self.output_dir / "calibration_subject_effects.parquet", index=False
        )
        _write_json(
            self.output_dir / "calibration_correlations.json",
            calibration_correlations.to_dict("records"),
        )

        error_rows: list[dict[str, Any]] = []
        source_frames: list[pd.DataFrame] = []
        for key, frame in predictions.items():
            track, model, seed = key
            if seed != 42 or track not in {"feature_window", "feature_sequence", "raw_eeg"}:
                continue
            analysis = calculate_error_analysis(frame)
            error_rows.append({"track": track, "model": model, "seed": seed, **analysis})
            source = summarize_by_source(frame, model=model)
            source["track"] = track
            source["unique_subjects_overall"] = source.attrs["unique_subjects_overall"]
            source["subject_counts_additive"] = False
            source_frames.append(source)
        source_summary = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame()
        error_path = self.output_dir / "error_analysis.json"
        _write_json(error_path, error_rows)
        source_summary.to_parquet(self.output_dir / "source_summary.parquet", index=False)
        difficult = _difficult_subjects(subject_metrics, calibration_effects)
        _write_json(
            self.output_dir / "difficult_subjects.json",
            {name: frame.to_dict("records") for name, frame in difficult.items()},
        )

        figure_paths = _build_figures(
            self.figures_dir,
            subject_metrics=subject_metrics,
            predictions=predictions,
            preprocessing_trials=preprocessing_trials,
            preprocessing_effects=preprocessing_effects,
            calibration_effects=calibration_effects,
            source_summary=source_summary,
        )

        training_rows = [
            {
                "track": entry.analysis_track,
                "model": entry.model,
                "seed": entry.seed,
                "prediction_unit": entry.prediction_unit,
                "n_predictions": entry.number_of_predictions,
                **_artifact_training_summary(entry),
            }
            for entry in selected
            if entry.analysis_track != "calibration"
        ]
        training_summary = pd.DataFrame(training_rows)

        statistical_report = self.reports_dir / "statistical_model_comparison.md"
        comparison_table = pd.DataFrame(paired_rows)
        primary_fold_metrics = fold_metrics.loc[
            (fold_metrics["seed"] == 42)
            & fold_metrics["track"].isin(
                ["feature_window", "feature_sequence", "raw_eeg"]
            ),
            [
                "track", "model", "outer_fold", "n_samples", "accuracy",
                "balanced_accuracy", "macro_f1", "auc",
            ],
        ]
        comparison_columns = [
            "family", "comparison", "metric", "n_subjects", "mean_difference",
            "median_difference", "ci_low", "ci_high",
            "probability_difference_gt_zero", "subjects_improved",
            "subjects_degraded", "ties", "fraction_improved",
            "fraction_degraded", "number_needed_to_improve", "rank_biserial",
            "wilcoxon_p_value",
            "holm_adjusted_p_value", "sign_test_p_value", "wilcoxon_status",
        ]
        statistical_report.write_text(
            "# Statistical model comparison\n\n"
            "The independent unit for every inferential comparison is the anonymized "
            "subject, never a window or sequence. Differences are left model minus "
            "right model. Confidence intervals use paired subject bootstrap "
            f"({self.bootstrap_samples:,} resamples, {self.confidence_level:.0%}, "
            f"seed {self.random_state}). Holm correction is applied separately inside "
            "feature-window, sequence, raw, and calibration families.\n\n"
            "## Subject-level model summaries\n\n"
            + _table(model_summary)
            + "\n\n## Paired comparisons\n\n"
            + _table(comparison_table[[column for column in comparison_columns if column in comparison_table]])
            + "\n\n## Fold-level descriptive metrics (seed 42)\n\n"
            + _table(primary_fold_metrics)
            + "\n\n## Sequence-model constraint\n\n"
            "LSTM and BiLSTM are exactly aligned length-10 runs and may be paired. "
            "The completed Transformer is length 8, so its IDs and sample count do not "
            "match; Transformer-versus-recurrent paired tests are blocked. Aggregate "
            "fold metrics remain descriptive and do not establish a paired winner.\n\n"
            "## Raw EEG models\n\n"
            "EEGNet and ShallowConvNet use the same 30,958 raw deduplicated windows "
            "within each seed. Raw and filtered variants are not mixed.\n\n"
            "## Preprocessing factorial summary\n\n"
            + _table(preprocessing_trials)
            + "\n\n### Seed-42 factorial main effects\n\n"
            + _table(preprocessing_effects)
            + "\n\n### A/B/E multiseed differences versus raw A\n\n"
            + _table(preprocessing_multiseed)
            + "\n\nBand-pass (B) and band-pass plus notch (E) change sign "
            "across seeds relative to raw A, so neither has a stable multiseed "
            "advantage. Every CAR-enabled seed-42 trial is below its corresponding "
            "non-CAR condition, consistent with a negative CAR effect for this model."
            + "\n\n## Calibration association analysis\n\n"
            + _table(calibration_correlations)
            + "\n\nSpearman intervals use a subject bootstrap capped at 2,000 "
            "resamples for computational tractability; the primary paired metric "
            f"intervals above use the predeclared {self.bootstrap_samples:,} resamples."
            + "\n\n## Training and representation metadata\n\n"
            + _table(training_summary),
            encoding="utf-8",
        )

        error_report = self.reports_dir / "error_analysis.md"
        compact_errors = pd.DataFrame([{
            "track": row["track"],
            "model": row["model"],
            "n_samples": row["n_samples"],
            "ordinal_mae": row["ordinal_mae"],
            "adjacent_accuracy": row["adjacent_accuracy"],
            "severe_error_rate": row["severe_error_rate"],
            "extreme_recall_0": row["extreme_class_recall_0"],
            "extreme_recall_4": row["extreme_class_recall_4"],
            "extreme_truth_predicted_centrally": row["extreme_truth_predicted_centrally"],
        } for row in error_rows])
        sections = [
            "# Error analysis",
            "",
            "All IDs below are the existing anonymized subject identifiers. Source "
            "metrics are descriptive: a subject present in both sources remains one "
            "statistical subject, and source subject counts are explicitly non-additive.",
            "",
            "## Class and ordinal errors",
            "",
            _table(compact_errors),
            "",
            "## Source performance",
            "",
            _table(source_summary.drop(columns=["subject_ids"], errors="ignore")),
        ]
        for name, frame in difficult.items():
            sections.extend(["", f"## {name.replace('_', ' ').title()}", "", _table(frame)])
        sections.extend(["", "## Per-model class details", ""])
        for row in error_rows:
            sections.extend([
                f"### {row['track']} / {row['model']}",
                "",
                _table(pd.DataFrame(row["per_class"])),
                "",
                "Row-normalized confusion matrix:",
                "",
                _table(pd.DataFrame(row["row_normalized_confusion_matrix"])),
                "",
            ])
        error_report.write_text("\n".join(sections), encoding="utf-8")

        calibration_summary = pd.DataFrame()
        if not calibration_effects.empty:
            calibration_summary = calibration_effects.groupby(
                ["calibration_method", "budget_seconds"], as_index=False
            ).agg(
                subjects=("subject_id", "nunique"),
                delta_balanced_accuracy_mean=("delta_balanced_accuracy", "mean"),
                delta_macro_f1_mean=("delta_macro_f1", "mean"),
                fraction_improved=("delta_balanced_accuracy", lambda values: float((values > 0).mean())),
                fraction_degraded=("delta_balanced_accuracy", lambda values: float((values < 0).mean())),
            )
        automl_stats = comparison_table.loc[
            comparison_table.get("family", pd.Series(dtype=str)) == "automl_pilot"
        ] if not comparison_table.empty else pd.DataFrame()
        article_report = self.reports_dir / "article_results_summary.md"
        article_report.write_text(
            "# Article-ready results summary\n\n"
            "## 1. Dataset and evaluation protocol\n\n"
            "The supervised target is five-level `label_q5`. Scientific model results "
            "use five-fold GroupKFold by `subject_id`; inner validation is confined to "
            "outer-train data. Subject is the independent inferential unit.\n\n"
            "## 2. Random-window sanity versus subject GroupKFold\n\n"
            "Earlier random-window runs remain technical sanity checks and are not "
            "included in inferential claims. GroupKFold is the defensible primary "
            "protocol because it prevents train/test subject overlap.\n\n"
            "## 3. Feature-window baselines\n\n"
            "Random Forest and Torch MLP are exactly aligned on 45,384 ten-second "
            "EEG+POW feature windows. Their paired subject results are reported with "
            "bootstrap intervals, Wilcoxon, exact sign tests, and within-family Holm "
            "adjustment.\n\n"
            "## 4. Sequence-model comparison\n\n"
            "The available LSTM/BiLSTM pair uses 43,828 gap-aware sequences of length "
            "10. Transformer uses 44,142 sequences of length 8. Therefore the recurrent "
            "pair can be tested against each other, but no exact paired Transformer "
            "comparison is currently defensible. Aggregate results can only be described.\n\n"
            "## 5. Raw EEG model comparison\n\n"
            "EEGNet and ShallowConvNet are exactly aligned on 30,958 raw deduplicated "
            "EEG windows for seeds 7, 42, and 123. Filtered runs are excluded from this "
            "model-family comparison.\n\n"
            "## 6. Logical-record deduplication\n\n"
            "Raw-model analyses use the logical-record-deduplicated dataset and its "
            "existing fold assignments; no cache or dataset was rebuilt.\n\n"
            "## 7. Preprocessing ablation\n\n"
            + _table(preprocessing_effects)
            + "\n\nA/B/E multiseed deltas versus raw A:\n\n"
            + _table(preprocessing_multiseed)
            + "\n\nThe full A–H factorial is evaluated at seed 42, while A/B/E have "
            "seeds 7, 42, and 123. Band-pass and notch deltas change sign across "
            "seeds; they do not show a stable advantage over raw input. CAR is "
            "consistently negative within the seed-42 matched factorial contrasts.\n\n"
            "## 8. Transformer AutoML pilot\n\n"
            + _table(automl_stats)
            + "\n\nThis is an outer-fold-1 case study only. It does not support a general "
            "claim about AutoML, and no p-value is reported for it.\n\n"
            "## 9. User calibration\n\n"
            + _table(calibration_summary)
            + "\n\nEvery calibrated method is compared with zero-shot predictions from the "
            "same subject, budget, and evaluation tail. Calibration inputs are excluded "
            "from evaluation metrics.\n\n"
            "## 10. Limitations\n\n"
            "The cohort contains 54 subjects, subject AUC is undefined when required "
            "classes are absent, recurrent and Transformer sequence definitions differ, "
            "the AutoML result covers one outer fold, and calibration effects are "
            "heterogeneous. No external-dataset generalization is evaluated here.\n\n"
            "## 11. Main defensible claims\n\n"
            "- GroupKFold performance is above the five-class chance reference for the "
            "main completed models, but varies materially across subjects.\n"
            "- Raw-model differences are assessed only on exact deduplicated windows.\n"
            "- CAR has a negative seed-42 factorial main effect for ShallowConvNet; "
            "band-pass/notch stability must be judged across the available seeds.\n"
            "- Head-only calibration has a small, heterogeneous matched effect, while "
            "short-interval subject normalization is harmful on average.\n"
            "- The outer-fold-1 AutoML pilot did not improve the baseline Transformer.\n\n"
            "## 12. Claims that cannot yet be made\n\n"
            "- Transformer cannot be declared superior to LSTM/BiLSTM from a paired "
            "test because sequence identities differ.\n"
            "- The AutoML pilot cannot be generalized beyond outer fold 1.\n"
            "- A preprocessing choice cannot be called universally optimal from this "
            "single architecture and limited seed set.\n"
            "- No clinical, causal, or external-population claim is supported.",
            encoding="utf-8",
        )

        summary = {
            "analysis_name": self.spec["analysis"]["name"],
            "canonical_runs": len(selected),
            "tracks": sorted({entry.analysis_track for entry in selected}),
            "subject_metric_rows": len(subject_metrics),
            "alignment_checks": len(alignment_rows),
            "aligned_comparisons": sum(row["aligned"] for row in alignment_rows),
            "blocked_comparisons": sum(not row["aligned"] for row in alignment_rows),
            "paired_statistic_rows": len(paired_rows),
            "bootstrap_samples": self.bootstrap_samples,
            "random_state": self.random_state,
            "artifacts": {
                "inventory_markdown": str(inventory_paths[0]),
                "inventory_json": str(inventory_paths[1]),
                "alignment_report": str(alignment_report),
                "subject_metrics": str(subject_metrics_path),
                "fold_metrics": str(fold_metrics_path),
                "model_subject_summary": str(model_summary_path),
                "paired_statistics": str(paired_path),
                "error_analysis_json": str(error_path),
                "preprocessing_trial_summary": str(
                    self.output_dir / "preprocessing_trial_summary.parquet"
                ),
                "preprocessing_multiseed_deltas": str(
                    self.output_dir / "preprocessing_multiseed_deltas.parquet"
                ),
                "calibration_subject_effects": str(
                    self.output_dir / "calibration_subject_effects.parquet"
                ),
                "calibration_correlations": str(
                    self.output_dir / "calibration_correlations.json"
                ),
                "statistical_model_comparison": str(statistical_report),
                "error_analysis_report": str(error_report),
                "article_results_summary": str(article_report),
                "figures": [str(path) for path in figure_paths],
            },
        }
        summary_path = self.output_dir / "analysis_summary.json"
        summary["artifacts"]["analysis_summary"] = str(summary_path)
        _write_json(summary_path, summary)
        return summary
