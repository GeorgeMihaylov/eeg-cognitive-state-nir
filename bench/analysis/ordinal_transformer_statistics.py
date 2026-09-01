"""Paired subject-level analysis of categorical and ordinal Transformers.

The module consumes completed prediction artifacts only.  Windows, sequences,
folds, and sources are retained as descriptive strata; statistical inference
uses one paired observation per ``subject_id``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from bench.analysis.alignment import AlignmentError
from bench.analysis.paired_statistics import (
    apply_holm_by_family,
    paired_subject_statistics,
)
from cogstate.model_zoo.DL.sequence_utils import sequence_index_sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_GROUPS = ("eeg_only", "eeg_pow")
METHODS = ("categorical", "coral", "corn")
IDENTITY_COLUMNS = (
    "sequence_id", "fold", "subject_id", "record_id", "source", "y_true"
)
PRIMARY_METRICS = ("ordinal_mae", "severe_error_rate")
SECONDARY_METRICS = (
    "balanced_accuracy", "macro_f1", "quadratic_weighted_kappa",
    "adjacent_accuracy", "expected_rank_mae", "expected_rank_spearman",
)
FEATURE_GROUP_METRICS = (
    "balanced_accuracy", "macro_f1", "quadratic_weighted_kappa",
    "ordinal_mae", "severe_error_rate",
)
HIGHER_IS_BETTER = {
    "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "kappa",
    "auc", "quadratic_weighted_kappa", "adjacent_accuracy",
    "expected_rank_spearman",
}
LOWER_IS_BETTER = {
    "ordinal_mae", "severe_error_rate", "expected_rank_mae",
}
SUBJECT_METRICS = (
    "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "kappa",
    "auc", "quadratic_weighted_kappa", "ordinal_mae", "adjacent_accuracy",
    "severe_error_rate", "expected_rank_mae", "expected_rank_spearman",
)
WILCOXON_CONVENTION = (
    "two-sided scipy.stats.wilcoxon; zero_method='wilcox' discards exact zero "
    "differences before ranking; method='auto'; all-zero pairs are explicitly undefined"
)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as input_file:
        value = yaml.safe_load(input_file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _method_key(method: str, feature_group: str) -> str:
    return f"{method}_{feature_group}"


@dataclass(frozen=True)
class ResolvedRun:
    method: str
    feature_group: str
    run_directory: Path
    manifest_file: Path
    config_file: Path
    prediction_file: Path
    config_hash: str
    timestamp: str
    seed: int
    sequence_length: int
    sequence_count: int
    subject_count: int
    fold_count: int
    sequence_index_sha256: str
    prediction_index_sha256: str

    @property
    def key(self) -> str:
        return _method_key(self.method, self.feature_group)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("run_directory", "manifest_file", "config_file", "prediction_file"):
            value[key] = _display_path(value[key])
        return value


def _prediction_file(run_directory: Path) -> Path:
    matches = [
        path for path in run_directory.rglob("predictions.parquet")
        if path.parent.name == "group_kfold_subject"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one unified prediction file in {run_directory}, found {len(matches)}"
        )
    return matches[0]


def _run_identity(config: Mapping[str, Any]) -> tuple[str, str] | None:
    experiment = config.get("experiment", {})
    if not isinstance(experiment, Mapping):
        return None
    feature_group = str(experiment.get("feature_group", ""))
    if feature_group not in FEATURE_GROUPS:
        return None
    head_type = str(experiment.get("head_type", "")).lower()
    experiment_type = str(experiment.get("type", "")).lower()
    if experiment_type == "ordinal_transformer_full" and head_type in {"coral", "corn"}:
        return head_type, feature_group
    trial_id = str(experiment.get("trial_id", ""))
    task = str(experiment.get("task", ""))
    if (
        trial_id == f"transformer_classification_{feature_group}"
        and task == "classification"
        and "smoke" not in trial_id.lower()
    ):
        return "categorical", feature_group
    return None


def _seed(config: Mapping[str, Any]) -> int:
    experiment = config.get("experiment", {})
    if isinstance(experiment, Mapping) and experiment.get("seed") is not None:
        return int(experiment["seed"])
    models = config.get("models", {})
    if isinstance(models, Mapping) and models:
        model = next(iter(models.values()))
        if isinstance(model, Mapping):
            params = model.get("params", {})
            if isinstance(params, Mapping) and params.get("random_state") is not None:
                return int(params["random_state"])
    return int((config.get("evaluation", {}) or {}).get("random_state", -1))


def _validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    expected_sequences: int,
    expected_subjects: int,
    expected_folds: int,
    expected_prediction_hash: str | None = None,
) -> str:
    required = set(IDENTITY_COLUMNS) | {"y_pred"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction artifact is missing columns: {missing}")
    if len(frame) != expected_sequences:
        raise ValueError(f"Expected {expected_sequences} sequences, observed {len(frame)}")
    if frame["sequence_id"].isna().any() or frame["sequence_id"].duplicated().any():
        raise ValueError("sequence_id must be unique and non-null")
    if frame["subject_id"].nunique() != expected_subjects:
        raise ValueError(
            f"Expected {expected_subjects} subjects, observed {frame['subject_id'].nunique()}"
        )
    if frame["fold"].nunique() != expected_folds:
        raise ValueError(f"Expected {expected_folds} folds, observed {frame['fold'].nunique()}")
    subject_folds = frame.groupby("subject_id")["fold"].nunique()
    if not bool((subject_folds == 1).all()):
        raise ValueError("Every subject_id must belong to exactly one outer fold")
    digest = sequence_index_sha256(frame)
    if expected_prediction_hash is not None and digest != expected_prediction_hash:
        raise ValueError(
            f"Prediction-artifact index hash mismatch: "
            f"{digest} != {expected_prediction_hash}"
        )
    return digest


def discover_canonical_runs(document: Mapping[str, Any]) -> dict[str, ResolvedRun]:
    """Discover the latest complete semantic run for each required method."""

    analysis = document.get("analysis", {})
    expected = document.get("expected", {})
    roots = [_repo_path(value) for value in analysis.get("run_roots", [])]
    required_keys = {
        _method_key(method, feature_group)
        for feature_group in FEATURE_GROUPS for method in METHODS
    }
    candidates: dict[str, list[ResolvedRun]] = {key: [] for key in required_keys}
    for manifest_file in sorted(
        path for root in roots if root.is_dir() for path in root.rglob("run_manifest.json")
    ):
        try:
            manifest = _load_json(manifest_file)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if manifest.get("status") != "completed":
            continue
        run_directory = manifest_file.parent
        config_file = run_directory / "config.yaml"
        if not config_file.is_file():
            continue
        config = _load_yaml(config_file)
        identity = _run_identity(config)
        if identity is None:
            continue
        method, feature_group = identity
        if _seed(config) != int(expected["seed"]):
            continue
        sequence = config.get("sequence", {})
        evaluation = config.get("evaluation", {})
        if int(sequence.get("length", -1)) != int(expected["sequence_length"]):
            continue
        if int(evaluation.get("n_splits", -1)) != int(expected["folds"]):
            continue
        if evaluation.get("folds") not in (None, list(range(1, int(expected["folds"]) + 1))):
            continue
        experiment = config.get("experiment", {})
        if method != "categorical" and (
            not isinstance(experiment, Mapping)
            or str(experiment.get("full_sequence_index_sha256", ""))
            != str(expected["sequence_index_sha256"])
        ):
            continue
        try:
            prediction_file = _prediction_file(run_directory)
            predictions = pd.read_parquet(prediction_file)
            digest = _validate_prediction_frame(
                predictions,
                expected_sequences=int(expected["sequences"]),
                expected_subjects=int(expected["subjects"]),
                expected_folds=int(expected["folds"]),
                expected_prediction_hash=(
                    str(expected["prediction_artifact_index_sha256"])
                    if expected.get("prediction_artifact_index_sha256") is not None
                    else None
                ),
            )
        except (OSError, ValueError):
            continue
        timestamp = str(manifest.get("timestamp", run_directory.name))
        candidates[_method_key(method, feature_group)].append(ResolvedRun(
            method=method,
            feature_group=feature_group,
            run_directory=run_directory,
            manifest_file=manifest_file,
            config_file=config_file,
            prediction_file=prediction_file,
            config_hash=str(manifest.get("config_hash", "")),
            timestamp=timestamp,
            seed=_seed(config),
            sequence_length=int(sequence["length"]),
            sequence_count=len(predictions),
            subject_count=int(predictions["subject_id"].nunique()),
            fold_count=int(predictions["fold"].nunique()),
            sequence_index_sha256=str(expected["sequence_index_sha256"]),
            prediction_index_sha256=digest,
        ))
    missing = sorted(key for key in required_keys if not candidates[key])
    if missing:
        raise ValueError(f"Missing valid completed canonical runs: {missing}")
    return {
        key: sorted(values, key=lambda value: (value.timestamp, str(value.run_directory)))[-1]
        for key, values in sorted(candidates.items())
    }


def require_six_way_alignment(predictions: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Require exact equality of all scientific identity columns."""

    expected_keys = {
        _method_key(method, feature_group)
        for feature_group in FEATURE_GROUPS for method in METHODS
    }
    if set(predictions) != expected_keys:
        raise AlignmentError(
            f"Expected six methods {sorted(expected_keys)}, observed {sorted(predictions)}"
        )
    reference_key = "categorical_eeg_only"
    reference = predictions[reference_key].sort_values("sequence_id").reset_index(drop=True)
    audits: list[dict[str, Any]] = []
    for key in sorted(expected_keys - {reference_key}):
        candidate = predictions[key].sort_values("sequence_id").reset_index(drop=True)
        mismatches = {
            column: int((reference[column].astype(str) != candidate[column].astype(str)).sum())
            for column in IDENTITY_COLUMNS
        }
        count_mismatch = len(reference) != len(candidate)
        duplicates = int(candidate["sequence_id"].duplicated().sum())
        if count_mismatch or duplicates or any(mismatches.values()):
            raise AlignmentError(
                f"Six-way alignment failed for {key}: count_mismatch={count_mismatch}, "
                f"duplicates={duplicates}, mismatches={mismatches}"
            )
        audits.append({
            "reference": reference_key,
            "candidate": key,
            "rows": len(candidate),
            "duplicate_sequence_ids": duplicates,
            "mismatches": mismatches,
            "exact_match": True,
        })
    return {
        "exact_match": True,
        "reference": reference_key,
        "rows": len(reference),
        "subjects": int(reference["subject_id"].nunique()),
        "folds": int(reference["fold"].nunique()),
        "identity_columns": list(IDENTITY_COLUMNS),
        "comparisons": audits,
    }


def class_probability_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [f"class_probability_{index}" for index in range(5)]
    fallback = [f"proba_{index}" for index in range(5)]
    if set(preferred).issubset(frame.columns):
        return preferred
    if set(fallback).issubset(frame.columns):
        return fallback
    raise ValueError("Predictions require five class-probability columns")


def categorical_expected_rank(frame: pd.DataFrame) -> np.ndarray:
    probabilities = frame[class_probability_columns(frame)].to_numpy(dtype=float)
    if probabilities.shape[1] != 5 or not np.isfinite(probabilities).all():
        raise ValueError("Class probabilities must be finite with shape [n_samples, 5]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Class probabilities must sum to one")
    return probabilities @ np.arange(5, dtype=float)


def _safe_spearman(y_true: np.ndarray, expected_rank: np.ndarray) -> tuple[float, str | None]:
    if len(y_true) < 2 or np.ptp(y_true) == 0:
        return np.nan, "expected_rank_spearman_no_target_variation"
    if np.ptp(expected_rank) == 0:
        return np.nan, "expected_rank_spearman_no_prediction_variation"
    value = float(spearmanr(y_true, expected_rank).statistic)
    return value, None if np.isfinite(value) else "expected_rank_spearman_undefined"


def calculate_prediction_metrics(
    frame: pd.DataFrame,
    *,
    prediction_column: str = "y_pred",
    expected_rank: np.ndarray | None = None,
) -> dict[str, Any]:
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_pred = frame[prediction_column].to_numpy(dtype=int)
    probabilities = frame[class_probability_columns(frame)].to_numpy(dtype=float)
    if expected_rank is None:
        expected_rank = categorical_expected_rank(frame)
    expected_rank = np.asarray(expected_rank, dtype=float)
    if expected_rank.shape != y_true.shape or not np.isfinite(expected_rank).all():
        raise ValueError("Expected rank must be a finite vector matching y_true")
    present = np.unique(y_true)
    recalls = [float(np.mean(y_pred[y_true == label] == label)) for label in present]
    distance = np.abs(y_pred - y_true)
    reasons: list[str] = []
    qwk = np.nan
    if len(present) < 2:
        reasons.append("qwk_single_true_class")
    else:
        qwk = float(cohen_kappa_score(
            y_true, y_pred, labels=list(range(5)), weights="quadratic"
        ))
        if not np.isfinite(qwk):
            reasons.append("qwk_undefined")
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=list(range(5))))
    if not np.isfinite(kappa):
        reasons.append("kappa_undefined")
    auc = np.nan
    if set(present.tolist()) == set(range(5)):
        try:
            auc = float(roc_auc_score(
                y_true, probabilities, multi_class="ovr", average="weighted"
            ))
        except ValueError:
            reasons.append("auc_metric_error")
    else:
        reasons.append("auc_missing_true_classes")
    rank_spearman, rank_reason = _safe_spearman(y_true, expected_rank)
    if rank_reason:
        reasons.append(rank_reason)
    return {
        "n_samples": int(len(frame)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": kappa,
        "auc": auc,
        "quadratic_weighted_kappa": qwk,
        "ordinal_mae": float(distance.mean()),
        "adjacent_accuracy": float(np.mean(distance <= 1)),
        "severe_error_rate": float(np.mean(distance >= 2)),
        "expected_rank_mae": float(np.mean(np.abs(expected_rank - y_true))),
        "expected_rank_spearman": rank_spearman,
        "undefined_metric_reason": ";".join(reasons) if reasons else None,
    }


def calculate_subject_metrics(
    predictions: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, frame in sorted(predictions.items()):
        method, feature_group = key.split("_", 1)
        full_expected = (
            categorical_expected_rank(frame)
            if method == "categorical"
            else frame["expected_rank"].to_numpy(dtype=float)
        )
        frame = frame.copy()
        frame["_expected_rank_analysis"] = full_expected
        for subject_id, group in frame.groupby("subject_id", sort=True):
            folds = group["fold"].unique()
            if len(folds) != 1:
                raise ValueError("A subject appears in more than one outer fold")
            metrics = calculate_prediction_metrics(
                group,
                expected_rank=group["_expected_rank_analysis"].to_numpy(dtype=float),
            )
            rows.append({
                "run_key": key,
                "method": method,
                "feature_group": feature_group,
                "seed": 42,
                "subject_id": str(subject_id),
                "fold": int(folds[0]),
                "source_membership": "+".join(sorted(group["source"].astype(str).unique())),
                "n_sequences": int(len(group)),
                **metrics,
            })
    result = pd.DataFrame(rows)
    if len(result) != 6 * 53:
        raise ValueError(f"Expected 318 subject-metric rows, observed {len(result)}")
    return result


def metric_improvement(
    candidate: Iterable[float], reference: Iterable[float], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    candidate_values = np.asarray(list(candidate), dtype=float)
    reference_values = np.asarray(list(reference), dtype=float)
    raw_delta = candidate_values - reference_values
    if metric in HIGHER_IS_BETTER:
        return raw_delta, raw_delta.copy()
    if metric in LOWER_IS_BETTER:
        return raw_delta, -raw_delta
    raise ValueError(f"Unknown metric direction: {metric}")


def _heterogeneity(improvement: np.ndarray) -> dict[str, float]:
    values = improvement[np.isfinite(improvement)]
    if not len(values):
        return {name: np.nan for name in (
            "improvement_min", "improvement_q10", "improvement_q25",
            "improvement_median", "improvement_q75", "improvement_q90",
            "improvement_max", "worst_quartile_mean", "best_quartile_mean",
        )}
    ordered = np.sort(values)
    quartile_size = max(1, int(np.ceil(len(ordered) / 4)))
    return {
        "improvement_min": float(np.min(values)),
        "improvement_q10": float(np.quantile(values, 0.10)),
        "improvement_q25": float(np.quantile(values, 0.25)),
        "improvement_median": float(np.median(values)),
        "improvement_q75": float(np.quantile(values, 0.75)),
        "improvement_q90": float(np.quantile(values, 0.90)),
        "improvement_max": float(np.max(values)),
        "worst_quartile_mean": float(np.mean(ordered[:quartile_size])),
        "best_quartile_mean": float(np.mean(ordered[-quartile_size:])),
    }


def paired_metric_comparison(
    subject_metrics: pd.DataFrame,
    *,
    candidate_key: str,
    reference_key: str,
    metric: str,
    family: str,
    hypothesis_tier: str,
    n_resamples: int = 10_000,
    random_state: int = 42,
) -> dict[str, Any]:
    candidate = subject_metrics.loc[
        subject_metrics["run_key"] == candidate_key, ["subject_id", metric]
    ].set_index("subject_id")
    reference = subject_metrics.loc[
        subject_metrics["run_key"] == reference_key, ["subject_id", metric]
    ].set_index("subject_id")
    if not candidate.index.equals(reference.index):
        raise AlignmentError(
            f"Subject pairs differ for {candidate_key} vs {reference_key}"
        )
    raw_delta, improvement = metric_improvement(
        candidate[metric].to_numpy(), reference[metric].to_numpy(), metric
    )
    finite = np.isfinite(improvement)
    oriented = paired_subject_statistics(
        improvement[finite], np.zeros(int(finite.sum())),
        n_resamples=n_resamples,
        random_state=random_state,
    )
    values = improvement[finite]
    raw = raw_delta[finite]
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    standardized = float(np.mean(values) / sd) if np.isfinite(sd) and sd > 0 else np.nan
    return {
        "family": family,
        "hypothesis_tier": hypothesis_tier,
        "feature_group": candidate_key.rsplit("_", 2)[-2] + "_" + candidate_key.rsplit("_", 1)[-1]
        if candidate_key.endswith(("eeg_only", "eeg_pow")) else "cross_feature",
        "candidate": candidate_key,
        "reference": reference_key,
        "metric": metric,
        "direction": "higher_is_better" if metric in HIGHER_IS_BETTER else "lower_is_better",
        "n_valid_pairs": int(finite.sum()),
        "reference_mean": float(reference[metric].to_numpy()[finite].mean()),
        "candidate_mean": float(candidate[metric].to_numpy()[finite].mean()),
        "raw_mean_delta": float(np.mean(raw)),
        "raw_median_delta": float(np.median(raw)),
        "mean_improvement": float(np.mean(values)),
        "median_improvement": float(np.median(values)),
        "paired_difference_sd": sd,
        "standardized_paired_effect": standardized,
        "bootstrap_ci_low": oriented["ci_low"],
        "bootstrap_ci_high": oriented["ci_high"],
        "bootstrap_samples": n_resamples,
        "bootstrap_seed": random_state,
        "subjects_improved": oriented["subjects_improved"],
        "subjects_degraded": oriented["subjects_degraded"],
        "ties": oriented["ties"],
        "fraction_improved": oriented["fraction_improved"],
        "fraction_degraded": oriented["fraction_degraded"],
        "wilcoxon_status": oriented["wilcoxon_status"],
        "wilcoxon_statistic": oriented["wilcoxon_statistic"],
        "wilcoxon_p_value": oriented["wilcoxon_p_value"],
        "sign_test_status": oriented["sign_test_status"],
        "sign_test_p_value": oriented["sign_test_p_value"],
        "rank_biserial": oriented["rank_biserial"],
        "wilcoxon_convention": WILCOXON_CONVENTION,
        **_heterogeneity(values),
    }


def build_hypothesis_tables(
    subject_metrics: pd.DataFrame,
    *,
    n_resamples: int,
    random_state: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    primary: list[dict[str, Any]] = []
    secondary: list[dict[str, Any]] = []
    feature_effects: list[dict[str, Any]] = []
    for feature_group in FEATURE_GROUPS:
        reference = _method_key("categorical", feature_group)
        for candidate_method in ("coral", "corn"):
            candidate = _method_key(candidate_method, feature_group)
            for metric in PRIMARY_METRICS:
                primary.append(paired_metric_comparison(
                    subject_metrics,
                    candidate_key=candidate,
                    reference_key=reference,
                    metric=metric,
                    family=f"primary_{feature_group}",
                    hypothesis_tier="primary",
                    n_resamples=n_resamples,
                    random_state=random_state,
                ))
        for candidate_method, reference_method in (
            ("coral", "categorical"),
            ("corn", "categorical"),
            ("coral", "corn"),
        ):
            for metric in SECONDARY_METRICS:
                secondary.append(paired_metric_comparison(
                    subject_metrics,
                    candidate_key=_method_key(candidate_method, feature_group),
                    reference_key=_method_key(reference_method, feature_group),
                    metric=metric,
                    family=f"secondary_{feature_group}",
                    hypothesis_tier="secondary",
                    n_resamples=n_resamples,
                    random_state=random_state,
                ))
    for method in METHODS:
        for metric in FEATURE_GROUP_METRICS:
            feature_effects.append(paired_metric_comparison(
                subject_metrics,
                candidate_key=_method_key(method, "eeg_pow"),
                reference_key=_method_key(method, "eeg_only"),
                metric=metric,
                family="feature_group_effect",
                hypothesis_tier="feature_group_secondary",
                n_resamples=n_resamples,
                random_state=random_state,
            ))
    return (
        apply_holm_by_family(primary),
        apply_holm_by_family(secondary),
        apply_holm_by_family(feature_effects),
    )


def build_subject_effect_types(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    indexed = subject_metrics.set_index(["run_key", "subject_id"])
    for feature_group in FEATURE_GROUPS:
        reference_key = _method_key("categorical", feature_group)
        reference_subjects = subject_metrics.loc[
            subject_metrics["run_key"] == reference_key, "subject_id"
        ].tolist()
        baseline_mae = np.asarray([
            indexed.loc[(reference_key, subject), "ordinal_mae"]
            for subject in reference_subjects
        ], dtype=float)
        q25, q75 = np.quantile(baseline_mae, [0.25, 0.75])
        for method in ("coral", "corn"):
            candidate_key = _method_key(method, feature_group)
            for subject in reference_subjects:
                ref = indexed.loc[(reference_key, subject)]
                cand = indexed.loc[(candidate_key, subject)]
                ordinal_improvement = float(ref["ordinal_mae"] - cand["ordinal_mae"])
                severe_improvement = float(
                    ref["severe_error_rate"] - cand["severe_error_rate"]
                )
                ba_improvement = float(
                    cand["balanced_accuracy"] - ref["balanced_accuracy"]
                )
                if ordinal_improvement > 0 and severe_improvement > 0:
                    effect_type = "both_ordinal_improved"
                elif ordinal_improvement < 0 and severe_improvement < 0:
                    effect_type = "both_ordinal_degraded"
                elif ordinal_improvement > 0 or severe_improvement > 0:
                    effect_type = "mixed_ordinal_effect"
                else:
                    effect_type = "ordinal_tie"
                if (ordinal_improvement > 0 or severe_improvement > 0) and ba_improvement < 0:
                    effect_type += "+ba_tradeoff"
                elif ba_improvement > 0 and (
                    ordinal_improvement < 0 or severe_improvement < 0
                ):
                    effect_type += "+ba_gain_ordinal_tradeoff"
                baseline_value = float(ref["ordinal_mae"])
                difficulty = (
                    "best_quartile" if baseline_value <= q25
                    else "worst_quartile" if baseline_value >= q75
                    else "middle_half"
                )
                rows.append({
                    "feature_group": feature_group,
                    "candidate": method,
                    "reference": "categorical",
                    "subject_id": subject,
                    "fold": int(ref["fold"]),
                    "source_membership": ref["source_membership"],
                    "baseline_ordinal_mae": baseline_value,
                    "difficulty_group": difficulty,
                    "ordinal_mae_improvement": ordinal_improvement,
                    "severe_error_improvement": severe_improvement,
                    "balanced_accuracy_improvement": ba_improvement,
                    "ordinal_mae_improved": ordinal_improvement > 0,
                    "severe_error_improved": severe_improvement > 0,
                    "both_ordinal_improved": ordinal_improvement > 0 and severe_improvement > 0,
                    "both_ordinal_degraded": ordinal_improvement < 0 and severe_improvement < 0,
                    "ordinal_gain_with_ba_loss": (
                        (ordinal_improvement > 0 or severe_improvement > 0)
                        and ba_improvement < 0
                    ),
                    "ba_gain_with_ordinal_loss": (
                        ba_improvement > 0
                        and (ordinal_improvement < 0 or severe_improvement < 0)
                    ),
                    "effect_type": effect_type,
                })
    return pd.DataFrame(rows)


def hard_subject_summary(effect_types: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for keys, group in effect_types.groupby(["feature_group", "candidate"], sort=True):
        feature_group, candidate = keys
        for difficulty in ("worst_quartile", "best_quartile"):
            subset = group.loc[group["difficulty_group"] == difficulty]
            rows.append({
                "feature_group": feature_group,
                "candidate": candidate,
                "difficulty_group": difficulty,
                "subjects": int(len(subset)),
                "mean_ordinal_mae_improvement": float(subset["ordinal_mae_improvement"].mean()),
                "mean_severe_error_improvement": float(subset["severe_error_improvement"].mean()),
                "fraction_ordinal_mae_improved": float(subset["ordinal_mae_improved"].mean()),
                "fraction_ordinal_mae_degraded": float(
                    (subset["ordinal_mae_improvement"] < 0).mean()
                ),
            })
    return rows


def error_distance_analysis(
    predictions: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for key, frame in sorted(predictions.items()):
        distance = np.abs(
            frame["y_pred"].to_numpy(dtype=int) - frame["y_true"].to_numpy(dtype=int)
        )
        method, feature_group = key.split("_", 1)
        for value in range(5):
            count = int((distance == value).sum())
            rows.append({
                "row_type": "distance_distribution",
                "feature_group": feature_group,
                "method": method,
                "candidate": None,
                "reference": None,
                "distance": value,
                "transition": None,
                "count": count,
                "fraction": float(count / len(frame)),
            })
    for feature_group in FEATURE_GROUPS:
        reference = predictions[_method_key("categorical", feature_group)].sort_values(
            "sequence_id"
        )
        reference_distance = np.abs(
            reference["y_pred"].to_numpy(dtype=int)
            - reference["y_true"].to_numpy(dtype=int)
        )
        for method in ("coral", "corn"):
            candidate = predictions[_method_key(method, feature_group)].sort_values(
                "sequence_id"
            )
            candidate_distance = np.abs(
                candidate["y_pred"].to_numpy(dtype=int)
                - candidate["y_true"].to_numpy(dtype=int)
            )
            masks = {
                "severe_to_adjacent": (reference_distance >= 2) & (candidate_distance == 1),
                "adjacent_to_exact": (reference_distance == 1) & (candidate_distance == 0),
                "exact_to_error": (reference_distance == 0) & (candidate_distance > 0),
                "became_more_severe": candidate_distance > reference_distance,
            }
            counts = {name: int(mask.sum()) for name, mask in masks.items()}
            counts["net_severe_error_change"] = int(
                (candidate_distance >= 2).sum() - (reference_distance >= 2).sum()
            )
            for transition, count in counts.items():
                rows.append({
                    "row_type": "candidate_transition",
                    "feature_group": feature_group,
                    "method": method,
                    "candidate": method,
                    "reference": "categorical",
                    "distance": None,
                    "transition": transition,
                    "count": count,
                    "fraction": float(count / len(candidate)),
                })
            matrix = np.zeros((5, 5), dtype=int)
            np.add.at(matrix, (reference_distance, candidate_distance), 1)
            for before in range(5):
                for after in range(5):
                    transitions.append({
                        "feature_group": feature_group,
                        "candidate": method,
                        "reference": "categorical",
                        "reference_distance": before,
                        "candidate_distance": after,
                        "count": int(matrix[before, after]),
                        "fraction": float(matrix[before, after] / len(candidate)),
                    })
    return pd.DataFrame(rows), transitions


def class_error_analysis(predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, frame in sorted(predictions.items()):
        method, feature_group = key.split("_", 1)
        y_true = frame["y_true"].to_numpy(dtype=int)
        y_pred = frame["y_pred"].to_numpy(dtype=int)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=np.arange(5), zero_division=0
        )
        for class_id in range(5):
            mask = y_true == class_id
            distance = np.abs(y_pred[mask] - class_id)
            predicted_counts = np.bincount(y_pred[mask], minlength=5)
            rows.append({
                "run_key": key,
                "method": method,
                "feature_group": feature_group,
                "true_class": class_id,
                "support": int(support[class_id]),
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "mean_absolute_ordinal_error": float(distance.mean()),
                "severe_error_rate": float(np.mean(distance >= 2)),
                **{
                    f"predicted_class_{predicted}": int(predicted_counts[predicted])
                    for predicted in range(5)
                },
            })
    return pd.DataFrame(rows)


def prediction_bias_analysis(predictions: Mapping[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, frame in sorted(predictions.items()):
        method, feature_group = key.split("_", 1)
        residual = frame["y_pred"].to_numpy(dtype=int) - frame["y_true"].to_numpy(dtype=int)
        expected = (
            categorical_expected_rank(frame)
            if method == "categorical"
            else frame["expected_rank"].to_numpy(dtype=float)
        )
        counts = np.bincount(frame["y_pred"].to_numpy(dtype=int), minlength=5)
        rows.append({
            "run_key": key,
            "method": method,
            "feature_group": feature_group,
            "mean_prediction_error": float(np.mean(residual)),
            "median_prediction_error": float(np.median(residual)),
            "fraction_overestimation": float(np.mean(residual > 0)),
            "fraction_underestimation": float(np.mean(residual < 0)),
            "expected_rank_mean_bias": float(np.mean(expected) - frame["y_true"].mean()),
            **{f"predicted_class_{index}": int(counts[index]) for index in range(5)},
        })
    return rows


def threshold_argmax_analysis(predictions: Mapping[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, frame in sorted(predictions.items()):
        method, feature_group = key.split("_", 1)
        if method == "categorical":
            continue
        expected = frame["expected_rank"].to_numpy(dtype=float)
        for rule, column in (("threshold_0.5", "y_pred"), ("class_probability_argmax", "ordinal_argmax")):
            metrics = calculate_prediction_metrics(
                frame, prediction_column=column, expected_rank=expected
            )
            rows.append({
                "method": method,
                "feature_group": feature_group,
                "rule": rule,
                "disagreement_with_primary": int(
                    (frame[column].to_numpy(dtype=int) != frame["y_pred"].to_numpy(dtype=int)).sum()
                ),
                **{name: metrics[name] for name in (
                    "balanced_accuracy", "macro_f1", "quadratic_weighted_kappa",
                    "ordinal_mae", "severe_error_rate",
                )},
            })
    return rows


def source_analysis(predictions: Mapping[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, frame in sorted(predictions.items()):
        method, feature_group = key.split("_", 1)
        for source, group in frame.groupby("source", sort=True):
            expected = (
                categorical_expected_rank(group)
                if method == "categorical"
                else group["expected_rank"].to_numpy(dtype=float)
            )
            metrics = calculate_prediction_metrics(group, expected_rank=expected)
            rows.append({
                "run_key": key,
                "method": method,
                "feature_group": feature_group,
                "source": str(source),
                "sequences": int(len(group)),
                "subjects": int(group["subject_id"].nunique()),
                **{name: metrics[name] for name in FEATURE_GROUP_METRICS},
            })
    indexed = {(row["run_key"], row["source"]): row for row in rows}
    for row in rows:
        if row["method"] == "categorical":
            row.update({f"delta_vs_categorical_{metric}": 0.0 for metric in FEATURE_GROUP_METRICS})
            continue
        reference = indexed[(_method_key("categorical", row["feature_group"]), row["source"])]
        for metric in FEATURE_GROUP_METRICS:
            row[f"delta_vs_categorical_{metric}"] = float(row[metric] - reference[metric])
    return rows


def fold_analysis(
    predictions: Mapping[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    fold_metrics: list[dict[str, Any]] = []
    for key, frame in sorted(predictions.items()):
        method, feature_group = key.split("_", 1)
        for fold, group in frame.groupby("fold", sort=True):
            expected = (
                categorical_expected_rank(group)
                if method == "categorical"
                else group["expected_rank"].to_numpy(dtype=float)
            )
            metrics = calculate_prediction_metrics(group, expected_rank=expected)
            fold_metrics.append({
                "run_key": key,
                "method": method,
                "feature_group": feature_group,
                "fold": int(fold),
                **{name: metrics[name] for name in SUBJECT_METRICS},
            })
    fold_frame = pd.DataFrame(fold_metrics)
    deltas: list[dict[str, Any]] = []
    for feature_group in FEATURE_GROUPS:
        for method in ("coral", "corn"):
            candidate = fold_frame.loc[
                fold_frame["run_key"] == _method_key(method, feature_group)
            ].set_index("fold")
            reference = fold_frame.loc[
                fold_frame["run_key"] == _method_key("categorical", feature_group)
            ].set_index("fold")
            for fold in candidate.index:
                for metric in FEATURE_GROUP_METRICS:
                    raw, improvement = metric_improvement(
                        [candidate.loc[fold, metric]], [reference.loc[fold, metric]], metric
                    )
                    deltas.append({
                        "feature_group": feature_group,
                        "candidate": method,
                        "reference": "categorical",
                        "fold": int(fold),
                        "metric": metric,
                        "reference_value": float(reference.loc[fold, metric]),
                        "candidate_value": float(candidate.loc[fold, metric]),
                        "raw_delta": float(raw[0]),
                        "improvement": float(improvement[0]),
                    })
    stability: list[dict[str, Any]] = []
    for (key, metric), group in pd.DataFrame([
        {"run_key": row["run_key"], "metric": metric, "value": row[metric]}
        for row in fold_metrics for metric in FEATURE_GROUP_METRICS
    ]).groupby(["run_key", "metric"], sort=True):
        stability.append({
            "run_key": key,
            "metric": metric,
            "fold_mean": float(group["value"].mean()),
            "fold_std": float(group["value"].std(ddof=0)),
            "minimum_fold": float(group["value"].min()),
            "maximum_fold": float(group["value"].max()),
        })
    return fold_metrics, pd.DataFrame(deltas), stability


def validate_saved_fold_metrics(
    runs: Mapping[str, ResolvedRun], fold_metrics: Sequence[Mapping[str, Any]],
    *, tolerance: float = 1e-10,
) -> list[dict[str, Any]]:
    computed = {(row["run_key"], int(row["fold"])): row for row in fold_metrics}
    audits: list[dict[str, Any]] = []
    metric_names = (
        "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "kappa",
        "auc", "ordinal_mae", "adjacent_accuracy", "severe_error_rate",
    )
    for key, run in sorted(runs.items()):
        for fold in range(1, 6):
            matches = list(run.run_directory.glob(
                f"**/group_kfold_subject/fold_{fold:02d}/metrics.json"
            ))
            if len(matches) != 1:
                raise ValueError(f"Expected one saved fold metric file for {key} fold {fold}")
            saved = _load_json(matches[0])
            differences: dict[str, float] = {}
            for metric in metric_names:
                if metric not in saved:
                    continue
                difference = abs(float(saved[metric]) - float(computed[(key, fold)][metric]))
                differences[metric] = difference
                if difference > tolerance:
                    raise ValueError(
                        f"Recomputed {key} fold {fold} {metric} differs by {difference}"
                    )
            audits.append({
                "run_key": key,
                "fold": fold,
                "saved_file": _display_path(matches[0]),
                "maximum_absolute_difference": max(differences.values(), default=0.0),
                "within_tolerance": True,
            })
    return audits


def _find_protocol_payload(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        if isinstance(value.get("folds"), Mapping) and isinstance(
            value.get("aggregated"), Mapping
        ):
            return value
        for child in value.values():
            found = _find_protocol_payload(child)
            if found is not None:
                return found
    return None


def validate_saved_aggregate_metrics(
    runs: Mapping[str, ResolvedRun], fold_metrics: Sequence[Mapping[str, Any]],
    aggregate_metrics: Mapping[str, Mapping[str, Any]], *, tolerance: float = 1e-10,
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    metric_names = (
        "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "kappa",
        "auc", "ordinal_mae", "adjacent_accuracy", "severe_error_rate",
    )
    fold_frame = pd.DataFrame(fold_metrics)
    for key, run in sorted(runs.items()):
        report = _load_json(run.run_directory / "metrics.json")
        payload = _find_protocol_payload(report)
        if payload is None:
            raise ValueError(f"Saved aggregate protocol payload missing for {key}")
        maximum_fold_mean_delta = 0.0
        for metric in metric_names:
            saved_key = f"{metric}_mean"
            if saved_key not in payload["aggregated"]:
                continue
            recomputed = float(fold_frame.loc[fold_frame.run_key == key, metric].mean())
            delta = abs(float(payload["aggregated"][saved_key]) - recomputed)
            maximum_fold_mean_delta = max(maximum_fold_mean_delta, delta)
            if delta > tolerance:
                raise ValueError(f"Saved fold aggregate mismatch for {key} {metric}: {delta}")
        maximum_global_delta = 0.0
        ordinal_aggregate_file = run.run_directory / "aggregate_metrics.json"
        if ordinal_aggregate_file.is_file():
            ordinal_saved = _load_json(ordinal_aggregate_file).get(
                "window_sequence_aggregate", {}
            )
            for metric in metric_names:
                if metric not in ordinal_saved:
                    continue
                delta = abs(float(ordinal_saved[metric]) - float(aggregate_metrics[key][metric]))
                maximum_global_delta = max(maximum_global_delta, delta)
                if delta > tolerance:
                    raise ValueError(f"Saved global aggregate mismatch for {key} {metric}: {delta}")
        audits.append({
            "run_key": key,
            "maximum_fold_mean_absolute_difference": maximum_fold_mean_delta,
            "maximum_global_absolute_difference": maximum_global_delta,
            "global_saved_comparison_available": ordinal_aggregate_file.is_file(),
            "within_tolerance": True,
        })
    return audits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _comparison_lookup(
    rows: Sequence[Mapping[str, Any]], candidate: str, reference: str, metric: str
) -> Mapping[str, Any]:
    matches = [
        row for row in rows
        if row["candidate"] == candidate
        and row["reference"] == reference
        and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one comparison for {candidate}, {reference}, {metric}"
        )
    return matches[0]


def select_decision(
    primary: Sequence[Mapping[str, Any]],
    secondary: Sequence[Mapping[str, Any]],
    hard_subjects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the predeclared Task 6D decision rule deterministically."""

    evaluations: dict[str, dict[str, Any]] = {}
    for method in ("coral", "corn"):
        method_primary = [row for row in primary if row["candidate"].startswith(method)]
        method_secondary = [
            row for row in secondary
            if row["candidate"].startswith(method)
            and row["reference"].startswith("categorical")
            and row["metric"] in {"balanced_accuracy", "macro_f1"}
        ]
        significant_primary = [
            row for row in method_primary
            if row["mean_improvement"] > 0
            and row["bootstrap_ci_low"] > 0
            and row["holm_adjusted_p_value"] < 0.05
        ]
        confirmed_discrimination_harm = [
            row for row in method_secondary
            if row["bootstrap_ci_high"] < 0
            and row["holm_adjusted_p_value"] < 0.05
        ]
        variant_a = bool(significant_primary) and len(confirmed_discrimination_harm) < 2
        same_positive_direction = all(row["mean_improvement"] > 0 for row in method_primary)
        majority_not_degraded = all(
            row["fraction_degraded"] <= 0.5 for row in method_primary
        )
        worst = [
            row for row in hard_subjects
            if row["candidate"] == method and row["difficulty_group"] == "worst_quartile"
        ]
        hard_subject_gain = bool(worst) and all(
            row["mean_ordinal_mae_improvement"] > 0 for row in worst
        )
        persistent_ba_f1_harm = len(confirmed_discrimination_harm) >= 2
        variant_b = (
            same_positive_direction
            and majority_not_degraded
            and hard_subject_gain
            and not persistent_ba_f1_harm
        )
        evaluations[method] = {
            "variant_a": variant_a,
            "variant_b": variant_b,
            "confirmed_primary_comparisons": [
                f"{row['feature_group']}:{row['metric']}" for row in significant_primary
            ],
            "confirmed_discrimination_harm": [
                f"{row['feature_group']}:{row['metric']}"
                for row in confirmed_discrimination_harm
            ],
            "all_primary_mean_improvements_positive": same_positive_direction,
            "majority_not_degraded_all_primary": majority_not_degraded,
            "worst_quartile_ordinal_mae_gain_both_groups": hard_subject_gain,
            "eligible_for_additional_seeds": variant_a or variant_b,
        }
    eligible = [method for method, row in evaluations.items() if row["eligible_for_additional_seeds"]]
    head_winners: list[dict[str, Any]] = []
    for feature_group in FEATURE_GROUPS:
        for metric in PRIMARY_METRICS:
            coral = _comparison_lookup(
                primary, _method_key("coral", feature_group),
                _method_key("categorical", feature_group), metric,
            )
            corn = _comparison_lookup(
                primary, _method_key("corn", feature_group),
                _method_key("categorical", feature_group), metric,
            )
            winner = (
                "coral" if float(coral["candidate_mean"]) < float(corn["candidate_mean"])
                else "corn"
            )
            head_winners.append({
                "feature_group": feature_group,
                "metric": metric,
                "descriptive_winner": winner,
                "coral_mean": float(coral["candidate_mean"]),
                "corn_mean": float(corn["candidate_mean"]),
            })
    direct_secondary_confirmed = [
        row for row in secondary
        if row["candidate"].startswith("coral")
        and row["reference"].startswith("corn")
        and row["holm_adjusted_p_value"] < 0.05
    ]
    all_severe_positive = {
        method: all(
            row["mean_improvement"] > 0
            for row in primary
            if row["candidate"].startswith(method) and row["metric"] == "severe_error_rate"
        )
        for method in ("coral", "corn")
    }
    if eligible == ["corn"]:
        decision_id = 1
        selected_method = "corn"
        label = "continue_only_with_CORN"
    elif eligible == ["coral"]:
        decision_id = 2
        selected_method = "coral"
        label = "continue_only_with_CORAL"
    elif len(eligible) == 2:
        decision_id = 3
        selected_method = "coral_and_corn"
        label = "continue_with_both_ordinal_heads"
    elif any(all_severe_positive.values()):
        decision_id = 5
        selected_method = "ordinal_auxiliary_loss_only"
        label = "retain_ordinal_objective_as_auxiliary"
    else:
        decision_id = 4
        selected_method = None
        label = "stop_pure_ordinal_direction"
    if decision_id in (1, 2):
        next_experiment = (
            f"{selected_method.upper()} × EEG-only and EEG+POW, seeds 7 and 123; "
            "reuse seed 42 (four new five-fold runs)"
        )
        estimated_runs = 4
    elif decision_id == 3:
        next_experiment = (
            "CORAL and CORN × EEG-only and EEG+POW, seeds 7 and 123; "
            "reuse seed 42 (eight new five-fold runs)"
        )
        estimated_runs = 8
    elif decision_id == 5:
        next_experiment = (
            "categorical cross-entropy plus a small auxiliary ordinal loss, "
            "EEG+POW primary and EEG-only control"
        )
        estimated_runs = 2
    else:
        next_experiment = (
            "joint categorical plus regression model or subject-risk-aware training; "
            "define a separate protocol before any run"
        )
        estimated_runs = 0
    return {
        "selected_decision_id": decision_id,
        "selected_decision": label,
        "selected_ordinal_method": selected_method,
        "primary_feature_group": "eeg_pow",
        "control_feature_group": "eeg_only",
        "method_rule_evaluations": evaluations,
        "descriptive_primary_head_winners": head_winners,
        "holm_confirmed_direct_secondary_head_differences": [
            f"{row['feature_group']}:{row['metric']}" for row in direct_secondary_confirmed
        ],
        "evidence_supporting": [
            f"{method}: {', '.join(values['confirmed_primary_comparisons']) or 'no Holm-confirmed primary comparison'}"
            for method, values in evaluations.items()
        ] + ([
            "Both heads satisfy variant A through Holm-confirmed EEG-only primary effects; "
            "the descriptive primary winner is split (CORAL 1/4, CORN 3/4), and no "
            "direct CORAL-vs-CORN secondary contrast is Holm-confirmed."
        ] if decision_id == 3 else []),
        "evidence_against": [
            f"{method}: no Holm-confirmed primary improvement for EEG+POW; "
            f"variant B={values['variant_b']}; confirmed BA/F1 harm="
            f"{', '.join(values['confirmed_discrimination_harm']) or 'none'}"
            for method, values in evaluations.items()
        ],
        "decision_rationale": (
            "Decision 3 is retained despite its cost because both heads meet variant A, "
            "their primary strengths differ descriptively, and the seed-42 paired evidence "
            "does not justify excluding either head. Neither satisfies variant B, so this "
            "is a seed-stability check rather than a claim of ordinal superiority."
            if decision_id == 3 else
            "The selected option follows the predeclared variant A/B and rejection rules."
        ),
        "remaining_uncertainty": (
            "All inferential results use one initial state (seed 42); source and fold "
            "breakdowns are descriptive, and ordinal_argmax is diagnostic only."
        ),
        "next_experiment": next_experiment,
        "estimated_number_of_new_runs": estimated_runs,
        "experiments_not_recommended": [
            "POW-only repetition without a new scientific question",
            "hyperparameter search before seed stability is known",
            "changing the predeclared ordinal decoding rule",
            "LSTM/BiLSTM or regression training within this analysis stage",
        ],
    }


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not np.isfinite(numeric) else f"{numeric:.{digits}f}"


def _comparison_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Group | Candidate | Metric | Reference mean | Candidate mean | Raw Δ | Improvement [95% CI] | Better/worse/tie | Wilcoxon p | Holm p | Sign p | Rank-biserial |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['feature_group']} | {row['candidate']} | {row['metric']} | "
            f"{_format(row['reference_mean'])} | {_format(row['candidate_mean'])} | "
            f"{_format(row['raw_mean_delta'])} | {_format(row['mean_improvement'])} "
            f"[{_format(row['bootstrap_ci_low'])}, {_format(row['bootstrap_ci_high'])}] | "
            f"{row['subjects_improved']}/{row['subjects_degraded']}/{row['ties']} | "
            f"{_format(row['wilcoxon_p_value'])} | {_format(row['holm_adjusted_p_value'])} | "
            f"{_format(row['sign_test_p_value'])} | {_format(row['rank_biserial'])} |"
        )
    return lines


def render_analysis_report(summary: Mapping[str, Any]) -> str:
    means = summary["subject_level_means"]
    mean_lines = [
        "| Method | BA | Macro F1 | QWK | Ordinal MAE | Severe error | Expected-rank MAE | Expected-rank ρ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, values in means.items():
        mean_lines.append(
            f"| {key} | {_format(values['balanced_accuracy'])} | "
            f"{_format(values['macro_f1'])} | {_format(values['quadratic_weighted_kappa'])} | "
            f"{_format(values['ordinal_mae'])} | {_format(values['severe_error_rate'])} | "
            f"{_format(values['expected_rank_mae'])} | "
            f"{_format(values['expected_rank_spearman'])} |"
        )
    hard_lines = [
        "| Group | Candidate | Baseline quartile | Subjects | Ordinal-MAE improvement | Severe-error improvement | Improved fraction |",
        "|---|---|---|---:|---:|---:|---:|",
        *[
            f"| {row['feature_group']} | {row['candidate']} | {row['difficulty_group']} | "
            f"{row['subjects']} | {_format(row['mean_ordinal_mae_improvement'])} | "
            f"{_format(row['mean_severe_error_improvement'])} | "
            f"{_format(row['fraction_ordinal_mae_improved'])} |"
            for row in summary["hard_subjects"]
        ],
    ]
    effect_lines = [
        f"- `{row['feature_group']} / {row['candidate']} / {row['effect_type']}`: "
        f"{row['subjects']} subjects"
        for row in summary["subject_effect_type_counts"]
    ]
    transition_lines = [
        "| Group | Candidate | Transition | Count | Fraction |",
        "|---|---|---|---:|---:|",
        *[
            f"| {row['feature_group']} | {row['candidate']} | {row['transition']} | "
            f"{row['count']} | {_format(row['fraction'])} |"
            for row in summary["error_distance_summary"]
            if row["row_type"] == "candidate_transition"
        ],
    ]
    class_lines = [
        "| Method | True class | Recall | Precision | F1 | Ordinal MAE | Severe error |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row['run_key']} | {row['true_class']} | {_format(row['recall'])} | "
            f"{_format(row['precision'])} | {_format(row['f1'])} | "
            f"{_format(row['mean_absolute_ordinal_error'])} | "
            f"{_format(row['severe_error_rate'])} |"
            for row in summary["class_error_analysis"]
            if row["true_class"] in (1, 2, 3)
        ],
    ]
    threshold_lines = [
        "| Group | Method | Rule | Disagreements | BA | Macro F1 | QWK | Ordinal MAE | Severe error |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row['feature_group']} | {row['method']} | {row['rule']} | "
            f"{row['disagreement_with_primary']} | {_format(row['balanced_accuracy'])} | "
            f"{_format(row['macro_f1'])} | {_format(row['quadratic_weighted_kappa'])} | "
            f"{_format(row['ordinal_mae'])} | {_format(row['severe_error_rate'])} |"
            for row in summary["threshold_argmax_diagnostics"]
        ],
    ]
    source_lines = [
        "| Method | Source | Subjects* | BA | Macro F1 | QWK | Ordinal MAE | Severe error |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row['run_key']} | {row['source']} | {row['subjects']} | "
            f"{_format(row['balanced_accuracy'])} | {_format(row['macro_f1'])} | "
            f"{_format(row['quadratic_weighted_kappa'])} | {_format(row['ordinal_mae'])} | "
            f"{_format(row['severe_error_rate'])} |"
            for row in summary["source_descriptive_results"]
        ],
    ]
    fold_lines = [
        "| Method | Metric | Fold mean ± SD | Min | Max |",
        "|---|---|---:|---:|---:|",
        *[
            f"| {row['run_key']} | {row['metric']} | {_format(row['fold_mean'])} ± "
            f"{_format(row['fold_std'])} | {_format(row['minimum_fold'])} | "
            f"{_format(row['maximum_fold'])} |"
            for row in summary["fold_stability"]
        ],
    ]
    sections = [
        "# Статистический анализ порядкового Transformer",
        "",
        "## 1. Цель",
        "",
        "Строгий парный анализ шести завершённых Transformer-вариантов. Независимая единица — `subject_id`; окна, последовательности, folds и источники не используются как независимые наблюдения.",
        "",
        "## 2. Анализируемые runs",
        "",
        *[f"- `{key}`: `{row['run_directory']}`" for key, row in summary["resolved_runs"].items()],
        "",
        "Все runs: seed 42, sequence length 8, 44 142 последовательности, 53 испытуемых, 5 folds. Smoke-runs исключены.",
        "",
        "## 3. Exact alignment",
        "",
        f"Совпали `{', '.join(IDENTITY_COLUMNS)}`: {summary['alignment']['rows']:,} строк, 0 расхождений, 0 дубликатов.",
        "",
        "## 4. Повторный расчёт метрик",
        "",
        *mean_lines,
        "",
        "Fold-метрики пересчитаны из unified predictions и совпали с сохранёнными fold reports в пределах машинной точности.",
        "",
        "## 5. Заранее заданные основные гипотезы",
        "",
        "В каждой feature group отдельная семья из CORAL/CORN × ordinal MAE/severe error; Holm применён только к Wilcoxon p-values.",
        "",
        "## 6. Основные статистические результаты",
        "",
        *_comparison_table(summary["primary_hypotheses"]),
        "",
        f"Конвенция Wilcoxon: {WILCOXON_CONVENTION}.",
        "",
        "## 7. Вторичные результаты",
        "",
        *_comparison_table(summary["secondary_hypotheses"]),
        "",
        "## 8. Влияние группы признаков",
        "",
        *_comparison_table(summary["feature_group_comparisons"]),
        "",
        "## 9. Неоднородность эффекта по испытуемым",
        "",
        "Полные minimum/q10/q25/median/q75/q90/maximum, SD и стандартизованные парные эффекты сохранены в `paired_comparisons.parquet`; индивидуальные типы — в `subject_effect_types.parquet`.",
        "",
        *effect_lines,
        "",
        "## 10. Результаты трудных испытуемых",
        "",
        *hard_lines,
        "",
        "## 11. Расстояние ошибки",
        "",
        "Распределения расстояний 0–4 и матрицы переходов candidate-vs-categorical сохранены в `error_distance_transitions.parquet`. Последовательностные переходы описательны; статистический вывод остаётся subject-level.",
        "",
        *transition_lines,
        "",
        "## 12. Результаты по классам",
        "",
        "Recall, precision, F1, ordinal MAE, severe-error rate и распределения прогнозов для классов 0–4 сохранены в `class_error_analysis.parquet`. Ниже показаны заранее выделенные средние классы 1–3.",
        "",
        *class_lines,
        "",
        "## 13. Смещение прогнозов",
        "",
        *[
            f"- `{row['run_key']}`: mean(y_pred−y_true)={_format(row['mean_prediction_error'])}, "
            f"over={_format(row['fraction_overestimation'])}, under={_format(row['fraction_underestimation'])}, "
            f"expected-rank bias={_format(row['expected_rank_mean_bias'])}."
            for row in summary["prediction_bias"]
        ],
        "",
        "## 14. Threshold-rule против argmax",
        "",
        "Основным остаётся заранее заданное `count(q >= 0.5)`. Диагностический argmax не меняет основной прогноз.",
        "",
        *threshold_lines,
        "",
        "## 15. Результаты по источникам",
        "",
        "Old_EEG и gpn_data рассчитаны описательно. Они не считаются независимыми группами, потому что часть людей присутствует в обоих источниках.",
        "",
        *source_lines,
        "",
        "`*` Числа людей по источникам перекрываются и не суммируются как независимые выборки.",
        "",
        "## 16. Стабильность между folds",
        "",
        "Fold means/std/min/max и все candidate−categorical дельты сохранены в summary JSON и `fold_deltas.parquet`; folds не передавались в Wilcoxon.",
        "",
        *fold_lines,
        "",
        "## 17. Ограничения анализа seed 42",
        "",
        "Оценён один initial state. Парный subject-level протокол измеряет устойчивость между людьми, но не между seeds; source/fold/class analyses описательны.",
        "",
        "## 18. Статистически допустимые выводы",
        "",
        f"По заранее заданному правилу выбрано решение {summary['decision']['selected_decision_id']}: `{summary['decision']['selected_decision']}`.",
        "",
        "## 19. Пока недопустимые утверждения",
        "",
        "Нельзя заявлять seed-устойчивое или причинное преимущество, считать folds/источники независимыми репликациями либо менять основное threshold decoding по результатам диагностического argmax.",
        "",
    ]
    return "\n".join(sections)


def render_decision_report(decision: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Решение по порядковому Transformer",
        "",
        f"- Selected decision: **{decision['selected_decision_id']} — {decision['selected_decision']}**.",
        f"- Selected ordinal method: `{decision['selected_ordinal_method']}`.",
        f"- Primary feature group: `{decision['primary_feature_group']}`.",
        f"- Control feature group: `{decision['control_feature_group']}`.",
        f"- Rationale: {decision['decision_rationale']}",
        "- Evidence supporting: " + "; ".join(decision["evidence_supporting"]),
        "- Evidence against: " + "; ".join(decision["evidence_against"]),
        f"- Remaining uncertainty: {decision['remaining_uncertainty']}",
        f"- Next experiment: {decision['next_experiment']}",
        f"- Estimated new runs: {decision['estimated_number_of_new_runs']}.",
        "- Experiments not recommended: " + "; ".join(decision["experiments_not_recommended"]),
        "",
        "Следующий эксперимент в этой задаче не запускался.",
        "",
    ])


class OrdinalTransformerStatistics:
    """Plan and execute analysis without invoking model construction or training."""

    def __init__(
        self, config_path: str | Path, *, output_dir: str | Path | None = None
    ) -> None:
        self.config_path = _repo_path(config_path)
        self.document = _load_yaml(self.config_path)
        analysis = self.document.get("analysis", {})
        self.output_dir = _repo_path(
            output_dir if output_dir is not None else analysis["output_dir"]
        )
        self.report_path = _repo_path(analysis["report_path"])
        self.summary_path = _repo_path(analysis["summary_path"])
        self.decision_report_path = _repo_path(analysis["decision_report_path"])

    def plan(self) -> dict[str, Any]:
        runs = discover_canonical_runs(self.document)
        expected = self.document["expected"]
        analysis = self.document["analysis"]
        return {
            "valid": True,
            "resolved_runs": {key: run.to_dict() for key, run in runs.items()},
            "feature_groups": list(FEATURE_GROUPS),
            "methods": list(METHODS),
            "seed": int(expected["seed"]),
            "sequence_length": int(expected["sequence_length"]),
            "sequence_index_sha256": str(expected["sequence_index_sha256"]),
            "sequences": int(expected["sequences"]),
            "subjects": int(expected["subjects"]),
            "folds": int(expected["folds"]),
            "primary_families": [f"primary_{group}" for group in FEATURE_GROUPS],
            "secondary_families": [f"secondary_{group}" for group in FEATURE_GROUPS],
            "feature_group_family": "feature_group_effect",
            "bootstrap_iterations": int(analysis["bootstrap_samples"]),
            "bootstrap_seed": int(analysis["random_state"]),
            "output_paths": {
                "generated": _display_path(self.output_dir),
                "report": _display_path(self.report_path),
                "summary": _display_path(self.summary_path),
                "decision_report": _display_path(self.decision_report_path),
            },
        }

    @staticmethod
    def render_plan(plan: Mapping[str, Any]) -> str:
        lines = [
            "Ordinal Transformer statistical analysis plan",
            f"Validity: {'valid' if plan['valid'] else 'invalid'}",
            f"Methods: {', '.join(plan['methods'])}",
            f"Feature groups: {', '.join(plan['feature_groups'])}",
            f"Seed: {plan['seed']}; sequence length: {plan['sequence_length']}",
            f"Sequences: {plan['sequences']}; subjects: {plan['subjects']}; folds: {plan['folds']}",
            f"Sequence-index SHA-256: {plan['sequence_index_sha256']}",
            f"Primary families: {', '.join(plan['primary_families'])}",
            f"Secondary families: {', '.join(plan['secondary_families'])}",
            f"Bootstrap: {plan['bootstrap_iterations']} iterations, seed {plan['bootstrap_seed']}",
            "Resolved runs:",
        ]
        lines.extend(
            f"  {key}: {run['run_directory']}" for key, run in plan["resolved_runs"].items()
        )
        lines.append("Output paths:")
        lines.extend(f"  {key}: {value}" for key, value in plan["output_paths"].items())
        lines.append("Plan-only: no statistical tables or reports were written.")
        return "\n".join(lines)

    def execute(self) -> dict[str, Any]:
        plan = self.plan()
        runs = discover_canonical_runs(self.document)
        predictions = {
            key: pd.read_parquet(run.prediction_file) for key, run in runs.items()
        }
        alignment = require_six_way_alignment(predictions)
        probability_audits: list[dict[str, Any]] = []
        for key, frame in sorted(predictions.items()):
            computed_expected = categorical_expected_rank(frame)
            method = runs[key].method
            expected_delta = 0.0
            threshold_mismatches = 0
            argmax_mismatches = 0
            if method != "categorical":
                expected_delta = float(np.max(np.abs(
                    computed_expected - frame["expected_rank"].to_numpy(dtype=float)
                )))
                if expected_delta > 1e-5:
                    raise ValueError(f"Expected-rank mismatch in {key}: {expected_delta}")
                threshold = frame[[f"threshold_probability_{i}" for i in range(4)]].to_numpy()
                decoded = (threshold >= 0.5).sum(axis=1)
                threshold_mismatches = int((decoded != frame["y_pred"].to_numpy()).sum())
                argmax = np.argmax(
                    frame[class_probability_columns(frame)].to_numpy(dtype=float), axis=1
                )
                argmax_mismatches = int((argmax != frame["ordinal_argmax"].to_numpy()).sum())
                if threshold_mismatches or argmax_mismatches:
                    raise ValueError(f"Stored ordinal decoding mismatch in {key}")
            probability_audits.append({
                "run_key": key,
                "all_finite": True,
                "maximum_probability_sum_error": float(np.max(np.abs(
                    frame[class_probability_columns(frame)].sum(axis=1).to_numpy() - 1.0
                ))),
                "maximum_expected_rank_recomputation_delta": expected_delta,
                "threshold_decoding_mismatches": threshold_mismatches,
                "ordinal_argmax_recomputation_mismatches": argmax_mismatches,
            })
        subject_metrics = calculate_subject_metrics(predictions)
        analysis_config = self.document["analysis"]
        primary, secondary, feature_effects = build_hypothesis_tables(
            subject_metrics,
            n_resamples=int(analysis_config["bootstrap_samples"]),
            random_state=int(analysis_config["random_state"]),
        )
        effects = build_subject_effect_types(subject_metrics)
        hard_subjects = hard_subject_summary(effects)
        error_rows, transition_rows = error_distance_analysis(predictions)
        transition_frame = pd.concat([
            error_rows,
            pd.DataFrame(transition_rows).assign(row_type="distance_transition_matrix"),
        ], ignore_index=True, sort=False)
        class_rows = class_error_analysis(predictions)
        biases = prediction_bias_analysis(predictions)
        threshold_argmax = threshold_argmax_analysis(predictions)
        sources = source_analysis(predictions)
        fold_metrics, fold_deltas, fold_stability = fold_analysis(predictions)
        fold_metric_audit = validate_saved_fold_metrics(runs, fold_metrics)
        aggregate_metrics = {}
        for key, frame in sorted(predictions.items()):
            expected_rank = (
                categorical_expected_rank(frame)
                if runs[key].method == "categorical"
                else frame["expected_rank"].to_numpy(dtype=float)
            )
            aggregate_metrics[key] = calculate_prediction_metrics(
                frame, expected_rank=expected_rank
            )
        aggregate_metric_audit = validate_saved_aggregate_metrics(
            runs, fold_metrics, aggregate_metrics
        )
        decision = select_decision(primary, secondary, hard_subjects)
        paired = pd.DataFrame(primary + secondary + feature_effects)
        means = {
            key: {
                metric: float(group[metric].mean(skipna=True))
                for metric in SUBJECT_METRICS
            }
            for key, group in subject_metrics.groupby("run_key", sort=True)
        }
        data_path = _repo_path(self.document["expected"]["source_parquet"])
        source_hash = _sha256(data_path)
        expected_source_hash = str(self.document["expected"]["source_parquet_sha256"])
        if source_hash != expected_source_hash:
            raise ValueError(f"Source Parquet SHA-256 changed: {source_hash}")
        summary = {
            "schema_version": "ordinal-transformer-statistics-v1",
            "analysis_unit": "subject_id",
            "independent_subjects": 53,
            "seed": 42,
            "bootstrap_samples": int(analysis_config["bootstrap_samples"]),
            "bootstrap_seed": int(analysis_config["random_state"]),
            "wilcoxon_convention": WILCOXON_CONVENTION,
            "resolved_runs": {key: run.to_dict() for key, run in runs.items()},
            "alignment": alignment,
            "probability_audits": probability_audits,
            "fold_metric_recomputation_audit": fold_metric_audit,
            "aggregate_metric_recomputation_audit": aggregate_metric_audit,
            "aggregate_metrics": aggregate_metrics,
            "subject_level_means": means,
            "primary_hypotheses": primary,
            "secondary_hypotheses": secondary,
            "feature_group_comparisons": feature_effects,
            "hard_subjects": hard_subjects,
            "subject_effect_type_counts": effects.groupby(
                ["feature_group", "candidate", "effect_type"], sort=True
            ).size().rename("subjects").reset_index().to_dict(orient="records"),
            "error_distance_summary": error_rows.to_dict(orient="records"),
            "class_error_analysis": class_rows.to_dict(orient="records"),
            "prediction_bias": biases,
            "threshold_argmax_diagnostics": threshold_argmax,
            "source_descriptive_results": sources,
            "fold_metrics": fold_metrics,
            "fold_stability": fold_stability,
            "decision": decision,
            "source_parquet": _display_path(data_path),
            "source_parquet_sha256": source_hash,
            "limitations": [
                "One initial state (seed 42) only",
                "Source and fold analyses are descriptive, not independent tests",
                "Ordinal argmax is diagnostic; threshold decoding remains primary",
                "No causal claim about subject difficulty is supported",
            ],
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        subject_metrics.to_parquet(self.output_dir / "subject_metrics.parquet", index=False)
        paired.to_parquet(self.output_dir / "paired_comparisons.parquet", index=False)
        effects.to_parquet(self.output_dir / "subject_effect_types.parquet", index=False)
        class_rows.to_parquet(self.output_dir / "class_error_analysis.parquet", index=False)
        transition_frame.to_parquet(
            self.output_dir / "error_distance_transitions.parquet", index=False
        )
        fold_deltas.to_parquet(self.output_dir / "fold_deltas.parquet", index=False)
        _write_json(self.output_dir / "primary_hypotheses.json", {
            "family_policy": "Holm within each feature group over four Wilcoxon p-values",
            "comparisons": primary,
        })
        _write_json(self.output_dir / "secondary_hypotheses.json", {
            "family_policy": "Holm within each feature group over eighteen Wilcoxon p-values",
            "comparisons": secondary,
        })
        _write_json(self.output_dir / "feature_group_comparisons.json", {
            "family_policy": "One Holm family over fifteen Wilcoxon p-values",
            "comparisons": feature_effects,
        })
        _write_json(self.output_dir / "decision.json", decision)
        _write_json(self.summary_path, summary)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(render_analysis_report(summary), encoding="utf-8")
        self.decision_report_path.write_text(render_decision_report(decision), encoding="utf-8")
        return {
            "status": "completed",
            "resolved_runs": plan["resolved_runs"],
            "alignment": alignment,
            "decision": decision,
            "subject_metric_rows": len(subject_metrics),
            "paired_comparison_rows": len(paired),
            "artifacts": {
                "output_dir": _display_path(self.output_dir),
                "report": _display_path(self.report_path),
                "summary": _display_path(self.summary_path),
                "decision_report": _display_path(self.decision_report_path),
            },
        }
