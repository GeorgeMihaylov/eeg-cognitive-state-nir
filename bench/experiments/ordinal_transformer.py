"""Technical one-fold smoke orchestration for ordinal Transformer heads."""

from __future__ import annotations

import gc
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from sklearn.model_selection import GroupKFold

from bench.bench_runner import (
    BenchmarkRunner,
    CompletedBenchmarkRun,
    benchmark_config_hash,
)
from bench.datasets.base_eeg_data_loader import (
    feature_list_sha256,
    resolve_feature_columns,
)
from bench.tasks.tasks_registry import get_task
from bench.validation.cross_val import CrossValidator
from model_zoo import build_model
from model_zoo.DL.sequence_utils import build_sequences, sequence_index_sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_HEAD_TYPES = ("categorical", "coral", "corn")
SMOKE_ALIGNMENT_COLUMNS = (
    "sequence_id",
    "outer_fold",
    "subject_id",
    "record_id",
    "source",
    "target_sample_id",
    "target_time",
    "y_true",
    "split",
)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _relative_path(value: str | Path) -> str:
    path = Path(value)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_frame_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> str:
    """Hash selected semantic columns after deterministic sequence sorting."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Smoke subset is missing hash columns: {missing}")
    selected = frame.loc[:, list(columns)].sort_values(
        "sequence_id", kind="mergesort"
    )
    if selected["sequence_id"].isna().any():
        raise ValueError("Smoke subset has missing sequence_id values")
    if selected["sequence_id"].duplicated().any():
        raise ValueError("Smoke subset has duplicate sequence_id values")
    digest = hashlib.sha256()
    for row in selected.itertuples(index=False, name=None):
        normalized = [
            value.item() if isinstance(value, np.generic) else value
            for value in row
        ]
        digest.update(json.dumps(
            normalized,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def audit_prediction_probabilities(
    predictions: pd.DataFrame,
    head_type: str,
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate saved class/threshold probabilities and prediction semantics."""
    normalized_head = str(head_type).strip().lower()
    if normalized_head not in SUPPORTED_HEAD_TYPES:
        raise ValueError(f"Unsupported head_type for probability audit: {head_type!r}")
    class_columns = [
        (
            f"proba_{index}"
            if normalized_head == "categorical"
            else f"class_probability_{index}"
        )
        for index in range(5)
    ]
    missing = sorted(set(class_columns) - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing class probabilities: {missing}")
    probabilities = predictions[class_columns].to_numpy(dtype=np.float64)
    if probabilities.shape != (len(predictions), 5):
        raise ValueError(f"Expected [N,5] class probabilities, got {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("Class probabilities contain NaN or infinite values")
    minimum_probability = float(probabilities.min(initial=np.inf))
    if minimum_probability < -tolerance:
        raise ValueError(
            f"Class probability is negative beyond tolerance: {minimum_probability}"
        )
    sum_error = np.abs(probabilities.sum(axis=1) - 1.0)
    maximum_sum_error = float(sum_error.max(initial=0.0))
    if maximum_sum_error > tolerance:
        raise ValueError(
            "Class probability sum differs from one beyond tolerance: "
            f"{maximum_sum_error}"
        )
    result: dict[str, Any] = {
        "rows": int(len(predictions)),
        "head_type": normalized_head,
        "class_probability_shape": list(probabilities.shape),
        "all_class_probabilities_finite": True,
        "minimum_class_probability": minimum_probability,
        "maximum_class_probability_sum_error": maximum_sum_error,
    }
    if normalized_head == "categorical":
        return result

    threshold_columns = [f"threshold_probability_{index}" for index in range(4)]
    required = {
        *threshold_columns,
        "expected_rank",
        "ordinal_argmax",
        "y_pred",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Ordinal predictions are missing columns: {missing}")
    cumulative = predictions[threshold_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(cumulative).all():
        raise ValueError("Threshold probabilities contain NaN or infinite values")
    monotonicity = cumulative[:, 1:] - cumulative[:, :-1]
    maximum_violation = float(max(0.0, monotonicity.max(initial=0.0)))
    if maximum_violation > tolerance:
        raise ValueError(
            "Threshold probabilities violate monotonicity: "
            f"maximum violation={maximum_violation}"
        )
    expected = predictions["expected_rank"].to_numpy(dtype=np.float64)
    if not np.isfinite(expected).all() or np.any((expected < 0) | (expected > 4)):
        raise ValueError("expected_rank must be finite and lie in [0,4]")
    y_pred = predictions["y_pred"].to_numpy(dtype=np.int64)
    argmax = predictions["ordinal_argmax"].to_numpy(dtype=np.int64)
    if np.any((y_pred < 0) | (y_pred > 4)):
        raise ValueError("Ordinal y_pred falls outside [0,4]")
    if np.any((argmax < 0) | (argmax > 4)):
        raise ValueError("ordinal_argmax falls outside [0,4]")
    recomputed_prediction = (cumulative >= 0.5).sum(axis=1)
    recomputed_expected = cumulative.sum(axis=1)
    prediction_mismatches = int(np.count_nonzero(recomputed_prediction != y_pred))
    expected_delta = float(
        np.max(np.abs(recomputed_expected - expected), initial=0.0)
    )
    if prediction_mismatches:
        raise ValueError(
            f"Saved ordinal y_pred has {prediction_mismatches} threshold-rule mismatches"
        )
    if expected_delta > tolerance:
        raise ValueError(
            f"Saved expected_rank differs from sum(q): maximum delta={expected_delta}"
        )
    raw_probabilities = np.concatenate([
        1.0 - cumulative[:, :1],
        cumulative[:, :-1] - cumulative[:, 1:],
        cumulative[:, -1:],
    ], axis=1)
    result.update({
        "all_threshold_probabilities_finite": True,
        "maximum_monotonicity_violation": maximum_violation,
        "round_off_correction_count": int(
            np.count_nonzero(raw_probabilities < 0.0)
        ),
        "y_pred_recomputation_mismatches": prediction_mismatches,
        "maximum_expected_rank_recomputation_delta": expected_delta,
        "ordinal_argmax_disagreement_count": int(np.count_nonzero(argmax != y_pred)),
        "ordinal_argmax_disagreement_fraction": float(np.mean(argmax != y_pred)),
        "expected_rank_min": float(expected.min(initial=np.inf)),
        "expected_rank_max": float(expected.max(initial=-np.inf)),
    })
    return result


def prediction_alignment(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, Any]:
    """Compare exact outer-test identities without comparing model outputs."""
    identity_columns = (
        "sequence_id",
        "fold",
        "subject_id",
        "record_id",
        "source",
        "target_sample_id",
        "target_time",
        "y_true",
        "split",
    )
    missing_reference = sorted(set(identity_columns) - set(reference.columns))
    missing_candidate = sorted(set(identity_columns) - set(candidate.columns))
    if missing_reference or missing_candidate:
        return {
            "exact_match": False,
            "missing_reference": missing_reference,
            "missing_candidate": missing_candidate,
        }
    left = reference.loc[:, list(identity_columns)].sort_values(
        "sequence_id", kind="mergesort"
    ).reset_index(drop=True)
    right = candidate.loc[:, list(identity_columns)].sort_values(
        "sequence_id", kind="mergesort"
    ).reset_index(drop=True)
    mismatches: dict[str, int] = {}
    if len(left) != len(right):
        mismatches["row_count"] = abs(len(left) - len(right))
    for column in identity_columns:
        if len(left) != len(right):
            mismatches[column] = max(len(left), len(right))
            continue
        if column in {"target_time", "y_true"}:
            mismatch = ~np.isclose(
                left[column].to_numpy(dtype=float),
                right[column].to_numpy(dtype=float),
                equal_nan=True,
            )
        else:
            mismatch = left[column].astype(str) != right[column].astype(str)
        mismatches[column] = int(np.count_nonzero(mismatch))
    return {
        "exact_match": bool(not any(mismatches.values())),
        "reference_rows": int(len(left)),
        "candidate_rows": int(len(right)),
        "mismatches": mismatches,
    }


def load_ordinal_transformer_spec(path: str | Path) -> dict[str, Any]:
    spec_path = _repo_path(path)
    if not spec_path.is_file():
        raise FileNotFoundError(f"Ordinal Transformer experiment not found: {spec_path}")
    document = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    required = {
        "experiment",
        "dataset",
        "task",
        "feature_group",
        "head_types",
        "seeds",
        "model",
        "sequence",
        "validation",
        "evaluation",
        "protocol",
    }
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"Ordinal experiment is missing sections: {missing}")
    heads = tuple(str(value).strip().lower() for value in document["head_types"])
    if heads != SUPPORTED_HEAD_TYPES:
        raise ValueError(
            "Smoke head_types must be exactly categorical, coral, corn in that order"
        )
    seeds = [int(value) for value in document["seeds"]]
    if seeds != [42]:
        raise ValueError("Technical smoke supports only seed 42")
    folds = [int(value) for value in document["evaluation"].get("folds", [])]
    if folds != [1]:
        raise ValueError("Technical smoke supports only outer fold 1")
    max_epochs = int(document["protocol"].get("max_epochs", 3))
    if max_epochs < 1 or max_epochs > 3:
        raise ValueError("Technical smoke max_epochs must be in [1,3]")
    return document


@dataclass(frozen=True)
class OrdinalTransformerTrialPlan:
    trial_id: str
    head_type: str
    feature_group: str
    feature_count: int
    feature_list_sha256: str
    input_shape: tuple[int, int]
    sequence_length: int
    full_sequence_count: int
    full_sequence_index_sha256: str
    smoke_sequence_subset_sha256: str
    outer_fold: int
    outer_train_sequences: int
    train_sequences: int
    validation_sequences: int
    test_sequences: int
    train_subjects: int
    validation_groups: int
    test_subjects: int
    class_counts: Mapping[str, Mapping[str, int]]
    model_parameter_count: int
    maximum_epochs: int
    output_dir: Path
    status: str
    invalid_reasons: tuple[str, ...]
    action: str
    resolved_config: Mapping[str, Any]
    config_hash: str
    completed_run: CompletedBenchmarkRun | None = None

    def to_dict(self, *, include_config: bool = False) -> dict[str, Any]:
        payload = {
            "trial_id": self.trial_id,
            "head_type": self.head_type,
            "feature_group": self.feature_group,
            "feature_count": self.feature_count,
            "feature_list_sha256": self.feature_list_sha256,
            "input_shape": list(self.input_shape),
            "sequence_length": self.sequence_length,
            "full_sequence_count": self.full_sequence_count,
            "full_sequence_index_sha256": self.full_sequence_index_sha256,
            "smoke_sequence_subset_sha256": self.smoke_sequence_subset_sha256,
            "outer_fold": self.outer_fold,
            "outer_train_sequences": self.outer_train_sequences,
            "train_sequences": self.train_sequences,
            "validation_sequences": self.validation_sequences,
            "test_sequences": self.test_sequences,
            "train_subjects": self.train_subjects,
            "validation_groups": self.validation_groups,
            "test_subjects": self.test_subjects,
            "class_counts": _jsonable(self.class_counts),
            "model_parameter_count": self.model_parameter_count,
            "maximum_epochs": self.maximum_epochs,
            "output_directory": _relative_path(self.output_dir),
            "validity_status": self.status,
            "invalid_reasons": list(self.invalid_reasons),
            "action": self.action,
            "reusable_completed_run": (
                None
                if self.completed_run is None
                else _relative_path(self.completed_run.run_directory)
            ),
            "config_hash": self.config_hash,
        }
        if include_config:
            payload["resolved_config"] = _jsonable(self.resolved_config)
        return payload


class OrdinalTransformerSmokeExperiment:
    """Resolve three heads and delegate all training to ``BenchmarkRunner``."""

    def __init__(
        self,
        spec_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        runner_factory: Callable[[dict[str, Any]], Any] = BenchmarkRunner,
        completed_run_finder: Callable[..., CompletedBenchmarkRun | None] = (
            BenchmarkRunner.find_completed_run
        ),
        context_builder: Callable[[], Mapping[str, Any]] | None = None,
        trial_auditor: Callable[
            [OrdinalTransformerTrialPlan, CompletedBenchmarkRun, Any],
            Mapping[str, Any],
        ] | None = None,
    ) -> None:
        self.spec_path = _repo_path(spec_path)
        self.document = load_ordinal_transformer_spec(self.spec_path)
        self.output_root = _repo_path(
            output_dir or self.document["experiment"]["output_dir"]
        )
        self.runner_factory = runner_factory
        self.completed_run_finder = completed_run_finder
        self.context_builder = context_builder
        self.trial_auditor = trial_auditor
        self._context: dict[str, Any] | None = None

    @property
    def data_path(self) -> Path:
        return _repo_path(self.document["dataset"]["data_path"])

    def _build_context(self) -> dict[str, Any]:
        if self._context is not None:
            return self._context
        if self.context_builder is not None:
            self._context = dict(self.context_builder())
            return self._context

        schema = list(pq.ParquetFile(self.data_path).schema.names)
        metadata_columns = [
            "subject_id", "record_id", "source", "t_start", "label_q5"
        ]
        frame = pd.read_parquet(self.data_path, columns=metadata_columns)
        frame.insert(0, "sample_id", np.arange(len(frame), dtype=np.int64))
        supervised = frame.loc[frame["label_q5"].notna()].copy()
        supervised["label_q5"] = supervised["label_q5"].astype(np.int64)
        n_splits = int(self.document["evaluation"].get("n_splits", 5))
        splitter = GroupKFold(n_splits=n_splits)
        supervised["fold"] = 0
        for fold, (_, test_index) in enumerate(splitter.split(
            supervised,
            supervised["label_q5"],
            supervised["subject_id"],
        ), start=1):
            supervised.iloc[
                test_index, supervised.columns.get_loc("fold")
            ] = fold

        sequence = self.document["sequence"]
        built = build_sequences(
            X=np.zeros((len(supervised), 1), dtype=np.float32),
            y=supervised["label_q5"].to_numpy(dtype=np.int64),
            metadata=supervised[[
                "source", "subject_id", "record_id", "sample_id", "t_start"
            ]],
            sequence_length=int(sequence["length"]),
            stride=int(sequence.get("stride", 1)),
            target_position=str(sequence.get("target_position", "last")),
            expected_step_seconds=sequence.get("expected_step_seconds"),
            max_gap_seconds=sequence.get("max_gap_seconds"),
        )
        canonical = built.metadata.copy()
        subject_folds = supervised.drop_duplicates("subject_id").set_index(
            "subject_id"
        )["fold"]
        canonical["fold"] = canonical["subject_id"].map(subject_folds).astype(int)
        canonical["y_true"] = built.y.astype(np.int64)
        canonical = canonical.sort_values(
            ["fold", "sequence_id"], kind="mergesort"
        ).reset_index(drop=True)
        full_hash = sequence_index_sha256(canonical)
        outer_fold = int(self.document["evaluation"]["folds"][0])
        outer_train = canonical.loc[canonical["fold"] != outer_fold].copy()
        outer_test = canonical.loc[canonical["fold"] == outer_fold].copy()

        feature_names = resolve_feature_columns(
            schema, str(self.document["feature_group"]["feature_set"])
        )
        feature_hash = feature_list_sha256(feature_names)
        model_params = deepcopy(self.document["model"]["params"])
        model_params.update({
            "head_type": "categorical",
            "max_epochs": int(self.document["protocol"]["max_epochs"]),
            "random_state": 42,
            "device": "cpu",
        })
        preview = build_model(
            "torch_transformer",
            "classification",
            input_shape=(int(sequence["length"]), len(feature_names)),
            num_outputs=5,
            params=model_params,
        )
        preview.set_validation_groups(
            outer_train["record_group_id"].astype(str).to_numpy(),
            subject_ids=outer_train["subject_id"].astype(str).to_numpy(),
            record_ids=outer_train["record_id"].astype(str).to_numpy(),
            outer_test_record_ids=outer_test["record_id"].astype(str).to_numpy(),
            strategy=str(self.document["validation"]["strategy"]),
            group_column=str(self.document["validation"]["group_column"]),
            validation_size=float(self.document["validation"]["validation_size"]),
            random_state=int(self.document["validation"]["random_state"]),
        )
        outer_train_labels = outer_train["y_true"].to_numpy(dtype=np.int64)
        train_index, validation_index = preview._group_validation_indices(
            outer_train_labels
        )
        validation_summary = preview._validation_summary(
            outer_train_labels, train_index, validation_index
        )

        split_manifest = canonical.copy()
        split_manifest["outer_fold"] = outer_fold
        split_manifest["split"] = "test"
        train_sequence_ids = outer_train.iloc[train_index]["sequence_id"]
        validation_sequence_ids = outer_train.iloc[validation_index]["sequence_id"]
        split_manifest.loc[
            split_manifest["sequence_id"].isin(train_sequence_ids), "split"
        ] = "train"
        split_manifest.loc[
            split_manifest["sequence_id"].isin(validation_sequence_ids), "split"
        ] = "validation"
        subset_hash = stable_frame_sha256(
            split_manifest, SMOKE_ALIGNMENT_COLUMNS
        )

        def counts(values: pd.Series) -> dict[str, int]:
            observed = values.value_counts().sort_index()
            return {str(int(key)): int(value) for key, value in observed.items()}

        self._context = {
            "supervised_rows": int(len(supervised)),
            "canonical": canonical,
            "split_manifest": split_manifest,
            "full_sequence_index_sha256": full_hash,
            "smoke_sequence_subset_sha256": subset_hash,
            "feature_names": feature_names,
            "feature_list_sha256": feature_hash,
            "outer_fold": outer_fold,
            "outer_train_sequences": int(len(outer_train)),
            "train_sequences": int(len(train_index)),
            "validation_sequences": int(len(validation_index)),
            "test_sequences": int(len(outer_test)),
            "train_subjects": int(
                outer_train.iloc[train_index]["subject_id"].nunique()
            ),
            "validation_groups": int(
                len(validation_summary["inner_validation_group_ids"])
            ),
            "test_subjects": int(outer_test["subject_id"].nunique()),
            "test_subject_ids": sorted(
                outer_test["subject_id"].astype(str).unique().tolist()
            ),
            "class_counts": {
                "train": counts(outer_train.iloc[train_index]["y_true"]),
                "validation": counts(
                    outer_train.iloc[validation_index]["y_true"]
                ),
                "test": counts(outer_test["y_true"]),
            },
            "validation_summary": validation_summary,
            "outer_subject_overlap": sorted(
                set(outer_train["subject_id"].astype(str))
                & set(outer_test["subject_id"].astype(str))
            ),
            "source_parquet_sha256": _file_sha256(self.data_path),
            "sequence_build_stats": built.stats,
        }
        return self._context

    def _resolved_config(
        self,
        head_type: str,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        dataset = self.document["dataset"]
        feature = self.document["feature_group"]
        model = self.document["model"]
        trial_id = f"{head_type}_eeg_only"
        params = deepcopy(model["params"])
        params.update({
            "head_type": head_type,
            "num_classes": 5,
            "max_epochs": int(self.document["protocol"]["max_epochs"]),
            "random_state": 42,
        })
        config = {
            "output_dir": str(self.output_root / "runs" / trial_id),
            "datasets": {
                str(dataset["name"]): {
                    "data_path": str(self.data_path),
                    "feature_set": str(feature["feature_set"]),
                    "feature_group": "eeg_only",
                    "target_col": str(dataset["target"]),
                    "subject_col": str(dataset.get("subject_col", "subject_id")),
                    "n_classes": 5,
                    "discretize": False,
                    "max_features": int(feature["feature_count"]),
                    "expected_feature_count": int(feature["feature_count"]),
                    "feature_list_sha256": str(feature["feature_list_sha256"]),
                }
            },
            "tasks": [str(self.document["task"]["benchmark_task"])],
            "models": {
                str(model["name"]): {
                    "type": str(model["type"]),
                    "task_type": "classification",
                    "params": params,
                }
            },
            "sequence": deepcopy(self.document["sequence"]),
            "validation": deepcopy(self.document["validation"]),
            "evaluation": deepcopy(self.document["evaluation"]),
            "task_config": {"random_state": 42},
            "run_within_subject": False,
            "run_loso": False,
            "experiment": {
                "name": str(self.document["experiment"]["name"]),
                "type": "ordinal_transformer_smoke",
                "trial_id": trial_id,
                "head_type": head_type,
                "feature_group": "eeg_only",
                "seed": 42,
                "outer_fold": int(context["outer_fold"]),
                "full_sequence_index_sha256": str(
                    context["full_sequence_index_sha256"]
                ),
                "smoke_sequence_subset_sha256": str(
                    context["smoke_sequence_subset_sha256"]
                ),
                "technical_only": True,
            },
        }
        return config

    def plan(self) -> list[OrdinalTransformerTrialPlan]:
        context = self._build_context()
        invalid_common: list[str] = []
        expected = self.document["dataset"]
        feature = self.document["feature_group"]
        if context["supervised_rows"] != int(expected["expected_supervised_rows"]):
            invalid_common.append("supervised row count mismatch")
        if len(context["canonical"]) != int(expected["expected_sequences"]):
            invalid_common.append("canonical sequence count mismatch")
        if context["full_sequence_index_sha256"] != str(
            expected["sequence_index_sha256"]
        ):
            invalid_common.append("canonical sequence-index hash mismatch")
        if context["source_parquet_sha256"] != str(expected["parquet_sha256"]):
            invalid_common.append("source Parquet hash mismatch")
        if len(context["feature_names"]) != int(feature["feature_count"]):
            invalid_common.append("feature count mismatch")
        if context["feature_list_sha256"] != str(feature["feature_list_sha256"]):
            invalid_common.append("feature-list hash mismatch")
        if context["outer_subject_overlap"]:
            invalid_common.append("outer train/test subject overlap")
        if context["validation_summary"]["group_overlap"]:
            invalid_common.append("inner train/validation group overlap")
        if any(len(counts) != 5 for counts in context["class_counts"].values()):
            invalid_common.append("one or more splits do not contain all five classes")

        plans: list[OrdinalTransformerTrialPlan] = []
        for head_type in SUPPORTED_HEAD_TYPES:
            config = self._resolved_config(head_type, context)
            output = Path(config["output_dir"])
            completed = self.completed_run_finder(
                config, search_directories=[output]
            )
            adapter = build_model(
                "torch_transformer",
                "classification",
                input_shape=(
                    int(self.document["sequence"]["length"]),
                    int(feature["feature_count"]),
                ),
                num_outputs=5,
                params=config["models"]["torch_transformer"]["params"],
            )
            reasons = list(invalid_common)
            plans.append(OrdinalTransformerTrialPlan(
                trial_id=str(config["experiment"]["trial_id"]),
                head_type=head_type,
                feature_group="eeg_only",
                feature_count=int(feature["feature_count"]),
                feature_list_sha256=str(context["feature_list_sha256"]),
                input_shape=(
                    int(self.document["sequence"]["length"]),
                    int(feature["feature_count"]),
                ),
                sequence_length=int(self.document["sequence"]["length"]),
                full_sequence_count=int(len(context["canonical"])),
                full_sequence_index_sha256=str(
                    context["full_sequence_index_sha256"]
                ),
                smoke_sequence_subset_sha256=str(
                    context["smoke_sequence_subset_sha256"]
                ),
                outer_fold=int(context["outer_fold"]),
                outer_train_sequences=int(context["outer_train_sequences"]),
                train_sequences=int(context["train_sequences"]),
                validation_sequences=int(context["validation_sequences"]),
                test_sequences=int(context["test_sequences"]),
                train_subjects=int(context["train_subjects"]),
                validation_groups=int(context["validation_groups"]),
                test_subjects=int(context["test_subjects"]),
                class_counts=deepcopy(context["class_counts"]),
                model_parameter_count=int(
                    adapter.model_metadata["parameter_count"]
                ),
                maximum_epochs=int(self.document["protocol"]["max_epochs"]),
                output_dir=output,
                status="valid" if not reasons else "invalid",
                invalid_reasons=tuple(reasons),
                action="reuse" if completed is not None else "run",
                resolved_config=config,
                config_hash=benchmark_config_hash(config),
                completed_run=completed,
            ))
            del adapter
        return plans

    @staticmethod
    def render_plan(plans: Sequence[OrdinalTransformerTrialPlan]) -> str:
        lines = [
            "# Ordinal Transformer technical smoke plan",
            "",
            "| Trial | Head | Features/hash | Input | Full sequences/hash | Smoke hash | Fold | Train/val/test | Subjects/groups | Classes | Params | Epochs | Output | Reusable | Status |",
            "| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | --- | --- | --- |",
        ]
        for plan in plans:
            lines.append(
                f"| `{plan.trial_id}` | {plan.head_type} | "
                f"{plan.feature_count} / `{plan.feature_list_sha256[:12]}` | "
                f"`{list(plan.input_shape)}` | {plan.full_sequence_count} / "
                f"`{plan.full_sequence_index_sha256[:12]}` | "
                f"`{plan.smoke_sequence_subset_sha256[:12]}` | "
                f"{plan.outer_fold} | {plan.train_sequences}/"
                f"{plan.validation_sequences}/{plan.test_sequences} | "
                f"{plan.train_subjects}/{plan.validation_groups}/"
                f"{plan.test_subjects} | `{json.dumps(plan.class_counts)}` | "
                f"{plan.model_parameter_count} | {plan.maximum_epochs} | "
                f"`{_relative_path(plan.output_dir)}` | "
                f"{'yes' if plan.completed_run else 'no'} | {plan.status} |"
            )
        lines.extend([
            "",
            "No sequence limit is applied: the repository has no safe limiter after "
            "canonical sequence construction. The complete outer fold 1 is used.",
            "Plan-only performs no training and writes no benchmark artifacts.",
        ])
        return "\n".join(lines)

    @staticmethod
    def _fold_result(
        completed: CompletedBenchmarkRun,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        results = json.loads(completed.result_file.read_text(encoding="utf-8"))
        dataset = results["emotiv_cognitive"]
        model = dataset["models"]["cognitive_load_5class"]["torch_transformer"]
        fold = model["group_kfold_subject"]["folds"]["fold_01"]
        return results, fold

    @staticmethod
    def _finite_checkpoint_state(payload: Mapping[str, Any]) -> bool:
        return all(
            not torch.is_floating_point(value) or bool(torch.isfinite(value).all())
            for value in payload["model_state_dict"].values()
        )

    def _rebuild_test_split(self, config: Mapping[str, Any]) -> Any:
        runner = BenchmarkRunner(deepcopy(dict(config)))
        dataset_name = next(iter(config["datasets"]))
        data = runner.load_dataset(dataset_name)
        task_name = str(config["tasks"][0])
        task = get_task(task_name, data, dict(config.get("task_config", {})))
        splits = CrossValidator(task).run_group_kfold(
            group_column=str(config["evaluation"]["group_column"]),
            n_splits=int(config["evaluation"]["n_splits"]),
            random_state=int(config["evaluation"].get("random_state", 42)),
        )
        return runner._build_sequence_split(splits["fold_01"])

    def _audit_trial(
        self,
        plan: OrdinalTransformerTrialPlan,
        completed: CompletedBenchmarkRun,
        split: Any,
    ) -> dict[str, Any]:
        _, fold = self._fold_result(completed)
        artifacts = {key: Path(value) for key, value in fold["artifacts"].items()}
        predictions = pd.read_parquet(artifacts["predictions"])
        probability = audit_prediction_probabilities(
            predictions, plan.head_type
        )
        training_log = pd.read_csv(artifacts["training_log"])
        for column in ("train_loss", "validation_loss", "learning_rate"):
            if column not in training_log or not np.isfinite(
                training_log[column].to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    f"Training log has a missing or non-finite {column!r} column"
                )
        if len(training_log) > plan.maximum_epochs:
            raise ValueError("Training exceeded the smoke maximum epoch count")
        if len(training_log) > 1 and float(
            np.ptp(training_log["train_loss"].to_numpy(dtype=float))
        ) == 0.0:
            raise ValueError("Training loss is exactly constant across all epochs")

        config = yaml.safe_load(
            (completed.run_directory / "config.yaml").read_text(encoding="utf-8")
        )
        configured_head = config["models"]["torch_transformer"]["params"].get(
            "head_type", "categorical"
        )
        if configured_head != plan.head_type:
            raise ValueError("Resolved config does not preserve head_type")
        model = build_model(
            "torch_transformer",
            "classification",
            input_shape=tuple(split.X_test.shape[1:]),
            num_outputs=5,
            params=config["models"]["torch_transformer"]["params"],
        )
        initial_state = {
            key: value.detach().cpu().clone()
            for key, value in model.model.state_dict().items()
        }
        try:
            checkpoint = torch.load(
                artifacts["model"], map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(artifacts["model"], map_location="cpu")
        if checkpoint.get("head_type", "categorical") != plan.head_type:
            raise ValueError("Checkpoint head_type does not match the trial")
        if not self._finite_checkpoint_state(checkpoint):
            raise ValueError("Checkpoint contains non-finite model parameters")
        model.load(artifacts["model"])
        detailed = model.predict_detailed(split.X_test)
        reloaded = pd.DataFrame({
            "sequence_id": np.asarray(split.row_metadata_test["sequence_id"]).astype(str),
            "y_pred_reloaded": detailed["y_pred"],
        })
        for index in range(5):
            reloaded[f"proba_{index}_reloaded"] = detailed[
                "class_probabilities"
            ][:, index]
        if plan.head_type != "categorical":
            for index in range(4):
                reloaded[f"threshold_{index}_reloaded"] = detailed[
                    "threshold_probabilities"
                ][:, index]
            reloaded["expected_rank_reloaded"] = detailed["expected_rank"]
        compared = predictions.merge(
            reloaded, on="sequence_id", how="outer", validate="one_to_one",
            indicator=True,
        )
        if not compared["_merge"].eq("both").all():
            raise ValueError("Reloaded checkpoint prediction membership differs")
        y_mismatches = int(np.count_nonzero(
            compared["y_pred"].to_numpy(dtype=int)
            != compared["y_pred_reloaded"].to_numpy(dtype=int)
        ))
        class_probability_delta = float(max(
            np.max(np.abs(
                compared[f"proba_{index}"].to_numpy(dtype=float)
                - compared[f"proba_{index}_reloaded"].to_numpy(dtype=float)
            ), initial=0.0)
            for index in range(5)
        ))
        threshold_probability_delta = None
        expected_rank_delta = None
        if plan.head_type != "categorical":
            threshold_probability_delta = float(max(
                np.max(np.abs(
                    compared[f"threshold_probability_{index}"].to_numpy(dtype=float)
                    - compared[f"threshold_{index}_reloaded"].to_numpy(dtype=float)
                ), initial=0.0)
                for index in range(4)
            ))
            expected_rank_delta = float(np.max(np.abs(
                compared["expected_rank"].to_numpy(dtype=float)
                - compared["expected_rank_reloaded"].to_numpy(dtype=float)
            ), initial=0.0))
        if y_mismatches or class_probability_delta > 1e-7:
            raise ValueError("Checkpoint reload changed categorical predictions")
        if (
            threshold_probability_delta is not None
            and threshold_probability_delta > 1e-7
        ):
            raise ValueError("Checkpoint reload changed threshold probabilities")
        if expected_rank_delta is not None and expected_rank_delta > 1e-7:
            raise ValueError("Checkpoint reload changed expected ranks")

        head_prefix = (
            "classifier." if plan.head_type == "categorical" else "ordinal_head."
        )
        changed_head_parameters = 0
        maximum_head_parameter_delta = 0.0
        for key, trained in checkpoint["model_state_dict"].items():
            if not key.startswith(head_prefix) or not torch.is_floating_point(trained):
                continue
            delta = float(torch.max(torch.abs(
                trained.detach().cpu() - initial_state[key]
            )).item())
            maximum_head_parameter_delta = max(maximum_head_parameter_delta, delta)
            changed_head_parameters += int(delta > 0.0)
        if changed_head_parameters == 0:
            raise ValueError("No output-head parameter changed during training")

        head_diagnostics = dict(
            fold["training"].get("head_diagnostics", {})
        )
        objective_diagnostics = dict(
            fold["training"].get("objective_training_diagnostics", {})
        )
        if plan.head_type == "coral":
            cutpoints = [head_diagnostics[f"cutpoint_{index}"] for index in range(4)]
            if not np.all(np.diff(cutpoints) > 0):
                raise ValueError("Saved CORAL cutpoints are not strictly increasing")
        if plan.head_type == "corn":
            risk_counts = [objective_diagnostics[f"risk_count_{index}"] for index in range(4)]
            if risk_counts[0] <= 0 or not np.all(np.diff(risk_counts) <= 0):
                raise ValueError("Saved CORN risk counts are invalid")

        required_metrics = {
            "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
            "auc", "kappa", "quadratic_weighted_kappa", "ordinal_mae",
            "adjacent_accuracy", "severe_error_rate",
        }
        if plan.head_type != "categorical":
            required_metrics.update({
                "expected_rank_mae", "expected_rank_spearman"
            })
        missing_metrics = sorted(required_metrics - set(fold["metrics"]))
        if missing_metrics:
            raise ValueError(f"Smoke metrics are missing: {missing_metrics}")
        nonfinite_metrics = sorted(
            name for name in required_metrics
            if not np.isfinite(float(fold["metrics"][name]))
        )
        if nonfinite_metrics:
            raise ValueError(f"Smoke metrics are non-finite: {nonfinite_metrics}")

        validation_split = json.loads(
            artifacts["validation_split"].read_text(encoding="utf-8")
        )
        if validation_split["group_overlap"]:
            raise ValueError("Inner train/validation groups overlap")
        if validation_split["outer_test_record_overlap"]:
            raise ValueError("Inner data overlap outer-test records")
        if fold["split_metadata"].get("subject_overlap"):
            raise ValueError("Outer train/test subjects overlap")

        diagnostics = {
            "trial_id": plan.trial_id,
            "head_type": plan.head_type,
            "run_directory": str(completed.run_directory),
            "artifacts": {key: str(value) for key, value in artifacts.items()},
            "epochs_trained": int(len(training_log)),
            "training_log": training_log.to_dict(orient="records"),
            "best_epoch": int(fold["training"]["best_epoch"]),
            "best_validation_loss": float(
                fold["training"]["best_validation_loss"]
            ),
            "training_time_seconds": float(fold["training_time"]),
            "device": str(fold["training"]["device"]),
            "device_name": str(fold["training"]["device_name"]),
            "parameter_count": int(fold["training"]["trainable_parameter_count"]),
            "head_diagnostics": head_diagnostics,
            "objective_training_diagnostics": objective_diagnostics,
            "probability_validation": probability,
            "metrics": fold["metrics"],
            "checkpoint_reload": {
                "strict_load": True,
                "y_pred_mismatches": y_mismatches,
                "maximum_class_probability_delta": class_probability_delta,
                "maximum_threshold_probability_delta": threshold_probability_delta,
                "maximum_expected_rank_delta": expected_rank_delta,
            },
            "checkpoint": {
                "head_type": checkpoint.get("head_type", "categorical"),
                "all_parameters_finite": True,
                "changed_output_head_parameters": changed_head_parameters,
                "maximum_output_head_parameter_delta": maximum_head_parameter_delta,
                "categorical_classifier_keys_preserved": (
                    sorted(
                        key for key in checkpoint["model_state_dict"]
                        if key.startswith("classifier.")
                    )
                    if plan.head_type == "categorical"
                    else None
                ),
            },
            "validation_split": validation_split,
        }
        artifact_dir = artifacts["predictions"].parent
        probability_path = artifact_dir / "probability_validation_summary.json"
        checkpoint_path = artifact_dir / "checkpoint_reload_audit.json"
        head_path = artifact_dir / "head_diagnostics.json"
        fold_manifest_path = artifact_dir / "fold_manifest.json"
        _write_json(probability_path, probability)
        _write_json(checkpoint_path, diagnostics["checkpoint_reload"])
        _write_json(head_path, {
            "head_type": plan.head_type,
            "head_diagnostics": head_diagnostics,
            "objective_training_diagnostics": objective_diagnostics,
        })
        diagnostics["technical_artifacts"] = {
            "probability_validation": str(probability_path),
            "checkpoint_reload_audit": str(checkpoint_path),
            "head_diagnostics": str(head_path),
            "fold_manifest": str(fold_manifest_path),
        }
        _write_json(fold_manifest_path, {
            "schema_version": 1,
            "status": "completed",
            "trial_id": plan.trial_id,
            "head_type": plan.head_type,
            "outer_fold": plan.outer_fold,
            "smoke_sequence_subset_sha256": (
                plan.smoke_sequence_subset_sha256
            ),
            "split_metadata": fold["split_metadata"],
            "training": fold["training"],
            "metrics": fold["metrics"],
            "standard_artifacts": diagnostics["artifacts"],
            "technical_artifacts": diagnostics["technical_artifacts"],
            "probability_validation": probability,
            "checkpoint_reload": diagnostics["checkpoint_reload"],
        })
        with (completed.run_directory / "resolved_config.yaml").open(
            "w", encoding="utf-8"
        ) as output:
            yaml.safe_dump(_jsonable(config), output, sort_keys=False)
        _write_json(
            completed.run_directory / "ordinal_smoke_trial_manifest.json",
            diagnostics,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return diagnostics

    def _combined_audit(
        self,
        plans: Sequence[OrdinalTransformerTrialPlan],
        completed: Mapping[str, CompletedBenchmarkRun],
        audits: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        prediction_frames: dict[str, pd.DataFrame] = {}
        normalizations: dict[str, dict[str, Any]] = {}
        validations: dict[str, dict[str, Any]] = {}
        for plan in plans:
            _, fold = self._fold_result(completed[plan.trial_id])
            artifacts = {key: Path(value) for key, value in fold["artifacts"].items()}
            prediction_frames[plan.head_type] = pd.read_parquet(
                artifacts["predictions"]
            )
            normalizations[plan.head_type] = json.loads(
                artifacts["normalization_stats"].read_text(encoding="utf-8")
            )
            validations[plan.head_type] = json.loads(
                artifacts["validation_split"].read_text(encoding="utf-8")
            )
        reference = prediction_frames["categorical"]
        alignments = {
            head: prediction_alignment(reference, frame)
            for head, frame in prediction_frames.items()
        }
        normalization_deltas: dict[str, Any] = {}
        reference_normalization = normalizations["categorical"]
        for head, normalization in normalizations.items():
            mean_delta = float(np.max(np.abs(
                np.asarray(normalization["mean"], dtype=float)
                - np.asarray(reference_normalization["mean"], dtype=float)
            ), initial=0.0))
            scale_delta = float(np.max(np.abs(
                np.asarray(normalization["scale"], dtype=float)
                - np.asarray(reference_normalization["scale"], dtype=float)
            ), initial=0.0))
            normalization_deltas[head] = {
                "mean_max_abs_delta": mean_delta,
                "scale_max_abs_delta": scale_delta,
                "feature_order_equal": (
                    normalization["feature_names"]
                    == reference_normalization["feature_names"]
                ),
            }
        validation_equal = {
            head: validation == validations["categorical"]
            for head, validation in validations.items()
        }
        subset_hashes = {
            plan.head_type: plan.smoke_sequence_subset_sha256 for plan in plans
        }
        ready = bool(
            all(value["exact_match"] for value in alignments.values())
            and all(value["mean_max_abs_delta"] == 0.0 for value in normalization_deltas.values())
            and all(value["scale_max_abs_delta"] == 0.0 for value in normalization_deltas.values())
            and all(value["feature_order_equal"] for value in normalization_deltas.values())
            and all(validation_equal.values())
            and len(set(subset_hashes.values())) == 1
            and all(
                audit["checkpoint_reload"]["y_pred_mismatches"] == 0
                for audit in audits.values()
            )
        )
        return {
            "status": "completed" if ready else "invalid",
            "technical_only": True,
            "scientific_quality_claim": False,
            "ready_for_full_experiment": ready,
            "sequence_alignment": alignments,
            "normalization_deltas": normalization_deltas,
            "validation_splits_equal": validation_equal,
            "smoke_subset_hashes": subset_hashes,
            "all_smoke_subset_hashes_equal": len(set(subset_hashes.values())) == 1,
            "trials": _jsonable(audits),
        }

    def execute(
        self,
        plans: Sequence[OrdinalTransformerTrialPlan],
        *,
        resume: bool,
    ) -> dict[str, Any]:
        invalid = [plan for plan in plans if plan.status != "valid"]
        if invalid:
            raise ValueError(
                "Invalid ordinal smoke trials: "
                + "; ".join(
                    f"{plan.trial_id}: {', '.join(plan.invalid_reasons)}"
                    for plan in invalid
                )
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        context = self._build_context()
        split_manifest_path = self.output_root / "smoke_sequence_split.parquet"
        context["split_manifest"].to_parquet(split_manifest_path, index=False)
        completed: dict[str, CompletedBenchmarkRun] = {}
        outcomes: list[dict[str, Any]] = []
        for plan in plans:
            existing = self.completed_run_finder(
                plan.resolved_config, search_directories=[plan.output_dir]
            )
            if resume and existing is not None:
                completed[plan.trial_id] = existing
                outcomes.append({**plan.to_dict(), "outcome": "resumed"})
                continue
            runner = self.runner_factory(deepcopy(dict(plan.resolved_config)))
            runner.run()
            completed_run = runner.completed_run()
            completed[plan.trial_id] = completed_run
            outcomes.append({**plan.to_dict(), "outcome": "completed"})
            del runner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        audits: dict[str, Mapping[str, Any]] = {}
        split = None
        if self.trial_auditor is None:
            split = self._rebuild_test_split(plans[0].resolved_config)
        for plan in plans:
            auditor = self.trial_auditor or self._audit_trial
            audits[plan.trial_id] = dict(
                auditor(plan, completed[plan.trial_id], split)
            )
        if self.trial_auditor is None:
            combined = self._combined_audit(plans, completed, audits)
        else:
            combined = {
                "status": "completed",
                "technical_only": True,
                "ready_for_full_experiment": True,
                "trials": _jsonable(audits),
            }
        manifest = {
            "experiment": str(self.document["experiment"]["name"]),
            "source_parquet_sha256": str(context["source_parquet_sha256"]),
            "full_sequence_index_sha256": str(
                context["full_sequence_index_sha256"]
            ),
            "smoke_sequence_subset_sha256": str(
                context["smoke_sequence_subset_sha256"]
            ),
            "sequence_split_manifest": str(split_manifest_path),
            "outcomes": outcomes,
            "audit": combined,
        }
        _write_json(self.output_root / "ordinal_transformer_smoke_manifest.json", manifest)
        return manifest


__all__ = [
    "OrdinalTransformerSmokeExperiment",
    "OrdinalTransformerTrialPlan",
    "SMOKE_ALIGNMENT_COLUMNS",
    "audit_prediction_probabilities",
    "load_ordinal_transformer_spec",
    "prediction_alignment",
    "stable_frame_sha256",
]
