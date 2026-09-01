"""Cross-fitted sensitivity audit for the legacy global ``label_q5``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import yaml

from bench.analysis.diagnostic_baselines import (
    align_with_canonical_predictions,
    assign_subject_folds,
    run_diagnostic_baselines,
)
from bench.analysis.label_target_audit import (
    _jsonable,
    _repo_path,
    _sha256_file,
    _write_json,
)
from bench.analysis.temporal_target_structure import (
    RECORD_COLUMNS,
    calculate_temporal_statistics,
    prepare_temporal_frame,
    previous_label_predictions,
    summarize_predictions,
)


SAFE_LABEL_COLUMN = "fold_train_quantile_label_q5"
GLOBAL_LABEL_COLUMN = "global_label_q5"
QUANTILES = (0.2, 0.4, 0.6, 0.8)
QUANTILE_NAMES = ("q20", "q40", "q60", "q80")
REQUIRED_COLUMNS = [
    "source",
    "subject_id",
    "record_id",
    "t_start",
    "t_end",
    "target_focus",
    "label_q5",
]


def apply_finite_thresholds(
    values: Sequence[float] | np.ndarray | pd.Series,
    thresholds: Sequence[float],
    *,
    n_classes: int = 5,
) -> np.ndarray:
    """Apply right-closed ``[-inf, q20, ..., +inf]`` bins to finite values."""

    array = np.asarray(values, dtype=float)
    edges = np.asarray(thresholds, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("All target values must be finite before thresholding")
    if len(edges) != n_classes - 1:
        raise ValueError(
            f"Expected {n_classes - 1} internal thresholds, got {len(edges)}"
        )
    if not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
        raise ValueError("Train-derived thresholds must be finite and strictly increasing")
    labels = np.searchsorted(edges, array, side="left").astype(np.int64)
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= n_classes:
        raise RuntimeError("Threshold application produced an out-of-range class")
    return labels


def _class_counts(labels: np.ndarray, n_classes: int) -> dict[str, int]:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_classes)
    return {str(class_id): int(counts[class_id]) for class_id in range(n_classes)}


def _class_fractions(labels: np.ndarray, n_classes: int) -> dict[str, float]:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_classes)
    fractions = counts / counts.sum()
    return {
        str(class_id): float(fractions[class_id]) for class_id in range(n_classes)
    }


def _balance_summary(labels: np.ndarray, n_classes: int) -> dict[str, Any]:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_classes)
    fractions = counts / counts.sum()
    uniform = np.full(n_classes, 1.0 / n_classes)
    return {
        "counts": _class_counts(labels, n_classes),
        "fractions": _class_fractions(labels, n_classes),
        "all_classes_present": bool(np.all(counts > 0)),
        "missing_classes": np.flatnonzero(counts == 0).astype(int).tolist(),
        "minimum_class_count": int(counts.min()),
        "minimum_class_fraction": float(fractions.min()),
        "maximum_absolute_deviation_from_uniform": float(
            np.max(np.abs(fractions - uniform))
        ),
        "total_variation_distance_from_uniform": float(
            0.5 * np.sum(np.abs(fractions - uniform))
        ),
    }


def build_cross_fitted_labels(
    folded: pd.DataFrame,
    *,
    target_col: str = "target_focus",
    global_label_col: str = "label_q5",
    fold_col: str = "outer_fold",
    global_thresholds: Sequence[float],
    n_classes: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit thresholds on outer-train targets and label each outer-test row once."""

    required = {
        target_col,
        global_label_col,
        fold_col,
        "sample_id",
        "subject_id",
        "record_id",
        "source",
    }
    missing = sorted(required - set(folded.columns))
    if missing:
        raise ValueError(f"Cross-fitted label input is missing columns: {missing}")
    global_edges = np.asarray(global_thresholds, dtype=float)
    if len(global_edges) != n_classes - 1:
        raise ValueError("Global threshold count does not match n_classes")
    if not folded["sample_id"].is_unique:
        raise ValueError("Every supervised sample_id must be unique")

    parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    fold_values = sorted(folded[fold_col].astype(int).unique().tolist())
    for fold in fold_values:
        train = folded.loc[folded[fold_col] != fold].copy()
        test = folded.loc[folded[fold_col] == fold].copy()
        train_subjects = sorted(train["subject_id"].astype(str).unique().tolist())
        test_subjects = sorted(test["subject_id"].astype(str).unique().tolist())
        overlap = sorted(set(train_subjects) & set(test_subjects))
        if overlap:
            raise RuntimeError(f"Subject overlap in fold {fold}: {overlap}")
        train_values = train[target_col].to_numpy(dtype=float)
        test_values = test[target_col].to_numpy(dtype=float)
        thresholds = np.quantile(train_values, QUANTILES, method="linear")
        unique_thresholds = int(len(np.unique(thresholds)))
        valid = bool(
            np.isfinite(thresholds).all()
            and unique_thresholds == n_classes - 1
            and np.all(np.diff(thresholds) > 0)
        )
        row: dict[str, Any] = {
            "fold": int(fold),
            "status": "valid" if valid else "invalid_non_unique_thresholds",
            "threshold_fit_partition": "outer_train_only",
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "subject_overlap": overlap,
            "train_windows": int(len(train)),
            "test_windows": int(len(test)),
            "unique_internal_thresholds": unique_thresholds,
            "duplicates_drop_would_reduce_classes": unique_thresholds < n_classes - 1,
        }
        for name, value, global_value in zip(
            QUANTILE_NAMES, thresholds, global_edges
        ):
            row[name] = float(value)
            row[f"delta_{name}_vs_global"] = float(value - global_value)
        if not valid:
            row["train_class_balance"] = None
            row["test_class_balance"] = None
            fold_rows.append(row)
            continue

        train_labels = apply_finite_thresholds(
            train_values, thresholds, n_classes=n_classes
        )
        test_labels = apply_finite_thresholds(
            test_values, thresholds, n_classes=n_classes
        )
        row["train_class_balance"] = _balance_summary(train_labels, n_classes)
        row["test_class_balance"] = _balance_summary(test_labels, n_classes)
        fold_rows.append(row)
        test[GLOBAL_LABEL_COLUMN] = test[global_label_col].astype(np.int64)
        test[SAFE_LABEL_COLUMN] = test_labels
        test["label_changed"] = (
            test[GLOBAL_LABEL_COLUMN] != test[SAFE_LABEL_COLUMN]
        )
        test["absolute_label_shift"] = (
            test[SAFE_LABEL_COLUMN] - test[GLOBAL_LABEL_COLUMN]
        ).abs().astype(np.int64)
        parts.append(test)

    invalid = [row for row in fold_rows if row["status"] != "valid"]
    threshold_result = {
        "quantiles": list(QUANTILES),
        "quantile_method": "numpy.quantile(method='linear')",
        "application_bins": "[-inf, q20], (q20, q40], ..., (q80, +inf]",
        "global_internal_thresholds": {
            name: float(value) for name, value in zip(QUANTILE_NAMES, global_edges)
        },
        "folds": fold_rows,
        "all_folds_valid": not invalid,
        "invalid_folds": [int(row["fold"]) for row in invalid],
    }
    if invalid:
        return pd.DataFrame(), threshold_result
    cross_fitted = pd.concat(parts, ignore_index=True, sort=False)
    if len(cross_fitted) != len(folded):
        raise RuntimeError(
            f"Cross-fitted rows {len(cross_fitted)} != supervised rows {len(folded)}"
        )
    if not cross_fitted["sample_id"].is_unique:
        raise RuntimeError("A supervised sample received more than one test label")
    if set(cross_fitted["sample_id"]) != set(folded["sample_id"]):
        raise RuntimeError("Cross-fitted labels do not cover every supervised sample")
    observed = sorted(cross_fitted[SAFE_LABEL_COLUMN].unique().astype(int).tolist())
    if observed != list(range(n_classes)):
        raise RuntimeError(f"Cross-fitted test labels have classes {observed}")
    return cross_fitted.sort_values("sample_id", kind="mergesort").reset_index(
        drop=True
    ), threshold_result


def summarize_thresholds(
    threshold_result: Mapping[str, Any],
) -> dict[str, Any]:
    if not threshold_result["all_folds_valid"]:
        return {"status": "invalid", "invalid_folds": threshold_result["invalid_folds"]}
    folds = threshold_result["folds"]
    global_values = threshold_result["global_internal_thresholds"]
    summary: dict[str, Any] = {"status": "valid"}
    for name in QUANTILE_NAMES:
        values = np.asarray([row[name] for row in folds], dtype=float)
        deltas = values - float(global_values[name])
        summary[name] = {
            "global": float(global_values[name]),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
            "maximum_absolute_deviation_from_global": float(
                np.max(np.abs(deltas))
            ),
        }
    return summary


def label_comparison_metrics(
    frame: pd.DataFrame,
    *,
    global_col: str = GLOBAL_LABEL_COLUMN,
    safe_col: str = SAFE_LABEL_COLUMN,
    n_classes: int = 5,
) -> dict[str, Any]:
    global_labels = frame[global_col].to_numpy(dtype=int)
    safe_labels = frame[safe_col].to_numpy(dtype=int)
    signed_shift = safe_labels - global_labels
    absolute_shift = np.abs(signed_shift)
    changed = absolute_shift > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kappa = float(cohen_kappa_score(global_labels, safe_labels))
        quadratic_kappa = float(
            cohen_kappa_score(global_labels, safe_labels, weights="quadratic")
        )
    return {
        "windows": int(len(frame)),
        "agreement_fraction": float(np.mean(~changed)),
        "changed_fraction": float(np.mean(changed)),
        "changed_windows": int(changed.sum()),
        "mean_absolute_class_shift": float(np.mean(absolute_shift)),
        "one_class_shift_fraction": float(np.mean(absolute_shift == 1)),
        "two_or_more_class_shift_fraction": float(np.mean(absolute_shift >= 2)),
        "one_class_fraction_among_changed": (
            float(np.mean(absolute_shift[changed] == 1)) if changed.any() else 0.0
        ),
        "two_or_more_fraction_among_changed": (
            float(np.mean(absolute_shift[changed] >= 2)) if changed.any() else 0.0
        ),
        "signed_shift_counts": {
            str(int(value)): int((signed_shift == value).sum())
            for value in sorted(np.unique(signed_shift))
        },
        "absolute_shift_counts": {
            str(int(value)): int((absolute_shift == value).sum())
            for value in sorted(np.unique(absolute_shift))
        },
        "transition_counts_global_to_safe": confusion_matrix(
            global_labels, safe_labels, labels=list(range(n_classes))
        ).tolist(),
        "cohens_kappa": kappa,
        "quadratic_weighted_kappa": quadratic_kappa,
    }


def comparison_by_group(
    frame: pd.DataFrame,
    group_columns: str | Sequence[str],
    *,
    n_classes: int = 5,
) -> list[dict[str, Any]]:
    columns = [group_columns] if isinstance(group_columns, str) else list(group_columns)
    grouping: str | list[str] = columns[0] if len(columns) == 1 else columns
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(grouping, sort=True, observed=True):
        key_values = (keys,) if len(columns) == 1 else keys
        row = {column: str(value) for column, value in zip(columns, key_values)}
        row.update(label_comparison_metrics(group, n_classes=n_classes))
        rows.append(row)
    return rows


def _threshold_report_table(folds: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Fold | Train / test windows | q20 (delta) | q40 (delta) | "
        "q60 (delta) | q80 (delta) | Changed |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in folds:
        lines.append(
            f"| {row['fold']} | {row['train_windows']} / {row['test_windows']} | "
            f"{row['q20']:.7f} ({row['delta_q20_vs_global']:+.7f}) | "
            f"{row['q40']:.7f} ({row['delta_q40_vs_global']:+.7f}) | "
            f"{row['q60']:.7f} ({row['delta_q60_vs_global']:+.7f}) | "
            f"{row['q80']:.7f} ({row['delta_q80_vs_global']:+.7f}) | "
            f"{row['comparison']['changed_fraction']:.4%} |"
        )
    return "\n".join(lines)


def load_label_sensitivity_spec(path: str | Path) -> dict[str, Any]:
    spec_path = _repo_path(path)
    with spec_path.open("r", encoding="utf-8") as stream:
        spec = yaml.safe_load(stream)
    if not isinstance(spec, dict) or not isinstance(spec.get("analysis"), dict):
        raise ValueError("Sensitivity YAML must contain an 'analysis' mapping")
    required = {"data_path", "output_dir", "report_path", "summary_path"}
    missing = sorted(required - set(spec["analysis"]))
    if missing:
        raise ValueError(f"Sensitivity config is missing keys: {missing}")
    return spec


@dataclass
class LabelDefinitionSensitivity:
    spec_path: Path
    spec: dict[str, Any]
    data_path: Path
    output_dir: Path
    report_path: Path
    summary_path: Path

    def __init__(
        self,
        spec_path: str | Path,
        *,
        output_dir: str | Path | None = None,
    ) -> None:
        self.spec_path = _repo_path(spec_path)
        self.spec = load_label_sensitivity_spec(self.spec_path)
        analysis = self.spec["analysis"]
        self.data_path = _repo_path(analysis["data_path"])
        self.output_dir = _repo_path(output_dir or analysis["output_dir"])
        self.report_path = _repo_path(analysis["report_path"])
        self.summary_path = _repo_path(analysis["summary_path"])

    def plan(self) -> dict[str, Any]:
        analysis = self.spec["analysis"]
        return {
            "analysis_name": analysis.get("name", "label_definition_sensitivity"),
            "data_path": self.data_path,
            "canonical_folds": "5-fold GroupKFold by subject_id",
            "canonical_reference_predictions": _repo_path(
                analysis["canonical_reference_predictions"]
            ),
            "global_internal_thresholds": analysis["global_internal_thresholds"],
            "fold_threshold_plan": "fit q20/q40/q60/q80 on outer-train target_focus only",
            "test_application": "[-inf, q20], ..., (q80, +inf]",
            "repeat_diagnostics_if_changed_fraction_exceeds": float(
                analysis.get("repeat_diagnostics_if_changed_fraction_exceeds", 0.05)
            ),
            "output_dir": self.output_dir,
            "report_path": self.report_path,
            "summary_path": self.summary_path,
            "models_trained": 0,
            "writes_performed": False,
        }

    @staticmethod
    def render_plan(plan: Mapping[str, Any]) -> str:
        return "\n".join(
            [
                "# Label definition sensitivity plan",
                "",
                f"- Dataset: `{_jsonable(plan['data_path'])}`",
                f"- Canonical folds: {plan['canonical_folds']}",
                f"- Canonical reference: `{_jsonable(plan['canonical_reference_predictions'])}`",
                f"- Global thresholds: {plan['global_internal_thresholds']}",
                f"- Fold thresholds: {plan['fold_threshold_plan']}",
                f"- Test bins: {plan['test_application']}",
                f"- Generated output: `{_jsonable(plan['output_dir'])}`",
                f"- Report: `{_jsonable(plan['report_path'])}`",
                "- Legacy label_q5 modified: no",
                "- Models trained: 0",
                "- Writes performed: no",
            ]
        )

    @staticmethod
    def _temporal_summary(
        statistics: Mapping[str, Any], prediction_metrics: Mapping[str, Any]
    ) -> dict[str, Any]:
        transitions = statistics["label_q5"]["transitions"]
        runs = statistics["label_q5"]["runs"]["overall"]
        return {
            "same_class_probability": transitions["same_class_probability"],
            "adjacent_class_probability": transitions["adjacent_class_probability"],
            "two_or_more_classes_probability": transitions[
                "two_or_more_classes_probability"
            ],
            "mean_run_length_windows": runs["length_windows"]["mean"],
            "median_run_length_windows": runs["length_windows"]["median"],
            "mean_run_duration_seconds": runs["duration_seconds"]["mean"],
            "median_run_duration_seconds": runs["duration_seconds"]["median"],
            "previous_label": prediction_metrics["overall"],
        }

    @staticmethod
    def _recommendation(
        comparison: Mapping[str, Any],
        sources: Sequence[Mapping[str, Any]],
        threshold_summary: Mapping[str, Any],
        global_temporal: Mapping[str, Any],
        safe_temporal: Mapping[str, Any],
    ) -> dict[str, Any]:
        max_threshold_delta = max(
            threshold_summary[name]["maximum_absolute_deviation_from_global"]
            for name in QUANTILE_NAMES
        )
        source_changes = [float(row["changed_fraction"]) for row in sources]
        persistence_delta = abs(
            float(safe_temporal["same_class_probability"])
            - float(global_temporal["same_class_probability"])
        )
        evidence = {
            "changed_fraction": float(comparison["changed_fraction"]),
            "two_or_more_class_shift_fraction": float(
                comparison["two_or_more_class_shift_fraction"]
            ),
            "quadratic_weighted_kappa": float(
                comparison["quadratic_weighted_kappa"]
            ),
            "maximum_threshold_delta": max_threshold_delta,
            "source_changed_fraction_range": max(source_changes) - min(source_changes),
            "persistence_absolute_delta": persistence_delta,
        }
        option_b = (
            evidence["changed_fraction"] < 0.05
            and evidence["two_or_more_class_shift_fraction"] == 0.0
            and evidence["quadratic_weighted_kappa"] > 0.98
            and evidence["maximum_threshold_delta"] < 0.01
            and evidence["source_changed_fraction_range"] < 0.01
            and evidence["persistence_absolute_delta"] < 0.02
        )
        return {
            "option": "B" if option_b else "A",
            "evidence": evidence,
            "recommendation": (
                "Keep global label_q5 as the predefined legacy task for reproducibility, "
                "and retain it as the predefined benchmark task for directly comparable "
                "experiments. Preserve the split-fitted result as a required sensitivity "
                "analysis and save its thresholds in every split artifact."
                if option_b
                else
                "Use fold-train quantile labels as the primary task in new experiments; "
                "retain global label_q5 only for reproducing legacy results."
            ),
            "baseline_repetition": (
                "A focused rerun of major RF/Transformer baselines is recommended later to "
                "confirm ranking stability, but it is not urgent given the small, adjacent, "
                "non-systematic shifts."
                if option_b
                else
                "Repeat the major RF/Transformer baselines with leakage-safe labels before "
                "making new comparative claims."
            ),
            "decision_basis": "joint threshold, shift, source, agreement, and temporal evidence",
        }

    def _render_report(self, summary: Mapping[str, Any]) -> str:
        comparison = summary["comparison"]
        thresholds = summary["thresholds"]
        threshold_summary = thresholds["summary"]
        temporal = summary["temporal"]
        top_subjects = summary["most_sensitive_subjects"][:10]
        lines = [
            "# Label definition sensitivity",
            "",
            "The existing `label_q5` remains unchanged. Four thresholds are fitted only "
            "from the outer-train subjects of each canonical GroupKFold split and applied "
            "unchanged to that fold's train and test targets.",
            "",
            "## Fold-specific thresholds",
            "",
            _threshold_report_table(thresholds["folds"]),
            "",
            "All folds retained four unique internal thresholds and all five classes. "
            "`duplicates='drop'` therefore has no effect in this sensitivity analysis.",
            "",
            "### Threshold distribution across folds",
            "",
            "| Threshold | Global | Mean | SD | Min | Max | Max absolute delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {name} | {threshold_summary[name]['global']:.7f} | "
                f"{threshold_summary[name]['mean']:.7f} | "
                f"{threshold_summary[name]['std']:.7f} | "
                f"{threshold_summary[name]['min']:.7f} | "
                f"{threshold_summary[name]['max']:.7f} | "
                f"{threshold_summary[name]['maximum_absolute_deviation_from_global']:.7f} |"
                for name in QUANTILE_NAMES
            ],
            "",
            "### Leakage-safe class balance",
            "",
            "Counts and fractions are ordered as classes `[0, 1, 2, 3, 4]`.",
            "",
            "| Fold | Train counts | Train fractions | Test counts | Test fractions | "
            "Max test deviation from 0.20 |",
            "| ---: | --- | --- | --- | --- | ---: |",
            *[
                "| "
                f"{row['fold']} | "
                f"{[row['train_class_balance']['counts'][str(i)] for i in range(5)]} | "
                f"{[round(row['train_class_balance']['fractions'][str(i)], 4) for i in range(5)]} | "
                f"{[row['test_class_balance']['counts'][str(i)] for i in range(5)]} | "
                f"{[round(row['test_class_balance']['fractions'][str(i)], 4) for i in range(5)]} | "
                f"{row['test_class_balance']['maximum_absolute_deviation_from_uniform']:.4%} |"
                for row in thresholds["folds"]
            ],
            "",
            "## Label agreement",
            "",
            f"- Agreement: {comparison['overall']['agreement_fraction']:.4%}",
            f"- Changed: {comparison['overall']['changed_fraction']:.4%} "
            f"({comparison['overall']['changed_windows']} windows)",
            f"- Mean absolute shift: "
            f"{comparison['overall']['mean_absolute_class_shift']:.6f}",
            f"- One-class shifts: "
            f"{comparison['overall']['one_class_shift_fraction']:.4%}",
            f"- Two-or-more-class shifts: "
            f"{comparison['overall']['two_or_more_class_shift_fraction']:.4%}",
            f"- Cohen's kappa: {comparison['overall']['cohens_kappa']:.6f}",
            f"- Quadratic weighted kappa: "
            f"{comparison['overall']['quadratic_weighted_kappa']:.6f}",
            "",
            "### Fold-level agreement",
            "",
            "| Fold | Agreement | Changed | One-class | Two-or-more | Kappa | QWK |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {row['outer_fold']} | {row['agreement_fraction']:.4%} | "
                f"{row['changed_fraction']:.4%} | "
                f"{row['one_class_shift_fraction']:.4%} | "
                f"{row['two_or_more_class_shift_fraction']:.4%} | "
                f"{row['cohens_kappa']:.6f} | "
                f"{row['quadratic_weighted_kappa']:.6f} |"
                for row in comparison["by_fold"]
            ],
            "",
            "### Sensitivity by class",
            "",
            "| Grouping | Class | Windows | Changed | Mean absolute shift |",
            "| --- | ---: | ---: | ---: | ---: |",
            *[
                f"| Global label | {row[GLOBAL_LABEL_COLUMN]} | {row['windows']} | "
                f"{row['changed_fraction']:.4%} | "
                f"{row['mean_absolute_class_shift']:.6f} |"
                for row in comparison["by_global_class"]
            ],
            *[
                f"| Fold-train label | {row[SAFE_LABEL_COLUMN]} | {row['windows']} | "
                f"{row['changed_fraction']:.4%} | "
                f"{row['mean_absolute_class_shift']:.6f} |"
                for row in comparison["by_safe_class"]
            ],
            "",
            "## Source sensitivity",
            "",
            "| Source | Windows | Changed | Mean absolute shift |",
            "| --- | ---: | ---: | ---: |",
            *[
                f"| {row['source']} | {row['windows']} | "
                f"{row['changed_fraction']:.4%} | "
                f"{row['mean_absolute_class_shift']:.6f} |"
                for row in comparison["by_source"]
            ],
            "",
            "## Most sensitive subjects",
            "",
            "| Subject | Windows | Changed |",
            "| --- | ---: | ---: |",
            *[
                f"| {row['subject_id']} | {row['windows']} | "
                f"{row['changed_fraction']:.4%} |"
                for row in top_subjects
            ],
            "",
            "## Temporal comparison",
            "",
            "| Label | Persistence | Adjacent transition | Severe transition | "
            "Mean run | Previous-label balanced accuracy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| Global | {temporal['global']['same_class_probability']:.4%} | "
            f"{temporal['global']['adjacent_class_probability']:.4%} | "
            f"{temporal['global']['two_or_more_classes_probability']:.4%} | "
            f"{temporal['global']['mean_run_length_windows']:.3f} | "
            f"{temporal['global']['previous_label']['balanced_accuracy']:.6f} |",
            f"| Fold-train | {temporal['fold_train_quantile']['same_class_probability']:.4%} | "
            f"{temporal['fold_train_quantile']['adjacent_class_probability']:.4%} | "
            f"{temporal['fold_train_quantile']['two_or_more_classes_probability']:.4%} | "
            f"{temporal['fold_train_quantile']['mean_run_length_windows']:.3f} | "
            f"{temporal['fold_train_quantile']['previous_label']['balanced_accuracy']:.6f} |",
            "",
            "Median run length is "
            f"{temporal['global']['median_run_length_windows']:.1f} window for the global "
            "label and "
            f"{temporal['fold_train_quantile']['median_run_length_windows']:.1f} window for "
            "the fold-train label.",
            "",
            "| Label | Accuracy | Balanced accuracy | Macro F1 | Ordinal MAE | "
            "Adjacent accuracy | Severe error |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {label} | {values['previous_label']['accuracy']:.6f} | "
                f"{values['previous_label']['balanced_accuracy']:.6f} | "
                f"{values['previous_label']['macro_f1']:.6f} | "
                f"{values['previous_label']['ordinal_mae']:.6f} | "
                f"{values['previous_label']['adjacent_accuracy']:.6f} | "
                f"{values['previous_label']['severe_error_rate']:.6f} |"
                for label, values in (
                    ("Global", temporal["global"]),
                    ("Fold-train", temporal["fold_train_quantile"]),
                )
            ],
            "",
            "D0-D3 were not repeated because the changed-label fraction did not exceed "
            "the configured 5% condition; no model was trained by this analysis.",
            "",
            "## Recommendation",
            "",
            f"**Option {summary['recommendation']['option']}.** "
            f"{summary['recommendation']['recommendation']}",
            "",
            summary["recommendation"]["baseline_repetition"],
            "",
            "The decision uses the threshold deviations, magnitude and direction of shifts, "
            "subject/source structure, weighted agreement, and temporal stability jointly; "
            "it is not based on a single acceptance cutoff.",
            "",
        ]
        return "\n".join(lines)

    def execute(self) -> dict[str, Any]:
        analysis = self.spec["analysis"]
        n_classes = int(analysis.get("n_classes", 5))
        global_thresholds = [
            float(value) for value in analysis["global_internal_thresholds"]
        ]
        before_hash = _sha256_file(self.data_path)
        before_size = self.data_path.stat().st_size
        source = pd.read_parquet(self.data_path, columns=REQUIRED_COLUMNS)
        legacy_labels_before = source["label_q5"].copy(deep=True)
        prepared = prepare_temporal_frame(source, n_classes=n_classes)
        expected_rows = analysis.get("expected_supervised_rows")
        if expected_rows is not None and len(prepared) != int(expected_rows):
            raise ValueError(
                f"Expected {expected_rows} supervised rows, observed {len(prepared)}"
            )
        folded, fold_metadata = assign_subject_folds(
            prepared, n_splits=int(analysis.get("n_splits", 5))
        )
        canonical_alignment = align_with_canonical_predictions(
            folded,
            _repo_path(analysis["canonical_reference_predictions"]),
            label_col="label_q5",
        )
        cross_fitted, threshold_result = build_cross_fitted_labels(
            folded,
            global_thresholds=global_thresholds,
            n_classes=n_classes,
        )
        threshold_summary = summarize_thresholds(threshold_result)
        if cross_fitted.empty or not threshold_result["all_folds_valid"]:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            threshold_result["summary"] = threshold_summary
            _write_json(self.output_dir / "fold_quantile_thresholds.json", threshold_result)
            raise ValueError(
                f"Train-derived thresholds are invalid in folds "
                f"{threshold_result['invalid_folds']}"
            )

        comparison_overall = label_comparison_metrics(
            cross_fitted, n_classes=n_classes
        )
        comparison_fold = comparison_by_group(
            cross_fitted, "outer_fold", n_classes=n_classes
        )
        comparison_source = comparison_by_group(
            cross_fitted, "source", n_classes=n_classes
        )
        comparison_subject = comparison_by_group(
            cross_fitted, "subject_id", n_classes=n_classes
        )
        comparison_global_class = comparison_by_group(
            cross_fitted, GLOBAL_LABEL_COLUMN, n_classes=n_classes
        )
        comparison_safe_class = comparison_by_group(
            cross_fitted, SAFE_LABEL_COLUMN, n_classes=n_classes
        )
        fold_comparison_map = {
            int(row["outer_fold"]): row for row in comparison_fold
        }
        for row in threshold_result["folds"]:
            row["comparison"] = fold_comparison_map[int(row["fold"])]
        threshold_result["summary"] = threshold_summary

        global_statistics, _ = calculate_temporal_statistics(
            cross_fitted, label_col=GLOBAL_LABEL_COLUMN, n_classes=n_classes
        )
        safe_statistics, _ = calculate_temporal_statistics(
            cross_fitted, label_col=SAFE_LABEL_COLUMN, n_classes=n_classes
        )
        global_previous = summarize_predictions(
            previous_label_predictions(cross_fitted, label_col=GLOBAL_LABEL_COLUMN)
        )
        safe_previous = summarize_predictions(
            previous_label_predictions(cross_fitted, label_col=SAFE_LABEL_COLUMN)
        )
        global_temporal = self._temporal_summary(global_statistics, global_previous)
        safe_temporal = self._temporal_summary(safe_statistics, safe_previous)

        repeat_threshold = float(
            analysis.get("repeat_diagnostics_if_changed_fraction_exceeds", 0.05)
        )
        repeat_diagnostics = comparison_overall["changed_fraction"] > repeat_threshold
        diagnostic_result: dict[str, Any]
        if repeat_diagnostics:
            _, safe_baseline_metrics = run_diagnostic_baselines(
                cross_fitted,
                label_col=SAFE_LABEL_COLUMN,
                n_classes=n_classes,
                spec=self.spec.get("diagnostic_baselines", {}),
            )
            diagnostic_result = {
                "status": "repeated",
                "reason": "changed fraction exceeded configured 5% condition",
                "metrics": safe_baseline_metrics,
            }
        else:
            diagnostic_result = {
                "status": "not_repeated",
                "reason": (
                    f"changed_fraction={comparison_overall['changed_fraction']:.8f} "
                    f"did not exceed {repeat_threshold:.8f}"
                ),
                "models_trained": 0,
            }

        recommendation = self._recommendation(
            comparison_overall,
            comparison_source,
            threshold_summary,
            global_temporal,
            safe_temporal,
        )
        comparison_output = cross_fitted[
            [
                "sample_id",
                "subject_id",
                "record_id",
                "source",
                "outer_fold",
                "target_focus",
                GLOBAL_LABEL_COLUMN,
                SAFE_LABEL_COLUMN,
                "label_changed",
                "absolute_label_shift",
            ]
        ].copy()
        detailed_temporal = {
            "global": {
                "statistics": global_statistics,
                "previous_label": global_previous,
            },
            "fold_train_quantile": {
                "statistics": safe_statistics,
                "previous_label": safe_previous,
            },
            "comparison": {
                "overall": comparison_overall,
                "by_fold": comparison_fold,
                "by_source": comparison_source,
                "by_subject": comparison_subject,
                "by_global_class": comparison_global_class,
                "by_safe_class": comparison_safe_class,
            },
            "diagnostic_baselines": diagnostic_result,
        }
        most_sensitive = sorted(
            comparison_subject,
            key=lambda row: (-float(row["changed_fraction"]), row["subject_id"]),
        )
        summary = {
            "analysis_name": analysis.get("name", "label_definition_sensitivity"),
            "analysis_only": True,
            "models_trained": 0 if not repeat_diagnostics else 30,
            "legacy_label_modified": False,
            "data_path": self.data_path,
            "input_sha256": before_hash,
            "input_size_bytes": before_size,
            "supervised_rows": int(len(cross_fitted)),
            "subjects": int(cross_fitted["subject_id"].nunique()),
            "records": int(cross_fitted[RECORD_COLUMNS].drop_duplicates().shape[0]),
            "canonical_alignment": canonical_alignment,
            "fold_metadata": fold_metadata,
            "thresholds": threshold_result,
            "comparison": {
                "overall": comparison_overall,
                "by_fold": comparison_fold,
                "by_source": comparison_source,
                "by_global_class": comparison_global_class,
                "by_safe_class": comparison_safe_class,
            },
            "most_sensitive_subjects": most_sensitive[:15],
            "temporal": {
                "global": global_temporal,
                "fold_train_quantile": safe_temporal,
                "delta": {
                    key: float(safe_temporal[key] - global_temporal[key])
                    for key in (
                        "same_class_probability",
                        "adjacent_class_probability",
                        "two_or_more_classes_probability",
                        "mean_run_length_windows",
                    )
                },
            },
            "diagnostic_baselines": diagnostic_result,
            "recommendation": recommendation,
            "artifacts": {
                "fold_quantile_thresholds": self.output_dir
                / "fold_quantile_thresholds.json",
                "cross_fitted_label_comparison": self.output_dir
                / "cross_fitted_label_comparison.parquet",
                "cross_fitted_temporal_statistics": self.output_dir
                / "cross_fitted_temporal_statistics.json",
                "report": self.report_path,
                "summary": self.summary_path,
            },
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.output_dir / "fold_quantile_thresholds.json", threshold_result)
        comparison_output.to_parquet(
            self.output_dir / "cross_fitted_label_comparison.parquet", index=False
        )
        _write_json(
            self.output_dir / "cross_fitted_temporal_statistics.json",
            detailed_temporal,
        )
        if not source["label_q5"].equals(legacy_labels_before):
            raise RuntimeError("In-memory legacy label_q5 changed during analysis")
        after_hash = _sha256_file(self.data_path)
        if after_hash != before_hash or self.data_path.stat().st_size != before_size:
            raise RuntimeError("Input Parquet changed during sensitivity analysis")
        summary["input_sha256_after"] = after_hash
        summary["input_modified"] = False
        _write_json(self.summary_path, summary)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            self._render_report(_jsonable(summary)), encoding="utf-8"
        )
        return _jsonable(summary)


__all__ = [
    "GLOBAL_LABEL_COLUMN",
    "LabelDefinitionSensitivity",
    "SAFE_LABEL_COLUMN",
    "apply_finite_thresholds",
    "build_cross_fitted_labels",
    "comparison_by_group",
    "label_comparison_metrics",
    "load_label_sensitivity_spec",
    "summarize_thresholds",
]
