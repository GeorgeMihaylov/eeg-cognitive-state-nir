"""Reproducible, training-free audit of the benchmark focus target.

The audit deliberately operates on the already built window-level Parquet.  It
validates the stored target lineage, reconstructs the global quintile labels,
and describes their distribution without importing or constructing a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REQUIRED_COLUMNS = (
    "source",
    "subject_id",
    "record_id",
    "t_start",
    "t_end",
    "PM.Focus.Scaled__mean",
    "target_focus",
    "target_main",
    "label_q5",
)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        try:
            return str(value.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
        except ValueError:
            return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _json_counts(series: pd.Series) -> str:
    counts = series.value_counts().sort_index()
    return json.dumps(
        {str(int(key)): int(value) for key, value in counts.items()},
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_values(series: pd.Series) -> str:
    values = sorted({str(value) for value in series.dropna().tolist()})
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _assert_numeric_equal(left: pd.Series, right: pd.Series, message: str) -> None:
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    if not np.array_equal(left_numeric.isna().to_numpy(), right_numeric.isna().to_numpy()):
        raise ValueError(f"{message}: missing-value masks differ")
    valid = left_numeric.notna()
    if not np.array_equal(
        left_numeric.loc[valid].to_numpy(), right_numeric.loc[valid].to_numpy()
    ):
        raise ValueError(message)


def reconstruct_global_quantiles(
    target: pd.Series,
    *,
    n_classes: int,
) -> tuple[pd.Series, list[float]]:
    """Recreate the builder's single global ``pd.qcut`` call."""

    numeric = pd.to_numeric(target, errors="coerce")
    labels, boundaries = pd.qcut(
        numeric,
        q=n_classes,
        labels=False,
        duplicates="drop",
        retbins=True,
    )
    if len(boundaries) != n_classes + 1:
        raise ValueError(
            f"Expected {n_classes} quantile classes, got {len(boundaries) - 1}; "
            "duplicate boundaries were dropped"
        )
    return labels.astype("Float64"), [float(value) for value in boundaries]


def calculate_class_statistics(
    supervised: pd.DataFrame,
    *,
    target_col: str,
    label_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for class_id, group in supervised.groupby(label_col, sort=True, observed=True):
        target = group[target_col]
        rows.append(
            {
                "class_id": int(class_id),
                "windows": int(len(group)),
                "subjects": int(group["subject_id"].nunique()),
                "records": int(group[["source", "subject_id", "record_id"]].drop_duplicates().shape[0]),
                "sources": int(group["source"].nunique()),
                "target_focus_min": float(target.min()),
                "target_focus_max": float(target.max()),
                "target_focus_mean": float(target.mean()),
                "target_focus_median": float(target.median()),
                "target_focus_std": float(target.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values("class_id").reset_index(drop=True)


def calculate_subject_statistics(
    supervised: pd.DataFrame,
    *,
    target_col: str,
    label_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject_id, group in supervised.groupby("subject_id", sort=True, observed=True):
        target = group[target_col]
        counts = group[label_col].value_counts()
        rows.append(
            {
                "subject_id": str(subject_id),
                "source_membership": _json_values(group["source"]),
                "records": int(group[["source", "record_id"]].drop_duplicates().shape[0]),
                "windows": int(len(group)),
                "target_focus_mean": float(target.mean()),
                "target_focus_std": float(target.std(ddof=1)),
                "target_focus_min": float(target.min()),
                "target_focus_max": float(target.max()),
                "classes_present": _json_values(group[label_col].astype(int)),
                "majority_class_fraction": float(counts.max() / len(group)),
                "class_counts": _json_counts(group[label_col]),
            }
        )
    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def calculate_record_statistics(
    supervised: pd.DataFrame,
    *,
    target_col: str,
    label_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["source", "subject_id", "record_id"]
    for keys, group in supervised.groupby(group_cols, sort=True, observed=True):
        source, subject_id, record_id = keys
        target = group[target_col]
        duration = float(group["t_end"].max() - group["t_start"].min())
        rows.append(
            {
                "source": str(source),
                "subject_id": str(subject_id),
                "record_id": str(record_id),
                "windows": int(len(group)),
                "duration": duration,
                "target_focus_mean": float(target.mean()),
                "target_focus_std": float(target.std(ddof=1)),
                "classes_present": _json_values(group[label_col].astype(int)),
                "class_counts": _json_counts(group[label_col]),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def calculate_variance_decomposition(
    supervised: pd.DataFrame,
    *,
    target_col: str,
) -> dict[str, Any]:
    """Population sum-of-squares decomposition plus a one-way ICC(1)."""

    frame = supervised[["source", "subject_id", "record_id", target_col]].copy()
    values = frame[target_col].to_numpy(dtype=float)
    grand_mean = float(np.mean(values))
    total_ss = float(np.sum((values - grand_mean) ** 2))
    if total_ss <= 0:
        raise ValueError("target_focus must have positive variance")

    subject_groups = list(frame.groupby("subject_id", sort=True, observed=True))
    subject_means = frame.groupby("subject_id", observed=True)[target_col].transform("mean")
    between_subject_ss = float(np.sum((subject_means.to_numpy() - grand_mean) ** 2))
    within_subject_ss = float(np.sum((values - subject_means.to_numpy()) ** 2))

    record_keys = ["subject_id", "source", "record_id"]
    record_means = frame.groupby(record_keys, observed=True)[target_col].transform("mean")
    between_record_within_subject_ss = float(
        np.sum((record_means.to_numpy() - subject_means.to_numpy()) ** 2)
    )
    within_record_ss = float(np.sum((values - record_means.to_numpy()) ** 2))

    source_means = frame.groupby("source", observed=True)[target_col].transform("mean")
    between_source_ss = float(np.sum((source_means.to_numpy() - grand_mean) ** 2))

    n = len(frame)
    subject_count = len(subject_groups)
    sizes = np.asarray([len(group) for _, group in subject_groups], dtype=float)
    ms_between = between_subject_ss / (subject_count - 1)
    ms_within = within_subject_ss / (n - subject_count)
    effective_group_size = float((n - np.sum(sizes**2) / n) / (subject_count - 1))
    icc_1 = float(
        (ms_between - ms_within)
        / (ms_between + (effective_group_size - 1.0) * ms_within)
    )

    source_rows: list[dict[str, Any]] = []
    for source, group in frame.groupby("source", sort=True, observed=True):
        series = group[target_col]
        source_rows.append(
            {
                "source": str(source),
                "windows": int(len(group)),
                "subjects": int(group["subject_id"].nunique()),
                "records": int(group[["subject_id", "record_id"]].drop_duplicates().shape[0]),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    source_difference: dict[str, Any] | None = None
    if len(source_rows) == 2:
        first, second = source_rows
        first_values = frame.loc[frame["source"].astype(str) == first["source"], target_col]
        second_values = frame.loc[frame["source"].astype(str) == second["source"], target_col]
        pooled_variance = (
            (len(first_values) - 1) * first_values.var(ddof=1)
            + (len(second_values) - 1) * second_values.var(ddof=1)
        ) / (len(first_values) + len(second_values) - 2)
        difference = float(first_values.mean() - second_values.mean())
        source_difference = {
            "contrast": f"{first['source']} - {second['source']}",
            "mean_difference": difference,
            "cohens_d_unadjusted": float(difference / np.sqrt(pooled_variance)),
        }

    return {
        "method": "population sum-of-squares decomposition on supervised windows",
        "total_variance_population": float(np.var(values, ddof=0)),
        "total_sum_of_squares": total_ss,
        "between_subject": {
            "sum_of_squares": between_subject_ss,
            "fraction_of_total": between_subject_ss / total_ss,
        },
        "within_subject": {
            "sum_of_squares": within_subject_ss,
            "fraction_of_total": within_subject_ss / total_ss,
        },
        "between_record_within_subject": {
            "sum_of_squares": between_record_within_subject_ss,
            "fraction_of_total": between_record_within_subject_ss / total_ss,
        },
        "within_record": {
            "sum_of_squares": within_record_ss,
            "fraction_of_total": within_record_ss / total_ss,
        },
        "between_source_unadjusted": {
            "sum_of_squares": between_source_ss,
            "fraction_of_total": between_source_ss / total_ss,
        },
        "hierarchical_reconstruction_error_fraction": abs(
            total_ss - between_subject_ss - between_record_within_subject_ss - within_record_ss
        ) / total_ss,
        "icc_1_one_way_random": icc_1,
        "icc_effective_group_size": effective_group_size,
        "source_statistics": source_rows,
        "source_difference": source_difference,
        "caveats": [
            "Fractions are descriptive window-weighted quantities, not causal effects.",
            "The source fraction is unadjusted for subjects represented in both sources.",
            "Repeated windows are temporally dependent, so window counts are not independent sample sizes.",
        ],
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    table = frame[columns]
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in table.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, float):
                rendered.append(f"{value:.6f}")
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, rule, *rows])


def load_label_target_audit_spec(path: str | Path) -> dict[str, Any]:
    spec_path = _repo_path(path)
    with spec_path.open("r", encoding="utf-8") as stream:
        spec = yaml.safe_load(stream)
    if not isinstance(spec, dict) or not isinstance(spec.get("audit"), dict):
        raise ValueError("Label target audit YAML must contain an 'audit' mapping")
    required = {"data_path", "output_dir", "report_path", "summary_path"}
    missing = sorted(required - set(spec["audit"]))
    if missing:
        raise ValueError(f"Audit config is missing required keys: {missing}")
    return spec


@dataclass
class LabelTargetAudit:
    spec_path: Path
    spec: dict[str, Any]
    data_path: Path
    output_dir: Path
    report_path: Path
    summary_path: Path
    target_col: str
    label_col: str
    n_classes: int

    def __init__(
        self,
        spec_path: str | Path,
        *,
        output_dir: str | Path | None = None,
    ) -> None:
        self.spec_path = _repo_path(spec_path)
        self.spec = load_label_target_audit_spec(self.spec_path)
        audit = self.spec["audit"]
        self.data_path = _repo_path(audit["data_path"])
        self.output_dir = _repo_path(output_dir or audit["output_dir"])
        self.report_path = _repo_path(audit["report_path"])
        self.summary_path = _repo_path(audit["summary_path"])
        self.target_col = str(audit.get("target_column", "target_focus"))
        self.label_col = str(audit.get("label_column", "label_q5"))
        self.n_classes = int(audit.get("n_classes", 5))
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least two")

    def plan(self) -> dict[str, Any]:
        """Return the resolved read-only plan without creating any path."""

        return {
            "analysis_name": self.spec["audit"].get("name", "label_target_audit"),
            "spec_path": self.spec_path,
            "data_path": self.data_path,
            "target_column": self.target_col,
            "label_column": self.label_col,
            "n_classes": self.n_classes,
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
                "# Label target audit plan",
                "",
                f"- Spec: `{_jsonable(plan['spec_path'])}`",
                f"- Input: `{_jsonable(plan['data_path'])}`",
                f"- Target: `{plan['target_column']}`",
                f"- Label: `{plan['label_column']}` ({plan['n_classes']} classes)",
                f"- Generated output: `{_jsonable(plan['output_dir'])}`",
                f"- Report: `{_jsonable(plan['report_path'])}`",
                "- Models trained: 0",
                "- Writes performed: no",
            ]
        )

    def _load_and_validate(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        if not self.data_path.is_file():
            raise FileNotFoundError(f"Audit dataset not found: {self.data_path}")
        columns = list(dict.fromkeys((*DEFAULT_REQUIRED_COLUMNS, self.target_col, self.label_col)))
        frame = pd.read_parquet(self.data_path, columns=columns)
        for column in ("t_start", "t_end", "PM.Focus.Scaled__mean", self.target_col, "target_main", self.label_col):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        _assert_numeric_equal(
            frame[self.target_col],
            frame["PM.Focus.Scaled__mean"],
            f"{self.target_col} is not an exact copy of PM.Focus.Scaled__mean",
        )
        _assert_numeric_equal(
            frame[self.target_col],
            frame["target_main"],
            f"{self.target_col} is not an exact copy of target_main",
        )
        if not np.array_equal(
            frame[self.target_col].isna().to_numpy(),
            frame[self.label_col].isna().to_numpy(),
        ):
            raise ValueError("target and label missing-value masks differ")

        supervised = frame.loc[frame[self.target_col].notna()].copy()
        labels = supervised[self.label_col]
        if not np.all(np.equal(labels.to_numpy(), np.floor(labels.to_numpy()))):
            raise ValueError(f"{self.label_col} contains non-integer values")
        observed = sorted(labels.astype(int).unique().tolist())
        expected = list(range(self.n_classes))
        if observed != expected:
            raise ValueError(
                f"{self.label_col} classes are {observed}; expected exactly {expected}"
            )

        reconstructed, boundaries = reconstruct_global_quantiles(
            frame[self.target_col], n_classes=self.n_classes
        )
        reconstructed_values = reconstructed.to_numpy(dtype=float, na_value=np.nan)
        stored_values = frame[self.label_col].to_numpy(dtype=float)
        if not np.array_equal(np.isnan(reconstructed_values), np.isnan(stored_values)):
            raise ValueError("Reconstructed qcut labels have a different missing mask")
        valid = np.isfinite(stored_values)
        if not np.array_equal(reconstructed_values[valid], stored_values[valid]):
            mismatches = int(np.sum(reconstructed_values[valid] != stored_values[valid]))
            raise ValueError(f"Stored labels differ from global qcut in {mismatches} rows")

        info = {
            "global_quantile_boundaries": boundaries,
            "stored_labels_match_recomputed_global_qcut": True,
            "missing_target_rows": int(frame[self.target_col].isna().sum()),
        }
        return frame, info

    def _validate_expected_counts(self, summary: Mapping[str, Any]) -> None:
        audit = self.spec["audit"]
        checks = {
            "expected_rows": "rows",
            "expected_supervised_rows": "supervised_rows",
            "expected_subjects": "subjects",
            "expected_records": "records",
        }
        for config_key, result_key in checks.items():
            if config_key in audit and int(audit[config_key]) != int(summary[result_key]):
                raise ValueError(
                    f"{config_key}={audit[config_key]} but observed {result_key}={summary[result_key]}"
                )

    def _render_report(
        self,
        summary: Mapping[str, Any],
        classes: pd.DataFrame,
        variance: Mapping[str, Any],
    ) -> str:
        boundaries = summary["label_q5"]["global_quantile_boundaries"]
        interval_lines = []
        for class_id in range(self.n_classes):
            left = "[" if class_id == 0 else "("
            interval_lines.append(
                f"- Class {class_id}: `{left}{boundaries[class_id]:.15g}, "
                f"{boundaries[class_id + 1]:.15g}]`"
            )
        source_rows = pd.DataFrame(variance["source_statistics"])
        source_boundary_lines = [
            f"- `{source}`: "
            + ", ".join(f"{value:.15g}" for value in values)
            for source, values in summary["label_q5"][
                "counterfactual_source_specific_boundaries"
            ].items()
        ]
        return "\n".join(
            [
                "# Label target audit",
                "",
                "## Provenance",
                "",
                "The verified construction path is:",
                "",
                "1. `src/02_build_emotiv_catalog.py` inventories exported Emotiv CSV/BZ2 "
                "records under both `data/raw/gpn_data` and `data/raw/Old_EEG`, preserving "
                "the acquisition source, subject, file layout, separator, and header metadata. "
                "Both source inventories expose the vendor-provided `PM.Focus.Scaled` field; "
                "this repository does not derive that upstream scale.",
                "2. `src/04_build_windowed_pm_dataset.py` reads the catalog and validated "
                "common columns. `pd.to_numeric(errors='coerce')` converts invalid PM values "
                "to missing; only rows with a missing `Timestamp` are explicitly dropped. "
                "Records are divided into absolute 10-second timestamp bins. "
                "`PM.Focus.Scaled` is aggregated as mean/std/min/max/last, and "
                "`target_focus = PM.Focus.Scaled__mean`; pandas' mean ignores individual "
                "missing focus samples, leaving a missing window target only when its focus "
                "mean is unavailable. If one logical time bin crosses CSV chunk boundaries, "
                "the current implementation takes an unweighted mean of its chunk-level "
                "aggregates during secondary aggregation.",
                "3. Within each record, `target_main = target_focus`. No additional target "
                "normalization and no `PM.Focus.IsActive` mask are applied.",
                "4. All record tables from both sources are concatenated. Only then is "
                "`label_q5` calculated from `target_main` by a single global `pd.qcut` call.",
                "5. `src/08_build_eeg_features.py` left-merges the EEG features into that "
                "PM/POW table. Direct comparison of the two processed Parquets showed the "
                "same row count and exact equality of `target_focus` and `label_q5`.",
                "6. Benchmark configs select the stored `label_q5` with `discretize: false`; "
                "the dataset loader and five-class task validate/use it but do not recreate "
                "the quantile labels.",
                "",
                "`label_q5` is produced after all records from both sources have been "
                "concatenated by `pd.qcut(target_main, q=5, labels=False, "
                "duplicates='drop')`; `target_main` and `target_focus` are exactly equal. It is not "
                "computed per source, subject, or record, and no train/test split exists at "
                "that point.",
                "",
                "## Dataset structure",
                "",
                f"- Rows: {summary['rows']:,}",
                f"- Subjects (all / supervised): {summary['subjects']} / {summary['supervised_subjects']}",
                f"- Records (all / supervised): {summary['records']} / {summary['supervised_records']}",
                f"- Sources: {summary['sources']}",
                f"- Non-null `target_focus`: {summary['target_focus_non_null']:,}",
                f"- Non-null `label_q5`: {summary['label_q5_non_null']:,}",
                f"- Subjects in both sources: {summary['subjects_in_both_sources']}",
                f"- Subjects in one source only: {summary['subjects_in_one_source']}",
                "",
                "## Reconstructed quintile boundaries",
                "",
                *interval_lines,
                "",
                "These one set of boundaries is applied to `Old_EEG` and `gpn_data`. For "
                "diagnosis only, independently refitting `qcut` inside each source would have "
                "produced the following different edges (these were not used):",
                "",
                *source_boundary_lines,
                "",
                "The stored label agrees exactly with a fresh global `qcut` on the current "
                "processed target. The numerical boundaries were not persisted by the "
                "original builder; these values are reconstructed from the current Parquet. "
                "Therefore the current labels are exactly reproducible, while a raw rebuild "
                "also depends on retaining the same raw exports, builder code, chunking, and "
                "record inventory.",
                "",
                "## Per-class statistics",
                "",
                _markdown_table(
                    classes,
                    [
                        "class_id", "windows", "subjects", "records", "sources",
                        "target_focus_min", "target_focus_max", "target_focus_mean",
                        "target_focus_median", "target_focus_std",
                    ],
                ),
                "",
                "## Source comparison",
                "",
                _markdown_table(
                    source_rows,
                    ["source", "windows", "subjects", "records", "mean", "std", "min", "max"],
                ),
                "",
                f"The unadjusted source mean contrast is "
                f"{variance['source_difference']['mean_difference']:.6f} "
                f"(`{variance['source_difference']['contrast']}`), with descriptive "
                f"Cohen's d {variance['source_difference']['cohens_d_unadjusted']:.6f}.",
                "",
                "## Variance decomposition",
                "",
                f"- Total population variance: {variance['total_variance_population']:.9f}",
                f"- Between subjects: {variance['between_subject']['fraction_of_total']:.4%}",
                f"- Within subjects: {variance['within_subject']['fraction_of_total']:.4%}",
                f"- Between records within subject: "
                f"{variance['between_record_within_subject']['fraction_of_total']:.4%}",
                f"- Within records: {variance['within_record']['fraction_of_total']:.4%}",
                f"- Unadjusted between sources: "
                f"{variance['between_source_unadjusted']['fraction_of_total']:.4%}",
                f"- One-way ICC(1): {variance['icc_1_one_way_random']:.6f}",
                "",
                "These are descriptive, window-weighted components. Temporally adjacent "
                "windows are not independent, and the source component is not adjusted for "
                "subjects observed in both sources.",
                "",
                "## Leakage assessment and scientific interpretation",
                "",
                "The EEG or feature values are not used to define the classes. However, the "
                "global class boundaries use target values from every subject and both sources "
                "before GroupKFold, LOSO, or cross-source evaluation. This is a methodological "
                "target-definition leakage (a transductive use of outer-test target "
                "distribution), even though it does not directly expose test EEG features to "
                "the estimator. It can make class balance and thresholds depend on the test "
                "cohort. Future confirmatory evaluations should freeze clinically or "
                "scientifically justified thresholds, or estimate thresholds on each outer "
                "training partition and apply them unchanged to its test partition.",
                "",
                "Scientifically, `label_q5` is an ordinal discretization of a proprietary "
                "device-derived focus metric averaged over a window. It is a weak proxy target, "
                "not a direct expert annotation, diagnosis, or independently validated "
                "cognitive-state ground truth. The five IDs encode ordered global quantile "
                "bands; treating them as nominal classes is an engineering benchmark choice.",
                "",
                "## Reproducibility and safety",
                "",
                f"- Input SHA-256 before/after: `{summary['input_sha256']}` / "
                f"`{summary['input_sha256_after']}`",
                "- Input Parquet modified: no",
                "- Models trained: 0",
                "- Generated tables are written outside tracked source files.",
                "",
            ]
        )

    def execute(self) -> dict[str, Any]:
        """Validate target lineage and write deterministic audit artifacts."""

        input_hash = _sha256_file(self.data_path)
        input_size = self.data_path.stat().st_size
        frame, label_info = self._load_and_validate()
        supervised = frame.loc[frame[self.target_col].notna()].copy()
        supervised[self.label_col] = supervised[self.label_col].astype(int)

        class_stats = calculate_class_statistics(
            supervised, target_col=self.target_col, label_col=self.label_col
        )
        subject_stats = calculate_subject_statistics(
            supervised, target_col=self.target_col, label_col=self.label_col
        )
        record_stats = calculate_record_statistics(
            supervised, target_col=self.target_col, label_col=self.label_col
        )
        variance = calculate_variance_decomposition(supervised, target_col=self.target_col)

        source_counts = supervised.groupby("subject_id", observed=True)["source"].nunique()
        counterfactual_boundaries = {
            str(source): reconstruct_global_quantiles(
                group[self.target_col], n_classes=self.n_classes
            )[1]
            for source, group in supervised.groupby("source", sort=True, observed=True)
        }
        summary: dict[str, Any] = {
            "analysis_name": self.spec["audit"].get("name", "label_target_audit"),
            "analysis_only": True,
            "models_trained": 0,
            "data_path": _jsonable(self.data_path),
            "input_size_bytes": input_size,
            "input_sha256": input_hash,
            "rows": int(len(frame)),
            "subjects": int(frame["subject_id"].nunique()),
            "records": int(frame[["source", "subject_id", "record_id"]].drop_duplicates().shape[0]),
            "sources": sorted(frame["source"].astype(str).unique().tolist()),
            "target_focus_non_null": int(frame[self.target_col].notna().sum()),
            "label_q5_non_null": int(frame[self.label_col].notna().sum()),
            "supervised_rows": int(len(supervised)),
            "supervised_subjects": int(supervised["subject_id"].nunique()),
            "supervised_records": int(
                supervised[["source", "subject_id", "record_id"]].drop_duplicates().shape[0]
            ),
            "subjects_in_both_sources": int((source_counts > 1).sum()),
            "subjects_in_one_source": int((source_counts == 1).sum()),
            "target_lineage": {
                "raw_export_column": "PM.Focus.Scaled",
                "window_aggregate_column": "PM.Focus.Scaled__mean",
                "target_column": self.target_col,
                "target_main_is_exact_copy": True,
                "additional_scaling_in_repository": False,
                "activity_mask_applied": False,
                "numeric_parse_failures_become_missing": True,
                "window_seconds": 10.0,
                "chunk_boundary_secondary_aggregation": "unweighted mean of chunk-level aggregates",
            },
            "label_q5": {
                **label_info,
                "algorithm": "pd.qcut(target_main, q=5, labels=False, duplicates='drop')",
                "scope": "all concatenated records, subjects, and sources before evaluation split",
                "per_source": False,
                "per_subject": False,
                "per_record": False,
                "boundaries_persisted_by_original_builder": False,
                "uses_outer_test_target_distribution": True,
                "methodological_leakage_risk": True,
                "counterfactual_source_specific_boundaries": counterfactual_boundaries,
            },
            "source_statistics": variance["source_statistics"],
            "variance_decomposition": variance,
            "artifacts": {
                "target_class_statistics": self.output_dir / "target_class_statistics.parquet",
                "subject_target_statistics": self.output_dir / "subject_target_statistics.parquet",
                "record_target_statistics": self.output_dir / "record_target_statistics.parquet",
                "variance_decomposition": self.output_dir / "variance_decomposition.json",
                "report": self.report_path,
                "summary": self.summary_path,
            },
        }
        self._validate_expected_counts(summary)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        class_stats.to_parquet(
            self.output_dir / "target_class_statistics.parquet", index=False
        )
        subject_stats.to_parquet(
            self.output_dir / "subject_target_statistics.parquet", index=False
        )
        record_stats.to_parquet(
            self.output_dir / "record_target_statistics.parquet", index=False
        )
        _write_json(self.output_dir / "variance_decomposition.json", variance)

        input_hash_after = _sha256_file(self.data_path)
        if input_hash_after != input_hash or self.data_path.stat().st_size != input_size:
            raise RuntimeError("Input Parquet changed during the read-only audit")
        summary["input_sha256_after"] = input_hash_after
        summary["input_modified"] = False
        _write_json(self.summary_path, summary)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            self._render_report(summary, class_stats, variance), encoding="utf-8"
        )
        return _jsonable(summary)


__all__ = [
    "LabelTargetAudit",
    "calculate_class_statistics",
    "calculate_record_statistics",
    "calculate_subject_statistics",
    "calculate_variance_decomposition",
    "load_label_target_audit_spec",
    "reconstruct_global_quantiles",
]
