"""Typed native target contracts for COG-BCI record-labelled tasks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

import numpy as np
import pandas as pd


TARGET_SCHEMA_VERSION = 1

ALL_COG_BCI_TASK_VARIANTS = (
    "zero_back",
    "one_back",
    "two_back",
    "matb_easy",
    "matb_medium",
    "matb_difficult",
    "pvt",
    "flanker",
    "rest_begin_eyes_open",
    "rest_begin_eyes_closed",
    "rest_end_eyes_open",
    "rest_end_eyes_closed",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def require_relative_path(value: str | Path, *, label: str) -> Path:
    """Reject machine-specific paths in tracked protocol configuration."""
    text = str(value).replace("\\", "/")
    if Path(text).is_absolute() or PureWindowsPath(text).is_absolute():
        raise ValueError(f"{label} must be a relative path, got {value!r}")
    if any(part == ".." for part in Path(text).parts):
        raise ValueError(f"{label} must not escape its root, got {value!r}")
    return Path(text)


@dataclass(frozen=True)
class COGBCITaskDefinition:
    """Immutable record-level target schema inherited by window rows."""

    task_id: str
    task_family: str
    target_name: str
    target_type: str
    class_names: tuple[str, ...]
    class_to_index_items: tuple[tuple[str, int], ...]
    ordered_classes: bool
    included_task_variants: tuple[str, ...]
    excluded_task_variants: tuple[str, ...]
    target_source: str = "record.task_variant"
    target_level: str = "record"
    schema_version: int = TARGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.target_type not in {
            "ordinal_classification",
            "categorical_classification",
        }:
            raise ValueError(f"Unsupported target_type {self.target_type!r}")
        mapping = self.class_to_index
        if tuple(mapping) != self.included_task_variants:
            raise ValueError(
                "class_to_index order must match included_task_variants"
            )
        expected = list(range(len(self.class_names)))
        if list(mapping.values()) != expected:
            raise ValueError(
                f"Class indices must be contiguous {expected}, got "
                f"{list(mapping.values())}"
            )
        if len(self.class_names) != len(mapping):
            raise ValueError("class_names and class mapping have different sizes")
        if set(self.included_task_variants) & set(self.excluded_task_variants):
            raise ValueError("Included and excluded task variants overlap")
        if self.target_level != "record":
            raise ValueError("COG-BCI canonical targets must be record-level")

    @property
    def class_to_index(self) -> dict[str, int]:
        return dict(self.class_to_index_items)

    @property
    def index_to_class(self) -> dict[int, str]:
        return {
            index: class_name
            for index, class_name in enumerate(self.class_names)
        }

    @property
    def schema_hash(self) -> str:
        return _stable_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        document = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "task_family": self.task_family,
            "target_name": self.target_name,
            "target_type": self.target_type,
            "class_names": list(self.class_names),
            "class_to_index": self.class_to_index,
            "ordered_classes": self.ordered_classes,
            "included_task_variants": list(self.included_task_variants),
            "excluded_task_variants": list(self.excluded_task_variants),
            "target_source": self.target_source,
            "target_level": self.target_level,
        }
        if include_hash:
            document["target_schema_hash"] = self.schema_hash
        return document


def _definition(
    *,
    task_id: str,
    task_family: str,
    target_name: str,
    variants: tuple[str, ...],
) -> COGBCITaskDefinition:
    return COGBCITaskDefinition(
        task_id=task_id,
        task_family=task_family,
        target_name=target_name,
        target_type="ordinal_classification",
        class_names=variants,
        class_to_index_items=tuple(
            (variant, index) for index, variant in enumerate(variants)
        ),
        ordered_classes=True,
        included_task_variants=variants,
        excluded_task_variants=tuple(
            variant
            for variant in ALL_COG_BCI_TASK_VARIANTS
            if variant not in variants
        ),
    )


COG_BCI_TASK_DEFINITIONS: dict[str, COGBCITaskDefinition] = {
    "cog_bci_nback_3class": _definition(
        task_id="cog_bci_nback_3class",
        task_family="n_back",
        target_name="n_back_level",
        variants=("zero_back", "one_back", "two_back"),
    ),
    "cog_bci_matb_3class": _definition(
        task_id="cog_bci_matb_3class",
        task_family="matb",
        target_name="matb_difficulty",
        variants=("matb_easy", "matb_medium", "matb_difficult"),
    ),
}


def get_cog_bci_task_definition(task_id: str) -> COGBCITaskDefinition:
    try:
        return COG_BCI_TASK_DEFINITIONS[str(task_id)]
    except KeyError as error:
        raise ValueError(
            f"Unknown COG-BCI task_id {task_id!r}; available="
            f"{sorted(COG_BCI_TASK_DEFINITIONS)}"
        ) from error


@dataclass(frozen=True)
class COGBCITargetRecord:
    """One record-level target assignment before window inheritance."""

    task_id: str
    target_name: str
    record_id: str
    subject_id: str
    session_id: str
    task_variant: str
    target: int
    class_name: str


@dataclass
class COGBCITargetIndex:
    """Validated window-level target index with stable semantic identity."""

    definition: COGBCITaskDefinition
    frame: pd.DataFrame
    target_index_hash: str

    @property
    def accepted(self) -> pd.DataFrame:
        return self.frame.loc[self.frame["included_for_supervised"]].copy()

    @property
    def records(self) -> tuple[COGBCITargetRecord, ...]:
        columns = [
            "record_id",
            "subject_id",
            "session_id",
            "task_variant",
            "target",
            "class_name",
        ]
        records = (
            self.frame[columns]
            .drop_duplicates()
            .sort_values("record_id", kind="stable")
        )
        return tuple(
            COGBCITargetRecord(
                task_id=self.definition.task_id,
                target_name=self.definition.target_name,
                **row,
            )
            for row in records.to_dict(orient="records")
        )

    @classmethod
    def from_window_index(
        cls,
        windows: pd.DataFrame,
        definition: COGBCITaskDefinition,
    ) -> "COGBCITargetIndex":
        required = {
            "sample_id",
            "dataset",
            "source",
            "subject_id",
            "session_id",
            "record_id",
            "record_group_id",
            "task_family",
            "task_variant",
            "window_index",
            "start_sample",
            "stop_sample",
            "start_time_seconds",
            "stop_time_seconds",
            "status",
        }
        missing = sorted(required - set(windows.columns))
        if missing:
            raise ValueError(f"Window index is missing columns: {missing}")
        if windows["sample_id"].isna().any() or windows["sample_id"].eq("").any():
            raise ValueError("Window index contains missing sample_id")
        duplicates = windows["sample_id"].duplicated(keep=False)
        if duplicates.any():
            values = windows.loc[duplicates, "sample_id"].astype(str).unique()
            raise ValueError(
                f"Window index contains duplicate sample_id: {values[:5].tolist()}"
            )
        for column in ("record_id", "subject_id", "session_id", "record_group_id"):
            if windows[column].isna().any() or windows[column].eq("").any():
                raise ValueError(f"Window index contains missing {column}")

        mapping = definition.class_to_index
        included = windows["task_variant"].isin(mapping)
        wrong_family = included & windows["task_family"].ne(
            definition.task_family
        )
        if wrong_family.any():
            raise ValueError(
                "Included task variants appear under the wrong task family"
            )
        selected = windows.loc[included].copy()
        if selected.empty:
            raise ValueError(
                f"No windows match {definition.included_task_variants}"
            )
        record_variants = selected.groupby("record_id")["task_variant"].nunique()
        if record_variants.gt(1).any():
            invalid = record_variants[record_variants.gt(1)].index.tolist()
            raise ValueError(
                "A record cannot receive two labels for one task: "
                f"{invalid[:5]}"
            )
        selected["target"] = (
            selected["task_variant"].map(mapping).astype(np.int64)
        )
        selected["class_name"] = selected["target"].map(
            definition.index_to_class
        )
        selected["target_name"] = definition.target_name
        selected["task_id"] = definition.task_id
        selected["target_level"] = definition.target_level
        selected["included_for_supervised"] = selected["status"].eq("accepted")
        selected["window_duration_seconds"] = (
            selected["stop_time_seconds"].astype(float)
            - selected["start_time_seconds"].astype(float)
        )
        output_columns = [
            "sample_id",
            "dataset",
            "source",
            "subject_id",
            "session_id",
            "record_id",
            "record_group_id",
            "task_family",
            "task_variant",
            "window_index",
            "start_sample",
            "stop_sample",
            "start_time_seconds",
            "stop_time_seconds",
            "window_duration_seconds",
            "status",
            "included_for_supervised",
            "target",
            "class_name",
            "target_name",
            "task_id",
            "target_level",
        ]
        selected = selected[output_columns].sort_values(
            ["record_id", "window_index", "sample_id"], kind="stable"
        ).reset_index(drop=True)
        record_targets = selected.groupby("record_id")["target"].nunique()
        if record_targets.gt(1).any():
            raise ValueError("A record received multiple target indices")

        identity_rows = selected[
            [
                "sample_id",
                "record_id",
                "status",
                "target",
                "task_variant",
            ]
        ].astype(str).values.tolist()
        target_index_hash = _stable_hash(
            {
                "target_schema_hash": definition.schema_hash,
                "rows": identity_rows,
            }
        )
        return cls(
            definition=definition,
            frame=selected,
            target_index_hash=target_index_hash,
        )


def build_cog_bci_target_index(
    windows: pd.DataFrame,
    task_id: str,
) -> COGBCITargetIndex:
    return COGBCITargetIndex.from_window_index(
        windows,
        get_cog_bci_task_definition(task_id),
    )


def task_definition_from_config(
    config: Mapping[str, Any],
) -> COGBCITaskDefinition:
    if "task_id" not in config:
        raise ValueError("COG-BCI protocol config requires task_id")
    definition = get_cog_bci_task_definition(str(config["task_id"]))
    configured_target = config.get("target_name")
    if configured_target not in (None, definition.target_name):
        raise ValueError(
            f"Configured target_name {configured_target!r} does not match "
            f"{definition.target_name!r}"
        )
    return definition
