"""Exact observation alignment for within-track paired comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


class AlignmentError(ValueError):
    """Raised when a paired comparison does not have exact observation identity."""


def _normalise_fold(value: Any) -> str:
    text = str(value)
    if text.startswith("fold_"):
        return text
    try:
        return f"fold_{int(float(text)):02d}"
    except ValueError:
        return text


def _fold_column(frame: pd.DataFrame) -> str | None:
    if "fold" in frame:
        return "fold"
    if "outer_fold" in frame:
        return "outer_fold"
    return None


@dataclass(frozen=True)
class AlignmentResult:
    left_model: str
    right_model: str
    prediction_unit: str
    identity_column: str
    aligned: bool
    reason: str
    left_predictions: int
    right_predictions: int
    matched_predictions: int
    left_duplicates: int
    right_duplicates: int
    missing_from_left: int
    missing_from_right: int
    fold_mismatches: int
    subject_mismatches: int
    target_mismatches: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_alignment(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_model: str,
    right_model: str,
    prediction_unit: str,
    identity_column: str | None = None,
    right_prediction_unit: str | None = None,
) -> AlignmentResult:
    """Check exact ID, fold, subject, and target equality without approximation."""

    if right_prediction_unit is not None and right_prediction_unit != prediction_unit:
        return AlignmentResult(
            left_model, right_model, prediction_unit, identity_column or "",
            False,
            f"prediction units differ: {prediction_unit} vs {right_prediction_unit}",
            len(left), len(right), 0, 0, 0, len(left), len(right), 0, 0, 0,
        )
    identity = identity_column or (
        "sequence_id"
        if prediction_unit == "sequence"
        else "sample_id"
    )
    required = {identity, "subject_id", "y_true"}
    missing_left = sorted(required - set(left.columns))
    missing_right = sorted(required - set(right.columns))
    if missing_left or missing_right:
        reason = f"required columns missing: left={missing_left}, right={missing_right}"
        return AlignmentResult(
            left_model, right_model, prediction_unit, identity, False, reason,
            len(left), len(right), 0, 0, 0, len(left), len(right), 0, 0, 0,
        )

    left_duplicates = int(left[identity].duplicated().sum())
    right_duplicates = int(right[identity].duplicated().sum())
    left_ids = set(left[identity])
    right_ids = set(right[identity])
    missing_from_left = len(right_ids - left_ids)
    missing_from_right = len(left_ids - right_ids)

    left_fold = _fold_column(left)
    right_fold = _fold_column(right)
    if left_fold is None or right_fold is None:
        return AlignmentResult(
            left_model, right_model, prediction_unit, identity, False,
            f"outer fold column missing: left={left_fold}, right={right_fold}",
            len(left), len(right), len(left_ids & right_ids),
            left_duplicates, right_duplicates, missing_from_left,
            missing_from_right, 0, 0, 0,
        )

    left_keys = left[[identity, left_fold, "subject_id", "y_true"]].rename(
        columns={left_fold: "outer_fold"}
    )
    right_keys = right[[identity, right_fold, "subject_id", "y_true"]].rename(
        columns={right_fold: "outer_fold"}
    )
    merged = left_keys.merge(
        right_keys,
        on=identity,
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one" if not left_duplicates and not right_duplicates else None,
    )
    fold_mismatches = int(
        (
            merged["outer_fold_left"].map(_normalise_fold)
            != merged["outer_fold_right"].map(_normalise_fold)
        ).sum()
    )
    subject_mismatches = int(
        (
            merged["subject_id_left"].astype(str)
            != merged["subject_id_right"].astype(str)
        ).sum()
    )
    target_mismatches = int(
        (~np.isclose(
            merged["y_true_left"].astype(float),
            merged["y_true_right"].astype(float),
            equal_nan=True,
        )).sum()
    )
    problems: list[str] = []
    if len(left) != len(right):
        problems.append(f"prediction counts differ ({len(left)} vs {len(right)})")
    if left_duplicates or right_duplicates:
        problems.append(
            f"duplicate IDs (left={left_duplicates}, right={right_duplicates})"
        )
    if missing_from_left or missing_from_right:
        problems.append(
            "prediction IDs differ "
            f"(missing left={missing_from_left}, missing right={missing_from_right})"
        )
    if fold_mismatches:
        problems.append(f"outer fold mismatches={fold_mismatches}")
    if subject_mismatches:
        problems.append(f"subject mismatches={subject_mismatches}")
    if target_mismatches:
        problems.append(f"target mismatches={target_mismatches}")
    return AlignmentResult(
        left_model=left_model,
        right_model=right_model,
        prediction_unit=prediction_unit,
        identity_column=identity,
        aligned=not problems,
        reason="exact alignment" if not problems else "; ".join(problems),
        left_predictions=len(left),
        right_predictions=len(right),
        matched_predictions=len(merged),
        left_duplicates=left_duplicates,
        right_duplicates=right_duplicates,
        missing_from_left=missing_from_left,
        missing_from_right=missing_from_right,
        fold_mismatches=fold_mismatches,
        subject_mismatches=subject_mismatches,
        target_mismatches=target_mismatches,
    )


def require_alignment(
    left: pd.DataFrame,
    right: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """Return exactly aligned rows or block the paired analysis."""

    result = check_alignment(left, right, **kwargs)
    if not result.aligned:
        raise AlignmentError(result.reason)
    identity = result.identity_column
    left_sorted = left.sort_values(identity).reset_index(drop=True)
    right_sorted = right.sort_values(identity).reset_index(drop=True)
    return left_sorted.add_suffix("_left").join(right_sorted.add_suffix("_right"))
