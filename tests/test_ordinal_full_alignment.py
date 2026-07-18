from __future__ import annotations

import pandas as pd

from bench.experiments.ordinal_transformer_full import full_prediction_alignment


def _predictions() -> pd.DataFrame:
    return pd.DataFrame({
        "sequence_id": ["s0", "s1"],
        "fold": [1, 2],
        "subject_id": ["u0", "u1"],
        "record_id": ["r0", "r1"],
        "source": ["Old_EEG", "gpn_data"],
        "target_sample_id": [7, 8],
        "target_time": [70.0, 80.0],
        "y_true": [1, 4],
    })


def test_full_alignment_accepts_identical_canonical_rows() -> None:
    expected = _predictions()
    result = full_prediction_alignment(expected, expected.sample(frac=1, random_state=42))
    assert result["exact_match"] is True
    assert not any(result["mismatches"].values())


def test_full_alignment_detects_identity_mismatch() -> None:
    expected = _predictions()
    observed = expected.copy()
    observed.loc[1, "subject_id"] = "wrong"
    result = full_prediction_alignment(expected, observed)
    assert result["exact_match"] is False
    assert result["mismatches"]["subject_id"] == 1


def test_full_alignment_detects_duplicate_sequence_id() -> None:
    expected = _predictions()
    observed = expected.copy()
    observed.loc[1, "sequence_id"] = "s0"
    result = full_prediction_alignment(expected, observed)
    assert result["exact_match"] is False
    assert result["duplicate_sequence_ids"]["candidate"] == 1
