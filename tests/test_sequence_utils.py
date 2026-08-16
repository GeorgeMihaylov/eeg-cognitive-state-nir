import numpy as np
import pandas as pd
import pytest

from model_zoo.DL.sequence_utils import build_sequences


def _metadata_for_records(record_lengths: list[int]) -> pd.DataFrame:
    rows = []
    sample_id = 100
    for record_index, length in enumerate(record_lengths):
        for time_index in range(length):
            rows.append({
                "source": "gpn_data" if record_index < 2 else "Old_EEG",
                "subject_id": f"S{record_index // 2}",
                "record_id": f"R{record_index}",
                "record_group_id": f"G{record_index}",
                "sample_id": sample_id,
                "t_start": float(time_index * 10),
            })
            sample_id += 1
    return pd.DataFrame(rows)


def test_sequences_do_not_cross_source_subject_or_record_and_sort_time() -> None:
    metadata = _metadata_for_records([4, 4, 4])
    shuffled = metadata.sample(frac=1.0, random_state=42)
    X = np.column_stack([
        shuffled["sample_id"].to_numpy(),
        shuffled["t_start"].to_numpy(),
    ]).astype(np.float32)
    y = shuffled["sample_id"].to_numpy(dtype=np.int64)

    result = build_sequences(
        X,
        y,
        shuffled.reset_index(drop=True),
        sequence_length=3,
    )

    assert result.X.shape == (6, 3, 2)
    assert result.stats["records_total"] == 3
    for sequence, row, target in zip(
        result.X, result.metadata.itertuples(index=False), result.y
    ):
        sequence_sample_ids = sequence[:, 0].astype(np.int64)
        original = metadata.set_index("sample_id").loc[sequence_sample_ids]
        assert original["source"].nunique() == 1 == len({row.source})
        assert original["subject_id"].nunique() == 1 == len({row.subject_id})
        assert original["record_id"].nunique() == 1 == len({row.record_id})
        assert np.all(np.diff(sequence[:, 1]) > 0)
        assert target == sequence_sample_ids[-1]
        assert row.target_sample_id == sequence_sample_ids[-1]
        assert row.sequence_end_sample_id == sequence_sample_ids[-1]


def test_legacy_metadata_derives_record_group_without_crossing_records() -> None:
    metadata = _metadata_for_records([4, 4]).drop(columns="record_group_id")
    X = np.column_stack([metadata["sample_id"], metadata["t_start"]]).astype(
        np.float32
    )
    y = metadata["sample_id"].to_numpy(dtype=np.int64)

    result = build_sequences(X, y, metadata, sequence_length=3)

    assert result.stats["record_group_id_source"] == (
        "derived_from_source_subject_record"
    )
    assert result.stats["records_total"] == 2
    assert result.metadata["record_group_id"].nunique() == 2
    assert all(
        row.record_id in row.record_group_id
        for row in result.metadata.itertuples(index=False)
    )


def test_sequence_length_stride_metadata_and_unique_ids() -> None:
    metadata = _metadata_for_records([14])
    X = np.arange(14 * 5, dtype=np.float32).reshape(14, 5)
    y = np.arange(14, dtype=np.int64) % 5

    result = build_sequences(X, y, metadata, sequence_length=10, stride=2)

    assert result.X.shape == (3, 10, 5)
    assert result.y.shape == (3,)
    assert result.metadata["target_sample_id"].tolist() == [109, 111, 113]
    assert result.metadata["sequence_length"].eq(10).all()
    assert result.metadata["sequence_id"].is_unique
    assert result.stats["stride"] == 2
    assert result.stats["target_position"] == "last"
    assert result.stats["time_column"] == "t_start"


def test_short_records_are_skipped_and_reported() -> None:
    metadata = _metadata_for_records([4, 10])
    X = np.ones((14, 3), dtype=np.float32)
    y = np.arange(14, dtype=np.int64) % 5

    result = build_sequences(X, y, metadata, sequence_length=10)

    assert result.X.shape == (1, 10, 3)
    assert result.stats["records_total"] == 2
    assert result.stats["records_used"] == 1
    assert result.stats["records_skipped_short"] == 1


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_sequence_builder_rejects_non_finite_features(bad_value: float) -> None:
    metadata = _metadata_for_records([10])
    X = np.ones((10, 2), dtype=np.float32)
    X[4, 1] = bad_value

    with pytest.raises(ValueError, match="NaN or infinite"):
        build_sequences(X, np.arange(10) % 5, metadata)


def test_train_and_test_sequences_remain_subject_isolated() -> None:
    metadata = _metadata_for_records([12, 12, 12, 12])
    X = np.ones((len(metadata), 2), dtype=np.float32)
    y = np.arange(len(metadata), dtype=np.int64) % 5
    train_mask = metadata["subject_id"].eq("S0").to_numpy()
    test_mask = metadata["subject_id"].eq("S1").to_numpy()

    train = build_sequences(X[train_mask], y[train_mask], metadata[train_mask])
    test = build_sequences(X[test_mask], y[test_mask], metadata[test_mask])

    assert set(train.metadata["subject_id"]).isdisjoint(
        set(test.metadata["subject_id"])
    )


def test_gap_aware_sequences_do_not_cross_time_gaps() -> None:
    times = np.asarray([0, 10, 20, 30, 100, 110, 120], dtype=float)
    metadata = _metadata_for_records([len(times)])
    metadata["t_start"] = times
    X = np.column_stack([metadata["sample_id"], times]).astype(np.float32)
    y = np.arange(len(times), dtype=np.int64)

    result = build_sequences(
        X,
        y,
        metadata,
        sequence_length=3,
        expected_step_seconds=10.0,
        max_gap_seconds=10.5,
    )

    assert result.X.shape == (3, 3, 2)
    assert result.metadata["segment_id"].tolist() == [0, 0, 1]
    assert result.metadata["max_internal_gap"].eq(10.0).all()
    assert result.metadata["sequence_start_time"].tolist() == [0.0, 10.0, 100.0]
    assert result.metadata["sequence_end_time"].tolist() == [20.0, 30.0, 120.0]
    assert not any(
        sequence[0, 1] < 100 <= sequence[-1, 1]
        for sequence in result.X
    )
    assert result.stats["continuous_segments_total"] == 2
    assert result.stats["records_with_gaps"] == 1
    assert result.stats["gaps_detected"] == 1
    assert result.stats["largest_observed_gap"] == 70.0
    assert result.stats["candidate_sequences_without_gap_check"] == 5
    assert result.stats["valid_sequences_after_gap_check"] == 3
    assert result.stats["sequences_rejected_due_to_gaps"] == 2


def test_non_increasing_time_starts_new_segment() -> None:
    times = np.asarray([0, 10, 10, 20, 30], dtype=float)
    metadata = _metadata_for_records([len(times)])
    metadata["t_start"] = times
    metadata["sample_id"] = [5, 7, 6, 8, 9]
    X = np.column_stack([metadata["sample_id"], times]).astype(np.float32)

    result = build_sequences(
        X,
        np.arange(len(times)),
        metadata,
        sequence_length=2,
        expected_step_seconds=10.0,
        max_gap_seconds=10.5,
    )

    assert result.stats["gaps_detected"] == 1
    assert result.stats["continuous_segments_total"] == 2
    assert result.metadata["segment_id"].tolist() == [0, 1, 1]
    assert result.metadata["max_internal_gap"].eq(10.0).all()


def test_short_gap_segments_are_skipped_and_counted() -> None:
    times = np.asarray([0, 10, 100, 110, 120], dtype=float)
    metadata = _metadata_for_records([len(times)])
    metadata["t_start"] = times
    X = np.ones((len(times), 2), dtype=np.float32)

    result = build_sequences(
        X,
        np.arange(len(times)),
        metadata,
        sequence_length=3,
        expected_step_seconds=10.0,
        max_gap_seconds=10.5,
    )

    assert result.X.shape == (1, 3, 2)
    assert result.stats["windows_excluded_due_to_gaps"] == 2
    assert result.stats["candidate_sequences_without_gap_check"] == 3
    assert result.stats["valid_sequences_after_gap_check"] == 1
    assert result.stats["sequences_rejected_due_to_gaps"] == 2


def test_gap_filter_preserves_continuous_sequence_count() -> None:
    metadata = _metadata_for_records([14])
    X = np.ones((14, 2), dtype=np.float32)
    y = np.arange(14) % 5

    old_result = build_sequences(X, y, metadata, sequence_length=10)
    gap_result = build_sequences(
        X,
        y,
        metadata,
        sequence_length=10,
        expected_step_seconds=10.0,
        max_gap_seconds=10.5,
    )

    assert len(gap_result.X) == len(old_result.X) == 5
    assert gap_result.stats["sequences_rejected_due_to_gaps"] == 0


def test_endpoint_targets_use_unlabelled_context_and_last_sample() -> None:
    metadata = _metadata_for_records([12])
    X = np.column_stack([metadata["sample_id"], metadata["t_start"]]).astype(np.float32)
    endpoints = {110: 2, 111: 1}

    result = build_sequences(
        X,
        np.empty(0, dtype=np.int64),
        metadata,
        sequence_length=10,
        endpoint_targets=endpoints,
    )

    assert result.metadata["target_sample_id"].tolist() == [110, 111]
    assert result.y.tolist() == [2, 1]
    assert result.X[:, -1, 0].astype(int).tolist() == [110, 111]
    assert result.stats["full_target_count"] == 2
    assert result.stats["sequence_endpoint_count"] == 2


def test_endpoint_stats_separate_missing_history_and_gap() -> None:
    metadata = _metadata_for_records([12])
    metadata["t_start"] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100, 110, 120]
    endpoints = {100: 0, 109: 1, 111: 2}

    result = build_sequences(
        np.ones((12, 2), dtype=np.float32),
        np.empty(0, dtype=np.int64),
        metadata,
        sequence_length=3,
        expected_step_seconds=10.0,
        max_gap_seconds=10.01,
        endpoint_targets=endpoints,
    )

    assert result.metadata["target_sample_id"].tolist() == [111]
    assert result.stats["dropped_no_history"] == 1
    assert result.stats["dropped_gap"] == 1
    assert result.stats["dropped_other"] == 0


def test_sequences_never_cross_logical_record_group() -> None:
    metadata = _metadata_for_records([6])
    metadata.loc[3:, "record_group_id"] = "G-other"
    X = np.column_stack([metadata["sample_id"], metadata["t_start"]]).astype(np.float32)

    result = build_sequences(X, np.arange(6), metadata, sequence_length=3)

    assert len(result.X) == 2
    assert set(result.metadata["record_group_id"]) == {"G0", "G-other"}
    assert result.metadata["record_group_id"].is_unique
