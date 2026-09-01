import numpy as np
import pytest

from bench.core.abstract_dataset import EEGData
from bench.tasks.cognitive_load import CognitiveLoad5ClassTask
from bench.validation.cross_val import CrossValidator


def make_cross_source_data() -> EEGData:
    rows = []

    def add_record(source, subject, logical_id, record_id, n_rows=25):
        start = len(rows)
        for offset in range(n_rows):
            rows.append({
                "sample_id": start + offset,
                "source": source,
                "subject_id": subject,
                "record_id": record_id,
                "record_group_id": logical_id,
                "label": offset % 5,
                "time": float(offset * 10),
            })

    for index in range(5):
        subject = f"G{index}"
        add_record("gpn_data", subject, f"g_{index}", f"gpn__g_{index}")
    for index in range(3):
        subject = f"O{index}"
        add_record("Old_EEG", subject, f"o_{index}", f"old__o_{index}")
    for index in range(3):
        subject = f"X{index}"
        duplicate = f"duplicate_{index}"
        add_record("gpn_data", subject, duplicate, f"gpn__{duplicate}")
        add_record("Old_EEG", subject, duplicate, f"old__{duplicate}")
        add_record(
            "gpn_data", subject, f"g_extra_{index}", f"gpn__extra_{index}"
        )
    add_record("Old_EEG", "X0", "old_extra_0", "old__extra_0")

    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    sample_ids = np.asarray([row["sample_id"] for row in rows], dtype=np.int64)
    features = np.column_stack([
        sample_ids,
        labels,
        np.sin(sample_ids),
        np.cos(sample_ids),
    ]).astype(np.float32)
    return EEGData(
        data=features,
        labels=labels,
        subject_ids=np.asarray([row["subject_id"] for row in rows]),
        sample_ids=sample_ids,
        record_ids=np.asarray([row["record_id"] for row in rows]),
        row_metadata={
            "source": np.asarray([row["source"] for row in rows]),
            "record_group_id": np.asarray([
                row["record_group_id"] for row in rows
            ]),
            "t_start": np.asarray([row["time"] for row in rows]),
            "t_center": np.asarray([row["time"] + 5 for row in rows]),
        },
    )


@pytest.fixture
def cross_validator() -> CrossValidator:
    data = make_cross_source_data()
    task = CognitiveLoad5ClassTask(data, {"random_state": 42})
    return CrossValidator(task)


@pytest.mark.parametrize(
    ("train_source", "test_source", "n_train_subjects", "n_test_subjects"),
    [
        ("gpn_data", "Old_EEG", 5, 3),
        ("Old_EEG", "gpn_data", 3, 5),
    ],
)
def test_source_exclusive_split_is_directional_and_leak_free(
    cross_validator,
    train_source,
    test_source,
    n_train_subjects,
    n_test_subjects,
):
    split = cross_validator.run_cross_source_holdout(
        train_source=train_source,
        test_source=test_source,
        subject_mode="source_exclusive",
        minimum_train_subjects=3,
    )

    assert split.metadata["status"] == "valid"
    assert split.metadata["n_train_subjects"] == n_train_subjects
    assert split.metadata["n_test_subjects"] == n_test_subjects
    assert set(split.row_metadata_train["source"]) == {train_source}
    assert set(split.row_metadata_test["source"]) == {test_source}
    assert split.metadata["subject_overlap"] == []
    assert split.metadata["logical_record_overlap"] == []
    assert split.metadata["record_overlap"] == []
    assert split.metadata["sample_overlap"] == []
    assert split.metadata["raw_interval_overlap"] == []


def test_shared_subject_split_removes_duplicates_symmetrically(
    cross_validator,
):
    split = cross_validator.run_cross_source_holdout(
        train_source="gpn_data",
        test_source="Old_EEG",
        subject_mode="shared_subject",
    )

    assert split.metadata["status"] == "invalid"
    assert len(split.metadata["excluded_logical_record_ids"]) == 3
    assert split.metadata["n_train_subjects"] == 1
    assert split.metadata["n_test_subjects"] == 1
    assert set(split.metadata["train_subject_ids"]) == {"X0"}
    assert set(split.metadata["test_subject_ids"]) == {"X0"}
    assert set(split.metadata["eligible_shared_subject_ids"]) == {"X0"}
    assert set(split.metadata["subject_overlap"]) == {"X0"}
    assert split.metadata["logical_record_overlap"] == []
    assert any("train subjects=1" in reason for reason in split.metadata["invalid_reasons"])
    assert any("test subjects=1" in reason for reason in split.metadata["invalid_reasons"])


def test_shared_subject_split_without_duplicate_removal_is_invalid(
    cross_validator,
):
    split = cross_validator.run_cross_source_holdout(
        train_source="gpn_data",
        test_source="Old_EEG",
        subject_mode="shared_subject",
        remove_logical_duplicates=False,
    )

    assert split.metadata["status"] == "invalid"
    assert set(split.metadata["logical_record_overlap"]) == {
        "duplicate_0", "duplicate_1", "duplicate_2"
    }
    assert any("logical recording overlap" in value for value in split.metadata["invalid_reasons"])


def test_cross_source_limits_are_deterministic_and_subject_balanced(
    cross_validator,
):
    kwargs = dict(
        train_source="gpn_data",
        test_source="Old_EEG",
        subject_mode="source_exclusive",
        max_train_windows=100,
        max_test_windows=60,
    )
    first = cross_validator.run_cross_source_holdout(**kwargs)
    second = cross_validator.run_cross_source_holdout(**kwargs)

    np.testing.assert_array_equal(first.sample_id_train, second.sample_id_train)
    np.testing.assert_array_equal(first.sample_id_test, second.sample_id_test)
    assert len(first.y_train) == 100
    assert len(first.y_test) == 60
    assert first.metadata["minimum_test_predictions_per_subject_actual"] == 20
    assert first.metadata["status"] == "valid"


def test_insufficient_protocol_is_explicit_not_silent(cross_validator):
    split = cross_validator.run_cross_source_holdout(
        train_source="Old_EEG",
        test_source="gpn_data",
        subject_mode="source_exclusive",
        minimum_train_subjects=5,
    )

    assert split.metadata["status"] == "invalid"
    assert split.metadata["invalid_reasons"] == [
        "train subjects=3 is below configured minimum 5"
    ]


def test_cross_source_rejects_missing_source(cross_validator):
    with pytest.raises(ValueError, match="sources are unavailable"):
        cross_validator.run_cross_source_holdout(
            train_source="missing",
            test_source="Old_EEG",
        )
