import numpy as np
import torch
from sklearn.model_selection import train_test_split

from cogstate.model_zoo import build_model
from cogstate.model_zoo.DL import TorchClassificationAdapter, TorchLSTMClassifier


def _classification_sequences() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    y = np.tile(np.arange(5), 20).astype(np.int64)
    X = rng.normal(size=(len(y), 4, 6)).astype(np.float32)
    X[:, :, 0] += y[:, None]
    return X, y


def _grouped_classification_sequences() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(123)
    n_records = 12
    sequences_per_record = 10
    record_ids = np.repeat(
        [f"R{record_index:02d}" for record_index in range(n_records)],
        sequences_per_record,
    )
    subject_ids = np.repeat(
        [f"S{record_index // 2:02d}" for record_index in range(n_records)],
        sequences_per_record,
    )
    y = np.tile(np.arange(5), n_records * 2).astype(np.int64)
    X = rng.normal(size=(len(y), 4, 6)).astype(np.float32)
    X[:, :, 0] += y[:, None]
    return X, y, record_ids, subject_ids


def _small_lstm(validation_seed: int = 42) -> TorchClassificationAdapter:
    model = build_model(
        "torch_lstm",
        "classification",
        input_shape=(4, 6),
        num_outputs=5,
        params={
            "hidden_size": 8,
            "classifier_hidden": 6,
            "batch_size": 32,
            "max_epochs": 1,
            "validation_size": 0.25,
            "early_stopping_patience": 1,
            "device": "cpu",
            "random_state": 42,
        },
    )
    model.validation_random_state_ = validation_seed
    return model


def test_lstm_and_bilstm_forward_passes() -> None:
    X = torch.randn(7, 10, 12)
    lstm = TorchLSTMClassifier(12, 5, hidden_size=8, classifier_hidden=6)
    bilstm = TorchLSTMClassifier(
        12, 5, hidden_size=8, classifier_hidden=6, bidirectional=True
    )

    assert lstm(X).shape == (7, 5)
    assert bilstm(X).shape == (7, 5)
    assert bilstm.classifier[0].in_features == 16


def test_factory_builds_lstm_and_bilstm() -> None:
    common = {
        "hidden_size": 8,
        "classifier_hidden": 6,
        "max_epochs": 1,
        "device": "cpu",
    }
    lstm = build_model("torch_lstm", "classification", (4, 6), 5, common)
    bilstm = build_model("torch_bilstm", "classification", (4, 6), 5, common)

    assert isinstance(lstm, TorchClassificationAdapter)
    assert isinstance(bilstm, TorchClassificationAdapter)
    assert lstm.model.bidirectional is False
    assert bilstm.model.bidirectional is True


def test_lstm_fit_predict_proba_and_checkpoint(tmp_path) -> None:
    X, y = _classification_sequences()
    model = build_model(
        "torch_lstm",
        "classification",
        input_shape=(4, 6),
        num_outputs=5,
        params={
            "hidden_size": 12,
            "classifier_hidden": 8,
            "batch_size": 20,
            "max_epochs": 2,
            "validation_size": 0.2,
            "early_stopping_patience": 2,
            "device": "cpu",
            "random_state": 42,
        },
    )

    model.fit(X, y)
    predictions = model.predict(X[:11])
    probabilities = model.predict_proba(X[:11])
    checkpoint = tmp_path / "model.pt"
    model.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")

    assert predictions.shape == (11,)
    assert probabilities.shape == (11, 5)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert payload["input_shape"] == (4, 6)
    assert payload["feature_mean"].shape == (6,)
    assert payload["model_metadata"]["model_type"] == "torch_lstm"


def test_3d_scaler_uses_feature_dimension_and_inner_train_only() -> None:
    X, y = _classification_sequences()
    model = build_model(
        "torch_lstm",
        "classification",
        input_shape=(4, 6),
        num_outputs=5,
        params={
            "hidden_size": 8,
            "classifier_hidden": 6,
            "batch_size": 32,
            "max_epochs": 1,
            "validation_size": 0.2,
            "early_stopping_patience": 1,
            "device": "cpu",
            "random_state": 42,
        },
    )
    inner_train, _, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y,
    )

    model.fit(X, y)

    expected_mean = inner_train.reshape(-1, X.shape[-1]).mean(axis=0)
    assert model.feature_mean_.shape == (6,)
    np.testing.assert_allclose(
        model.feature_mean_, expected_mean, rtol=1e-6, atol=1e-7
    )


def test_group_record_validation_is_disjoint_and_scaler_uses_inner_train() -> None:
    X, y, record_ids, subject_ids = _grouped_classification_sequences()
    model = _small_lstm()
    model.set_validation_groups(
        record_ids,
        subject_ids=subject_ids,
        record_ids=record_ids,
        outer_test_record_ids=np.asarray(["OUTER_R01", "OUTER_R02"]),
        validation_size=0.25,
        random_state=42,
    )

    model.fit(X, y)

    split = model.validation_split_
    assert split is not None
    assert split["record_overlap"] == []
    assert split["inner_record_overlap"] == 0
    assert split["outer_test_record_overlap"] == []
    assert set(split["inner_train_record_ids"]).isdisjoint(
        split["inner_validation_record_ids"]
    )
    train_idx = model.inner_train_indices_
    assert train_idx is not None
    expected_mean = X[train_idx].reshape(-1, X.shape[-1]).mean(axis=0)
    np.testing.assert_allclose(
        model.feature_mean_, expected_mean, rtol=1e-6, atol=1e-7
    )


def test_group_record_validation_is_reproducible_and_seeded() -> None:
    _, y, record_ids, subject_ids = _grouped_classification_sequences()

    def validation_records(seed: int) -> set[str]:
        model = _small_lstm()
        model.set_validation_groups(
            record_ids,
            subject_ids=subject_ids,
            record_ids=record_ids,
            validation_size=0.25,
            random_state=seed,
        )
        _, validation_idx = model._validation_indices(y)
        return set(record_ids[validation_idx])

    first = validation_records(42)
    repeated = validation_records(42)
    changed = validation_records(7)

    assert first == repeated
    assert first != changed


def test_group_record_validation_rejects_too_few_records() -> None:
    _, y, _, subject_ids = _grouped_classification_sequences()
    one_record = np.full(len(y), "R00")
    model = _small_lstm()
    model.set_validation_groups(
        one_record,
        subject_ids=subject_ids,
        record_ids=one_record,
    )

    with np.testing.assert_raises_regex(ValueError, "at least two unique records"):
        model._validation_indices(y)
