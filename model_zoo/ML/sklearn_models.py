from typing import Any, Dict, Mapping, Optional

from sklearn.base import BaseEstimator


CLASSIFICATION_MODEL_NAMES = frozenset(
    {"logistic_regression", "mlp", "random_forest", "svm", "xgboost"}
)
REGRESSION_MODEL_NAMES = frozenset(
    {"mean_regressor", "mlp", "random_forest", "svm", "xgboost"}
)
SKLEARN_MODEL_NAMES = CLASSIFICATION_MODEL_NAMES | REGRESSION_MODEL_NAMES


def _normalize_task_type(task_type: str) -> str:
    normalized = task_type.strip().lower()
    aliases = {
        "classification": "classification",
        "classifier": "classification",
        "regression": "regression",
        "regressor": "regression",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "task_type must be 'classification' or 'regression', "
            f"got {task_type!r}"
        ) from exc


def build_sklearn_model(
    model_name: str,
    task_type: str,
    params: Optional[Mapping[str, Any]] = None,
) -> BaseEstimator:
    """Create a supported sklearn-compatible estimator."""
    normalized_name = model_name.strip().lower()
    normalized_task = _normalize_task_type(task_type)
    model_params: Dict[str, Any] = dict(params or {})

    supported_names = (
        CLASSIFICATION_MODEL_NAMES
        if normalized_task == "classification"
        else REGRESSION_MODEL_NAMES
    )
    if normalized_name not in supported_names:
        raise ValueError(
            f"Model {model_name!r} is not available for {normalized_task}. "
            f"Available: {sorted(supported_names)}"
        )

    if normalized_task == "classification":
        if normalized_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(**model_params)
        if normalized_name == "svm":
            from sklearn.svm import SVC

            return SVC(**model_params)
        if normalized_name == "logistic_regression":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(**model_params)
        if normalized_name == "mlp":
            from sklearn.neural_network import MLPClassifier

            return MLPClassifier(**model_params)
        if normalized_name == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(**model_params)

    if normalized_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**model_params)
    if normalized_name == "mean_regressor":
        from sklearn.dummy import DummyRegressor

        model_params.setdefault("strategy", "mean")
        return DummyRegressor(**model_params)
    if normalized_name == "svm":
        from sklearn.svm import SVR

        return SVR(**model_params)
    if normalized_name == "mlp":
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(**model_params)
    if normalized_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**model_params)

    raise AssertionError(f"Unhandled sklearn model: {normalized_name}")
