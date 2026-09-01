from typing import Dict, Type, Any
from .cognitive_load import (
    CognitiveLoad5ClassTask,
    CognitiveLoadTask,
    FoldLocalQ3Task,
    FocusRegressionTask,
    PerformanceMetricsRegressionTask,
)
from .wesad_task import WESADTask
from ..core.abstract_task import BaseTask
from ..core.abstract_dataset import EEGData

TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    'cognitive_load_3class': CognitiveLoadTask,
    'cognitive_load_5class': CognitiveLoad5ClassTask,
    'focus_regression': FocusRegressionTask,
    'performance_metrics_regression': PerformanceMetricsRegressionTask,
    'wesad_4class': WESADTask,
    'pm_attention_regression': FocusRegressionTask,
    'pm_engagement_regression': FocusRegressionTask,
    'pm_excitement_regression': FocusRegressionTask,
    'pm_stress_regression': FocusRegressionTask,
    'pm_relaxation_regression': FocusRegressionTask,
    'pm_interest_regression': FocusRegressionTask,
    'pm_focus_regression': FocusRegressionTask,
    'pm_multioutput_regression_7': PerformanceMetricsRegressionTask,
    'label_focus_q5_legacy': CognitiveLoad5ClassTask,
    **{
        f'pm_{metric}_q3_fold_local': FoldLocalQ3Task
        for metric in (
            'attention', 'engagement', 'excitement', 'stress',
            'relaxation', 'interest', 'focus',
        )
    },
}

TASK_TARGET_IDS = {
    'cognitive_load_5class': 'label_focus_q5_legacy',
    'focus_regression': 'pm_focus_regression',
    'performance_metrics_regression': 'pm_multioutput_regression_7',
    'label_focus_q5_legacy': 'label_focus_q5_legacy',
    'pm_multioutput_regression_7': 'pm_multioutput_regression_7',
    **{
        f'pm_{metric}_regression': f'pm_{metric}_regression'
        for metric in (
            'attention', 'engagement', 'excitement', 'stress',
            'relaxation', 'interest', 'focus',
        )
    },
    **{
        f'pm_{metric}_q3_fold_local': f'pm_{metric}_q3_fold_local'
        for metric in (
            'attention', 'engagement', 'excitement', 'stress',
            'relaxation', 'interest', 'focus',
        )
    },
}


def get_task(name: str, data: EEGData, config: Dict[str, Any]) -> BaseTask:
    if name not in TASK_REGISTRY:
        raise ValueError(
            f"Task '{name}' not found. Available: {list(TASK_REGISTRY.keys())}"
        )

    expected_target_id = TASK_TARGET_IDS.get(name)
    actual_target_id = data.metadata.get('target_id')
    configured_target_id = config.get('target_id')
    if (
        configured_target_id is not None
        and expected_target_id is not None
        and str(configured_target_id) != expected_target_id
    ):
        raise ValueError(
            f"Task {name!r} requires target_id {expected_target_id!r}, got "
            f"{configured_target_id!r}"
        )
    if (
        actual_target_id is not None
        and expected_target_id is not None
        and str(actual_target_id) != expected_target_id
    ):
        raise ValueError(
            f"Task {name!r} requires target_id {expected_target_id!r}, but "
            f"dataset provides {actual_target_id!r}"
        )

    task_class = TASK_REGISTRY[name]
    return task_class(data, config)
