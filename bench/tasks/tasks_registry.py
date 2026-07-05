from typing import Dict, Type, Any
from .base import BaseEEGTask
from .cognitive_load import CognitiveLoadTask
from ..core.task import BaseTask
from ..core.dataset import EEGData

TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    'cognitive_load_3class': CognitiveLoadTask,
}


def get_task(name: str, data: EEGData, config: Dict[str, Any]) -> BaseTask:
    if name not in TASK_REGISTRY:
        raise ValueError(
            f"Task '{name}' not found. Available: {list(TASK_REGISTRY.keys())}"
        )

    task_class = TASK_REGISTRY[name]
    return task_class(data, config)
