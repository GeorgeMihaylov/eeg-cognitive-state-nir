from typing import Dict, Type, Any
from ..core.abstract_dataset import BaseDataset

# Используем этот файл для централизованной регистрации датасетов

DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {}


def get_dataset(name: str, config: Dict[str, Any]) -> BaseDataset:
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{name}' not found. Available: {list(DATASET_REGISTRY.keys())}"
        )

    dataset_class = DATASET_REGISTRY[name]

    if 'data_path' not in config:
        raise ValueError(f"data_path is required for dataset '{name}'")

    return dataset_class(config)
