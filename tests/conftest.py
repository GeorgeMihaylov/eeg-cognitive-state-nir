# tests/conftest.py
"""
Конфигурация pytest
"""

import sys
import os
from pathlib import Path

# Добавление корневой директории проекта в PYTHONPATH
# Получаем корневую директорию проекта (родительскую для tests)
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Опционально: можно добавить и другие пути
# sys.path.insert(0, str(root_dir / 'src'))

import pytest
import numpy as np


# Настройка random seed для воспроизводимости
np.random.seed(42)


@pytest.fixture(autouse=True)
def set_random_seed():
    """Автоматическая установка random seed для всех тестов"""
    np.random.seed(42)
    yield


def pytest_configure(config):
    """Настройка маркеров для pytest"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )