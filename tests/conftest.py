import sys
import os
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import pytest
import numpy as np

np.random.seed(42)

@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(42)
    yield


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )