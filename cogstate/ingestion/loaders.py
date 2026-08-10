"""Small, format-agnostic loaders for tabular biosignal exports."""
from pathlib import Path
from typing import Iterable
import numpy as np


def load_timeseries(path: str | Path, *, delimiter: str = ",", skip_header: int = 0) -> np.ndarray:
    """Load CSV/TSV numeric data as ``[samples, channels]``."""
    array = np.genfromtxt(Path(path), delimiter=delimiter, skip_header=skip_header)
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not len(array):
        raise ValueError("A non-empty two-dimensional time series is required")
    return array


def load_eeg(path: str | Path, *, delimiter: str = ",", skip_header: int = 0) -> np.ndarray:
    return load_timeseries(path, delimiter=delimiter, skip_header=skip_header)


def load_behavior_log(path: str | Path, *, delimiter: str = ",", skip_header: int = 1) -> np.ndarray:
    return load_timeseries(path, delimiter=delimiter, skip_header=skip_header)
