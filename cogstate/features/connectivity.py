"""
connectivity.py — связностные признаки ЭЭГ (10.2.3).

Характеризуют не отдельный канал, а взаимодействие между парами
каналов — это отдельный класс признаков, дополняющий спектральные и
статистические: рост синхронизации между областями коры часто
сопровождает изменения когнитивной нагрузки и внимания.

Реализованы три меры попарной связности:
    - когерентность (coherence) по частотным ритмам;
    - phase locking value (PLV) — синхронность фазы, независимо от амплитуды;
    - корреляция Пирсона во временной области (самый дешёвый вариант).

Число пар каналов растёт квадратично, поэтому матрицы связности не
идут в модель как есть — сводятся к скалярным сводкам (среднее,
максимум) на уровне окна, см. summarize_connectivity_matrix().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import coherence, hilbert

from .spectral import DEFAULT_BANDS


@dataclass
class ConnectivityConfig:
    sample_rate: float
    bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_BANDS))
    nperseg: int = 128
    max_channel_pairs: int = 50   # ограничение на число пар для полной матрицы (вычислительный бюджет)


def compute_correlation_matrix(window: np.ndarray) -> np.ndarray:
    """window: [n_samples, n_channels] -> [n_channels, n_channels] матрица корреляции Пирсона."""
    return np.corrcoef(window, rowvar=False)


def compute_coherence_matrix(window: np.ndarray, config: ConnectivityConfig, band: Tuple[float, float]) -> np.ndarray:
    """
    Матрица средней когерентности в заданной полосе частот между
    всеми парами каналов. Симметрична, диагональ = 1.
    """
    n_channels = window.shape[1]
    matrix = np.eye(n_channels)
    nperseg = min(config.nperseg, window.shape[0])

    pairs = list(combinations(range(n_channels), 2))[: config.max_channel_pairs]
    for ch1, ch2 in pairs:
        freqs, coh = coherence(window[:, ch1], window[:, ch2], fs=config.sample_rate, nperseg=nperseg)
        mask = (freqs >= band[0]) & (freqs <= band[1])
        value = np.mean(coh[mask]) if np.any(mask) else 0.0
        matrix[ch1, ch2] = matrix[ch2, ch1] = value

    return matrix


def compute_plv_matrix(window: np.ndarray, config: ConnectivityConfig) -> np.ndarray:
    """
    Phase Locking Value между всеми парами каналов на основе фазы
    аналитического сигнала (преобразование Гильберта). В отличие от
    когерентности не чувствителен к амплитудным различиям между
    каналами — устойчивее при разном контакте электродов.
    """
    n_channels = window.shape[1]
    phases = np.angle(hilbert(window, axis=0))  # [n_samples, n_channels]

    matrix = np.eye(n_channels)
    pairs = list(combinations(range(n_channels), 2))[: config.max_channel_pairs]
    for ch1, ch2 in pairs:
        phase_diff = phases[:, ch1] - phases[:, ch2]
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        matrix[ch1, ch2] = matrix[ch2, ch1] = plv

    return matrix


def summarize_connectivity_matrix(matrix: np.ndarray) -> Dict[str, float]:
    """
    Свести полную матрицу связности к нескольким скалярам —
    признакам уровня окна, а не пары каналов, чтобы не раздувать
    размерность признакового пространства с ростом числа электродов.
    """
    n = matrix.shape[0]
    off_diagonal = matrix[~np.eye(n, dtype=bool)]
    if off_diagonal.size == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(off_diagonal)),
        "std": float(np.std(off_diagonal)),
        "max": float(np.max(off_diagonal)),
    }


def extract_connectivity_features(window: np.ndarray, config: ConnectivityConfig) -> Dict[str, np.ndarray]:
    """
    window: [n_samples, n_channels]
    return: {имя_признака: скаляр} — по одной сводной статистике на
    (метод связности × ритм), а не полная матрица, для совместимости
    с плоским вектором признаков окна (features/pipeline.py).
    """
    features: Dict[str, np.ndarray] = {}

    corr_summary = summarize_connectivity_matrix(compute_correlation_matrix(window))
    for stat_name, value in corr_summary.items():
        features[f"correlation_{stat_name}"] = np.array([value])

    for band_name, band in config.bands.items():
        coh_summary = summarize_connectivity_matrix(compute_coherence_matrix(window, config, band))
        for stat_name, value in coh_summary.items():
            features[f"coherence_{band_name}_{stat_name}"] = np.array([value])

    plv_summary = summarize_connectivity_matrix(compute_plv_matrix(window, config))
    for stat_name, value in plv_summary.items():
        features[f"plv_{stat_name}"] = np.array([value])

    return features


def feature_names(config: ConnectivityConfig) -> List[str]:
    names = [f"correlation_{s}" for s in ("mean", "std", "max")]
    for band_name in config.bands:
        names += [f"coherence_{band_name}_{s}" for s in ("mean", "std", "max")]
    names += [f"plv_{s}" for s in ("mean", "std", "max")]
    return names
