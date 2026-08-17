"""
entropy.py — энтропийные признаки ЭЭГ (10.2.3).

Энтропийные меры оценивают сложность/непредсказуемость сигнала —
класс признаков, отдельный от спектральных и статистических, но
хорошо зарекомендовавший себя при анализе ЭЭГ (снижение энтропии
часто связывают с ростом когнитивной нагрузки и синхронизацией
нейронной активности).

Реализованы три меры:
    - спектральная энтропия  — "плоскостность" распределения мощности по частотам;
    - энтропия перестановок  — сложность порядка значений во времени, устойчива к шуму;
    - sample entropy         — регулярность/предсказуемость временного ряда.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List

import numpy as np
from scipy.signal import welch
from scipy.special import xlogy

from .spectral import PowerSpectrum


@dataclass
class EntropyConfig:
    sample_rate: float
    permutation_order: int = 3      # длина паттерна для энтропии перестановок
    permutation_delay: int = 1
    sample_entropy_m: int = 2       # длина шаблона для sample entropy
    sample_entropy_r_ratio: float = 0.2  # порог как доля от std сигнала
    spectral_low_hz: float = 1.0
    spectral_high_hz: float | None = None

    def __post_init__(self) -> None:
        high = self.spectral_high_hz or self.sample_rate / 2.0
        if self.sample_rate <= 0 or not 0 <= self.spectral_low_hz < high <= self.sample_rate / 2.0:
            raise ValueError("Invalid spectral entropy frequency range")


def spectral_entropy_from_psd(
    frequencies: np.ndarray,
    psd: np.ndarray,
    *,
    band: tuple[float, float],
) -> np.ndarray:
    """Normalized Shannon entropy for every PSD column."""
    mask = (frequencies >= band[0]) & (frequencies <= band[1])
    selected = np.asarray(psd[mask], dtype=float)
    if selected.shape[0] < 2:
        return np.zeros(psd.shape[1])
    probabilities = selected / np.maximum(
        np.sum(selected, axis=0, keepdims=True), np.finfo(float).tiny
    )
    entropy = -np.sum(xlogy(probabilities, probabilities), axis=0) / np.log(2.0)
    return entropy / np.log2(selected.shape[0])


def spectral_entropy_1d(x: np.ndarray, sample_rate: float) -> float:
    """Энтропия Шеннона нормированного спектра мощности одного канала."""
    freqs, psd = welch(x, fs=sample_rate, nperseg=min(len(x), 256))
    value = spectral_entropy_from_psd(
        freqs, psd[:, None], band=(1.0, sample_rate / 2.0)
    )
    return float(value[0])


def permutation_entropy_1d(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """
    Энтропия распределения "ординальных паттернов" — относительного
    порядка `order` последовательных (с шагом delay) отсчётов.
    Устойчива к монотонным нелинейным искажениям и умеренному шуму,
    что удобно для сигналов с потребительских EEG-гарнитур.
    """
    n = len(x)
    if n < order * delay + 1:
        return 0.0

    all_patterns = list(permutations(range(order)))
    pattern_index = {p: i for i, p in enumerate(all_patterns)}
    counts = np.zeros(len(all_patterns))

    for i in range(n - (order - 1) * delay):
        window = x[i: i + order * delay: delay]
        pattern = tuple(np.argsort(window))
        counts[pattern_index[pattern]] += 1

    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy / np.log2(len(all_patterns)))  # нормировка на [0, 1]


def sample_entropy_1d(x: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    """
    Sample entropy (Richman & Moorman, 2000): -log(A/B), где B — число
    похожих пар шаблонов длины m, A — длины m+1, "похожесть" по
    Чебышёвскому расстоянию < r. Не учитывает самосовпадения (в
    отличие от approximate entropy), что даёт менее смещённую оценку.

    Вычислительно затратна (O(n^2)) — приемлемо для окон в несколько
    секунд при типичной частоте дискретизации потребительских
    EEG-гарнитур (128-256 Гц), но не для длинных записей целиком.
    """
    n = len(x)
    if n < m + 2:
        return 0.0

    r = r_ratio * np.std(x)
    if r == 0:
        return 0.0

    def _count_matches(template_len: int) -> int:
        templates = np.array([x[i:i + template_len] for i in range(n - template_len + 1)])
        count = 0
        for i in range(len(templates)):
            dist = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1) if i + 1 < len(templates) else np.array([])
            count += np.sum(dist < r)
        return count

    b = _count_matches(m)
    a = _count_matches(m + 1)

    if b == 0 or a == 0:
        return 0.0
    return float(-np.log(a / b))


def extract_entropy_features(
    window: np.ndarray,
    config: EntropyConfig,
    *,
    spectrum: PowerSpectrum | None = None,
) -> Dict[str, np.ndarray]:
    """
    window: [n_samples, n_channels]
    return: {имя_признака: [n_channels]}
    """
    n_channels = window.shape[1]

    if spectrum is None:
        spectral_ent = np.array(
            [
                spectral_entropy_1d(window[:, ch], config.sample_rate)
                for ch in range(n_channels)
            ]
        )
    else:
        spectral_ent = spectral_entropy_from_psd(
            spectrum.frequencies,
            spectrum.psd,
            band=(
                config.spectral_low_hz,
                config.spectral_high_hz or config.sample_rate / 2.0,
            ),
        )
    permutation_ent = np.array([
        permutation_entropy_1d(window[:, ch], config.permutation_order, config.permutation_delay)
        for ch in range(n_channels)
    ])
    sample_ent = np.array([
        sample_entropy_1d(window[:, ch], config.sample_entropy_m, config.sample_entropy_r_ratio)
        for ch in range(n_channels)
    ])

    return {
        "spectral_entropy": spectral_ent,
        "permutation_entropy": permutation_ent,
        "sample_entropy": sample_ent,
    }


def feature_names(config: EntropyConfig) -> List[str]:
    return ["spectral_entropy", "permutation_entropy", "sample_entropy"]
