import numpy as np
from scipy import signal
from typing import Tuple


class EEGFilter:
    def __init__(
            self,
            highpass: float = 1.0,
            lowpass: float = 50.0,
            notch: float = 50.0,
            sampling_rate: float = 128.0,
            order: int = 4
    ):
        self.highpass = highpass
        self.lowpass = lowpass
        self.notch = notch
        self.sampling_rate = sampling_rate
        self.order = order

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Применить фильтры к данным.

        Args:
            data: (n_samples, n_channels, n_timesteps) или (n_channels, n_timesteps)

        Returns:
            Отфильтрованные данные
        """
        original_shape = data.shape

        if data.ndim == 3:
            n_samples, n_channels, n_timesteps = data.shape
            data_flat = data.reshape(-1, n_timesteps)
        else:
            n_channels, n_timesteps = data.shape
            data_flat = data

        filtered = []
        for channel_data in data_flat:
            if self.highpass:
                b, a = signal.butter(
                    self.order,
                    self.highpass / (self.sampling_rate / 2),
                    btype='high'
                )
                channel_data = signal.filtfilt(b, a, channel_data)

            if self.lowpass:
                b, a = signal.butter(
                    self.order,
                    self.lowpass / (self.sampling_rate / 2),
                    btype='low'
                )
                channel_data = signal.filtfilt(b, a, channel_data)

            if self.notch:
                b, a = signal.iirnotch(self.notch, 30, self.sampling_rate)
                channel_data = signal.filtfilt(b, a, channel_data)

            filtered.append(channel_data)

        filtered = np.array(filtered)

        if data.ndim == 3:
            filtered = filtered.reshape(n_samples, n_channels, -1)

        return filtered

    def get_params(self) -> dict:
        return {
            'highpass': self.highpass,
            'lowpass': self.lowpass,
            'notch': self.notch,
            'sampling_rate': self.sampling_rate,
            'order': self.order
        }