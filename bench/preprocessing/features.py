import numpy as np
from scipy import signal
from typing import List, Dict, Tuple


class EEGFeatureExtractor:
    def __init__(
            self,
            sampling_rate: float = 128.0,
            freq_bands: Dict[str, Tuple[float, float]] = None,
            window_s: float = 2.0
    ):
        self.sampling_rate = sampling_rate
        self.window_s = window_s

        if freq_bands is None:
            self.freq_bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 50)
            }
        else:
            self.freq_bands = freq_bands

    def extract_spectral_features(self, data: np.ndarray) -> np.ndarray:
        """
        Извлечь спектральные признаки.

        Args:
            data: (n_epochs, n_channels, n_timesteps)

        Returns:
            (n_epochs, n_spectral_features)
        """
        n_epochs, n_channels, n_timesteps = data.shape

        features = []
        for epoch_idx in range(n_epochs):
            epoch_features = []
            for channel_idx in range(n_channels):
                channel_data = data[epoch_idx, channel_idx]

                # Welch PSD
                freqs, psd = signal.welch(
                    channel_data,
                    fs=self.sampling_rate,
                    nperseg=min(256, len(channel_data)),
                    noverlap=min(128, len(channel_data) // 2)
                )

                for band_name, (low, high) in self.freq_bands.items():
                    mask = (freqs >= low) & (freqs < high)
                    if np.any(mask):
                        power = np.mean(psd[mask])
                        epoch_features.append(power)
                    else:
                        epoch_features.append(0.0)

            features.append(epoch_features)

        return np.array(features)

    def extract_statistical_features(self, data: np.ndarray) -> np.ndarray:
        """
        Извлечь статистические признаки во временном домене

        Args:
            data: (n_epochs, n_channels, n_timesteps)

        Returns:
            (n_epochs, n_stat_features)
        """
        n_epochs, n_channels, n_timesteps = data.shape

        features = []
        for epoch_idx in range(n_epochs):
            epoch_features = []
            for channel_idx in range(n_channels):
                channel_data = data[epoch_idx, channel_idx]

                epoch_features.append(np.mean(channel_data))
                epoch_features.append(np.std(channel_data))
                epoch_features.append(np.min(channel_data))
                epoch_features.append(np.max(channel_data))
                epoch_features.append(np.median(channel_data))
                epoch_features.append(np.mean(np.square(channel_data)))  # energy

            features.append(epoch_features)

        return np.array(features)

    def extract_connectivity_features(self, data: np.ndarray) -> np.ndarray:
        """
        Извлечь признаки функциональной связности

        Args:
            data: (n_epochs, n_channels, n_timesteps)

        Returns:
            (n_epochs, n_connectivity_features)
        """
        n_epochs, n_channels, n_timesteps = data.shape

        features = []
        for epoch_idx in range(n_epochs):
            epoch_data = data[epoch_idx]
            epoch_features = []
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    corr = np.corrcoef(epoch_data[i], epoch_data[j])[0, 1]
                    if not np.isnan(corr):
                        epoch_features.append(corr)
                    else:
                        epoch_features.append(0.0)

            features.append(epoch_features)

        return np.array(features)

    def extract_all_features(self, data: np.ndarray) -> np.ndarray:
        spectral = self.extract_spectral_features(data)
        statistical = self.extract_statistical_features(data)
        connectivity = self.extract_connectivity_features(data)

        return np.hstack([spectral, statistical, connectivity])