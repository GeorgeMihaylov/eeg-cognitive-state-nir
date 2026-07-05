import numpy as np
from typing import Tuple, Optional


class FASTERArtifactRemoval:
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def detect_artifacts(self, data: np.ndarray) -> np.ndarray:
        """
        Найти артефактные эпохи.

        Args:
            data: (n_epochs, n_channels, n_timesteps)

        Returns:
            boolean массив (n_epochs,) — True для артефактов
        """
        n_epochs, n_channels, n_timesteps = data.shape
        artifacts = np.zeros(n_epochs, dtype=bool)

        for epoch_idx in range(n_epochs):
            epoch_data = data[epoch_idx]

            for channel_idx in range(n_channels):
                channel_data = epoch_data[channel_idx]
                mean = np.mean(channel_data)
                std = np.std(channel_data)

                if std > 0:
                    z_scores = np.abs((channel_data - mean) / std)
                    if np.any(z_scores > self.threshold):
                        artifacts[epoch_idx] = True
                        break

        return artifacts

    def remove(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[
        np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Удалить артефактные эпохи.

        Returns:
            (очищенные данные, очищенные метки, маска артефактов)
        """
        artifacts = self.detect_artifacts(data)
        clean_mask = ~artifacts

        if labels is not None:
            return data[clean_mask], labels[clean_mask], artifacts
        return data[clean_mask], None, artifacts

    def get_artifact_percentage(self, artifacts: np.ndarray) -> float:
        return (np.sum(artifacts) / len(artifacts)) * 100