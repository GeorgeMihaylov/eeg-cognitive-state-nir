import numpy as np
from typing import Optional, List, Dict, Any
from .filters import EEGFilter
from .artifacts import FASTERArtifactRemoval
from .features import EEGFeatureExtractor


class PreprocessingPipeline:
    def __init__(
            self,
            filter_config: Optional[Dict[str, Any]] = None,
            artifact_config: Optional[Dict[str, Any]] = None,
            feature_config: Optional[Dict[str, Any]] = None
    ):
        filter_config = filter_config or {}
        artifact_config = artifact_config or {}
        feature_config = feature_config or {}

        self.filter = EEGFilter(**filter_config)
        self.artifact_removal = FASTERArtifactRemoval(**artifact_config)
        self.feature_extractor = EEGFeatureExtractor(**feature_config)

    def fit_transform(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[
        np.ndarray, Optional[np.ndarray]]:
        """
        Выполнить весь пайплайн препроцессинга

        Args:
            data: (n_epochs, n_channels, n_timesteps)
            labels: (n_epochs,)

        Returns:
            (признаки, очищенные метки)
        """
        data_filtered = self.filter.apply(data)
        data_clean, labels_clean, artifacts = self.artifact_removal.remove(
            data_filtered, labels
        )
        features = self.feature_extractor.extract_all_features(data_clean)

        return features, labels_clean

    def get_params(self) -> Dict[str, Any]:
        return {
            'filter': self.filter.get_params(),
            'artifact_threshold': self.artifact_removal.threshold,
            'feature_bands': self.feature_extractor.freq_bands
        }