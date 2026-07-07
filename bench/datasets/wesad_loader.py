import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

from ..core.abstract_dataset import EEGData, BaseDataset

logger = logging.getLogger(__name__)


class WESADDataset(BaseDataset):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_path = Path(config.get('data_path', ''))
        self.subject_ids = config.get('subject_ids', None)
        self.modalities = config.get('modalities', ['chest', 'wrist'])
        self.signals = config.get('signals', {
            'chest': ['ACC', 'ECG', 'EDA', 'EMG', 'RESP', 'TEMP'],
            'wrist': ['ACC', 'BVP', 'EDA', 'TEMP']
        })
        self.labels = config.get('labels', [1, 2, 3, 4])
        self.sampling_rate = config.get('sampling_rate', 700)
        self.window_size = config.get('window_size', 256)
        self.step_size = config.get('step_size', 128)

        self._validate_path()

    def _validate_path(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"WESAD data path not found: {self.data_path}")

    def _get_subject_folders(self) -> List[Path]:
        if self.subject_ids:
            subjects = [f"S{id}" for id in self.subject_ids]
            return [p for p in self.data_path.iterdir() if p.is_dir() and p.name in subjects]
        else:
            return [p for p in self.data_path.iterdir() if p.is_dir() and p.name not in ['S1', 'S12']]

    def _load_subject_data(self, subject_folder: Path) -> Dict[str, Any]:
        pkl_file = subject_folder / f"{subject_folder.name}.pkl"

        if not pkl_file.exists():
            logger.warning(f"PKL file not found for {subject_folder.name}")
            return None

        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            logger.error(f"Failed to load {pkl_file}: {e}")
            return None

    def _extract_signals(self, data: Dict[str, Any], modalities: List[str]) -> Dict[str, np.ndarray]:
        signals = {}

        for modality in modalities:
            if modality in data['signal']:
                modality_data = data['signal'][modality]

                for signal_name in self.signals.get(modality, []):
                    if signal_name in modality_data:
                        signals[f"{modality}_{signal_name}"] = modality_data[signal_name]

        return signals

    def _get_labels(self, data: Dict[str, Any]) -> np.ndarray:
        labels = data.get('label', None)
        if labels is None:
            raise ValueError("Labels not found in data")
        return labels

    def _filter_by_labels(self, signals: Dict[str, np.ndarray], labels: np.ndarray) -> Tuple[
        Dict[str, np.ndarray], np.ndarray]:
        valid_mask = np.isin(labels, self.labels)

        filtered_signals = {}
        for name, signal in signals.items():
            if len(signal) == len(labels):
                filtered_signals[name] = signal[valid_mask]
            else:
                logger.warning(f"Signal {name} has different length: {len(signal)} vs {len(labels)}")
                filtered_signals[name] = signal[:len(labels)][valid_mask]

        return filtered_signals, labels[valid_mask]

    def _aggregate_windows(self, windowed_signals: Dict[str, np.ndarray]) -> np.ndarray:
        all_features = []
        n_windows = None

        for signal_name, windows in windowed_signals.items():
            if n_windows is None:
                n_windows = windows.shape[0]

            if windows.shape[0] != n_windows:
                windows = windows[:n_windows]

            for window_idx in range(n_windows):
                window = windows[window_idx]

                if len(window.shape) == 1:
                    features = [
                        np.mean(window),
                        np.std(window),
                        np.min(window),
                        np.max(window),
                        np.median(window),
                        np.percentile(window, 25),
                        np.percentile(window, 75),
                    ]
                else:
                    features = []
                    for channel_idx in range(window.shape[1]):
                        channel = window[:, channel_idx]
                        features.extend([
                            np.mean(channel),
                            np.std(channel),
                            np.min(channel),
                            np.max(channel),
                            np.median(channel),
                        ])

                all_features.append(features)

        if not all_features:
            return np.array([])

        return np.array(all_features)

    def _create_windows(self, signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        windowed_signals = {}
        min_length = min(len(signal) for signal in signals.values())

        for name, signal in signals.items():
            windows = []
            for start in range(0, min_length - self.window_size + 1, self.step_size):
                window = signal[start:start + self.window_size]
                if len(window) == self.window_size:
                    windows.append(window)

            if windows:
                windowed_signals[name] = np.array(windows)

        return windowed_signals

    def _get_labels_for_windows(self, labels: np.ndarray) -> np.ndarray:
        window_labels = []
        for start in range(0, len(labels) - self.window_size + 1, self.step_size):
            window_labels.append(labels[start])
        return np.array(window_labels[:len(window_labels)])

    def load(self) -> EEGData:
        logger.info("Loading WESAD dataset...")

        subject_folders = self._get_subject_folders()
        logger.info(f"Found {len(subject_folders)} subject folders")

        all_features = []
        all_labels = []
        all_subject_ids = []

        for subject_folder in subject_folders:
            subject_id = subject_folder.name
            logger.info(f"Loading subject {subject_id}...")

            subject_data = self._load_subject_data(subject_folder)

            if subject_data is None:
                continue

            signals = self._extract_signals(subject_data, self.modalities)

            if not signals:
                logger.warning(f"No signals found for {subject_id}")
                continue

            labels = self._get_labels(subject_data)

            filtered_signals, filtered_labels = self._filter_by_labels(signals, labels)

            if len(filtered_labels) == 0:
                logger.warning(f"No valid labels for {subject_id}")
                continue

            windowed_signals = self._create_windows(filtered_signals)

            if not windowed_signals:
                logger.warning(f"No windows created for {subject_id}")
                continue

            features = self._aggregate_windows(windowed_signals)

            if features.shape[0] == 0:
                logger.warning(f"No features extracted for {subject_id}")
                continue

            window_labels = self._get_labels_for_windows(filtered_labels)

            n_windows = min(features.shape[0], len(window_labels))

            all_features.append(features[:n_windows])
            all_labels.extend(window_labels[:n_windows])
            all_subject_ids.extend([subject_id] * n_windows)

        if all_features:
            X = np.vstack(all_features).astype(np.float32)
            y = np.array(all_labels[:X.shape[0]])
            subject_ids = np.array(all_subject_ids[:X.shape[0]])

            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            metadata = {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_subjects': len(np.unique(subject_ids)),
                'n_classes': len(np.unique(y)),
                'classes': np.unique(y).tolist(),
                'source': 'WESAD',
                'sampling_rate': self.sampling_rate,
                'modalities': self.modalities,
                'labels_used': self.labels,
                'window_size': self.window_size,
                'step_size': self.step_size
            }

            logger.info(f"Loaded WESAD dataset: {X.shape[0]} samples, {X.shape[1]} features, "
                        f"{len(np.unique(subject_ids))} subjects, {len(np.unique(y))} classes")

            return EEGData(
                data=X,
                labels=y,
                subject_ids=subject_ids,
                feature_names=feature_names,
                sampling_rate=self.sampling_rate,
                metadata=metadata
            )
        else:
            raise ValueError("No data loaded from WESAD dataset")

    def get_description(self) -> Dict[str, Any]:
        return {
            'name': self.__class__.__name__,
            'data_path': str(self.data_path),
            'subject_ids': self.subject_ids,
            'modalities': self.modalities,
            'labels': self.labels,
            'sampling_rate': self.sampling_rate,
            'window_size': self.window_size,
            'step_size': self.step_size,
            'signals': self.signals
        }
