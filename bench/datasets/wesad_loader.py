import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy import signal
from scipy.stats import entropy

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
        self.target_sampling_rate = config.get('target_sampling_rate', 700)
        self.original_sampling_rates = {
            'chest': 700,
            'wrist': 64
        }
        self.window_size = config.get('window_size', 256)
        self.step_size = config.get('step_size', 128)
        self.spectral_bands = config.get('spectral_bands', {
            'total': (0.5, 50),
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 50)
        })
        self._validate_path()
        self.feature_names = None

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

    def _extract_signals(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        signals = {}
        for modality in self.modalities:
            if modality in data['signal']:
                modality_data = data['signal'][modality]
                for signal_name in self.signals.get(modality, []):
                    if signal_name in modality_data:
                        key = f"{modality}_{signal_name}"
                        signals[key] = modality_data[signal_name]
        return signals

    def _resample_signals(self, signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        resampled = {}
        target_fs = self.target_sampling_rate
        for name, data in signals.items():
            modality = name.split('_')[0]
            orig_fs = self.original_sampling_rates.get(modality, 700)
            if orig_fs == target_fs:
                resampled[name] = data
            else:
                up = target_fs
                down = orig_fs
                g = np.gcd(up, down)
                up //= g
                down //= g
                resampled[name] = signal.resample_poly(data, up, down)
        if resampled:
            min_len = min(len(v) for v in resampled.values())
            for name in resampled:
                if len(resampled[name]) > min_len:
                    resampled[name] = resampled[name][:min_len]
        return resampled

    def _get_labels(self, data: Dict[str, Any]) -> np.ndarray:
        labels = data.get('label', None)
        if labels is None:
            raise ValueError("Labels not found in data")
        return labels

    def _filter_by_labels(self, signals: Dict[str, np.ndarray], labels: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        valid_mask = np.isin(labels, self.labels)
        for name, sig in signals.items():
            if len(sig) != len(labels):
                if len(sig) > len(labels):
                    signals[name] = sig[:len(labels)]
                else:
                    signals[name] = np.pad(sig, (0, len(labels) - len(sig)))
        filtered_signals = {name: sig[valid_mask] for name, sig in signals.items()}
        return filtered_signals, labels[valid_mask]

    def _create_windows(self, signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not signals:
            return {}
        min_length = min(len(sig) for sig in signals.values())
        if min_length < self.window_size:
            logger.warning(f"Signal too short ({min_length}) for window_size={self.window_size}")
            return {}
        windowed = {}
        for name, sig in signals.items():
            windows = []
            for start in range(0, min_length - self.window_size + 1, self.step_size):
                window = sig[start:start + self.window_size]
                if len(window) == self.window_size:
                    windows.append(window)
            if windows:
                windowed[name] = np.array(windows)
        return windowed

    def _extract_spectral_features(self, window: np.ndarray, fs: float) -> Dict[str, float]:
        freqs, psd = signal.welch(window, fs=fs, nperseg=min(64, len(window)), noverlap=0)
        spectral_features = {}
        for band_name, (low, high) in self.spectral_bands.items():
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                band_power = np.trapz(psd[mask], freqs[mask])
                spectral_features[band_name] = band_power
            else:
                spectral_features[band_name] = 0.0
        psd_norm = psd / (np.sum(psd) + 1e-12)
        spectral_entropy = entropy(psd_norm)
        spectral_features['spectral_entropy'] = spectral_entropy
        return spectral_features

    def _aggregate_windows(self, windowed_signals: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        if not windowed_signals:
            return np.array([]), []
        n_windows = min(windows.shape[0] for windows in windowed_signals.values())
        for name in windowed_signals:
            if windowed_signals[name].shape[0] > n_windows:
                windowed_signals[name] = windowed_signals[name][:n_windows]

        all_features = []
        feature_names = []

        for name, windows in windowed_signals.items():
            if windows.ndim == 3:
                n_channels = windows.shape[2]
            else:
                n_channels = 1
                windows = windows[:, :, np.newaxis] if windows.ndim == 2 else windows[:, np.newaxis, :]

            for ch in range(n_channels):
                channel_windows = windows[:, :, ch] if windows.ndim == 3 else windows[:, :]
                prefix = f"{name}_ch{ch}"

                stat_names = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'rms', 'var', 'ptp']
                for stat_name in stat_names:
                    feature_names.append(f"{prefix}_{stat_name}")

                for band_name in self.spectral_bands.keys():
                    feature_names.append(f"{prefix}_{band_name}")
                feature_names.append(f"{prefix}_spectral_entropy")

        for window_idx in range(n_windows):
            window_features = []
            for name, windows in windowed_signals.items():
                if windows.ndim == 3:
                    n_channels = windows.shape[2]
                    window = windows[window_idx]
                else:
                    n_channels = 1
                    window = windows[window_idx][:, np.newaxis] if windows.ndim == 2 else windows[window_idx]

                for ch in range(n_channels):
                    channel = window[:, ch] if n_channels > 1 else window
                    stat_features = [
                        np.mean(channel),
                        np.std(channel),
                        np.min(channel),
                        np.max(channel),
                        np.median(channel),
                        np.percentile(channel, 25),
                        np.percentile(channel, 75),
                        np.sqrt(np.mean(channel**2)),
                        np.var(channel),
                        np.ptp(channel)
                    ]
                    window_features.extend(stat_features)

                    fs = self.target_sampling_rate
                    spec = self._extract_spectral_features(channel, fs)
                    for band_name in self.spectral_bands.keys():
                        window_features.append(spec[band_name])
                    window_features.append(spec['spectral_entropy'])

            all_features.append(window_features)

        self.feature_names = feature_names
        return np.array(all_features), feature_names

    def _get_labels_for_windows(self, labels: np.ndarray) -> np.ndarray:
        window_labels = []
        for start in range(0, len(labels) - self.window_size + 1, self.step_size):
            window_labels.append(labels[start])
        return np.array(window_labels)

    def load(self) -> EEGData:
        logger.info("Loading WESAD dataset...")
        subject_folders = self._get_subject_folders()
        logger.info(f"Found {len(subject_folders)} subject folders")
        all_features, all_labels, all_subject_ids = [], [], []
        for subject_folder in subject_folders:
            subject_id = subject_folder.name
            logger.info(f"Loading subject {subject_id}...")
            subject_data = self._load_subject_data(subject_folder)
            if subject_data is None:
                continue
            raw_signals = self._extract_signals(subject_data)
            if not raw_signals:
                logger.warning(f"No signals found for {subject_id}")
                continue
            resampled_signals = self._resample_signals(raw_signals)
            if not resampled_signals:
                logger.warning(f"Resampling failed for {subject_id}")
                continue
            labels = self._get_labels(subject_data)
            filtered_signals, filtered_labels = self._filter_by_labels(resampled_signals, labels)
            if len(filtered_labels) == 0:
                logger.warning(f"No valid labels for {subject_id}")
                continue
            windowed_signals = self._create_windows(filtered_signals)
            if not windowed_signals:
                logger.warning(f"No windows created for {subject_id}")
                continue
            features, _ = self._aggregate_windows(windowed_signals)
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
            feature_names = self.feature_names if self.feature_names is not None else [f"feature_{i}" for i in range(X.shape[1])]
            metadata = {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_subjects': len(np.unique(subject_ids)),
                'n_classes': len(np.unique(y)),
                'classes': np.unique(y).tolist(),
                'source': 'WESAD',
                'target_sampling_rate': self.target_sampling_rate,
                'modalities': self.modalities,
                'labels_used': self.labels,
                'window_size': self.window_size,
                'step_size': self.step_size,
                'feature_names': feature_names
            }
            logger.info(f"Loaded WESAD dataset: {X.shape[0]} samples, {X.shape[1]} features, "
                        f"{len(np.unique(subject_ids))} subjects, {len(np.unique(y))} classes")
            return EEGData(
                data=X,
                labels=y,
                subject_ids=subject_ids,
                feature_names=feature_names,
                sampling_rate=self.target_sampling_rate,
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
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'step_size': self.step_size,
            'signals': self.signals,
            'spectral_bands': self.spectral_bands
        }
