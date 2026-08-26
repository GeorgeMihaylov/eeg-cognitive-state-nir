"""MNE-FASTER calibration bundles and causal streaming application."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MANIFEST_NAME = "mne-faster-manifest.json"
ICA_NAME = "calibration-ica.fif"
BUNDLE_SCHEMA_VERSION = 1


def _dependencies() -> tuple[Any, Any]:
    try:
        import mne
        import mne_faster
    except ImportError as exc:  # pragma: no cover - exercised without optional deps
        raise RuntimeError(
            "MNE-FASTER support requires the preprocessing dependencies: "
            "install requirements-preprocessing.txt"
        ) from exc
    return mne, mne_faster


def _epochs_array(
    epochs: object,
    *,
    sample_rate: float,
    channel_names: Sequence[str],
    montage_name: str,
    input_scale_to_volts: float,
    preprocessing_contract: Mapping[str, Any] | None = None,
) -> Any:
    mne, _ = _dependencies()
    values = np.asarray(epochs, dtype=float)
    if values.ndim != 3 or not all(values.shape):
        raise ValueError("epochs must be [epochs, samples, channels]")
    if values.shape[2] != len(channel_names):
        raise ValueError("epoch channel count does not match channel_names")
    if not np.isfinite(values).all():
        raise ValueError("epochs contain non-finite values")
    if sample_rate <= 0 or input_scale_to_volts <= 0:
        raise ValueError("sample_rate and input_scale_to_volts must be positive")

    info = mne.create_info(
        ch_names=list(channel_names), sfreq=sample_rate, ch_types="eeg"
    )
    contract = preprocessing_contract or {}
    with info._unlock():
        if "bandpass_low_hz" in contract:
            info["highpass"] = float(contract["bandpass_low_hz"])
        if "bandpass_high_hz" in contract:
            info["lowpass"] = float(contract["bandpass_high_hz"])
    montage = mne.channels.make_standard_montage(montage_name)
    info.set_montage(montage, on_missing="ignore")
    missing_positions = [
        channel["ch_name"]
        for channel in info["chs"]
        if not np.isfinite(channel["loc"][:3]).all()
        or np.linalg.norm(channel["loc"][:3]) <= np.finfo(float).eps
    ]
    if missing_positions:
        raise ValueError(
            f"Montage {montage_name!r} has no positions for: "
            + ", ".join(missing_positions)
        )
    data = np.transpose(values, (0, 2, 1)) * input_scale_to_volts
    return mne.EpochsArray(data, info, tmin=0.0, baseline=None, verbose=False)


def _supports_line_noise(sample_rate: float) -> bool:
    return sample_rate / 2.0 > 60.0


def _channel_metrics(sample_rate: float) -> tuple[str, ...]:
    metrics = ["variance", "correlation", "hurst", "kurtosis"]
    if _supports_line_noise(sample_rate):
        metrics.append("line_noise")
    return tuple(metrics)


def _component_metrics(sample_rate: float, *, has_eog: bool = False) -> tuple[str, ...]:
    metrics = ["kurtosis", "power_gradient", "hurst", "median_gradient"]
    if has_eog:
        metrics.insert(0, "eog_correlation")
    if _supports_line_noise(sample_rate):
        metrics.append("line_noise")
    return tuple(metrics)


def _local_channel_metrics(sample_rate: float) -> tuple[str, ...]:
    metrics = ["amplitude", "variance", "deviation", "median_gradient"]
    if _supports_line_noise(sample_rate):
        metrics.append("line_noise")
    return tuple(metrics)


def _clean_local_channels(epochs: Any, local_bads: Sequence[Sequence[str]]) -> None:
    """Interpolate global and epoch-local bads in place, as MNE-FASTER does."""
    global_bads = tuple(epochs.info.get("bads", ()))
    for index, bads in enumerate(local_bads):
        combined = list(dict.fromkeys((*global_bads, *bads)))
        if not combined:
            continue
        epoch = epochs[index]
        epoch.info["bads"] = combined
        epoch.interpolate_bads(reset_bads=True, verbose=False)
        epochs._data[index] = epoch.get_data(copy=False)[0]
    epochs.info["bads"] = []


def _find_local_bads(mne_faster: Any, epochs: Any, *, threshold: float, max_iter: int) -> Any:
    # scipy.stats warns when a metric is identical across all channels. In that
    # case MNE returns no z-score outlier, which is the intended outcome.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Precision loss occurred.*", category=RuntimeWarning
        )
        return mne_faster.find_bad_channels_in_epochs(
            epochs,
            thres=threshold,
            max_iter=max_iter,
            use_metrics=_local_channel_metrics(float(epochs.info["sfreq"])),
        )


@dataclass(frozen=True)
class MNEFasterConfig:
    """Configuration for subject/record-level MNE-FASTER calibration."""

    sample_rate: float
    channel_names: tuple[str, ...]
    montage_name: str = "standard_1020"
    input_scale_to_volts: float = 1e-6
    global_channel_threshold: float = 3.0
    threshold: float = 3.0
    max_iter: int = 1
    ica_n_components: int | float | None = None
    ica_method: str = "fastica"
    ica_random_state: int = 42
    ica_max_iter: int = 500
    power_gradient_range_hz: tuple[float, float] = (1.0, 45.0)
    preprocessing_contract: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or not self.channel_names:
            raise ValueError("A positive sample_rate and channel_names are required")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel_names must be unique")
        if self.global_channel_threshold <= 0 or self.threshold <= 0:
            raise ValueError("FASTER thresholds must be positive")
        if self.max_iter < 1 or self.ica_max_iter < 1:
            raise ValueError("iteration limits must be positive")
        low, high = self.power_gradient_range_hz
        if not 0 < low < high < self.sample_rate / 2.0:
            raise ValueError("power_gradient_range_hz must be inside Nyquist")


@dataclass(frozen=True)
class MNEFasterReport:
    global_bad_channels: tuple[str, ...]
    bad_epochs: tuple[int, ...]
    bad_components: tuple[int, ...]
    local_bad_channels: tuple[tuple[str, ...], ...]
    kept_epoch_mask: np.ndarray
    channel_bads_by_metric: Mapping[str, tuple[str, ...]]
    epoch_bads_by_metric: Mapping[str, tuple[int, ...]]
    component_bads_by_metric: Mapping[str, tuple[int, ...]]


@dataclass
class MNEFasterBundle:
    """Serializable calibration state applied unchanged to streaming windows."""

    version: str
    sample_rate: float
    channel_names: tuple[str, ...]
    montage_name: str
    input_scale_to_volts: float
    global_bad_channels: tuple[str, ...]
    preprocessing_contract: dict[str, Any]
    threshold: float
    max_iter: int
    power_gradient_range_hz: tuple[float, float]
    ica: Any

    def validate(
        self,
        *,
        sample_rate: float,
        channel_names: Sequence[str],
        preprocessing_contract: Mapping[str, Any],
    ) -> None:
        errors: list[str] = []
        if not np.isclose(self.sample_rate, sample_rate):
            errors.append(f"sample_rate {self.sample_rate} != {sample_rate}")
        if self.channel_names != tuple(channel_names):
            errors.append("channel order differs")
        for name, expected in preprocessing_contract.items():
            actual = self.preprocessing_contract.get(name)
            if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                if actual is None or not np.isclose(float(actual), float(expected)):
                    errors.append(f"preprocessing.{name} {actual!r} != {expected!r}")
            elif actual != expected:
                errors.append(f"preprocessing.{name} {actual!r} != {expected!r}")
        if errors:
            raise ValueError("Incompatible MNE-FASTER bundle: " + "; ".join(errors))

    def transform(self, signal: object) -> np.ndarray:
        _, mne_faster = _dependencies()
        values = np.asarray(signal, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.channel_names):
            raise ValueError("signal must be [samples, channels] in bundle channel order")
        epochs = _epochs_array(
            values[None, ...],
            sample_rate=self.sample_rate,
            channel_names=self.channel_names,
            montage_name=self.montage_name,
            input_scale_to_volts=self.input_scale_to_volts,
            preprocessing_contract=self.preprocessing_contract,
        )
        epochs.info["bads"] = list(self.global_bad_channels)
        self.ica.apply(epochs, verbose=False)
        local_bads = _find_local_bads(
            mne_faster,
            epochs,
            threshold=self.threshold,
            max_iter=self.max_iter,
        )
        _clean_local_channels(epochs, local_bads)
        epochs.set_eeg_reference("average", verbose=False)
        cleaned = epochs.get_data(copy=False)[0].T / self.input_scale_to_volts
        return np.asarray(cleaned, dtype=values.dtype)

    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.ica.save(target / ICA_NAME, overwrite=True, verbose=False)
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "version": self.version,
            "sample_rate": self.sample_rate,
            "channel_names": list(self.channel_names),
            "montage_name": self.montage_name,
            "input_scale_to_volts": self.input_scale_to_volts,
            "global_bad_channels": list(self.global_bad_channels),
            "bad_components": list(self.ica.exclude),
            "preprocessing_contract": self.preprocessing_contract,
            "threshold": self.threshold,
            "max_iter": self.max_iter,
            "power_gradient_range_hz": list(self.power_gradient_range_hz),
            "ica_file": ICA_NAME,
        }
        (target / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, directory: str | Path) -> "MNEFasterBundle":
        mne, _ = _dependencies()
        target = Path(directory)
        manifest = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ValueError("Unsupported MNE-FASTER bundle schema")
        ica = mne.preprocessing.read_ica(target / manifest["ica_file"], verbose=False)
        declared = tuple(int(index) for index in manifest.get("bad_components", ()))
        if tuple(ica.exclude) != declared:
            raise ValueError("ICA exclusions differ from the MNE-FASTER manifest")
        return cls(
            version=str(manifest["version"]),
            sample_rate=float(manifest["sample_rate"]),
            channel_names=tuple(manifest["channel_names"]),
            montage_name=str(manifest["montage_name"]),
            input_scale_to_volts=float(manifest["input_scale_to_volts"]),
            global_bad_channels=tuple(manifest.get("global_bad_channels", ())),
            preprocessing_contract=dict(manifest.get("preprocessing_contract", {})),
            threshold=float(manifest["threshold"]),
            max_iter=int(manifest["max_iter"]),
            power_gradient_range_hz=tuple(manifest["power_gradient_range_hz"]),
            ica=ica,
        )


class MNEFasterCalibrator:
    """Fit the reference MNE-FASTER stages and retain deployable state."""

    def __init__(self, config: MNEFasterConfig) -> None:
        self.config = config

    def fit_transform(
        self, epochs: object, *, version: str = "mne-faster-calibration-v1"
    ) -> tuple[np.ndarray, MNEFasterBundle, MNEFasterReport]:
        mne, mne_faster = _dependencies()
        cfg = self.config
        prepared = _epochs_array(
            epochs,
            sample_rate=cfg.sample_rate,
            channel_names=cfg.channel_names,
            montage_name=cfg.montage_name,
            input_scale_to_volts=cfg.input_scale_to_volts,
            preprocessing_contract=cfg.preprocessing_contract,
        )

        channel_by_metric = mne_faster.find_bad_channels(
            prepared,
            thres=cfg.global_channel_threshold,
            max_iter=cfg.max_iter,
            use_metrics=_channel_metrics(cfg.sample_rate),
            return_by_metric=True,
        )
        global_bads = tuple(
            sorted({name for names in channel_by_metric.values() for name in names})
        )
        prepared.info["bads"] = list(global_bads)

        epoch_by_metric = mne_faster.find_bad_epochs(
            prepared,
            thres=cfg.threshold,
            max_iter=cfg.max_iter,
            return_by_metric=True,
        )
        bad_epochs = tuple(
            sorted({int(index) for indices in epoch_by_metric.values() for index in indices})
        )
        kept_mask = np.ones(len(prepared), dtype=bool)
        kept_mask[list(bad_epochs)] = False
        if not kept_mask.any():
            raise ValueError("MNE-FASTER rejected every calibration epoch")
        calibration = prepared[kept_mask]

        picks = mne.pick_types(
            calibration.info, meg=False, eeg=True, eog=False, exclude="bads"
        )
        if len(picks) < 2:
            raise ValueError("At least two good EEG channels are required for ICA")
        ica = mne.preprocessing.ICA(
            n_components=cfg.ica_n_components,
            method=cfg.ica_method,
            random_state=cfg.ica_random_state,
            max_iter=cfg.ica_max_iter,
        )
        ica.fit(calibration, picks=picks, verbose=False)
        component_by_metric = mne_faster.find_bad_components(
            ica,
            calibration,
            thres=cfg.threshold,
            max_iter=cfg.max_iter,
            use_metrics=_component_metrics(cfg.sample_rate),
            prange=cfg.power_gradient_range_hz,
            return_by_metric=True,
        )
        bad_components = tuple(
            sorted(
                {int(index) for indices in component_by_metric.values() for index in indices}
            )
        )
        ica.exclude = list(bad_components)

        cleaned = calibration.copy()
        ica.apply(cleaned, verbose=False)
        local_bads = _find_local_bads(
            mne_faster,
            cleaned,
            threshold=cfg.threshold,
            max_iter=cfg.max_iter,
        )
        _clean_local_channels(cleaned, local_bads)
        cleaned.set_eeg_reference("average", verbose=False)

        bundle = MNEFasterBundle(
            version=version,
            sample_rate=cfg.sample_rate,
            channel_names=cfg.channel_names,
            montage_name=cfg.montage_name,
            input_scale_to_volts=cfg.input_scale_to_volts,
            global_bad_channels=global_bads,
            preprocessing_contract=dict(cfg.preprocessing_contract),
            threshold=cfg.threshold,
            max_iter=cfg.max_iter,
            power_gradient_range_hz=cfg.power_gradient_range_hz,
            ica=ica,
        )
        report = MNEFasterReport(
            global_bad_channels=global_bads,
            bad_epochs=bad_epochs,
            bad_components=bad_components,
            local_bad_channels=tuple(tuple(names) for names in local_bads),
            kept_epoch_mask=kept_mask,
            channel_bads_by_metric={
                key: tuple(value) for key, value in channel_by_metric.items()
            },
            epoch_bads_by_metric={
                key: tuple(int(index) for index in value)
                for key, value in epoch_by_metric.items()
            },
            component_bads_by_metric={
                key: tuple(int(index) for index in value)
                for key, value in component_by_metric.items()
            },
        )
        values = cleaned.get_data(copy=False).transpose(0, 2, 1)
        values = values / cfg.input_scale_to_volts
        return values, bundle, report
