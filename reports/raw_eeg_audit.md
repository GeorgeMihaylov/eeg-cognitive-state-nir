# Raw EEG audit

Audited records: **120**.
Canonical signal channels: **14**.

## Canonical schema

Channel order: `EEG.AF3, EEG.F7, EEG.F3, EEG.FC5, EEG.T7, EEG.P7, EEG.O1, EEG.O2, EEG.P8, EEG.T8, EEG.FC6, EEG.F4, EEG.F8, EEG.AF4`.

`Timestamp` is Unix time in seconds. Windows are aligned to the same absolute 10-second bins used by the processed dataset. EEG service columns such as `EEG.Counter`, `EEG.Interpolated`, battery and marker fields are not model inputs.

## Sources

| source | records | rows | size GiB | compression | nominal rates | measured sfreq min/median/max | duplicate-ts records | channel-NaN records | gaps >1.5 samples | sampled amplitude p01/p50/p99 |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---|
| Old_EEG | 49 | 61263878 | 34.11 | none:49 | 128Hz:1, 256Hz:48 | 128.074/256.141/256.156 | 0 | 0 | 14 | 4012.122/4293.333/4632.163 |
| gpn_data | 71 | 67823579 | 3.15 | bz2:71 | 128Hz:3, 256Hz:68 | 128.074/256.141/256.156 | 0 | 1 | 9 | 4061.592/4291.539/4609.985 |

## Decisions

- Use the intersection of the 14 named Emotiv signal channels in the fixed order above.
- Preserve 256 Hz when the measured source rate is within 0.5%; the four measured 128 Hz exports are upsampled to the common 256 Hz target with polyphase resampling.
- Select data by timestamps, collapse duplicate timestamps, regularize small jitter, and reject a window when more than 2% of an expected channel grid is absent.
- Keep raw amplitudes in exported numeric units; physical units are not asserted because the exports do not provide a verified calibration field.
- `Old_EEG` is uncompressed while `gpn_data` is mostly bzip2-compressed; this is an I/O difference, not a channel-schema difference.

The machine-readable per-record measurements are in `data/interim/raw_eeg_schema.json`.
