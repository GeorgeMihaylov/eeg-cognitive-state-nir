# Label target audit

Примечание об архитектуре: упоминания `src/02`, `src/04` и `src/08`
ниже фиксируют исторические пути на момент исходного аудита. Текущие
реализации находятся соответственно в
`bench/datasets/emotiv_catalog_builder.py`,
`bench/datasets/emotiv_pm_window_builder.py` и
`bench/features/legacy_emotiv_eeg_features.py`; актуальные CLI находятся в
`scripts/data/`.

## Provenance

The verified construction path is:

1. `src/02_build_emotiv_catalog.py` inventories exported Emotiv CSV/BZ2 records under both `data/raw/gpn_data` and `data/raw/Old_EEG`, preserving the acquisition source, subject, file layout, separator, and header metadata. Both source inventories expose the vendor-provided `PM.Focus.Scaled` field; this repository does not derive that upstream scale.
2. `src/04_build_windowed_pm_dataset.py` reads the catalog and validated common columns. `pd.to_numeric(errors='coerce')` converts invalid PM values to missing; only rows with a missing `Timestamp` are explicitly dropped. Records are divided into absolute 10-second timestamp bins. `PM.Focus.Scaled` is aggregated as mean/std/min/max/last, and `target_focus = PM.Focus.Scaled__mean`; pandas' mean ignores individual missing focus samples, leaving a missing window target only when its focus mean is unavailable. If one logical time bin crosses CSV chunk boundaries, the current implementation takes an unweighted mean of its chunk-level aggregates during secondary aggregation.
3. Within each record, `target_main = target_focus`. No additional target normalization and no `PM.Focus.IsActive` mask are applied.
4. All record tables from both sources are concatenated. Only then is `label_q5` calculated from `target_main` by a single global `pd.qcut` call.
5. `src/08_build_eeg_features.py` left-merges the EEG features into that PM/POW table. Direct comparison of the two processed Parquets showed the same row count and exact equality of `target_focus` and `label_q5`.
6. Benchmark configs select the stored `label_q5` with `discretize: false`; the dataset loader and five-class task validate/use it but do not recreate the quantile labels.

`label_q5` is produced after all records from both sources have been concatenated by `pd.qcut(target_main, q=5, labels=False, duplicates='drop')`; `target_main` and `target_focus` are exactly equal. It is not computed per source, subject, or record, and no train/test split exists at that point.

## Dataset structure

- Rows: 51,308
- Subjects (all / supervised): 55 / 54
- Records (all / supervised): 120 / 119
- Sources: ['Old_EEG', 'gpn_data']
- Non-null `target_focus`: 45,384
- Non-null `label_q5`: 45,384
- Subjects in both sources: 31
- Subjects in one source only: 23

## Reconstructed quintile boundaries

- Class 0: `[0.004077, 0.330177]`
- Class 1: `(0.330177, 0.387786]`
- Class 2: `(0.387786, 0.444458]`
- Class 3: `(0.444458, 0.526585]`
- Class 4: `(0.526585, 0.991193]`

These one set of boundaries is applied to `Old_EEG` and `gpn_data`. For diagnosis only, independently refitting `qcut` inside each source would have produced the following different edges (these were not used):

- `Old_EEG`: 0.004077, 0.3310062, 0.3902908, 0.4491266, 0.533429, 0.991193
- `gpn_data`: 0.033242, 0.329431, 0.385454, 0.440328, 0.521398, 0.991193

The stored label agrees exactly with a fresh global `qcut` on the current processed target. The numerical boundaries were not persisted by the original builder; these values are reconstructed from the current Parquet. Therefore the current labels are exactly reproducible, while a raw rebuild also depends on retaining the same raw exports, builder code, chunking, and record inventory.

## Per-class statistics

| class_id | windows | subjects | records | sources | target_focus_min | target_focus_max | target_focus_mean | target_focus_median | target_focus_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 9080 | 53 | 112 | 2 | 0.004077 | 0.330177 | 0.272814 | 0.286450 | 0.049831 |
| 1 | 9075 | 54 | 116 | 2 | 0.330181 | 0.387786 | 0.360452 | 0.361106 | 0.016447 |
| 2 | 9075 | 53 | 115 | 2 | 0.387791 | 0.444454 | 0.415188 | 0.414493 | 0.016371 |
| 3 | 9078 | 53 | 114 | 2 | 0.444459 | 0.526585 | 0.482278 | 0.480582 | 0.023428 |
| 4 | 9076 | 53 | 107 | 2 | 0.526592 | 0.991193 | 0.619558 | 0.598831 | 0.078203 |

## Source comparison

| source | windows | subjects | records | mean | std | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Old_EEG | 21558 | 43 | 48 | 0.433588 | 0.129357 | 0.004077 | 0.991193 |
| gpn_data | 23826 | 42 | 71 | 0.426845 | 0.120829 | 0.033242 | 0.991193 |

The unadjusted source mean contrast is 0.006743 (`Old_EEG - gpn_data`), with descriptive Cohen's d 0.053961.

## Variance decomposition

- Total population variance: 0.015623800
- Between subjects: 16.0948%
- Within subjects: 83.9052%
- Between records within subject: 6.9945%
- Within records: 76.9106%
- Unadjusted between sources: 0.0726%
- One-way ICC(1): 0.163054

These are descriptive, window-weighted components. Temporally adjacent windows are not independent, and the source component is not adjusted for subjects observed in both sources.

## Leakage assessment and scientific interpretation

The EEG or feature values are not used to define the classes. However, the global class boundaries use target values from every subject and both sources before GroupKFold, LOSO, or cross-source evaluation. This is a methodological target-definition leakage (a transductive use of outer-test target distribution), even though it does not directly expose test EEG features to the estimator. It can make class balance and thresholds depend on the test cohort. Future confirmatory evaluations should freeze clinically or scientifically justified thresholds, or estimate thresholds on each outer training partition and apply them unchanged to its test partition.

Scientifically, `label_q5` is an ordinal discretization of a proprietary device-derived focus metric averaged over a window. It is a weak proxy target, not a direct expert annotation, diagnosis, or independently validated cognitive-state ground truth. The five IDs encode ordered global quantile bands; treating them as nominal classes is an engineering benchmark choice.

## Reproducibility and safety

- Input SHA-256 before/after: `26b7d71f7c71cc575098888150f12dcf24132075c7e692c9059cf675200954f8` / `26b7d71f7c71cc575098888150f12dcf24132075c7e692c9059cf675200954f8`
- Input Parquet modified: no
- Models trained: 0
- Generated tables are written outside tracked source files.
