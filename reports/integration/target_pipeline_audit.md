# Target pipeline audit

## Purpose

This audit was triggered because the limited smoke run and the raw Parquet
inspection reported inconsistent class coverage.

## Raw dataset facts

| Property | Value |
|---|---:|
| Total rows | 51308 |
| Non-null `label_q5` | 45384 |
| Non-null `target_focus` | 45384 |
| All subjects | 55 |
| Subjects with non-null `label_q5` | 54 |
| Subjects with non-null `target_focus` | 54 |

### Full `label_q5` distribution

| Value | Count |
|---|---:|
| `0.0` | 9080 |
| `1.0` | 9075 |
| `2.0` | 9075 |
| `3.0` | 9078 |
| `4.0` | 9076 |
| `nan` | 5924 |


### First 1000 raw rows

| Value | Count |
|---|---:|
| `0.0` | 241 |
| `1.0` | 205 |
| `2.0` | 165 |
| `3.0` | 121 |
| `4.0` | 212 |
| `nan` | 56 |


## Observed benchmark behavior

The successful full-data smoke reported:

- 51,302 loaded samples;
- 55 subjects;
- five classes;
- 40,920 training rows and 10,382 test rows.

This does not directly match the 45,384 rows with non-null `label_q5`.

## Root configuration

```json
{
    "data_path":  "./data/processed/windowed_eeg_pm_dataset_w10.parquet",
    "feature_set":  "pow_plus_eeg",
    "n_classes":  5,
    "discretize":  true,
    "max_features":  500
}
```

Tasks:

```json
"cognitive_load_5class"
```

## Relevant source matches

```text
cli.py:46:                 'discretize': True,
cli.py:50:         'tasks': ['cognitive_load_5class'],
cli.py:289:     max_windows = getattr(args, 'max_windows', None)
cli.py:290:     if max_windows is not None:
cli.py:291:         max_windows = int(max_windows)
cli.py:292:         if max_windows <= 0:
cli.py:295:             dataset['max_windows'] = max_windows
cli.py:296:         logger.info(f"Datasets limited to {max_windows} windows")
cli.py:898:             max_windows=args.max_windows,
cli.py:964:             max_windows=args.max_windows,
configs.yaml:8:     discretize: true
configs.yaml:12:   - cognitive_load_5class
bench\tasks\cognitive_load.py:155:         return 'cognitive_load_5class'
bench\tasks\cognitive_load.py:159:     """Continuous ``target_focus`` task using the shared split machinery."""
bench\tasks\tasks_registry.py:13:     'cognitive_load_5class': CognitiveLoad5ClassTask,
bench\datasets\base_eeg_data_loader.py:54:         'target_main', 'label_q5',
bench\datasets\base_eeg_data_loader.py:102:     def _discretize_target(self, y: np.ndarray) -> np.ndarray:
bench\datasets\base_eeg_data_loader.py:103:         if not self.config.get('discretize', True):
bench\datasets\emotiv_loader.py:141:             target_candidates = ['target_main', 'label_q5'] + [c for c in df.columns if c.startswith('target_')]
bench\datasets\emotiv_loader.py:150:         y = self._discretize_target(y)
bench\datasets\emotiv_loader.py:175:         max_windows = self.config.get('max_windows')
bench\datasets\emotiv_loader.py:176:         if max_windows is not None:
bench\datasets\emotiv_loader.py:177:             max_windows = int(max_windows)
bench\datasets\emotiv_loader.py:178:             if max_windows <= 0:
bench\datasets\emotiv_loader.py:179:                 raise ValueError('max_windows must be positive')
bench\datasets\emotiv_loader.py:180:             if len(X) > max_windows:
bench\datasets\emotiv_loader.py:209:                 ).head(max_windows)['position'].to_numpy(
bench\datasets\emotiv_loader.py:239:                 'max_windows': max_windows,
bench\datasets\logical_recordings.py:87:             ).dropna()
bench\datasets\logical_recordings.py:188:         label_counts = group_rows["label_q5"].value_counts().sort_index()
bench\datasets\logical_recordings.py:193:             .dropna().astype(int).unique().tolist()
bench\datasets\raw_eeg_window_dataset.py:433:         splitter.split(np.zeros((len(frame), 1)), frame["label_q5"], groups), start=1
bench\datasets\raw_eeg_window_dataset.py:449:     """Join label_q5 windows to catalog records without reading raw signals."""
bench\datasets\raw_eeg_window_dataset.py:452:         "record_id", "source", "subject_id", "t_start", "t_end", "label_q5"
bench\datasets\raw_eeg_window_dataset.py:457:     supervised = processed.loc[processed["label_q5"].notna()].copy()
bench\datasets\raw_eeg_window_dataset.py:459:     supervised["label_q5"] = supervised["label_q5"].astype(np.int64)
bench\datasets\raw_eeg_window_dataset.py:535:         "t_start", "t_end", "absolute_t_start", "absolute_t_end", "label_q5",
bench\datasets\raw_eeg_window_dataset.py:916:         target_column = str(self.config.get("target_col", "label_q5"))
bench\datasets\raw_eeg_window_dataset.py:917:         if target_column != "label_q5":
bench\datasets\raw_eeg_window_dataset.py:920:                 f"'label_q5' only, got {target_column!r}"
bench\datasets\raw_eeg_window_dataset.py:992:         max_windows = self.config.get("max_windows")
bench\datasets\raw_eeg_window_dataset.py:993:         if max_windows is not None:
bench\datasets\raw_eeg_window_dataset.py:994:             limit = int(max_windows)
bench\datasets\raw_eeg_window_dataset.py:996:                 raise ValueError("max_windows must be positive")
bench\datasets\raw_eeg_window_dataset.py:1000:                     "max_windows must be large enough to retain every subject; "
bench\datasets\raw_eeg_window_dataset.py:1006:                 remaining.groupby(["outer_fold", "label_q5"], observed=True).ngroups,
bench\datasets\raw_eeg_window_dataset.py:1010:                 ["outer_fold", "label_q5"], sort=False, observed=True
bench\datasets\raw_eeg_window_dataset.py:1023:             sorted(accepted["preprocessing_hash"].dropna().astype(str).unique())
```

## Questions that must be resolved

1. Does the five-class task use the stored `label_q5` column or recompute
   classes from `target_focus`?
2. At what point is `discretize: true` applied?
3. At what point is `max_windows` applied?
4. Are rows with null `label_q5` excluded before splitting?
5. Why does the full loader report 51,302 samples instead of 51,308 raw rows
   or 45,384 supervised rows?
6. Why are 55 subjects loaded when the prior supervised benchmark used
   54 subjects?
7. Is the target-building behavior identical across the root config and the
   experiment configs?

## Integration decision

Do not use the current smoke metrics as scientific results.

Do not commit `canonical_smoke_validation.md` until the questions above are
resolved. The separate smoke configurations may remain untracked while the
target path is audited.
