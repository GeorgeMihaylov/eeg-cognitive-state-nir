# Absolute EEG lag: implementation and launch readiness

Status: implementation validated; confirmatory training **not started**.
Branch: `analysis/pm-eeg-absolute-lag-20260928`.
Inspected HEAD: `807ca0e8526d27ba2facf94c3c31ab1e74ce5e24`.

## Fixed scientific contract

- Five start-to-start absolute lags: -30, -20, **-10 reference**, 0, +10 seconds.
- Seven explicit continuous PM targets converted to LOW/HIGH using the existing
  outer-train Q(1/3)/Q(2/3) contract; strict middle excluded.
- Thresholds fit once per PM/fold on **all canonical outer-train continuous complete
  cases**, preserving the established LOW/HIGH threshold universe. They are not
  refitted separately by lag or on outer-test/common-support test values.
- Common support: target-valid samples with an exact EEG pair for every lag,
  separately retained within the existing five subject folds. Train and test
  support are both matched; -10 is newly trained on this support.
- Canonical float32 371 features and unchanged cached preprocessing. LightGBM:
  200 trees, four threads, seed 42; no tuning or auxiliary fits. 175 model fits.
- Primary endpoint: equal-weight participant balanced accuracy. Secondary macro-F1,
  ROC-AUC, average precision (PR-AUC), recalls, precision and accuracy. Undefined
  participant metrics stay undefined, with valid-participant counts per metric.
- All paired differences use -10. Fold mean/SD and participant paired bootstrap
  (10,000 draws, seed 42) are saved. No window bootstrap or automatic lag selection.

## Timestamp semantics and real audit

`absolute_lag_s = EEG.t_start - PM.t_start`. The builder
`bench/datasets/emotiv_pm_window_builder.py` aggregates PM within 10-second bins
and records relative window start/end. These are **not PM emission timestamps**.
The old lag lookup's negative sign already means earlier EEG; no sign inversion
is needed. The experiment forces `t_start` arithmetic, ignoring nominal window IDs.
All pairings verify actual timestamp differences and record/subject/fold identity.
Missing exact pairs are excluded; the previous available row is never substituted.

For lags <=0 the available-information cutoff is **PM-window end**. Thus 0 is
contemporaneous, not prediction at PM-window start. +10 is explicitly non-causal.

Real example: subject `0012905a`, recording
`gpn_data__0012905a__day1__2part__2023.11.28T12.51.55p03.00`, PM window [35,45] s:

| Absolute lag | EEG start | EEG end | Actual start difference |
|---:|---:|---:|---:|
| -30 | 5 | 15 | -30 |
| -20 | 15 | 25 | -20 |
| -10 | 25 | 35 | -10 |
| 0 | 35 | 45 | 0 |
| +10 | 45 | 55 | +10 |

`alignment_audit.csv` saves three real common-support examples per fold and lag.
Every available pairing, not only these examples, is checked in memory.

## Reuse and changed files

- `bench/experiments/pm_low_high_q3_extremes_confirmatory.py`: extracted the existing
  binary fit/artifact body into `execute_binary_run`; historical `execute_run`
  retains its fixed -10/XGBoost restrictions. No parallel model-training path.
- `bench/experiments/pm_eeg_absolute_lag.py` (new): experiment preparation,
  matching/audits, manifest, resume, paired aggregation and figures.
- `scripts/analysis/run_pm_eeg_absolute_lag.py` (new): thin CLI.
- `experiments/pm_diagnostics/pm_eeg_absolute_lag_v1.json` (new): frozen specification.
- `tests/test_pm_eeg_absolute_lag.py` (new): targeted contract and execution tests.
- This diagnostics directory contains deterministic preflight summaries and a
  separate `smoke/` result. Prediction-level `smoke/runs/` files are runtime only.

Also reused: canonical feature cache validation/loading; `build_threshold_audit`,
binary labels, participant metrics and resumable-summary handling; exact lag lookup
from `bench/analysis/pm_eeg_lag_sweep.py`; `subject_bootstrap_interval` from
`bench/analysis/paired_statistics.py`; canonical `cogstate.model_zoo.build_model`.

## Validation on 2026-09-28

- `py_compile`: four touched/new Python files passed.
- Targeted and related tests: **56 passed** across absolute lag, LOW/HIGH extremes,
  lag confirmation, model robustness, temporal context and lag regression modules.
- CLI `--help`, full `--dry-run`, real `--smoke` and real smoke resume passed.
- Dry-run: 30,958 canonical rows; common complete-case support 29,104 Attention
  and 30,392 for each other PM. Zero cross-record, cross-subject or cross-fold pairs.
- Smoke: Attention/fold 1, ten trees, all five lags; 15,088 training examples from
  42 participants, 4,202 test examples from 11 participants, identical across lags.
- Initial smoke: 5 executed. Resume verification: **0 executed, 5 resumed**.
- Figures visually inspected; reference heatmap column equals exactly zero.
- `git diff --check` passed. Full unrelated test suite was not run.

| Lag | Smoke participant BA | Delta vs -10 |
|---:|---:|---:|
| -30 | 0.608403 | -0.052909 |
| -20 | 0.631869 | -0.029443 |
| -10 | 0.661313 | 0.000000 |
| 0 | 0.640070 | -0.021243 |
| +10 | 0.604316 | -0.056997 |

These are execution diagnostics, **not confirmation of the best lag**.

Confirmatory protocol hash:
`38a65e564a8103de5f3a34f5216af0aff1dddcbea2ee8484af4e3acd7adb1fdf`.
Smoke protocol hash:
`0c9b2b184a516a8851e79d5aae74a33e0576ec34f48455225aca075407a4c2a9`.
Per-fit specification hashes are in `run_matrix.csv`. The manifest also contains
feature-value SHA256, target/timestamp table hash, cache/feature/preprocessing/fold
identity, thresholds/support/pairing hashes, effective model defaults, package
versions, source hashes and Git provenance. No model weights are saved.

## Windows / PowerShell launch

The following uses the verified local interpreter and canonical cache. LightGBM
4.6.0 was installed only into the local ignored runtime directory; no shared
environment was modified. Runtime paths are provided here solely for reproducibility.

```powershell
Set-Location F:\EEG
$env:PYTHONPATH = 'F:\EEG\benchmark_results\pm_absolute_lag_runtime;F:\EEG'
& 'F:\venvs\ml312\Scripts\python.exe' `
  'scripts\analysis\run_pm_eeg_absolute_lag.py' `
  --feature-cache-dir 'F:\eeg-preproc-integration\benchmark_results\cogstate_features_emotiv14_canonical_v1' `
  --run
```

Use `--dry-run` instead of `--run` for no-training preflight; `--smoke` writes to
the separate `smoke/` directory by default. Resume is automatic and verifies
protocol/specification/threshold hashes plus prediction and participant-file
checksums. Completed cells are not retrained. A different protocol cannot reuse
an output directory containing runs; interrupted/incompatible artifacts fail closed.

Expected full outputs: `protocol.json`, `dry_run_summary.json`, `run_matrix.csv`,
`threshold_audit.csv`, `common_support_audit.csv`, `alignment_audit.csv`,
`overall.csv`, `fold_metrics.csv` (including paired deltas),
`participant_metrics.csv`, `participant_paired_deltas.csv`,
`participant_bootstrap.csv`, `balanced_accuracy_vs_lag.png`, `delta_heatmap.png`,
`REPORT.md`, `execution_status.json`, and 175 runtime `runs/*/` bundles containing
predictions, participant metrics and run summaries.

## Limits and repository hygiene

No full scientific outcome is available yet. Smoke uses one target/fold and fewer
trees. Participant CIs are descriptive, unadjusted for multiplicity, and conditional
on overlapping training folds. PM autocorrelation remains relevant. A high +10
result would motivate a separate smoothing/overlap/timestamp audit, not a new
causal production protocol or a claim about physiological delay.

LightGBM/sklearn emits a benign feature-name warning when predicting from the
NumPy matrix. No feature ordering, preprocessing or model parameters were changed
to suppress it. Actual feature values and all 371 columns are preserved.

The pre-existing modified nested-AutoML protocol and unrelated untracked files
were untouched. No staging, commit, push, merge, rebase, reset or cleanup occurred.
