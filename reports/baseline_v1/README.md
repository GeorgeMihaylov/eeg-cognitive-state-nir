# Baseline v1

This directory contains the integrated baseline summary for the EEG latent proxy-state project.

## Baseline definition

- Target space: `slow_pca_1`, `slow_pca_2`, `slow_pca_3`
- Feature set: `pow_plus_eeg`
- Sequence length: `8`
- Main model: Transformer sequence regressor
- Personal calibration: head-only calibration
- Calibration fraction: `0.2`
- Split protocol: subject-wise train/validation/test

## Main command

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\52_run_integrated_baseline_v1.py `
  --root . `
  --mode summarize
```

## Outputs

- `baseline_v1_report.md`
- `baseline_v1_summary.json`
- `hypothesis_baseline_matrix.csv`
- `artifact_index.csv`
- `commands_used.md`

## Interpretation

This baseline is a project-level reproducibility and reporting baseline. It integrates feature ablation, temporal model comparison, personal calibration diagnostics, split-seed robustness, and optional naive baselines into one hypothesis-driven summary.
