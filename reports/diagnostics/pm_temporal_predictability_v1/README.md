# Predictive EEG history infrastructure

Status: **preflight complete; smoke only**. The primary full 175-fit experiment has NOT run. The CLI deliberately exposes only `--dry-run` and `--smoke`; full execution requires a separate authorized change. No new tuning, feature extraction, preprocessing or folds.

## Frozen comparison

Seven PM, five fixed subject folds, canonical 371 float32 features. H10/H20/H30/H60/H120 concatenate 1/2/3/6/12 exact previous ten-second feature blocks, oldest first and canonical -10 s last (371/742/1113/2226/4452 columns). Actual record-relative timestamps are checked; every intervening block must exist. No gap, record, subject or fold crossing and no substitution.

Primary `strict_common_support_h120`: one identical retained target universe per PM/fold for every horizon. Sample, ordered sample, label, participant, fold and threshold hashes are checked. Q33/Q67 are reused from full canonical outer-train continuous complete cases, before history filtering; LOW<=Q33, HIGH>=Q67. Missing middle targets consume time but never become binary training labels.

Secondary `horizon_specific_support` has a separate sensitivity run matrix section. The preflight describes 175 primary + 175 optional sensitivity cells; this is **not** 350 authorized fits. No sensitivity fit was executed. Future reporting must keep these supports separate; secondary deltas alone cannot isolate history effects.

Fixed LightGBM: n_estimators=200, n_jobs=4, random_state=42. Primary endpoint: participant-macro balanced accuracy; secondary Macro-F1, ROC-AUC, PR-AUC, recalls, precision and accuracy. Existing target fitting, model factory, binary fit/artifact path, participant metrics and subject bootstrap utilities are reused.

## Preflight

History support: H10 30,806; H20 30,663; H30 30,524; H60 30,140; H120 29,437 windows. Threshold invariance and primary identity hashes pass. Run matrix, input/config/source hashes, counts, alignment examples and feature names are generated. `protocol.json` records source cache identity; derived history views do not modify that cache.

## Bounded smoke

One run: Attention, fold 1, all five horizons, n_estimators=10. **5 executed, 0 restored, 5 complete**. Every cell uses 14,756 training and 4,098 test examples.

| Horizon | Participant-macro BA | Macro-F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| H10 | 0.6557 | 0.5998 | 0.7963 | 0.8207 |
| H20 | 0.6680 | 0.6124 | 0.8020 | 0.8260 |
| H30 | 0.6762 | 0.6239 | 0.7996 | 0.8177 |
| H60 | 0.6728 | 0.6178 | 0.7970 | 0.8176 |
| H120 | 0.6598 | 0.6025 | 0.7921 | 0.8165 |

These are smoke metrics, **not evidence for a preferred horizon or model**. No bootstrap inference is made from this smoke. Participant metrics and paired H10 deltas are saved. Runtime prediction/history-provenance artifacts have SHA256 checksums; resume rejects incomplete or corrupt results instead of silently retraining.

## Order control and future reporting

`history_matrix(..., shuffled=True, sample_ids=...)` independently permutes older blocks per target example using a stable sample-ID seed. It keeps all 371 features of each block together and fixes the -10 block. A single global column permutation would not destroy order information for a tree model, so it is not used. This control has synthetic tests but no real-data fits. H10/H20 cannot have a nontrivial older-block order permutation.

`paired_history_results` pairs each participant/PM/horizon with H10 within its support mode and can use the existing subject bootstrap (10,000 resamples, seed 42). Final comparisons require metric-specific valid participant counts, descriptive intervals and multiplicity caution. No outcome-driven architecture or horizon choice is made here.

## Reproduce

```powershell
python scripts/analysis/run_pm_temporal_predictability.py --feature-cache-dir <canonical-cache> --dry-run
python scripts/analysis/run_pm_temporal_predictability.py --feature-cache-dir <canonical-cache> --smoke
```

An existing smoke with matching protocol/checksums is restored without new fits. Incompatible output is rejected. Do not invoke another training smoke to select settings.

Validation: 40 targeted tests passed across the three new test modules and the existing absolute-lag regression module. Tests cover entropy and SVD/FNN synthetic behavior, length gates, exact history/gaps/boundaries, dimensions/order, train-only threshold invariance, matched hashes, deterministic shuffle/smoke manifests, paired participant bootstrap and rejection of full execution. A first test attempt encountered permissions on the pre-existing pytest runtime folder; the passing run used a fresh task-specific `--basetemp` without deleting user files.

Next step: review the [diagnostic report](../pm_temporal_predictability_audit_v1/README.md), then separately authorize and enable the full matched predictive experiment. Diagnostic temporal structure motivates that comparison but does not yet justify a new sequence architecture.
