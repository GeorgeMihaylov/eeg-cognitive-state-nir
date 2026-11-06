# Current scientific context — 2026-11-06

This is the compact current scientific context. Code/tests and frozen experiment artifacts remain the source of truth. [README](README.md) is the project entry point; [AGENTS](AGENTS.md) defines operational invariants. Historical snapshots are retained with explicit banners.

## Question and current decision

What temporal information in EEG supports cross-subject prediction of PM-defined LOW/HIGH states, and how much can limited participant calibration improve it?

**Temporal Predictability Audit is complete**: support, ACF, MI/surrogates, PE, SSA/Hankel, restricted FNN; full matched H10–H120 **175/175**; conditional H120 order control **25/25**. History value is target-dependent. H120 improves 5/7 PM after BH across 28 contrasts; correct older-block order improves 4/5 gated PM after BH across five contrasts. Stress order evidence is inconclusive.

Next: **one bounded, prespecified H120 sequence-model comparison against ordered-H120 LightGBM**, planned, not executed. No broad neural search or automatic AutoML/Mamba/PatchTST/TimesNet sweep. Wavelet and synchronization ablations remain separate subsequent hypotheses; representation and architecture should not change simultaneously. Later: one confirmatory personalization comparison and independent external validation.

## Data and representation

Internal sources: gpn_data + Old_EEG; these are provenance sources, not independent domains because logical recordings can overlap. Canonical deduplicated cohort: 30,958 ten-second windows, 54 participants, 86 logical recordings. Raw EEG per example [1,14,2560], 256 Hz. Attention has 29,569 continuous complete cases before temporal/label filtering; other PM have 30,958. Each experiment's support is authoritative.

Current canonical feature cache: **371 = 182 spectral + 140 statistical + 28 entropy + 21 connectivity**. Historical EEG 168 + POW 280 = 448 features is a separate pipeline, not the origin or current definition of the 371 features.

## Frozen current target and validation contract

Seven targets: target_attention, target_engagement, target_excitement, target_stress, target_relaxation, target_interest, target_focus.

For each PM and fixed outer fold, fit Q33/Q67 on **canonical outer-train continuous complete cases**, once before history filtering. **LOW <= Q33; HIGH >= Q67**. Exclude the middle third from binary training/evaluation labels. Middle samples still consume elapsed time and can contribute to history-support accounting. Apply thresholds unchanged to outer-test and all matched horizons. No test-derived target thresholds, preprocessing, feature selection, calibration or hyperparameter selection.

Five fixed subject-disjoint outer folds by subject_id. Participant-first aggregation; primary classification endpoint **participant-macro balanced accuracy**. Secondary Macro-F1, ROC-AUC, PR-AUC; undefined AUCs are omitted only for that metric with valid participant counts recorded. Paired bootstrap resamples participants, not EEG windows. Random-window splits are diagnostics, not evidence of cross-subject generalization.

Canonical dataset-level temporal alignment: **EEG(t-10 s) -> PM(t)**. PM [T,T+10] uses EEG [T-10,T]. Longer history extends backwards, exact record-relative ten-second lookup, oldest first and canonical block last. No gaps/substitution, future blocks, subject/record/fold crossing. This does not establish a physiological delay, brain mechanism or proprietary PM algorithm.

## Baselines and nested AutoML

[Model robustness](reports/diagnostics/pm_low_high_model_robustness_v1/summary_by_model.csv), mean of 35 participant-macro PM × fold rows:

| Model | BA | Macro-F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| LightGBM | 0.752780 | 0.716281 | 0.878977 | 0.868075 |
| XGBoost | 0.750079 | 0.715152 | 0.873250 | 0.862414 |

[Nested LOW/HIGH AutoML](reports/diagnostics/pm_low_high_nested_automl_v1/README.md) is complete: 2,730 inner trainings and 35 final outer fits. Historical label_q5 Transformer AutoML was only a pilot; it is not the current AutoML status.

Earlier LOW/HIGH matched LSTM/Transformer comparisons did not establish a stable advantage over tabular baselines. Their support/context differ from the newly justified H120 comparison; the latter has not run.

## Completed temporal diagnostics

[Diagnostic report](reports/diagnostics/pm_temporal_predictability_audit_v1/README.md): H120 retains 29,437 windows, 95.56% of H10 support, 53 participants, 84 records before LOW/HIGH filtering.

| Delay (s) | 10 | 20 | 30 | 60 | 120 |
|---|---:|---:|---:|---:|---:|
| Median all-feature record ACF | .2655 | .1597 | .1313 | .0991 | .0402 |
| Median excess normalized MI | .0493 | .0255 | .0204 | .0132 | .0075 |

MI measures a more general dependence class than correlation; it does not prove nonlinearity. PE is near maximal, with a small systematic order effect. SSA observed/surrogate effective ranks 41.9408/53.3163, rank90 43.5/47.0 and leading energy .1172/.0462 indicate structured but multicomponent trajectories. Restricted FNN has only 14–16 eligible records: no reliable embedding dimension, physical attractor or Takens claim. These are diagnostics, separate from PM predictive evidence.

## Completed matched prediction and order control

[Full matched evidence](reports/diagnostics/pm_temporal_predictability_v1/README.md): strict_common_support_h120; LightGBM 200 estimators, n_jobs=4, random_state=42. All 175 fits executed, no restores/failures. Identical target samples, labels, subjects, folds, thresholds and counts across horizons within PM/fold.

| Horizon | BA | Macro-F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| H10 | .7505 | .7175 | .8823 | .8687 |
| H20 | .7608 | .7277 | .8996 | .8859 |
| H30 | .7663 | .7331 | .9052 | .8905 |
| H60 | .7708 | .7382 | .9073 | .8927 |
| H120 | .7819 | .7501 | .9113 | .8973 |

Equal-PM means are descriptive, not replacements for target-specific paired inference. H120-vs-H10 BA gains pass BH for excitement (+.0197), focus (+.1071), interest (+.0345), relaxation (+.0195), stress (+.0341): **5/7**. Attention/engagement have no confirmed H120 gain. Participant bootstrap: 10,000, seed 42; BH over exactly 28 primary contrasts.

[Order control](reports/diagnostics/pm_temporal_predictability_v1/order_control_h120/README.md): **25/25**, no failures/restores, ordered H120 reused. Same five gated PM, H120, five folds. Existing deterministic per-sample shuffle permutes only the 11 older blocks (-120 through -20 s); canonical -10 stays fixed. Hashes/provenance verify data identity and block preservation.

| PM | Ordered BA | Shuffled BA | Delta | BH q |
|---|---:|---:|---:|---:|
| excitement | .86061 | .85321 | +.00740 | .00850 |
| focus | .81268 | .73349 | +.07919 | .00025 |
| interest | .76239 | .75380 | +.00859 | .00850 |
| relaxation | .80252 | .78817 | +.01435 | .00025 |
| stress | .77320 | .76352 | +.00969 | .07079 |

Ordered-minus-shuffled BA: 10,000 participant-paired bootstrap resamples, seed 42, percentile 95% CI, centered-null two-sided p, BH across **five** conditional contrasts. Four pass; stress CI crosses zero and is **inconclusive**, not equivalence. Focus: 52/53 participants and 5/5 folds favor ordered. The two-stage selection uses the same cohort, one shuffle realization and overlapping training folds; these are not unconditional independent tests across all seven PM. Secondary metrics are descriptive.

## Personalization and supporting tracks

[LOW/HIGH weighted augmentation](reports/diagnostics/pm_low_high_personalized_weighted_augmentation_v1/summary_by_estimand.csv): 30-minute chronological calibration. Operational support 50 participants/345 participant-PM pairs with fallback; adaptation applied to 285 pairs/48 participants. XGBoost delta ROC-AUC +.0162159, PR-AUC +.0147896; LightGBM +.0146802/+.0152810. Original outer-train target thresholds stay fixed. Gains are modest; do not launch another broad sweep.

Continuous seven-PM regression remains supporting evidence: [RF versus mean baseline](reports/integration/pm_multioutput_regression.md), [alignment regression](reports/diagnostics/pm_eeg_lag_final_conclusion.md). Regression and historical Q3 classification metrics must not be relabelled as LOW/HIGH.

External COG-BCI N-Back (zero/one/two-back), MATB (easy/medium/difficult) and CL-Drive/CLARE subjective_cognitive_load_3class_fixed are **native three-class tasks**, not obsolete internal PM Q3. Synthetic three-class fixtures are also valid. WESAD is physiological, not EEG. External results use different targets/support and do not validate all seven internal PM conclusions automatically.

Historical label_q5 DANN, FOMAML and contrastive tracks remain supporting/negative evidence; FOMAML do_not_proceed and closed transfer screenings are not invitations to repeat sweeps. Streaming replay/demo is implemented; software-window latency is not measured sensor-to-user latency.

## History, limitations and authority

Internal label_q5 (five-level Focus) and fold-local Q3 LOW/MIDDLE/HIGH are **historical/legacy**, not current primary classification. Preserve their code, configurations, metrics, seeds, folds and provenance. Native external three-class semantics are unchanged.

[2026-08-29 final snapshot](reports/integration/project_final_state.md) and [historical final package](reports/summary/final_project_results.md) describe earlier evidence, not competing current states. [Current experiment registry](reports/summary/experiment_registry.yaml) separates scientific_status from execution status. Current summaries must not rank incompatible cohorts/tasks as one benchmark.

PM are device-derived proxies, not direct cognitive ground truth. Internal independent sample size is at most 54 participants; H120 uses 53. Proprietary smoothing may influence alignment. No physiological causality, small brain attractor, universal history benefit or guaranteed Transformer advantage is established. Negative findings and uncertainty remain part of the scientific record.
