# F3/F4 channel-drop pilot

Status: complete; diagnostic descriptive screening. No automatic winner or next-channel selection.

## Protocol and validation

Frozen 30,958 EEG windows / 54 participants / five fixed subject-disjoint folds / seven PM. Exact previous-window EEG(t-10s) -> PM(t); existing outer-train-only Q33/Q67 thresholds; MIDDLE excluded. Pre-training checks cover all 105 representation/fold/PM combinations, including LOW/HIGH/MIDDLE IDs, subjects, thresholds and pairing.

35 baseline LightGBM runs reused after automatic artifact verification; 70 new LightGBM fits. n_estimators=200, n_jobs=4, random_state=42. Connectivity recomputed on the remaining 13 raw channels.

Overall aggregation: mean across available PM within each participant, then equal participant weight. Undefined metrics remain undefined and are excluded only from that metric's mean; valid counts are in the CSVs. Attention and Engagement have 53 evaluated participants; the other PM and overall have 54.

Paired bootstrap: 10,000 resamples of participants, seed 42, percentile 95% CI for mean BA difference. Overall bootstrap uses one available-PM-averaged paired difference per participant. Folds are not bootstrap units. Target-specific intervals are descriptive and not multiplicity-adjusted.

## Descriptive result

| variant | balanced_accuracy | macro_f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- |
| baseline_14ch | 0.753663 | 0.714697 | 0.878822 | 0.867690 |
| drop_F3_13ch | 0.747051 | 0.710956 | 0.875780 | 0.864779 |
| drop_F4_13ch | 0.753621 | 0.715354 | 0.879919 | 0.869994 |

## Paired result

| variant | pm | n_subjects | mean_delta_ba | median_delta_ba | fraction_participants_improved | fraction_participants_tied | fraction_participants_worsened | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drop_F3_13ch | overall | 54 | -0.006612 | -0.002112 | 0.370370 | 0.000000 | 0.629630 | -0.015615 | -0.000453 |
| drop_F4_13ch | overall | 54 | -0.000042 | -0.001429 | 0.444444 | 0.018519 | 0.537037 | -0.003064 | 0.003067 |

## PM-specific results

| pm | variant | n_subjects | balanced_accuracy | macro_f1 | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| attention | baseline_14ch | 53 | 0.721523 | 0.687718 | 0.864209 | 0.845185 |
| engagement | baseline_14ch | 53 | 0.733829 | 0.711132 | 0.862784 | 0.847298 |
| excitement | baseline_14ch | 54 | 0.843620 | 0.814411 | 0.946638 | 0.940761 |
| stress | baseline_14ch | 54 | 0.733751 | 0.687327 | 0.866030 | 0.877967 |
| relaxation | baseline_14ch | 54 | 0.790072 | 0.768392 | 0.930141 | 0.928068 |
| interest | baseline_14ch | 54 | 0.736632 | 0.678217 | 0.860406 | 0.855508 |
| focus | baseline_14ch | 54 | 0.709827 | 0.664301 | 0.821547 | 0.779043 |
| attention | drop_F3_13ch | 53 | 0.710966 | 0.676145 | 0.855663 | 0.839741 |
| engagement | drop_F3_13ch | 53 | 0.732623 | 0.710738 | 0.861534 | 0.844113 |
| excitement | drop_F3_13ch | 54 | 0.821045 | 0.801561 | 0.947583 | 0.942186 |
| stress | drop_F3_13ch | 54 | 0.736919 | 0.692258 | 0.867206 | 0.875871 |
| relaxation | drop_F3_13ch | 54 | 0.787704 | 0.767381 | 0.928824 | 0.925374 |
| interest | drop_F3_13ch | 54 | 0.734420 | 0.677072 | 0.851973 | 0.852434 |
| focus | drop_F3_13ch | 54 | 0.706486 | 0.663642 | 0.817675 | 0.773734 |
| attention | drop_F4_13ch | 53 | 0.715327 | 0.682275 | 0.864142 | 0.847110 |
| engagement | drop_F4_13ch | 53 | 0.735752 | 0.714534 | 0.864488 | 0.848245 |
| excitement | drop_F4_13ch | 54 | 0.842797 | 0.814221 | 0.948672 | 0.945818 |
| stress | drop_F4_13ch | 54 | 0.737037 | 0.693038 | 0.872400 | 0.879594 |
| relaxation | drop_F4_13ch | 54 | 0.786114 | 0.763350 | 0.927338 | 0.926042 |
| interest | drop_F4_13ch | 54 | 0.738489 | 0.678266 | 0.857194 | 0.857384 |
| focus | drop_F4_13ch | 54 | 0.713368 | 0.670379 | 0.825198 | 0.785764 |

| variant | pm | n_subjects | mean_delta_ba | median_delta_ba | fraction_participants_improved | fraction_participants_tied | fraction_participants_worsened | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drop_F3_13ch | attention | 53 | -0.010557 | -0.000494 | 0.471698 | 0.018868 | 0.509434 | -0.021857 | -0.000019 |
| drop_F3_13ch | engagement | 53 | -0.001206 | -0.004545 | 0.452830 | 0.018868 | 0.528302 | -0.007597 | 0.005263 |
| drop_F3_13ch | excitement | 54 | -0.022575 | -0.002521 | 0.425926 | 0.018519 | 0.555556 | -0.062928 | 0.000080 |
| drop_F3_13ch | focus | 54 | -0.003340 | 0.001413 | 0.537037 | 0.018519 | 0.444444 | -0.011834 | 0.004529 |
| drop_F3_13ch | interest | 54 | -0.002213 | -0.002445 | 0.388889 | 0.074074 | 0.537037 | -0.011068 | 0.006854 |
| drop_F3_13ch | relaxation | 54 | -0.002368 | -0.001978 | 0.444444 | 0.018519 | 0.537037 | -0.007642 | 0.003025 |
| drop_F3_13ch | stress | 54 | 0.003168 | 0.000000 | 0.481481 | 0.037037 | 0.481481 | -0.007321 | 0.013129 |
| drop_F4_13ch | attention | 53 | -0.006196 | -0.000912 | 0.471698 | 0.000000 | 0.528302 | -0.015768 | 0.002331 |
| drop_F4_13ch | engagement | 53 | 0.001923 | 0.001108 | 0.528302 | 0.037736 | 0.433962 | -0.005115 | 0.008911 |
| drop_F4_13ch | excitement | 54 | -0.000823 | -0.002833 | 0.407407 | 0.037037 | 0.555556 | -0.006450 | 0.005391 |
| drop_F4_13ch | focus | 54 | 0.003541 | -0.003038 | 0.407407 | 0.037037 | 0.555556 | -0.007087 | 0.016450 |
| drop_F4_13ch | interest | 54 | 0.001856 | 0.000000 | 0.462963 | 0.074074 | 0.462963 | -0.006049 | 0.009903 |
| drop_F4_13ch | relaxation | 54 | -0.003958 | -0.003063 | 0.351852 | 0.111111 | 0.537037 | -0.009189 | 0.001343 |
| drop_F4_13ch | stress | 54 | 0.003286 | 0.001507 | 0.537037 | 0.037037 | 0.425926 | -0.003595 | 0.010814 |

## Interpretation

Removing F3 reduced overall participant-first BA; its paired mean-difference CI is below zero. Effects vary across participants and PM; the mean is more negative than the median. Removing F4 was approximately neutral overall, with a CI including zero and mixed participant/target effects. These results do not establish equivalence or justify selecting additional channels. No post-hoc success threshold was introduced.

## Feature caches and provenance

- baseline_14ch: [30958, 371], float32; existing baseline (not rematerialized).
  Cache identity: `5062cac1e84a73c2e1c783f6b2c02f074e6177f2059515726e5b0433d166745f`.
  Feature hash: `454ea5db886fce8c981dc28f8910ca07355956033b001550020a04fd4605832e`.
- drop_F3_13ch: [30958, 346], float32; 1255.126 seconds.
  Cache identity: `faade1778bbc877fffacaeabe3ea76b43b1e966cee949113e498b7e13ae6ccea`.
  Feature hash: `5fd7838215855215b8dd043921a8056168b7754082620aac9c7fbb787de38d20`.
- drop_F4_13ch: [30958, 346], float32; 1279.861 seconds.
  Cache identity: `931ebf27250d6ac6f0270d62b6062b47838bb60c6ccf9c17bd1d3f2b8eb84fee`.
  Feature hash: `eb4256e2b003fd361a23ebde407f97b414a3f09696b24bece97ebfdfc55448f8`.

Protocol hash: `5881a400ddd292ce50f5b5b184478b67e4f5913f64adedfeedc647c79e367870`.
Branch: `analysis/pm-target-validity-audit-20260827`; HEAD: `222dbaf6ebe090b4211308ca673607aa04aabd55`.
Uncommitted channel-drop implementation; pre-existing unrelated modified/untracked artifacts retained. No commit created.

Code hashes, runtime versions, channel lists, cohort/fold/pairing identities and full configuration: `run_manifest.json`.

Artifacts: `summary_by_variant.csv`, `summary_by_target.csv`, `participant_metrics.csv`, `participant_deltas.csv`, `fold_target_results.csv`, `bootstrap_summary.csv`, `pretraining_contract.csv`, `cohort_equivalence.csv`.
