# LOW/HIGH confirmatory post-hoc robustness audit

## 1. Scope

This is a descriptive post-hoc robustness audit of the completed LOW-vs-HIGH
confirmatory experiment. It performs no training, tuning, participant removal,
lag search, threshold refitting or protocol modification. The scientific object
is **extreme-state separability**, not a deployable selective classifier.

## 2. Existing preregistered protocol

- preregistration code HEAD: `1e28fdae3b2ce2d75a4d90489960492299600a46`
- protocol hash: `ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431`
- targets/folds/runs: `7 / 5 / 35`
- alignment: fixed exact record-local `EEG(t-10 s) -> PM(t)`
- target: outer-train Q33/Q67 LOW/HIGH proxy; middle tertile excluded
- model: fixed XGBoost classifier, seed 42

This audit does not compare LOW/HIGH Macro-F1 directly with the distinct
three-class Q3 task.

## 3. Artifact integrity

All 35 run directories contain `predictions.parquet`, `participant_metrics.csv`
and `run_summary.json`. Independent participant recomputation matched the stored
metrics within tolerance; maximum absolute difference was
`2.22e-16`.

- prediction rows audited: `142750`
- PM-participant rows: `376`
- unique participants: `54`
- undefined participant ROC-AUC / PR-AUC: `5 / 5`
- fold×PM subjects with zero retained extreme rows: `1`
- duplicate prediction rows or within-run target IDs: `0 / 0`
- subject/fold anomalies: `0`
- protocol/specification/sample hashes: consistent

## 4. Participant-level distribution

| PM | valid n | mean BA | median BA | BA IQR | BA > .50 | BA >= .70 |
| --- | --- | --- | --- | --- | --- | --- |
| attention | 53 | 0.7175 | 0.7177 | [0.6612, 0.7763] | 53/53 | 34/53 |
| engagement | 53 | 0.7287 | 0.7256 | [0.6634, 0.7861] | 53/53 | 33/53 |
| excitement | 54 | 0.8405 | 0.8374 | [0.8041, 0.8917] | 54/54 | 51/54 |
| stress | 54 | 0.7353 | 0.7527 | [0.6672, 0.8128] | 51/54 | 36/54 |
| relaxation | 54 | 0.7911 | 0.8240 | [0.7228, 0.8612] | 54/54 | 43/54 |
| interest | 54 | 0.7360 | 0.7385 | [0.6729, 0.8139] | 53/54 | 35/54 |
| focus | 54 | 0.7010 | 0.7010 | [0.6554, 0.7544] | 52/54 | 27/54 |

Across the descriptive pooled PM-participant rows,
`370/376`
(`98.4%`) exceed BA 0.50,
`343` (`91.2%`) reach BA >= 0.60,
`259` (`68.9%`) reach BA >= 0.70,
and `136` (`36.2%`) reach BA >= 0.80.
These thresholds are descriptive and are not significance tests.

## 5. Bootstrap uncertainty

Percentile 95% intervals use
`10,000` deterministic bootstrap
replicates, seed `42`.
The pooled analysis resamples unique `subject_id` clusters and carries all PM
rows belonging to each sampled participant.

| Metric | Observed mean | Clustered 95% CI | clusters |
| --- | --- | --- | --- |
| balanced_accuracy | 0.7501 | [0.7342, 0.7661] | 54 |
| macro_f1 | 0.7149 | [0.6928, 0.7365] | 54 |
| roc_auc | 0.8728 | [0.8607, 0.8846] | 53 |

These are descriptive uncertainty intervals, not formal confirmatory tests.

## 6. Fold robustness

The worst BA fold×PM cell is `attention` fold 4
with BA `0.6764`. The best is `excitement`
fold 2 with BA `0.8812`.
The weakest PM mean is `focus` (`0.7010`),
and the strongest is `excitement` (`0.8405`).
No protocol element is changed in response to these post-hoc observations.

## 7. Class-balance analysis

Pooled repeated-measures Spearman correlations were:

- absolute class imbalance vs BA: rho `-0.1067`;
- absolute class imbalance vs Macro-F1: rho `-0.2294`;
- test-window count vs BA: rho `-0.0282`.

P-values in the CSV are explicitly exploratory. The pooled rows repeat
participants across PM and therefore do not supply independent inferential units.
Correlations are descriptive associations and cannot establish a causal role for
class balance or sample count.

## 8. LOW/HIGH recall symmetry

| PM | mean LOW | mean HIGH | HIGH-LOW | HIGH > LOW | LOW > HIGH |
| --- | --- | --- | --- | --- | --- |
| attention | 0.7345 | 0.7004 | -0.0340 | 52.8% | 47.2% |
| engagement | 0.7031 | 0.7544 | 0.0512 | 60.4% | 39.6% |
| excitement | 0.8364 | 0.8385 | 0.0021 | 47.2% | 52.8% |
| stress | 0.7093 | 0.7702 | 0.0609 | 54.7% | 45.3% |
| relaxation | 0.7818 | 0.7924 | 0.0106 | 45.3% | 54.7% |
| interest | 0.7294 | 0.7326 | 0.0031 | 49.1% | 50.9% |
| focus | 0.6890 | 0.7206 | 0.0316 | 56.6% | 43.4% |

LOW and HIGH recall are paired within participant and are not treated as
independent observations.

## 9. Cross-PM participant difficulty

The median off-diagonal Spearman correlation between participant BA profiles
across PM is `0.2088`. This quantifies whether relative participant
difficulty tends to recur across outcomes; it is descriptive and based on
pairwise available participants.

Bottom 10 participants by mean BA across available PM:

| Subject | Mean BA | Median BA | Min BA | PM | Windows | Min PM n | Mean minority | One-class PM | PM-specific BA / n / minority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3110e0c7 | 0.6312 | 0.6220 | 0.3732 | 7 | 1781 | 109 | 0.3563 | 0 | attention:BA=0.537,n=109,minority=0.38; engagement:BA=0.654,n=261,minority=0.36; excitement:BA=0.816,n=310,minority=0.30; focus:BA=0.622,n=291,minority=0.22; interest:BA=0.612,n=307,minority=0.41; relaxation:BA=0.806,n=284,minority=0.45; stress:BA=0.373,n=219,minority=0.37 |
| 50c02189 | 0.6339 | 0.6061 | 0.5123 | 7 | 4794 | 479 | 0.3898 | 0 | attention:BA=0.705,n=586,minority=0.24; engagement:BA=0.813,n=744,minority=0.31; excitement:BA=0.740,n=479,minority=0.43; focus:BA=0.512,n=657,minority=0.46; interest:BA=0.515,n=795,minority=0.45; relaxation:BA=0.545,n=689,minority=0.37; stress:BA=0.606,n=844,minority=0.47 |
| c112918e | 0.6413 | 0.5837 | 0.5233 | 7 | 3718 | 347 | 0.3919 | 0 | attention:BA=0.584,n=608,minority=0.38; engagement:BA=0.710,n=541,minority=0.46; excitement:BA=0.804,n=566,minority=0.35; focus:BA=0.756,n=555,minority=0.26; interest:BA=0.548,n=494,minority=0.43; relaxation:BA=0.523,n=607,minority=0.43; stress:BA=0.564,n=347,minority=0.44 |
| 8191f1d9 | 0.6657 | 0.6846 | 0.5352 | 7 | 4180 | 508 | 0.2846 | 0 | attention:BA=0.749,n=508,minority=0.30; engagement:BA=0.659,n=623,minority=0.19; excitement:BA=0.730,n=547,minority=0.42; focus:BA=0.764,n=539,minority=0.35; interest:BA=0.535,n=688,minority=0.13; relaxation:BA=0.685,n=629,minority=0.28; stress:BA=0.539,n=646,minority=0.32 |
| 0182e16c | 0.6682 | 0.6457 | 0.5305 | 7 | 2539 | 331 | 0.2482 | 0 | attention:BA=0.766,n=384,minority=0.32; engagement:BA=0.646,n=355,minority=0.34; excitement:BA=0.916,n=401,minority=0.08; focus:BA=0.530,n=351,minority=0.21; interest:BA=0.674,n=375,minority=0.29; relaxation:BA=0.605,n=331,minority=0.13; stress:BA=0.541,n=342,minority=0.37 |
| b0700166 | 0.6729 | 0.6704 | 0.5704 | 7 | 3405 | 430 | 0.3436 | 0 | attention:BA=0.570,n=501,minority=0.33; engagement:BA=0.730,n=479,minority=0.39; excitement:BA=0.692,n=430,minority=0.42; focus:BA=0.670,n=476,minority=0.40; interest:BA=0.636,n=548,minority=0.21; relaxation:BA=0.773,n=467,minority=0.30; stress:BA=0.638,n=504,minority=0.35 |
| 81f1f0fe | 0.6816 | 0.7256 | 0.4897 | 7 | 3120 | 373 | 0.3448 | 0 | attention:BA=0.626,n=373,minority=0.47; engagement:BA=0.726,n=443,minority=0.21; excitement:BA=0.787,n=381,minority=0.39; focus:BA=0.490,n=438,minority=0.16; interest:BA=0.738,n=524,minority=0.47; relaxation:BA=0.772,n=426,minority=0.43; stress:BA=0.633,n=535,minority=0.29 |
| 30c140ca | 0.6866 | 0.7356 | 0.3519 | 7 | 2088 | 253 | 0.3847 | 0 | attention:BA=0.776,n=280,minority=0.44; engagement:BA=0.669,n=253,minority=0.41; excitement:BA=0.804,n=266,minority=0.38; focus:BA=0.736,n=301,minority=0.38; interest:BA=0.352,n=352,minority=0.21; relaxation:BA=0.715,n=331,minority=0.43; stress:BA=0.754,n=305,minority=0.44 |
| 7150e10a | 0.6876 | 0.6977 | 0.5882 | 7 | 3417 | 235 | 0.3066 | 0 | attention:BA=0.755,n=555,minority=0.39; engagement:BA=0.709,n=533,minority=0.31; excitement:BA=0.831,n=545,minority=0.19; focus:BA=0.639,n=504,minority=0.17; interest:BA=0.588,n=235,minority=0.31; relaxation:BA=0.594,n=483,minority=0.40; stress:BA=0.698,n=562,minority=0.37 |
| 7092f07b | 0.6929 | 0.7018 | 0.5505 | 7 | 2328 | 278 | 0.3699 | 0 | attention:BA=0.550,n=377,minority=0.42; engagement:BA=0.771,n=301,minority=0.36; excitement:BA=0.739,n=280,minority=0.33; focus:BA=0.655,n=305,minority=0.29; interest:BA=0.659,n=414,minority=0.34; relaxation:BA=0.702,n=278,minority=0.40; stress:BA=0.773,n=373,minority=0.45 |

Top 10 participants:

| Subject | Mean BA | Median BA | Min BA | PM | Windows | Min PM n | Mean minority | One-class PM | PM-specific BA / n / minority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 71e10186 | 0.9446 | 0.9499 | 0.8963 | 7 | 3935 | 489 | 0.4224 | 0 | attention:BA=0.948,n=499,minority=0.40; engagement:BA=0.950,n=591,minority=0.48; excitement:BA=0.952,n=516,minority=0.42; focus:BA=0.896,n=489,minority=0.37; interest:BA=0.918,n=546,minority=0.47; relaxation:BA=0.997,n=643,minority=0.39; stress:BA=0.950,n=651,minority=0.44 |
| d111e017 | 0.8795 | 0.9047 | 0.7567 | 7 | 6333 | 829 | 0.4682 | 0 | attention:BA=0.841,n=866,minority=0.48; engagement:BA=0.757,n=829,minority=0.50; excitement:BA=0.943,n=896,minority=0.46; focus:BA=0.855,n=839,minority=0.50; interest:BA=0.911,n=897,minority=0.42; relaxation:BA=0.945,n=958,minority=0.46; stress:BA=0.905,n=1048,minority=0.46 |
| 71a251fa | 0.8475 | 0.8987 | 0.6515 | 7 | 1502 | 80 | 0.3198 | 0 | attention:BA=0.806,n=230,minority=0.44; engagement:BA=0.917,n=265,minority=0.16; excitement:BA=0.977,n=219,minority=0.21; focus:BA=0.652,n=222,minority=0.11; interest:BA=0.899,n=241,minority=0.46; relaxation:BA=0.903,n=245,minority=0.40; stress:BA=0.779,n=80,minority=0.46 |
| d18000a3 | 0.8214 | 0.8366 | 0.6988 | 7 | 2913 | 249 | 0.3833 | 0 | attention:BA=0.699,n=493,minority=0.43; engagement:BA=0.786,n=432,minority=0.48; excitement:BA=0.837,n=430,minority=0.37; focus:BA=0.760,n=440,minority=0.36; interest:BA=0.894,n=391,minority=0.41; relaxation:BA=0.861,n=478,minority=0.46; stress:BA=0.913,n=249,minority=0.17 |
| f0f2a1e1 | 0.8136 | 0.8129 | 0.7283 | 7 | 2346 | 233 | 0.2952 | 0 | attention:BA=0.797,n=282,minority=0.36; engagement:BA=0.861,n=321,minority=0.45; excitement:BA=0.838,n=399,minority=0.16; focus:BA=0.728,n=399,minority=0.09; interest:BA=0.813,n=233,minority=0.38; relaxation:BA=0.868,n=317,minority=0.31; stress:BA=0.789,n=395,minority=0.32 |
| 0110f12e | 0.8096 | 0.8252 | 0.7182 | 7 | 3021 | 371 | 0.4411 | 0 | attention:BA=0.785,n=446,minority=0.43; engagement:BA=0.826,n=451,minority=0.44; excitement:BA=0.825,n=412,minority=0.50; focus:BA=0.718,n=386,minority=0.43; interest:BA=0.797,n=371,minority=0.36; relaxation:BA=0.854,n=465,minority=0.45; stress:BA=0.861,n=490,minority=0.47 |
| b1c2f044 | 0.8086 | 0.8331 | 0.6150 | 7 | 2716 | 236 | 0.3072 | 0 | attention:BA=0.615,n=369,minority=0.49; engagement:BA=0.780,n=435,minority=0.26; excitement:BA=0.914,n=439,minority=0.29; focus:BA=0.756,n=430,minority=0.25; interest:BA=0.833,n=418,minority=0.19; relaxation:BA=0.915,n=389,minority=0.31; stress:BA=0.847,n=236,minority=0.36 |
| 5001d09a | 0.8069 | 0.7868 | 0.7602 | 7 | 1451 | 152 | 0.3547 | 0 | attention:BA=0.787,n=152,minority=0.46; engagement:BA=0.787,n=217,minority=0.41; excitement:BA=0.880,n=217,minority=0.24; focus:BA=0.760,n=219,minority=0.16; interest:BA=0.779,n=253,minority=0.46; relaxation:BA=0.843,n=159,minority=0.41; stress:BA=0.813,n=234,minority=0.35 |
| 40009139 | 0.8026 | 0.8065 | 0.6559 | 7 | 1707 | 191 | 0.4318 | 0 | attention:BA=0.656,n=264,minority=0.44; engagement:BA=0.873,n=251,minority=0.36; excitement:BA=0.912,n=276,minority=0.49; focus:BA=0.806,n=278,minority=0.38; interest:BA=0.814,n=191,minority=0.49; relaxation:BA=0.804,n=251,minority=0.44; stress:BA=0.753,n=196,minority=0.43 |
| 9192c107 | 0.8000 | 1.0000 | 0.5000 | 5 | 8 | 1 | 0.0000 | 5 | excitement:BA=1.000,n=1,minority=0.00; focus:BA=0.500,n=2,minority=0.00; interest:BA=1.000,n=1,minority=0.00; relaxation:BA=1.000,n=2,minority=0.00; stress:BA=0.500,n=2,minority=0.00 |

No participant is removed as a result of this ranking.

## 10. Worst-case behavior

Worst and best PM/fold cells remain visible in `fold_robustness.csv`. The bottom
participant table reports total and minimum PM-specific extreme-window counts,
class balance and one-class PM counts. This distinguishes genuine broad
difficulty from obvious tiny-sample or single-class cases without post-hoc
exclusion. Across the bottom 10, the smallest PM-specific test set contains
`109` extreme windows and the total number of one-class PM rows is
`0`. In contrast, the nominal top-10 entry `9192c107`
has only eight windows across five available PM and all five rows are one-class;
its high mean BA is therefore explicitly treated as sparse descriptive behavior,
not evidence of broadly strong generalization.

## 11. Limitations

- PM×participant rows are repeated measures across outcomes.
- PM targets are device-derived proxy measures, not ground-truth cognitive states.
- Nominal windows are temporally autocorrelated and not independent trials.
- Bootstrap intervals describe participant heterogeneity; they are not a new
  preregistered hypothesis test.
- Correlation with balance or window count is observational and non-causal.
- LOW/HIGH excludes the middle tertile and is a different task from Q3.

## 12. Scientific conclusion

**A. Majority versus driven subset.** The result is broadly distributed rather
than produced by a small high-performing subset: `370/376` PM-participant rows
have BA > 0.50, `343/376` have BA >= 0.60, and the pooled median BA is
`0.7539` (IQR
`0.6813`-`0.8261`). Every
PM has at least 88.9% of its available participants at BA >= 0.60. These are
descriptive thresholds, not participant-level significance tests.

**B. Class balance.** There is no evidence here that high extreme-state
separability is a trivial consequence of favorable balance or more test windows:
pooled absolute imbalance versus BA is weakly negative (rho
`-0.1067`), while window count versus BA is near zero
(rho `-0.0282`). Imbalance relates more strongly and
negatively to Macro-F1 (rho `-0.2294`), which is
directionally consistent with imbalance hurting, rather than creating, the score.
These repeated-measures associations remain exploratory and non-causal.

**C. Fold stability.** No fold×PM BA cell approaches chance: the worst is
`attention` fold 4 at
`0.6764`, versus the best `excitement`
fold 2 at `0.8812`.
`interest` has the largest five-fold BA range
(`0.1227`), so it is the clearest relative instability,
but not a protocol-breaking anomaly.

**D. Difficult participants.** Difficulty has modest cross-outcome persistence:
the median off-diagonal BA correlation is `0.2088`. The three
lowest mean-BA participants are `3110e0c7, 50c02189, c112918e`; all have seven PM available,
and the bottom 10 collectively contain no one-class PM row. Their lower scores
therefore cannot be dismissed as a single-class artifact, although outcome-
specific performance still varies and no participant is excluded.

**E. Stage decision.** Artifact integrity is complete and no protocol defect was
found. The LOW/HIGH confirmatory stage can remain closed and the project can move
to a separately preregistered model-robustness comparison. This is a workflow
recommendation based on the completed protocol plus descriptive robustness audit,
not a newly invented confirmatory selection criterion.
