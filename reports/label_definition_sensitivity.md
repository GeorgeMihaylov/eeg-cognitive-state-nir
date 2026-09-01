# Label definition sensitivity

The existing `label_q5` remains unchanged. Four thresholds are fitted only from the outer-train subjects of each canonical GroupKFold split and applied unchanged to that fold's train and test targets.

## Fold-specific thresholds

| Fold | Train / test windows | q20 (delta) | q40 (delta) | q60 (delta) | q80 (delta) | Changed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36261 / 9123 | 0.3342750 (+0.0040980) | 0.3910700 (+0.0032840) | 0.4477690 (+0.0033110) | 0.5283060 (+0.0017210) | 3.6282% |
| 2 | 36392 / 8992 | 0.3293360 (-0.0008410) | 0.3858676 (-0.0019184) | 0.4417284 (-0.0027296) | 0.5218646 (-0.0047204) | 3.1028% |
| 3 | 36277 / 9107 | 0.3304860 (+0.0003090) | 0.3863070 (-0.0014790) | 0.4415964 (-0.0028616) | 0.5220132 (-0.0045718) | 2.5475% |
| 4 | 36328 / 9056 | 0.3273968 (-0.0027802) | 0.3867420 (-0.0010440) | 0.4444634 (+0.0000054) | 0.5300206 (+0.0034356) | 1.5459% |
| 5 | 36278 / 9106 | 0.3293054 (-0.0008716) | 0.3888320 (+0.0010460) | 0.4476856 (+0.0032276) | 0.5315140 (+0.0049290) | 2.5807% |

All folds retained four unique internal thresholds and all five classes. `duplicates='drop'` therefore has no effect in this sensitivity analysis.

### Threshold distribution across folds

| Threshold | Global | Mean | SD | Min | Max | Max absolute delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q20 | 0.3301770 | 0.3301598 | 0.0025536 | 0.3273968 | 0.3342750 | 0.0040980 |
| q40 | 0.3877860 | 0.3877637 | 0.0021699 | 0.3858676 | 0.3910700 | 0.0032840 |
| q60 | 0.4444580 | 0.4446486 | 0.0030347 | 0.4415964 | 0.4477690 | 0.0033110 |
| q80 | 0.5265850 | 0.5267437 | 0.0045309 | 0.5218646 | 0.5315140 | 0.0049290 |

### Leakage-safe class balance

Counts and fractions are ordered as classes `[0, 1, 2, 3, 4]`.

| Fold | Train counts | Train fractions | Test counts | Test fractions | Max test deviation from 0.20 |
| ---: | --- | --- | --- | --- | ---: |
| 1 | [7253, 7254, 7250, 7252, 7252] | [0.2, 0.2, 0.1999, 0.2, 0.2] | [2392, 1830, 1685, 1519, 1697] | [0.2622, 0.2006, 0.1847, 0.1665, 0.186] | 6.2194% |
| 2 | [7280, 7277, 7278, 7278, 7279] | [0.2, 0.2, 0.2, 0.2, 0.2] | [1694, 1568, 1712, 1803, 2215] | [0.1884, 0.1744, 0.1904, 0.2005, 0.2463] | 4.6330% |
| 3 | [7256, 7256, 7254, 7255, 7256] | [0.2, 0.2, 0.2, 0.2, 0.2] | [1877, 1519, 1636, 1850, 2225] | [0.2061, 0.1668, 0.1796, 0.2031, 0.2443] | 4.4318% |
| 4 | [7266, 7265, 7266, 7265, 7266] | [0.2, 0.2, 0.2, 0.2, 0.2] | [1434, 2002, 1998, 2078, 1544] | [0.1583, 0.2211, 0.2206, 0.2295, 0.1705] | 4.1652% |
| 5 | [7256, 7256, 7255, 7256, 7255] | [0.2, 0.2, 0.2, 0.2, 0.2] | [1712, 2130, 2044, 1765, 1455] | [0.188, 0.2339, 0.2245, 0.1938, 0.1598] | 4.0215% |

## Label agreement

- Agreement: 97.3184%
- Changed: 2.6816% (1217 windows)
- Mean absolute shift: 0.026816
- One-class shifts: 2.6816%
- Two-or-more-class shifts: 0.0000%
- Cohen's kappa: 0.966480
- Quadratic weighted kappa: 0.993307

### Fold-level agreement

| Fold | Agreement | Changed | One-class | Two-or-more | Kappa | QWK |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 96.3718% | 3.6282% | 3.6282% | 0.0000% | 0.954405 | 0.991420 |
| 2 | 96.8972% | 3.1028% | 3.1028% | 0.0000% | 0.961103 | 0.992547 |
| 3 | 97.4525% | 2.5475% | 2.5475% | 0.0000% | 0.968044 | 0.994062 |
| 4 | 98.4541% | 1.5459% | 1.5459% | 0.0000% | 0.980588 | 0.995669 |
| 5 | 97.4193% | 2.5807% | 2.5807% | 0.0000% | 0.967626 | 0.992940 |

### Sensitivity by class

| Grouping | Class | Windows | Changed | Mean absolute shift |
| --- | ---: | ---: | ---: | ---: |
| Global label | 0 | 9080 | 1.1674% | 0.011674 |
| Global label | 1 | 9075 | 3.1295% | 0.031295 |
| Global label | 2 | 9075 | 3.5813% | 0.035813 |
| Global label | 3 | 9078 | 4.0648% | 0.040648 |
| Global label | 4 | 9076 | 1.4654% | 0.014654 |
| Fold-train label | 0 | 9109 | 1.4821% | 0.014821 |
| Fold-train label | 1 | 9049 | 2.8511% | 0.028511 |
| Fold-train label | 2 | 9075 | 3.5813% | 0.035813 |
| Fold-train label | 3 | 9015 | 3.3943% | 0.033943 |
| Fold-train label | 4 | 9136 | 2.1125% | 0.021125 |

## Source sensitivity

| Source | Windows | Changed | Mean absolute shift |
| --- | ---: | ---: | ---: |
| Old_EEG | 21558 | 2.7554% | 0.027554 |
| gpn_data | 23826 | 2.6148% | 0.026148 |

## Most sensitive subjects

| Subject | Windows | Changed |
| --- | ---: | ---: |
| 7150e10a | 1555 | 5.9807% |
| 40f0714a | 155 | 4.5161% |
| d18000a3 | 680 | 4.4118% |
| a1721173 | 592 | 4.3919% |
| c112918e | 844 | 4.2654% |
| 7072a0e0 | 1353 | 3.9911% |
| f0f2a1e1 | 507 | 3.9448% |
| 41e2010c | 682 | 3.8123% |
| 71f0603f | 1025 | 3.8049% |
| 21a031f6 | 716 | 3.7709% |

## Temporal comparison

| Label | Persistence | Adjacent transition | Severe transition | Mean run | Previous-label balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global | 58.4047% | 34.4798% | 7.1155% | 2.389 | 0.584250 |
| Fold-train | 58.3450% | 34.5484% | 7.1067% | 2.386 | 0.583155 |

Median run length is 1.0 window for the global label and 1.0 window for the fold-train label.

| Label | Accuracy | Balanced accuracy | Macro F1 | Ordinal MAE | Adjacent accuracy | Severe error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Global | 0.584047 | 0.584250 | 0.584269 | 0.496935 | 0.928845 | 0.071155 |
| Fold-train | 0.583450 | 0.583155 | 0.583165 | 0.497422 | 0.928933 | 0.071067 |

D0-D3 were not repeated because the changed-label fraction did not exceed the configured 5% condition; no model was trained by this analysis.

## Recommendation

**Option B.** Keep global label_q5 as the predefined legacy task for reproducibility, and retain it as the predefined benchmark task for directly comparable experiments. Preserve the split-fitted result as a required sensitivity analysis and save its thresholds in every split artifact.

A focused rerun of major RF/Transformer baselines is recommended later to confirm ranking stability, but it is not urgent given the small, adjacent, non-systematic shifts.

The decision uses the threshold deviations, magnitude and direction of shifts, subject/source structure, weighted agreement, and temporal stability jointly; it is not based on a single acceptance cutoff.
