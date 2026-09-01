# Cross-source generalization

Strict source-exclusive transfer uses disjoint subjects. Shared-subject transfer is reported separately and is not a subject-independent estimate.

| Direction | Mode | Model | Status | N test | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Kappa | AUC | Severe error | In-domain test-source BA | Delta BA |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpn_data -> Old_EEG | source_exclusive | random_forest | completed | 6717 | 0.2955 | 0.2890 | 0.2774 | 0.2842 | 0.1167 | 0.6161 | 0.3594 | 0.2819 | +0.0072 |
| gpn_data -> Old_EEG | source_exclusive | torch_transformer | completed | 6496 | 0.3819 | 0.3755 | 0.3648 | 0.3727 | 0.2271 | 0.7057 | 0.2534 | 0.3370 | +0.0385 |
| gpn_data -> Old_EEG | shared_subject | random_forest | invalid |  |  |  |  |  |  |  |  |  |  |
|  |  | invalid reasons | train subjects=1 is below configured minimum 5; test subjects=1 is below configured minimum 3 |  |  |  |  |  |  |  |  |  |  |
| gpn_data -> Old_EEG | shared_subject | torch_transformer | invalid |  |  |  |  |  |  |  |  |  |  |
|  |  | invalid reasons | train subjects=1 is below configured minimum 5; test subjects=1 is below configured minimum 3 |  |  |  |  |  |  |  |  |  |  |
| Old_EEG -> gpn_data | source_exclusive | random_forest | completed | 6348 | 0.2941 | 0.2946 | 0.2883 | 0.2902 | 0.1157 | 0.6116 | 0.3538 | 0.2951 | -0.0006 |
| Old_EEG -> gpn_data | source_exclusive | torch_transformer | completed | 6165 | 0.3392 | 0.3364 | 0.3146 | 0.3127 | 0.1673 | 0.6733 | 0.2996 | 0.3439 | -0.0075 |
| Old_EEG -> gpn_data | shared_subject | random_forest | invalid |  |  |  |  |  |  |  |  |  |  |
|  |  | invalid reasons | train subjects=1 is below configured minimum 5; test subjects=1 is below configured minimum 3 |  |  |  |  |  |  |  |  |  |  |
| Old_EEG -> gpn_data | shared_subject | torch_transformer | invalid |  |  |  |  |  |  |  |  |  |  |
|  |  | invalid reasons | train subjects=1 is below configured minimum 5; test subjects=1 is below configured minimum 3 |  |  |  |  |  |  |  |  |  |  |

## In-domain references

| Source | Model | Accuracy mean/std | Balanced accuracy mean/std | Macro F1 mean/std |
|---|---|---:|---:|---:|
| gpn_data | random_forest | 0.2967 / 0.0408 | 0.2951 / 0.0369 | 0.2900 / 0.0384 |
| gpn_data | torch_transformer | 0.3437 / 0.0253 | 0.3439 / 0.0279 | 0.3314 / 0.0214 |
| Old_EEG | random_forest | 0.2886 / 0.0264 | 0.2819 / 0.0179 | 0.2678 / 0.0232 |
| Old_EEG | torch_transformer | 0.3591 / 0.0207 | 0.3370 / 0.0338 | 0.3196 / 0.0290 |

## Training details

| Direction | Model | Device | Epochs | Best epoch | Best validation loss | Parameters | Training seconds |
|---|---|---|---:|---:|---:|---:|---:|
| gpn_data -> Old_EEG | random_forest |  |  |  |  |  | 0.384 |
| gpn_data -> Old_EEG | torch_transformer | NVIDIA GeForce RTX 5060 Ti | 12 | 8 | 1.5432 | 340869 | 4.921 |
| Old_EEG -> gpn_data | random_forest |  |  |  |  |  | 0.410 |
| Old_EEG -> gpn_data | torch_transformer | NVIDIA GeForce RTX 5060 Ti | 5 | 1 | 1.3937 | 340869 | 1.921 |

## Subject-level performance

Subject metrics are descriptive aggregates over test subjects; RF windows and Transformer sequences remain different prediction units.

| Direction | Model | Subjects | Balanced accuracy mean/std | Macro F1 mean/std | Severe error mean/std |
|---|---|---:|---:|---:|---:|
| gpn_data -> Old_EEG | random_forest | 12 | 0.2612 / 0.0396 | 0.2399 / 0.0448 | 0.3546 / 0.0671 |
| gpn_data -> Old_EEG | torch_transformer | 12 | 0.3329 / 0.0559 | 0.3000 / 0.0728 | 0.2604 / 0.0965 |
| Old_EEG -> gpn_data | random_forest | 11 | 0.2405 / 0.0386 | 0.2152 / 0.0468 | 0.3784 / 0.1073 |
| Old_EEG -> gpn_data | torch_transformer | 11 | 0.2748 / 0.0421 | 0.2439 / 0.0560 | 0.2907 / 0.0825 |

## Class-level error analysis

Rows in each confusion matrix are true classes and columns are predicted classes. Counts use each model's own prediction unit.

### gpn_data -> Old_EEG / random_forest

- True class counts: 0:1602 / 1:1277 / 2:1172 / 3:1282 / 4:1384.
- Predicted class counts: 0:1388 / 1:1824 / 2:939 / 3:711 / 4:1855.
- Per-class recall: 0:0.3196 / 1:0.3540 / 2:0.1638 / 3:0.1201 / 4:0.4877.

| True / predicted | 0 | 1 | 2 | 3 | 4 |
|---:|---:|---:|---:|---:|---:|
| 0 | 512 | 497 | 183 | 133 | 277 |
| 1 | 319 | 452 | 186 | 86 | 234 |
| 2 | 223 | 360 | 192 | 116 | 281 |
| 3 | 208 | 302 | 230 | 154 | 388 |
| 4 | 126 | 213 | 148 | 222 | 675 |

### gpn_data -> Old_EEG / torch_transformer

- True class counts: 0:1527 / 1:1242 / 2:1148 / 3:1259 / 4:1320.
- Predicted class counts: 0:1091 / 1:1329 / 2:1023 / 3:1139 / 4:1914.
- Per-class recall: 0:0.3798 / 1:0.3140 / 2:0.2038 / 3:0.2653 / 4:0.7144.

| True / predicted | 0 | 1 | 2 | 3 | 4 |
|---:|---:|---:|---:|---:|---:|
| 0 | 580 | 461 | 212 | 163 | 111 |
| 1 | 257 | 390 | 268 | 192 | 135 |
| 2 | 114 | 275 | 234 | 241 | 284 |
| 3 | 94 | 173 | 217 | 334 | 441 |
| 4 | 46 | 30 | 92 | 209 | 943 |

### Old_EEG -> gpn_data / random_forest

- True class counts: 0:1475 / 1:1468 / 2:1248 / 3:1097 / 4:1060.
- Predicted class counts: 0:1422 / 1:1346 / 2:1192 / 3:1002 / 4:1386.
- Per-class recall: 0:0.3607 / 1:0.2657 / 2:0.2220 / 3:0.1595 / 4:0.4651.

| True / predicted | 0 | 1 | 2 | 3 | 4 |
|---:|---:|---:|---:|---:|---:|
| 0 | 532 | 358 | 230 | 199 | 156 |
| 1 | 342 | 390 | 295 | 236 | 205 |
| 2 | 255 | 297 | 277 | 218 | 201 |
| 3 | 186 | 185 | 220 | 175 | 331 |
| 4 | 107 | 116 | 170 | 174 | 493 |

### Old_EEG -> gpn_data / torch_transformer

- True class counts: 0:1409 / 1:1421 / 2:1220 / 3:1068 / 4:1047.
- Predicted class counts: 0:2611 / 1:754 / 2:842 / 3:882 / 4:1076.
- Per-class recall: 0:0.6629 / 1:0.1499 / 2:0.1730 / 3:0.1910 / 4:0.5053.

| True / predicted | 0 | 1 | 2 | 3 | 4 |
|---:|---:|---:|---:|---:|---:|
| 0 | 934 | 193 | 157 | 91 | 34 |
| 1 | 770 | 213 | 195 | 169 | 74 |
| 2 | 507 | 180 | 211 | 189 | 133 |
| 3 | 293 | 100 | 165 | 204 | 306 |
| 4 | 107 | 68 | 114 | 229 | 529 |

## Interpretation

- CS1 is strict unseen-subject transfer: subjects, logical recordings, source records, sample IDs and canonical raw intervals have zero train/test overlap.
- CS2 is not estimable with the configured minimums after removing 33 exact logical-record duplicates: only one subject retains data in both sources, below both train and test subject minimums.
- All four completed CS1 accuracies exceed the five-class random reference of 0.20. Transformer exceeds RF in both directions.
- Transfer is asymmetric, especially for Transformer: gpn_data -> Old_EEG is stronger than Old_EEG -> gpn_data. This is descriptive, not a significance claim.
- Relative to source-only GroupKFold on the destination source, the observed balanced-accuracy deltas are small and can be positive or negative; the evaluated subject populations differ, so these are contextual references rather than paired estimates.
- Source-identity predictability was not trained or selected in this experiment; no separability claim is made from target-test labels.
- Both sources cover all five classes in every valid outer partition. No additional seeds, target fine-tuning or preprocessing variants were run.
- Middle classes 2 and 3 have the weakest recall in most completed trials; class 1 is additionally weak for the reverse-direction Transformer. Class-frequency and prediction-distribution shifts are shown above and may contribute to, but do not prove the cause of, the directional gap.

No difference is described as statistically significant without a separate paired analysis. A five-class random baseline is 0.20.
