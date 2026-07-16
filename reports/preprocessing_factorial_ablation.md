# ShallowConvNet preprocessing factorial ablation

## Scope and protocol

The complete 2×2×2 preprocessing factorial was run once with seed 42. The
model was `torch_shallow_convnet`; the target was `label_q5`; the input shape
was determined as `[1, 14, 2560]`. Training used CUDA on an NVIDIA GeForce RTX
5060 Ti, five precomputed outer subject folds, inner validation grouped by
`record_group_id`, at most 15 epochs, patience 4, batch size 128, learning rate
0.001, and weight decay 0.0001.

The trial mapping is:

| Trial | Band-pass | Notch | CAR | Pipeline |
|---|---:|---:|---:|---|
| A | 0 | 0 | 0 | raw |
| B | 1 | 0 | 0 | band-pass |
| C | 0 | 1 | 0 | notch |
| D | 0 | 0 | 1 | CAR |
| E | 1 | 1 | 0 | band-pass + notch |
| F | 1 | 0 | 1 | band-pass + CAR |
| G | 0 | 1 | 1 | notch + CAR |
| H | 1 | 1 | 1 | band-pass + notch + CAR |

## Integrity checks

All eight trials completed successfully. Every trial has five fold directories,
30,958 unified predictions, 54 test subjects across the five folds, and no
duplicate `sample_id`. `sample_id`, fold, and `y_true` match trial A exactly for
every trial. Outer subject overlap and inner logical-record overlap are both
zero in every fold. All `proba_0`–`proba_4` values are finite; the largest
absolute row-sum error is `2.38e-7`.

Every fold contains `model.pt`, `metrics.json`, `training_log.csv`,
`predictions.parquet`, `validation_split.json`, `normalization_stats.json`,
`preprocessing_metadata.json`, `selected_logical_records.parquet`, and
`rejected_windows.parquet`. The full run reused the eight prevalidated caches;
it did not invoke cache building.

## Trial ranking

Trials are ordered by mean balanced accuracy, with macro F1 as the tie-breaker.
Standard deviations are across the five outer folds.

| Rank | Trial | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Kappa | AUC | Epochs | Best val loss | Training (s) | Δ balanced vs A |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | E | 0.2849 ± 0.0145 | 0.2889 ± 0.0148 | 0.2659 ± 0.0250 | 0.2654 ± 0.0285 | 0.1087 ± 0.0174 | 0.6151 ± 0.0189 | 11.8 ± 3.2 | 1.5286 | 803.3 | +0.0065 |
| 2 | B | 0.2847 ± 0.0157 | 0.2873 ± 0.0156 | 0.2653 ± 0.0271 | 0.2650 ± 0.0305 | 0.1074 ± 0.0193 | 0.6145 ± 0.0193 | 11.8 ± 3.2 | 1.5288 | 786.8 | +0.0049 |
| 3 | C | 0.2807 ± 0.0177 | 0.2833 ± 0.0170 | 0.2611 ± 0.0250 | 0.2616 ± 0.0252 | 0.1031 ± 0.0202 | 0.6039 ± 0.0237 | 11.0 ± 3.4 | 1.5500 | 737.7 | +0.0009 |
| 4 | A | 0.2793 ± 0.0164 | 0.2824 ± 0.0170 | 0.2599 ± 0.0150 | 0.2599 ± 0.0141 | 0.1010 ± 0.0189 | 0.6031 ± 0.0172 | 10.6 ± 3.3 | 1.5413 | 713.5 | 0.0000 |
| 5 | G | 0.2602 ± 0.0033 | 0.2634 ± 0.0072 | 0.2433 ± 0.0098 | 0.2438 ± 0.0080 | 0.0780 ± 0.0058 | 0.5844 ± 0.0119 | 11.0 ± 3.4 | 1.5629 | 794.7 | -0.0190 |
| 6 | D | 0.2596 ± 0.0081 | 0.2620 ± 0.0115 | 0.2396 ± 0.0132 | 0.2398 ± 0.0138 | 0.0759 ± 0.0108 | 0.5823 ± 0.0130 | 10.0 ± 3.2 | 1.5679 | 694.2 | -0.0204 |
| 7 | F | 0.2489 ± 0.0204 | 0.2514 ± 0.0243 | 0.2315 ± 0.0158 | 0.2314 ± 0.0146 | 0.0628 ± 0.0270 | 0.5659 ± 0.0217 | 10.4 ± 2.5 | 1.5527 | 736.9 | -0.0310 |
| 8 | H | 0.2484 ± 0.0208 | 0.2510 ± 0.0246 | 0.2315 ± 0.0164 | 0.2313 ± 0.0153 | 0.0622 ± 0.0275 | 0.5660 ± 0.0218 | 10.4 ± 2.5 | 1.5519 | 731.0 | -0.0314 |

The sum of fold training times is 5,998.1 seconds (100.0 minutes); end-to-end
wall time for the sequential command was 6,018.5 seconds (100.3 minutes).

## Fold-level balanced accuracy

Each cell is `balanced accuracy (paired delta versus A on the same fold)`.
Full accuracy, macro/weighted F1, kappa, AUC, training metadata, and all paired
deltas are in `reports/preprocessing_factorial_ablation_folds.csv`.

| Trial | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---:|---:|---:|---:|---:|
| A | 0.2783 (+0.0000) | 0.2925 (+0.0000) | 0.2529 (+0.0000) | 0.3035 (+0.0000) | 0.2847 (+0.0000) |
| B | 0.2913 (+0.0130) | 0.3088 (+0.0163) | 0.2612 (+0.0084) | 0.2931 (-0.0104) | 0.2821 (-0.0026) |
| C | 0.2706 (-0.0077) | 0.2990 (+0.0065) | 0.2693 (+0.0164) | 0.3086 (+0.0051) | 0.2689 (-0.0158) |
| D | 0.2529 (-0.0254) | 0.2609 (-0.0316) | 0.2482 (-0.0046) | 0.2667 (-0.0368) | 0.2811 (-0.0037) |
| E | 0.2911 (+0.0128) | 0.3067 (+0.0142) | 0.2616 (+0.0088) | 0.2927 (-0.0108) | 0.2923 (+0.0075) |
| F | 0.2344 (-0.0439) | 0.2661 (-0.0264) | 0.2172 (-0.0357) | 0.2523 (-0.0512) | 0.2872 (+0.0025) |
| G | 0.2527 (-0.0256) | 0.2644 (-0.0281) | 0.2579 (+0.0051) | 0.2694 (-0.0341) | 0.2724 (-0.0124) |
| H | 0.2355 (-0.0428) | 0.2663 (-0.0262) | 0.2154 (-0.0375) | 0.2510 (-0.0525) | 0.2867 (+0.0020) |

E is above A on four of five folds for balanced accuracy, while B is above A
on three. CAR-containing F and H are below A on four folds and nearly tied on
fold 5. No statistical significance is claimed from five folds and one seed.

## Factorial effects

Factors were coded as -1/+1. Each reported effect is the mean response when
the corresponding coded product is +1 minus the mean when it is -1. Thus the
main effects are marginal high-minus-low differences, and interaction effects
are orthogonal factorial contrasts.

| Effect | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Kappa | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Band-pass | -0.00321 | -0.00309 | -0.00245 | -0.00297 | -0.00422 | -0.00305 |
| Notch | +0.00042 | +0.00085 | +0.00136 | +0.00150 | +0.00124 | +0.00090 |
| CAR | -0.02812 | -0.02854 | -0.02654 | -0.02643 | -0.03531 | -0.03451 |
| Band-pass × notch | -0.00058 | -0.00030 | -0.00109 | -0.00134 | -0.00093 | -0.00054 |
| Band-pass × CAR | -0.00798 | -0.00837 | -0.00750 | -0.00747 | -0.01022 | -0.01438 |
| Notch × CAR | -0.00036 | -0.00037 | +0.00048 | +0.00047 | -0.00047 | +0.00018 |
| Band-pass × notch × CAR | +0.00004 | -0.00063 | -0.00080 | -0.00072 | -0.00046 | -0.00044 |

For balanced accuracy, the across-fold mean ± standard deviation of the
factorial effect was:

| Effect | Mean ± std across folds |
|---|---:|
| Band-pass | -0.00309 ± 0.01158 |
| Notch | +0.00085 ± 0.00340 |
| CAR | -0.02854 ± 0.01496 |
| Band-pass × notch | -0.00030 ± 0.00528 |
| Band-pass × CAR | -0.00837 ± 0.00795 |
| Notch × CAR | -0.00037 ± 0.00145 |
| Band-pass × notch × CAR | -0.00063 ± 0.00216 |

The marginal band-pass effect is negative because it interacts with CAR. With
CAR disabled, band-pass improves balanced accuracy by +0.00495 for B versus A
and +0.00560 for E versus C. With CAR enabled, it decreases balanced accuracy
by -0.01053 for F versus D and -0.01239 for H versus G. The notch main effect
is small. CAR is the dominant negative factor: its balanced-accuracy effect is
negative on all five folds (`-0.03897`, `-0.03735`, `-0.02657`, `-0.03965`,
and `-0.00017`), although fold 5 is effectively neutral.

## Interpretation and next checkpoint

All trial means are above the five-class random level of 0.20, ranging from
0.2510 to 0.2889 balanced accuracy. E and B are the top two variants, but their
advantages over raw A are small (+0.0065 and +0.0049), and E exceeds B by only
0.0015. G has the lowest fold standard deviation, but at a materially lower
mean. No leakage, probability, sample-identity, or artifact anomaly was found;
the strong negative CAR result is a scientific observation to replicate, not
evidence of a technical failure.

For additional seeds, the recommended checkpoint set is E, B, and A: the two
leaders plus the raw control. C ranks third numerically, but its +0.0009 gain
over A is too small to justify dropping the control. Seeds 7 and 123, EEGNet,
artifact rejection, and further preprocessing variants were not run here.

Full artifacts are under
`benchmark_results/preprocessing_ablation_shallowconvnet/full/trial_A/seed_42`
through `trial_H/seed_42`. Machine-readable trial and fold tables are in
`reports/preprocessing_factorial_ablation_trials.csv` and
`reports/preprocessing_factorial_ablation_folds.csv`.
