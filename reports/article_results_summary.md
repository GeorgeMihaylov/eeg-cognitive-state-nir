# Article-ready results summary

## 1. Dataset and evaluation protocol

The supervised target is five-level `label_q5`. Scientific model results use five-fold GroupKFold by `subject_id`; inner validation is confined to outer-train data. Subject is the independent inferential unit.

## 2. Random-window sanity versus subject GroupKFold

Earlier random-window runs remain technical sanity checks and are not included in inferential claims. GroupKFold is the defensible primary protocol because it prevents train/test subject overlap.

## 3. Feature-window baselines

Random Forest and Torch MLP are exactly aligned on 45,384 ten-second EEG+POW feature windows. Their paired subject results are reported with bootstrap intervals, Wilcoxon, exact sign tests, and within-family Holm adjustment.

## 4. Sequence-model comparison

The available LSTM/BiLSTM pair uses 43,828 gap-aware sequences of length 10. Transformer uses 44,142 sequences of length 8. Therefore the recurrent pair can be tested against each other, but no exact paired Transformer comparison is currently defensible. Aggregate results can only be described.

## 5. Raw EEG model comparison

EEGNet and ShallowConvNet are exactly aligned on 30,958 raw deduplicated EEG windows for seeds 7, 42, and 123. Filtered runs are excluded from this model-family comparison.

## 6. Logical-record deduplication

Raw-model analyses use the logical-record-deduplicated dataset and its existing fold assignments; no cache or dataset was rebuilt.

## 7. Preprocessing ablation

| factor   |   on_mean |   off_mean |   main_effect_on_minus_off |
|:---------|----------:|-----------:|---------------------------:|
| bandpass |  0.269653 |   0.272745 |               -0.00309246  |
| notch    |  0.271623 |   0.270774 |                0.000848908 |
| car      |  0.256928 |   0.28547  |               -0.0285418   |

A/B/E multiseed deltas versus raw A:

| trial   |   seed |   delta_balanced_accuracy_vs_A |   delta_macro_f1_vs_A |   delta_auc_vs_A |
|:--------|-------:|-------------------------------:|----------------------:|-----------------:|
| B       |      7 |                    0.00354512  |           -0.0029283  |     -0.00177725  |
| B       |     42 |                    0.00494522  |            0.00533351 |      0.0114326   |
| B       |    123 |                   -0.00359353  |           -0.0070361  |     -0.000199419 |
| E       |      7 |                    0.000793635 |           -0.00574786 |     -0.00357641  |
| E       |     42 |                    0.00649453  |            0.00591693 |      0.0120512   |
| E       |    123 |                   -0.00354354  |           -0.0073782  |     -0.000172312 |

The full A–H factorial is evaluated at seed 42, while A/B/E have seeds 7, 42, and 123. Band-pass and notch deltas change sign across seeds; they do not show a stable advantage over raw input. CAR is consistently negative within the seed-42 matched factorial contrasts.

## 8. Transformer AutoML pilot

| comparison                                   | family       | track        | left_model        | right_model          |   seed | metric            | budget_seconds   | inferential   | status   |   n_subjects |   nonzero_differences |   mean_difference |   median_difference |     ci_low |     ci_high |   probability_difference_gt_zero |   subjects_improved |   subjects_degraded |   ties |   fraction_improved |   fraction_degraded |   number_needed_to_improve | wilcoxon_status          |   wilcoxon_statistic | wilcoxon_p_value   | sign_test_status         | sign_test_p_value   |   rank_biserial | holm_adjusted_p_value   |
|:---------------------------------------------|:-------------|:-------------|:------------------|:---------------------|-------:|:------------------|:-----------------|:--------------|:---------|-------------:|----------------------:|------------------:|--------------------:|-----------:|------------:|---------------------------------:|--------------------:|--------------------:|-------:|--------------------:|--------------------:|---------------------------:|:-------------------------|---------------------:|:-------------------|:-------------------------|:--------------------|----------------:|:------------------------|
| tuned_minus_baseline_transformer_outer_fold1 | automl_pilot | automl_pilot | tuned_transformer | baseline_transformer |     42 | balanced_accuracy |                  | False         | ok       |           11 |                    11 |        -0.0335926 |          -0.038215  | -0.0513401 | -0.0146393  |                           0.0007 |                   1 |                  10 |      0 |           0.0909091 |            0.909091 |                   11       | not_run_pilot_case_study |                    5 |                    | not_run_pilot_case_study |                     |       -0.848485 |                         |
| tuned_minus_baseline_transformer_outer_fold1 | automl_pilot | automl_pilot | tuned_transformer | baseline_transformer |     42 | macro_f1          |                  | False         | ok       |           11 |                    11 |        -0.0181423 |          -0.0289572 | -0.0431285 |  0.00931242 |                           0.0895 |                   3 |                   8 |      0 |           0.272727  |            0.727273 |                    3.66667 | not_run_pilot_case_study |                   16 |                    | not_run_pilot_case_study |                     |       -0.515152 |                         |
| tuned_minus_baseline_transformer_outer_fold1 | automl_pilot | automl_pilot | tuned_transformer | baseline_transformer |     42 | auc               |                  | False         | ok       |           11 |                    11 |        -0.0201142 |          -0.0167591 | -0.0413879 |  0.00158602 |                           0.0349 |                   2 |                   9 |      0 |           0.181818  |            0.818182 |                    5.5     | not_run_pilot_case_study |                   15 |                    | not_run_pilot_case_study |                     |       -0.545455 |                         |

This is an outer-fold-1 case study only. It does not support a general claim about AutoML, and no p-value is reported for it.

## 9. User calibration

| calibration_method    |   budget_seconds |   subjects |   delta_balanced_accuracy_mean |   delta_macro_f1_mean |   fraction_improved |   fraction_degraded |
|:----------------------|-----------------:|-----------:|-------------------------------:|----------------------:|--------------------:|--------------------:|
| full_model            |              180 |         53 |                     0.00155651 |            0.00430998 |            0.45283  |            0.528302 |
| full_model            |              300 |         53 |                     0.00422703 |            0.00859423 |            0.584906 |            0.415094 |
| full_model            |              600 |         53 |                     0.00425732 |            0.00900231 |            0.641509 |            0.358491 |
| head_only             |              180 |         53 |                     0.00507361 |            0.00632683 |            0.641509 |            0.358491 |
| head_only             |              300 |         53 |                     0.00782603 |            0.0108902  |            0.603774 |            0.396226 |
| head_only             |              600 |         53 |                     0.00741206 |            0.0109374  |            0.641509 |            0.358491 |
| subject_normalization |              180 |         53 |                    -0.072877   |           -0.100797   |            0.150943 |            0.849057 |
| subject_normalization |              300 |         53 |                    -0.0687598  |           -0.0936721  |            0.207547 |            0.792453 |
| subject_normalization |              600 |         53 |                    -0.0750587  |           -0.0959628  |            0.207547 |            0.792453 |

Every calibrated method is compared with zero-shot predictions from the same subject, budget, and evaluation tail. Calibration inputs are excluded from evaluation metrics.

## 10. Limitations

The cohort contains 54 subjects, subject AUC is undefined when required classes are absent, recurrent and Transformer sequence definitions differ, the AutoML result covers one outer fold, and calibration effects are heterogeneous. No external-dataset generalization is evaluated here.

## 11. Main defensible claims

- GroupKFold performance is above the five-class chance reference for the main completed models, but varies materially across subjects.
- Raw-model differences are assessed only on exact deduplicated windows.
- CAR has a negative seed-42 factorial main effect for ShallowConvNet; band-pass/notch stability must be judged across the available seeds.
- Head-only calibration has a small, heterogeneous matched effect, while short-interval subject normalization is harmful on average.
- The outer-fold-1 AutoML pilot did not improve the baseline Transformer.

## 12. Claims that cannot yet be made

- Transformer cannot be declared superior to LSTM/BiLSTM from a paired test because sequence identities differ.
- The AutoML pilot cannot be generalized beyond outer fold 1.
- A preprocessing choice cannot be called universally optimal from this single architecture and limited seed set.
- No clinical, causal, or external-population claim is supported.