# User calibration report

Base run: `F:\EEG\benchmark_results\groupkfold_torch_transformer_label_q5\20260716_191246`
Base config hash: `ea4dbe39293c14d7c171901c46b53a3a9aa2edb6825c1b70cae379cafa220416`
Calibration run: `benchmark_results\user_calibration_transformer_label_q5\20260716_194723`
Full-model supplementary run: `benchmark_results\user_calibration_transformer_full_model_label_q5\20260716_195204`
Elapsed seconds: 283.0

Chronological splits are created from original windows before sequence
building. Seven windows are purged at an intra-record boundary; record
and >10.5 second gap boundaries remain enforced by the canonical builder.
Evaluation data is never used for normalization or early stopping.

## Subject-level aggregate

| method                |   budget_seconds |   valid_subjects |   balanced_accuracy_mean |   balanced_accuracy_subject_sd |   macro_f1_mean |   macro_f1_subject_sd |   delta_balanced_accuracy_vs_zero_shot |   subjects_improved |   subjects_degraded |
|:----------------------|-----------------:|-----------------:|-------------------------:|-------------------------------:|----------------:|----------------------:|---------------------------------------:|--------------------:|--------------------:|
| full_model            |                0 |                0 |               nan        |                    nan         |      nan        |           nan         |                           nan          |                   0 |                   0 |
| full_model            |               60 |                0 |               nan        |                    nan         |      nan        |           nan         |                           nan          |                   0 |                   0 |
| full_model            |              180 |               53 |                 0.355434 |                      0.0797648 |        0.322053 |             0.0939409 |                             0.00155651 |                  24 |                  28 |
| full_model            |              300 |               53 |                 0.360444 |                      0.0793935 |        0.329179 |             0.0943915 |                             0.00422703 |                  31 |                  22 |
| full_model            |              600 |               53 |                 0.361071 |                      0.0809836 |        0.33109  |             0.0954344 |                             0.00425732 |                  34 |                  19 |
| head_only             |                0 |                0 |               nan        |                    nan         |      nan        |           nan         |                           nan          |                   0 |                   0 |
| head_only             |               60 |                0 |               nan        |                    nan         |      nan        |           nan         |                           nan          |                   0 |                   0 |
| head_only             |              180 |               53 |                 0.358951 |                      0.0773937 |        0.32407  |             0.0952632 |                             0.00507361 |                  34 |                  19 |
| head_only             |              300 |               53 |                 0.364043 |                      0.0778779 |        0.331475 |             0.0944529 |                             0.00782603 |                  32 |                  21 |
| head_only             |              600 |               53 |                 0.364226 |                      0.0777407 |        0.333025 |             0.0936418 |                             0.00741206 |                  34 |                  19 |
| subject_normalization |                0 |                0 |               nan        |                    nan         |      nan        |           nan         |                           nan          |                   0 |                   0 |
| subject_normalization |               60 |                0 |               nan        |                    nan         |      nan        |           nan         |                           nan          |                   0 |                   0 |
| subject_normalization |              180 |               53 |                 0.281001 |                      0.0721718 |        0.216946 |             0.0934635 |                            -0.072877   |                   8 |                  45 |
| subject_normalization |              300 |               53 |                 0.287457 |                      0.0704543 |        0.226913 |             0.0880199 |                            -0.0687598  |                  11 |                  42 |
| subject_normalization |              600 |               53 |                 0.281755 |                      0.0707738 |        0.226125 |             0.0774979 |                            -0.0750587  |                  11 |                  42 |
| zero_shot             |                0 |               53 |                 0.35143  |                      0.0781313 |        0.313471 |             0.0937191 |                             0          |                   0 |                   0 |
| zero_shot             |               60 |               53 |                 0.352867 |                      0.0782154 |        0.315764 |             0.0944911 |                             0          |                   0 |                   0 |
| zero_shot             |              180 |               53 |                 0.353878 |                      0.0790985 |        0.317743 |             0.09484   |                             0          |                   0 |                   0 |
| zero_shot             |              300 |               53 |                 0.356217 |                      0.0793446 |        0.320585 |             0.0949537 |                             0          |                   0 |                   0 |
| zero_shot             |              600 |               53 |                 0.356814 |                      0.0793489 |        0.322088 |             0.0946838 |                             0          |                   0 |                   0 |

## Data sufficiency

A 60-second prefix contains six 10-second windows and cannot form a
length-8 Transformer sequence. Normalization and fine-tuning therefore
receive `insufficient_sequence_context` at 60 seconds rather than
borrowing future windows. Fifty-three of 54 subjects are valid at the
longer budgets; the subject(s) below have fewer than 20 evaluation
sequences:

9192c107

## Calibration class coverage

|   budget_seconds |   number_of_classes |   subjects |   mean_delta_balanced_accuracy |
|-----------------:|--------------------:|-----------:|-------------------------------:|
|              180 |                   1 |          6 |                    0.00648033  |
|              180 |                   2 |         13 |                    0.00490211  |
|              180 |                   3 |         18 |                    0.00713335  |
|              180 |                   4 |         13 |                    0.00238791  |
|              180 |                   5 |          3 |                    0.00228293  |
|              300 |                   1 |          2 |                    0.0188559   |
|              300 |                   2 |          6 |                   -0.0020799   |
|              300 |                   3 |         12 |                   -0.000182661 |
|              300 |                   4 |         22 |                    0.00843173  |
|              300 |                   5 |         11 |                    0.0187492   |
|              600 |                   3 |          7 |                    0.00060721  |
|              600 |                   4 |         16 |                   -0.00352919  |
|              600 |                   5 |         30 |                    0.0148352   |

## Interpretation

Only 14 subjects had monotonically non-decreasing head-only balanced accuracy across 180/300/600 seconds. The calibration class-coverage/delta correlation was 0.129.

Full-model fine-tuning was run after the predeclared head-only criterion was met. It remained positive on average but was weaker than head-only at every valid budget, so head-only remains the preferred adaptation method.

All reported deltas use a matched zero-shot prediction on the same
budget-specific evaluation tail. No statistical-significance claim
is made.

Figures: `user_calibration_balanced_accuracy.svg`,
`user_calibration_macro_f1.svg`,
`user_calibration_subject_delta_heatmap.svg`, and
`user_calibration_class_coverage.svg`.
