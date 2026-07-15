# Raw EEG CNN model comparison

## Protocol

EEGNet and ShallowConvNet were evaluated on the same 30,958 accepted,
deduplicated raw EEG windows with target `label_q5`. Both models use the same
precomputed five outer folds grouped by `subject_id`, the same selected logical
recordings, raw preprocessing, train-only channel normalization and grouped
inner validation by `record_group_id`. Seeds 42, 7 and 123 were run for both
models. Every model/seed run contains exactly one prediction per accepted sample,
and all `(sample_id, fold, y_true)` rows are identical across the six runs.

Integrity checks for every run found:

- five completed outer folds and all 54 subjects represented in test once;
- 30,958 unique unified predictions and zero duplicate `sample_id` values;
- zero outer train/test subject overlap;
- zero inner train/validation logical-record overlap;
- finite `proba_0`–`proba_4` with row sums within `1e-6` of one;
- complete model, metrics, log, prediction, normalization, preprocessing,
  logical-selection and rejected-window artifacts for every fold.

## Seed 42 comparison

Values are mean ± population standard deviation across the same five folds.

| Model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Kappa | AUC | Epochs | Best validation loss | Training time | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EEGNet | 0.259637 ± 0.015150 | 0.258578 ± 0.018799 | 0.228363 ± 0.023192 | 0.227848 ± 0.024191 | 0.072336 ± 0.024740 | 0.585061 ± 0.019674 | 9.0 ± 2.280 | 1.564113 ± 0.022517 | 653.057 s | 8,501 |
| ShallowConvNet | **0.279308 ± 0.016372** | **0.282386 ± 0.016985** | **0.259939 ± 0.015034** | **0.259900 ± 0.014072** | **0.100963 ± 0.018949** | **0.603077 ± 0.017154** | 10.6 ± 3.262 | 1.541270 ± 0.058268 | 713.475 s | 1,925 |

## Seed 42 fold-level results

| Model | Fold | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Kappa | AUC | Epochs | Best epoch | Best validation loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EEGNet | 1 | 0.277738 | 0.284423 | 0.223801 | 0.218885 | 0.103806 | 0.607137 | 6 | 2 | 1.590619 |
| EEGNet | 2 | 0.277132 | 0.276959 | 0.266504 | 0.268530 | 0.098433 | 0.604798 | 9 | 5 | 1.576492 |
| EEGNet | 3 | 0.251781 | 0.252133 | 0.238408 | 0.239707 | 0.066635 | 0.570879 | 12 | 8 | 1.526380 |
| EEGNet | 4 | 0.239958 | 0.237967 | 0.215537 | 0.213144 | 0.044012 | 0.586665 | 11 | 7 | 1.552248 |
| EEGNet | 5 | 0.251578 | 0.241407 | 0.197567 | 0.198972 | 0.048794 | 0.555826 | 7 | 3 | 1.574824 |
| ShallowConvNet | 1 | 0.279036 | 0.278331 | 0.263675 | 0.268199 | 0.100181 | 0.596458 | 6 | 2 | 1.646105 |
| ShallowConvNet | 2 | 0.295543 | 0.292509 | 0.267596 | 0.270846 | 0.118490 | 0.625883 | 13 | 9 | 1.494395 |
| ShallowConvNet | 3 | 0.250787 | 0.252852 | 0.230897 | 0.232796 | 0.067996 | 0.588405 | 8 | 4 | 1.496508 |
| ShallowConvNet | 4 | 0.295187 | 0.303500 | 0.274162 | 0.268069 | 0.120675 | 0.620905 | 11 | 7 | 1.563805 |
| ShallowConvNet | 5 | 0.275988 | 0.284739 | 0.263363 | 0.259592 | 0.097473 | 0.583733 | 15 | 13 | 1.505536 |

## Seed 42 paired fold differences

Differences are `ShallowConvNet − EEGNet` on identical test samples.

| Fold | Accuracy Δ | Balanced accuracy Δ | Macro F1 Δ | Weighted F1 Δ | Kappa Δ | AUC Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.001299 | −0.006092 | +0.039875 | +0.049314 | −0.003625 | −0.010679 |
| 2 | +0.018411 | +0.015550 | +0.001092 | +0.002316 | +0.020058 | +0.021085 |
| 3 | −0.000994 | +0.000720 | −0.007511 | −0.006911 | +0.001361 | +0.017526 |
| 4 | +0.055229 | +0.065533 | +0.058624 | +0.054925 | +0.076663 | +0.034240 |
| 5 | +0.024410 | +0.043332 | +0.065795 | +0.060620 | +0.048679 | +0.027906 |

ShallowConvNet is higher on seed-42 accuracy in 4/5 folds, balanced accuracy in
4/5 folds and macro F1 in 4/5 folds. These paired descriptive differences were
not subjected to an inferential significance test.

## Seed-level means

| Model | Seed | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Kappa | AUC | Mean epochs | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EEGNet | 7 | 0.244684 | 0.246108 | 0.226756 | 0.226645 | 0.056544 | 0.565488 | 7.8 | 563.386 s |
| EEGNet | 42 | 0.259637 | 0.258578 | 0.228363 | 0.227848 | 0.072336 | 0.585061 | 9.0 | 653.057 s |
| EEGNet | 123 | 0.251364 | 0.252938 | 0.215774 | 0.214522 | 0.064913 | 0.575225 | 7.4 | 560.111 s |
| ShallowConvNet | 7 | 0.284179 | 0.285366 | 0.265779 | 0.265774 | 0.105865 | 0.612997 | 14.0 | 952.988 s |
| ShallowConvNet | 42 | 0.279308 | 0.282386 | 0.259939 | 0.259900 | 0.100963 | 0.603077 | 10.6 | 713.475 s |
| ShallowConvNet | 123 | 0.284114 | 0.283838 | 0.268422 | 0.269133 | 0.104172 | 0.597974 | 10.8 | 759.132 s |

## Across-seed summary

`Seed std` is the population standard deviation of the three seed-level fold
means. `Fold/run std` is the population standard deviation across all 15
seed-fold observations.

| Model | Metric | Mean across folds/seeds | Seed std | Fold/run std |
|---|---|---:|---:|---:|
| EEGNet | Accuracy | 0.251895 | 0.006116 | 0.022980 |
| EEGNet | Balanced accuracy | 0.252541 | 0.005099 | 0.022694 |
| EEGNet | Macro F1 | 0.223631 | 0.005595 | 0.027209 |
| EEGNet | Weighted F1 | 0.223005 | 0.006018 | 0.028363 |
| EEGNet | Kappa | 0.064598 | 0.006451 | 0.029262 |
| EEGNet | AUC | 0.575258 | 0.007990 | 0.027135 |
| ShallowConvNet | Accuracy | **0.282534** | **0.002281** | **0.013387** |
| ShallowConvNet | Balanced accuracy | **0.283864** | **0.001217** | **0.013857** |
| ShallowConvNet | Macro F1 | **0.264713** | **0.003545** | **0.013236** |
| ShallowConvNet | Weighted F1 | **0.264936** | **0.003816** | **0.014993** |
| ShallowConvNet | Kappa | **0.103667** | **0.002033** | **0.015914** |
| ShallowConvNet | AUC | **0.604683** | **0.006237** | **0.018281** |

Across three seeds, ShallowConvNet exceeds EEGNet by 0.030639 accuracy,
0.031322 balanced accuracy, 0.041082 macro F1 and 0.041931 weighted F1 on the
mean of the 15 matched seed-fold observations. It also has lower between-seed
and fold/run variability for accuracy, balanced accuracy and macro F1.

## Artifacts

- EEGNet seed 42: `benchmark_results/groupkfold_torch_eegnet_raw_dedup_label_q5/20260715_082819`
- EEGNet seed 7: `benchmark_results/groupkfold_torch_eegnet_raw_dedup_label_q5_seed7/20260715_095859`
- EEGNet seed 123: `benchmark_results/groupkfold_torch_eegnet_raw_dedup_label_q5_seed123/20260715_100852`
- ShallowConvNet seed 42: `benchmark_results/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed42/20260715_091552`
- ShallowConvNet seed 7: `benchmark_results/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed7/20260715_092857`
- ShallowConvNet seed 123: `benchmark_results/groupkfold_torch_shallow_convnet_raw_dedup_label_q5_seed123/20260715_094513`

The companion `reports/raw_eeg_cnn_model_comparison.csv` contains all 30
model/seed/fold metric rows, training histories and parameter counts.

## Short conclusion for the meeting

The second raw-EEG architecture, `torch_shallow_convnet`, is integrated through
the shared factory and adapter. Full five-fold subject-grouped evaluation was
completed for seeds 42, 7 and 123, with no subject or logical-record leakage.
ShallowConvNet outperformed EEGNet on the mean accuracy, balanced accuracy and
macro F1 for all three seeds. Its accuracy stayed near 0.28 with lower
between-seed and fold variability than EEGNet, and both models remained above
the five-class random accuracy level of 0.20. ShallowConvNet achieved this with
1,925 parameters versus EEGNet's 8,501, although its total training time was
longer because it generally trained for more epochs. After the meeting, the next
step is to review class-wise errors and only then register any architecture or
training changes as a new controlled experiment.

No statistical-significance claim is made from these descriptive fold/seed
comparisons.
