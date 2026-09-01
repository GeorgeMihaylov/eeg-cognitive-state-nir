# Transformer benchmark report

## Legacy source and migration

The migrated architecture comes from `origin/latent-state-alignment`, commit `0c6aea0` (`transformer_add`), file `src/42_train_transformer_latent_trajectory_model.py`. The reusable scientific idea was a feature-sequence Transformer over EEG+POW windows with learned positions, a pre-norm TransformerEncoder and last-token pooling. The regression target/head, private DataLoader/training loop, random split, gap-unaware sequence builder, custom CLI and custom artifact layout were rejected.

The current `torch_transformer` is a pure `torch.nn.Module`. It is constructed by `model_zoo/factory.py`, trained by the existing `TorchClassificationAdapter`, receives gap-aware sequences from `model_zoo/DL/sequence_utils.py`, and is evaluated/serialized by the canonical `BenchmarkRunner`.

## Current representation and architecture

```text
EEG + POW feature windows [batch, 8, 448]
  -> Linear(448, 128)
  -> learned positional encoding
  -> 2 x pre-norm TransformerEncoderLayer
       nhead=4, dim_feedforward=256, GELU, dropout=0.1
  -> last valid token
  -> LayerNorm + MLP classification head
  -> 5 label_q5 logits
```

Sequences never cross `source`, `subject_id`, `record_id` or a time gap above 10.5 seconds. The module supports boolean padding masks and last-valid, masked-mean and CLS pooling; the scientific default is last-valid to preserve the legacy last-window target idea. Padded tokens are excluded from attention keys and pooling. The benchmark currently produces fixed-length, unpadded sequences.

Trainable parameters: **340,869**. At batch 128 the nominal attention-score tensor is 32,768 FP32 elements (about 128 KiB) per layer. Smoke peak allocated GPU memory was 46,653,440 bytes (about 44.5 MiB) on an NVIDIA GeForce RTX 5060 Ti.

## Technical smoke

The canonical CLI smoke used one fold, 1,000 balanced windows, at most three epochs and seed 42. It automatically resolved input shape `[8, 448]`, trained on CUDA and wrote the standard manifest, config, model, training log, validation split and predictions.

- accuracy: 0.5403;
- balanced accuracy: 0.3875;
- macro F1: 0.3867;
- epochs: 3 (best epoch 2);
- best validation loss: 1.3595;
- mean epoch time: 0.166 s including CUDA warm-up;
- config hash: `e074510d4b4049b653e4b794cc4fa383bd668c96985901f745c30c5aedb8e7db`.

These smoke metrics are not a scientific estimate.

## Seed 42 comparison

All models use the same five outer subject-fold assignments. RF/MLP operate on individual windows; LSTM/BiLSTM use gap-aware sequences of length 10; Transformer uses gap-aware sequences of length 8. Consequently the outer protocol is comparable, but sequence-level predictions are not one-to-one across every model.

| Model | Parameters | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Kappa | AUC | Epochs | Training sec |
|---|---|---|---|---|---|---|---|---|---|
| random_forest | n/a | 0.3021 +/- 0.0241 | 0.3059 +/- 0.0255 | 0.2955 +/- 0.0217 | 0.2953 +/- 0.0226 | 0.1297 +/- 0.0310 | 0.6219 +/- 0.0257 | n/a | 9.8 |
| torch_mlp | 148,485 | 0.2786 +/- 0.0147 | 0.2822 +/- 0.0168 | 0.2740 +/- 0.0126 | 0.2736 +/- 0.0115 | 0.1005 +/- 0.0179 | 0.5956 +/- 0.0216 | 15.0 +/- 0.0 | 27.9 |
| torch_lstm_gapaware | 304,517 | 0.3673 +/- 0.0182 | 0.3697 +/- 0.0239 | 0.3555 +/- 0.0273 | 0.3566 +/- 0.0241 | 0.2095 +/- 0.0242 | 0.7072 +/- 0.0162 | 6.8 +/- 1.7 | 24.2 |
| torch_bilstm_gapaware | 608,645 | 0.3653 +/- 0.0130 | 0.3681 +/- 0.0194 | 0.3570 +/- 0.0168 | 0.3572 +/- 0.0122 | 0.2069 +/- 0.0171 | 0.7091 +/- 0.0125 | 6.2 +/- 1.0 | 26.2 |
| torch_transformer | 340,869 | 0.3664 +/- 0.0132 | 0.3687 +/- 0.0189 | 0.3615 +/- 0.0169 | 0.3620 +/- 0.0101 | 0.2080 +/- 0.0178 | 0.7036 +/- 0.0145 | 11.2 +/- 2.9 | 86.8 |

Transformer is within 0.0011 balanced accuracy of LSTM and 0.0006 above BiLSTM. It has the best sequence-model macro F1 and weighted F1, with balanced-accuracy fold variability slightly below both LSTM baselines. Its AUC is slightly lower and its training time is higher. No statistical-significance claim is made.

| Comparison | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Kappa | AUC |
|---|---|---|---|---|---|---|
| Transformer - LSTM | -0.0010 | -0.0010 | +0.0060 | +0.0053 | -0.0014 | -0.0036 |
| Transformer - BiLSTM | +0.0011 | +0.0006 | +0.0044 | +0.0047 | +0.0011 | -0.0055 |

## Fold-level seed 42 results

| Model | Fold | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Kappa | AUC | Epochs | Training sec |
|---|---|---|---|---|---|---|---|---|---|
| random_forest | fold_01 | 0.3443 | 0.3497 | 0.3315 | 0.3324 | 0.1845 | 0.6608 | n/a | 2.0 |
| random_forest | fold_02 | 0.3105 | 0.3069 | 0.3046 | 0.3103 | 0.1390 | 0.6438 | n/a | 2.0 |
| random_forest | fold_03 | 0.2751 | 0.2714 | 0.2671 | 0.2726 | 0.0940 | 0.5950 | n/a | 2.0 |
| random_forest | fold_04 | 0.2945 | 0.3080 | 0.2913 | 0.2821 | 0.1208 | 0.6085 | n/a | 1.9 |
| random_forest | fold_05 | 0.2860 | 0.2937 | 0.2827 | 0.2792 | 0.1104 | 0.6015 | n/a | 1.9 |
| torch_bilstm_gapaware | fold_01 | 0.3609 | 0.3711 | 0.3557 | 0.3529 | 0.2076 | 0.7139 | 6 | 5.6 |
| torch_bilstm_gapaware | fold_02 | 0.3763 | 0.3710 | 0.3623 | 0.3722 | 0.2218 | 0.7208 | 8 | 6.1 |
| torch_bilstm_gapaware | fold_03 | 0.3417 | 0.3312 | 0.3263 | 0.3363 | 0.1740 | 0.6851 | 5 | 4.6 |
| torch_bilstm_gapaware | fold_04 | 0.3751 | 0.3811 | 0.3644 | 0.3599 | 0.2151 | 0.7162 | 6 | 5.0 |
| torch_bilstm_gapaware | fold_05 | 0.3723 | 0.3863 | 0.3765 | 0.3649 | 0.2162 | 0.7093 | 6 | 4.9 |
| torch_lstm_gapaware | fold_01 | 0.3747 | 0.3777 | 0.3695 | 0.3730 | 0.2216 | 0.7178 | 7 | 5.7 |
| torch_lstm_gapaware | fold_02 | 0.3853 | 0.3806 | 0.3729 | 0.3821 | 0.2332 | 0.7162 | 10 | 6.2 |
| torch_lstm_gapaware | fold_03 | 0.3327 | 0.3222 | 0.3012 | 0.3127 | 0.1631 | 0.6831 | 6 | 4.4 |
| torch_lstm_gapaware | fold_04 | 0.3762 | 0.3829 | 0.3696 | 0.3636 | 0.2159 | 0.7256 | 6 | 4.1 |
| torch_lstm_gapaware | fold_05 | 0.3678 | 0.3853 | 0.3641 | 0.3518 | 0.2136 | 0.6931 | 5 | 3.8 |
| torch_mlp | fold_01 | 0.2586 | 0.2622 | 0.2568 | 0.2579 | 0.0784 | 0.5648 | 15 | 6.6 |
| torch_mlp | fold_02 | 0.2970 | 0.2971 | 0.2875 | 0.2907 | 0.1236 | 0.6324 | 15 | 5.3 |
| torch_mlp | fold_03 | 0.2666 | 0.2619 | 0.2632 | 0.2686 | 0.0818 | 0.5943 | 15 | 5.2 |
| torch_mlp | fold_04 | 0.2927 | 0.3002 | 0.2883 | 0.2820 | 0.1152 | 0.5957 | 15 | 5.3 |
| torch_mlp | fold_05 | 0.2781 | 0.2897 | 0.2742 | 0.2688 | 0.1033 | 0.5907 | 15 | 5.5 |
| torch_transformer | fold_01 | 0.3707 | 0.3834 | 0.3687 | 0.3651 | 0.2211 | 0.7212 | 15 | 23.4 |
| torch_transformer | fold_02 | 0.3753 | 0.3725 | 0.3506 | 0.3595 | 0.2221 | 0.7079 | 9 | 14.0 |
| torch_transformer | fold_03 | 0.3413 | 0.3342 | 0.3359 | 0.3456 | 0.1769 | 0.6773 | 12 | 18.5 |
| torch_transformer | fold_04 | 0.3661 | 0.3660 | 0.3666 | 0.3628 | 0.1992 | 0.7023 | 7 | 11.3 |
| torch_transformer | fold_05 | 0.3785 | 0.3874 | 0.3856 | 0.3769 | 0.2209 | 0.7092 | 13 | 19.7 |

## Additional Transformer seeds

The predeclared condition for additional seeds was met at seed 42: comparable balanced accuracy, better macro F1 than the relevant sequence baselines and lower fold variability than LSTM. Seeds 7 and 123 therefore used the same config and folds, changing only random state.

| Seed | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Kappa | AUC | Epochs | Training sec |
|---|---|---|---|---|---|---|---|---|
| 7 | 0.3625 +/- 0.0189 | 0.3648 +/- 0.0231 | 0.3575 +/- 0.0202 | 0.3578 +/- 0.0181 | 0.2031 +/- 0.0247 | 0.6967 +/- 0.0197 | 9.2 +/- 3.2 | 73.1 |
| 42 | 0.3664 +/- 0.0132 | 0.3687 +/- 0.0189 | 0.3615 +/- 0.0169 | 0.3620 +/- 0.0101 | 0.2080 +/- 0.0178 | 0.7036 +/- 0.0145 | 11.2 +/- 2.9 | 86.8 |
| 123 | 0.3600 +/- 0.0198 | 0.3629 +/- 0.0279 | 0.3515 +/- 0.0342 | 0.3512 +/- 0.0274 | 0.2004 +/- 0.0256 | 0.7032 +/- 0.0172 | 7.6 +/- 2.1 | 61.8 |

Across the three seed-level means:

| Metric | Mean across seeds | Between-seed SD |
|---|---|---|
| accuracy | 0.3629 | 0.0032 |
| balanced_accuracy | 0.3655 | 0.0030 |
| macro_f1 | 0.3568 | 0.0050 |
| weighted_f1 | 0.3570 | 0.0054 |
| kappa | 0.2038 | 0.0039 |
| auc | 0.7012 | 0.0038 |

The Transformer remains above random chance and close to LSTM/BiLSTM across all seeds. Seed-to-seed movement is modest for accuracy and balanced accuracy but larger for macro F1/fold variability at seed 123. The result is scientifically interesting as an alternative temporal encoder, but it is not a clear overall winner over recurrent baselines.

## Validation

For every Transformer seed:

- five folds and all 54 test subjects, each subject in test exactly once;
- 44,142 unique sequence and target sample predictions;
- finite `proba_0` through `proba_4`, row sums approximately one;
- zero outer subject overlap;
- zero inner `record_group_id` overlap;
- identical subject-fold assignments to RF, MLP, LSTM and BiLSTM;
- standard `model.pt`, `training_log.csv`, `validation_split.json`, fold predictions and unified predictions.

## Artifact directories

| Run | Seed | Standard benchmark run directory |
|---|---|---|
| smoke | 42 | benchmark_results/smoke_torch_transformer_label_q5/20260716_191010 |
| full | 7 | benchmark_results/groupkfold_torch_transformer_label_q5_seed7/20260716_191618 |
| full | 42 | benchmark_results/groupkfold_torch_transformer_label_q5/20260716_191246 |
| full | 123 | benchmark_results/groupkfold_torch_transformer_label_q5_seed123/20260716_191837 |

The machine-readable fold table is `reports/transformer_benchmark_folds.csv`.
