# COG-BCI contrastive EEGNet transfer screening

- Branch: `integration/benchmark-unification`.
- Source HEAD: `22440253ddd3d3f674146ca437f2404d4ac35d2b`.
- Result status: `diagnostic`.
- Decision: `do_not_proceed`.

## Input and pretraining contract

The immutable `emotiv_common` raw cache contributed 56,903 accepted windows from 1,044 records, 29 subjects and 3 sessions.
All task families were used without labels: flanker, matb, n_back, pvt, resting_state.
Input is `[B, 1, 14, 2560]`, float32, from the existing raw cache. No band-pass, notch, demean, CAR, or resampling was added.
Subject-disjoint pretraining split: 24 train / 5 validation subjects; 47,123 / 9,780 windows; split hash `d1cc6a469ae75126c76889cc49a736273cddbfa9cea00a590799971016fa0bf9`.

The fixed augmentation order was Gaussian noise, amplitude scaling, time masking, channel masking, and temporal shift. The existing ProjectionHead and normalized in-batch NT-Xent objective were used; checkpoint selection used only pretraining-validation contrastive loss.

## Pretraining result

EEGNet latent width: 1280; encoder parameters: 2,096; projection parameters: 180,480.
Training completed 30 epochs in 2019.9 s; best epoch 27, validation NT-Xent 5.911507.
Collapse audit: fatal=False, reasons=[].
Encoder checkpoint SHA-256: `1c5aa561630ac3da787701aaa6589dc34b0ff6f052bd1d8e198e1744b4aebb4c`.

## Downstream fold-1 comparison

The canonical deduplicated raw `label_q5` dataset and its precomputed subject GroupKFold outer fold 1 were retained. All three modes used the same subject-disjoint inner split, training budget, preprocessing, seed, batching, metrics, and inner-validation macro-F1 checkpoint selection.

| Mode | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Epochs | Best val macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| random_init | 0.2975 | 0.2997 | 0.2702 | 0.2724 | 15 | 0.2539 |
| head_only | 0.2562 | 0.2505 | 0.2435 | 0.2486 | 8 | 0.2122 |
| full_model | 0.2760 | 0.2760 | 0.2692 | 0.2711 | 13 | 0.2585 |

Confusion matrices and per-class recall are preserved in the runtime JSON and mode metrics. Subject-level window metrics are also saved.

## Leakage and checkpoint audit

Leakage safe: True; outer train/test subject and sample overlap, inner train/validation subject, record and sample overlap are all zero. The outer test was not used for pretraining, augmentation, epoch, or mode selection.
Checkpoint valid: True. The projection head was not transferred; downstream heads were new five-output heads.

## Decision

The preregistered deterministic screening rule returned `do_not_proceed`. This is not a statistical-significance claim.

## Limitations and next step

This is one outer fold and one seed. COG-BCI and the project share channel order and sample count but not sampling rate (500 versus 256 Hz); the EEGNet kernel shape is fixed to the downstream 256-Hz architecture for strict encoder transfer. No full five-fold or multi-seed experiment is justified unless the preregistered screening threshold is met.
