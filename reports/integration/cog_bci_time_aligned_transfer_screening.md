# COG-BCI time-aligned contrastive transfer screening

Status: `diagnostic`.

## Repository and temporal contract

- Branch: `integration/benchmark-unification`.
- Audited HEAD: `57ac7d3cc036d04663a4471f742d225f8c96d14d`.
- Project contract: 256 Hz, 10.0 s, 2,560 samples, 14 channels, `PROJECT_EMOTIV_CHANNEL_ORDER`.
- The previous screening was shape-compatible but used COG-BCI at 500 Hz for 5.12 s; it was not physically time-aligned.

## Resampling and cache

- Whole records were selected to `emotiv_common`, loaded once, resampled by explicit polyphase ratio 64/125, then windowed.
- Explicit anti-alias FIR: 2,501 taps, Kaiser beta 5.0, normalized cutoff 1/125, constant zero padding.
- No demean, experimental band-pass, notch, CAR, rereference, Cz interpolation, or per-window resampling was applied.
- Cache config hash: `64dc6c82e1a780bd9c832c0abbcc04e52a5b40889ae3630f586f39589436235e`.
- Resampling hash: `8d14d5defb844e7affd6b3d2ef48084cfcfa536d1c94fdd338b7ed2ae8138d09`.
- Accepted windows: 28,910; rejected tails: 917; size: 3.887 GiB.
- Windows by family: `{"flanker": 5221, "matb": 7569, "n_back": 8601, "pvt": 5490, "resting_state": 2029}`.
- Duration absolute error maximum: 0.003875000 s.
- Event timing absolute error maximum: 0.001937500 s; metadata mismatches: 0.
- QC: 1,044 records, 29 subjects, 3 sessions, 14 channels, 2,560 samples, no ECG1, NaN, Inf, duplicate ID, invalid bound, or record crossing.

## Contrastive pretraining

- Subject assignment is unchanged from the shape-only run: 24 train / 5 validation; validation subjects: `sub-08, sub-17, sub-19, sub-24, sub-25`.
- New combined split hash: `c02613f05a85a99ea76cea7b6c100ac582e5eb1cea783df84c82f73227cb405b`.
- Epochs: 30; best epoch: 29; time: 982.0 s.
- Best validation NT-Xent: 5.605642; collapse: False.
- Shape-only encoder: `1c5aa561630ac3da787701aaa6589dc34b0ff6f052bd1d8e198e1744b4aebb4c`.
- Time-aligned encoder: `bfcfe47a9764884717cd7ba076e1205c97f49453a24299f35d69660becb3d495`.
- Encoder / projection parameters: 2,096 / 180,480; latent dimension: 1280.

| Pretraining | Train NT-Xent | Validation NT-Xent | Train-val gap | Validation effective rank | Positive-negative gap | Feature std |
|---|---:|---:|---:|---:|---:|---:|
| shape_only | 1.486529 | 5.911507 | 4.424979 | 15.605995 | 0.187998 | 0.084163 |
| time_aligned | 1.539139 | 5.605642 | 4.066503 | 14.881308 | 0.287489 | 0.083446 |

Time alignment reduced validation NT-Xent and the train-validation loss gap, and increased the positive-negative similarity gap. Effective rank decreased slightly; neither run met the collapse criteria.

## Fold-2 controlled downstream screening

| Mode | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Epochs | Best val macro F1 | Best val loss | Train s | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_init | 0.2770 | 0.2757 | 0.2705 | 0.2728 | 15 | 0.2249 | 1.6302 | 184.9 | 1.5448 |
| shape_only | 0.2532 | 0.2530 | 0.2504 | 0.2526 | 7 | 0.2277 | 1.6071 | 86.7 | 1.5669 |
| time_aligned | 0.2699 | 0.2677 | 0.2685 | 0.2719 | 12 | 0.2288 | 1.6439 | 147.5 | 1.5306 |

Fold 2 contains 6,192 test windows from 11 subjects. Inner train/validation and outer test subjects are pairwise disjoint.

Per-class recall:

| Mode | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 |
|---|---:|---:|---:|---:|---:|
| random_init | 0.2373 | 0.3045 | 0.3539 | 0.1048 | 0.3780 |
| shape_only | 0.3732 | 0.2448 | 0.2474 | 0.1152 | 0.2842 |
| time_aligned | 0.3126 | 0.1998 | 0.3263 | 0.1410 | 0.3586 |

All three modes used the same fold 2, subject-disjoint inner validation, seed, optimizer, budget, train-only channel standardization, checkpoint criterion, and test objects. Projection heads were not loaded downstream.

## Integrity and decision

- Checkpoint audit: True.
- Leakage audit: True.
- Outer train/test subject overlap, inner train/validation subject/record/sample overlap, and pretraining train/validation subject overlap are all zero.
- Unified predictions contain the same 6,192 sample, subject, record, target and fold identities in every mode; probabilities are finite and sum to one.
- Protected input hashes (old COG cache, old encoder checkpoint, project raw manifest and split inputs) are unchanged.
- Preregistration SHA-256: `59451c9d3c5b3d24d912f6e7f30cc6913477c1c5b42ea16a669e1ba90a982671`.
- Decision: `close_transfer_track`.
- Metric deltas: `{"time_aligned_minus_random_balanced_accuracy": -0.008026758197976214, "time_aligned_minus_random_macro_f1": -0.001943832635148135, "time_aligned_minus_shape_only_balanced_accuracy": 0.014715442184264849, "time_aligned_minus_shape_only_macro_f1": 0.018097501417228712}`.

This is a second sequential one-fold, one-seed diagnostic screening, not a pooled estimate with fold 1 and not a statistical significance test.

The COG-BCI contrastive-transfer track should be closed for the current project; no new augmentation or architecture search is recommended.
