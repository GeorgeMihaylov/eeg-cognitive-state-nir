# COG-BCI N-Back three-class diagnostic baseline

## 1. Repository state

- Branch: `integration/benchmark-unification`
- Starting HEAD: `925dfaa feat(data): add COG-BCI task and split protocols`
- Starting worktree and staging area: clean
- Result status: `diagnostic`
- Model seed: `42`

No commit, push, merge, rebase, reset, branch switch, or staging operation was
performed. Source EEG files, the window cache, task protocol, and split
manifests were treated as immutable inputs.

## 2. Input artifacts and provenance hashes

The run reused:

- `benchmark_results/cog_bci_windows/emotiv_common_full`
- `benchmark_results/cog_bci_protocols/nback_3class`

| Contract | SHA-256 |
|---|---|
| Window-cache configuration | `4f60a6c7cd9d0dd6613a9338834691d9ee289a749a08551045502aef4da80d72` |
| Task protocol | `5f7e01bc2dc2967737c6704819d7ef13ac2ba2919b1d4d9a46b36560cea63598` |
| Outer split | `5874a0a93bff6f8a504cbc75e15c48588bf12cd51f460eb0f9d16ff94809ac01` |
| Inner split | `d84f3853e244f2be47f6ad4431b533a1ec8d6e0b8d41687991d6b2fc32922ef9` |
| Channel mapping | `5735b5e22f3344b12bf0058b83da83bd00231905274a88ce1d221eeb09774a4d` |

The post-run SHA-256 of the cache window index was
`d9ec8adffb08e97b2de1e4eade9dc278691929c6869baa417b5b8042045535fa`,
matching the pre-run value. No cache build or target/split reconstruction was
performed.

## 3. Task

The task is categorical N-Back load classification:

| Task variant | Class |
|---|---:|
| `zero_back` | 0 |
| `one_back` | 1 |
| `two_back` | 2 |

The objective is `CrossEntropyLoss`. The ordinal relationship among classes is
used only in additional evaluation metrics. MATB-II windows are excluded.

## 4. Dataset and balance

The loader selected accepted, supervised N-Back windows from the existing
14-channel `emotiv_common` cache.

| Property | Value |
|---|---:|
| Accepted windows | 16,927 |
| Logical records | 261 |
| Subjects | 29 |
| Sessions | 3 |
| Input shape | `[1, 14, 2560]` |
| Sampling rate | 500 Hz |
| Window duration | 5.12 s |

| Class | Windows | Records | Subjects |
|---:|---:|---:|---:|
| 0 | 5,579 | 87 | 29 |
| 1 | 5,638 | 87 | 29 |
| 2 | 5,710 | 87 | 29 |

The channel order is the fixed 14-channel Emotiv contract. `ECG1` is absent.

## 5. Input-scale audit

The audit used the first 256 `sample_id`-sorted fold-1 inner-train windows and
did not inspect outer-test data.

| Statistic | Value |
|---|---:|
| dtype | `float32` |
| shape | `[256, 1, 14, 2560]` |
| minimum | -0.04122134 |
| maximum | 0.03707881 |
| mean | 0.00588079 |
| standard deviation | 0.00991955 |
| median absolute value | 0.00714043 |
| percentile 0.1 | -0.03804346 |
| percentile 1 | -0.02780181 |
| percentile 50 | 0.00593184 |
| percentile 99 | 0.02887710 |
| percentile 99.9 | 0.03541357 |

All values were finite. The cache is populated from `MNE Raw.get_data()`, whose
contract is an SI representation. The original physical-unit metadata is not
exposed by the reader, so units are not inferred from numerical magnitude
alone. The cache and baseline loader apply no hidden multiplication:
`scale_factor = 1.0`. The shared Torch adapter fits per-channel mean and scale
on each fold's inner-train subset only and applies that fixed transformation to
inner validation and outer test. Model input is therefore standardized and
dimensionless, with no double scaling.

## 6. Models

Both existing production raw-EEG models and the shared encoder contract were
reused.

| Model | Parameters | Input | Output |
|---|---:|---|---|
| EEGNet | 7,395 | `[B, 1, 14, 2560]` | `[B, 3]` |
| ShallowConvNet | 1,843 | `[B, 1, 14, 2560]` | `[B, 3]` |

`encode()`, `forward_head()`, and ordinary `forward()` retain their existing
roles. No new model implementation or private training loop was introduced.

## 7. Training configurations

Tracked diagnostic configurations:

- `experiments/cog_bci/nback_eegnet_baseline.json`
- `experiments/cog_bci/nback_shallowconvnet_baseline.json`

Shared settings:

| Setting | Value |
|---|---|
| seed | 42 |
| device | `auto` (resolved to CUDA) |
| maximum epochs | 30 |
| batch size | 128 |
| optimizer | AdamW |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| early-stopping patience | 8 |
| checkpoint criterion | inner-validation record macro F1, maximize |
| loss | CrossEntropyLoss |
| feature normalization | inner-train per-channel standardization |

No scheduler, class weights, sampling rebalance, label smoothing, focal loss,
or hyperparameter search was used.

## 8. Split protocol

The run consumed the exact manifests created by the preceding protocol task.

- Outer evaluation: five-fold subject-disjoint GroupKFold, no shuffle.
- Inner validation: subject-disjoint assignment inside outer train only.
- Outer-test observations were not used for normalization, early stopping,
  learning-rate selection, architecture selection, or checkpoint selection.
- Every model was rebuilt from a fresh factory instance for every fold.

## 9. Early stopping

The shared Torch adapter was extended backward-compatibly to accept an exact
validation index partition and to monitor record-level macro F1. For each
epoch, inner-validation probabilities are averaged within record before
computing the selection metric. The best state is restored before outer-test
inference.

| Model | Fold | Epochs trained | Best epoch | Best validation record macro F1 | Validation loss at best state |
|---|---:|---:|---:|---:|---:|
| EEGNet | 1 | 12 | 4 | 0.325758 | 1.113774 |
| EEGNet | 2 | 11 | 3 | 0.389075 | 1.101981 |
| EEGNet | 3 | 9 | 1 | 0.356790 | 1.103921 |
| EEGNet | 4 | 11 | 3 | 0.360386 | 1.097781 |
| EEGNet | 5 | 10 | 2 | 0.311247 | 1.100799 |
| ShallowConvNet | 1 | 30 | 24 | 0.438095 | 1.715947 |
| ShallowConvNet | 2 | 9 | 1 | 0.388889 | 1.178434 |
| ShallowConvNet | 3 | 11 | 3 | 0.414288 | 1.239925 |
| ShallowConvNet | 4 | 12 | 4 | 0.446649 | 1.171332 |
| ShallowConvNet | 5 | 10 | 2 | 0.389363 | 1.227286 |

ShallowConvNet fold 1 reached the configured 30-epoch ceiling; the other nine
fold-model combinations stopped earlier.

## 10. Checkpoint contract

Each fold checkpoint contains model and AdamW optimizer states, epoch log,
best monitor value, input shape, three-class contract, architecture metadata,
seed, fold, channel order, input-scale metadata, and the five provenance
hashes. Each checkpoint was loaded into a fresh factory-built model and
verified by matching probabilities. Outer-test metrics do not participate in
checkpoint selection.

## 11. Window-level results

Pooled out-of-fold metrics over 16,927 windows:

| Model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Macro precision | Macro recall |
|---|---:|---:|---:|---:|---:|---:|
| EEGNet | 0.356472 | 0.356299 | 0.356090 | 0.356199 | 0.356151 | 0.356299 |
| ShallowConvNet | 0.353341 | 0.352956 | 0.351928 | 0.352138 | 0.352796 | 0.352956 |

These window-level estimates are secondary because windows from a recording are
dependent and longer records contribute more observations.

## 12. Record-level results

Probabilities were averaged over all accepted windows within each record,
followed by `argmax`. Pooled metrics over 261 out-of-fold records:

| Model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Macro precision | Macro recall |
|---|---:|---:|---:|---:|---:|---:|
| EEGNet | 0.356322 | 0.356322 | 0.355945 | 0.355945 | 0.355991 | 0.356322 |
| ShallowConvNet | 0.356322 | 0.356322 | 0.355701 | 0.355701 | 0.356705 | 0.356322 |

Fold-level primary metrics:

| Model | Fold | Balanced accuracy | Macro F1 | Ordinal MAE | QWK |
|---|---:|---:|---:|---:|---:|
| EEGNet | 1 | 0.400000 | 0.398652 | 0.844444 | 0.062500 |
| EEGNet | 2 | 0.351852 | 0.328373 | 0.888889 | 0.075000 |
| EEGNet | 3 | 0.351852 | 0.327901 | 0.907407 | 0.049383 |
| EEGNet | 4 | 0.351852 | 0.323290 | 0.814815 | 0.000000 |
| EEGNet | 5 | 0.333333 | 0.327452 | 0.814815 | 0.062500 |
| ShallowConvNet | 1 | 0.288889 | 0.276569 | 0.933333 | 0.031250 |
| ShallowConvNet | 2 | 0.388889 | 0.388163 | 0.833333 | 0.028169 |
| ShallowConvNet | 3 | 0.407407 | 0.405577 | 0.740741 | 0.222222 |
| ShallowConvNet | 4 | 0.351852 | 0.345408 | 0.814815 | 0.060606 |
| ShallowConvNet | 5 | 0.333333 | 0.332924 | 0.944444 | -0.051948 |

Across-fold mean ± population standard deviation:

| Model | Balanced accuracy | Macro F1 | Ordinal MAE | QWK |
|---|---:|---:|---:|---:|
| EEGNet | 0.357778 ± 0.022296 | 0.341133 ± 0.028816 | 0.854074 ± 0.038031 | 0.049877 ± 0.026221 |
| ShallowConvNet | 0.354074 ± 0.041811 | 0.349728 ± 0.045276 | 0.853333 ± 0.076501 | 0.058060 ± 0.090188 |

## 13. Subject-level summary

Metrics were computed from each test subject's nine record predictions, not by
collapsing the subject to one class.

| Model / metric | Mean | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| EEGNet accuracy | 0.356322 | 0.045009 | 0.333333 | 0.333333 | 0.444444 |
| EEGNet balanced accuracy | 0.356322 | 0.045009 | 0.333333 | 0.333333 | 0.444444 |
| EEGNet macro F1 | 0.275831 | 0.067209 | 0.259259 | 0.166667 | 0.444444 |
| EEGNet ordinal MAE | 0.854406 | 0.105625 | 0.888889 | 0.666667 | 1.000000 |
| ShallowConvNet accuracy | 0.356322 | 0.089363 | 0.333333 | 0.222222 | 0.555556 |
| ShallowConvNet balanced accuracy | 0.356322 | 0.089363 | 0.333333 | 0.222222 | 0.555556 |
| ShallowConvNet macro F1 | 0.306194 | 0.102042 | 0.285714 | 0.133333 | 0.547619 |
| ShallowConvNet ordinal MAE | 0.850575 | 0.173305 | 0.888889 | 0.444444 | 1.222222 |

## 14. Confusion matrices

Rows are true classes and columns are predicted classes.

EEGNet, windows:

```text
[[1921, 1765, 1893],
 [1820, 1903, 1915],
 [1646, 1854, 2210]]
```

EEGNet, records:

```text
[[30, 27, 30],
 [29, 29, 29],
 [25, 28, 34]]
```

ShallowConvNet, windows:

```text
[[1751, 1841, 1987],
 [1577, 1881, 2180],
 [1661, 1700, 2349]]
```

ShallowConvNet, records:

```text
[[29, 29, 29],
 [24, 29, 34],
 [25, 27, 35]]
```

## 15. Ordinal metrics

| Level / model | Ordinal MAE | Within one class | Quadratic weighted kappa |
|---|---:|---:|---:|
| Window / EEGNet | 0.852602 | 0.790926 | 0.051764 |
| Window / ShallowConvNet | 0.862173 | 0.784486 | 0.038663 |
| Record / EEGNet | 0.854406 | 0.789272 | 0.051282 |
| Record / ShallowConvNet | 0.850575 | 0.793103 | 0.057143 |

## 16. Leakage and coverage audit

For every fold, all of the following overlaps are zero:

- outer train/test subject, record, record-group, and sample overlap;
- inner train/validation subject, record, record-group, and sample overlap;
- inner partition/outer-test subject, record, and sample overlap.

Both models produced one prediction for every accepted window and record:

- 16,927 rows and 16,927 unique `sample_id` values;
- 261 rows and 261 unique `record_id` values;
- all 29 subjects and folds 1–5 represented;
- all probability values finite;
- maximum probability-sum error: `1.19e-7` for EEGNet and `1.79e-7` for
  ShallowConvNet.

## 17. Runtime

| Model | Total wall time | Sum of fold training times | Peak allocated CUDA memory |
|---|---:|---:|---:|
| EEGNet | 563.93 s | 554.43 s | 519,623,168 bytes |
| ShallowConvNet | 582.29 s | 572.41 s | 1,600,241,664 bytes |

The technical one-fold smoke-runs completed before the full runs. They used two
epochs and four training batches per epoch, with a full validation pass and
full fold-1 test inference. Smoke checkpoint reload, finite loss, probability
normalization, record aggregation, and split isolation all passed.

| Smoke model | Fold-1 record balanced accuracy | Record macro F1 | Record ordinal MAE | Training time |
|---|---:|---:|---:|---:|
| EEGNet | 0.377778 | 0.375404 | 0.822222 | 7.31 s |
| ShallowConvNet | 0.333333 | 0.240196 | 0.933333 | 6.84 s |

These smoke values are implementation checks and are not scientific results.

## 18. Device and library versions

- Device: NVIDIA GeForce RTX 5060 Ti
- Python: 3.11.15
- PyTorch: 2.11.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- NumPy: 2.4.6
- pandas: 3.0.3
- scikit-learn: 1.9.0

## 19. Single-seed limitation

This is one seed and one fixed protocol. Fold dispersion is reported, but it is
not a substitute for between-seed uncertainty. The small advantage above the
balanced three-class chance level of approximately 0.333 is diagnostic and is
not evidence of a statistically stable effect.

## 20. Unequal windows per record

The training loader shuffles ordinary windows. No record-balanced sampler was
introduced, so longer records contribute more gradient updates. Runtime
artifacts preserve window counts per record and class for a future controlled
comparison. Record-level evaluation prevents long test recordings from
directly dominating the primary pooled metric, but it does not remove the
training-side weighting effect.

## 21. Model comparison

The two models are effectively tied on pooled record balanced accuracy
(`0.356322`). EEGNet has marginally higher pooled record macro F1
(`0.355945` versus `0.355701`) and lower fold and subject variability.
ShallowConvNet has marginally lower record ordinal MAE (`0.850575` versus
`0.854406`) and higher record QWK (`0.057143` versus `0.051282`), but its fold
variability is substantially larger. Window-level metrics favor EEGNet by
roughly 0.3–0.4 percentage points. Neither architecture establishes a strong
N-Back signal in this single-seed whole-record baseline.

## 22. Readiness for a multi-seed experiment

The data contract, immutable manifest use, scale audit, model factory,
record-aware early stopping, prediction exports, checkpoint reload, leakage
audit, and resume path have all been exercised end to end. The infrastructure
is technically ready for a predeclared multi-seed experiment. Scientifically,
such a run should occur only after deciding whether this weak diagnostic signal
justifies the compute and after explicitly retaining record-level metrics as
primary.

## 23. Recommended next step

First review the near-chance result and the known whole-record/source-filter
limitations. If the result is still scientifically useful, predeclare a small
seed set and compare the two architectures on the unchanged manifests and
configs, reporting between-seed uncertainty. Do not expand to MATB-II,
62-channel input, transfer learning, or architecture search as part of that
confirmation.

## Runtime artifacts

Full outputs:

- `benchmark_results/cog_bci_baselines/nback_3class/eegnet_seed42`
- `benchmark_results/cog_bci_baselines/nback_3class/shallowconvnet_seed42`

Each directory contains the resolved configuration, run and aggregate
summaries, fold and subject metrics, window and record predictions, confusion
matrices, training history, scale and leakage audits, per-fold checkpoints, and
an errors table. These runtime outputs remain ignored and are not intended for
Git.

## Verification

- Python compilation: passed.
- New baseline tests: 10 passed.
- Related targeted suite: 80 passed, 2 warnings.
- Full `tests` discovery: 919 passed, 12 warnings.
- Full repository discovery: 919 passed, 12 warnings.
- `git diff --check`: passed.
