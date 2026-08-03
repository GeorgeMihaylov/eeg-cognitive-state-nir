# Raw-deduplicated FOMAML diagnostic on `label_q5`

## Status and scope

- Branch: `integration/benchmark-unification`.
- Base HEAD: `dda4254` (`feat(meta): add raw-deduplicated FOMAML protocol`).
- Result status: `diagnostic`.
- Decision: `do_not_proceed`.
- Executed scope: production EEGNet, outer fold 1, seed 42, CUDA, one run.
- Not executed: other folds, seeds, architectures, hyperparameter search, raw-cache rebuild, or outer-split rebuild.

The scientific hypothesis was that a production EEGNet initialization learned
with First-Order MAML would improve equal-budget full-model personalization for
new participants relative to a conventionally supervised initialization. The
comparison used identical raw support/query samples and identical adaptation
parameters. This one-fold diagnostic does not support the hypothesis on its
primary participant-level macro-F1 criterion.

## Immutable protocol and preregistration

- Protocol ID: `fomaml_label_q5_raw_deduplicated_v2`.
- Protocol hash:
  `e73703a443aea3b34f62606efa76bd592ff70099a30cdca80d292f1d76a1fd60`.
- Raw-universe hash:
  `308fdd96523565417d0fc2f3b3bfdea74639224ce1ee44f5e87af6cc0b7e94cf`.
- Outer-fold source artifact SHA-256:
  `41ec5a244e11b5dd4ff25faa7361f2bca302dd719612fea8cbc54a55b6ff3341`.
- Execution preregistration SHA-256:
  `07c57c2f957125e603fa2afaad2078dc0a206f3697f7f01926dcae55133a18e3`.
- Disabled task-8C preregistration remained byte-identical at SHA-256
  `dc998ca72142678394e6f85e10d4b89b1fd0205a6a87be5cae2ba26c37c98692`.

The execution preregistration was created before data loading or any gradient
step. It fixed the architecture, device, seed, episode IDs, optimization
parameters, BatchNorm policies, checkpoint criteria, metrics, policy-selection
rule, decision rule, and output directory. The source protocol, episode index,
raw-universe manifest, error table, and disabled preregistration remained
unchanged throughout execution.

## Architecture and data contract

Production EEGNet passed its pre-training audit:

| Property | Value |
|---|---:|
| input | `[B,1,14,2560]` |
| output classes | 5 |
| latent dimension | 1,280 |
| parameters | 8,501 |
| architecture signature | `248d244e050af2b9e5bd1de0b706665efd3cb13a642044a965fc85602bbe23a7` |

The existing raw-deduplicated cache contains 30,958 float32 windows at 256 Hz
and was reused without rebuilding. Canonical channel order and representative
mmap tensor shape/finiteness checks passed before training.

The fixed protocol contains 11 meta-train, five meta-validation, and five
eligible protected outer-test participants, one episode per participant.
Support is one complete earliest logical record; query is every complete later
logical record. No record truncation, replacement, oversampling, window-level
fallback, sample remapping, or feature-level episode reuse occurred. Each
support and query partition contains classes 0--4.

## Training and meta-validation

The supervised baseline used 6,673 meta-train samples for optimizer steps and
1,269 meta-validation query samples only for early stopping/checkpoint
selection. It trained for nine epochs, selected epoch 6, took 41.45 seconds,
and achieved meta-validation macro F1 0.215615 and balanced accuracy 0.231422.

FOMAML used `create_graph=false`, one inner step, inner learning rate 0.01,
Adam meta learning rate 0.001, meta-batches of four episodes, gradient clipping
at 5.0, and meta-train participants only for updates. Query loss supplied the
first-order meta-gradient but query labels never entered inner adaptation.

| BatchNorm policy | epochs | best epoch | training seconds | meta-validation macro F1 | balanced accuracy |
|---|---:|---:|---:|---:|---:|
| `frozen_global` | 8 | 6 | 51.48 | 0.127903 | 0.233135 |
| `support_local` | 5 | 2 | 31.33 | 0.127360 | 0.238944 |

The macro-F1 difference was -0.000542 for `support_local` minus
`frozen_global`, within the preregistered 0.005 tie region. Therefore
`frozen_global` was selected as the simpler policy before outer-test was
opened. The selected checkpoint SHA-256 is
`2377eba652e9e1f8f44eee12dff2c8dcec9575e41539c8f07e8313ffb78b6a22`.

## Outer-test unlock and aggregate results

The immutable outer-test unlock manifest was written only after fixing the
supervised checkpoint, selected FOMAML checkpoint, BatchNorm policy, and all
adaptation parameters. Its SHA-256 is
`4e9cd1edf65e62fac2c22633c5002703a63b4ddee38ec468a1e222e18c7b4eb1`.

Participant means are the primary aggregation. Window-level values are shown
only as descriptive diagnostics; windows are not treated as independent
statistical observations.

| Mode | participant mean accuracy | balanced accuracy | macro F1 | weighted F1 | ordinal MAE |
|---|---:|---:|---:|---:|---:|
| zero-shot supervised | 0.279847 | 0.224124 | 0.210094 | 0.275182 | 1.185835 |
| supervised full-model | 0.271127 | 0.216438 | 0.198521 | 0.257170 | 1.217170 |
| selected FOMAML | 0.227180 | 0.255491 | 0.152184 | 0.173125 | 1.666263 |

| Mode | window accuracy | balanced accuracy | macro F1 | weighted F1 | ordinal MAE |
|---|---:|---:|---:|---:|---:|
| zero-shot supervised | 0.262974 | 0.255197 | 0.247747 | 0.250916 | 1.239650 |
| supervised full-model | 0.254227 | 0.257351 | 0.247672 | 0.245410 | 1.282799 |
| selected FOMAML | 0.237901 | 0.237090 | 0.179834 | 0.180325 | 1.686297 |

All three modes evaluated the same 1,715 query windows from the same five
participants. Each mode has unique sample IDs; sample ID, participant and
`y_true` identities match exactly across modes. All probabilities are finite
and their maximum row-sum error is `2.38e-7`.

## Participant-level results

| participant | support/query windows | zero-shot macro F1 | supervised-adapted macro F1 | FOMAML macro F1 | supervised balanced acc. | FOMAML balanced acc. |
|---|---:|---:|---:|---:|---:|---:|
| `3110e0c7` | 240 / 202 | 0.207207 | 0.177133 | 0.197082 | 0.195816 | 0.238605 |
| `7150e10a` | 688 / 176 | 0.178580 | 0.137668 | 0.028540 | 0.154000 | 0.210333 |
| `71f0603f` | 650 / 374 | 0.182542 | 0.168961 | 0.055608 | 0.210706 | 0.208153 |
| `c112918e` | 624 / 219 | 0.257351 | 0.280363 | 0.274180 | 0.286148 | 0.373453 |
| `d111e017` | 581 / 744 | 0.224788 | 0.228481 | 0.205509 | 0.235523 | 0.246913 |

## Paired comparison and support budget

For selected FOMAML minus supervised full-model:

- mean / median macro-F1 difference: -0.046338 / -0.022972;
- mean / median balanced-accuracy difference: +0.039053 / +0.042790;
- mean / median ordinal-MAE difference: +0.449093 / +0.247525, where
  positive is worse;
- macro-F1 wins/losses/ties: 1/4/0.

The 10,000-resample participant-level diagnostic bootstrap gave a macro-F1
mean-difference interval of [-0.093587, 0.000912], balanced-accuracy interval
of [0.012092, 0.066013], and ordinal-MAE interval of [0.116061, 0.803109].
With only five participants these intervals are unstable and are not evidence
of statistical significance.

Outer-test support varied from 240 to 688 windows because complete records
were preserved. The descriptive association between support size and FOMAML
macro-F1 gain was Pearson -0.726 and Spearman -0.800. At `n=5`, this is neither
causal nor a reliable estimate of a population association.

## Leakage and buffer audit

- Meta-train/meta-validation/outer-test subject intersections: zero.
- Support/query sample and logical-record intersections: zero.
- Missing raw IDs and duplicate episode sample references: zero.
- Meta-validation and outer-test samples were absent from optimizer steps.
- Outer-test was absent from policy, checkpoint and epoch selection.
- Query data was absent from inner adaptation and did not update BatchNorm
  buffers.
- Participant fast states were isolated and discarded after each episode.
- Original supervised and FOMAML checkpoints were unchanged during outer-test.
- All compared modes used identical support/query order, inner steps, inner
  learning rate, loss, gradient clipping, metrics and selected buffer policy.

## Decision and limitations

The preregistered rule returns `do_not_proceed` because selected FOMAML won
participant-level macro F1 for no more than one of five participants. Balanced
accuracy improved on average, but the primary macro-F1 metric declined and
ordinal error worsened. No hyperparameter, policy, checkpoint or model choice
was changed after outer-test inspection.

This is one fold and one seed with five eligible outer-test participants. It
does not estimate five-fold or multi-seed stability and does not establish
statistical significance. Plausible future work, if separately approved,
includes investigating why first-order meta-training collapses macro F1 for
two participants, comparing a better preregistered supervised initialization,
or revisiting the loss/episode objective using meta-validation only. No such
experiment is automatically authorized by this diagnostic.

## Runtime artifacts and command

Ignored runtime artifacts are under
`benchmark_results/meta_learning_fomaml_label_q5_raw_diagnostic/`. They include
the execution preregistration, protocol reference, architecture audit,
supervised and both FOMAML checkpoints/histories/meta-validation predictions,
policy selection, outer-test unlock, unified predictions, participant and
aggregate metrics, paired comparison, support-budget analysis, leakage/buffer
audits, decision, summary, errors, and runtime report. No runtime artifact
contains an absolute local path.

Executed command:

```powershell
python scripts\run_fomaml_label_q5_raw_diagnostic.py `
  --config experiments\meta_learning\fomaml_label_q5_raw_diagnostic.json `
  --verbose
```

Pre-run verification completed with 16 new targeted tests, 79 related
meta/FOMAML tests, and two full suites of 1,103 tests. Final verification is
reported separately in the task handoff.
