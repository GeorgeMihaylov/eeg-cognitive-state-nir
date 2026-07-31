# Diagnostic FOMAML experiment on `label_q5`

## Status

- Branch: `integration/benchmark-unification`.
- Base HEAD: `2b5d071` (`feat(meta): add production-safe FOMAML contract`).
- Intended status: one-fold, one-seed diagnostic.
- Actual status: `blocked_protocol_raw_sample_mismatch`.
- Supervised training, FOMAML training, BatchNorm-policy selection, checkpoint
  selection, and outer-test evaluation were not performed.

The scientific hypothesis was that a production EEGNet initialization learned
with First-Order MAML on outer-train participants would improve equal-budget
personalization for new participants relative to a conventionally supervised
initialization. The hypothesis remains untested because a mandatory
pre-training data-contract check failed.

## Preregistration

The immutable preregistration was written before loading any episode tensors
or executing a gradient step.

- SHA-256:
  `54f21e907ff1a414d45c1594e422c4caede0a449ca9acf02374bb50502122754`.
- Outer fold: 1.
- Seed: 42.
- Model: production EEGNet.
- Support/query budget: 32/64 windows.
- Inner steps: 1.
- Inner learning rate: 0.01.
- Meta learning rate: 0.001.
- Meta-batch size: 4.
- Maximum supervised/meta epochs: 12/8.
- Buffer policies: `frozen_global`, `support_local`.
- Device resolved before preregistration: CUDA.

No preregistered value was changed after the file was written.

## Architecture and split contract

The production architecture audit passed before the blocker was reached:

- input: `[B,1,14,2560]`;
- sampling rate and window: 256 Hz, 10 seconds;
- latent dimension: 1280;
- output width: 5;
- parameters: 8,501;
- architecture signature:
  `248d244e050af2b9e5bd1de0b706665efd3cb13a642044a965fc85602bbe23a7`.

The immutable task-8F protocol also passed its metadata-only leakage audit:

- protocol hash:
  `a3e6ff5ee2dbfa1638ffee9180ddff582dbab8aa6186e164320dd92f082871e8`;
- outer train/test subjects: 43/11;
- meta-train/meta-validation subjects: 34/9;
- materialized meta-train/meta-validation/outer-test episodes: 23/9/8;
- subject, support/query sample, and support/query record overlaps: zero;
- within-record fallback: absent.

The 14 participants already skipped by task 8F remain skipped: 13 had fewer
than two records, and one had only three eligible support windows. Their
reasons were not modified.

## Blocking raw-data alignment audit

Task 8F materialized its episodes from the feature-level Random Forest
prediction artifact, whose supervised universe contains 45,384 windows. Task
8X requires the canonical raw-deduplicated EEG universe, which contains 30,958
accepted windows. Before normalization or training, every preregistered
support/query `sample_id`, subject, and target was checked against this raw
cache.

| scope | requested IDs | missing IDs | affected episodes | fully available episodes |
|---|---:|---:|---:|---:|
| meta-train | 2,208 | 578 | 10 / 23 | 13 |
| meta-validation | 864 | 193 | 4 / 9 | 5 |
| outer-test | 768 | 130 | 3 / 8 | 5 |
| total | 3,840 | 901 | 17 / 40 | 23 |

For the 2,939 IDs that are present, subject identity and `label_q5` agree with
the task-8F manifest without a single mismatch. Most missing blocks are entire
query records excluded by logical-record deduplication; a small number are
individual boundary windows.

There is no permitted automatic repair:

- retaining only fully available episodes would add 17 new participant skips;
- using the all-source raw universe would violate the required deduplicated
  dataset mode;
- mapping discarded source records to selected logical records would change
  support/query sample IDs and episode hashes;
- rebuilding raw episodes would change the fixed task-8F protocol hash and the
  already written preregistration.

Each option is explicitly outside task 8X, so the experiment fails closed.

## Training, policies, and outer-test

No supervised training history or checkpoint exists. Neither
`frozen_global` nor `support_local` entered real-data meta-training, so there
are no meta-validation policy metrics and no selected policy. No FOMAML
checkpoint exists.

The immutable pre-outer-test decision manifest was never created and
outer-test episode tensors were never materialized. Consequently there are no
zero-shot, supervised full-model, or FOMAML predictions, no participant
metrics, no paired differences, and no win/loss count. Reporting placeholder
zeros would be scientifically incorrect, so these artifacts are intentionally
absent.

## Leakage and immutability audit

- Preregistration was created before training and remains byte-identical.
- Source protocol, episode, and error artifacts remain unchanged.
- Outer-test was not used for policy, checkpoint, epoch, learning-rate, or
  support-budget selection.
- Query data never entered adaptation or BatchNorm updates.
- No participant state was created or reused.
- No source checkpoint was changed.
- No unsafe sample remapping, episode rebuild, deduplication bypass, or
  additional participant drop was applied.

## Runtime evidence

The blocked run preserves:

- `experiment_preregistration.json` and its hash sidecar;
- copies of the immutable protocol manifest and episode index;
- `raw_cache_alignment_audit.json` with partition-level missing IDs;
- combined original and alignment errors;
- `leakage_audit.json`;
- `decision.json`;
- `diagnostic_summary.json`;
- `diagnostic_report.md`.

All are runtime artifacts under
`benchmark_results/meta_learning_fomaml_label_q5_diagnostic/` and are not
tracked by Git. They contain no local absolute paths.

## Verification and next step

- New blocker-focused tests: 16 passed, 1 existing pytest-config warning.
- Related FOMAML tests: 36 passed before the blocker audit extension.
- Pre-training full suite: 1,074 passed, 13 warnings.
- Final `python -m pytest -q tests`: 1,075 passed, 13 warnings.
- Final root `python -m pytest -q`: 1,075 passed, 13 warnings.

The diagnostic decision is `blocked_protocol_raw_sample_mismatch`, not
`proceed`, `inconclusive`, or `do_not_proceed`, because no model comparison
was executed. A future task must explicitly authorize materializing a new
raw-deduplicated episode protocol, assign it a new protocol hash, and create a
new preregistration before this scientific experiment can be run.
