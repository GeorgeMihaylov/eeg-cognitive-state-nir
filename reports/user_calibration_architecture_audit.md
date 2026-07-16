# User calibration architecture audit

## Existing canonical path

The completed zero-shot Transformer run is
`benchmark_results/groupkfold_torch_transformer_label_q5/20260716_191246`.
Its resolved config hash is
`ea4dbe39293c14d7c171901c46b53a3a9aa2edb6825c1b70cae379cafa220416`.
The run contains five subject-disjoint GroupKFold checkpoints, their original
train-only normalization statistics, validation manifests, predictions, and
the canonical resolved config.

The current call graph is:

```text
BenchmarkRunner
  -> dataset registry / task registry
  -> CrossValidator.run_group_kfold(subject_id)
  -> BenchmarkRunner._build_sequence_split
       -> sequence_utils.build_sequences
  -> model_zoo.factory.build_model
       -> TorchClassificationAdapter
  -> standard metrics and fold artifacts
```

`CrossValidator` deterministically reconstructs the same five outer folds.
`build_sequences` groups by `source`, `subject_id`, and `record_id`, and splits
segments at non-positive or greater-than-configured time gaps. The Transformer
base run uses length 8, stride 1, a 10 second expected step, and a 10.5 second
maximum gap.

## Base checkpoint loading

Each fold checkpoint stores the module state, architecture metadata, input
shape, class count, training configuration, training summary, train-only
feature mean/scale, and validation manifest. The factory can reconstruct the
adapter from the resolved benchmark config, but the adapter currently has no
public checkpoint-loading method. The minimal extension is an instance
`load(path)` method that validates input shape/class count/model metadata,
restores model state and normalization statistics, marks the adapter fitted,
and never changes the checkpoint on disk.

The calibration experiment must validate the referenced benchmark run and
fold checkpoint before use. `model_reference.json` will retain the base run,
fold, checkpoint, and base config hash instead of copying the zero-shot model.

## Independent per-subject models

The zero-shot fold adapter must remain immutable. A small adapter `clone()`
operation can deep-copy the loaded fitted adapter for every test subject. All
normalization replacement or weight updates occur only on that clone. Loading
the checkpoint again would also be independent, but an explicit clone makes
the protocol and tests clearer. The implementation must compare the base state
before and after every subject to detect accidental mutation.

## Chronological calibration split

The split must be made at the original window level, before sequence building:

1. select one outer-test subject only;
2. order windows deterministically by `source`, `record_id`, time, and
   `sample_id`;
3. consume the earliest real-time interval for calibration;
4. purge at least seven subsequent windows at an intra-record boundary;
5. build calibration and evaluation sequences separately with the canonical
   gap-aware builder.

Building both partitions separately proves that no sequence can reuse a source
window across the boundary. Record boundaries and gaps remain enforced by the
existing builder. A second chronological split inside calibration data creates
an 80/20 train/validation partition with its own purge. If this cannot produce
usable train and validation sequences, calibration uses a configured fixed
small epoch count and records `fixed_epochs_no_validation`; evaluation data is
never supplied to the adapter.

Budgets are measured from actual timestamps plus the configured window
duration, not from the number of overlapping Transformer sequences. A 20%
fractional budget can use the same window-level mechanism even though the first
full experiment uses duration budgets only.

## Normalization semantics

The checkpoint retains outer-train normalization statistics. Zero-shot and
head-only/full-model fine-tuning continue to use those statistics.
Subject-normalization computes mean and scale from unique calibration windows
only and replaces the clone's transform statistics for both calibration and
evaluation inference. It does not blend in evaluation statistics and does not
alter the base checkpoint. Both train and subject statistics are recorded in
metadata so the replacement rule is explicit and reproducible.

## Fine-tuning through the existing adapter

The adapter owns AdamW, CrossEntropyLoss, DataLoader, device handling, seed,
logging, prediction, and probability code. The experiment layer must not add a
training loop. The minimal adapter extension is `fine_tune(...)` with explicit
calibration train and optional calibration-validation arrays:

- `head_only`: only parameters under the Transformer `classifier` module are
  trainable (LayerNorm and classification head);
- `full_model`: all parameters are trainable for future use;
- an explicit validation subset enables early stopping;
- absent validation uses a configured fixed epoch count;
- one-class calibration labels remain legal because missing future classes are
  expected and must not exclude a subject.

No architecture or factory special case is needed. The same neutral adapter
API can later be used by LSTM or ShallowConvNet with a model-specific trainable
parameter selector.

## Artifact and metric reuse

`MetricsCalculator.calculate_all_metrics` remains the metric entry point. The
two requested ordinal diagnostics should be added there once, rather than
implemented privately in the experiment. Calibration writes the existing
prediction convention (`y_true`, `y_pred`, `proba_*`) plus calibration identity
columns. Per-subject metrics are the primary aggregation units.

The experiment layer writes, per fold/subject/budget/method:

```text
calibration_spec.yaml
calibration_split.json
calibration_training_log.csv
calibration_metrics.json
predictions.parquet
model_reference.json
calibrated_head.pt (head-only only)
```

It also writes a deterministic top-level manifest, unified predictions, and
subject/summary tables. Calibration observations never appear in evaluation
predictions or metrics.

## Reusable entry points

- `BenchmarkRunner.validate_completed_run` and benchmark config hashing;
- `BenchmarkRunner.load_dataset` for the registered feature dataset;
- dataset and task registries plus `CrossValidator.run_group_kfold`;
- `sequence_utils.build_sequences` for every calibration partition;
- `model_zoo.factory.build_model` to reconstruct the Transformer;
- `TorchClassificationAdapter` for loading, cloning, normalization, training,
  prediction, and checkpoint data;
- `MetricsCalculator.calculate_all_metrics` for evaluation.

## Required minimal changes

1. Add checkpoint loading/cloning and calibration fine-tuning to the existing
   adapter.
2. Add the requested ordinal diagnostics to the existing metrics calculator.
3. Add `bench/experiments/user_calibration.py` for split planning, experiment
   orchestration, aggregation, and artifacts.
4. Add one experiment YAML and route it through the existing `cli.py`.
5. Add focused leakage, independence, parameter-freezing, resolver, and smoke
   tests.

`BenchmarkRunner`, `CrossValidator`, dataset loaders, model factory, Transformer
architecture, and sequence builder require no behavioral rewrite.

## Compatibility risks

- Calibration intervals often contain one class; reusing the ordinary `fit`
  label checks would incorrectly reject them.
- Overlapping sequences cannot be split directly without leaking source
  windows; all splits therefore occur before sequence construction.
- Subject statistics computed from overlapping sequences would overweight
  middle windows; statistics must use unique calibration windows.
- A base adapter `fit` call resets to its initial random state, so calibration
  requires a dedicated adapter method that starts from loaded weights.
- Zero-shot comparisons at different budgets use different remaining tails.
  A zero-shot control is therefore evaluated on every budget-specific
  evaluation subset.
- Some short subjects/budgets cannot support a purged validation split. The
  fallback epoch mode must be explicit rather than using evaluation loss.
- The current user `.gitignore` ignores CSV files. Required CSV reports can be
  created without modifying `.gitignore`, but remain ignored locally.
