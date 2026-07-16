# Transformer legacy implementation audit

## Source branch/commit

The primary legacy Transformer implementation is on
`origin/latent-state-alignment` in commit `0c6aea0` (`transformer_add`):

```text
src/42_train_transformer_latent_trajectory_model.py
```

That commit is a descendant of `b5c8cb9` (`Add EEG PM baselines and MHA
experiments`), which contains the earlier
`src/11_train_multihead_attention_short.py`. The earlier model is useful as
historical evidence for the value of local temporal context, but `src/42` is
the model selected for migration because it is explicitly the project's
Transformer trajectory architecture.

## Original files

- `src/42_train_transformer_latent_trajectory_model.py`: dataset loading,
  feature selection, sequence construction, splitting, preprocessing,
  Transformer definition, training, metrics and artifact generation in one
  1,446-line script.
- `src/43_train_transformer_latent_trajectory_with_calibration.py`: a separate
  calibration experiment built around the trajectory model. Calibration is
  outside the current task.
- `src/11_train_multihead_attention_short.py`: the earlier PM regression
  experiment using short, centred sequences and a TransformerEncoder.
- `README.md` on `origin/latent-state-alignment`: documents the MHA experiments
  and reports that temporal context was useful, but attention did not
  consistently outperform a context-tabular baseline.

No `.pt` checkpoints or completed `src/42` result directory are tracked in Git.
The repository history therefore provides architecture code, not reusable
weights or a reproducible completed classification result.

## Original architecture

The `TransformerRegressor` in `src/42` used:

```text
[batch, sequence_length, n_features]
  -> Linear(n_features, d_model)
  -> learned positional embedding [1, sequence_length, d_model]
  -> TransformerEncoderLayer(
       nhead,
       dim_feedforward,
       GELU,
       norm_first=True,
       batch_first=True
     ) x num_layers
  -> last-token or unmasked mean pooling
  -> LayerNorm
  -> Linear(d_model, d_model)
  -> GELU
  -> Dropout
  -> Linear(d_model, output_dim)
```

Default architecture parameters were `d_model=128`, `n_heads=4`,
`num_layers=2`, `dim_feedforward=256`, `dropout=0.1`, learned positional
encoding and last-token pooling. The earlier `src/11` model used a smaller
`d_model=64`, one encoder layer and centre-token pooling for centred PM
sequences.

## Original input representation

The selected model is feature-based, not raw EEG. Its intended input is a
sequence of aggregated EEG and power features:

```text
[batch, sequence_length, 448]
```

The default sequence length was 8 and the target was taken from the last
window. Sequences were grouped by every available identifier among `source`,
`subject_id` and `record_id`, then sorted by window start. This preserves
record boundaries and matches the scientific idea of trajectory modelling.

The legacy target was continuous slow latent state regression
(`slow_pca_1` through `slow_pca_4`). The current benchmark target is the
five-class `label_q5`, so only the representation and encoder architecture are
portable.

## Original training procedure

The script implemented a private training pipeline:

- fold-train-only median imputation and standardisation;
- target standardisation for regression;
- AdamW, MSE loss and gradient clipping at 1.0;
- `batch_size=128`, up to 50 epochs, patience 8 and minimum delta `1e-4`;
- a private DataLoader loop and early stopping state;
- private prediction, metrics, checkpoint and CSV/plot/report serialization.

These responsibilities now belong to `TorchClassificationAdapter` and
`BenchmarkRunner` and must not be copied.

## Original splitting procedure

The script supported three independent modes:

- random sequence split (enabled by default together with GroupKFold);
- GroupKFold by `subject_id`;
- cross-source evaluation.

GroupKFold by subject is the only directly relevant scientific outer protocol.
The current benchmark already owns the canonical precomputed subject folds and
record-grouped inner validation, so none of the legacy split implementation is
needed.

## Reusable components

- Feature-sequence Transformer rather than raw-sample self-attention.
- Input projection from 448 EEG+POW features into a compact embedding.
- Learned positional embedding.
- Pre-norm `TransformerEncoder` with GELU.
- Last-valid-token pooling, matching prediction of the final window.
- Configurable `d_model`, heads, layers, feed-forward dimension, dropout,
  activation, positional encoding and pooling.
- Sequence grouping intent: never cross source, subject or record boundaries.
- Fold-local model initialization and train-only feature scaling as concepts;
  their current implementation comes from the canonical benchmark.

## Incompatible components

- Regression head, MSE loss and target scaler.
- The private Dataset/DataLoader/training/early-stopping loop.
- The private CLI and `Config` dataclass.
- The private split implementation and default random split.
- Ad-hoc feature discovery and optional variance-based truncation.
- CSV-only predictions, custom report directories and custom checkpoints.
- Plotting and Ridge baseline code.
- The calibration script.
- Legacy weights: none are tracked, and regression weights would not match the
  five-class target even if available.

## Leakage risks

1. The default random sequence split can put overlapping sequences, windows,
   records and subjects on both sides of evaluation.
2. Legacy sequence construction respected record boundaries but did not split
   records at missing or non-monotonic time gaps. A sequence could therefore
   bridge a recording interruption.
3. `--max-rows` sampled individual rows before sequence construction, which
   destroys temporal continuity while the old builder had no gap check.
4. Mean pooling had no padding mask. This did not affect fixed-length legacy
   sequences, but would be incorrect for padded variable-length input.
5. Learned positional embeddings had a fixed maximum length and the forward
   method accepted no padding mask.

The current `model_zoo/DL/sequence_utils.py` already prevents source, subject
and record crossing and provides gap-aware segmentation. It must remain the
single sequence builder.

## Migration plan

1. Implement one `torch.nn.Module` classifier in
   `model_zoo/DL/transformer.py`, preserving input projection, learned
   positional encoding, pre-norm encoder and last-token pooling.
2. Provide explicit padding-mask support. Fixed-length benchmark sequences do
   not require padding, but masked mean/last/CLS behaviour will be testable and
   padded tokens will never be pooled.
3. Build it as canonical `torch_transformer` through `model_zoo/factory.py` and
   the existing `TorchClassificationAdapter`. Unknown parameters must fail and
   `d_model % nhead == 0` must be validated.
4. Mark `torch_transformer` as a sequence model so `BenchmarkRunner` invokes
   the existing gap-aware sequence builder before factory construction.
5. Use EEG+POW features, `label_q5`, sequence length 8, last target, expected
   10-second step and maximum 10.5-second gap.
6. Use the existing 5-fold subject GroupKFold and record-grouped inner
   validation. Do not introduce a new runner, adapter, CLI, config schema or
   artifact format.
7. Run unit/integration tests, then a one-fold/1,000-window/3-epoch smoke via
   the main CLI. Estimate resources before the full seed-42 run.
8. After a successful full run, compare with the existing feature-sequence
   LSTM/BiLSTM and feature baselines using standard benchmark artifacts.
