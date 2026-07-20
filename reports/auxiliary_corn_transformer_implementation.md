# Auxiliary CORN Transformer implementation

Implementation date: 2026-07-19. Source revision: `26c20fd`.

This implementation completes task 7B only. It adds the joint categorical + auxiliary
CORN model path and synthetic/integration tests. It does not train on the EEG dataset,
does not select lambda, and does not run GroupKFold.

## 1. Implemented architecture

The canonical model name remains `torch_transformer`. The new resolved head type is:

```yaml
head_type: categorical_corn
auxiliary_weight: 0.5
```

A single call to `TorchFeatureTransformerClassifier.encode()` produces the pooled
representation. The unchanged `classifier.*` categorical head produces five logits,
and the existing `CornOrdinalHead` is registered as
`auxiliary_ordinal_head.*` and produces four conditional logits.

The forward contract is the named tuple:

```python
CategoricalCornOutput(
    categorical_logits: Tensor,  # [B, 5]
    ordinal_logits: Tensor,      # [B, 4]
)
```

The legacy categorical state keys remain unchanged:
`classifier.0.*`, `classifier.1.*`, and `classifier.4.*`.

## 2. Composite objective

`CategoricalCornObjectiveHandler` reuses the existing categorical cross-entropy and
canonical CORN risk-set implementation. For a batch:

```text
categorical_loss = CE(categorical_logits, y)
ordinal_loss     = CORN(ordinal_logits, y)
total_loss       = categorical_loss + auxiliary_weight * ordinal_loss
```

The CORN term remains normalized by the number of valid risk-set elements. Epoch
aggregation keeps independent numerators and denominators for categorical and ordinal
components before combining their means. This avoids bias from variable per-batch
CORN risk-set sizes.

`auxiliary_weight` must be finite and non-negative. It is rejected for single-head
categorical, CORAL, and CORN configurations. It is included in model metadata,
checkpoint objective metadata, and the resolved configuration hash.

## 3. Training and early stopping

`TorchClassificationAdapter` retains its single training loop. Objective handlers now
expose named loss components and a combination rule. Existing single-head objectives
map to one component and preserve their previous behavior.

Joint training logs contain:

```text
train_total_loss
train_categorical_loss
train_ordinal_loss
validation_total_loss
validation_categorical_loss
validation_ordinal_loss
early_stopping_metric
```

The legacy `train_loss` and `validation_loss` fields remain aliases of total loss.
For `categorical_corn`, checkpoint selection and early stopping monitor
`validation_categorical_loss`, as specified before implementation. The best-epoch
summary stores the monitor value and all validation components.

## 4. Prediction semantics

The primary model prediction remains categorical:

```text
class_probabilities = softmax(categorical_logits)
y_pred = argmax(class_probabilities)
```

Only these probabilities are used for the primary AUC and standard benchmark metrics.
The primary categorical expected rank is stored separately.

The auxiliary CORN head is decoded independently into:

```text
aux_threshold_probabilities
aux_class_probabilities
aux_expected_rank
aux_ordinal_prediction
aux_ordinal_argmax
aux_conditional_probabilities
```

No averaging or selection between heads is performed.

## 5. Artifacts and metrics

Joint `predictions.parquet` keeps the standard identity and `proba_*` columns and adds:

```text
head_type
class_probability_0 ... class_probability_4
categorical_expected_rank
aux_threshold_logit_0 ... aux_threshold_logit_3
aux_threshold_probability_0 ... aux_threshold_probability_3
aux_conditional_probability_0 ... aux_conditional_probability_3
aux_class_probability_0 ... aux_class_probability_4
aux_expected_rank
aux_ordinal_prediction
aux_ordinal_argmax
auxiliary_weight
```

The runner calculates primary metrics from the categorical head and writes auxiliary
metrics with an `aux_` prefix. It also records categorical/auxiliary prediction
agreement. `auxiliary_corn_metadata.json` audits primary and auxiliary probability
normalization and auxiliary cumulative monotonicity.

Existing categorical, CORAL, and CORN prediction schemas are not populated with empty
`aux_*` columns.

## 6. Checkpoint compatibility

A joint checkpoint contains both `classifier.*` and `auxiliary_ordinal_head.*` keys.
Strict reload into an identically configured joint model is supported. Reload rejects:

- a different `head_type`;
- a different `auxiliary_weight`;
- implicit conversion between pure categorical, pure CORN, and joint checkpoints.

Legacy categorical checkpoint structure and strict loading remain unchanged.

## 7. Calibration behavior

Calibration remains supported only for pure categorical models. Both the experiment
layer and adapter fail before optimization for `categorical_corn` with:

```text
Auxiliary CORN calibration is not supported yet.
```

## 8. Synthetic validation

Synthetic sequence tests used CPU data with shape `[N, 8, 6]` and five classes. The
joint model was built through the canonical factory for auxiliary weights 0.25, 0.5,
and 1.0. Tests covered forward shapes, the exact loss equation, backpropagation,
optimizer training, detailed prediction, probability validity, artifacts, strict
checkpoint reload, and calibration guards.

Synthetic one-epoch results (technical only):

| auxiliary weight | train total | train categorical | train CORN | validation total | validation categorical | validation CORN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 1.83055 | 1.67371 | 0.62737 | 1.86458 | 1.70170 | 0.65153 |
| 0.50 | 1.98739 | 1.67371 | 0.62737 | 2.02735 | 1.70162 | 0.65145 |
| 1.00 | 2.30108 | 1.67371 | 0.62737 | 2.35324 | 1.70189 | 0.65135 |

The full canonical parameter counts are 322,313 for EEG-only and 358,153 for
EEG+POW. Primary and auxiliary class-probability row sums differed from one by at
most `1.79e-7` in the synthetic checks.

New auxiliary-CORN tests:

```text
24 passed
```

Focused legacy and new model tests:

```text
115 passed, 3 skipped
```

Repository-wide test run available without the private data and generated benchmark
artifacts:

```text
334 passed, 3 skipped, 7 deselected, 11 warnings
```

The seven deselected tests require either
`data/processed/windowed_eeg_pm_dataset_w10.parquet` or completed canonical benchmark
runs, neither of which was included in the supplied Git archive. Running the complete
suite without deselection produced 330 passes and seven failures exclusively due to
those missing external artifacts. No failure was caused by the new model path.

## 9. Files changed

Core implementation:

```text
model_zoo/DL/ordinal.py
model_zoo/DL/transformer.py
model_zoo/DL/adapter.py
model_zoo/DL/__init__.py
bench/bench_runner.py
bench/experiments/user_calibration.py
```

New tests:

```text
tests/test_auxiliary_corn_transformer.py
tests/test_auxiliary_corn_objective.py
tests/test_auxiliary_corn_adapter.py
tests/test_auxiliary_corn_artifacts.py
```

Reports:

```text
reports/auxiliary_corn_transformer_implementation.md
reports/auxiliary_corn_transformer_implementation_summary.json
```

## 10. Remaining work for task 7V

The next stage should add only a smoke experiment on real EEG data:

- outer fold 1;
- EEG+POW;
- seed 42;
- three epochs;
- auxiliary weights 0.25, 0.5, and 1.0;
- identical inner split and normalization for all candidates;
- early stopping by categorical validation loss;
- artifact and checkpoint audit.

Nested lambda selection and the full 90-fold-fit experiment remain out of scope until
the smoke pipeline is validated.
