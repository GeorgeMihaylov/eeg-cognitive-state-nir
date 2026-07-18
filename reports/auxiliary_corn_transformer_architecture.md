# Auxiliary CORN Transformer architecture

Design date: 2026-07-18. Source revision: `d8428d9dfadb17078fd523b03ebda3c1e805b996`.
This document is a design specification only: no model code was changed, no model was
trained, and no checkpoint or benchmark artifact was created.

## 1. Motivation from multiseed ordinal results

The categorical EEG+POW Transformer remains the primary reference. Across seeds 7,
42, and 123 it achieved approximately 0.367 balanced accuracy, 0.369 macro F1,
0.985 ordinal MAE, and 0.252 severe-error rate. The pure CORN head improved subject-
level ordinal MAE by 0.03385 (95% paired bootstrap CI [0.01237, 0.05698], Holm
`p=0.01196`) and severe-error rate by 0.01981 (CI [0.00995, 0.03044], Holm
`p=0.00302`) for EEG+POW. Its mean balanced-accuracy change was -0.01114 and its
macro-F1 change was -0.00168. The ordinal-MAE effect was positive in two of three
seeds, while the severe-error effect was positive in all three.

The effect was largest for difficult subjects: the worst categorical-performance
quartile improved by 0.07663 ordinal-MAE units and 0.03308 severe-error units under
EEG+POW CORN. This motivates a shared representation trained with ordinal information
without replacing the categorical decision rule. The hypothesis is not that two
predictions should be ensembled; it is that CORN can regularize the shared encoder
while the categorical head remains the sole primary inference head.

## 2. Existing reusable implementation

The current implementation already has the required seams:

- `TorchFeatureTransformerClassifier.encode()` performs projection, positional
  encoding, the Transformer encoder, pooling, and returns one pooled representation;
- the categorical head is registered as `classifier`;
- `CornOrdinalHead` already implements the desired CORN feature block and four-logit
  output for five classes;
- `corn_loss_parts()` already applies the canonical risk-set mask and exposes a
  numerator and denominator;
- `ClassificationObjectiveHandler` centralizes existing categorical, CORAL, and CORN
  decoding semantics;
- `TorchClassificationAdapter` owns the one DataLoader/AdamW/validation/early-
  stopping/checkpoint loop;
- `BenchmarkRunner` determines `(sequence_length, n_features)` and `num_outputs` from
  each split and task, and writes standard fold artifacts;
- the experiment layer already invokes `BenchmarkRunner` in-process and audits exact
  sequence, fold, validation-group, normalization, and prediction alignment;
- `benchmark_config_hash()` hashes the complete resolved scientific config except
  `output_dir`, so a resolved `head_type` and `auxiliary_weight` naturally affect run
  identity.

The canonical cohort remains 44,142 sequences from 53 subjects, sequence-index
SHA-256 `1d0a1fe8ab8aad1c6da8637cb882c4365c01acb80c0822e34ced21ef7ee36afa`.
The input is `[B, 8, 168]` for EEG-only and `[B, 8, 448]` for EEG+POW.

## 3. Shared encoder

The canonical model name remains `torch_transformer`; no new model-factory name or
second Transformer class is introduced. The resolved configuration selects the new
mode:

```yaml
model:
  name: torch_transformer
  params:
    head_type: categorical_corn
    num_classes: 5
    auxiliary_weight: 0.5
```

The forward graph is:

```text
x [B,T,F]
  -> TorchFeatureTransformerClassifier.encode(x, padding_mask)
  -> pooled h [B,128]
       |-> existing classifier               -> categorical_logits [B,5]
       `-> existing CornOrdinalHead instance -> ordinal_logits     [B,4]
```

`encode()` must be called exactly once. Both heads consume the same `pooled` tensor,
so the projection, positional encoding, attention layers, and pooling are neither
duplicated nor executed twice. The encoder architecture, padding-mask rules, sequence
length, pooling, and feature normalization remain unchanged.

## 4. Categorical head

For both `head_type: categorical` and `head_type: categorical_corn`, `self.classifier`
must be the current `nn.Sequential` in the current order:

```text
classifier.0 = LayerNorm(d_model)
classifier.1 = Linear(d_model, d_model)
classifier.2 = GELU
classifier.3 = Dropout(dropout)
classifier.4 = Linear(d_model, num_classes)
```

The registered state keys therefore remain `classifier.0.*`, `classifier.1.*`, and
`classifier.4.*`. A private construction helper is acceptable, provided the resulting
module is still assigned to `self.classifier` and categorical initialization/order is
unchanged. Pure categorical checkpoints and head-only calibration continue to use the
same names.

## 5. Auxiliary CORN head

The auxiliary module must be an instance of the existing `CornOrdinalHead` and be
registered as:

```text
auxiliary_ordinal_head.*
```

For `d_model=128` and five classes it applies the existing
`LayerNorm -> Linear -> GELU -> Dropout -> Linear(128, 4)` path. Its output is four
conditional logits. The existing target construction, risk-set masks, conditional-to-
cumulative conversion, class-probability conversion, threshold prediction, and
expected-rank functions are reused without numerical changes.

Projected trainable parameter counts are 322,313 for EEG-only and 358,153 for
EEG+POW. These are the existing shared encoder plus both unchanged heads; they must be
verified by the factory tests in the implementation task rather than hard-coded into
model logic.

## 6. Forward output type

Use a named tuple defined beside the ordinal output contracts, not a plain tuple and
not shape inference:

```python
class CategoricalCornOutput(NamedTuple):
    categorical_logits: torch.Tensor
    ordinal_logits: torch.Tensor
```

`TorchFeatureTransformerClassifier.forward()` returns `CategoricalCornOutput` only
for `head_type == "categorical_corn"`. Existing categorical, CORAL, and CORN modes
continue to return a single `Tensor`; their public and checkpoint behavior does not
change. A `NamedTuple` gives explicit field names, remains a lightweight tuple of
PyTorch tensors, is straightforward to test, and does not affect `state_dict` or
`torch.save`.

The adapter/objective layer must validate the object type and both exact widths. It
must never infer that a tensor is categorical or ordinal merely from its second
dimension.

## 7. Composite objective

The exact optimization objective is:

```text
CE = sum(CrossEntropy(categorical_logits, y, reduction="none")) / N

CORN = sum(mask * BCEWithLogits(ordinal_logits, corn_targets))
       / sum(mask)

total = CE + auxiliary_weight * CORN
```

`auxiliary_weight` is a finite scalar greater than or equal to zero. Class weights,
subject weights, changed risk sets, probability-consistency penalties, and regression
losses are out of scope.

The clean extension is a new `CategoricalCornObjectiveHandler` implementing the same
small objective protocol as existing handlers. `ClassificationObjectiveHandler`
retains its current single-head behavior. The shared protocol should expose:

```text
loss_components(outputs, targets) -> named LossParts components
combine(component_means)          -> scalar optimization loss
decode(outputs)                    -> explicit primary and auxiliary outputs
to_metadata()                      -> serializable objective semantics
```

The adapter aggregates the CE numerator/denominator and CORN numerator/denominator
separately across an epoch, then recomputes the aggregate total. This preserves the
current CORN normalization even when batch risk-set sizes differ.

For joint runs, `training_log.csv` must contain finite values for:

```text
train_total_loss
train_categorical_loss
train_ordinal_loss
validation_total_loss
validation_categorical_loss
validation_ordinal_loss
```

The legacy `train_loss` and `validation_loss` columns remain aliases of total loss so
existing artifact checks do not break. The checkpoint summary additionally records
`early_stopping_monitor`, `best_monitor_value`, and the three component values at the
best epoch.

## 8. Prediction semantics

Only the categorical head defines the primary prediction:

```text
class_probabilities = softmax(categorical_logits)
y_pred = argmax(class_probabilities)
```

These values, and only these values, are passed to balanced accuracy, macro F1, AUC,
Cohen's kappa, QWK, ordinal MAE, adjacent accuracy, and severe-error rate. Categorical
expected rank is `sum(class_id * class_probability)` and is the continuous primary-
head diagnostic.

The auxiliary CORN output is decoded separately:

```text
conditional probabilities = sigmoid(ordinal_logits)
cumulative probabilities  = cumulative product of conditional probabilities
auxiliary class probabilities = adjacent differences of cumulative probabilities
auxiliary ordinal prediction  = count(cumulative_probability >= 0.5)
auxiliary ordinal argmax       = argmax(auxiliary class probabilities)
auxiliary expected rank        = sum(cumulative probabilities)
```

The standard predictions artifact retains its identity columns and primary
`proba_0` ... `proba_4` compatibility columns. Joint runs additionally write:

```text
head_type = categorical_corn
class_probability_0 ... class_probability_4
categorical_expected_rank
aux_threshold_probability_0 ... aux_threshold_probability_3
aux_class_probability_0 ... aux_class_probability_4
aux_expected_rank
aux_ordinal_prediction
aux_ordinal_argmax
auxiliary_weight
```

The `aux_threshold_probability_*` fields are cumulative `P(y > k)`, not raw
conditional probabilities. Raw auxiliary logits and conditional probabilities may be
kept in an optional diagnostic artifact, but are not substituted for the required
fields. Per-sample losses are not written to `predictions.parquet`; component losses
belong in `training_log.csv` and fold metrics/metadata.

No probability averaging, outer-test head selection, or auxiliary threshold rule is
used for the primary prediction in the first experiment.

## 9. Checkpoint format

The resolved config and checkpoint must identify the joint model before strict state
loading:

```text
head_type: categorical_corn
num_classes: 5
num_thresholds: 4
auxiliary_weight: <resolved scalar>
objective_type: categorical_plus_auxiliary_corn
early_stopping_monitor: validation_categorical_loss
```

The state dict contains the existing encoder and `classifier.*` keys plus
`auxiliary_ordinal_head.*`. `model_metadata`, `objective`, `training_config`,
`training_summary`, `training_log`, validation split, and normalization statistics
all retain their present checkpoint locations and gain only explicit joint-objective
metadata.

Loading is strict. The factory-built input shape, class count, `head_type`, objective
schema version, and state keys must match. A pure categorical, CORAL, or CORN
checkpoint is rejected before `load_state_dict`; no implicit key conversion or partial
load is allowed. The first experiment initializes the joint model from scratch with
the declared seed. A future explicit warm-start mode could load only encoder and
categorical keys while recording provenance, but it is not part of tasks 7B-7D.

## 10. Calibration compatibility

The current user-calibration experiment already rejects any configured head other
than `categorical`, and adapter `fine_tune()` rejects non-categorical objectives.
The joint mode must therefore fail early and explicitly:

```text
head_type == categorical_corn -> calibration not supported yet
```

It must not silently fine-tune `classifier.*`, ignore the auxiliary objective, or save
a partially defined calibration checkpoint. A future protocol may study categorical-
head-only calibration of a frozen joint encoder, but that is a separate hypothesis
and is not part of task 7B or the first full experiment.

## 11. Required extension points

The later implementation can remain local:

1. `model_zoo/DL/ordinal.py`: typed joint output, objective protocol/composite handler,
   existing CORN reuse, and joint decoded output.
2. `model_zoo/DL/transformer.py`: one additional head branch using one `encode()` call;
   preserve ordinary branches and names.
3. `model_zoo/DL/adapter.py`: component-aware aggregation/logging, categorical-loss
   early stopping for the joint handler, and separate detailed outputs.
4. `model_zoo/DL/__init__.py`: export the new typed contracts/handler.
5. `bench/bench_runner.py`: write named auxiliary fields and guarantee that primary
   metrics/AUC receive categorical probabilities.
6. A later experiment orchestrator: train lambda candidates without evaluating
   outer-test, select on the shared inner validation split, then evaluate only the
   selected checkpoint.
7. Tests for forward/loss/gradients/prediction/checkpoint/hash/calibration/artifacts and
   unchanged legacy behavior.

`model_zoo/factory.py` keeps `torch_transformer`; it requires no new model type. The
builder validates the resolved head and weight and supplies the appropriate handler.
`bench/validation/metrics.py` already accepts primary `y_pred`, five-column
probabilities, and expected rank, so no new metric formulas are required.

The current `BenchmarkRunner.run()` fits and immediately evaluates an outer fold. It
must not be called once per lambda candidate because that would expose and persist
outer-test results before selection. The minimal future runner seam is to separate
"fit one split" from "predict/save a fitted split" while keeping `_evaluate_split()`
as a backward-compatible composition of those two operations.

## 12. Components that must not be duplicated

Do not add a second Transformer encoder/class, `torch_transformer_*` factory name,
training loop, DataLoader path, normalization routine, split builder, CORN target/mask
implementation, probability conversion, metrics implementation, artifact directory
scheme, or sequence builder. Do not rename `classifier.*`, modify legacy head output
types, rebuild the sequence cohort, or introduce CORAL/regression/ensemble behavior.

## 13. Risks

1. Calling `encode()` once per head doubles encoder work and can introduce dropout-
   inconsistent representations.
2. Renaming `classifier.*` breaks strict categorical checkpoint load and head-only
   calibration.
3. Returning an untyped tuple or guessing semantics from shape can swap the two
   four/five-wide outputs after future changes.
4. Averaging loss values by batch count instead of their explicit denominators changes
   CORN weighting.
5. Early stopping on total loss can select a checkpoint with better CORN loss but a
   worse primary categorical head; the experiment therefore monitors categorical CE.
6. Passing auxiliary probabilities to AUC changes the scientific estimand.
7. Running the ordinary full runner for every lambda touches outer-test before lambda
   selection and creates an avoidable leakage/audit risk.
8. Recomputing inner splits or normalization independently with different seeds breaks
   candidate comparability.
9. Partial checkpoint loading can silently produce an untrained auxiliary head.
10. Automatically enabling existing calibration would optimize an undefined joint
    objective and produce incompatible artifacts.
11. Selecting lambda from outer-test, combining heads after seeing outcomes, or
    changing the grid after results invalidates the preregistered comparison.
12. The motivating effects are internal to the same 53-subject dataset; the proposed
    experiment is still an internal model comparison, not external validation.
