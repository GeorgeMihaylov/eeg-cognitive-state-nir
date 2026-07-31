# Synthetic First-Order MAML contract

## Scope and status

- Branch: `integration/benchmark-unification`.
- Base HEAD: `b384f7c`.
- Runtime status: `diagnostic`.
- Decision: `infrastructure_only`.
- Device: CPU.
- Real EEG or COG-BCI training: not performed.

This task implements a minimal, model-independent First-Order
Model-Agnostic Meta-Learning (FOMAML) algorithm contract using the episodic
infrastructure introduced in task 8T. It is a synthetic mathematical and
software smoke test, not an EEG experiment and not evidence about model
quality.

## Historical prototype mapping

The historical source was inspected with `git show` at
`feature/benchmarking@8ecbee9:bench/tasks/mixin/metalearning.py`.

| Historical component | Current treatment |
|---|---|
| support/query tasks | retained through `MetaEpisode` and its validated IDs |
| participant adaptation | retained as an episode contract; synthetic entities only here |
| cloned learner | replaced by independent functional fast weights |
| query loss for meta-update | retained |
| `learn2learn.MAML` | rejected; no optional dependency or fallback |
| global `self.data` | rejected; the algorithm receives explicit episode tensors |
| embedded training loop | replaced by importable orchestration and one meta-step API |
| hard-coded ways/shots | replaced by typed configuration |
| random task/window mixing | rejected; synthetic IDs and partitions are deterministic |

## FOMAML rather than full MAML

For every episode the implementation creates an ordered mapping of cloned
parameter tensors, adapts it on support data, computes query gradients with
respect to the final fast weights, maps those gradients by exact name and
shape to the base model, averages them across the meta-batch, clips after
averaging, and performs one explicit Adam meta-optimizer step.

Every inner gradient call uses `create_graph=False`. Each updated fast tensor
is detached before the next step. Therefore query backpropagation cannot
differentiate through the support-gradient operation and no Hessian-vector
product is constructed. This is FOMAML, not Second-Order MAML.

## Functional adaptation and fast weights

The installed environment uses PyTorch `2.11.0+cu128`; the smoke uses its
`torch.func.functional_call` API on CPU. Parameters and buffers are supplied
as separate dictionaries.

Fast-weight invariants:

- names and deterministic order equal `model.named_parameters()`;
- shapes exactly match the base model;
- every tensor has independent storage;
- fast-weight creation and all inner/query operations leave the base
  `state_dict` unchanged;
- unsupported missing, extra, or shape-mismatched parameters fail explicitly;
- support adaptation does not receive query features or labels;
- all losses and gradients must be finite;
- a numerically zero meta-gradient is rejected.

The base model changes only after averaged query gradients have been assigned
and the explicit meta-optimizer step has run.

## Buffer policy

The only accepted configuration is `buffer_policy=frozen`. The synthetic
model contains no BatchNorm, Dropout, or mutable buffers. Production EEGNet
and ShallowConvNet contain BatchNorm running statistics, so real adaptation
is explicitly blocked rather than silently updating either base or copied
statistics.

For production models, a read-only functional forward is executed in eval
mode using copied parameters and buffers. Parameter names, shapes, latent
dimension, head width, output shape, and unchanged state are audited. This
proves functional-call compatibility but not safe real-data adaptation.

## Synthetic episodes

The deterministic generator creates three Gaussian classes in two
dimensions. Each task uses a fixed rotation, scale, shift, and seeded noise:

- support: 5 samples per class, 15 total;
- query: 10 different samples per class, 30 total;
- meta-train episodes: 12;
- meta-validation episodes: 6;
- support and query use distinct sample and record IDs;
- all episodes pass the existing `MetaEpisodeBuilder` and leakage validator.

The model is `Linear(2,16) -> ReLU -> Linear(16,3)` and is not registered in
the production model zoo.

## Inner adaptation and meta-gradient

Configuration was used exactly as preregistered: two inner steps, inner
learning rate 0.1, meta-learning rate 0.01, four episodes per meta-batch, 20
meta-steps, gradient clip norm 5.0, CPU, and seed 42. No hyperparameter search
or alternate seed was run.

Synthetic results:

| Audit | Result |
|---|---:|
| mean support loss before adaptation | 0.511857 |
| mean support loss after two steps | 0.375752 |
| mean meta-train query loss | 0.383325 |
| mean meta-validation query loss | 0.218641 |
| mean meta-gradient norm before clipping | 0.666833 |
| minimum / maximum meta-gradient norm | 0.259201 / 1.080797 |
| parameters updated at every meta-step | 4 / 4 |

All support/query losses, inner gradients, query gradients, optimizer states,
and final parameters were finite. Support adaptation reduced mean loss. The
base model remained unchanged before each meta-step and changed after every
meta-step.

## Leakage and first-order audits

- support/query sample overlap: zero;
- query data is absent from the `adapt` signature;
- changing query features and labels leaves final fast weights identical;
- changing support labels changes fast weights;
- outer project and COG-BCI manifests were hash-protected and unchanged;
- no dataset loader is imported or called;
- the historical mixin, `learn2learn`, and `higher` are not runtime
  dependencies.

## Determinism

The complete smoke is executed twice from the same config, seed, episodes,
and initial state. Episode IDs, every history value, losses, gradient norms,
and model hashes match exactly.

- Initial model hash:
  `f4924857f92346fd2646c0725a93e83b2c97d9d98e956f9d4ea66786d80b3246`.
- Final model hash:
  `38128e34e86cde953e6483a513edc50c81ce9d392ba3e6c7952146c9ed9499ff`.
- Scientific result hash:
  `02ddaf37fb7b357886154f05f0e69609ab302535702205ea914f5cca44821a9b`.

No runtime summary contains timestamps or local absolute paths.

## Production compatibility

| Model | Functional eval | State unchanged | Latent/head | Adaptation status |
|---|---|---|---|---|
| EEGNet | yes | yes | 128 / 3 | blocked by frozen stateful buffers |
| ShallowConvNet | yes | yes | 4 / 3 | blocked by frozen stateful buffers |

The algorithm contract works for the approved buffer-free synthetic model.
Because a scientifically approved BatchNorm policy is still absent, the
overall status remains `infrastructure_only`, not `algorithm_contract_ready`
for production EEG.

## Runtime artifacts

Ignored artifacts are written under
`benchmark_results/meta_learning_fomaml_synthetic/`:

- `resolved_config.json`;
- `synthetic_episode_manifest.json`;
- `initial_model_manifest.json`;
- `meta_training_history.csv`;
- `episode_metrics.parquet`;
- `gradient_audit.csv`;
- `parameter_update_audit.json`;
- `leakage_audit.json`;
- `determinism_audit.json`;
- `final_model_state.pt`;
- `final_model_manifest.json`;
- `smoke_summary.json`;
- `errors.csv`;
- `smoke_report.md`.

## Work required before EEG

A separately approved protocol must define real meta-train,
meta-validation, and outer-test query semantics. A stateful-buffer policy
must decide whether BatchNorm statistics are globally frozen, adapted from
support only, replaced, or handled with another validated normalization
strategy. The choice requires leakage tests, checkpoint/resume semantics,
adapter integration, and a new diagnostic approval. Until then, no real EEG
FOMAML run is permitted.
