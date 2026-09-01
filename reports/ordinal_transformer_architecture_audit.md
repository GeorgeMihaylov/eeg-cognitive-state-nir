# Ordinal Transformer architecture audit

Дата аудита: 2026-07-18. Исходный commit: `d234f71de345620b27ff097a85c1eef69c3e7b46`.
Аудит выполнен без изменения кода, обучения моделей и записи benchmark-артефактов.

## Existing Transformer implementation

Каноническая модель реализована классом `TorchFeatureTransformerClassifier` в
`model_zoo/DL/transformer.py`. Она принимает тензор `[B, T, F]`, где для опубликованных
экспериментов `T=8`, а `F=168` для EEG-only и `F=448` для EEG+POW. Фактический путь:

```text
[B,T,F]
  -> input_projection: Linear(F, 128)
  -> learned positional_encoding: [1, 8, 128]
  -> 2 x TransformerEncoderLayer
       nhead=4, dim_feedforward=256, GELU, dropout=0.1,
       batch_first=True, norm_first=True
  -> last-token pooling
  -> classifier:
       LayerNorm(128)
       Linear(128,128)
       GELU
       Dropout(0.1)
       Linear(128,5)
  -> 5 categorical logits
```

Модель также поддерживает `mean` и `cls` pooling, learned или sinusoidal positions и
padding mask. Проверяются размерность входа, ширина признаков, максимальная длина и
наличие хотя бы одного непустого token. Encoder и pooling уже отделимы логически;
минимальная безопасная точка расширения — вынести существующую часть forward до
`pooled` в метод `encode`, не переименовывая зарегистрированные модули.

Фабрика `build_model` знает единственный тип `torch_transformer`, строит его после
загрузки данных по фактическому `input_shape`, а логическое число выходов получает как
`task.n_classes`. Для `cognitive_load_5class` это 5. Отдельного ordinal task type сейчас
нет и для этого эксперимента он не требуется: target остаётся пятиуровневой
classification-задачей, меняется только output semantics модели.

## Existing adapter and training path

`TorchClassificationAdapter` является общей sklearn-подобной оболочкой для всех
PyTorch-классификаторов. Он уже централизует:

- проверку shapes и NaN/Inf, NumPy/lazy inputs и labels;
- CPU/CUDA `device: auto`, seed, DataLoader и AdamW;
- inner validation, в том числе `group_record` по `record_group_id`;
- standardization только по inner-train;
- early stopping и восстановление лучшего состояния;
- training log, save/load, clone и fine-tuning.

Сейчас семантика categorical зашита в трёх местах:

1. `fit` и `fine_tune` создают `nn.CrossEntropyLoss`;
2. validation accuracy использует `logits.argmax(dim=1)`;
3. `predict_proba` применяет softmax, а `predict` берёт argmax.

Создание отдельного ordinal adapter скопировало бы почти весь training loop и создало
бы риск расхождения validation split, normalization, stopping и checkpoint logic.
Рекомендуется расширить существующий adapter небольшим objective/output handler с
categorical реализацией по умолчанию.

Handler должен предоставлять как минимум:

```text
loss_parts(logits, labels) -> numerator, denominator
class_probabilities(logits) -> [N, 5]
primary_predictions(logits) -> [N]
prediction_outputs(logits) -> logits/probabilities/expected_rank/head_type
```

`loss_parts` нужен вместо одного scalar loss: для categorical denominator равен `N`,
для CORAL — `N * 4`, для CORN — числу допустимых условных наблюдений. Optimizer получает
`numerator / denominator` текущего batch, а epoch/validation loss считается как сумма
числителей, делённая на сумму знаменателей. Так logging и early stopping не зависят от
состава mini-batches.

## Checkpoint compatibility

Проверен реальный checkpoint fold 1 из
`benchmark_results/groupkfold_torch_transformer_label_q5/20260716_191246`. Payload
содержит:

```text
feature_mean
feature_scale
input_shape = (8, 448)
model_metadata
model_state_dict
num_classes = 5
training_config
training_log
training_summary
validation_split
```

Transformer metadata не содержит `head_type`. Значимые ключи state dict:

```text
input_projection.weight/bias
positional_encoding.encoding
encoder.layers.0.*
encoder.layers.1.*
classifier.0.weight/bias
classifier.1.weight/bias
classifier.4.weight/bias
```

Последний слой имеет shapes `(5, 128)` и `(5,)`. `load` проверяет `input_shape` и
`num_classes`, затем вызывает `load_state_dict(..., strict=True)`.

Правило совместимости для реализации:

- отсутствующий `head_type` в YAML или checkpoint означает `categorical`;
- categorical-модуль и все перечисленные имена параметров остаются без изменений;
- для categorical по-прежнему создаётся `self.classifier` с тем же `nn.Sequential`;
- ordinal-параметры регистрируются под новым префиксом `ordinal_head.`;
- новый checkpoint явно сохраняет `head_type`, `num_classes=5`,
  `num_thresholds=4`, loss normalization и prediction rule;
- загрузчик до strict load проверяет соответствие head semantics; старый checkpoint
  получает только обратносуместимый default, без переписывания payload.

Это позволяет старым checkpoints загрузиться в фабрично созданную categorical-модель
с теми же ключами. Загрузка categorical weights в ordinal head или наоборот должна
завершаться понятной ошибкой до `load_state_dict`.

## Calibration compatibility

Пользовательская настройка строит модель через factory по `input_shape` и
`num_classes` checkpoint, затем вызывает adapter `load`. Режим `head_only` жёстко
передаёт `trainable_parameter_prefixes=("classifier.",)`, а тесты проверяют, что
изменяются только параметры с этим префиксом. Fine-tuning также использует
CrossEntropyLoss и argmax accuracy.

Поэтому на этапе 6Б:

- categorical calibration должна остаться байт-совместимой по именам и поведению;
- ordinal checkpoints нельзя молча передавать текущему calibration pipeline;
- до отдельного ordinal-calibration дизайна следует явно отклонять `head_type !=
  categorical` в `UserCalibrationExperiment`;
- будущий общий интерфейс `head_parameter_prefixes()` может вернуть
  `("classifier.",)` или `("ordinal_head.",)`, но его использование для ordinal
  fine-tuning не входит в первый эксперимент.

## Reusable components

Без изменения научного протокола переиспользуются:

- input projection, positional encoding, TransformerEncoder и pooling;
- model factory и позднее определение `(sequence_length, n_features)`;
- `CognitiveLoad5ClassTask` и `label_q5`;
- gap-aware sequence builder и canonical `sequence_id`;
- 5-fold GroupKFold по `subject_id`;
- inner `group_record` validation и train-only normalization;
- DataLoader, AdamW, early stopping, best-state restoration и save/load;
- runner artifact directories, unified predictions и fold artifacts;
- categorical metrics, feature-group alignment и subject-level paired utilities;
- CLI seed/fold overrides и существующие categorical baseline runs.

Фактическая baseline-выборка: 44 142 последовательности длины 8, 53 испытуемых,
sequence-index SHA-256
`1d0a1fe8ab8aad1c6da8637cb882c4365c01acb80c0822e34ced21ef7ee36afa`.

## Required extension points

Минимальный будущий patch задачи 6Б должен затронуть следующие точки:

1. `model_zoo/DL/ordinal.py`: cumulative targets, CORN masks, handlers, losses,
   probability conversion и ordinal heads.
2. `model_zoo/DL/transformer.py`: общий `encode`; default categorical branch без
   изменения module names; `ordinal_head` для CORAL/CORN.
3. `model_zoo/DL/adapter.py`: objective handler вместо жёстких CE/softmax/argmax;
   один training loop; `predict_outputs` для дополнительных артефактов.
4. `model_zoo/factory.py` и `model_zoo/DL/__init__.py`: передача/экспорт `head_type`;
   model type остаётся `torch_transformer`.
5. `bench/validation/metrics.py`: QWK и continuous expected-rank diagnostics.
6. `bench/bench_runner.py`: передача class probabilities в AUC и сохранение
   дополнительных ordinal prediction fields через необязательный интерфейс.
7. Тесты adapter/factory/checkpoint/calibration/runner; отдельные experiment configs
   появляются только на smoke/full этапах.

Task registry менять семантически не нужно. `num_classes` должен по-прежнему приходить
из `task.n_classes`; дублирующий `model.params.num_classes`, если разрешить его для
читаемости, обязан либо отсутствовать, либо строго совпадать с `num_outputs`.

## Components that must not be duplicated

Не должны появляться второй TransformerEncoder, второй PyTorch training loop, отдельная
реализация split/normalization/early stopping, ordinal-копия factory или собственный
runner. Также нельзя повторно строить sequence cohort: ordinal и categorical варианты
сравниваются только на идентичных `sequence_id`, fold, subject, record, source и
`y_true`.

## Risks

1. Переименование `classifier.*` сломает strict checkpoint load и head-only calibration.
2. Softmax над четырьмя threshold logits даст неверные class probabilities и AUC.
3. Argmax threshold logits не является ordinal prediction rule.
4. Четыре независимых binary outputs без structural constraint нельзя называть CORAL.
5. Неверная CORN mask или усреднение по batch size сместит loss к нижним порогам.
6. Разности немонотонных `q` создают отрицательные class probabilities; молчаливый clip
   скроет ошибку модели.
7. Передача threshold probabilities в `roc_auc_score` нарушит ожидаемую форму `[N,5]`.
8. Весовые коэффициенты, рассчитанные по outer-test, создадут leakage. Основной запуск
   должен быть невзвешенным.
9. Изменение sequence builder, folds или normalization уничтожит парность с baseline.
10. Автоматическое включение ordinal checkpoints в текущую calibration ветку применит
    неверные CE/argmax и неверный prefix.

## Recommended architecture

Оставить один `TorchFeatureTransformerClassifier` с `head_type` из множества
`categorical | coral | corn`, default `categorical`. Метод `encode` возвращает pooled
representation. Categorical branch использует неизменный `classifier`; ordinal branch
использует `ordinal_head`, но тот же encoder и одинаковый предголовной блок
`LayerNorm -> Linear -> GELU -> Dropout`.

Adapter получает сериализуемый objective handler, выбранный builder по `head_type`.
Handler инкапсулирует только различающиеся loss/output semantics, а весь процесс
обучения остаётся общим. `predict_proba` всегда означает вероятности пяти классов
`[N,5]`; расширенный `predict_outputs` возвращает threshold logits, cumulative
threshold probabilities, class probabilities, expected rank и head type. Такой
контракт минимален, сохраняет текущий benchmark API и не смешивает категориальные,
пороговые и условные вероятности.
