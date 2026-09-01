# Ordinal Transformer infrastructure implementation

Дата проверки: 2026-07-18

Ветка: `feature/ordinal-transformer`

Базовый commit проекта: `0cd4570`

## 1. Реализованные классы и функции

Нейтральная порядковая логика вынесена в `model_zoo/DL/ordinal.py`. Модуль
содержит `ClassificationObjectiveHandler`, `CoralOrdinalHead`,
`CornOrdinalHead`, проверяемое кодирование CORAL/CORN targets и risk-set masks,
loss-функции с явными numerator/denominator, преобразование logits в накопленные
и классовые вероятности, threshold decoding и expected rank.

`TorchFeatureTransformerClassifier` поддерживает один канонический тип модели
`torch_transformer` и три значения `head_type`: `categorical` (по умолчанию),
`coral`, `corn`. Метод `encode()` выполняет прежние input projection,
positional encoding, `TransformerEncoder` и pooling. Существующий `LayerNorm`
не переносился в `encode()`: он остаётся `classifier.0` для сохранения прежнего
вычислительного пути и ключей checkpoint. Ordinal heads содержат эквивалентный
неглубокий предголовный блок `LayerNorm -> Linear -> GELU -> Dropout`.

## 2. Структура CORAL

После общего ordinal pre-head одна score-функция вычисляет
`s = w^T h + b`. Четыре logits имеют вид `z_k = s - c_k`. Первый cutpoint
свободен, а остальные параметризованы как
`c_k = c_(k-1) + softplus(delta_k) + 1e-6`. Поэтому cutpoints строго возрастают,
а `sigmoid(z_k)` структурно не возрастает по номеру порога. Начальная точка —
`[-1.5, -0.5, 0.5, 1.5]`; сортировки внутри `forward()` нет. Фактические пороги
доступны через `CoralOrdinalHead.cutpoints()`.

## 3. Структура CORN

После такого же неглубокого pre-head линейный слой выдаёт `K-1=4` условных
logits. Отдельная глубокая CORN-сеть не добавлена. Выход `k` интерпретируется как
logit вероятности `P(y > k | y > k-1)`; для `k=0` risk set включает всю порцию.

## 4. Точная агрегация CORN loss

Для класса `y` используются `t_k = 1[y > k]` и `m_k = 1[y >= k]`.

```text
numerator   = sum_i sum_k m[i,k] * BCEWithLogits(z[i,k], t[i,k])
denominator = sum_i sum_k m[i,k]
L_CORN      = numerator / denominator
```

Порог с пустым risk set не вносит вклад. Полностью нулевой denominator вызывает
контролируемую ошибку, хотя при корректном `k=0` он невозможен. Loss по эпохе
агрегируется теми же numerator/denominator, а не средним от batch means.

CORAL использует невзвешенный BCE sum, делённый на `N*(K-1)`.

## 5. Преобразование вероятностей

Для CORAL `q_k = sigmoid(z_k)`. Для CORN сначала
`r_k = sigmoid(z_k)`, затем `q_k = product_(j<=k) r_j` через `torch.cumprod`.
Из накопленных вероятностей строятся:

```text
p0 = 1-q0; p1 = q0-q1; p2 = q1-q2; p3 = q2-q3; p4 = q3
```

`predict_proba()` для всех heads возвращает только классовые вероятности формы
`[N,5]`; threshold probabilities никогда не передаются в AUC как классовые.

## 6. Численные проверки

Перед преобразованием проверяются shape, floating dtype, конечность, диапазон и
монотонность. Допуск равен `1e-7`, как в проекте 6А. Нарушение больше допуска
вызывает ошибку. Только малые отрицательные разности ограничиваются нулём;
нормализация разрешена лишь при малой ошибке суммы. Проверяются конечность,
неотрицательность и единичная сумма результата. Fold metadata будет хранить
максимальное нарушение монотонности, максимальную ошибку суммы и число реально
исправленных отрицательных round-off разностей.

## 7. Интеграция с адаптером

`TorchClassificationAdapter` получает один `ClassificationObjectiveHandler` и
использует его в существующем train/validation loop. Device, DataLoader,
standardization, validation split, early stopping, best-state restoration и
training log не дублировались. `predict_detailed()` выполняет единый inference и
возвращает classes, raw outputs и доступные ordinal diagnostics.

`predict()` сохраняет categorical argmax, а для CORAL/CORN использует
`count(q_k >= 0.5)`. `predict_proba()` неизменно возвращает `numpy.ndarray [N,5]`.

## 8. Совместимость checkpoints

Categorical `classifier` оставлен на прежнем пути и сохраняет ключи
`classifier.0.*`, `classifier.1.*`, `classifier.4.*`. Config без `head_type`
разрешается как categorical. Реальный checkpoint fold 1 из существующего запуска
загружен через `strict=True`; refactored forward и прежнее ручное вычисление
совпали с `atol=0, rtol=0`, как и softmax вероятности.

Новые checkpoints сохраняют `head_type` и objective metadata. Старый checkpoint
без поля трактуется как categorical. Несовпадение head в любую сторону вызывает
понятную ошибку; автоматического преобразования весов нет.

## 9. Совместимость персональной настройки

Categorical calibration path не изменён и его regression tests проходят.
`UserCalibrationExperiment` и `fine_tune()` явно отклоняют CORAL/CORN до начала
настройки сообщением `Ordinal calibration is not supported yet`.

## 10. Поддерживаемые метрики

Существующие accuracy, balanced accuracy, macro/weighted F1, weighted multiclass
OVR AUC, Cohen kappa, ordinal MAE, adjacent accuracy и severe error rate сохранены.
Добавлены quadratic weighted kappa по primary `y_pred`, expected-rank MAE и
Spearman между `expected_rank` и `y_true`. Expected-rank показатели появляются
только при наличии expected rank; неопределённый Spearman остаётся `NaN`.

## 11. Формат будущих predictions

Стандартный `predictions.parquet` остаётся единственным источником истины.
Ordinal runs добавляют `threshold_logit_0..3`,
`threshold_probability_0..3`, `class_probability_0..4`, `expected_rank`,
`ordinal_argmax`, `head_type` и для CORN `conditional_probability_0..3`.
Существующие `proba_0..4` точно равны class probabilities. Проектное имя
`y_pred_argmax` сохранено как compatibility alias, а каноническое имя задачи 6Б —
`ordinal_argmax`. Categorical artifacts не получают пустых ordinal-колонок.

## 12. Результаты синтетической проверки

Использованы 50 искусственных последовательностей `[50,8,6]`, пять классов,
CPU, seed 42 и ровно один optimizer step на модель. Реальный EEG не читался.

| head | raw output | class proba | parameters | train loss | validation loss | finite | max sum error |
|---|---:|---:|---:|---:|---:|---:|---:|
| categorical | `[8,5]` | `[8,5]` | 853 | 1.669043 | 1.668207 | yes | 1.19e-7 |
| coral | `[8,4]` | `[8,5]` | 821 | 0.564735 | 0.576752 | yes | 1.19e-7 |
| corn | `[8,4]` | `[8,5]` | 844 | 0.646598 | 0.660020 | yes | 5.96e-8 |

Все forward/loss/backward/optimizer/predict/predict_proba результаты конечны.
Это техническая проверка интерфейса, не оценка качества модели.

Тесты: 55 новых ordinal/compatibility тестов прошли; 68 выбранных прежних
Transformer/factory/calibration/runner тестов прошли; полный suite —
`280 passed, 10 warnings`.

## 13. Известные ограничения

- Реальное обучение EEG, GroupKFold, дополнительные seeds и научное сравнение не
  выполнялись.
- Ordinal user calibration пока намеренно не поддержана.
- Class/threshold weighting и regression objective не реализованы.
- CORAL/CORN не калибровались по outer-test и не имеют post-hoc threshold tuning.
- В factory не потребовался отдельный dispatch: существующий общий
  `build_model()` уже передаёт params в `build_torch_transformer`; валидация heads
  добавлена в Transformer builder без новых имён моделей.

## 14. Что осталось для задачи 6В

Нужно создать утверждённые experiment configs, выполнить полный leakage-safe
GroupKFold на неизменном каноническом sequence index, проверить все fold artifacts
и sample identities, затем сравнить categorical/CORAL/CORN парно на одинаковых
folds и только после этого формулировать научные выводы. Отдельно потребуется
решение о дополнительных seeds и ordinal calibration; они не запускаются
автоматически.
