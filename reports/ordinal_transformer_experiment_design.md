# Ordinal Transformer experiment design

Дата проекта: 2026-07-18. Этот документ определяет будущую реализацию и эксперимент;
модели в задаче 6А не обучались.

## Scientific motivation

`label_q5` — пять упорядоченных уровней, полученных квантованием `target_focus`.
Соседние классы ближе друг к другу, чем крайние. Временная диагностика показала
сохранение класса 58.40%, соседний переход 34.48% и переход на два или более класса
7.12%. Категориальный Transformer не использует порядок непосредственно.

Опубликованный seed-42 categorical baseline:

| Feature group | Balanced accuracy | Macro F1 | Ordinal MAE | Severe error |
| --- | ---: | ---: | ---: | ---: |
| EEG-only | 0.3456 +/- 0.0232 | 0.3403 +/- 0.0210 | 1.0565 | 0.2784 |
| EEG+POW | 0.3687 +/- 0.0189 | 0.3615 +/- 0.0169 | 0.9838 | 0.2513 |

Ordinal head проверяет, можно ли уменьшить расстояние и долю тяжёлых ошибок без
неприемлемой потери разделения пяти классов. Это отдельная проверка target formulation,
не замена leakage-safe label sensitivity и не регрессия.

## Research questions

1. Уменьшают ли CORAL или CORN ordinal MAE и severe error относительно categorical
   Transformer на тех же последовательностях и folds?
2. Сохраняются ли balanced accuracy и macro F1?
3. Согласуется ли направление эффекта между EEG-only и EEG+POW?
4. Даёт ли условная постановка CORN преимущество над rank-consistent CORAL?
5. Улучшаются ли QWK и continuous expected-rank diagnostics?

## Categorical baseline

Baseline не переобучается: используются существующие seed-42 runs из
`feature_group_transformer_ablation`:

```text
EEG-only:
benchmark_results/feature_group_transformer_ablation/runs/
  transformer_classification_eeg_only/20260718_124023

EEG+POW:
benchmark_results/feature_group_transformer_ablation/runs/
  transformer_classification_eeg_pow/20260718_124412
```

Архитектура и optimization остаются: sequence length 8, `d_model=128`, 4 heads,
2 layers, FF=256, GELU, dropout=0.1, last pooling, batch 128, max 15 epochs,
AdamW lr 0.001, weight decay 0.0001, patience 4, train-only standardization,
device auto, seed 42. Ordinal runs различаются только head/loss semantics.

## CORAL formulation

Для `K=5` классов строятся `K-1=4` cumulative targets:

```text
t[i,k] = 1[y_i > k],  k = 0,1,2,3

y=0 -> [0,0,0,0]
y=1 -> [1,0,0,0]
y=2 -> [1,1,0,0]
y=3 -> [1,1,1,0]
y=4 -> [1,1,1,1]
```

После общего Transformer encoder и предголовного блока получается representation
`h_i`. CORAL head использует одну общую score-функцию и четыре упорядоченных cutpoints:

```text
s_i = w^T h_i
c_0 = a
c_k = c_(k-1) + softplus(delta_k) + epsilon, k=1..3
z[i,k] = s_i - c_k
q[i,k] = sigmoid(z[i,k]) = P(y_i > k)
```

`epsilon=1e-6`. Поскольку cutpoints строго возрастают, logits и probabilities
не возрастают по `k`: `q_0 >= q_1 >= q_2 >= q_3`. Это структурная гарантия, а не
post-hoc сортировка. Инициализация воспроизводима и не использует labels: cutpoints
`[-1.5,-0.5,0.5,1.5]`, что задаётся через `a=-1.5` и inverse-softplus unit increments;
score weights инициализируются стандартным seed-controlled PyTorch способом.

Основной loss невзвешенный:

```text
L_CORAL = sum_i sum_k BCEWithLogits(z[i,k], t[i,k]) / (N * 4)
```

Никакие class/threshold weights в первом полном запуске не используются. Если они
будут исследоваться позже, вычислять их можно только по inner-train каждого outer fold.

## CORN formulation

CORN выдаёт четыре условных logits:

```text
r[i,k] = sigmoid(z[i,k]) = P(y_i > k | y_i > k-1)
```

Для `k=0` условие считается истинным для всех наблюдений. Маски и цели:

```text
m[i,0] = 1
m[i,k] = 1[y_i > k-1], k=1..3
t[i,k] = 1[y_i > k]
```

То есть threshold `k` обучается только на объектах, дошедших до соответствующего
условного risk set. Невзвешенный loss:

```text
numerator   = sum_i sum_k m[i,k] * BCEWithLogits(z[i,k], t[i,k])
denominator = sum_i sum_k m[i,k]
L_CORN      = numerator / denominator
```

Это определение используется и для batch optimization, и для точного epoch/validation
aggregation. Пустой risk set отдельного верхнего threshold не вызывает BCE над пустым
тензором: elementwise loss умножается на mask, а threshold с нулевой mask даёт ноль в
числитель и denominator. Общий denominator всегда положителен, потому что `m[:,0]=1`.

Условные вероятности превращаются в cumulative:

```text
q[i,0] = r[i,0]
q[i,k] = product(r[i,0:k+1]), k=1..3
```

Так `q` структурно монотонны без ограничений на четыре conditional logits.

## Probability conversion

Для обоих методов `q_k=P(y>k)`. Вероятности пяти классов:

```text
p_0 = 1 - q_0
p_1 = q_0 - q_1
p_2 = q_1 - q_2
p_3 = q_2 - q_3
p_4 = q_3
```

Алгоритм численной проверки фиксируется заранее:

1. Проверить finite, диапазон `q` с tolerance `1e-7` и монотонность
   `q_k + 1e-7 >= q_(k+1)`.
2. При нарушении больше tolerance завершить inference ошибкой: это defect модели,
   checkpoint или conversion, а не случай для молчаливой коррекции.
3. Сформировать `p` разностями. Только значения в `[-1e-7,0)` считать round-off и
   заменить нулём; значение ниже `-1e-7` является ошибкой.
4. Проверить положительную finite row sum и нормировать `p` на неё. Нормировка удаляет
   только накопленную машинную погрешность.
5. Сохранить исходные, не скорректированные cumulative `q` в артефактах и записать
   число round-off corrections в `ordinal_metadata.json`.

`predict_proba` всегда возвращает именно `p` формы `[N,5]`. AUC рассчитывается только
по этим class probabilities; threshold probabilities для AUC не используются.

## Prediction rule

До просмотра outer-test результатов основным правилом фиксируется:

```text
y_pred = sum_k 1[q_k >= 0.5]
```

Равенство 0.5 относится к положительной стороне (`>=`). Результат всегда в `[0,4]`.
Дополнительно:

```text
expected_rank = sum_k q_k
y_pred_argmax = argmax_j p_j
```

`expected_rank` лежит в `[0,4]` и используется только для continuous diagnostics.
`y_pred_argmax` сохраняется как secondary diagnostic, но не заменяет primary `y_pred`
после просмотра результатов.

## Metrics

Основные categorical metrics:

- balanced accuracy;
- macro F1;
- weighted multiclass OVR AUC по `[N,5]` class probabilities;
- Cohen's kappa.

Основные ordinal metrics:

- ordinal MAE: `mean(abs(y_pred-y_true))`;
- adjacent accuracy: `mean(abs(error)<=1)`;
- severe error rate: `mean(abs(error)>=2)`;
- quadratic weighted kappa: `cohen_kappa_score(..., labels=0..4,
  weights="quadratic")`.

Дополнительные:

- expected-rank MAE: `mean(abs(expected_rank-y_true))`;
- Spearman correlation между expected rank и `y_true`; если одна переменная константна,
  результат `NaN` с явной причиной, без подстановки нуля.

Главные показатели решения: ordinal MAE, severe error rate, balanced accuracy и macro
F1. Метрики считаются per fold и pooled descriptively; statistical inference использует
subject-level значения, а не окна как независимые наблюдения.

## Evaluation protocol

Протокол полностью совпадает с categorical baseline:

```text
dataset: data/processed/windowed_eeg_pm_dataset_w10.parquet
target: label_q5, classes 0..4
supervised windows: 45 384
sequences: 44 142
sequence length/stride: 8 / 1
sequence subjects: 53 (из 54 supervised)
sequence-index SHA-256:
  1d0a1fe8ab8aad1c6da8637cb882c4365c01acb80c0822e34ced21ef7ee36afa
outer: GroupKFold(n_splits=5), group=subject_id, shuffle=False
inner: group_record, group=record_group_id, validation_size=0.15
normalization: inner-train only
seed: 42
```

Перед сравнением каждый ordinal run обязан иметь точное совпадение по
`sequence_id`, `fold`, `subject_id`, `record_id`, `source`, `target_sample_id`,
`target_time` и `y_true` с соответствующим categorical run. В каждом outer fold
subject overlap равен нулю; inner record-group overlap равен нулю. Одно наблюдение
должно иметь ровно одно outer-test prediction.

## Feature groups

Первое полное исследование включает только:

| Group | Features | Count | Ordered feature SHA-256 |
| --- | --- | ---: | --- |
| EEG-only | EEG aggregate features | 168 | `6e822ee172422e7138945b47b2b27c947393b828b72d96b7a8e22850aded8aca` |
| EEG+POW | EEG and POW | 448 | `8cd5d70faa8ff30fb4290dd9d9a2dde0e81f50e7682d05668b5fb47df511fd51` |

POW-only, LSTM/BiLSTM, regression и joint objectives не входят в эту матрицу.

## Experimental matrix

| Run | Feature group | Head | Loss | Status | Folds | Seed |
| --- | --- | --- | --- | --- | ---: | ---: |
| categorical_eeg_only | EEG-only | categorical, 5 logits | CE | existing | 5 | 42 |
| coral_eeg_only | EEG-only | CORAL, 4 threshold logits | unweighted CORAL | future | 5 | 42 |
| corn_eeg_only | EEG-only | CORN, 4 conditional logits | unweighted CORN | future | 5 | 42 |
| categorical_eeg_pow | EEG+POW | categorical, 5 logits | CE | existing | 5 | 42 |
| coral_eeg_pow | EEG+POW | CORAL, 4 threshold logits | unweighted CORAL | future | 5 | 42 |
| corn_eeg_pow | EEG+POW | CORN, 4 conditional logits | unweighted CORN | future | 5 | 42 |

Таким образом, задача 6Г потребует четыре новых полных запуска. Architecture,
optimization, folds, sequence cohort и feature lists неизменны; отдельный подбор
гиперпараметров CORAL/CORN не выполняется.

## Statistical analysis

Главная единица — `subject_id` (53 парных наблюдения при полном наличии predictions).
Для каждой пары методов внутри feature group рассчитываются subject-level differences,
paired subject bootstrap с 10 000 resamples и 95% CI, Wilcoxon signed-rank, exact/asymptotic
sign test по применимому правилу, rank-biserial correlation и доли improved/degraded/ties.

Семьи Holm correction разделены заранее:

- EEG-only methods: CORAL vs categorical, CORN vs categorical, CORAL vs CORN;
- EEG+POW methods: те же три сравнения.

Каждая из двух confirmatory families включает 12 гипотез: три method contrasts на
четырёх decision-primary metrics (ordinal MAE, severe error, balanced accuracy и macro
F1). Holm correction применяется внутри этих 12 гипотез и не объединяет EEG-only и
EEG+POW в одну искусственно большую семью. QWK, AUC, kappa и expected-rank diagnostics
на первом этапе являются поддерживающими; их p-values, если показаны, явно помечаются
как exploratory. Анализ source остаётся описательным из-за пересечения людей между
источниками.

## Success criteria

Один жёсткий порог не используется. Метод считается перспективным, если выполнены хотя
бы два условия из следующих семи и нет явного противоположного сигнала в paired CI:

1. уменьшается ordinal MAE;
2. уменьшается severe error rate;
3. растёт QWK;
4. balanced accuracy не демонстрирует убедительного ухудшения;
5. macro F1 не демонстрирует убедительного ухудшения;
6. улучшается нижний квартиль subject-level результата;
7. направление эффекта совпадает для EEG-only и EEG+POW.

Выражение «не демонстрирует убедительного ухудшения» оценивается по paired estimate и
95% CI, а не по произвольной заранее назначенной дельте. Статистическая значимость не
заявляется только по fold means.

## Artifact schema

Сохраняются стандартные `model.pt`, `training_log.csv`, `metrics.json`,
`class_metrics.json`, `validation_split.json`, normalization/feature/sequence manifests,
fold `predictions.parquet`, unified `predictions.parquet`, benchmark JSON и summary CSV.

Ordinal `predictions.parquet` сохраняет все canonical columns, включая:

```text
sequence_id, fold, subject_id, record_id, source, y_true, y_pred
```

и добавляет:

```text
threshold_logit_0 ... threshold_logit_3
threshold_probability_0 ... threshold_probability_3   # cumulative q, не CORN r
class_probability_0 ... class_probability_4
proba_0 ... proba_4                                    # compatibility aliases
expected_rank
y_pred_argmax
head_type
```

Для CORN raw conditional probabilities `r` не смешиваются с cumulative q: они либо
сохраняются в отдельных `conditional_probability_0..3`, либо не пишутся в основной
Parquet. Рекомендуется сохранить их для диагностики. `proba_k` обязаны быть точно
равны `class_probability_k`, чтобы текущий AUC/analysis code продолжал работать.

Отдельный компактный `ordinal_metadata.json` на fold хранит method, target encoding,
loss formula/normalization, class/threshold count, prediction rule, tolerance,
round-off correction count и monotonicity audit. Per-batch tensors и duplicated
sequence metadata в отдельные файлы не записываются.

Checkpoint metadata дополнительно содержит `head_type`, `num_thresholds=4`,
`output_semantics`, `probability_conversion_version` и `prediction_rule`. Categorical
checkpoint format остаётся допустимым без этих новых полей.

## Test plan

Минимальный набор задачи 6Б/6В состоит из 25 проверок:

1. cumulative targets для labels 0..4 совпадают с заданной матрицей;
2. неверные labels, shape и non-integer values отклоняются;
3. CORN masks для labels 0..4 построены точно;
4. CORAL forward имеет shape `[B,4]`;
5. CORN forward имеет shape `[B,4]`;
6. categorical default по-прежнему имеет shape `[B,5]`;
7. CORAL cutpoints строго возрастают;
8. CORAL cumulative probabilities finite и монотонны;
9. CORN conditional probabilities finite и находятся в `[0,1]`;
10. CORN cumulative products finite и монотонны;
11. class probabilities неотрицательны в пределах tolerance;
12. class probabilities finite и sum-to-one;
13. monotonicity violation больше tolerance вызывает ошибку, а не silent clip;
14. round-off correction применяется только в `[-1e-7,0)` и фиксируется;
15. классы 0 и 4 кодируются/декодируются корректно;
16. threshold prediction с `>=0.5` детерминирован, включая tie;
17. expected rank находится в `[0,4]` и равен сумме q;
18. CORAL loss совпадает с вручную рассчитанным synthetic примером;
19. CORN loss/masks/denominator совпадают с ручным примером;
20. пустой верхний CORN risk set не создаёт NaN/Inf;
21. factory создаёт categorical/CORAL/CORN, default categorical и проверяет
    `num_classes == num_outputs`;
22. реальный legacy categorical checkpoint загружается strict и даёт те же logits,
    predictions и probabilities;
23. categorical head-only calibration по `classifier.*` продолжает работать, ordinal
    calibration до поддержки отклоняется явно;
24. runner smoke сохраняет standard и ordinal artifacts, `predict_proba` имеет `[N,5]`,
    а AUC получает class, не threshold probabilities;
25. sequence IDs/folds не меняются; `.gitignore` и исходный Parquet не изменяются.

Дополнительно в regression suite следует оставить все существующие adapter, factory,
Transformer, runner и calibration tests.

## Implementation stages

### Task 6Б — infrastructure

Реализовать targets, handlers, losses, heads, probability conversion, adapter/factory,
metrics/artifacts и unit/integration tests. Полное обучение не запускать. Legacy
checkpoint equivalence является release blocker.

### Task 6В — technical smoke

Один fold, не более 2 000 EEG-only sequences, 3 epochs, отдельно CORAL и CORN.
Проверить finite losses/probabilities, schema, strict alignment и отсутствие leakage.

### Task 6Г — full seed-42 experiment

Четыре новых runs: CORAL/CORN x EEG-only/EEG+POW, 5 folds, canonical 44 142 sequences.
Существующие categorical runs не переобучать.

### Task 6Д — analysis and decision

Выполнить subject-level paired analysis и выбрать: оставить лучший ordinal approach,
перейти к joint regression, исследовать subject-robust training либо остановить
ordinal направление. Дополнительные seeds назначать после seed-42 решения, не в задаче
6А.
