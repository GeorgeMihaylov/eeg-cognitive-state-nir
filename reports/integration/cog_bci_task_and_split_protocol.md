# COG-BCI: целевые переменные и протокол разбиения

## Статус аудита

- Ветка: `integration/benchmark-unification`.
- Исходный HEAD: `5c2d407 feat(data): add COG-BCI window materialization`.
- Исходное рабочее дерево и staging area были чистыми.
- Использован существующий полный cache
  `benchmark_results/cog_bci_windows/emotiv_common_full`.
- SHA-256 входного `window_index.parquet` до и после построения протоколов:
  `d9ec8addfb08e97b2de1e4eade9dc278691929c6869baa417b5b8042045535fa`.
- Исходные записи, shards и индекс окон не изменялись. Модели не обучались.

Изучены существующие dataset/task contracts, реестр задач, runner,
GroupKFold/LOSO и group-aware inner validation, COG-BCI loader, channel
contracts, window materialization, CLI-оболочки и runtime manifests. Новый
слой использует общий deterministic GroupKFold helper из validation layer,
а не отдельную реализацию алгоритма разбиения.

## Рассмотренные научные задачи

В корпусе представлены N-Back, MATB-II, PVT, Flanker и четыре записи покоя.
Каноническими на текущем этапе выбраны две независимые задачи:

1. `cog_bci_nback_3class`: явно заданные уровни zero/one/two back;
2. `cog_bci_matb_3class`: явно заданные easy/medium/difficult conditions.

Они не объединяются в общую «нагрузку»: протоколы экспериментов, значения
условий и возможные смешивающие факторы различаются. PVT и Flanker не
получают искусственных классов сложности. Resting-state отложен до аудита
эффектов eyes open/closed и begin/end. KSS/RSME не объявлены targets из-за
пропусков, отрицательных KSS и пока не подтверждённой привязки к
record/session. Opaque MATLAB tables не декодировались.

## Target layer

### N-Back

| Поле | Значение |
|---|---|
| `task_id` | `cog_bci_nback_3class` |
| `target_name` | `n_back_level` |
| `target_type` | `ordinal_classification` |
| `target_source` | `record.task_variant` |
| `target_level` | `record` |
| `ordered_classes` | `true` |
| Классы | `zero_back → 0`, `one_back → 1`, `two_back → 2` |

### MATB-II

| Поле | Значение |
|---|---|
| `task_id` | `cog_bci_matb_3class` |
| `target_name` | `matb_difficulty` |
| `target_type` | `ordinal_classification` |
| `target_source` | `record.task_variant` |
| `target_level` | `record` |
| `ordered_classes` | `true` |
| Классы | `matb_easy → 0`, `matb_medium → 1`, `matb_difficult → 2` |

Ordinal contract задаёт порядок, но не предполагает равенство численных
расстояний между соседними уровнями. Метка назначается записи и наследуется
только её окнами. Отклонённый неполный хвост сохраняется в target index для
аудита, но не входит в supervised sample.

Target identity включает стабильные hashes схемы и отсортированного индекса.
Неизвестные и исключённые task variants не получают target. `sample_id`,
`subject_id`, `record_id` и `record_group_id` валидируются; дубликаты и
пропуски вызывают ошибку.

## Фактический баланс

Обе задачи содержат 29 участников, 3 сессии и 261 запись: 87 записей каждого
класса. Каждый класс присутствует у каждого участника и в каждой сессии.

### N-Back

| class | records | accepted windows | rejected tails | duration, h | windows/record min–max | mean ± SD |
|---|---:|---:|---:|---:|---:|---:|
| zero_back | 87 | 5 579 | 87 | 7.9346 | 60–80 | 64.126 ± 4.051 |
| one_back | 87 | 5 638 | 87 | 8.0185 | 60–80 | 64.805 ± 4.239 |
| two_back | 87 | 5 710 | 87 | 8.1209 | 61–88 | 65.632 ± 5.047 |

Итого: 16 927 accepted окон и 261 rejected tail. Record-level баланс точный;
на window-level максимальный класс на 131 окно (2.35%) больше минимального.
Это следствие длительности записей, а не изменения target mapping.

### MATB-II

| class | records | accepted windows | rejected tails | duration, h | windows/record |
|---|---:|---:|---:|---:|---:|
| matb_easy | 87 | 5 046 | 87 | 7.1765 | 58 |
| matb_medium | 87 | 5 046 | 87 | 7.1765 | 58 |
| matb_difficult | 87 | 5 046 | 87 | 7.1765 | 58 |

Итого: 15 138 accepted окон и 261 rejected tail. Баланс точный на уровнях
record, subject, session, window и accepted duration. Автоматический
undersampling/oversampling и class weights не применялись. Для будущего
моделирования допустимо отдельно сравнить равный вес окон, записей или
участников, а также record-/subject-balanced sampler, но только внутри
outer-train.

## События и ограничение `record_full`

N-Back содержит 53 186 event rows, MATB-II — 283 428. Для каждой из 261
записи каждой задачи присутствуют boundary и task-end; унифицированных
task-start markers нет. Поэтому текущая сегментация `record_full` остаётся
диагностическим baseline: target описывает условие всей записи, а не
декодированный trial. Высокая и неодинаковая event density MATB отражает
многокомпонентный протокол и может быть смешивающим фактором. Trial-level
targets требуют отдельного семантического аудита событий.

## Внешний, внутренний и LOSO протоколы

Внешний протокол — sklearn-compatible `GroupKFold(n_splits=5)` по
`subject_id`, без shuffle и seed. Каждый участник вместе со всеми сессиями,
записями и окнами встречается в test ровно одного fold.

Для каждого outer fold внутренний split строится заново только из
outer-train: первый fold детерминированного пятифолдового GroupKFold по
`subject_id`. Это подготовленный manifest для model selection/early
stopping; outer-test помечен `outer_test_excluded` и в inner train/validation
не входит.

Также подготовлено 29 LOSO manifests для каждой задачи. Основным первым
baseline остаётся пятифолдовый GroupKFold; LOSO предназначен для более
детальной subject-level оценки.

| Task | protocol hash | outer split hash | inner split hash | LOSO split hash |
|---|---|---|---|---|
| N-Back | `5f7e01bc2dc2967737c6704819d7ef13ac2ba2919b1d4d9a46b36560cea63598` | `5874a0a93bff6f8a504cbc75e15c48588bf12cd51f460eb0f9d16ff94809ac01` | `d84f3853e244f2be47f6ad4431b533a1ec8d6e0b8d41687991d6b2fc32922ef9` | `58d28ad27aaca5d649078175ca4cc25c37955fcd696ec02060d07be7b5a4d51a` |
| MATB-II | `b482c1a8b9ea4561f3e296de870e887997b92e98e7a337094c0f8dd5678fd8b3` | `783b8f0a588fece85ab5dff6970b740d678c1e451b899b020d964da826387dd9` | `00908ef07bb3856ae4d0951a97d0b4ace1387ff1e69db3620acb90374ee793fb` | `d5a236cd42bd5ced4c31d858f157e27aa1741c634302d5202b05ac7546dba8bd` |

Protocol hash зависит от dataset/task/schema, полного target index,
отсортированных subjects/records/sample IDs, config hash существующего
window cache и точной семантики всех splitters.

## Leakage audit и статистический уровень

Во всех 5 outer folds, 5 inner manifests и 29 LOSO folds для обеих задач:

- subject overlap равен нулю;
- `record_id` overlap равен нулю;
- `record_group_id` overlap равен нулю;
- `sample_id` overlap равен нулю;
- outer-test не участвует в inner split;
- target mapping заранее задан схемой и не вычисляется из fold;
- scaler, preprocessing selection, class weights и sampler не создаются;
- обучение не запускается.

Окна одной записи являются зависимыми наблюдениями. Window-level метрики
могут быть диагностическими, но основной уровень статистических выводов и
интервалов должен быть subject или record. Manifest сохраняет идентификаторы
для агрегации на всех трёх уровнях: window, record, subject.

## Межнаборный статус

- Channel contract COG-BCI ↔ Emotiv: ready.
- Window shape contract `[14, 2560]`: ready.
- Shared supervised target: **not available**.

`n_back_level` и `matb_difficulty` нельзя автоматически сопоставлять с
`label_q5` или `target_focus`. COG-BCI пригоден для самостоятельных native
задач и потенциального self-supervised/contrastive encoder pretraining.
DANN не является готовым научным протоколом: не определены общая target
семантика, направление source/target и финальный evaluation domain.

## Рекомендуемый первый baseline

Рекомендуется `cog_bci_nback_3class`. MATB-II технически имеет идеальный
баланс длительностей и окон, но N-Back задаёт более прозрачную и
интерпретируемую ordinal манипуляцию zero/one/two back. Небольшой
window-level дисбаланс N-Back документирован и не требует sampling до
наблюдения реального train-only поведения. MATB-II следует оставить вторым
native baseline и отдельно учитывать многокомпонентность и event density.

Перед первым обучением нужно подключить target index и split assignments к
общему dataset/runner contract, определить train-only normalization и
основные subject-/record-level metrics, выполнить один CPU smoke на первом
outer fold и проверить стандартные prediction/split/preprocessing
артефакты. Текущий этап не содержит обучения или выбора модели.
