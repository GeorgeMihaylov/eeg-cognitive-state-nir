# Полный аудит реестра целевых переменных

Статус решения: **target_registry_ready**. Аудит является read-only: модели не обучались, новые классовые цели и кэши не материализовались.

## 1. Причина аудита

Существующий benchmark исторически называл Focus-derived `label_q5` общей целью когнитивного состояния. Реестр разделяет непрерывные PM устройства, нативные activity-прокси, производные порядковые прокси и legacy aliases.

## 2. Фактическая схема датасета

Канонический Parquet содержит 51,308 окон и 508 столбцов. Признаковые группы: 168 EEG, 280 POW, 448 EEG+POW; хэши списков сохранены в YAML. Все `target_*`, `PM.*`, идентификаторы и временные поля исключены из признаков.

## 3. Семь основных PM

Канонический порядок: Attention, Engagement, Excitement, Stress, Relaxation, Interest, Focus. Для каждой метрики исходная схема обоих источников содержит Raw, Scaled, Min, Max и IsActive. Текущая непрерывная цель — среднее Scaled по 10-секундному окну.

## 4. PM.LongTermExcitement

Поле присутствует во всех 120 исходных записях и обоих источниках, но не выбрано текущим window builder и отсутствует в processed Parquet. Его связь с Excitement, шкала, пропуски и межисточниковая семантика не подтверждены; это отдельный кандидат, не восьмой канонический output.

## 5. Происхождение target_*

`bench/datasets/emotiv_pm_window_builder.py::read_and_aggregate_record` агрегирует `PM.<Metric>.Scaled` функциями mean/std/min/max/last и затем точно копирует `Scaled__mean` в `target_<metric>`. Для всех семи пар, обоих источников и масок пропусков найдено ноль расхождений; максимальная абсолютная разность равна 0.

## 6. Происхождение target_main

В каноническом Parquet `target_main` полностью совпадает с `target_focus` и `PM.Focus.Scaled__mean`: 45 384 конечных строки, ноль расхождений. В builder это условный legacy alias: при отсутствии столбца Focus выбирается Attention, затем Engagement. Loader по-прежнему использует `target_main` по умолчанию, поэтому канонические configs обязаны задавать цель явно.

## 7. Происхождение label_q5

`make_quality_labels` применяет один глобальный `pd.qcut(target_main, q=5, labels=False, duplicates='drop')` после объединения всех записей и до outer split. Полные границы bins: `[0.004077, 0.330177, 0.387786, 0.4444580000000001, 0.526585, 0.991193]`. Непустых меток: 45,384, участников: 54, source records: 119; классы 0–4. Реконструкция byte-for-value совпала со столбцом. Каноническое отображаемое имя — `label_focus_q5`, физический столбец остаётся `label_q5`.

## 8. Нативные индикаторы IsActive

Builder нормализует bool-like значения в 0/1, числовые значения сохраняет, затем берёт оконное среднее. Во всём processed Parquet каждый из семи `IsActive__mean` имеет только значения 0/1, промежуточных окон нет. Это эмпирический факт об агрегате, а не исчерпывающая повторная проверка 129 млн исходных CSV-строк; семантика флага остаётся кандидатом на валидацию.

## 9. Производные прокси-кандидаты

Для каждой непрерывной PM зарегистрированы q3/q5-кандидаты, но они не материализованы. Любые будущие границы должны оцениваться только на outer-train. Существующий глобальный Focus q5 сохраняется как legacy benchmark и sensitivity analysis, а не как шаблон для новых целей.

## 10. Multi-label постановка

Совместный вектор семи IsActive возможен технически для 51,124 окон. Распределение числа активных метрик: `{"0":141,"1":2702,"2":2857,"3":40,"6":2210,"7":43174}`. До подтверждения device semantics, зависимости меток и missing-label loss это только кандидат.

## 11. Доступность по источникам

Все семь исходных PM-семейств и LongTermExcitement перечислены в validated-columns для `gpn_data` и `Old_EEG`. Фактические window counts, missing rates, описательная статистика и квантили по каждому источнику сохранены в `target_availability_by_source.csv`.

## 12. Target-specific когорты

Размеры непрерывных когорт различаются. Семивыходная complete-case когорта: 43,174 окон, 53 участника, 117 source records. Отдельные target-, source-, activity-, label- и raw-deduplicated-compatible когорты находятся в `target_cohort_counts.csv`.

## 13. Входные представления

Feature mode поддерживает EEG (168), POW (280) и EEG+POW (448). Feature-sequence используется существующими temporal моделями. Raw mode имеет контракт `[1, 14, 2560]` при 256 Гц и 10 с. Признаки определены по схеме и хэшированы без чтения всех 508 столбцов.

## 14. Текущее покрытие задачами

Семь PM имеют совместную feature-based регрессию и персонализацию; это не заменяет отдельные научные результаты по каждому proxy. `label_q5` покрыт feature, sequence, raw, personalization, FOMAML и DANN только как Focus-derived цель. Activity, q3/q5-кандидаты и LongTermExcitement не обучались. Матрица находится в `target_task_coverage.csv`.

## 15. Ограничения raw loader

`RawEEGWindowDataset.load` жёстко принимает только `target_col == 'label_q5'`; raw index builder также присваивает folds по `label_q5`. Это документированное ограничение, не исправленное в данном аудите.

## 16. Риски утечек

Главный методический риск — глобальные границы label_q5 до subject-disjoint outer split. Также опасен неявный target_main fallback. Feature contamination сейчас контролируется: EEG/POW-списки не содержат PM или target columns. Полный реестр мер — `target_leakage_risk.csv`.

## 17. Канонический реестр

`reports/summary/target_registry.yaml` является машинно-читаемым источником истины. Он содержит обязательные provenance, semantics, input, task, risk, status и limitation поля для каждой цели и кандидата.

## 18. Рекомендуемая матрица будущих экспериментов

Этапы: (1) обобщить target specification/loaders; (2) поддержать отдельную регрессию семи PM; (3) валидировать IsActive; (4) добавить fold-fitted ordinal proxies; (5) запустить EEG/POW/EEG+POW baselines; (6) обобщить raw loader; (7) выбрать научно обоснованные deep targets; (8) лишь затем решать вопрос повторов DANN/FOMAML.

## 19. Какие существующие результаты сохраняются

Инфраструктура benchmark, subject-disjoint splits, logical-record deduplication, feature/raw caches, модели, метрики и artifact pipeline сохраняются. Результаты label_q5 сохраняются как результаты Focus; семивыходная PM-регрессия сохраняется. FOMAML и DANN не обобщаются на остальные цели.

## 20. Открытые вопросы

Нужны внешняя семантическая валидация IsActive и LongTermExcitement, решение о missing-label loss для multi-label, научное обоснование отдельных ordinal targets и безопасное расширение raw manifest/loader. Никакая из этих задач не была молча объявлена готовой к обучению.
