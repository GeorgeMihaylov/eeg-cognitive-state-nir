# Карта соответствия требованиям проекта

## 1. Источники требований

Основной источник: `README.md` (`project_plan`). Статус источников: `project_plan_only`.

- Официальный файл технического задания не найден в tracked-дереве или Git history.
- README фиксирует цели проекта, но не задаёт формальную обязательность и приоритет всех deliverables.
- AGENTS.md и PROJECT_CONTEXT.md используются только как внутренние планы, а не как официальный текст ТЗ.

## 2. Метод оценки

Реестр курируется вручную; генератор проверяет источники, evidence, иерархию, семь измерений покрытия и логические противоречия статусов.

## 3. Сводный статус

| Статус | Количество |
|---|---|
| complete | 10 |
| failed_acceptance_criterion | 1 |
| needs_clarification | 2 |
| not_applicable | 0 |
| not_started | 2 |
| partial | 12 |

Всего требований: **27**.

## 4. Полностью выполненные требования

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-DATA-01 | Унификация gpn_data и Old_EEG | complete | P3 | — |
| R-EVAL | Научно корректная оценка | complete | P3 | — |
| R-EVAL-01 | Outer/inner leakage protection | complete | P3 | — |
| R-EVAL-02 | Метрики, артефакты и статистические сравнения | complete | P3 | — |
| R-FEAT-01 | EEG, POW, raw и sequence representations | complete | P3 | — |
| R-MODEL-01 | Classical ML и Torch model zoo | complete | P3 | — |
| R-MODEL-02 | Классификация label_q5 и многовыходная PM-регрессия | complete | P1 | — |
| R-PERS-01 | Leakage-safe classification и PM personalization | complete | P3 | — |
| R-PLAT | Воспроизводимая интегрированная платформа | complete | P3 | — |
| R-PLAT-01 | Registries, CLI, manifests, resume и summaries | complete | P1 | — |

## 5. Частично выполненные требования

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-DATA | Данные и унификация источников | partial | P1 | Нет единого актуального data-card, включающего собственные и внешние треки.; Внешние datasets не имеют сопоставимого полного experiment contract. |
| R-DATA-02 | Внешний WESAD benchmark | partial | P2 | Нет текущего канонического WESAD experiment config и записи experiment registry.; Исторические WESAD scripts не являются текущим integrated pipeline. |
| R-DOC | Документация и итоговые материалы | partial | P1 | README датирован 20 июля и не отражает последние PM/personalization/config этапы.; Нет единого architecture document и актуального runbook.; Презентация, preprint и финальный deliverable не найдены. |
| R-FEAT | Представления и признаки | partial | P1 | Нет единого feature dictionary с происхождением всех 448 колонок. |
| R-FEAT-02 | Энтропия, связность и снижение размерности | partial | P2 | Энтропийные признаки отсутствуют.; Connectivity prototype не встроен в canonical feature dataset.; PCA и feature selection не имеют train-only integrated path. |
| R-MODEL | Интегрированный набор моделей | partial | P1 | Model selection недоступен в конечном demo-сценарии. |
| R-MODEL-03 | AutoML | partial | P3 | Выполнен diagnostic pilot, а не полный nested AutoML experiment.; До streaming/demo AutoML не закрывает ближайший обязательный deliverable. |
| R-PERS | Персонализация и перенос | partial | P1 | Accuracy 0.75 не достигнута.; Селективное online назначение калибровки не реализовано. |
| R-PERS-02 | Transfer, domain adaptation, meta- и contrastive learning | partial | P3 | Нет обоснованной domain definition для DANN.; Нет production episodic или shared encoder contracts. |
| R-PREP | Предобработка и контроль качества EEG | partial | P1 | Не все упоминаемые artifact-removal методы интегрированы в canonical pipeline. |
| R-PREP-01 | Band-pass, notch, CAR и deduplication | partial | P2 | Полный A-H factorial выполнен только для seed 42.; Diagnostic experiment не устанавливает универсально лучший preprocessing. |
| R-PREP-02 | Artifact removal, ICA и FASTER | partial | P2 | ICA отсутствует.; FASTER prototype не является полноценной реализацией алгоритма и не подключён к runner.; Нет leakage tests, config provenance или experiment result. |

## 6. Невыполненные требования

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-DEMO | Минимальный демонстрационный интерфейс | not_started | P1 | Нет UI или demo CLI, model selection, visualization и export flow. |
| R-STREAM | Потоковый replay и измерение latency | not_started | P1 | Нет replay API, latency/throughput/memory benchmark или error contract. |

## 7. Недостигнутые критерии качества

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-PERS-Q01 | Accuracy персонализации не ниже 0.75 | failed_acceptance_criterion | P0 | Максимальная наблюдавшаяся accuracy 0.6349206349 ниже 0.75.; Авторитетный первоисточник численного критерия не найден. |

## 8. Требования, нуждающиеся в уточнении

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-DATA-03 | STEW и другие открытые EEG-наборы | needs_clarification | P1 | Нет loader, task, config или current result для STEW.; Не найден официальный признак обязательности. |
| R-MULTI | Мультимодальность | needs_clarification | P1 | Нет модели с совместными EEG и wearable inputs.; WESAD относится к отдельной задаче и не синхронизирован с Emotiv.; Официальный multimodal deliverable не найден. |

## 9. Данные

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-DATA | Данные и унификация источников | partial | P1 | Нет единого актуального data-card, включающего собственные и внешние треки.; Внешние datasets не имеют сопоставимого полного experiment contract. |
| R-DATA-01 | Унификация gpn_data и Old_EEG | complete | P3 | — |
| R-DATA-02 | Внешний WESAD benchmark | partial | P2 | Нет текущего канонического WESAD experiment config и записи experiment registry.; Исторические WESAD scripts не являются текущим integrated pipeline. |
| R-DATA-03 | STEW и другие открытые EEG-наборы | needs_clarification | P1 | Нет loader, task, config или current result для STEW.; Не найден официальный признак обязательности. |

## 10. Предобработка

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-PREP | Предобработка и контроль качества EEG | partial | P1 | Не все упоминаемые artifact-removal методы интегрированы в canonical pipeline. |
| R-PREP-01 | Band-pass, notch, CAR и deduplication | partial | P2 | Полный A-H factorial выполнен только для seed 42.; Diagnostic experiment не устанавливает универсально лучший preprocessing. |
| R-PREP-02 | Artifact removal, ICA и FASTER | partial | P2 | ICA отсутствует.; FASTER prototype не является полноценной реализацией алгоритма и не подключён к runner.; Нет leakage tests, config provenance или experiment result. |

## 11. Признаки

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-FEAT | Представления и признаки | partial | P1 | Нет единого feature dictionary с происхождением всех 448 колонок. |
| R-FEAT-01 | EEG, POW, raw и sequence representations | complete | P3 | — |
| R-FEAT-02 | Энтропия, связность и снижение размерности | partial | P2 | Энтропийные признаки отсутствуют.; Connectivity prototype не встроен в canonical feature dataset.; PCA и feature selection не имеют train-only integrated path. |

## 12. Модели

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-MODEL | Интегрированный набор моделей | partial | P1 | Model selection недоступен в конечном demo-сценарии. |
| R-MODEL-01 | Classical ML и Torch model zoo | complete | P3 | — |
| R-MODEL-02 | Классификация label_q5 и многовыходная PM-регрессия | complete | P1 | — |
| R-MODEL-03 | AutoML | partial | P3 | Выполнен diagnostic pilot, а не полный nested AutoML experiment.; До streaming/demo AutoML не закрывает ближайший обязательный deliverable. |

## 13. Оценка качества

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-EVAL | Научно корректная оценка | complete | P3 | — |
| R-EVAL-01 | Outer/inner leakage protection | complete | P3 | — |
| R-EVAL-02 | Метрики, артефакты и статистические сравнения | complete | P3 | — |

## 14. Персонализация и перенос

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-PERS | Персонализация и перенос | partial | P1 | Accuracy 0.75 не достигнута.; Селективное online назначение калибровки не реализовано. |
| R-PERS-01 | Leakage-safe classification и PM personalization | complete | P3 | — |
| R-PERS-02 | Transfer, domain adaptation, meta- и contrastive learning | partial | P3 | Нет обоснованной domain definition для DANN.; Нет production episodic или shared encoder contracts. |
| R-PERS-Q01 | Accuracy персонализации не ниже 0.75 | failed_acceptance_criterion | P0 | Максимальная наблюдавшаяся accuracy 0.6349206349 ниже 0.75.; Авторитетный первоисточник численного критерия не найден. |

## 15. Воспроизводимая платформа

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-PLAT | Воспроизводимая интегрированная платформа | complete | P3 | — |
| R-PLAT-01 | Registries, CLI, manifests, resume и summaries | complete | P1 | — |

## 16. Потоковый режим

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-STREAM | Потоковый replay и измерение latency | not_started | P1 | Нет replay API, latency/throughput/memory benchmark или error contract. |

## 17. Демонстрация

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-DEMO | Минимальный демонстрационный интерфейс | not_started | P1 | Нет UI или demo CLI, model selection, visualization и export flow. |

## 18. Мультимодальность

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-MULTI | Мультимодальность | needs_clarification | P1 | Нет модели с совместными EEG и wearable inputs.; WESAD относится к отдельной задаче и не синхронизирован с Emotiv.; Официальный multimodal deliverable не найден. |

## 19. Документация и результаты

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-DOC | Документация и итоговые материалы | partial | P1 | README датирован 20 июля и не отражает последние PM/personalization/config этапы.; Нет единого architecture document и актуального runbook.; Презентация, preprint и финальный deliverable не найдены. |

## 20. Критические пробелы

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-PERS-Q01 | Accuracy персонализации не ниже 0.75 | failed_acceptance_criterion | P0 | Максимальная наблюдавшаяся accuracy 0.6349206349 ниже 0.75.; Авторитетный первоисточник численного критерия не найден. |

## 21. Что не требует новых экспериментов

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-DATA | Данные и унификация источников | partial | P1 | Нет единого актуального data-card, включающего собственные и внешние треки.; Внешние datasets не имеют сопоставимого полного experiment contract. |
| R-DATA-02 | Внешний WESAD benchmark | partial | P2 | Нет текущего канонического WESAD experiment config и записи experiment registry.; Исторические WESAD scripts не являются текущим integrated pipeline. |
| R-DATA-03 | STEW и другие открытые EEG-наборы | needs_clarification | P1 | Нет loader, task, config или current result для STEW.; Не найден официальный признак обязательности. |
| R-DEMO | Минимальный демонстрационный интерфейс | not_started | P1 | Нет UI или demo CLI, model selection, visualization и export flow. |
| R-DOC | Документация и итоговые материалы | partial | P1 | README датирован 20 июля и не отражает последние PM/personalization/config этапы.; Нет единого architecture document и актуального runbook.; Презентация, preprint и финальный deliverable не найдены. |
| R-FEAT | Представления и признаки | partial | P1 | Нет единого feature dictionary с происхождением всех 448 колонок. |
| R-FEAT-02 | Энтропия, связность и снижение размерности | partial | P2 | Энтропийные признаки отсутствуют.; Connectivity prototype не встроен в canonical feature dataset.; PCA и feature selection не имеют train-only integrated path. |
| R-MODEL | Интегрированный набор моделей | partial | P1 | Model selection недоступен в конечном demo-сценарии. |
| R-MODEL-03 | AutoML | partial | P3 | Выполнен diagnostic pilot, а не полный nested AutoML experiment.; До streaming/demo AutoML не закрывает ближайший обязательный deliverable. |
| R-MULTI | Мультимодальность | needs_clarification | P1 | Нет модели с совместными EEG и wearable inputs.; WESAD относится к отдельной задаче и не синхронизирован с Emotiv.; Официальный multimodal deliverable не найден. |
| R-PERS | Персонализация и перенос | partial | P1 | Accuracy 0.75 не достигнута.; Селективное online назначение калибровки не реализовано. |
| R-PERS-02 | Transfer, domain adaptation, meta- и contrastive learning | partial | P3 | Нет обоснованной domain definition для DANN.; Нет production episodic или shared encoder contracts. |
| R-PERS-Q01 | Accuracy персонализации не ниже 0.75 | failed_acceptance_criterion | P0 | Максимальная наблюдавшаяся accuracy 0.6349206349 ниже 0.75.; Авторитетный первоисточник численного критерия не найден. |
| R-PREP | Предобработка и контроль качества EEG | partial | P1 | Не все упоминаемые artifact-removal методы интегрированы в canonical pipeline. |
| R-PREP-01 | Band-pass, notch, CAR и deduplication | partial | P2 | Полный A-H factorial выполнен только для seed 42.; Diagnostic experiment не устанавливает универсально лучший preprocessing. |
| R-PREP-02 | Artifact removal, ICA и FASTER | partial | P2 | ICA отсутствует.; FASTER prototype не является полноценной реализацией алгоритма и не подключён к runner.; Нет leakage tests, config provenance или experiment result. |
| R-STREAM | Потоковый replay и измерение latency | not_started | P1 | Нет replay API, latency/throughput/memory benchmark или error contract. |

## 22. Рекомендуемый порядок закрытия

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-PERS-Q01 | Accuracy персонализации не ниже 0.75 | failed_acceptance_criterion | P0 | Максимальная наблюдавшаяся accuracy 0.6349206349 ниже 0.75.; Авторитетный первоисточник численного критерия не найден. |
| R-DATA | Данные и унификация источников | partial | P1 | Нет единого актуального data-card, включающего собственные и внешние треки.; Внешние datasets не имеют сопоставимого полного experiment contract. |
| R-DATA-03 | STEW и другие открытые EEG-наборы | needs_clarification | P1 | Нет loader, task, config или current result для STEW.; Не найден официальный признак обязательности. |
| R-DEMO | Минимальный демонстрационный интерфейс | not_started | P1 | Нет UI или demo CLI, model selection, visualization и export flow. |
| R-DOC | Документация и итоговые материалы | partial | P1 | README датирован 20 июля и не отражает последние PM/personalization/config этапы.; Нет единого architecture document и актуального runbook.; Презентация, preprint и финальный deliverable не найдены. |
| R-FEAT | Представления и признаки | partial | P1 | Нет единого feature dictionary с происхождением всех 448 колонок. |
| R-MODEL | Интегрированный набор моделей | partial | P1 | Model selection недоступен в конечном demo-сценарии. |
| R-MULTI | Мультимодальность | needs_clarification | P1 | Нет модели с совместными EEG и wearable inputs.; WESAD относится к отдельной задаче и не синхронизирован с Emotiv.; Официальный multimodal deliverable не найден. |
| R-PERS | Персонализация и перенос | partial | P1 | Accuracy 0.75 не достигнута.; Селективное online назначение калибровки не реализовано. |
| R-PREP | Предобработка и контроль качества EEG | partial | P1 | Не все упоминаемые artifact-removal методы интегрированы в canonical pipeline. |
| R-STREAM | Потоковый replay и измерение latency | not_started | P1 | Нет replay API, latency/throughput/memory benchmark или error contract. |
| R-DATA-02 | Внешний WESAD benchmark | partial | P2 | Нет текущего канонического WESAD experiment config и записи experiment registry.; Исторические WESAD scripts не являются текущим integrated pipeline. |
| R-FEAT-02 | Энтропия, связность и снижение размерности | partial | P2 | Энтропийные признаки отсутствуют.; Connectivity prototype не встроен в canonical feature dataset.; PCA и feature selection не имеют train-only integrated path. |
| R-PREP-01 | Band-pass, notch, CAR и deduplication | partial | P2 | Полный A-H factorial выполнен только для seed 42.; Diagnostic experiment не устанавливает универсально лучший preprocessing. |
| R-PREP-02 | Artifact removal, ICA и FASTER | partial | P2 | ICA отсутствует.; FASTER prototype не является полноценной реализацией алгоритма и не подключён к runner.; Нет leakage tests, config provenance или experiment result. |
| R-MODEL-03 | AutoML | partial | P3 | Выполнен diagnostic pilot, а не полный nested AutoML experiment.; До streaming/demo AutoML не закрывает ближайший обязательный deliverable. |
| R-PERS-02 | Transfer, domain adaptation, meta- и contrastive learning | partial | P3 | Нет обоснованной domain definition для DANN.; Нет production episodic или shared encoder contracts. |
