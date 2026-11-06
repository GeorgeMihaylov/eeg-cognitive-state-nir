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
| complete | 17 |
| failed_acceptance_criterion | 0 |
| needs_clarification | 1 |
| not_applicable | 1 |
| not_started | 0 |
| partial | 10 |

Всего требований: **29**.

## 4. Полностью выполненные требования

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-DATA-01 | Унификация gpn_data и Old_EEG | complete | P3 | — |
| R-EVAL | Научно корректная оценка | complete | P3 | — |
| R-EVAL-01 | Outer/inner leakage protection | complete | P3 | — |
| R-EVAL-02 | Метрики, артефакты и статистические сравнения | complete | P3 | — |
| R-FEAT | Canonical 371 engineered features | complete | P1 | — |
| R-FEAT-01 | EEG, POW, raw и sequence representations | complete | P3 | — |
| R-FEAT-02 | Entropy, connectivity and historical feature selection | complete | P1 | — |
| R-MODEL | Integrated model factory and current primary references | complete | P1 | — |
| R-MODEL-01 | Classical ML и Torch model zoo | complete | P3 | — |
| R-MODEL-02 | Seven-PM LOW/HIGH classification and supporting continuous PM regression | complete | P1 | — |
| R-MODEL-03 | Nested LOW/HIGH AutoML | complete | P1 | — |
| R-TEMPORAL | Completed temporal predictability and H120 order control | complete | P1 | — |
| R-PERS-01 | Leakage-safe LOW/HIGH adaptation and supporting PM regression | complete | P1 | — |
| R-PLAT | Воспроизводимая интегрированная платформа | complete | P3 | — |
| R-PLAT-01 | Registries, CLI, manifests, resume и summaries | complete | P1 | — |
| R-PREP-01 | Band-pass, notch, CAR and deduplication | complete | P1 | — |
| R-PREP-02 | Completed ICA and MNE-FASTER comparisons | complete | P1 | — |

## 5. Частично выполненные требования

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-DATA | Internal canonical data and external task contracts | partial | P1 | A unified publication data card spanning current internal and external tracks remains to be consolidated. |
| R-DATA-02 | Внешний WESAD benchmark | partial | P2 | Нет текущего канонического WESAD experiment config и записи experiment registry.; Исторические WESAD scripts не являются текущим integrated pipeline. |
| R-DEMO | Implemented streaming CLI/dashboard/API; delivery acceptance pending | partial | P1 | Final demo acceptance and hardware deployment scope need confirmation. |
| R-DOC | Current documentation and publication deliverables | partial | P1 | Final manuscript, presentation and acceptance scope remain to be finalized. |
| R-SEQUENCE | Planned bounded H120 sequence comparison | partial | P1 | One architecture and evaluation protocol still need to be frozen; experiment not executed. |
| R-MULTI | Completed external EEG/peripheral/fusion comparisons | partial | P1 | External labels differ from seven internal PM; temporal findings have not been independently replicated. |
| R-PERS | LOW/HIGH personalization and transfer | partial | P1 | Deployment of the validated adaptation protocol with real hardware remains separate from offline evidence. |
| R-PERS-02 | Completed transfer/DANN/FOMAML/contrastive tracks | partial | P1 | Independent external validation of the current temporal conclusion remains open. |
| R-PREP | Preprocessing and EEG quality control | partial | P1 | Further representation changes require a separate scientific hypothesis. |
| R-STREAM | Implemented replay/API and software latency; hardware latency pending | partial | P1 | Sensor-to-user end-to-end latency with physical hardware has not been measured. |

## 6. Невыполненные требования

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|

## 7. Недостигнутые критерии качества

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|

## 8. Требования, нуждающиеся в уточнении

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-DATA-03 | STEW и другие открытые EEG-наборы | needs_clarification | P1 | Нет loader, task, config или current result для STEW.; Не найден официальный признак обязательности. |

## 9. Данные

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-DATA | Internal canonical data and external task contracts | partial | P1 | A unified publication data card spanning current internal and external tracks remains to be consolidated. |
| R-DATA-01 | Унификация gpn_data и Old_EEG | complete | P3 | — |
| R-DATA-02 | Внешний WESAD benchmark | partial | P2 | Нет текущего канонического WESAD experiment config и записи experiment registry.; Исторические WESAD scripts не являются текущим integrated pipeline. |
| R-DATA-03 | STEW и другие открытые EEG-наборы | needs_clarification | P1 | Нет loader, task, config или current result для STEW.; Не найден официальный признак обязательности. |

## 10. Предобработка

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-PREP | Preprocessing and EEG quality control | partial | P1 | Further representation changes require a separate scientific hypothesis. |
| R-PREP-01 | Band-pass, notch, CAR and deduplication | complete | P1 | — |
| R-PREP-02 | Completed ICA and MNE-FASTER comparisons | complete | P1 | — |

## 11. Признаки

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-FEAT | Canonical 371 engineered features | complete | P1 | — |
| R-FEAT-01 | EEG, POW, raw и sequence representations | complete | P3 | — |
| R-FEAT-02 | Entropy, connectivity and historical feature selection | complete | P1 | — |

## 12. Модели

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-MODEL | Integrated model factory and current primary references | complete | P1 | — |
| R-MODEL-01 | Classical ML и Torch model zoo | complete | P3 | — |
| R-MODEL-02 | Seven-PM LOW/HIGH classification and supporting continuous PM regression | complete | P1 | — |
| R-MODEL-03 | Nested LOW/HIGH AutoML | complete | P1 | — |
| R-SEQUENCE | Planned bounded H120 sequence comparison | partial | P1 | One architecture and evaluation protocol still need to be frozen; experiment not executed. |
| R-TEMPORAL | Completed temporal predictability and H120 order control | complete | P1 | — |

## 13. Оценка качества

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-EVAL | Научно корректная оценка | complete | P3 | — |
| R-EVAL-01 | Outer/inner leakage protection | complete | P3 | — |
| R-EVAL-02 | Метрики, артефакты и статистические сравнения | complete | P3 | — |

## 14. Персонализация и перенос

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-PERS | LOW/HIGH personalization and transfer | partial | P1 | Deployment of the validated adaptation protocol with real hardware remains separate from offline evidence. |
| R-PERS-01 | Leakage-safe LOW/HIGH adaptation and supporting PM regression | complete | P1 | — |
| R-PERS-02 | Completed transfer/DANN/FOMAML/contrastive tracks | partial | P1 | Independent external validation of the current temporal conclusion remains open. |
| R-PERS-Q01 | Historical label_q5 accuracy threshold 0.75 (not current LOW/HIGH criterion) | not_applicable | P1 | — |

## 15. Воспроизводимая платформа

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-PLAT | Воспроизводимая интегрированная платформа | complete | P3 | — |
| R-PLAT-01 | Registries, CLI, manifests, resume и summaries | complete | P1 | — |

## 16. Потоковый режим

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-STREAM | Implemented replay/API and software latency; hardware latency pending | partial | P1 | Sensor-to-user end-to-end latency with physical hardware has not been measured. |

## 17. Демонстрация

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-DEMO | Implemented streaming CLI/dashboard/API; delivery acceptance pending | partial | P1 | Final demo acceptance and hardware deployment scope need confirmation. |

## 18. Мультимодальность

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-MULTI | Completed external EEG/peripheral/fusion comparisons | partial | P1 | External labels differ from seven internal PM; temporal findings have not been independently replicated. |

## 19. Документация и результаты

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-DOC | Current documentation and publication deliverables | partial | P1 | Final manuscript, presentation and acceptance scope remain to be finalized. |

## 20. Критические пробелы

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|

## 21. Что не требует новых экспериментов

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-DATA | Internal canonical data and external task contracts | partial | P1 | A unified publication data card spanning current internal and external tracks remains to be consolidated. |
| R-DATA-02 | Внешний WESAD benchmark | partial | P2 | Нет текущего канонического WESAD experiment config и записи experiment registry.; Исторические WESAD scripts не являются текущим integrated pipeline. |
| R-DATA-03 | STEW и другие открытые EEG-наборы | needs_clarification | P1 | Нет loader, task, config или current result для STEW.; Не найден официальный признак обязательности. |
| R-DEMO | Implemented streaming CLI/dashboard/API; delivery acceptance pending | partial | P1 | Final demo acceptance and hardware deployment scope need confirmation. |
| R-DOC | Current documentation and publication deliverables | partial | P1 | Final manuscript, presentation and acceptance scope remain to be finalized. |
| R-SEQUENCE | Planned bounded H120 sequence comparison | partial | P1 | One architecture and evaluation protocol still need to be frozen; experiment not executed. |
| R-MULTI | Completed external EEG/peripheral/fusion comparisons | partial | P1 | External labels differ from seven internal PM; temporal findings have not been independently replicated. |
| R-PERS | LOW/HIGH personalization and transfer | partial | P1 | Deployment of the validated adaptation protocol with real hardware remains separate from offline evidence. |
| R-PERS-02 | Completed transfer/DANN/FOMAML/contrastive tracks | partial | P1 | Independent external validation of the current temporal conclusion remains open. |
| R-PREP | Preprocessing and EEG quality control | partial | P1 | Further representation changes require a separate scientific hypothesis. |
| R-STREAM | Implemented replay/API and software latency; hardware latency pending | partial | P1 | Sensor-to-user end-to-end latency with physical hardware has not been measured. |

## 22. Рекомендуемый порядок закрытия

| ID | Требование | Статус | Приоритет закрытия | Пробел |
|---|---|---|---|---|
| R-DATA | Internal canonical data and external task contracts | partial | P1 | A unified publication data card spanning current internal and external tracks remains to be consolidated. |
| R-DATA-03 | STEW и другие открытые EEG-наборы | needs_clarification | P1 | Нет loader, task, config или current result для STEW.; Не найден официальный признак обязательности. |
| R-DEMO | Implemented streaming CLI/dashboard/API; delivery acceptance pending | partial | P1 | Final demo acceptance and hardware deployment scope need confirmation. |
| R-DOC | Current documentation and publication deliverables | partial | P1 | Final manuscript, presentation and acceptance scope remain to be finalized. |
| R-MULTI | Completed external EEG/peripheral/fusion comparisons | partial | P1 | External labels differ from seven internal PM; temporal findings have not been independently replicated. |
| R-PERS | LOW/HIGH personalization and transfer | partial | P1 | Deployment of the validated adaptation protocol with real hardware remains separate from offline evidence. |
| R-PERS-02 | Completed transfer/DANN/FOMAML/contrastive tracks | partial | P1 | Independent external validation of the current temporal conclusion remains open. |
| R-PREP | Preprocessing and EEG quality control | partial | P1 | Further representation changes require a separate scientific hypothesis. |
| R-SEQUENCE | Planned bounded H120 sequence comparison | partial | P1 | One architecture and evaluation protocol still need to be frozen; experiment not executed. |
| R-STREAM | Implemented replay/API and software latency; hardware latency pending | partial | P1 | Sensor-to-user end-to-end latency with physical hardware has not been measured. |
| R-DATA-02 | Внешний WESAD benchmark | partial | P2 | Нет текущего канонического WESAD experiment config и записи experiment registry.; Исторические WESAD scripts не являются текущим integrated pipeline. |
