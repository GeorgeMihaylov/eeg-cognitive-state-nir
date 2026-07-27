# Оставшиеся работы по проекту

## P0

### R-PERS-Q01 — Accuracy персонализации не ниже 0.75

- Пробел: Максимальная наблюдавшаяся accuracy 0.6349206349 ниже 0.75.; Авторитетный первоисточник численного критерия не найден..
- Минимальное действие: Подтвердить происхождение и обязательность порога; не запускать неограниченные fine-tuning sweeps автоматически.

- Новый эксперимент: нет.
- Зависимости: официальный acceptance criterion.
- Критерий завершения: Порог официально подтверждён, пересмотрен или снят с обоснованием.

## P1

### R-DATA — Данные и унификация источников

- Пробел: Нет единого актуального data-card, включающего собственные и внешние треки.; Внешние datasets не имеют сопоставимого полного experiment contract..
- Минимальное действие: Подготовить единый data-card и явно зафиксировать scope внешних наборов.
- Новый эксперимент: нет.
- Зависимости: официальный scope данных.
- Критерий завершения: Для каждого включённого источника описаны происхождение, задача, splits, идентификаторы, ограничения и каноническая точка загрузки.


### R-DEMO — Минимальный демонстрационный интерфейс

- Пробел: Нет UI или demo CLI, model selection, visualization и export flow..
- Минимальное действие: Построить минимальный demo поверх streaming replay и model registry.
- Новый эксперимент: нет.
- Зависимости: R-STREAM; canonical artifact selection.
- Критерий завершения: Документированный end-to-end demo запускается на одной записи.

### R-DOC — Документация и итоговые материалы

- Пробел: README датирован 20 июля и не отражает последние PM/personalization/config этапы.; Нет единого architecture document и актуального runbook.; Презентация, preprint и финальный deliverable не найдены..
- Минимальное действие: После streaming/demo обновить README, architecture/runbook и собрать финальный комплект отчёта и презентации.

- Новый эксперимент: нет.
- Зависимости: R-STREAM; R-DEMO; официальный deliverable scope.
- Критерий завершения: README, architecture, runbook, final report и presentation согласованы с generated registries и фактическими артефактами.


### R-FEAT — Представления и признаки

- Пробел: Нет единого feature dictionary с происхождением всех 448 колонок..
- Минимальное действие: Создать feature dictionary с группой, формулой и provenance.
- Новый эксперимент: нет.
- Зависимости: нет.
- Критерий завершения: Все 448 feature columns трассируются до исходного преобразования.

### R-MODEL — Интегрированный набор моделей

- Пробел: Model selection недоступен в конечном demo-сценарии..
- Минимальное действие: Подключить выбор канонической модели в минимальном demo.
- Новый эксперимент: нет.
- Зависимости: R-DEMO.
- Критерий завершения: Пользователь выбирает зарегистрированную модель без изменения кода.

### R-PERS — Персонализация и перенос

- Пробел: Accuracy 0.75 не достигнута.; Селективное online назначение калибровки не реализовано..
- Минимальное действие: Перенести validated personalization в будущий demo/streaming сценарий без новых budget sweeps.

- Новый эксперимент: нет.
- Зависимости: R-STREAM; R-DEMO.
- Критерий завершения: Demo использует zero-shot/head-only/full-model protocol без leakage.

### R-PREP — Предобработка и контроль качества EEG

- Пробел: Не все упоминаемые artifact-removal методы интегрированы в canonical pipeline..
- Минимальное действие: Зафиксировать обязательный минимальный preprocessing scope и отдельно решить необходимость artifact removal.

- Новый эксперимент: нет.
- Зависимости: официальный preprocessing scope.
- Критерий завершения: Обязательные операции и deferred методы явно разделены.

### R-STREAM — Потоковый replay и измерение latency

- Пробел: Нет replay API, latency/throughput/memory benchmark или error contract..
- Минимальное действие: Реализовать минимальный offline replay поверх canonical model artifact.
- Новый эксперимент: нет.
- Зависимости: выбор canonical model artifact; input/output API.
- Критерий завершения: Replay обрабатывает запись по окнам и сохраняет latency, throughput, memory и prediction timeline.


## P2

### R-DATA-02 — Внешний WESAD benchmark

- Пробел: Нет текущего канонического WESAD experiment config и записи experiment registry.; Исторические WESAD scripts не являются текущим integrated pipeline..
- Минимальное действие: Уточнить, обязателен ли WESAD deliverable; при положительном решении интегрировать один ограниченный baseline через текущий runner.

- Новый эксперимент: нет.
- Зависимости: официальный scope внешних данных.
- Критерий завершения: Scope-решение принято; при включении есть config, test и baseline.

### R-FEAT-02 — Энтропия, связность и снижение размерности

- Пробел: Энтропийные признаки отсутствуют.; Connectivity prototype не встроен в canonical feature dataset.; PCA и feature selection не имеют train-only integrated path..
- Минимальное действие: Уточнить обязательные advanced feature families до реализации.
- Новый эксперимент: нет.
- Зависимости: официальный feature scope.
- Критерий завершения: Есть утверждённый список обязательных feature families.

### R-PREP-01 — Band-pass, notch, CAR и deduplication

- Пробел: Полный A-H factorial выполнен только для seed 42.; Diagnostic experiment не устанавливает универсально лучший preprocessing..
- Минимальное действие: Зафиксировать raw как текущий reference и границы обобщения ablation; не запускать полный повтор без отдельной гипотезы.

- Новый эксперимент: нет.
- Зависимости: нет.
- Критерий завершения: README не представляет diagnostic ablation как финальный выбор.

### R-PREP-02 — Artifact removal, ICA и FASTER

- Пробел: ICA отсутствует.; FASTER prototype не является полноценной реализацией алгоритма и не подключён к runner.; Нет leakage tests, config provenance или experiment result..
- Минимальное действие: Сначала решить, входит ли artifact removal в обязательный scope; не интегрировать prototype без методической спецификации.

- Новый эксперимент: нет.
- Зависимости: artifact-removal acceptance definition.
- Критерий завершения: Принято документированное scope-решение и prototype не называется production FASTER.

## P3

### R-MODEL-03 — AutoML

- Пробел: Выполнен diagnostic pilot, а не полный nested AutoML experiment.; До streaming/demo AutoML не закрывает ближайший обязательный deliverable..
- Минимальное действие: Отложить full AutoML до закрытия platform deliverables.
- Новый эксперимент: нет.
- Зависимости: R-STREAM; R-DEMO; официальный AutoML scope.
- Критерий завершения: Есть отдельное решение, нужен ли full AutoML.

### R-PERS-02 — Transfer, domain adaptation, meta- и contrastive learning

- Пробел: Нет обоснованной domain definition для DANN.; Нет production episodic или shared encoder contracts..
- Минимальное действие: Не продолжать deferred mixins без новой научной постановки.
- Новый эксперимент: нет.
- Зависимости: domain definition; shared encoder API.
- Критерий завершения: Deferred status сохраняется либо появляется утверждённая постановка.

## Требуется уточнение scope

- **R-DATA-03**: Зафиксировать включение или исключение STEW в утверждённом scope.
- **R-MULTI**: Определить, требуется ли истинная multimodal fusion или отдельные benchmarks достаточны.

## Не рекомендуется выполнять сейчас

- **Повторные personalization budget sweeps** — Validated 20% multi-seed результат уже отвечает на текущий вопрос; ближайшие пробелы — platform deliverables.
- **DANN** — Нет обоснованной domain definition и source/target runner contract.
- **MAML** — Нет обязательного episodic protocol и production integration.
- **Contrastive pretraining** — Нет shared encoder API; 448 aggregated features нельзя выдавать за raw EEG.
- **Массовое перемещение конфигов** — Config curation завершена без необходимости физической реорганизации.
- **Унификация всех config loaders** — Специализированные loaders валидны; общий refactor не закрывает ближайший deliverable.
- **Full AutoML** — Streaming, demo и итоговая документация имеют более прямую связь с завершением платформы.
- **Дополнительные модели ради количества** — Model zoo уже покрывает классические, recurrent, Transformer и raw-CNN families.
