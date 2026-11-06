# Оставшиеся работы по проекту

## P0

_Нет задач._

## P1

### R-DATA — Internal canonical data and external task contracts

- Пробел: A unified publication data card spanning current internal and external tracks remains to be consolidated..
- Минимальное действие: Consolidate provenance, deduplication, native external labels and support; no data/cache rebuild.
- Новый эксперимент: нет.
- Зависимости: нет.
- Критерий завершения: Consolidate provenance, deduplication, native external labels and support; no data/cache rebuild.

### R-DEMO — Implemented streaming CLI/dashboard/API; delivery acceptance pending

- Пробел: Final demo acceptance and hardware deployment scope need confirmation..
- Минимальное действие: Package existing CLI/API/dashboard with a compatible model bundle; no new training.
- Новый эксперимент: нет.
- Зависимости: нет.
- Критерий завершения: Package existing CLI/API/dashboard with a compatible model bundle; no new training.

### R-DOC — Current documentation and publication deliverables

- Пробел: Final manuscript, presentation and acceptance scope remain to be finalized..
- Минимальное действие: Maintain current README/context/registries; assemble publication material from existing evidence.
- Новый эксперимент: нет.
- Зависимости: нет.
- Критерий завершения: Maintain current README/context/registries; assemble publication material from existing evidence.

### R-MULTI — Completed external EEG/peripheral/fusion comparisons

- Пробел: External labels differ from seven internal PM; temporal findings have not been independently replicated..
- Минимальное действие: Keep MEFAR/CL-Drive/CLARE native tasks separate; do not merge labels.
- Новый эксперимент: нет.
- Зависимости: нет.
- Критерий завершения: Keep MEFAR/CL-Drive/CLARE native tasks separate; do not merge labels.

### R-PERS — LOW/HIGH personalization and transfer

- Пробел: Deployment of the validated adaptation protocol with real hardware remains separate from offline evidence..
- Минимальное действие: Preserve 30-minute calibration, fixed target thresholds, operational fallback and future-tail evaluation.
- Новый эксперимент: нет.
- Зависимости: нет.
- Критерий завершения: Preserve 30-minute calibration, fixed target thresholds, operational fallback and future-tail evaluation.

### R-PERS-02 — Completed transfer/DANN/FOMAML/contrastive tracks

- Пробел: Independent external validation of the current temporal conclusion remains open..
- Минимальное действие: Keep historical negative/conditional findings; do not restart closed tracks without a new hypothesis.
- Новый эксперимент: нет.
- Зависимости: нет.
- Критерий завершения: Keep historical negative/conditional findings; do not restart closed tracks without a new hypothesis.

### R-PREP — Preprocessing and EEG quality control

- Пробел: Further representation changes require a separate scientific hypothesis..
- Минимальное действие: Document tested protocols and preserve negative preprocessing evidence.
- Новый эксперимент: нет.
- Зависимости: нет.
- Критерий завершения: Document tested protocols and preserve negative preprocessing evidence.

### R-SEQUENCE — Planned bounded H120 sequence comparison

- Пробел: One architecture and evaluation protocol still need to be frozen; experiment not executed..
- Минимальное действие: Prespecify one bounded H120 comparison against ordered-H120 LightGBM; execution requires a separate request.
- Новый эксперимент: нет.
- Зависимости: нет.
- Критерий завершения: Prespecify one bounded H120 comparison against ordered-H120 LightGBM; execution requires a separate request.

### R-STREAM — Implemented replay/API and software latency; hardware latency pending

- Пробел: Sensor-to-user end-to-end latency with physical hardware has not been measured..
- Минимальное действие: Document measured software replay separately from pending hardware latency.
- Новый эксперимент: нет.
- Зависимости: нет.
- Критерий завершения: Document measured software replay separately from pending hardware latency.

## P2

### R-DATA-02 — Внешний WESAD benchmark

- Пробел: Нет текущего канонического WESAD experiment config и записи experiment registry.; Исторические WESAD scripts не являются текущим integrated pipeline..
- Минимальное действие: Уточнить, обязателен ли WESAD deliverable; при положительном решении интегрировать один ограниченный baseline через текущий runner.

- Новый эксперимент: нет.
- Зависимости: официальный scope внешних данных.
- Критерий завершения: Scope-решение принято; при включении есть config, test и baseline.

## P3

_Нет задач._

## Требуется уточнение scope

- **R-DATA-03**: Зафиксировать включение или исключение STEW в утверждённом scope.

## Не рекомендуется выполнять сейчас

- **Broad architecture or AutoML search** — Nested AutoML is complete; temporal results justify one bounded comparison, not a sweep.
- **Repeated historical Q3/label_q5 or order-control fits** — Existing evidence retained; no new training is needed for documentation.
- **DANN/FOMAML/contrastive sweeps** — Infrastructure exists; tested directions are closed negative or conditional and require a new hypothesis.
- **Simultaneous architecture and representation changes** — Keep wavelet/synchronization hypotheses separate from the planned H120 sequence comparison.
