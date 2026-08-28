# Финальное покрытие требований

Карта разделяет формальное закрытие, научную ценность и сервисные/
демонстрационные пункты. Авторитетный юридический текст ТЗ в tracked-дереве
не найден, поэтому статусы являются инженерной трассировкой утверждённого
плана проекта.

## Сводка

| status | count |
|---|---|
| closed | 10 |
| not_required_for_article | 3 |
| open | 2 |
| partially_closed | 12 |

## Требования

| requirement_id | requirement | category | status | remaining_gap | recommended_closure_form |
|---|---|---|---|---|---|
| R-DATA | Данные и унификация источников | ('scientific',) | partially_closed | Нет единого актуального data-card, включающего собственные и внешние треки./Внешние datasets не имеют сопоставимого полного experiment contract. | Подготовить единый data-card и явно зафиксировать scope внешних наборов. |
| R-DATA-01 | Унификация gpn_data и Old_EEG | ('scientific',) | closed |  | Поддерживать data provenance при следующих изменениях набора. |
| R-DATA-02 | Внешний WESAD benchmark | ('scientific',) | not_required_for_article | Нет текущего канонического WESAD experiment config и записи experiment registry./Исторические WESAD scripts не являются текущим integrated pipeline. | Уточнить, обязателен ли WESAD deliverable; при положительном решении интегрировать один ограниченный baseline через текущий runner.
 |
| R-DATA-03 | STEW и другие открытые EEG-наборы | ('scientific',) | not_required_for_article | Нет loader, task, config или current result для STEW./Не найден официальный признак обязательности. | Зафиксировать включение или исключение STEW в утверждённом scope. |
| R-PREP | Предобработка и контроль качества EEG | ('scientific',) | partially_closed | Не все упоминаемые artifact-removal методы интегрированы в canonical pipeline. | Зафиксировать обязательный минимальный preprocessing scope и отдельно решить необходимость artifact removal.
 |
| R-PREP-01 | Band-pass, notch, CAR и deduplication | ('scientific',) | partially_closed | Полный A-H factorial выполнен только для seed 42./Diagnostic experiment не устанавливает универсально лучший preprocessing. | Зафиксировать raw как текущий reference и границы обобщения ablation; не запускать полный повтор без отдельной гипотезы.
 |
| R-PREP-02 | Artifact removal, ICA и FASTER | ('scientific',) | partially_closed | ICA отсутствует./FASTER prototype не является полноценной реализацией алгоритма и не подключён к runner./Нет leakage tests, config provenance или experiment result. | Сначала решить, входит ли artifact removal в обязательный scope; не интегрировать prototype без методической спецификации.
 |
| R-FEAT | Представления и признаки | ('scientific',) | partially_closed | Нет единого feature dictionary с происхождением всех 448 колонок. | Создать feature dictionary с группой, формулой и provenance. |
| R-FEAT-01 | EEG, POW, raw и sequence representations | ('scientific',) | closed |  | Поддерживать representation contracts в архитектурной документации. |
| R-FEAT-02 | Энтропия, связность и снижение размерности | ('scientific',) | partially_closed | Энтропийные признаки отсутствуют./Connectivity prototype не встроен в canonical feature dataset./PCA и feature selection не имеют train-only integrated path. | Уточнить обязательные advanced feature families до реализации. |
| R-MODEL | Интегрированный набор моделей | ('scientific',) | partially_closed | Model selection недоступен в конечном demo-сценарии. | Подключить выбор канонической модели в минимальном demo. |
| R-MODEL-01 | Classical ML и Torch model zoo | ('scientific',) | closed |  | Поддерживать model matrix в README и summary registry. |
| R-MODEL-02 | Классификация label_q5 и многовыходная PM-регрессия | ('scientific',) | closed |  | Обновить устаревшее ограничение README о regression track. |
| R-MODEL-03 | AutoML | ('scientific',) | partially_closed | Выполнен diagnostic pilot, а не полный nested AutoML experiment./До streaming/demo AutoML не закрывает ближайший обязательный deliverable. | Отложить full AutoML до закрытия platform deliverables. |
| R-EVAL | Научно корректная оценка | ('scientific',) | closed |  | Поддерживать единое описание evaluation policy. |
| R-EVAL-01 | Outer/inner leakage protection | ('scientific',) | closed |  | Сохранять leakage checklist для новых experiment families. |
| R-EVAL-02 | Метрики, артефакты и статистические сравнения | ('scientific',) | closed |  | Поддерживать artifact schema в architecture docs. |
| R-PERS | Персонализация и перенос | ('scientific',) | partially_closed | Accuracy 0.75 не достигнута./Селективное online назначение калибровки не реализовано. | Перенести validated personalization в будущий demo/streaming сценарий без новых budget sweeps.
 |
| R-PERS-01 | Leakage-safe classification и PM personalization | ('scientific',) | closed |  | Поддерживать финальные personalization reports и configs. |
| R-PERS-Q01 | Accuracy персонализации не ниже 0.75 | ('scientific',) | partially_closed | Максимальная наблюдавшаяся accuracy 0.6349206349 ниже 0.75./Авторитетный первоисточник численного критерия не найден. | Подтвердить происхождение и обязательность порога; не запускать неограниченные fine-tuning sweeps автоматически.
 |
| R-PERS-02 | Transfer, domain adaptation, meta- и contrastive learning | ('scientific',) | partially_closed | DANN is partially confirmed only in Old_EEG to gpn_data; FOMAML diagnostic is do_not_proceed; reverse DANN and a target-supervised upper bound remain untested. | Сохранить частично подтверждённый DANN и отрицательный FOMAML; новые sweeps запускать только по новой научной постановке. |
| R-PLAT | Воспроизводимая интегрированная платформа | ('formal',) | closed |  | Поддерживать registries и architecture documentation. |
| R-PLAT-01 | Registries, CLI, manifests, resume и summaries | ('formal',) | closed |  | Синхронизировать README с текущими registries. |
| R-STREAM | Потоковый replay и измерение latency | ('service/demo',) | open | Нет replay API, latency/throughput/memory benchmark или error contract. | Реализовать минимальный offline replay поверх canonical model artifact. |
| R-DEMO | Минимальный демонстрационный интерфейс | ('service/demo',) | open | Нет UI или demo CLI, model selection, visualization и export flow. | Построить минимальный demo поверх streaming replay и model registry. |
| R-MULTI | Мультимодальность | ('scientific',) | not_required_for_article | Нет модели с совместными EEG и wearable inputs./WESAD относится к отдельной задаче и не синхронизирован с Emotiv./Официальный multimodal deliverable не найден. | Определить, требуется ли истинная multimodal fusion или отдельные benchmarks достаточны. |
| R-DOC | Документация и итоговые материалы | ('formal',) | partially_closed | README датирован 20 июля и не отражает последние PM/personalization/config этапы./Нет единого architecture document и актуального runbook./Презентация, preprint и финальный deliverable не найдены. | После streaming/demo обновить README, architecture/runbook и собрать финальный комплект отчёта и презентации.
 |

## Оставшаяся работа

**Обязательно:** итоговая документация, description/data card,
reproducibility section, финальный отчёт, таблицы/рисунки, презентация.

**Желательно для статьи:** выбрать центральную гипотезу, зафиксировать
статистически корректные сравнения, related work, вклад, ограничения и
приложение отрицательных результатов.

**Исключить без новой гипотезы:** дальнейший DANN search, FOMAML sweep,
новый contrastive search, полный 62-канальный cache, дополнительные COG-BCI
CNN seeds, AutoML и новые внешние наборы. Уже выполненный confirmatory DANN
сохраняется как `partially_confirmed` evidence.
