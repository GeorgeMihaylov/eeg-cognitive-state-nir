# Аудит воспроизводимости итогового пакета

Дата актуализации: 2026-08-17.

## Контракт

- Tracked configs и отчёты не содержат локальных абсолютных путей.
- Основной evaluation — participant-disjoint; inner groups и train-only fitted
  transforms фиксируются в manifests.
- Runtime predictions, checkpoints и caches остаются вне Git.
- Метрики разных задач, cohorts и уровней агрегации не объединяются.
- Smoke/preliminary/diagnostic не выдаются за confirmatory/final.
- Отрицательные outcomes отделены от незапущенных экспериментов.

## Проверенные первичные артефакты

| Контур | Проверка | Результат |
|---|---|---|
| PM temporal quality | Пять manifests по 56 runs, 280 metrics JSON | Complete, failed runs отсутствуют; legacy `result_status=diagnostic` — metadata-only discrepancy |
| LightGBM selection | `execution_manifest.json`, 140 run summaries | 140/140 complete, failed=0, unique specification hashes=140, participant overlap=0 |
| Artifact removal | protocol manifest и smoke artifacts | Full matrix 140 не выполнена; smoke 4/4 complete |
| DANN | preregistration, locks, global audits, primary aggregates | 28 новых trainings complete; primary seeds 123/2026 отделены от seed-42 sensitivity |
| Multimodal | fold manifests, `summary_xgboost.csv`, `summary_shallow.csv` | MEFAR 15, CL-Drive 25 и CLARE 25 units представлены; test cohorts совпадают внутри folds |
| Streaming | protocol/model/replay manifests, latency и API snapshots | Научный bundle, exact replay alignment, quality/postprocessing и API path подтверждены |

Ключевые protocol hashes:

- LightGBM selection:
  `f9c3898cd2ce20055082e3d8e746c830fcadf71a72dff0a55c760880f3b736bf`;
- artifact-removal plan:
  `297eaf71684c790d3514b888eebc553e9612e6826e495b22b481757a1eeee23b`;
- selected-model seven-PM plan:
  `3981d726ace6cb91bc42cc9fa0c04dea89e85d8278cf34bf511969b00d029d76`;
- DANN confirmatory v2:
  `1ce582a3d73a7ae4393e77cc2f3b2cb7749ddbb30c1cb8fcad0056c6d326c368`.

## Ограничения provenance

Сводный inventory от 4 августа фиксирует 45 более ранних experiment records и
не включает все поздние PM-quality, LightGBM, external multimodal и streaming
результаты. Поэтому старые totals 39/45 не используются как текущая полнота
проекта; канонической для приёмки является ручная матрица требований.

Авторитетный исходный файл ТЗ в tracked-дереве и доступной Git history не
найден. Из-за этого обязательность двух подготовленных full benchmarks нельзя
вывести из репозитория. Это limitation формальной трассируемости, а не
scientific leakage.

PM temporal runtime находится в отдельном PM-quality рабочем дереве, тогда как
tracked отчёт и config интегрированы здесь. Исторические fold manifests имеют
`result_status=diagnostic` вследствие ранее исправленной propagation-ошибки;
метрики, predictions, cohorts и protocol execution не затронуты. Metadata
migration в ходе этого аудита не выполнялась.

Последняя полная проверка кода, зафиксированная в README на текущем базовом
состоянии: **1475 passed, 1 skipped, 37 warnings**. Настоящее изменение только
документальное; длительные эксперименты не запускались.
