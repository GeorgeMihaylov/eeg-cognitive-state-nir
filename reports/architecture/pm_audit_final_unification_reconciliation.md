# PM audit ↔ final package unification: reconciliation audit

## Статус

Семантическая адаптация PM target-validity/lag-кода к финальной структуре пакетов
выполнена без повторного массового копирования, обучения моделей и изменения
существующих научных результатов. Staging area остаётся пустым.

## Sources

| Роль | Рабочая копия / ветка | HEAD |
|---|---|---|
| Destination | `F:\EEG`, `analysis/pm-target-validity-audit-20260827` | `10812874535be6885228be2ed55aacdf1f4aa924` |
| Final-package source | `F:\eeg-final-unification`, `refactor/final-package-unification-20260828` | `7da5e922608c07efb0786ada6f3bfd7c3c9f0962` |

Общий merge base:
`3b50f35d45522cb6faa1c40da10409b32089b081`.

## Mechanical transfer

Ранее выполненный ручной перенос подтверждён и повторно не запускался:

- 46 additions из final-package-unification;
- 136 заменённых tracked-файлов;
- 44 удалённых legacy-файла;
- исходный mechanical tracked diff: 180 файлов, 4 102 добавления,
  17 914 удалений;
- пересечение PM audit paths с final-package paths: отсутствует;
- staging после переноса и после адаптации: пустой.

Текущее объединённое рабочее дерево шире mechanical diff, поскольку сохраняет
PM-ветку и выполненную ниже семантическую адаптацию. На момент финальной проверки
tracked diff содержит 160 modified и 44 deleted файла (204 tracked-файла,
4 915 добавлений, 19 628 удалений). Эти числа не переопределяют исходный
mechanical inventory.

## Semantic adaptation

### Канонические model imports

В следующих модулях удалён импорт из legacy top-level `model_zoo` и применён
канонический `cogstate.model_zoo`:

- `bench/experiments/pm_eeg_lag_confirmatory.py`;
- `bench/experiments/pm_eeg_lag_regression_confirmatory.py`.

Экспериментальные спецификации не изменены: classification продолжает
использовать XGBoost classifier, regression — `XGBRegressor`; в обоих случаях
сохранены `n_estimators=200`, `n_jobs=4`, `random_state=42`.

Legacy compatibility package не создавался. Каталоги `src/`, top-level
`model_zoo/` и top-level `automl/` отсутствуют.

### Library modules и thin CLI

Научная логика четырёх длинных скриптов перенесена без изменения CLI-контракта:

| Thin CLI | Импортируемый модуль |
|---|---|
| `scripts/analysis/postprocess_pm_target_validity.py` | `bench.analysis.pm_target_validity_postprocess` |
| `scripts/analysis/analyze_pm_subject_structure.py` | `bench.analysis.pm_subject_structure` |
| `scripts/analysis/analyze_pm_temporal_structure.py` | `bench.analysis.pm_temporal_structure` |
| `scripts/analysis/run_pm_eeg_lag_sweep.py` | `bench.analysis.pm_eeg_lag_sweep` |

CLI-файлы содержат только минимальный bootstrap корня репозитория и вызов
`runpy.run_module(..., run_name="__main__")`. Форматы аргументов и выходных
артефактов сохранены. Добавлен `tests/test_pm_analysis_entrypoints.py`, который
проверяет импортируемость, thin-CLI границу, record-safe lag pairing,
finite-pair filtering и Q3 subject diagnostics.

Операционные указания в локальном игнорируемом `AGENTS.md` синхронизированы с
каноническим путём `cogstate/model_zoo`; файл не входит в Git diff.

### Разрешённые конфликты

- PM-код адаптирован к удалению legacy `model_zoo`, а не наоборот.
- Научная логика вынесена из `scripts/`, не дублируется вторым pipeline.
- Существующие PM configs, результаты classification и runtime artifacts не
  перезаписывались.
- Сторонние untracked-файлы пользователя не удалялись и не редактировались.

## Preserved experiments

Сохранены и импортируются:

- основной PM target-validity audit и streaming/raw audit;
- postprocess, subject-structure и temporal-structure diagnostics;
- EEG lag sweep;
- classification `pm_eeg_lag_confirmatory`, его CLI/config/tests и результаты;
- regression `pm_eeg_lag_regression_confirmatory`, его CLI/config/tests и
  dry-run protocol.

Classification results не переобучались и не изменялись. Существующий набор
содержит 70 run directories / 70 summaries и подтверждает:

- 35/35 положительных delta Macro-F1, pooled mean delta
  `0.05300340847046983`;
- 35/35 положительных delta balanced accuracy, pooled mean delta
  `0.05627339902785858`;
- 7/7 PM имеют положительный mean delta по обеим метрикам.

Tracked diff внутри `reports/diagnostics/pm_eeg_lag_confirmatory_v1` отсутствует.

## Validation

### Syntax, imports и CLI

- `py_compile` для затронутых PM library/CLI/test modules: успешно;
- imports PM audit, streaming, analysis и обоих confirmatory experiments:
  успешно;
- `--help` для семи PM entry points: успешно;
- executable-import scan: активных `from/import model_zoo`, `automl` или `src`
  не найдено;
- активных импортов удалённых legacy `bench.automl` / `bench.meta` модулей не
  найдено.

### Tests

- PM-only suite: **33 passed**, 1 предупреждение о неизвестной pytest-опции
  `cache_dir`;
- объединённый targeted suite: **74 passed, 1 failed**, 1 warning;
- architecture suite без единственного unrelated thin-script guard:
  **14 passed, 1 deselected**, 1 warning;
- единственный targeted failure — architecture guard на пользовательском
  untracked `scripts/audit_raw_isactive.py` (1 149 строк); второй аналогичный
  untracked `scripts/audit_raw_isactive_v2.py` также сохранён. Эти файлы не
  относятся к PM semantic adaptation и сознательно не менялись.

Полный pytest выполнен один раз: **1 697 passed, 1 skipped, 7 failed,
45 errors, 40 warnings**. Ни один PM target-validity/lag test не входит в список
failures/errors. Оставшиеся случаи относятся к уже существующим
config/provenance и runtime/generated-artifact контрактам DANN, FOMAML,
personalization и robust-shrinkage, а также к указанному untracked thin-script
guard. Их исправление находится вне текущего scope; aggregate нельзя напрямую
сопоставлять с другим worktree, где доступен иной набор локальных artifacts.

### Regression dry-run

Dry-run выполнен во временный каталог; обучение не запускалось и существующие
результаты не перезаписывались.

| Инвариант | Значение |
|---|---:|
| canonical feature rows | 30 958 |
| feature count | 371 |
| dtype | `float32` |
| subjects | 54 |
| records | 86 |
| exact temporal pairs | 30 806 |
| first-window losses | 86 |
| temporal-gap losses | 66 |
| cross-record / subject / fold pairs | 0 / 0 / 0 |
| Attention complete / matched | 29 569 / 29 444 |
| each other PM complete / matched | 30 958 / 30 806 |
| planned fits | 70 |
| training executed | `false` |

Проверенные hashes:

- cache identity:
  `5062cac1e84a73c2e1c783f6b2c02f074e6177f2059515726e5b0433d166745f`;
- feature hash:
  `454ea5db886fce8c981dc28f8910ca07355956033b001550020a04fd4605832e`;
- fixed-fold hash:
  `2a176aad988fb814175c9edbf1e08266809f5c0a4ed4bf0dfd987ebf8c43c5dd`;
- dry-run protocol hash:
  `96b99b28533af365aa15b1a0464ce151ddbc34a51bac45645e4103acecfeb026`.

## Excluded artifacts

Сознательно не включались, не удалялись и не помещались в staging:

- `.codex_*` patches, scratch directories и temporary helpers;
- `pytest_tmp/` и временные test outputs;
- runtime caches и generated benchmark results;
- временные/локальные diagnostic reports;
- пользовательские PowerShell helpers;
- `scripts/audit_raw_isactive.py` и `scripts/audit_raw_isactive_v2.py`;
- любые иные unrelated user untracked files.

Git inventory содержит большое количество уже существовавших untracked runtime
деревьев; некоторые вложенные pytest-каталоги недоступны для перечисления из-за
Windows ACL. Поэтому `git ls-files --others --exclude-standard` сообщает
предупреждения доступа, а его числовой итог нельзя трактовать как полный
достоверный inventory. Ни один такой путь текущей задачей не очищался.

## Итог и риски

Финальная пакетная архитектура сохранена, PM-код использует канонические пакеты,
classification evidence не изменён, regression protocol воспроизводит все
зафиксированные dry-run инварианты. Основной оставшийся риск — несвязанные
локальные runtime/config artifacts и два пользовательских длинных untracked
скрипта, влияющие на полный/architecture pytest. До фиксации объединённого
дерева их следует рассмотреть отдельно, не смешивая с PM semantic adaptation.

