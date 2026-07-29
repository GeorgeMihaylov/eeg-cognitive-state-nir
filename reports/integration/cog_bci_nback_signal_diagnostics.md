# COG-BCI N-Back: диагностика слабого сигнала

## 1. Состояние репозитория

- Ветка: `integration/benchmark-unification`.
- Исходный HEAD: `bb81e16 feat(experiments): add COG-BCI N-Back baselines`.
- Исходное рабочее дерево и staging area: чистые.
- Статус результатов: `diagnostic`.

Исходные EEG, cache, target protocol, outer/inner assignments и существующие
прогнозы не изменялись. Commit, push, merge, rebase, reset, переключение ветки
и операции со staging area не выполнялись.

## 2. Неизменность входных артефактов

SHA-256 до и после диагностики совпадают:

| Артефакт | SHA-256 |
|---|---|
| `window_index.parquet` | `d9ec8addfb08e97b2de1e4eade9dc278691929c6869baa417b5b8042045535fa` |
| `task_definition.json` | `517f238d2880dbcd3cefb7bdfc88a4fad1c3cd70bf3965f4fa889c9106b517b7` |
| `target_index.parquet` | `a2f265a983a6517f0c3deb8fe7913da85fbd786cafa08d8ed2a8ad82adf2e5fa` |
| `outer_assignments.parquet` | `f66474c12b603fc045df7f73d06c23464682d5568228b5564aac1d075ecb38ca` |
| `outer_folds.json` | `f81052f3bdf21eb2aaf56dc0a0556d03623a7a4598ba83c139525833370cb512` |
| `inner_assignments.parquet` | `3737da49f33b2c2fe0ce3b812b6520e64f54db8f71890439b4b1d09dc9ba6d08` |
| `inner_folds.json` | `f4968dd0237d0f1900400d8ab4ac611addeefc68c27e317fcbae83c09ab6d328` |

Диагностические маски и признаки записаны только в отдельный runtime-каталог.

## 3. Аудит физических единиц

Проверены 12 файлов: `sub-01` и `sub-10`, две сессии, по одной записи
`zero_back`, `one_back`, `two_back`.

Установлено:

- EEGLAB header не содержит явного unit field;
- `EEG.data` и `EEG.datfile` ссылаются на внешний `float32` FDT;
- `EEG.ref` равен `common`;
- `EEG.chanlocs` не задаёт физические единицы;
- `EEG.etc` содержит служебные поля и версию EEGLAB, но не единицы;
- `MNE raw._orig_units` пуст;
- MNE назначает каналам FIFF unit `V`, calibration `1e-6`;
- прямые FDT-значения, умноженные на `1e-6`, точно совпадают с MNE output.

Поэтому корректный статус исходной физической единицы:

```text
physical_unit: unresolved
```

MNE output является вольтами по reader-конвенции, но это не доказывает, что
dataset-specific исходные значения действительно были записаны в микровольтах.
Прямой диапазон в проверенных фрагментах: `−16458.5 … 36494.9`; после MNE:
`−0.01646 … 0.03649`. Наблюдаемый коэффициент во всех файлах: `1e-6`.

## 4. Амплитуда, DC и межсубъектная вариативность

Для 3 654 комбинаций `record × channel` вычислены mean, median, standard
deviation, MAD, min/max, семь квантилей, near-zero fraction, DC и линейный
тренд.

- медианный абсолютный DC: `0.007878`;
- медианный within-record/channel standard deviation: `0.0001575`;
- отношение этих величин: `50.0`;
- медианный window-level `DC² / AC variance`: `15 955`;
- median DC по классам: `0.005893 / 0.005916 / 0.005690`;
- median DC по сессиям: `0.007956 / 0.004027 / 0.005649`;
- диапазон subject-level median DC: `−0.014294 … 0.014111`.

Классы почти не отличаются по DC, тогда как участники и сессии отличаются
сильно. Глобальная inner-train channel standardization исходного CNN baseline
не удаляет отдельный DC каждой записи или окна.

## 5. Спектральная диагностика

Для всех 16 927 окон вычислены 112 модельных признаков:

- log-power delta, theta, alpha, beta, low gamma по 14 каналам;
- log theta/alpha и theta/beta;
- log channel variance.

Welch использует одинаковые полосы, `nperseg=512`, `noverlap=256` и
`detrend=constant`. Все признаки конечны.

Дополнительный FFT-аудит показал:

- median power ratio `49–51 Hz / 1–45 Hz`: `4.526`;
- 99-й процентиль этого отношения: `1622.9`;
- robust total-power outliers с `|z|>6`: `0`;
- median log total 1–45 Hz по классам:
  `−2.866 / −2.835 / −2.781`.

Следовательно, 50-Гц компонента выражена очень сильно, но общий broadband
power не содержит небольшого числа экстремальных окон. Наибольшие
описательные class mean ranges наблюдаются для theta/alpha: до `0.285`
pooled standard deviation. Это не является статистическим доказательством
эффекта.

Для record-level mean spectral features медианная описательная доля дисперсии:

| Группировка | Median eta² | Mean eta² |
|---|---:|---:|
| Класс | 0.0046 | 0.0077 |
| Участник | 0.7670 | 0.7377 |
| Сессия | 0.0096 | 0.0111 |

Subject identity доминирует в абсолютной структуре признаков, а class effect
мал, но не равен нулю.

## 6. Границы задания

Для каждой из 261 N-Back записей существует ровно один class-specific end
marker (`601`, `611`, `621`). Он является последним событием:

- событий после end marker: `0`;
- median tail после marker: `0.004 s`;
- унифицированных task-start markers: `0`;
- первый non-boundary event: median `10.552 s`, диапазон
  `1.532 … 86.208 s`.

End boundary подтверждена. Начало задания по events не подтверждено, поэтому
5- и 10-секундные варианты остаются диагностическими масками, а не новой
канонической сегментацией.

## 7. Баланс metadata-масок

| Маска | Класс 0 | Класс 1 | Класс 2 | Всего |
|---|---:|---:|---:|---:|
| `record_full` | 5 579 | 5 638 | 5 710 | 16 927 |
| `to_end_marker` | 5 579 | 5 638 | 5 710 | 16 927 |
| exclude first 5 s | 5 492 | 5 551 | 5 623 | 16 666 |
| exclude first 10 s | 5 405 | 5 464 | 5 536 | 16 405 |

Пустых записей нет. End-mask не исключает accepted окна: текущая нарезка уже
заканчивается до подтверждённого marker.

## 8. Лёгкий subject-disjoint baseline

Использованы исходные пять outer folds и точные inner assignments. Scaler и
модели fit только на `inner_train`; outer test не участвовал в выборе признаков
или параметров.

| Модель | Accuracy | Balanced accuracy | Macro F1 | Ordinal MAE | QWK |
|---|---:|---:|---:|---:|---:|
| Multinomial logistic regression | 0.463602 | 0.463602 | 0.462664 | 0.701149 | 0.242254 |
| HistGradientBoosting | 0.429119 | 0.429119 | 0.425632 | 0.750958 | 0.207650 |

Logistic fold balanced accuracy: `0.356, 0.426, 0.519, 0.500, 0.500`.
Этот DC-устойчивый baseline существенно выше CNN baseline `0.356`, поэтому
разметка не является полностью неразделимой.

## 9. Within-subject/session-disjoint диагностика

В каждой из трёх ротаций две сессии всех 29 участников использованы для train,
третья — для test. Subject overlap (`29`) намеренный; record и sample overlap
равны нулю.

| Модель | Accuracy | Balanced accuracy | Macro F1 | Ordinal MAE | QWK |
|---|---:|---:|---:|---:|---:|
| Multinomial logistic regression | 0.463602 | 0.463602 | 0.462968 | 0.659004 | 0.329545 |
| HistGradientBoosting | 0.406130 | 0.406130 | 0.405627 | 0.727969 | 0.257143 |

Logistic balanced accuracy по held-out sessions:
`0.471 / 0.460 / 0.460`. Within-subject результат не выше
subject-disjoint, поэтому межсубъектная вариативность, хотя и очень велика,
не является единственным объяснением слабости исходных CNN.

## 10. Длительность записей

Диагностические terciles: `322.04 s` и `333.70 s`.

- duration vs class: Spearman `rho=0.165`;
- window count vs class: `rho=0.162`;
- EEGNet window count vs correctness: `rho=−0.008`;
- ShallowConvNet window count vs correctness: `rho=−0.018`;
- duration vs max probability: `−0.117` для EEGNet и `0.070` для
  ShallowConvNet.

P-values сохранены только как описательные и не использованы для выбора
модели. Correctness практически не связан с числом окон. Более длинные записи
дают больше train updates, но post-hoc short/medium/long метрики не показывают
монотонного улучшения.

## 11. Анализ существующих CNN-прогнозов

| Показатель | EEGNet | ShallowConvNet |
|---|---:|---:|
| Mean normalized entropy | 0.9955 | 0.8298 |
| Mean max probability | 0.3656 | 0.5627 |
| Median max probability | 0.3581 | 0.5424 |

EEGNet почти равномерно неуверен. ShallowConvNet существенно увереннее, но
accuracy не растёт вместе с confidence; в самом высоком confidence bin
accuracy составляет около `0.387`.

- модели согласны только на `29.9%` записей;
- обе верны: `11.1%`;
- обе ошибаются: `39.8%`;
- ровно одна верна: `49.0%`;
- mean Jensen–Shannon divergence: `0.0493`.

Это указывает не на единый набор безнадёжных записей, а на разные нестабильные
decision boundaries.

## 12. Ограниченная deep-проверка

После амплитудного аудита выполнена ровно одна разрешённая проверка:

```text
EEGNet
fold 1
seed 42
per-window channel mean removal
15 epochs maximum
unchanged inner split
```

| Fold 1 | Original | Per-window demean | Delta |
|---|---:|---:|---:|
| Record balanced accuracy | 0.400000 | 0.355556 | −0.044444 |
| Record macro F1 | 0.398652 | 0.328571 | −0.070080 |
| Record ordinal MAE | 0.844444 | 0.933333 | +0.088889 |
| Window balanced accuracy | 0.403577 | 0.333462 | −0.070115 |

Лучшей была эпоха 8, обучение завершило разрешённые 15 эпох. Простое удаление
DC не улучшило matched fold, поэтому постоянное смещение само по себе не
является достаточным объяснением.

## 13. Наиболее вероятная причина

Доступная разметка содержит умеренный спектральный сигнал: fixed-feature
logistic regression достигает `0.464` subject-disjoint balanced accuracy.
Главная проблема исходного CNN baseline — сочетание:

1. очень большой subject/record-specific DC и subject variance;
2. сильной 50-Гц компоненты в cache с preprocessing `none`;
3. слабого class effect относительно subject effect;
4. отсутствия подтверждённого task-start marker;
5. недостаточной устойчивости текущих CNN к этим nuisance-компонентам.

Фактическая полная слабость label и только межсубъектная вариативность не
согласуются со всеми наблюдениями.

## 14. Альтернативные объяснения

- Reader-конвенция `1e-6` может быть семантически неверна для конкретного
  источника, поскольку header не объявляет unit.
- Короткие 5.12-секундные окна могут недостаточно стабильно оценивать workload.
- Первые non-boundary events неодинаковы по времени; ранние окна могут включать
  служебный участок, но унифицированный start marker отсутствует.
- Архитектура и фиксированные гиперпараметры могли быть не оптимальны.
- Source preprocessing history остаётся неизвестной.

## 15. Ограничения

- Один seed для CNN и lightweight models.
- Within-subject protocol является диагностическим, не оценкой новых
  участников.
- Eta² и корреляции описательные, без correction for multiple comparisons.
- 5/10-second masks не использовались для model selection.
- Не выполнялись notch/band-pass CNN checks: они требуют заранее
  зафиксированного preprocessing contract.
- Не перестраивался cache и не исследовались 62 канала или MATB-II.

## 16. Решение о multi-seed

Полный multi-seed повтор исходного unfiltered CNN сейчас не рекомендуется.
Он с высокой вероятностью лишь уточнит вариативность слабой и методически
неоптимальной постановки. Инфраструктура готова, но preprocessing question
нужно закрыть до расхода compute.

## 17. Рекомендуемый следующий эксперимент

Сначала подтвердить исходные единицы и acquisition/filter history по
документации или producer metadata. Затем заранее зафиксировать один
leakage-safe preprocessing comparison без нового target/split:

```text
current raw
vs
DC removal + 1–45 Hz band-pass + 50 Hz notch
```

Проверить его сначала lightweight spectral baseline и одним EEGNet fold.
Только если улучшение воспроизводится без outer-test selection, переходить к
полному пятифолдовому и multi-seed запуску.

## Runtime artifacts

Runtime-результаты находятся в
`benchmark_results/cog_bci_diagnostics/nback_signal_audit/` и содержат unit,
amplitude, spectral, boundary, duration, prediction, lightweight baseline и
single-fold deep-check артефакты. Они игнорируются Git.

## Проверки

- `py_compile`: успешно;
- связанные COG-BCI/raw-model тесты: `102 passed, 5 warnings`;
- полный `pytest tests`: `930 passed, 13 warnings`;
- полный корневой `pytest`: `930 passed, 13 warnings`;
- `git diff --check`: exit code 0;
- staging area и `.gitignore`: без изменений.
