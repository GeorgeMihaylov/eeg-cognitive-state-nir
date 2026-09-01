# Каноническая матрица покрытия технического задания

Дата аудита: 2026-08-17. Это основной документ приёмочной трассировки текущей
ветки. Авторитетный исходный файл ТЗ с точными формулировками разделов 10.2 и
11 в tracked-дереве и доступной Git history не найден. Поэтому evidence и
фактические статусы установлены однозначно, а нормативная обязательность двух
неисполненных full benchmarks вынесена как вопрос руководителям.

Используется единая система статусов:

- `CLOSED` — реализация и необходимая проверка присутствуют;
- `PARTIAL` — часть требования закрыта, явно указан остаток;
- `NEGATIVE_RESULT` — эксперимент выполнен, количественная гипотеза не
  подтверждена;
- `HARDWARE_PENDING` — завершение требует физического EEG-устройства;
- `EXPERIMENT_PENDING` — нужен новый experiment, если соответствующая
  формулировка ТЗ обязательна;
- `DOCUMENTATION_PENDING` — требуется только оформление, не обучение.

## Сводка пунктов 10.2

| Пункт | Статус | Краткий итог |
|---|---|---|
| 10.2.1 Анализ и подготовка PM | CLOSED | Семь PM, raw/median/EMA/Hampel, 280-run downstream и complete-case правила проверены; raw PM выбран каноническим. |
| 10.2.2 Предобработка EEG | PARTIAL | Band-pass/notch/CAR ablation и fold-safe FASTER-like/ICA smoke есть; полного quantitative FASTER-like/ICA сравнения нет. |
| 10.2.3 Признаки и автоматический отбор | CLOSED | Spectral/statistical/entropy/connectivity реализованы; 448 → 50 LightGBM experiment завершён 140/140. |
| 10.2.4 Модели | PARTIAL | Model zoo и исторические full benchmarks есть; seven-PM selected-model confirmatory protocol подготовлен, но не выполнен. |
| 10.2.5 Перенос и персонализация | PARTIAL | Leakage-safe personalization, DANN, contrastive и FOMAML проверены; Accuracy ≥75% — валидный отрицательный результат. |
| 10.2.6 Работа в реальном времени | PARTIAL | Software replay/worker/inference/latency закрыты; physical end-to-end не проверен. |
| 10.2.7 Мультимодальный анализ | NEGATIVE_RESULT | Три набора и две model families проверены; универсальные +5–10% не подтверждены. |
| 10.2.8 Демонстрационный контур | PARTIAL | Replay, FastAPI и WebSocket работают без устройства; live headset demo отсутствует. |

## Детальная матрица 10.2

| Пункт ТЗ | Требование | Реализация | Эксперимент / evidence | Результат | Статус | Что ещё требуется |
|---|---|---|---|---|---|---|
| 10.2.1 | Семь PM: Attention, Engagement, Excitement, Stress, Relaxation, Interest, Focus | [`target_registry.py`](../../bench/tasks/target_registry.py), target audit, fixed continuous order | [`full_target_registry_audit.md`](../integration/full_target_registry_audit.md) | Все семь целей имеют provenance; complete-case 43 174 окна / 53 участника | CLOSED | Только поддерживать реестр |
| 10.2.1 | Raw PM, causal median, EMA, Hampel | [`pm_temporal_quality.py`](../../bench/analysis/pm_temporal_quality.py), config [`pm_temporal_quality_rf_final_v1.json`](../../experiments/pm_quality/pm_temporal_quality_rf_final_v1.json) | [`pm_temporal_quality_v1.md`](../pm_quality/pm_temporal_quality_v1.md); 280 metrics JSON | Raw: Macro F1 0.473036, BA 0.479122, MAE 0.098373; smoothing не даёт универсального выигрыша | CLOSED | Metadata migration legacy `result_status` — отдельная безопасная операция, не training |
| 10.2.1 | Missing values и сопоставимые cohorts | Target-specific complete cases внутри фиксированных folds; seven-output требует все 7 целей | [`pm_all_targets_feature_baseline.md`](../integration/pm_all_targets_feature_baseline.md) | Когорты не подменяют folds; single/multi-output сравниваются только на совпадающих samples | CLOSED | Нет |
| 10.2.1 | Поведенческие данные | Old_EEG annotation events и gpn markers инвентаризированы | PM temporal report, Behavioral audit | Old_EEG имеет time-spent/correctness; gpn markers без внешней семантики нельзя называть outcomes | CLOSED | Ограничение явно сохранять в тексте |
| 10.2.2 | Band-pass, notch, CAR, deduplication | Raw preprocessing registry и record-safe raw cache | [`preprocessing_selected_trials_multiseed.md`](../preprocessing_selected_trials_multiseed.md), [`preprocessing_factorial_ablation.md`](../preprocessing_factorial_ablation.md) | Raw наиболее стабилен; CAR −0.0285 BA в seed-42 factorial | CLOSED | Не делать CAR default |
| 10.2.2 | FASTER-like и ICA без leakage | [`fold_artifact_transform.py`](../../bench/preprocessing/fold_artifact_transform.py), outer-train-only ICA, per-window FASTER-like | `benchmark_results/artifact_removal_ablation_v1/smoke_report.md`, protocol hash `297eaf...` | 4/4 smoke complete; finite artifacts; mean-channel interpolation, не spherical FASTER | CLOSED | Для функционального требования достаточно |
| 10.2.2 | Количественное сравнение artifact removal | Full matrix заранее определена: 7 PM × 4 variants × 5 folds | `benchmark_results/artifact_removal_ablation_v1/protocol_manifest.json` | 140-run full result отсутствует | EXPERIMENT_PENDING | Выполнить только если ТЗ требует именно quantitative comparison; иначе не обязательно |
| 10.2.3 | Spectral/statistical/entropy/connectivity | Канонический target-free [`FeaturePipeline`](../../cogstate/features/pipeline.py) и group-specific modules | Feature tests и scientific streaming profiles | Все группы реализованы; lightweight исключает expensive entropy/connectivity | CLOSED | Подготовить компактный feature dictionary |
| 10.2.3 | EEG/POW и 448 исходных признаков | 168 EEG + 280 POW в каноническом parquet | [`README.md`](../../README.md), dataset/feature reports | 51 308 × 448 до target filtering | CLOSED | Нет |
| 10.2.3 | Автоматический fold-local отбор 448 → 50 | [`lightgbm_feature_selection.py`](../../bench/experiments/lightgbm_feature_selection.py), selector fitted только outer-train | `benchmark_results/lightgbm_feature_selection_v1/execution_manifest.json` | 140/140 complete; размерность −88.84%, fit ×6.78, качество немного ниже | CLOSED | Нет нового experiment |
| 10.2.4 | RF, LightGBM, XGBoost, MLP, LSTM, BiLSTM, Transformer, EEGNet, ShallowConvNet, ordinal | Общий [`model_zoo`](../../model_zoo/) и factory | [`experiment_summary.md`](../summary/experiment_summary.md), raw/Transformer/ordinal reports | Все перечисленные модели реализованы и имеют smoke или benchmark evidence | CLOSED | Поддерживать compatibility tests |
| 10.2.4 | Полноценные исторические benchmarks | Five-fold `label_q5`, raw CNN multiseed, seven-PM RF/LightGBM baselines | [`colleague_metrics_summary.md`](../summary/colleague_metrics_summary.md) | Валидные baselines присутствуют, но решают разные target/representation задачи | CLOSED | Не смешивать рейтинги разных cohorts |
| 10.2.4 | Selected-model seven-PM confirmatory | 280-cell plan, 245 supported training units; 224 новых trainings | [`pm_confirmatory_benchmark_plan.md`](../pm_confirmatory_benchmark_plan.md) | Plan-only complete, обучение не выполнялось | EXPERIMENT_PENDING | Нужен, если ТЗ требует confirmatory cross-model сравнение всех семи PM; обязательность требует решения руководителей |
| 10.2.5 | Leakage-safe personalization | Chronological calibration/final evaluation, seeds 7/42/2026 | [`personalization_multiseed_20pct.md`](../integration/personalization_multiseed_20pct.md), [`pm_regression_personalization_multiseed_20pct.md`](../integration/pm_regression_personalization_multiseed_20pct.md) | Семь PM: MAE gain 0.002685; classification effect мал и heterogeneous | CLOSED | Нет новых budget sweeps без гипотезы |
| 10.2.5 | Accuracy ≥75% после personalization | Тот же 53-subject multiseed protocol | Participant and per-seed aggregates | Средняя Accuracy ≈0.3138, max 0.634921, 0/53 ≥0.75 | NEGATIVE_RESULT | Ничего не переобучать ради формального «закрытия» |
| 10.2.5 | Cross-source/DANN | Target-label firewall, matched update budget, fixed source validation | [`dann_label_q5_confirmatory_v2.md`](../integration/dann_label_q5_confirmatory_v2.md) | ΔMacro F1 +0.008048; CI включает 0; `partially_confirmed` | CLOSED | Reverse direction только по новой гипотезе |
| 10.2.5 | Contrastive и FOMAML | Shared encoder, protected downstream, episodic/raw protocols | COG-BCI transfer и [`fomaml_label_q5_raw_diagnostic.md`](../integration/fomaml_label_q5_raw_diagnostic.md) | Contrastive без устойчивого gain; FOMAML ΔMacro F1 −0.046338, `do_not_proceed` | NEGATIVE_RESULT | Не продолжать без новой гипотезы |
| 10.2.6 | 10 s window, 1 s update, replay/LSL, quality, postprocessing | [`apps/streaming_worker`](../../apps/streaming_worker/), scientific configs | `benchmark_results/streaming_scientific_*_v1/run_summary.json` | Lightweight 336: Total P95 12.215 ms; full 399: 3052.311 ms | CLOSED | Online default — lightweight |
| 10.2.6 | Physical end-to-end latency | LSL source готов, но headset chain не запускалась | Scope note в latency manifests | Software latency не равна sensor-to-user latency | HARDWARE_PENDING | Реальный headset, LSL transport и измерение до API/UI |
| 10.2.7 | EEG-only, peripheral-only, fusion на MEFAR/CL-Drive/CLARE | Единый external multimodal protocol; XGBoost и ShallowFusion | [`multimodal_external_dataset_recommendation.md`](../external_datasets/multimodal_external_dataset_recommendation.md) | XGB ΔMacro F1: +0.113961/+0.011120/−0.037978; Shallow: −0.070163/−0.112538 | CLOSED | Нет |
| 10.2.7 | Универсальное улучшение fusion 5–10% | Matched participant folds и modality baselines | Те же fold-level summaries | Не подтверждено; на MEFAR peripheral-only лучше fusion | NEGATIVE_RESULT | Формулировать dataset-specific |
| 10.2.8 | Software demo без устройства | Replay worker, FastAPI health/status/latest/start/stop, WebSocket | [`apps/streaming_worker/api/README.md`](../../apps/streaming_worker/api/README.md), API snapshots | Демонстрационный software path реализован | CLOSED | При необходимости добавить presentation polish |
| 10.2.8 | Live headset demo | LSL adapter существует | Физического запуска нет | Нельзя подтвердить acquisition и live end-to-end | HARDWARE_PENDING | Устройство и демонстрационный сеанс |

## Количественные требования раздела 11

| Требование | Критерий | Фактический результат | Статус | Вывод |
|---|---|---|---|---|
| Computational real-time | Обработка должна укладываться в update interval | Lightweight Total P95 12.215 ms при 1000 ms interval | CLOSED | Software latency проходит с большим запасом |
| Full feature online | 399-feature профиль в том же 1 s budget | Total P95 3052.311 ms | NEGATIVE_RESULT | Full profile не использовать online без оптимизации |
| Personalized Accuracy | Accuracy ≥0.75 | Mean ≈0.3138; max 0.634921; 0/53 passed | NEGATIVE_RESULT | Критерий проверен и не достигнут |
| Multimodal improvement | Универсальные +5–10% | Эффект меняет знак между dataset/model | NEGATIVE_RESULT | Универсальный критерий не выполнен |
| End-to-end latency | Headset-to-user/API chain | Не измерялась | HARDWARE_PENDING | Требует физического EEG-устройства |
| Demo prototype | Программный replay/API/WebSocket | Реализован и имеет runtime snapshots | CLOSED | Software demo готов; live headset — отдельный hardware row |
| Методические рекомендации | Формальный компактный набор рекомендаций | Собран в [`project_scientific_conclusions.md`](../integration/project_scientific_conclusions.md) | CLOSED | Включить раздел в финальный текст/презентацию |

## Что действительно осталось для закрытия ТЗ

### A. Требует оборудования

| Работа | Зачем / пункт ТЗ | Масштаб | Обязательность |
|---|---|---|---|
| Live headset end-to-end test | 10.2.6, 10.2.8 и end-to-end latency раздела 11 | Небольшой инженерный сеанс после доступа к устройству; запись latency/quality/error contract | Обязательно, если ТЗ требует физический контур; иначе явно оформить limitation |

### B. Требует нового эксперимента

| Работа | Зачем / пункт ТЗ | Масштаб | Обязательность |
|---|---|---|---|
| Selected-model seven-PM confirmatory | Закрыть строгую интерпретацию 10.2.4 как full cross-model comparison | 245 supported units, 224 новых trainings; оценка плана ≈2.34 GPU/CPU wall-hours без overhead | **Условно обязательно**; нужен ответ руководителей по исходной формулировке ТЗ |
| Quantitative FASTER-like/ICA ablation | Закрыть строгую интерпретацию 10.2.2 как сравнение качества artifact removal | 140 ShallowConvNet runs; существенно дороже smoke | **Условно обязательно**; не нужно для требования только об implementation/smoke |

Однозначный ответ по selected-model benchmark: научно полезен и нужен для
confirmatory утверждения «какая модель лучше на семи PM», но его формальную
обязательность репозиторий не доказывает. До решения руководителей статус —
`EXPERIMENT_PENDING`, не скрытый `CLOSED`.

Однозначный ответ по FASTER-like/ICA: для доказательства работоспособности
текущих fold-safe implementation + smoke достаточно; для количественного
вывода о влиянии на качество нужен full benchmark. Нормативный выбор между
этими трактовками требует текста ТЗ или решения руководителей.

### C. Требует только оформления

| Работа | Зачем / пункт ТЗ | Масштаб | Обязательность |
|---|---|---|---|
| Финальный текст, презентация и data/feature dictionary | Формальные deliverables и понятное описание 448/399/336/50 profiles | Низкий/средний; без обучения | Обязательно для сдачи |
| PM metadata migration | Синхронизировать legacy `result_status` manifests | Низкий; только после отдельной ручной команды и audit JSON | Желательно для provenance, не влияет на метрики |

Не входят в остаток: новые DANN/FOMAML/contrastive sweeps, 62-channel COG-BCI,
повтор personalization и поиск универсального multimodal gain. Они уже имеют
валидный частичный или отрицательный результат и требуют новой научной
гипотезы, а не «дозакрытия» ТЗ.
