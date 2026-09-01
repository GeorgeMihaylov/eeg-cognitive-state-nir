# Интеграция preprocessing и EEG features — 2026-08-18

## Область и источники

Интеграция выполнена семантически, без merge/rebase/cherry-pick, на базе
`integration/benchmark-unification` (`0079bde`). Сопоставлены текущая ветка,
`origin/feature/final-pipeline` (`74d6f75`) и read-only worktree
`F:/EEG-artifact-v2`. Полный FASTER в ветке коллеги и `full_faster.py` из
artifact-v2 имеют одинаковый SHA-256; за основу offline-реализации взят этот
проверенный общий вариант, дополненный rank-safe ICA diagnostics.

Тяжёлый preprocessing cache, 280-run benchmark и DL-обучение не запускались.

## Матрица решений

| Компонент | Текущая интеграция | feature/final-pipeline | artifact-v2 | Выбор и причина | Проверки |
|---|---|---|---|---|---|
| Filtering | строгая конфигурация, optional band-pass/notch, корректный streaming state/reset | полезный `apply_causal`, но слабее validation и optional-stage contract | не применимо | сохранена текущая реализация, добавлен делегирующий `apply_causal`; causal whole-record и chunks используют один код | Nyquist, startup, chunk equivalence, reset, disabled stages, invalid/non-finite data |
| Online FASTER | per-window bad-channel detection и mean interpolation, без epoch rejection/ICA fit | назван как общий FASTER | тот же lightweight режим выделен отдельно | сохранён для streaming как `apply_faster_online`; он никогда не отбрасывает будущие epochs и не обучает ICA | existing preprocessing tests и явный package API |
| Full FASTER | прежний `run_faster` был неполным (без ICA stage и rich report) | полный 4-stage FASTER | идентичен реализации коллеги и связан с record-local v2 cache | добавлен отдельный offline `full_faster.py`; package-level `run_faster` означает только полный вариант | bad channel/epoch/component/channel-within-epoch, original mapping, spherical interpolation, determinism, rank/convergence |
| Fixed ICA | rank-aware `ArtifactICA`, channel/shape checks, convergence properties | более простой ICA без rank contract | собственный более простой дубль | сохранён единственный текущий `ArtifactICA`; удалён дубль из перенесённого full module; v2 явно выбирает `component_metric_profile=full_faster`, legacy callers сохраняют прежнюю математику | rank reduction, deterministic fit, transform shape, convergence diagnostics |
| Referencing | отсутствовало как общий модуль | CAR, median, robust average | не применимо | перенесено; robust reference использует локальную variance-outlier оценку и не зависит от большого FASTER module | CAR zero mean, extreme-channel exclusion, no input mutation |
| Detrend/baseline/wavelet | отсутствовали как общий offline API | полноценные операции | не применимо | перенесены с усиленной finite/shape/config validation; PyWavelets является явной optional dependency | drift, baseline, deterministic denoising, short/invalid input |
| EOG regression | отсутствовала | fit/transform ridge regression | EOG correlation используется в component diagnostics | перенесена отдельной fit-dependent операцией; отсутствие EOG безопасно, EOG diagnostics и regression не смешаны | fit requirement, correlation reduction, channel mismatch |
| Offline pipeline | streaming pipeline был единственным общим entry point | composable pipeline, но требовал уточнения stage order и fit scope | full FASTER работает отдельно на record epochs | добавлен identity-by-default `OfflinePreprocessingPipeline`; порядок явно включённых стадий: filter → detrend → reference/bad-channel interpolation → fitted EOG → fixed ICA → optional wavelet; полный epoch rejection вызывается отдельно | stage report/order, causal equivalence, optional stages, finite output |
| Spectral features | shared Welch внутри extractor, deterministic order/schema | reusable spectrum, standard engagement, optional spectral-edge band | не применимо | добавлены `PowerSpectrum`, opt-in engagement и opt-in edge band; legacy defaults не изменены | constant/near-constant, Nyquist, standard engagement, band-limited edge |
| Connectivity | explicit computed pairs, all 91 by default, efficient coherence, broadband PLV | matrix API, pair budget, band-limited PLV | не применимо | сохранены validation/order/defaults; добавлены matrix coherence и opt-in band PLV. Uncomputed pairs остаются NaN и исключаются из summary | pair budget, finite constant case, band PLV, deterministic pairs |
| Entropy/statistical | зрелые finite guards, short-window/zero-variance semantics, configurable sample entropy | функционально уже покрыто, но с менее строгой общей validation infrastructure | не применимо | текущие файлы сохранены без изменения | существующие constant/noise/determinism tests |
| Feature pipeline/selection/streaming | schema/hash, deterministic names, leakage-safe selector, streaming profiles | существенно сокращены | не применимо | полностью сохранены; новые options автоматически входят в config/specification hash | 371-feature legacy contract, selection и streaming tests |
| Model bundle | feature/preprocessing hashes, imputer/scaler/selector, PM order/probability checks, provenance | упрощённый bundle + отдельное schema-version поле | не применимо | текущий bundle не изменён: предлагаемый patch удалял более сильные invariants; существующий hash contract уже отклоняет несовместимые bundles | `test_streaming_scientific.py` |
| Artifact ablation v2 | отсутствовал, был только v1 fold transform | отсутствовал | 30 958-window target-independent record-local cache, 280-cell protocol | v2 перенесён как отдельный эксперимент, v1 не изменён; source cache/experiment API сохранены и адаптированы к единственному rank-safe ICA | synthetic cache/resume/identity/hash/leakage tests; real-data tests skip при отсутствии local data links |

## A. Сохранено из integration/benchmark-unification

- `_validation.py`, canonical feature pipeline, selection и обе streaming feature
  схемы;
- default 371-feature profile, имена, порядок и `FEATURE_SCHEMA_VERSION`;
- optional band-pass/notch semantics и инициализация causal filter state первым
  отсчётом;
- rank-safe `ArtifactICA` и его public diagnostics;
- полный model-bundle contract с scientific hashes, PM ordering, imputer,
  normalization/selector и provenance;
- v1 artifact-removal ablation и все существующие benchmark/target/split contracts.

## B. Адаптировано из feature/final-pipeline

- `apply_causal` как точный whole-record эквивалент `StreamingFilter`;
- reusable Welch `PowerSpectrum`, стандартный engagement index и ограничиваемый
  spectral edge как opt-in возможности;
- matrix API для coherence и band-limited PLV как opt-in;
- referencing, detrending, baseline correction, wavelet denoising, EOG regression
  и composable offline pipeline;
- полезные требования тестов, но не сокращённая архитектура исходной ветки.

## C. Адаптировано из artifact-removal-v2

- полный четырёхстадийный FASTER: global channels → bad epochs → ICA components
  → channels within retained epochs;
- per-metric diagnostics, original epoch mapping, retained mask, spherical spline
  interpolation и explicit interpolation method;
- target-independent record-local cache с retained/rejected `sample_id`, atomic
  shards, implementation/config hashes, resume verification и no-fallback errors;
- отдельный v2 protocol: 7 PM × 4 variants × 2 tasks × 5 folds = 280, с
  target masking после preprocessing и inner validation по `record_group_id`.

В full FASTER сохранена математика v2: second-order Hurst estimator, mean PSD
power-gradient metric, pairwise mean channel correlation, channel kurtosis,
global epoch deviation и per-epoch/across-channel local outliers. Это исправляет
главное расхождение старого lightweight `detect_bad_channel_epoch_pairs`, где
нормирование выполнялось across epochs. Full ICA дополнительно ограничивает
`n_components` фактическим рангом и сообщает warning/diagnostics вместо
скрытого fallback.

## D. Намеренно отвергнутые изменения

- удаление `_validation.py` и сокращение pipeline/selection/streaming;
- замена полного model bundle упрощённым вариантом и перенос model imports в
  параллельный `cogstate.model_zoo`;
- изменение default PLV с broadband на band-specific, добавление engagement в
  default vector и изменение spectral-edge диапазона без schema migration;
- замена deterministic lexicographic pair budget другой политикой без нового
  config field;
- упрощённый ICA коллеги без rank/channel/convergence contract;
- spherical interpolation без coordinates: explicit `spherical` завершается
  ошибкой, `auto` документированно выбирает mean только при отсутствии позиций.

## E. Публичный preprocessing API

Новый package-level API разделяет режимы:

- causal: `FilterConfig`, `StreamingFilter`, `apply_causal`,
  `StreamingPreprocessingPipeline`, `apply_faster_online`;
- offline: `apply_offline`, `OfflinePreprocessingPipeline`, `run_faster` /
  `run_faster_full`, `FullFasterConfig`, reference/denoising/EOG API;
- learned fixed transform: единый `ArtifactICA` + `IcaConfig`.

Backward-compatible lightweight epoch helper доступен под явным именем
`run_faster_lightweight`. Внутренний streaming pipeline теперь также импортирует
online функцию под явным именем.

## F. Feature schema

`FEATURE_SCHEMA_VERSION` остаётся `cogstate-features-v1`: default count, names,
ordering и formulas не изменены. Новые возможности только opt-in и полностью
входят в `feature_specification`/hash:

- `SpectralConfig.include_engagement_index=true`;
- `SpectralConfig.spectral_edge_band_hz=[low, high]`;
- `ConnectivityConfig.plv_mode=band`.

Такой opt-in профиль обязан иметь новый feature hash и обучаться заново. Нельзя
загружать старый bundle с новым hash. Deployed legacy engagement formula в
immutable streaming-v1 профиле намеренно не менялась в этой интеграции.

## G. Leakage considerations

- Streaming preprocessing не использует будущие chunks; fixed ICA может только
  transform и должна быть обучена на разрешённой train/calibration части.
- `EOGRegression.fit`, feature selection, normalization и любая learned
  calibration выполняются только на outer/inner train согласно caller contract.
- Record-local unsupervised cleanup может видеть целую одну логическую запись,
  но не объединяет participants/`record_group_id` и не использует targets.
- v2 сначала формирует общий target-independent signal universe, затем кэширует
  record-local transform, и лишь после этого применяет PM complete-case mask.
- Outer folds остаются participant-disjoint, inner validation —
  `record_group_id`-disjoint; Q3 thresholds fit только по outer train.
- Retained и rejected identities сохраняются явно; silent raw fallback запрещён.

## H. Stale experiments

Существующие результаты default `cogstate-features-v1`, v1 artifact ablation и
streaming bundles не становятся stale. Любой будущий эксперимент, включающий
opt-in engagement, edge-band или band-PLV, требует нового feature materialization
и полного переобучения по новому hash.

Кэши v2 из другого worktree нельзя переносить как текущие: hash включает
реализации cache/full FASTER, а rank/convergence patch меняет implementation
SHA-256. Их следует пересоздавать только отдельной подтверждённой командой; в
этой задаче cache build не выполнялся.

## I. Оставшиеся риски

- PyWavelets доступен в текущем `eeg_benchmark` environment, но в репозитории нет
  единого dependency manifest; импорт package остаётся безопасным, а попытка
  создать wavelet config без зависимости даёт явную ошибку.
- Mean interpolation остаётся fallback без channel coordinates и не является
  spherical scalp interpolation.
- FastICA может не сойтись; состояние не подменяется, warning и diagnostics
  сохраняются. Научный runner должен заранее определить политику допуска таких
  records, а не менять thresholds постфактум.
- Real-data v2 plan tests требуют canonical data links в конкретном worktree;
  без них они корректно skip, synthetic invariants продолжают проверяться.
- Новые opt-in feature profiles пока не зарегистрированы как канонические
  experiment configs и не должны использоваться случайно.

## Проверки интеграции

- Baseline до изменений: 61 passed, 2 skipped.
- `py_compile`: успешно для всех изменённых Python-модулей с отдельным
  `PYTHONPYCACHEPREFIX` (обычный локальный `__pycache__` в worktree недоступен
  sandbox-пользователю).
- Итоговый целевой набор preprocessing/features/selection/streaming/model
  bundle: 134 passed, 6 skipped, 1 deselected. Исключён только старый v1
  real-plan тест, требующий отсутствующий canonical raw index.
- Повтор наиболее затронутых тестов после финального API rename: 75 passed,
  6 skipped.
- Общий suite: 1355 passed, 11 skipped; 77 failures и 74 setup errors вызваны
  отсутствующими в этом изолированном worktree datasets, runtime experiment
  artifacts/generated CSV и корневыми `AGENTS.md`/`PROJECT_CONTEXT.md`. Полный
  список был получен; regression в целевом наборе не обнаружено.
- Synthetic API smoke: identity offline shape `[512, 4]`; legacy profile
  371 features и hash
  `a06eb9e844c229366e604768c3e9a47a16790731e5be2b85622376f3bac2b493`;
  opt-in profile 397 features с другим hash; full FASTER сохранил 8/8 synthetic
  epochs; v2 matrix содержит ровно 280 run specs.
