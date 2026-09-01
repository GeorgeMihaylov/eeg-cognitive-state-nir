# Итоговое состояние проекта

Дата актуализации: 2026-08-17. Состояние сверено с текущим кодом, configs и
доступными runtime manifests. Обучение, новые folds/seeds и перестроение кэшей
в ходе аудита не выполнялись.

## Закрытый контур

- Семь PM и исторический `label_q5` имеют явные target contracts.
- Основной outer protocol — participant-disjoint GroupKFold; inner validation
  и любые fitted transforms используют только train-группы.
- Реализованы feature-window, feature-sequence и raw-EEG модели, ordinal heads,
  единые predictions/manifests и resume-аудит.
- Temporal-quality RF matrix завершена: 280 runs; raw PM остаётся reference.
- LightGBM feature selection завершён: 140/140 runs, 448 → 50 признаков.
- Band-pass/notch/CAR ablation завершена; CAR не поддержан как default.
- Персонализация classification и семи PM проверена в leakage-safe
  chronological protocol и на нескольких seeds.
- DANN confirmatory, contrastive screening и FOMAML diagnostic завершены с
  честно сохранёнными частичными/отрицательными решениями.
- MEFAR, CL-Drive и CLARE имеют завершённые participant-disjoint multimodal
  сравнения.
- Streaming replay, worker, LSL source, модельный inference, quality checks,
  postprocessing, FastAPI и WebSocket реализованы; lightweight профиль
  вычислительно быстрее real-time.

## Ключевые решения

1. Главный научный контур — семь PM, а `label_q5` служит историческим
   Focus-specific benchmark.
2. Raw PM — канонический target-вариант: smoothing не дал универсального
   downstream преимущества.
3. 50-feature профиль полезен для ограниченных вычислений, но не повышает
   качество относительно 448 признаков.
4. Сложные neural models не гарантируют универсального выигрыша: результат
   зависит от representation, target и cohort.
5. Персонализация даёт небольшой средний эффект, но Accuracy ≥75% не
   достигнута.
6. DANN имеет статус `partially_confirmed`; contrastive track закрыт, FOMAML —
   `do_not_proceed`.
7. Multimodal fusion зависит от dataset и модели; универсального 5–10%
   улучшения нет.
8. Software real-time подтверждён только для lightweight профиля; физический
   end-to-end путь не проверен.

## Два условно незакрытых научных эксперимента

Полный selected-model seven-PM benchmark и полная quantitative
FASTER-like/ICA ablation подготовлены, но не выполнены. В tracked-дереве нет
авторитетного исходного текста ТЗ, поэтому их формальная обязательность не
может быть установлена по репозиторию. Если п. 10.2.4 требует именно
confirmatory сравнение выбранных моделей на всех семи PM, первый эксперимент
нужен. Если п. 10.2.2 требует количественно сравнить artifact-removal методы,
нужен второй. Если формулировки требуют только реализовать и функционально
проверить методы, текущих implementation + smoke достаточно.

## Что осталось безусловно

- live end-to-end проверка с физическим EEG-устройством;
- финальная presentation/report packaging и data/feature dictionary;
- решение научных руководителей по двум условным full benchmarks.

Каноническая матрица статусов:
[`final_requirement_coverage.md`](../requirements/final_requirement_coverage.md).
