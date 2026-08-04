# Итоговый пакет результатов EEG-бенчмарка

Дата консолидации: 2026-08-04. Этот документ агрегирует только уже
существующие runtime-артефакты. Обучение, перестроение кэшей и изменение
научных decision rules не выполнялись. Работа не объявляется полностью
завершённой.

## 1. Цель проекта

Единая воспроизводимая платформа для EEG/POW задач, subject-disjoint оценки,
персонализации, transfer/meta-learning и унифицированных артефактов.

## 2. Наборы данных

Основной benchmark объединяет `gpn_data` и `Old_EEG`; COG-BCI используется
как отдельный внешний диагностический трек. Источники Emotiv считаются
provenance-доменами, а не автоматически разными устройствами.

## 3. Каноническая выборка

Классификационная supervised-выборка содержит 45 384 окна, 54 участника и
пять классов `label_q5`. Raw-deduplicated DANN universe содержит 30 958 окон,
54 участника и 86 logical records с формой `[1, 14, 2560]`.

## 4. Схема валидации

Основной outer protocol — subject-disjoint GroupKFold. Inner validation,
персонализация, meta-episodes и DANN source validation используют отдельные
group-aware partitions; target-test не участвует в выборе модели.

## 5. Базовые модели

Random Forest и MLP остаются воспроизводимыми feature-window baselines.

## 6. Глубокие модели

LSTM, BiLSTM и Transformer используют временной контекст; EEGNet и
ShallowConvNet работают с raw окнами через общий adapter/encoder contract.

## 7. Preprocessing ablation

Factorial raw-EEG ablation не поддержала CAR как default для
ShallowConvNet; исходные численные решения не пересматривались.

## 8. Персонализация

Leakage-safe calibration отделяет calibration от final evaluation. Эффект
зависит от участника; full-model tuning не объявляется универсально лучшим.

## 9. Контрастивное обучение

Shape-only и time-aligned screening не улучшили downstream macro F1;
решение `close_transfer_track` сохраняется.

## 10. COG-BCI

14-channel cache сохранён; 62-channel expansion отклонён по заранее заданному
правилу. CNN и spectral результаты остаются diagnostic/negative evidence.

## 11. FOMAML

Participant-level outer-test: zero-shot macro F1
0.210094, supervised
full-model 0.198521, selected
FOMAML 0.152184. FOMAML против
supervised full-model: Δmacro F1
-0.046338,
Δbalanced accuracy
+0.039053,
Δordinal MAE
+0.449093;
W/L/T 1/4/0. Решение `do_not_proceed`. Это один fold, seed 42, пять
участников и EEGNet; инфраструктурная готовность не означает успех метода.

## 12. DANN

Диагностический fold 1 / seed 42: Δmacro F1 +0.013364, Δbalanced accuracy
+0.019079, Δordinal MAE −0.069330, W/L/T 6/2/0. Его bootstrap interval
включает ноль, поэтому статус — diagnostic `proceed`, не подтверждение.

## 13. Подтверждающий анализ

Primary analysis использует folds 1–5 и seeds 123/2026. DANN против
source-only: Δmacro F1
+0.008048,
Δbalanced accuracy
+0.008332,
Δordinal MAE
-0.034008.
Четыре из пяти folds и оба primary seeds положительны; 54.76% участников
улучшились, bootstrap 95% CI включает ноль. Решение `partially_confirmed`.
Seed 42 — sensitivity-only; fold 1 / seed 42 не переобучался и не входил в
primary decision. Всего выполнено 28 новых trainings.

## 14. Отрицательные результаты

Канонический список находится в `final_result_tables/negative_result_summary.csv`.
FOMAML `do_not_proceed` отделён от успешной episodic infrastructure.

## 15. Ограничения

Абсолютный macro F1 низок; source-validation содержит мало участников;
domain head значительно больше EEGNet; проверено только направление
`Old_EEG → gpn_data`; reverse direction и target-supervised upper bound не
выполнялись; эффекты неоднородны между участниками и seeds.

## 16. Требования проекта

Покрытие находится в `final_result_tables/requirement_coverage.csv` и
различает implementation, scientific evidence и незакрытые deliverables.

## 17. Воспроизводимость

Protocol/preregistration hashes, immutable unlock manifests, subject-level
splits и target-label firewall сохранены в runtime. Checkpoints,
predictions и кэши намеренно не отслеживаются Git.

## 18. Научные выводы

Проверенный FOMAML не поддержан. DANN показывает небольшой, но неоднородный
положительный эффект со статусом `partially_confirmed`; статистическая
значимость и полная доменная инвариантность не установлены.

## 19. Открытые направления

Нужны финальная публикационная интерпретация, presentation/demo scope и,
только при новой утверждённой гипотезе, reverse DANN или target-supervised
upper bound. Автоматические DANN/FOMAML sweeps не планируются.
