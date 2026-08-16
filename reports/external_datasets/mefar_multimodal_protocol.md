# MEFAR: leakage-safe мультимодальный протокол

## Данные и уровень анализа

Исходный архив `data/raw/mefar/archives/MEFAR Dataset Neurophysiological and Biosignal Data.zip` имеет размер 90 189 241 байт и SHA-256 `c591ac136150032f58365248adbe52c68d063bc80a8846d22a32f29ad202048a`. В корпусе 23 участника и по две сессии каждого участника — morning и evening, всего 46 participant/session samples. Основной анализ остаётся session-level: один sample соответствует одной сессии участника; window-level fusion не используется.

## Целевая переменная

Основной target — `mefar_cfs_fatigue_binary`, построенный из Likert total 11-item Chalder Fatigue Scale:

- CFS < 12: class 0, `non_fatigue`;
- CFS >= 12: class 1, `fatigue`.

Значение каждой пары `participant_id + session_id` проверяется по двум местам книги `general_info.xlsx`: итоговой таблице `Subject List` и индивидуальному листу участника с ответами CFS. Все 46 соответствий совпали. CFS score и отдельные ответы CFS не входят в признаки. Morning/evening хранится только как metadata и диагностический covariate `mefar_session_time_proxy`; он не является target или feature в 15 основных запусках.

Баланс основной цели: class 0 — 22 сессии, class 1 — 24. Morning: 16/7; evening: 6/17. Между сессиями класс меняют 12 участников, в одном классе остаются 11.

## Признаки и fusion

- EEG-only: 56 признаков — семь session-level статистик для восьми физиологических band-power колонок;
- wearable-only: 57 признаков — session-level статистики BVP, EDA, TEMP, ACC x/y/z, HR и IBI плюс RMSSD;
- EEG + wearable: 113 признаков.

`Attention`, `Meditation`, `Derived`, исходный `class`, CFS/target-поля и morning/evening не входят в основной feature contract. `MEFAR_DOWN.csv`, `MEFAR_MID.csv` и `MEFAR_UP.csv` исключены из эксперимента. Пустые IBI-derived значения заполняются медианой, рассчитанной только на outer-train.

## Разбиение и модель

Сохранён существующий deterministic 5-fold `GroupKFold` по `participant_id`: каждый участник встречается в test ровно один раз, participant overlap во всех folds равен нулю. Распределение class 0 / class 1:

| Fold | Train | Test |
|---:|---:|---:|
| 1 | 18 / 18 | 4 / 6 |
| 2 | 18 / 18 | 4 / 6 |
| 3 | 16 / 20 | 6 / 4 |
| 4 | 17 / 21 | 5 / 3 |
| 5 | 19 / 19 | 3 / 5 |

Оба класса представлены в train и test каждого fold, поэтому разбиение пригодно для Macro F1 и Balanced Accuracy; переход на `StratifiedGroupKFold` не нужен.

Во всех трёх режимах используется один фиксированный Random Forest без подбора гиперпараметров: 300 деревьев, `max_depth=12`, `min_samples_leaf=2`, `random_state=42`. Oversampling и глобальное масштабирование запрещены. Матрица содержит 3 режима × 5 folds = 15 запусков на одинаковых sample IDs.

Основные метрики: Macro F1 и Balanced Accuracy. Дополнительные: Accuracy, per-class precision/recall и confusion matrix. Основной эффект — `MacroF1(EEG + wearable) - MacroF1(EEG-only)`; дополнительный — `MacroF1(wearable-only) - MacroF1(EEG-only)`.

Protocol hash: `5a3339cab659e53f67b21da4b083191de9ce1e6c6a3eb7bb8bd593e852400ff3`.

## Ограничения

MEFAR EEG — одноканальный NeuroSky-derived feature stream, а не многоканальный raw EEG. Сессии EEG используют относительное время, E4 — Unix UTC; общий надёжный marker отсутствует для 44 из 46 сессий. Поэтому протокол подтверждает только session-level multimodal fusion внутри MEFAR и не доказывает перенос результата на 14-канальный Emotiv benchmark.
