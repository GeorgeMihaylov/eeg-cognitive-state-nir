# Научные выводы проекта

## Основная классификация

**Гипотеза.** EEG/POW и временной контекст позволяют предсказывать
пятиуровневый `label_q5` между испытуемыми. **Протокол.** Пятифолдовый
subject-disjoint GroupKFold, train-only preprocessing и group-aware inner
validation. **Результат.** Последовательные LSTM/BiLSTM/Transformer достигают
macro F1 около 0.36, превосходя RF/MLP и raw CNN baselines. **Решение.**
Transformer и recurrent модели остаются основными feature-based references.
**Ограничение.** Цель инерционна во времени и основана на глобальных
квантилях. **Статус для статьи:** основной результат с обязательным
sensitivity analysis разметки.

## Порядковая постановка

**Гипотеза.** Учёт порядка классов снизит тяжёлые ошибки без потери
категориального качества. **Протокол.** Три seeds, пять folds, subject-level
paired analysis; auxiliary weight выбирался только на inner validation.
**Результат.** CORN снижает ordinal MAE и severe-error rate, но balanced
accuracy не улучшается устойчиво; auxiliary policy также не поддержана.
**Решение.** Категориальный Transformer — основной baseline, CORN —
дополнительный анализ. **Ограничение.** Один набор и три seeds. **Статус для
статьи:** отрицательный/компромиссный результат.

## Регрессия и персонализация

**Гипотеза.** EEG+POW позволяют оценивать семь PM и адаптироваться к новому
пользователю. **Протокол.** Пятифолдовая RF-регрессия и leakage-safe
chronological 20% calibration. **Результат.** RF превосходит mean baseline;
full-model PM fine-tuning даёт небольшой устойчивый macro-MAE gain, но
классификационный full-model не универсально лучше head-only. **Решение.**
Для статьи показывать эффект и межсубъектную вариативность, не утверждать
универсальное превосходство полной настройки. **Ограничение.** Один бюджет
20%. **Статус:** основной результат с осторожной интерпретацией.

## COG-BCI

**Гипотеза.** Внешний N-Back корпус может подтвердить raw CNN, преимущества
62 каналов или перенос энкодера. **Протокол.** Record-safe caches,
subject-disjoint folds и заранее защищённый downstream fold. **Результат.**
CNN близки к chance; 62 канала дают только +0.0077 BA; shape-only и
time-aligned transfer не превосходят random initialization. Физическое
согласование улучшило contrastive representation diagnostics, но не
downstream. **Решение.** `retain_14_channel_cache`, `close_transfer_track`.
**Ограничение.** Transfer — screening на одном downstream fold. **Статус:**
диагностический отрицательный результат и приложение статьи.

## FOMAML и DANN

Эпизодическая инфраструктура и безопасные BatchNorm-контракты подтверждены
инженерно. В raw-deduplicated FOMAML diagnostic выбранная policy ухудшила
participant macro F1 и ordinal MAE относительно обычной supervised
full-model адаптации; решение — `do_not_proceed`. DANN в направлении
`Old_EEG → gpn_data` дал малый положительный средний эффект: четыре из пяти
folds и оба primary seeds положительны по macro F1. Статус
`partially_confirmed`: средний эффект ниже +0.01, win fraction ниже 60%, а
participant bootstrap interval включает ноль. Статистическая значимость не
установлена; source/target являются provenance-доменами, а не доказанно
разными устройствами.
