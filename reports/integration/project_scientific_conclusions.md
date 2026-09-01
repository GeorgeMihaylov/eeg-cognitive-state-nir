# Научные выводы проекта

## Цели и протокол

Главный контур проекта — семь непрерывных Performance Metrics: Attention,
Engagement, Excitement, Stress, Relaxation, Interest и Focus. Исторический
`label_q5` остаётся полезной Focus-specific задачей сопоставимости, но не
представляет всё пространство когнитивных состояний. Основная оценка —
subject-disjoint GroupKFold; preprocessing, thresholds и selection fitted
только на train.

## PM и признаки

Дополнительное causal smoothing уменьшает кратковременную вариативность PM, но
не даёт универсального downstream выигрыша. Raw PM имеет лучшие итоговые
classification, R² и correlation показатели и остаётся каноническим.

Fold-local LightGBM selection уменьшает 448 EEG+POW признаков до 50 и ускоряет
fit примерно в 6.78 раза, но немного ухудшает средние classification и
regression метрики. Это обоснованный lightweight/offline профиль, а не метод
повышения качества. Полный профиль следует сохранять там, где качество важнее
стоимости.

## Модели и порядковая постановка

LSTM, BiLSTM и Transformer сильнее оконных и raw-CNN baselines в историческом
`label_q5` benchmark, однако neural complexity сама по себе не гарантирует
выигрыш. CORN снижает ordinal errors, но не даёт устойчивого прироста Balanced
Accuracy; categorical Transformer остаётся основной reference-моделью, ordinal
варианты — sensitivity analysis.

Полный selected-model seven-PM protocol подготовлен, но не выполнен. Поэтому
preliminary fold-1 ranking нельзя переносить на все folds как confirmatory
вывод.

## EEG preprocessing

Band-pass/notch дают малые и seed-dependent изменения. CAR имеет отрицательный
средний описательный эффект по Balanced Accuracy (−0.0285) и не должен быть
default без новой гипотезы. Fold-safe FASTER-like/ICA реализованы и прошли
smoke; quantitative 5-fold effect неизвестен. Mean-channel interpolation не
следует называть канонической сферической FASTER-интерполяцией.

## Персонализация и перенос

Chronological personalization семи PM даёт небольшой воспроизводимый эффект:
full-model MAE улучшение 0.002685 и Spearman +0.011985. Эффект неоднороден
между участниками. Для `label_q5` Accuracy ≥75% не достигнута: средняя Accuracy
после full-model около 0.3138, наблюдаемый максимум 0.634921, достигших порога
участников 0/53.

DANN `Old_EEG → gpn_data` дал небольшой partially-confirmed эффект
(ΔMacro F1 +0.008048; bootstrap CI включает ноль). Источники —
provenance-домены, а не доказанно разные устройства. Contrastive transfer не
дал устойчивого downstream improvement; FOMAML ухудшил Macro F1 относительно
обычной supervised adaptation и получил `do_not_proceed`. Без новой гипотезы
новые FOMAML/contrastive sweeps не обоснованы.

## Мультимодальность

Fusion не имеет универсального знака эффекта. XGBoost улучшает Macro F1 на
MEFAR и немного на CL-Drive, но ухудшает на CLARE; ShallowFusion ухудшает
результат на обоих raw-EEG внешних наборах. На MEFAR wearable-only лучше
fusion. Следовательно, мультимодальность должна проектироваться и оцениваться
отдельно для каждого dataset/target/model.

## Потоковая обработка

Lightweight 336-feature streaming имеет Total P95 12.215 ms при 1-секундном
шаге; full 399-feature профиль имеет 3052.311 ms и в этот бюджет не
укладывается. Это software replay latency, не физическая end-to-end задержка.
Для online-контура рекомендован lightweight профиль; полноценная проверка
`headset → transport → processing → API/UI` требует устройства.

## Методические рекомендации

1. Использовать subject-level GroupKFold и явно сохранять outer/inner group
   audits.
2. Fit preprocessing, imputation, scaling, clipping, thresholds и feature
   selection только на train текущего fold.
3. Рассматривать семь PM как основной target-space; `label_q5` — только как
   историческую sensitivity-задачу.
4. Использовать raw PM как baseline; smoothing вводить лишь по новой
   предварительно сформулированной гипотезе.
5. Использовать 50-feature профиль при ограниченных ресурсах, а полный профиль
   — при приоритете качества.
6. Не включать CAR по умолчанию; FASTER-like/ICA не объявлять улучшающими
   качество до quantitative experiment.
7. Оценивать multimodal fusion dataset-specific и всегда сравнивать с каждой
   отдельной модальностью.
8. Не продолжать FOMAML/contrastive search без новой гипотезы и decision rule.
9. Для online-контура использовать lightweight streaming; full profile
   оставлять offline или оптимизировать отдельно.

Эти выводы не утверждают статистическую значимость там, где confidence
interval включает ноль или эксперимент имеет diagnostic/preliminary статус.
