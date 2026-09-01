# Персонализация многовыходной регрессии Performance Metrics, бюджет 20%

## Постановка

Проверена leakage-safe персонализация одной глобальной `torch_mlp` для семи
непрерывных Performance Metrics. Гипотеза состояла в том, что индивидуальная
калибровка уровня и масштаба PM может быть полезнее, чем для глобальных
квантильных классов. Сравнивались `zero_shot`, `bias_correction`,
`affine_calibration`, `head_only` и `full_model`.

Канонический complete-case набор содержит 43 174 окна, 53 испытуемых, 448
EEG + POW признаков и цели в фиксированном порядке:
`attention`, `engagement`, `excitement`, `stress`, `relaxation`, `interest`,
`focus`. Вход PM, `target_*`, `label_q5` и служебные столбцы не используются
как признаки.

## Протокол

- Outer evaluation: 5-fold `GroupKFold` по `subject_id`, seed 42.
- Глобальная модель обучается один раз на fold: всего 5 обучений.
- Inner validation: `group_holdout` по `subject_id`, fraction 0.15, seed 42.
- Preprocessing: `standard_clip`, percentiles 0.5/99.5, fit только на inner
  train; frozen state применяется к calibration, adaptation validation и
  final evaluation.
- Для каждого нового пользователя первые 20% детерминированно
  отсортированных окон (`source`, `record_id`, `t_start`, `sample_id`)
  образуют calibration pool, последние 80% — общий final evaluation.
- Calibration pool делится хронологически 80/20 на adaptation train и
  adaptation validation. Bias и Ridge используют только adaptation train.
- `head_only`: до 5 эпох, LR 0.001. `full_model`: до 5 эпох, LR 0.0001.
- Один model/split seed 42; paired bootstrap — 1000 ресэмплов по
  испытуемым.

Глобальная MLP имеет 66 183 параметра. При `head_only` обучается 455
параметров, 65 728 остаются frozen; при `full_model` обучаются все 66 183.
Loss — канонический MSE, нормализованный по числу samples × outputs.

## CUDA и выполнение

Запуск выполнен на PyTorch 2.11.0+cu128 и NVIDIA GeForce RTX 5060 Ti.
Пиковая память, зарегистрированная orchestration-run, — 20 028 416 байт
(пиковая память глобального fold — 20 894 720 байт). Глобальное обучение
заняло 18.85 с, персонализация — 25.08 с, полное выполнение с немедленным
обновлением объединённых Parquet/CSV после каждого условия — 133.37 с.

Все пять глобальных моделей обучались 8 эпох. Best validation loss по folds:

| Fold | Best epoch | Best validation loss | Training time, s |
|---|---:|---:|---:|
| 1 | 5 | 0.024299 | 4.96 |
| 2 | 8 | 0.016969 | 3.53 |
| 3 | 8 | 0.020921 | 3.46 |
| 4 | 8 | 0.018391 | 3.46 |
| 5 | 8 | 0.016763 | 3.43 |

Fine-tuning `head_only` использовал в среднем 4.74 ± 0.45 эпохи (диапазон
4–5), средний best validation loss 0.015054. `full_model` использовал
4.77 ± 0.42 эпохи (4–5), средний best validation loss 0.014507.

## Subject-level результат

Значения ниже — mean по 53 испытуемым; std отражает межсубъектную
вариабельность. Корреляционные macro-метрики усредняются только по
определённым целям, без замены NaN на ноль.

| Method | Macro MAE | Macro RMSE | Macro R² | Macro Pearson | Macro Spearman | Macro abs bias |
|---|---:|---:|---:|---:|---:|---:|
| zero_shot | 0.103848 ± 0.017078 | 0.131560 ± 0.019702 | -0.148883 ± 0.429485 | 0.413205 ± 0.101196 | 0.380569 ± 0.100487 | 0.042615 ± 0.015807 |
| bias_correction | 0.106756 ± 0.020944 | 0.135132 ± 0.022415 | -0.208332 ± 0.435793 | 0.413205 ± 0.101196 | 0.380569 ± 0.100487 | 0.050393 ± 0.021008 |
| affine_calibration | 0.113140 ± 0.023221 | 0.144498 ± 0.024761 | -0.265613 ± 0.333684 | 0.333112 ± 0.156077 | 0.313668 ± 0.146980 | 0.054581 ± 0.022947 |
| head_only | 0.102325 ± 0.016839 | 0.130435 ± 0.019587 | -0.140564 ± 0.500781 | 0.415344 ± 0.100738 | 0.383056 ± 0.100397 | 0.040315 ± 0.014132 |
| full_model | **0.101279 ± 0.016929** | **0.129105 ± 0.019458** | **-0.120986 ± 0.502735** | **0.427875 ± 0.106782** | **0.394836 ± 0.104918** | **0.039722 ± 0.013779** |

`full_model` дал лучшую среднюю macro MAE, R², Pearson, Spearman и absolute
bias. Улучшение macro MAE относительно zero-shot наблюдалось у 34/53
(64.15%) испытуемых; для `head_only` — у 32/53 (60.38%), bias — у 24/53
(45.28%), affine — у 15/53 (28.30%). В среднем full-model улучшил MAE у
4.21 из семи целей на пользователя; у 7 пользователей улучшились все цели,
у 30 — большинство целей.

## Paired gains и bootstrap

Положительный gain означает улучшение: уменьшение ошибки/bias или рост
R²/корреляции.

| Method vs zero-shot | Mean macro MAE gain | 95% bootstrap CI | Positive fraction | Mean Spearman gain | 95% CI | Mean abs-bias gain | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| bias_correction | -0.002908 | [-0.005937, -0.000245] | 45.28% | 0.000000 | [0, 0] | -0.007778 | [-0.012888, -0.002509] |
| affine_calibration | -0.009293 | [-0.013264, -0.005725] | 28.30% | -0.066901 | [-0.103261, -0.039012] | -0.011966 | [-0.018301, -0.006078] |
| head_only | 0.001523 | [0.000393, 0.002722] | 60.38% | 0.002487 | [0.000914, 0.004355] | 0.002300 | [-0.000293, 0.005273] |
| full_model | **0.002569** | **[0.001498, 0.003740]** | **64.15%** | **0.014267** | **[0.006376, 0.023523]** | **0.002893** | **[0.000568, 0.005468]** |

Full-model также превосходит head-only: средний paired macro MAE gain
0.001046, 95% CI [0.000276, 0.001851], positive fraction 66.04%;
Spearman gain 0.011780, CI [0.004662, 0.020138]. Это modest effect; интервалы
являются bootstrap-интервалами эффекта, а не заявлением о статистической
значимости.

## Семь показателей

Для `full_model` mean MAE gain положителен для всех семи целей:

| Target | Zero-shot MAE | Full-model MAE | Gain | Subjects improved |
|---|---:|---:|---:|---:|
| attention | 0.099411 | 0.098270 | 0.001141 | 58.49% |
| engagement | 0.095711 | 0.093593 | 0.002118 | 54.72% |
| excitement | 0.149897 | 0.142938 | **0.006959** | 79.25% |
| stress | 0.095353 | 0.094048 | 0.001305 | 54.72% |
| relaxation | 0.123884 | 0.121094 | 0.002790 | 69.81% |
| interest | 0.068531 | 0.068013 | 0.000518 | 52.83% |
| focus | 0.094147 | 0.090998 | 0.003149 | 50.94% |

Наибольший mean gain получен для excitement, наименьший — для interest.
Bias correction в среднем улучшила только excitement (+0.000539) и
relaxation (+0.000309), а остальные пять целей ухудшила. Affine calibration
ухудшила среднюю MAE всех семи целей; Ridge fallback не потребовался ни в
одном реальном условии.

## Анализ по поднаборам данных

Испытуемые распределены как `Old_EEG` — 12, `gpn_data` — 11, `both` — 30;
пользователи `both` не дублировались. Descriptive macro MAE:

| Method | Old_EEG | gpn_data | both |
|---|---:|---:|---:|
| zero_shot | 0.111392 | 0.097673 | 0.103094 |
| bias_correction | 0.120092 | 0.100735 | 0.103628 |
| affine_calibration | 0.134669 | 0.107394 | 0.106636 |
| head_only | 0.110765 | 0.095112 | 0.101594 |
| full_model | **0.110290** | **0.094024** | **0.100335** |

Это описательный анализ по поднаборам, не domain adaptation и не
cross-source transfer.

## Leakage, checkpoint и prediction audit

- Outer train/test subject overlap: 0 во всех folds.
- Target subject в global inner train/validation: 0/0 для всех условий.
- Calibration/final-evaluation overlap: 0.
- Adaptation train/validation overlap: 0; adaptation/final overlap: 0.
- Duplicate target sample IDs: 0.
- Все 53 пользователя и пять folds представлены.
- Завершено 265/265 условий, failed/incomplete: 0.
- Final evaluation идентичен для пяти методов каждого пользователя.
- Unified predictions: 1 209 565 long-строк =
  34 559 evaluation samples × 5 methods × 7 targets.
- Prediction key (`subject_id`, `sample_id`, `outer_fold`, `method`,
  `target_name`) уникален; все `y_true/y_pred_before/y_pred_after` конечны.
- `y_true` одинаков между методами; порядок семи targets канонический.
- Все independent clones начинают с global model state.
- Для zero-shot/bias/affine global state не изменяется.
- Для head-only frozen body hash не изменяется.
- Для full-model initial state совпадает с global checkpoint.

## Вывод

Простая гипотеза об индивидуальном постоянном смещении на этих
хронологических 20% данных не поддержана: bias correction и особенно
независимая affine Ridge-калибровка в среднем ухудшают качество. Вероятная
причина — calibration prefix недостаточно стабильно оценивает поздний уровень
PM при временной нестационарности; это требует отдельной проверки, а не
постфактум изменения протокола. Neural fine-tuning даёт небольшой, но
согласованный выигрыш, и полный fine-tuning лучше head-only по macro MAE и
Spearman. Основной кандидат следующего шага — `full_model`, но перед сильным
научным выводом необходимы дополнительные model/split seeds и, отдельно,
другие заранее заданные calibration budgets.

Ограничения текущего этапа: один seed, только MLP и только бюджет 20%.

## Артефакты

Runtime root:
`benchmark_results/pm_regression_personalization_20pct/`.

Главные файлы: `run_manifest.json`, `progress.json`, `global_fold_summary.csv`,
`personalization_subject_metrics.csv`, `target_metric_summary.csv`,
`aggregate_metrics.csv`, `paired_comparisons.csv`,
`calibration_parameters.csv`, `calibration_split_audit.csv`,
`checkpoint_audit.csv`, `predictions.parquet`. Каждый из 265 condition
каталогов содержит `model.pt`, `metrics.json`, `training_log.csv`,
`predictions.parquet`, calibration parameters, split audit и checkpoint audit.

