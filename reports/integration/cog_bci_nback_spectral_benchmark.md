# COG-BCI N-Back: 14- и 62-канальный спектральный benchmark

- Ветка: `integration/benchmark-unification`.
- HEAD: `5966a6133ba7a9c70dd24ef3f0bb5dce9b9a772b`.
- Статус: `diagnostic`.
- Исходные EEG, raw cache, task protocol и split manifests не изменены.

## Feature contract

Raw-сигнал; Welch 512/256, constant detrend. Для каждого канала вычислены log/relative band powers, theta/alpha, theta/beta и log variance. DC и 49–51 Hz доступны только в явно именованном `spectral_plus_nuisance`.

Окна агрегированы внутри каждой записи через mean, median, std и IQR. Основной объект оценки — 261 запись; split — исходный пятифолдовый GroupKFold по subject_id с готовым subject-disjoint inner split.

Размерности record-level feature sets: 14-channel channel-wise `728/896`,
62-channel channel-wise `3224/3968`, global summary для обеих политик
`260/320` (`spectral_only/spectral_plus_nuisance`).

SHA-256 всех входных manifests и channel contracts сохранены в
`benchmark_summary.json`; hashes до и после запуска совпадают.

## Pooled record-level metrics

| Policy | Representation | Model | Seed | BA | Macro F1 | Ordinal MAE | QWK |
|---|---|---|---:|---:|---:|---:|---:|
| cog_bci_common | channel_wise | hist_gradient_boosting | 42 | 0.4176 | 0.4161 | 0.7931 | 0.1315 |
| cog_bci_common | channel_wise | hist_gradient_boosting | 43 | 0.4176 | 0.4161 | 0.7931 | 0.1315 |
| cog_bci_common | channel_wise | hist_gradient_boosting | 44 | 0.4176 | 0.4161 | 0.7931 | 0.1315 |
| cog_bci_common | channel_wise | hist_gradient_boosting | 45 | 0.4176 | 0.4161 | 0.7931 | 0.1315 |
| cog_bci_common | channel_wise | hist_gradient_boosting | 46 | 0.4176 | 0.4161 | 0.7931 | 0.1315 |
| cog_bci_common | channel_wise | multinomial_logistic_regression | 42 | 0.4444 | 0.4441 | 0.7088 | 0.2493 |
| cog_bci_common | global_summary | hist_gradient_boosting | 42 | 0.4100 | 0.4081 | 0.7969 | 0.1319 |
| cog_bci_common | global_summary | hist_gradient_boosting | 43 | 0.4100 | 0.4081 | 0.7969 | 0.1319 |
| cog_bci_common | global_summary | hist_gradient_boosting | 44 | 0.4100 | 0.4081 | 0.7969 | 0.1319 |
| cog_bci_common | global_summary | hist_gradient_boosting | 45 | 0.4100 | 0.4081 | 0.7969 | 0.1319 |
| cog_bci_common | global_summary | hist_gradient_boosting | 46 | 0.4100 | 0.4081 | 0.7969 | 0.1319 |
| cog_bci_common | global_summary | multinomial_logistic_regression | 42 | 0.4253 | 0.4242 | 0.7510 | 0.1910 |
| emotiv_common | channel_wise | hist_gradient_boosting | 42 | 0.4330 | 0.4277 | 0.7510 | 0.1978 |
| emotiv_common | channel_wise | hist_gradient_boosting | 43 | 0.4330 | 0.4277 | 0.7510 | 0.1978 |
| emotiv_common | channel_wise | hist_gradient_boosting | 44 | 0.4330 | 0.4277 | 0.7510 | 0.1978 |
| emotiv_common | channel_wise | hist_gradient_boosting | 45 | 0.4330 | 0.4277 | 0.7510 | 0.1978 |
| emotiv_common | channel_wise | hist_gradient_boosting | 46 | 0.4330 | 0.4277 | 0.7510 | 0.1978 |
| emotiv_common | channel_wise | multinomial_logistic_regression | 42 | 0.4368 | 0.4364 | 0.7318 | 0.2006 |
| emotiv_common | global_summary | hist_gradient_boosting | 42 | 0.4176 | 0.4164 | 0.7739 | 0.1611 |
| emotiv_common | global_summary | hist_gradient_boosting | 43 | 0.4176 | 0.4164 | 0.7739 | 0.1611 |
| emotiv_common | global_summary | hist_gradient_boosting | 44 | 0.4176 | 0.4164 | 0.7739 | 0.1611 |
| emotiv_common | global_summary | hist_gradient_boosting | 45 | 0.4176 | 0.4164 | 0.7739 | 0.1611 |
| emotiv_common | global_summary | hist_gradient_boosting | 46 | 0.4176 | 0.4164 | 0.7739 | 0.1611 |
| emotiv_common | global_summary | multinomial_logistic_regression | 42 | 0.4483 | 0.4461 | 0.7433 | 0.1923 |

## Fold-level primary Logistic Regression

| Policy | Representation | Fold | Feature set | Features | BA | Macro F1 |
|---|---|---:|---|---:|---:|---:|
| emotiv_common | channel_wise | 1 | spectral_plus_nuisance | 896 | 0.3778 | 0.3647 |
| emotiv_common | channel_wise | 2 | spectral_only | 728 | 0.5370 | 0.5402 |
| emotiv_common | channel_wise | 3 | spectral_plus_nuisance | 896 | 0.4444 | 0.4284 |
| emotiv_common | channel_wise | 4 | spectral_plus_nuisance | 896 | 0.4444 | 0.4247 |
| emotiv_common | channel_wise | 5 | spectral_plus_nuisance | 896 | 0.3704 | 0.3432 |
| emotiv_common | global_summary | 1 | spectral_plus_nuisance | 320 | 0.3778 | 0.3013 |
| emotiv_common | global_summary | 2 | spectral_only | 260 | 0.4259 | 0.4117 |
| emotiv_common | global_summary | 3 | spectral_plus_nuisance | 320 | 0.5000 | 0.4997 |
| emotiv_common | global_summary | 4 | spectral_plus_nuisance | 320 | 0.5556 | 0.5544 |
| emotiv_common | global_summary | 5 | spectral_only | 260 | 0.3704 | 0.3315 |
| cog_bci_common | channel_wise | 1 | spectral_plus_nuisance | 3968 | 0.4444 | 0.4444 |
| cog_bci_common | channel_wise | 2 | spectral_only | 3224 | 0.4259 | 0.4215 |
| cog_bci_common | channel_wise | 3 | spectral_plus_nuisance | 3968 | 0.5185 | 0.5145 |
| cog_bci_common | channel_wise | 4 | spectral_only | 3224 | 0.4815 | 0.4792 |
| cog_bci_common | channel_wise | 5 | spectral_plus_nuisance | 3968 | 0.3519 | 0.3121 |
| cog_bci_common | global_summary | 1 | spectral_plus_nuisance | 320 | 0.4222 | 0.3966 |
| cog_bci_common | global_summary | 2 | spectral_only | 260 | 0.3519 | 0.3274 |
| cog_bci_common | global_summary | 3 | spectral_plus_nuisance | 320 | 0.5000 | 0.4978 |
| cog_bci_common | global_summary | 4 | spectral_only | 260 | 0.4815 | 0.4798 |
| cog_bci_common | global_summary | 5 | spectral_plus_nuisance | 320 | 0.3704 | 0.3396 |

Inner-selected Logistic `C` по folds:

- `emotiv_common/channel_wise`: `[0.1,10.0,1.0,0.1,0.1]`.
- `emotiv_common/global_summary`: `[10.0,10.0,10.0,0.1,0.1]`.
- `cog_bci_common/channel_wise`: `[1.0,10.0,1.0,0.01,0.1]`.
- `cog_bci_common/global_summary`: `[1.0,0.1,0.1,1.0,1.0]`.

HGB использовал две фиксированные конфигурации: simple
`lr=0.05/leaves=7/l2=0.001` и extended
`lr=0.08/leaves=15/l2=0.0001`; точный fold-level выбор сохранён в
`hyperparameter_selection.csv`.

## 62 минус 14 каналов

| Model | Representation | Seed | Δ BA | Δ Macro F1 | Δ Ordinal MAE | Folds 62>14 | Subjects 62>14 |
|---|---|---:|---:|---:|---:|---:|---:|
| multinomial_logistic_regression | channel_wise | 42 | +0.0077 | +0.0077 | -0.0230 | 3 | 9 |
| multinomial_logistic_regression | global_summary | 42 | -0.0230 | -0.0219 | +0.0077 | 1 | 5 |
| hist_gradient_boosting | channel_wise | 42 | -0.0153 | -0.0117 | +0.0421 | 2 | 9 |
| hist_gradient_boosting | channel_wise | 43 | -0.0153 | -0.0117 | +0.0421 | 2 | 9 |
| hist_gradient_boosting | channel_wise | 44 | -0.0153 | -0.0117 | +0.0421 | 2 | 9 |
| hist_gradient_boosting | channel_wise | 45 | -0.0153 | -0.0117 | +0.0421 | 2 | 9 |
| hist_gradient_boosting | channel_wise | 46 | -0.0153 | -0.0117 | +0.0421 | 2 | 9 |
| hist_gradient_boosting | global_summary | 42 | -0.0077 | -0.0082 | +0.0230 | 3 | 7 |
| hist_gradient_boosting | global_summary | 43 | -0.0077 | -0.0082 | +0.0230 | 3 | 7 |
| hist_gradient_boosting | global_summary | 44 | -0.0077 | -0.0082 | +0.0230 | 3 | 7 |
| hist_gradient_boosting | global_summary | 45 | -0.0077 | -0.0082 | +0.0230 | 3 | 7 |
| hist_gradient_boosting | global_summary | 46 | -0.0077 | -0.0082 | +0.0230 | 3 | 7 |

## Descriptive effects

| Policy | Representation | Feature set | Effect | Mean eta² | Median eta² |
|---|---|---|---|---:|---:|
| emotiv_common | channel_wise | spectral_only | class | 0.0061 | 0.0042 |
| emotiv_common | channel_wise | spectral_only | subject | 0.6564 | 0.6838 |
| emotiv_common | channel_wise | spectral_only | session | 0.0181 | 0.0134 |
| emotiv_common | channel_wise | spectral_plus_nuisance | class | 0.0055 | 0.0038 |
| emotiv_common | channel_wise | spectral_plus_nuisance | subject | 0.5982 | 0.6220 |
| emotiv_common | channel_wise | spectral_plus_nuisance | session | 0.0180 | 0.0146 |
| emotiv_common | global_summary | spectral_only | class | 0.0064 | 0.0045 |
| emotiv_common | global_summary | spectral_only | subject | 0.6547 | 0.6699 |
| emotiv_common | global_summary | spectral_only | session | 0.0165 | 0.0138 |
| emotiv_common | global_summary | spectral_plus_nuisance | class | 0.0057 | 0.0038 |
| emotiv_common | global_summary | spectral_plus_nuisance | subject | 0.5980 | 0.6247 |
| emotiv_common | global_summary | spectral_plus_nuisance | session | 0.0166 | 0.0146 |
| cog_bci_common | channel_wise | spectral_only | class | 0.0063 | 0.0045 |
| cog_bci_common | channel_wise | spectral_only | subject | 0.6631 | 0.6934 |
| cog_bci_common | channel_wise | spectral_only | session | 0.0165 | 0.0110 |
| cog_bci_common | channel_wise | spectral_plus_nuisance | class | 0.0057 | 0.0041 |
| cog_bci_common | channel_wise | spectral_plus_nuisance | subject | 0.6041 | 0.6422 |
| cog_bci_common | channel_wise | spectral_plus_nuisance | session | 0.0168 | 0.0137 |
| cog_bci_common | global_summary | spectral_only | class | 0.0061 | 0.0044 |
| cog_bci_common | global_summary | spectral_only | subject | 0.6365 | 0.6538 |
| cog_bci_common | global_summary | spectral_only | session | 0.0153 | 0.0115 |
| cog_bci_common | global_summary | spectral_plus_nuisance | class | 0.0054 | 0.0037 |
| cog_bci_common | global_summary | spectral_plus_nuisance | subject | 0.5783 | 0.5853 |
| cog_bci_common | global_summary | spectral_plus_nuisance | session | 0.0160 | 0.0146 |

## Subject bootstrap и решение

Primary comparison: `multinomial_logistic_regression` / `channel_wise`, seed 42. Δ balanced accuracy = `+0.0077`; 95% subject-bootstrap interval `[-0.0345, +0.0498]`, positive fraction `0.633`.

Решение: `retain_14_channel_cache`.

Bootstrap используется как диагностика устойчивости знака эффекта и не интерпретируется как доказательство статистической значимости.

## Сравнение с CNN и ограничения

Сохранённый 14-канальный CNN baseline имеет record balanced accuracy около 0.356 для EEGNet и ShallowConvNet. Настоящий benchmark сравнивает только лёгкие record-level модели и не запускает глубокое обучение.

Ограничения: 261 записи от 29 участников; признаки агрегируют полные record_full записи; acquisition units/filter history остаются частично неразрешёнными. Outer-test не использовался для выбора feature set, гиперпараметров или scaler.

Рекомендуемый следующий этап определяется `decision.json`; полный 62-канальный raw cache не строился.
