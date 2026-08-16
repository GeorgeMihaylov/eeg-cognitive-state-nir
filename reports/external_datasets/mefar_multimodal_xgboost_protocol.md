# MEFAR XGBoost: адаптация к общему external multimodal protocol

## Научный контракт

Новый experiment `mefar_multimodal_xgboost_v1` меняет только модель Random Forest на фиксированный XGBoost. Неизменными остаются:

- 23 участника и 46 participant/session samples;
- target `mefar_cfs_fatigue_binary`;
- порог Chalder Fatigue Scale: `< 12` — class 0, `>= 12` — class 1;
- class distribution 22/24;
- session-level fusion;
- feature contracts 56/57/113;
- пять существующих participant-disjoint folds;
- evaluation sample IDs;
- outer-train median imputation, отсутствие scaler и oversampling.

Существующий Random Forest namespace `benchmark_results/mefar_multimodal_v1` рассматривается как неизменяемый источник data/split contract. Его fold manifest копируется побайтно, а protocol, fold и sample hashes проверяются до построения нового плана.

## Идентификаторы

- Исходный RF protocol hash: `5a3339cab659e53f67b21da4b083191de9ce1e6c6a3eb7bb8bd593e852400ff3`.
- Fold manifest semantic hash: `c8b9e80fbe9978eb1252e9fea60d172eee86ed0c91db4689e75e9f7116310232`.
- Общий sample IDs hash: `c700c71a533686f949e544bc5a759821d5a22050ee99db8ab0f2f98fb13a9adf`.
- Новый XGBoost protocol hash: `2ab377f5c0bab1cf94373e1411bea99bfae0e0e0aa15665ebeb4b6d56970fa80`.

Новый hash намеренно отличается от RF protocol, поскольку модель и model specification изменились.

## Режимы и модель

Исторические MEFAR mode IDs сохранены без переименования:

| MEFAR mode | Общая semantic role | Признаки |
|---|---|---:|
| `eeg_only` | `eeg_only` | 56 |
| `wearable_only` | `peripheral_only` | 57 |
| `eeg_wearable` | `eeg_peripheral` | 113 |

`Attention`, `Meditation`, `Derived`, CFS и target-derived поля исключены из primary features.

Параметры XGBoost совпадают с CLARE/CL-Drive: 300 trees, depth 6, learning rate 0,05, subsample и colsample 0,8, `hist`, seed 42. Единственное техническое отличие для бинарной цели — `objective=binary:logistic` и `eval_metric=logloss` вместо трёхклассовых `multi:softprob`/`mlogloss`. Hyperparameter search отсутствует.

Матрица состоит из 15 units: три режима × пять существующих folds. Во всех режимах каждого fold test sample IDs идентичны, participant overlap равен нулю.

## Ограничение ShallowConvNet

MEFAR содержит NeuroSky-derived band powers, а не пригодный сырой многоканальный EEG tensor. Поэтому `--run-shallow` отклоняется до обучения с явной ошибкой; silent fallback отсутствует.

## Команды

```powershell
& 'C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe' scripts\run_external_multimodal_protocol.py --config experiments\external_datasets\mefar_multimodal_xgboost_v1.json --plan-only
```

Будущий полный XGBoost запуск:

```powershell
& 'C:\Users\George\miniconda3\envs\eeg_benchmark\python.exe' scripts\run_external_multimodal_protocol.py --config experiments\external_datasets\mefar_multimodal_xgboost_v1.json --run-xgboost
```

В текущей задаче обучение не выполнялось: `models_trained = 0`, `summary_xgboost.csv` и каталог `runs/` отсутствуют.
