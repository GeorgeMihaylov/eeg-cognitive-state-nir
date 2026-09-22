# QC-driven F8/F7 channel-drop confirmatory experiment

## Status

**COMPLETE**

Experiment: `pm_qc_channel_drop_confirmatory_v1`

## Goal

Проверить, улучшает ли удаление каналов F8 или F7, заранее выбранных по label-free QC audit, межсубъектную классификацию LOW/HIGH.

Проверялись:

- `baseline_14ch`
- `drop_F8_13ch`
- `drop_F7_13ch`

После получения результатов дополнительные каналы не выбирались.

## Data contract

- frozen EEG windows: 30,958
- participants: 54
- outer folds: 5
- PM targets: 7
- baseline feature width: 371
- 13-channel feature width: 346
- фиксированное соответствие `EEG(t-10s) -> PM(t)`
- одинаковые sample IDs, folds, thresholds и LOW/HIGH/MIDDLE masks

## Primary endpoint

Participant-first balanced accuracy:

1. метрика рассчитывается для participant × PM;
2. PM усредняются внутри участника;
3. участники имеют одинаковый вес.

## Overall results

| Variant | Participant-first BA | Mean delta BA | 95% paired CI |
|---|---:|---:|---:|
| baseline_14ch | 0.753663 | — | — |
| drop_F8_13ch | 0.747378 | -0.006285 | [-0.011430; -0.001981] |
| drop_F7_13ch | 0.741902 | -0.011761 | [-0.028656; -0.001573] |

Paired bootstrap:

- unit: participant
- resamples: 10,000
- seed: 42

Для обоих вариантов 95% доверительный интервал изменения balanced accuracy полностью ниже нуля.

## Secondary metrics

Baseline:

- Macro-F1: 0.714697
- ROC-AUC: 0.878822
- PR-AUC: 0.867690

Drop F8:

- Macro-F1: 0.709736
- ROC-AUC: 0.874060
- PR-AUC: 0.865173

Drop F7:

- Macro-F1: 0.707310
- ROC-AUC: 0.872533
- PR-AUC: 0.862864

## Comparison with redundancy controls

Ранее были проверены:

- `drop_F3_13ch`: отрицательный эффект;
- `drop_F4_13ch`: примерно нейтральный эффект.

В текущем QC-driven эксперименте:

- `drop_F8_13ch`: отрицательный эффект;
- `drop_F7_13ch`: отрицательный эффект.

Таким образом, ни redundancy-driven, ни QC-driven удаление каналов не дало улучшения основной метрики.

## Interpretation

F8 и F7 были заранее выбраны по аномально высокой вариативности raw EEG.

Однако их удаление ухудшило participant-first balanced accuracy.

Следовательно, повышенная вариативность raw-сигнала сама по себе не означает, что канал мешает модели. Эти каналы, вероятно, сохраняют полезную информацию для межсубъектного прогноза.

В совокупности с анализом межканальных временных лагов и предыдущей F3/F4 абляцией результаты не поддерживают дальнейший post-hoc поиск удаляемых каналов или фиксированное temporal alignment.

## Validation

- targeted tests: 54 passed
- py_compile: PASS
- feature caches: complete
- required artifacts: present
- git diff --check: PASS
- screen: PASS

## Artifacts

- `summary_by_variant.csv`
- `summary_by_target.csv`
- `participant_deltas.csv`
- `fold_target_results.csv`
- `bootstrap_summary.csv`
- `comparison_with_f3_f4.csv`
- `run_manifest.json`
- `validation.json`

Feature caches:

`benchmark_results/pm_qc_channel_drop_confirmatory_v1`

## Conclusion

Ветка `channel removal / fixed inter-channel temporal alignment` считается завершённой в рамках текущего протокола.

Следующие исследования целесообразно направить на:

- межсубъектный сдвиг распределений;
- валидность и шум PM-таргетов;
- ограничения текущего представления EEG-признаков.
