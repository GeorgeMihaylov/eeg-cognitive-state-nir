# Закрытые отрицательные результаты

Отрицательные результаты ниже считаются научными исходами проверенных
гипотез, а не ошибками реализации. Они не должны смешиваться с финальными
положительными результатами.

| direction | result | decision | status | report_path |
|---|---|---|---|---|
| raw-deduplicated FOMAML | Selected FOMAML reduced participant macro F1 by 0.046338 and increased ordinal MAE by 0.449093 versus supervised full-model adaptation. | do_not_proceed | closed_negative | reports/integration/fomaml_label_q5_raw_diagnostic.md |
| ShallowConvNet CAR | CAR reduced mean balanced accuracy in the factorial raw-EEG ablation. | Do not adopt CAR as the default for this dataset/model. | closed_negative | reports/preprocessing_factorial_ablation.md |
| ordinal Transformer losses | Ordinal objectives reduced ordinal errors but did not produce a stable balanced-accuracy gain. | Keep categorical Transformer as the primary classification reference. | closed_negative | reports/ordinal_transformer_multiseed_statistics.md |
| classification personalization | Full-model fine-tuning was not consistently superior to head-only tuning across subjects. | Report both; avoid claiming universal full-model superiority. | closed_negative | reports/integration/personalization_multiseed_20pct.md |
| COG-BCI CNN | EEGNet and ShallowConvNet only slightly exceeded the 0.333 three-class chance level. | Close CNN-only N-Back exploration. | closed_negative | reports/integration/cog_bci_nback_baseline.md |
| COG-BCI preprocessing | Filtering suppressed nuisance contamination but did not improve the inner criterion. | Do not run a broader preprocessing search. | closed_negative | reports/integration/cog_bci_nback_preprocessing_ablation.md |
| COG-BCI 62 channels | The 62-channel spectral advantage was +0.0077 balanced accuracy, below the +0.03 decision threshold. | retain_14_channel_cache | closed_negative | reports/integration/cog_bci_nback_spectral_benchmark.md |
| shape-only contrastive transfer | Pretraining did not outperform random initialization downstream. | Do not extend shape-only transfer to more folds or seeds. | closed_negative | reports/integration/cog_bci_contrastive_transfer_screening.md |
| time-aligned contrastive transfer | Physical time alignment improved representation diagnostics but not downstream macro F1. | close_transfer_track | closed_negative | reports/integration/cog_bci_time_aligned_transfer_screening.md |
