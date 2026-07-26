# Статус исторических mixin-прототипов

| Mixin | Проверен | Запускается | Интегрирован | Решение | Причина |
|---|---:|---:|---:|---|---|
| Contrastive learning | tested | synthetic_only | not_integrated | defer | Encoder не подключён downstream; нужен shared encoder API, а 448 features не являются raw EEG. |
| Domain adaptation | tested | no | not_integrated | defer | Нет runner/source-target contract и обоснованной cross-device domain-задачи. |
| Meta-learning | tested | no | not_integrated | defer | Нет learn2learn, runnable task и обоснованного episodic protocol. |
| Transfer learning | tested | diagnostic_only | integrated_as_reimplemented_pipeline | keep | Старый prototype сбрасывал pretrained weights; назначение реализовано через leakage-safe fine-tuning. |

`TransferLearningMixin` не считается production-ready: его назначение интегрировано как заново реализованный leakage-safe fine-tuning pipeline. DANN, MAML и contrastive pretraining не интегрированы.
