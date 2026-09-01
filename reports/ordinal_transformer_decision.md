# Решение по порядковому Transformer

- Selected decision: **3 — continue_with_both_ordinal_heads**.
- Selected ordinal method: `coral_and_corn`.
- Primary feature group: `eeg_pow`.
- Control feature group: `eeg_only`.
- Rationale: Decision 3 is retained despite its cost because both heads meet variant A, their primary strengths differ descriptively, and the seed-42 paired evidence does not justify excluding either head. Neither satisfies variant B, so this is a seed-stability check rather than a claim of ordinal superiority.
- Evidence supporting: coral: eeg_only:ordinal_mae, eeg_only:severe_error_rate; corn: eeg_only:ordinal_mae, eeg_only:severe_error_rate; Both heads satisfy variant A through Holm-confirmed EEG-only primary effects; the descriptive primary winner is split (CORAL 1/4, CORN 3/4), and no direct CORAL-vs-CORN secondary contrast is Holm-confirmed.
- Evidence against: coral: no Holm-confirmed primary improvement for EEG+POW; variant B=False; confirmed BA/F1 harm=none; corn: no Holm-confirmed primary improvement for EEG+POW; variant B=False; confirmed BA/F1 harm=none
- Remaining uncertainty: All inferential results use one initial state (seed 42); source and fold breakdowns are descriptive, and ordinal_argmax is diagnostic only.
- Next experiment: CORAL and CORN × EEG-only and EEG+POW, seeds 7 and 123; reuse seed 42 (eight new five-fold runs)
- Estimated new runs: 8.
- Experiments not recommended: POW-only repetition without a new scientific question; hyperparameter search before seed stability is known; changing the predeclared ordinal decoding rule; LSTM/BiLSTM or regression training within this analysis stage

Следующий эксперимент в этой задаче не запускался.
