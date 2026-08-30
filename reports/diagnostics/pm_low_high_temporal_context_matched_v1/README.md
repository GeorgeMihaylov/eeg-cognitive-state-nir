# PM LOW/HIGH matched temporal-context comparison v1

All models use the exact same 10-window-eligible LOW/HIGH target cohort.

Information sets:
- LightGBM: 371 features at EEG(t-10s)
- XGBoost: 371 features at EEG(t-10s)
- LSTM: EEG feature history t-100s..t-10s
- Transformer: last 8 windows t-80s..t-10s

Execution:
- matrix cells: 140
- new fits: 105
- reused LSTM fits: 35
- LSTM reuse allowed only after exact hash/config/artifact audit
- protocol hash: `e09f28dab2b37321dd665cc55653cfc08a5a29afc38927ee26bc2d2c6cc988e7`

Primary metric: participant-macro balanced accuracy.
Paired temporal-vs-tabular comparisons use identical participant/PM cohorts.
Clustered bootstrap resamples subject_id with 10,000 replicates, seed 42.

Personalization transition rule: rank all four models by mean participant-macro
balanced accuracy over 35 fold×PM rows; advance at most two, with the second
advancing only when within 0.01 absolute BA of the best.
