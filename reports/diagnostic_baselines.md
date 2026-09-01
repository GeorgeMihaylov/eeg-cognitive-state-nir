# Diagnostic baselines without EEG

All results use the canonical five subject GroupKFold partitions. Model fitting, one-hot encoding, and numeric scaling are confined to each outer-train partition. No EEG, POW, subject ID, record ID, future label, or test statistic is an input.

| Diagnostic | Model | Features | Accuracy | Balanced accuracy | Macro F1 | Ordinal MAE | Adjacent accuracy | Severe error rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 | majority_outer_train | outer-train majority | 0.171779 | 0.171779 | 0.148320 | 1.724727 | 0.483474 | 0.516526 |
| D1 | logistic_regression | source | 0.177970 | 0.177969 | 0.170379 | 1.785673 | 0.465032 | 0.534968 |
| D1 | random_forest | source | 0.177970 | 0.177969 | 0.170379 | 1.785673 | 0.465032 | 0.534968 |
| D2 | logistic_regression | normalized_record_progress, absolute_window_index, record_duration | 0.215935 | 0.215941 | 0.199647 | 1.617861 | 0.537392 | 0.462608 |
| D2 | random_forest | normalized_record_progress, absolute_window_index, record_duration | 0.214657 | 0.214656 | 0.207381 | 1.662150 | 0.512361 | 0.487639 |
| D3 | logistic_regression | source, normalized_record_progress, absolute_window_index, record_duration | 0.215120 | 0.215127 | 0.199658 | 1.619558 | 0.535012 | 0.464988 |
| D3 | random_forest | source, normalized_record_progress, absolute_window_index, record_duration | 0.213886 | 0.213885 | 0.202261 | 1.682686 | 0.503041 | 0.496959 |

D0 is the class mode from outer-train only. D1 uses source; D2 uses normalized record progress, zero-based absolute window index, and record duration; D3 combines D1 and D2. Record progress and duration use complete record metadata and are therefore retrospective covariates, not necessarily available in an online setting.

The strongest pooled accuracy is the D2 logistic control at `0.215935`, a delta of `+0.044157` from D0. Its balanced-accuracy delta from D0 is `+0.044162`. Adding source in D3 does not improve that result, so the modest signal is primarily record-position/duration structure rather than acquisition source.

Source-stratified and subject-stratified metrics are retained in `diagnostic_metrics.json`; fold means and standard deviations are retained in the summary JSON.

These controls quantify target structure and acquisition context. They are not evidence that a cognitive state is decoded from EEG.
