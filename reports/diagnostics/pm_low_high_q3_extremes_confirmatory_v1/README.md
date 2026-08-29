# PM LOW-vs-HIGH q3 extremes confirmatory v1

This preregistered protocol evaluates whether canonical EEG features distinguish
strongly LOW from strongly HIGH continuous PM states. The middle outer-train
tertile is excluded. Thresholds are fitted on continuous outer-train targets only
and applied unchanged to the paired outer-test targets.

- alignment: `EEG(t-10 s) -> PM(t)` only
- pairing: exact 10-second, record-local; gaps are excluded without substitution
- protocol hash: `ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431`
- canonical matrix: `30958 x 371`
- targets/folds/planned fits: `7 / 5 / 35`
- model: `XGBClassifier`
- hyperparameters: `{"n_estimators": 200, "n_jobs": 4, "random_state": 42}`
- participant-macro primary metrics: balanced accuracy, Macro-F1, ROC-AUC
- participant AUC policy: one-class subsets are undefined and excluded only from that metric
- training executed by dry-run: `false`

`results_by_fold.csv`, `summary_by_pm.csv` and `pooled_summary.csv` are created
only by an explicitly requested full run. Per-run predictions remain under
`runs/` as runtime artifacts.
