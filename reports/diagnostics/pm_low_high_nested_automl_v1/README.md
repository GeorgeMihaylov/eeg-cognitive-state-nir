# PM LOW/HIGH nested AutoML v1

Preregistered nested model-family/hyperparameter selection protocol.

- seven PM, one shared selected candidate per outer fold
- exact `EEG(t-10s) -> PM(t)` pairing and 371 canonical features
- five frozen subject-disjoint outer folds
- three deterministic subject-disjoint inner folds
- inner Q33/Q67 fitted on inner-train continuous labels only
- 13 distinct XGBoost + 13 distinct LightGBM candidates
- 2730 planned inner fits and 35 planned final outer fits
- dry-run training, inference and performance evaluation: false

Protocol hash: `0fc645050599906196d0dd31f8b44947444ac95291e361352b18f207c83acab0`
Candidate matrix hash: `4a0b9d390670c6121775c578a96171bea46329df070ae1fd33a455e6bf688e3b`
Inner split hash: `c05b6636cd47a1e17d56533846eabc05cc98948f4ace7c2373874c046f6a7d2b`
