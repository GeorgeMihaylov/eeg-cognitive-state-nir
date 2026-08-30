# PM LOW/HIGH personalization duration response v1

Confirmatory duration-response experiment.

- models: XGBoost + LightGBM
- base-model training: none
- base-model inference: none
- stored outer-test HIGH probabilities are reused
- calibration budgets: 300 / 600 / 900 seconds
- threshold strategy: median_midpoint only
- calibration eligibility: full budget and >=2 LOW + >=2 HIGH
- ineligible/non-separated calibration: zero-shot threshold 0.5 fallback
- common evaluation for every budget: strictly after +900 seconds
- fixed evaluation readiness: >=20 extremes and both classes
- expected evaluation-ready participant-PM cells: 364
- expected result rows: 2184
- primary contrast: 900 s minus 300 s
- secondary contrast: 600 s minus 300 s
- aggregation: PM within participant, then participants
- bootstrap: subject-clustered, 10,000 replicates

Protocol hash:
`14f6fb28ebd748a1f897df0df6d8e7e5a03302733cddde8c33017524d6335035`
