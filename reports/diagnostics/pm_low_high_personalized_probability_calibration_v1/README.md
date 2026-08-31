# PM LOW/HIGH personalized probability calibration v1

- 30-minute calibration budget only
- all seven PM
- XGBoost + LightGBM stored probabilities
- no base-model training or new inference
- candidate: intercept-only logit offset
- slope fixed to 1
- eligibility: full 1800 s + >=10 LOW + >=10 HIGH
- eligibility fixed from feasibility only: 285/345
- classification reference: frozen 30-minute median-midpoint policy
- classification fallback when logit-ineligible: frozen median-midpoint policy
- probability fallback when logit-ineligible: zero-shot probability
- primary metric: Brier score
- secondary probability metric: log loss
- secondary classification metrics: balanced accuracy and Macro-F1
- participant-first aggregation
- subject-clustered bootstrap: 10,000 replicates

Protocol hash: `d0a2e21e333a1ec70c9d1f5ca28604961d184ac8254f7f5fc93a8af62c70ba58`
