# PM LOW/HIGH long-duration personalization response v1

- models: XGBoost + LightGBM
- no base-model training or inference
- stored outer-test probabilities only
- budgets: 900 / 1200 / 1800 seconds
- common evaluation: strictly after +1800 seconds
- full budget: source duration AND canonical feature-grid span
- calibration: >=2 LOW and >=2 HIGH
- threshold: median_midpoint only
- fallback: threshold 0.5
- expected evaluation-ready participant-PM: 345
- expected result rows: 2070
- primary contrast: 1800 - 900 s
- secondary contrast: 1200 - 900 s
- participant-first aggregation
- subject-clustered bootstrap, 10,000 replicates

Protocol hash: `7fdc10bccad792c1f2d113ee063469230f229bd3dabf0cf0cb21e4f9b88e5caf`
