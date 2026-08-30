# PM LOW/HIGH personalized threshold v1

This experiment reuses completed XGBoost and LightGBM outer-test probabilities.
It does not retrain either base model and does not repeat model inference.

- PM: 7
- models: XGBoost, LightGBM
- budgets: 60 s exploratory, 120 s secondary, 300 s primary
- 30 s omitted from adaptation because feasibility found 0/378 cells with >=2 LOW and >=2 HIGH
- calibration eligibility: fully available budget and >=2 LOW + >=2 HIGH
- fixed evaluation: strictly after +300 s, >=20 extreme samples and both classes
- ineligible calibration: zero-shot threshold 0.5 fallback
- primary strategy: midpoint between calibration LOW/HIGH median probabilities
- sensitivity strategy: calibration balanced-accuracy maximizing threshold
- primary metric: balanced accuracy
- protocol hash: `578c359c6c56115aff8ccea29af18bf755989641464dc6672e22753349018af0`
- base training by dry-run: false
- base inference by dry-run: false
- threshold calibration by dry-run: false
