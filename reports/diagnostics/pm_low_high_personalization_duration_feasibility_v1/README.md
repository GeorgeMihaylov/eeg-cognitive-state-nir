# PM LOW/HIGH personalization duration feasibility v1

No model training, inference, or performance evaluation is performed.

Frozen duration question:
- calibration budgets: 300 / 600 / 900 seconds
- calibration record: earliest logical recording by selected-record UTC start
- calibration never crosses or stitches logical recordings
- no scanning forward until LOW/HIGH classes appear
- common evaluation for every budget: exact-lag targets strictly after +900 s
- outer-train Q33/Q67 thresholds remain unchanged
- alignment remains EEG(t-10 s) -> PM(t)
- later-record UTC overlap follows completed feasibility v1 trimming
- descriptive calibration support: both classes, 2+2, 3+3, 5+5
- future threshold method: median_midpoint only
- future models: XGBoost + LightGBM

Reference feasibility protocol:
`94c568d7e41344478c0550f573b0abf8893783831f6c7241b92c8e4fdd25c9cd`

Protocol hash:
`6bd91b39eef1869125e3f2c57125cfbef017f3b0c0a5538b135b4e32563075bb`

Dry-run:
- audit executed: false
- model training: false
- model inference: false
- performance evaluation: false
