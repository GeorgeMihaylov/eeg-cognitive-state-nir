# PM LOW/HIGH long-duration personalization feasibility v1

No model training, inference, or performance evaluation.

Frozen question:
- budgets: 900 / 1200 / 1800 seconds (15 / 20 / 30 minutes)
- control: 900 seconds
- common evaluation: strictly after +1800 seconds
- earliest logical record only; no record stitching
- a budget is fully available only when both source duration and canonical
  feature-grid span cover the complete interval
- threshold contract remains outer-train Q33/Q67
- temporal alignment remains EEG(t-10s) -> PM(t)
- future threshold strategy remains median_midpoint
- future models remain XGBoost + LightGBM
- calibration support reports 2+2, 3+3, 5+5, and 10+10 LOW/HIGH

Protocol hash:
`34e0aa3350f84198383cd0e6a1d213711983132dcc30aa14fcc9edaafbc1095f`
