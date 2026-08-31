# PM LOW/HIGH personalized weighted augmentation v1

Preregistered model-level personalization protocol. Dry-run only at this stage.

- all seven PM through one identical path
- exact `EEG(t-10s) -> PM(t)` temporal pairing
- canonical 371 engineered features
- fixed five subject-disjoint outer folds
- original outer-train Q33/Q67 LOW/HIGH thresholds
- fixed 1800-second earliest-record calibration; no stitching or extension
- middle consumes elapsed time and never enters binary fitting
- eligibility: full 1800 s + LOW>=10 + HIGH>=10 + fixed suffix ready
- method: subject-equivalent class-balanced weighted augmentation
- models: XGBoost and LightGBM with exact frozen hyperparameters
- expected operational / eligible / participants: 345 / 285 / 48
- planned personalized fits: 570
- dry-run fit, inference and performance evaluation: false

Protocol hash: `26a4d73a9c40aabffff8b1424e130331757add53bbf2c91513beb80f99dc6e69`
