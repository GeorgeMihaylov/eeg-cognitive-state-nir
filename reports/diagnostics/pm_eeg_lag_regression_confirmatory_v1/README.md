# PM EEG lag regression confirmatory v1

This protocol compares continuous-PM regression from `X(t)` and `X(t-10s)`
on one exact, PM-specific matched target cohort per outcome.

The -10 s lag was fixed from the preceding classification lag analysis before
inspecting regression results. No regression-specific lag selection is performed.

- protocol hash: `96b99b28533af365aa15b1a0464ce151ddbc34a51bac45645e4103acecfeb026`
- canonical features: `371`
- canonical rows: `30958`
- targets: `7` continuous PM
- model: `XGBRegressor`
- hyperparameters: `{"n_estimators": 200, "n_jobs": 4, "random_state": 42}`
- seed: `42`
- planned fits: `70`
- training executed by dry-run: `false`
