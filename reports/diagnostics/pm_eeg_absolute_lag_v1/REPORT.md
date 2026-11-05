# Absolute EEG-window lag / LOW-HIGH PM

Status: **confirmatory**. Reference: **-10 s, retrained on common support**.
Protocol: `38a65e564a8103de5f3a34f5216af0aff1dddcbea2ee8484af4e3acd7adb1fdf`.

Primary endpoint is equal-weight participant balanced accuracy. Deltas below are
alternative minus -10. Fold spread is sample SD across paired fold deltas.
95% percentile CIs resample participants (10,000 draws, seed 42), never windows.
CIs are descriptive and unadjusted for multiple comparisons; training folds overlap.

| PM | Lag (s) | Participant delta BA | 95% CI | Fold mean / SD | Favorable folds | Status |
|---|---:|---:|---|---|---:|---|
| target_attention | -30 | -0.092595 | [-0.108172, -0.077494] | -0.092130 / 0.014159 | 0 | causal_previous_window |
| target_attention | -20 | -0.041301 | [-0.055273, -0.028152] | -0.041110 / 0.009497 | 0 | causal_previous_window |
| target_attention | +0 | -0.061977 | [-0.078949, -0.046478] | -0.061310 / 0.023120 | 0 | contemporaneous_at_pm_window_end |
| target_attention | +10 | -0.084395 | [-0.100438, -0.069063] | -0.083593 / 0.020916 | 0 | non_causal_diagnostic |
| target_engagement | -30 | -0.097888 | [-0.114202, -0.080810] | -0.098074 / 0.018799 | 0 | causal_previous_window |
| target_engagement | -20 | -0.047488 | [-0.060456, -0.033587] | -0.047966 / 0.015519 | 0 | causal_previous_window |
| target_engagement | +0 | -0.054492 | [-0.069074, -0.040011] | -0.054739 / 0.023939 | 0 | contemporaneous_at_pm_window_end |
| target_engagement | +10 | -0.091460 | [-0.110025, -0.071930] | -0.091890 / 0.031413 | 0 | non_causal_diagnostic |
| target_excitement | -30 | -0.250435 | [-0.274119, -0.226633] | -0.250817 / 0.012757 | 0 | causal_previous_window |
| target_excitement | -20 | -0.166176 | [-0.184369, -0.147704] | -0.166286 / 0.006789 | 0 | causal_previous_window |
| target_excitement | +0 | -0.116731 | [-0.142613, -0.092202] | -0.116714 / 0.008735 | 0 | contemporaneous_at_pm_window_end |
| target_excitement | +10 | -0.236137 | [-0.260378, -0.212211] | -0.236640 / 0.016479 | 0 | non_causal_diagnostic |
| target_stress | -30 | -0.139394 | [-0.163960, -0.113680] | -0.139011 / 0.021088 | 0 | causal_previous_window |
| target_stress | -20 | -0.089471 | [-0.109901, -0.067616] | -0.089688 / 0.021021 | 0 | causal_previous_window |
| target_stress | +0 | -0.063537 | [-0.081784, -0.043721] | -0.063538 / 0.023541 | 0 | contemporaneous_at_pm_window_end |
| target_stress | +10 | -0.131542 | [-0.152783, -0.109057] | -0.131558 / 0.017965 | 0 | non_causal_diagnostic |
| target_relaxation | -30 | -0.158797 | [-0.182578, -0.134487] | -0.159903 / 0.029477 | 0 | causal_previous_window |
| target_relaxation | -20 | -0.096120 | [-0.112758, -0.079359] | -0.096944 / 0.026620 | 0 | causal_previous_window |
| target_relaxation | +0 | -0.103127 | [-0.121477, -0.084742] | -0.103547 / 0.016455 | 0 | contemporaneous_at_pm_window_end |
| target_relaxation | +10 | -0.160294 | [-0.184340, -0.135658] | -0.161109 / 0.023302 | 0 | non_causal_diagnostic |
| target_interest | -30 | -0.148557 | [-0.172723, -0.123688] | -0.149045 / 0.024561 | 0 | causal_previous_window |
| target_interest | -20 | -0.113342 | [-0.130022, -0.096045] | -0.113679 / 0.023411 | 0 | causal_previous_window |
| target_interest | +0 | -0.070359 | [-0.086960, -0.054649] | -0.070457 / 0.010136 | 0 | contemporaneous_at_pm_window_end |
| target_interest | +10 | -0.135260 | [-0.157728, -0.112442] | -0.135659 / 0.015552 | 0 | non_causal_diagnostic |
| target_focus | -30 | -0.009141 | [-0.025134, +0.007392] | -0.008929 / 0.011861 | 1 | causal_previous_window |
| target_focus | -20 | -0.000246 | [-0.012423, +0.011740] | -0.000005 / 0.010900 | 3 | causal_previous_window |
| target_focus | +0 | -0.082360 | [-0.098578, -0.067007] | -0.082169 / 0.018186 | 0 | contemporaneous_at_pm_window_end |
| target_focus | +10 | -0.111162 | [-0.129552, -0.092127] | -0.110943 / 0.009535 | 0 | non_causal_diagnostic |

## Interpretation rules

A. Assess the canonical -10 reference using all five folds and paired participant intervals;
this experiment never automatically changes the canonical project alignment.
B. For -20/-30, examine both paired effect size and consistency across folds for each PM.
C. The 0 column measures contemporaneous-window differences relative to -10.
D. Positive +10 differences are non-causal diagnostic evidence only. A conspicuously high
+10 result warrants a separate audit of smoothing, temporal overlap and timestamp semantics;
these are hypotheses, not established mechanisms or a production recommendation.

Absolute lag means EEG-window start minus PM-window start. Lag 0 is available only at
PM-window end and is not a prediction at PM-window start. Proprietary PM internal support
and physiological delays are not established by this comparison.

LOW/HIGH are derived PM proxies. Thresholds use only outer-train continuous complete cases
under the existing contract; train/test retained support and labels are identical across lags.
Single-class participant AUC/AP remain undefined, with metric-specific valid counts.
