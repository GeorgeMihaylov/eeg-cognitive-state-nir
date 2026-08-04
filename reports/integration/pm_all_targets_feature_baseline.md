# Canonical seven-PM feature baseline

## Status

`pm_feature_baseline_complete` on branch `integration/benchmark-unification`, commit `d3a56542f75ae326a83b5d1bc38411df791d0d52`.

## Protocol and preregistration

- Experiment: `pm_all_targets_feature_baseline_v1`
- Protocol hash: `41b2ef22cf69fda7dfbc6a6b5fd4a9b61f98b102e24853b77d1496bfd01967af`
- Run-matrix hash: `476cb5624a34c9f7b17bc0b381997be60b48607b7c2dd3a8c5010ad0e7f23a7b`
- Runs: 1125 complete / 1125 planned; 0 failed.
- Five immutable outer folds by `subject_id`; reference assignments match the existing label-Q5 benchmark.

## Targets and cohorts

Seven continuous PM targets and the fixed-order seven-output target are included. Cohort window counts: `{"pm_attention_regression": 43175, "pm_engagement_regression": 48254, "pm_excitement_regression": 50983, "pm_focus_regression": 45384, "pm_interest_regression": 45440, "pm_multioutput_regression_7": 43174, "pm_relaxation_regression": 45394, "pm_stress_regression": 45384}`. Target-specific complete cases are applied inside fixed folds.

## Features, models and seeds

EEG=168, POW=280, EEG+POW=448. Device POW columns are stored engineered power features, not spectra recomputed from raw EEG. PM, target, label and identity columns are excluded. Dummy mean, Ridge, Random Forest and single-output HistGradientBoosting are fixed baselines. Random Forest uses seeds 42, 123 and 2026; deterministic models run once. LightGBM was not installed and was not added.

## Leakage audit

Subject overlap is zero. Median imputation and Ridge scaling are fitted only on outer-train. Outer-test is not used for fitting, early stopping, selection or target statistics. The machine-readable audit is `global_leakage_audit.json`.

## Dummy mean, EEG+POW

| target_name | feature_set | model | mae_mean | rmse_mean | r2_mean | pearson_mean | spearman_mean | normalized_mae_mean | participants |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention | eeg_pow | dummy_mean | 0.096316 | 0.121170 | -0.089764 |  |  | 0.755824 | 53.000000 |
| engagement | eeg_pow | dummy_mean | 0.102767 | 0.130708 | -0.123634 |  |  | 0.780960 | 54.000000 |
| excitement | eeg_pow | dummy_mean | 0.183405 | 0.225757 | -0.239410 |  |  | 0.781520 | 54.000000 |
| focus | eeg_pow | dummy_mean | 0.098026 | 0.122009 | -35.481327 |  |  | 0.785942 | 54.000000 |
| interest | eeg_pow | dummy_mean | 0.069190 | 0.090142 | -0.099758 |  |  | 0.711325 | 54.000000 |
| relaxation | eeg_pow | dummy_mean | 0.135635 | 0.163698 | -1.706714 |  |  | 0.810155 | 54.000000 |
| stress | eeg_pow | dummy_mean | 0.098992 | 0.129968 | -0.705154 |  |  | 0.717249 | 54.000000 |

## Single-output PM regressions, EEG+POW

Random Forest rows combine all three preregistered seeds.

| target_name | feature_set | model | mae_mean | rmse_mean | r2_mean | pearson_mean | spearman_mean | normalized_mae_mean | participants |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention | eeg_pow | hist_gradient_boosting | 0.092163 | 0.114906 | -0.014396 | 0.490863 | 0.472630 | 0.723047 | 53.000000 |
| attention | eeg_pow | random_forest | 0.092387 | 0.115532 | -0.008261 | 0.425975 | 0.408044 | 0.724673 | 53.000000 |
| attention | eeg_pow | ridge | 0.142105 | 0.325919 | -56.188369 | 0.317381 | 0.411475 | 1.112983 | 53.000000 |
| engagement | eeg_pow | hist_gradient_boosting | 0.089059 | 0.112071 | 0.153835 | 0.519839 | 0.439529 | 0.676937 | 54.000000 |
| engagement | eeg_pow | random_forest | 0.090171 | 0.113567 | 0.131154 | 0.477355 | 0.394650 | 0.685446 | 54.000000 |
| engagement | eeg_pow | ridge | 0.160993 | 0.343006 | -88.891152 | 0.356400 | 0.361778 | 1.232095 | 54.000000 |
| excitement | eeg_pow | hist_gradient_boosting | 0.144968 | 0.184568 | 0.117289 | 0.561497 | 0.513175 | 0.617669 | 54.000000 |
| excitement | eeg_pow | random_forest | 0.145376 | 0.185198 | 0.147154 | 0.542323 | 0.501129 | 0.619370 | 54.000000 |
| excitement | eeg_pow | ridge | 0.204790 | 0.370611 | -16.072274 | 0.432605 | 0.479681 | 0.871770 | 54.000000 |
| focus | eeg_pow | hist_gradient_boosting | 0.091213 | 0.113994 | -50.924119 | 0.305550 | 0.284113 | 0.731261 | 54.000000 |
| focus | eeg_pow | random_forest | 0.091445 | 0.114649 | -55.487996 | 0.276326 | 0.243768 | 0.732953 | 54.000000 |
| focus | eeg_pow | ridge | 0.135299 | 0.329443 | -115.389938 | 0.228463 | 0.247416 | 1.079408 | 54.000000 |
| interest | eeg_pow | hist_gradient_boosting | 0.066819 | 0.084501 | -0.380038 | 0.460601 | 0.425461 | 0.686769 | 54.000000 |
| interest | eeg_pow | random_forest | 0.065438 | 0.083856 | -0.101102 | 0.449275 | 0.419789 | 0.672404 | 54.000000 |
| interest | eeg_pow | ridge | 0.103747 | 0.236122 | -47.088863 | 0.359388 | 0.390320 | 1.067422 | 54.000000 |
| relaxation | eeg_pow | hist_gradient_boosting | 0.115373 | 0.142957 | -3.013485 | 0.493587 | 0.465769 | 0.688724 | 54.000000 |
| relaxation | eeg_pow | random_forest | 0.115970 | 0.143728 | -2.105649 | 0.482583 | 0.461449 | 0.692414 | 54.000000 |
| relaxation | eeg_pow | ridge | 0.174728 | 0.362776 | -32.889051 | 0.364991 | 0.414345 | 1.048746 | 54.000000 |
| stress | eeg_pow | hist_gradient_boosting | 0.093683 | 0.122716 | -0.946682 | 0.444667 | 0.397773 | 0.678296 | 54.000000 |
| stress | eeg_pow | random_forest | 0.091675 | 0.121477 | -0.706163 | 0.448565 | 0.422294 | 0.663801 | 54.000000 |
| stress | eeg_pow | ridge | 0.170769 | 0.367456 | -100.489353 | 0.356983 | 0.366962 | 1.236735 | 54.000000 |

## Seven-output regression, EEG+POW

| target_name | feature_set | model | mae_mean | rmse_mean | r2_mean | pearson_mean | spearman_mean | normalized_mae_mean | participants |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attention | eeg_pow | dummy_mean | 0.096316 | 0.121170 | -0.089768 |  |  | 0.755822 | 53.000000 |
| attention | eeg_pow | random_forest | 0.094985 | 0.119421 | -0.060991 | 0.210573 | 0.200144 | 0.745397 | 53.000000 |
| attention | eeg_pow | ridge | 0.142749 | 0.327521 | -57.223495 | 0.317510 | 0.411388 | 1.118006 | 53.000000 |
| engagement | eeg_pow | dummy_mean | 0.101523 | 0.128813 | -0.179369 |  |  | 0.781147 | 53.000000 |
| engagement | eeg_pow | random_forest | 0.093415 | 0.117320 | 0.009206 | 0.383448 | 0.297260 | 0.718897 | 53.000000 |
| engagement | eeg_pow | ridge | 0.164055 | 0.397412 | -132.407561 | 0.351809 | 0.355691 | 1.273010 | 53.000000 |
| excitement | eeg_pow | dummy_mean | 0.178237 | 0.220660 | -0.246750 |  |  | 0.791278 | 53.000000 |
| excitement | eeg_pow | random_forest | 0.146684 | 0.185678 | 0.095553 | 0.517600 | 0.475613 | 0.651260 | 53.000000 |
| excitement | eeg_pow | ridge | 0.212874 | 0.480370 | -26.092193 | 0.407499 | 0.466785 | 0.945949 | 53.000000 |
| focus | eeg_pow | dummy_mean | 0.098450 | 0.122481 | -0.326867 |  |  | 0.792281 | 53.000000 |
| focus | eeg_pow | random_forest | 0.092083 | 0.115280 | -0.180149 | 0.269400 | 0.234721 | 0.740944 | 53.000000 |
| focus | eeg_pow | ridge | 0.143681 | 0.400866 | -102.102329 | 0.221845 | 0.258011 | 1.151613 | 53.000000 |
| interest | eeg_pow | dummy_mean | 0.067854 | 0.088629 | -0.077581 |  |  | 0.712187 | 53.000000 |
| interest | eeg_pow | random_forest | 0.064733 | 0.083210 | -0.015521 | 0.429186 | 0.395384 | 0.679771 | 53.000000 |
| interest | eeg_pow | ridge | 0.089441 | 0.207369 | -36.727600 | 0.384006 | 0.423997 | 0.938853 | 53.000000 |
| relaxation | eeg_pow | dummy_mean | 0.134793 | 0.163333 | -0.100575 |  |  | 0.809673 | 53.000000 |
| relaxation | eeg_pow | random_forest | 0.117194 | 0.144597 | 0.119874 | 0.454106 | 0.435621 | 0.703756 | 53.000000 |
| relaxation | eeg_pow | ridge | 0.174113 | 0.417776 | -44.920802 | 0.352814 | 0.425913 | 1.051312 | 53.000000 |
| stress | eeg_pow | dummy_mean | 0.101076 | 0.131638 | -0.141896 |  |  | 0.729719 | 53.000000 |
| stress | eeg_pow | random_forest | 0.092919 | 0.122446 | -0.161377 | 0.419393 | 0.366251 | 0.671066 | 53.000000 |
| stress | eeg_pow | ridge | 0.153254 | 0.334026 | -65.738842 | 0.361086 | 0.390259 | 1.103978 | 53.000000 |

## Paired feature-view differences

Differences are right minus left on identical participant/fold/seed units. Negative MAE and positive correlations indicate improvement.

| model | comparison | mae_difference | r2_difference | pearson_difference | spearman_difference | normalized_mae_difference |
| --- | --- | --- | --- | --- | --- | --- |
| dummy_mean | eeg_pow_minus_eeg | 0.000000 | 0.000000 |  |  | 0.000000 |
| dummy_mean | eeg_pow_minus_pow | 0.000000 | 0.000000 |  |  | 0.000000 |
| dummy_mean | pow_minus_eeg | 0.000000 | 0.000000 |  |  | 0.000000 |
| hist_gradient_boosting | eeg_pow_minus_eeg | -0.002798 | 0.681908 | 0.052885 | 0.052008 | -0.019020 |
| hist_gradient_boosting | eeg_pow_minus_pow | -0.003585 | -6.283521 | 0.038235 | 0.030808 | -0.020639 |
| hist_gradient_boosting | pow_minus_eeg | 0.000787 | 6.965430 | 0.014650 | 0.021200 | 0.001619 |
| random_forest | eeg_pow_minus_eeg | -0.003010 | -1.191242 | 0.060953 | 0.057000 | -0.021312 |
| random_forest | eeg_pow_minus_pow | -0.002873 | -4.339365 | 0.035514 | 0.031541 | -0.015724 |
| random_forest | pow_minus_eeg | -0.000137 | 3.148122 | 0.025438 | 0.025459 | -0.005588 |
| ridge | eeg_pow_minus_eeg | 0.048379 | -57.010213 | 0.001113 | 0.039716 | 0.347611 |
| ridge | eeg_pow_minus_pow | -0.043594 | 150.854247 | 0.077086 | 0.040935 | -0.303216 |
| ridge | pow_minus_eeg | 0.091973 | -207.864460 | -0.075973 | -0.001219 | 0.650827 |

## Paired multi-output minus single-output

These rows use only the identical seven-target complete-case cohort.

| target_name | feature_set | model | mae_multi_minus_single | r2_multi_minus_single | pearson_multi_minus_single | spearman_multi_minus_single | normalized_mae_multi_minus_single |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attention | eeg | random_forest | 0.003113 | -0.057893 | -0.149162 | -0.145433 | 0.024739 |
| attention | eeg | ridge | -0.000000 | 0.000007 | 0.000001 | -0.000001 | -0.000000 |
| attention | eeg_pow | random_forest | 0.002636 | -0.053063 | -0.214762 | -0.208580 | 0.021038 |
| attention | eeg_pow | ridge | 0.000000 | -0.000671 | -0.000000 | -0.000003 | 0.000003 |
| attention | pow | random_forest | 0.002424 | -0.047384 | -0.200985 | -0.197347 | 0.019306 |
| attention | pow | ridge | -0.000000 | 0.000815 | -0.000002 | 0.000005 | -0.000001 |
| engagement | eeg | random_forest | 0.000631 | 0.023114 | -0.069776 | -0.074160 | 0.004482 |
| engagement | eeg | ridge | 0.000000 | 0.000004 | -0.000001 | 0.000002 | 0.000001 |
| engagement | eeg_pow | random_forest | 0.002822 | -0.027594 | -0.078077 | -0.089208 | 0.021496 |
| engagement | eeg_pow | ridge | 0.000001 | -0.005020 | 0.000001 | -0.000003 | 0.000008 |
| engagement | pow | random_forest | 0.001855 | -0.009138 | -0.046872 | -0.058528 | 0.014274 |
| engagement | pow | ridge | -0.000003 | 0.039652 | -0.000002 | 0.000004 | -0.000025 |
| excitement | eeg | random_forest | 0.001026 | 0.004985 | -0.008232 | -0.015826 | 0.004544 |
| excitement | eeg | ridge | 0.000000 | -0.000026 | -0.000001 | 0.000006 | 0.000001 |
| excitement | eeg_pow | random_forest | 0.001007 | 0.003327 | -0.014653 | -0.017112 | 0.004458 |
| excitement | eeg_pow | ridge | 0.000013 | -0.010117 | 0.000010 | 0.000013 | 0.000058 |
| excitement | pow | random_forest | 0.001639 | -0.019986 | -0.010961 | -0.021310 | 0.007258 |
| excitement | pow | ridge | 0.000003 | -0.020929 | 0.000001 | -0.000009 | 0.000015 |
| focus | eeg | random_forest | -0.001646 | 0.048891 | 0.017175 | 0.004867 | -0.013249 |
| focus | eeg | ridge | 0.000001 | -0.000041 | -0.000001 | -0.000003 | 0.000008 |
| focus | eeg_pow | random_forest | -0.000167 | 0.012290 | 0.002316 | -0.000945 | -0.001221 |
| focus | eeg_pow | ridge | -0.000006 | 0.019051 | 0.000007 | -0.000002 | -0.000049 |
| focus | pow | random_forest | 0.000796 | -0.023864 | -0.005371 | -0.008433 | 0.006368 |
| focus | pow | ridge | 0.000005 | -0.012959 | 0.000000 | 0.000002 | 0.000038 |
| interest | eeg | random_forest | -0.001205 | 0.142857 | -0.013185 | -0.019296 | -0.012503 |
| interest | eeg | ridge | 0.000000 | -0.000042 | -0.000005 | -0.000000 | 0.000001 |
| interest | eeg_pow | random_forest | -0.000417 | 0.081657 | -0.034499 | -0.044376 | -0.003988 |
| interest | eeg_pow | ridge | 0.000000 | -0.006235 | -0.000005 | -0.000001 | 0.000000 |
| interest | pow | random_forest | -0.001822 | 0.154183 | -0.019020 | -0.020285 | -0.018709 |
| interest | pow | ridge | -0.000005 | 0.008273 | -0.000003 | -0.000011 | -0.000051 |
| relaxation | eeg | random_forest | 0.001272 | 0.006312 | -0.019592 | -0.018624 | 0.007733 |
| relaxation | eeg | ridge | 0.000000 | 0.000009 | -0.000002 | -0.000005 | 0.000001 |
| relaxation | eeg_pow | random_forest | 0.002225 | -0.015662 | -0.026010 | -0.027473 | 0.013283 |
| relaxation | eeg_pow | ridge | 0.000009 | -0.015028 | -0.000004 | -0.000001 | 0.000058 |
| relaxation | pow | random_forest | 0.002512 | -0.015843 | -0.024102 | -0.025728 | 0.015024 |
| relaxation | pow | ridge | -0.000001 | -0.005814 | -0.000011 | 0.000004 | -0.000008 |
| stress | eeg | random_forest | -0.001508 | 0.006896 | -0.024014 | -0.034192 | -0.010356 |
| stress | eeg | ridge | 0.000001 | -0.000033 | -0.000003 | -0.000004 | 0.000005 |
| stress | eeg_pow | random_forest | 0.000187 | -0.045178 | -0.039796 | -0.054754 | 0.002223 |
| stress | eeg_pow | ridge | -0.000013 | 0.024057 | -0.000010 | -0.000012 | -0.000093 |
| stress | pow | random_forest | -0.002022 | 0.255469 | -0.005433 | -0.020413 | -0.014046 |
| stress | pow | ridge | -0.000012 | 0.031703 | -0.000013 | -0.000001 | -0.000083 |

## Descriptive source slices

| model | source | mae_mean | r2_mean | pearson_mean | spearman_mean |
| --- | --- | --- | --- | --- | --- |
| dummy_mean | Old_EEG | 0.114375 | -5.920220 |  |  |
| dummy_mean | gpn_data | 0.107924 | -6.568943 |  |  |
| dummy_mean | overall | 0.111961 | -5.394359 |  |  |
| hist_gradient_boosting | Old_EEG | 0.101597 | -8.570835 | 0.457678 | 0.426754 |
| hist_gradient_boosting | gpn_data | 0.095607 | -9.406945 | 0.460391 | 0.409690 |
| hist_gradient_boosting | overall | 0.098894 | -7.713410 | 0.468391 | 0.428412 |
| random_forest | Old_EEG | 0.101496 | -9.032357 | 0.433257 | 0.409342 |
| random_forest | gpn_data | 0.095455 | -9.982383 | 0.435376 | 0.387633 |
| random_forest | overall | 0.098791 | -8.151134 | 0.443408 | 0.407267 |
| ridge | Old_EEG | 0.166285 | -72.502337 | 0.343121 | 0.386222 |
| ridge | gpn_data | 0.107565 | -15.356408 | 0.334424 | 0.364850 |
| ridge | overall | 0.154823 | -63.958937 | 0.346058 | 0.381898 |

Source slices are descriptive and are not treated as independent confirmation datasets.

## Participant variability and undefined metrics

Participant metrics use equal subject weights. The standard deviations in the result tables capture participant variability. Undefined R-squared/Pearson/Spearman values remain missing with explicit reasons; they are never replaced by zero.

| model | participants | undefined_r2 | undefined_pearson | undefined_spearman |
| --- | --- | --- | --- | --- |
| dummy_mean | 2244.000000 | 0.000000 | 2244.000000 | 2244.000000 |
| hist_gradient_boosting | 1131.000000 | 0.000000 | 0.000000 | 0.000000 |
| random_forest | 10071.000000 | 0.000000 | 0.000000 | 0.000000 |
| ridge | 3357.000000 | 0.000000 | 0.000000 | 0.000000 |

Detailed participant-, fold-, seed-, source-, comparison-, dummy-improvement- and undefined-metric tables are stored under `benchmark_results/pm_all_targets_feature_baseline_v1/`.

## Limitations and final status

This is a classical engineered-feature baseline, not a raw-EEG, personalization, FOMAML or DANN experiment. Different PM targets have different complete-case cohorts; direct single-versus-multioutput comparisons therefore use only the identical seven-output cohort. Negative R-squared values are retained. No target is declared solved from relative ranking alone.

Participant-level R-squared is unstable for subjects whose within-subject target variance is near zero; it must be interpreted together with MAE, normalized MAE and correlations. Ridge emitted ill-conditioned-matrix warnings and produced poor finite predictions despite train-only scaling, so those negative results are retained rather than hidden or used to retune the preregistered model.
