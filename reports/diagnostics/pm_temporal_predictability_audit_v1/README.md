# Temporal Predictability Audit — history support

This stage audits exact EEG history availability before any predictive model fitting.

## Contract

- Canonical feature cache: 371 float32 EEG features.
- Canonical block: exact 10 s EEG window.
- History uses real record-relative `t_start`.
- No previous-available substitution is allowed.
- History never crosses subject, record or outer-fold identity.
- Horizons: H10, H20, H30, H60 and H120.
- Primary comparison support: `strict_common_support_h120`.
- Secondary sensitivity support: `horizon_specific_support`.
- LOW/HIGH thresholds are not refit in this support audit.
- No models are fitted.

## Global support

| horizon | required_blocks | n_windows | retention_vs_h10 | n_subjects | n_records |
| --- | --- | --- | --- | --- | --- |
| H10 | 1 | 30806 | 1.0000 | 54 | 86 |
| H20 | 2 | 30663 | 0.9954 | 54 | 86 |
| H30 | 3 | 30524 | 0.9908 | 53 | 85 |
| H60 | 6 | 30140 | 0.9784 | 53 | 84 |
| H120 | 12 | 29437 | 0.9556 | 53 | 84 |

## Key result

H120 retains 95.56% of H10 support (29437 windows), with 53 subjects and 84 records.

The minimum H120 retention across target/test-fold combinations is 94.60%.

The retained support is sufficient to use H120 as the matched support for the primary H10–H120 comparison.
