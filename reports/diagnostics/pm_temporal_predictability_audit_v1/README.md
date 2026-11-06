# Temporal Predictability Audit

Diagnostic/exploratory analysis of canonical engineered EEG trajectories. This report combines the completed support/ACF/MI audits with reviewed PE and new SSA/FNN diagnostics. No classifier is fitted by these diagnostic scripts; the separate predictive experiments are now complete: matched 175/175 and conditional H120 order control 25/25.

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

## History support

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

The retained support is sufficient to use H120 as the matched support for the primary H10-H120 comparison.

## ACF

Exact record-relative endpoint pairs at 10/20/30/60/120 seconds, computed within recordings and aggregated as feature medians within records and then equal-record medians. See `acf_summary.csv` and `scripts/analysis/audit_pm_temporal_acf.py`.

| Delay, s | 10 | 20 | 30 | 60 | 120 |
|---|---:|---:|---:|---:|---:|
| All-feature median record correlation | 0.2655 | 0.1597 | 0.1313 | 0.0991 | 0.0402 |

Connectivity: 0.3436 at 10 s and 0.1019 at 120 s; entropy: 0.3755 and 0.0987. Dependence decays quickly, with a weak remaining association at 120 s. These are endpoint-pair statistics, not an assertion that every intervening window exists. Available record counts vary by delay (84 to 82); records within a participant are dependent. No participant-level significance claim is made.

## Mutual Information

Within-record quantile discretization (five bins), ten temporal permutation surrogates, seed 42; each feature's valid-pair mask is preserved. NMI is normalized by the geometric mean of marginal entropies. Equal-record medians from `mi_surrogate_summary.csv`:

| Delay, s | 10 | 20 | 30 | 60 | 120 |
|---|---:|---:|---:|---:|---:|
| Excess NMI over surrogate | 0.0493 | 0.0255 | 0.0204 | 0.0132 | 0.0075 |

Surrogate NMI is approximately 0.014; connectivity excess NMI at 120 s is 0.0115. MI can capture nonlinear dependence, but these results do not prove nonlinearity. Quantization and finite-sample bias constrain interpretation. No PM-based delay optimization was performed; new FNN uses prespecified tau=1.

## Permutation Entropy

Final results and protocol: [pe_final_v1](pe_final_v1/README.md). Original parent-directory PE CSVs remain preliminary historical outputs.

Longest exact-contiguous segment per record, >=100 windows; m=3/4/5, tau=1, normalized H/log(m!), ten surrogates, seed 42. Review confirmed injective base-m encoding of stable argsort permutations. Ties use temporal-index order and their fractions are reported. The valid-pattern gate is now max(50,5*m!) = 50/120/600, rather than 50 for every order.

| m | Eligible records | Observed PE | Surrogate PE | Median paired surrogate-minus-observed |
|---|---:|---:|---:|---:|
| 3 | 78 | 0.9953 | 0.9975 | 0.0015 |
| 4 | 77 | 0.9859 | 0.9898 | 0.0032 |
| 5 | 8 | 0.9760 | 0.9816 | 0.0054 |

All-feature record medians have positive deltas in all eligible records. The ordinal-order effect is small; PE is near maximal. The eight-record m5 subset is not representative of the full cohort, and order comparisons change both finite-sample bias and support. The length gate is a conservative heuristic, not a guarantee of unbiased entropy estimation.

## SSA / Hankel

Exploratory feature-trajectory analysis, not raw EEG or forecasting. Longest exact-contiguous segment per record, N>=100, median/IQR scaling, fixed L=min(60,floor(N/2)). This caps the trajectory width at 600 s and half the segment length without selecting L from results. Missing trajectories and zero IQR are skipped without interpolation. Energy is squared singular values normalized to sum one, computed through equivalent symmetric Hankel Gram eigenvalues (tested against direct SVD).

Ten within-segment temporal permutations, seed 42. Feature medians are taken within each record/family, then equal-record medians globally. The `all` summary weights usable features equally within a record, not the four families equally. See `ssa_protocol.json`, `ssa_summary.csv`, `ssa_by_record_family.csv`, and `state_space_execution.json`.

| All features, 78 records / 52 participants | Observed | Surrogate |
|---|---:|---:|
| Effective rank exp(H(energy)) | 41.9408 | 53.3163 |
| Participation ratio | 27.3453 | 47.8938 |
| Rank for 90% energy | 43.5 | 47.0 |
| Leading singular-value energy | 0.1172 | 0.0462 |

Median paired surrogate-minus-observed effective rank: 9.8805. Median differences need not equal differences of medians. Of record-feature trajectories, 27,844 were valid, 1,094 had zero IQR, and 2,968 were too short. The observed ordering concentrates energy more than permutation, but the rank remains substantial: this is not evidence for a small physical attractor. Trends/nonstationarity may also lower effective rank; permutation does not isolate a physiological mechanism. Records can share participants, and differing N/L affect spectra. Detailed spectra and cumulative energy are runtime-only in `benchmark_results/pm_temporal_state_space_v1/ssa_spectra.json`; no trajectory matrices are stored.

## FNN / Takens

Restricted exploratory diagnostic: first five canonical feature IDs per family (20 total), fixed before outcomes; tau=1, dimensions 1..8. Kennel-style false-neighbor criteria are distance expansion >10 OR expanded distance / trajectory standard deviation >2. Temporal neighbours within max(12,m*tau) windows are excluded. Require max(500,50*(m+1)) embedding vectors and >=500 usable neighbours; zero-distance cases are reported and >10% causes a skip. Ten within-segment permutations, seed 42. Exact features and rules are in `fnn_protocol.json`.

| Dimension | Observed FNN | Surrogate FNN | Records / participants |
|---|---:|---:|---:|
| 1 | 0.9568 | 0.9515 | 16 / 16 |
| 2 | 0.5276 | 0.5624 | 16 / 16 |
| 3 | 0.2170 | 0.2376 | 16 / 16 |
| 4 | 0.1447 | 0.1724 | 15 / 15 |
| 5 | 0.1432 | 0.1679 | 15 / 15 |
| 6 | 0.1499 | 0.1866 | 15 / 15 |
| 7 | 0.1681 | 0.2131 | 15 / 15 |
| 8 | 0.2025 | 0.2615 | 14 / 14 |

There were 2,438 valid record-feature-dimension comparisons, 11,320 `insufficient_sequence_length`, one `unreliable_neighbors` and one `unreliable_surrogates`. FNN does not approach zero or establish a robust saturation dimension. High-dimensional sparsity, measurement noise, changing support and the deliberately limited representative subset prevent a reliable Takens-dimension claim. No embedding dimension or neural architecture is selected from these curves.

## Reproduction and interpretation

```powershell
python scripts/analysis/audit_pm_temporal_permutation_entropy.py --feature-cache-dir <canonical-cache>
python scripts/analysis/audit_pm_temporal_ssa.py --feature-cache-dir <canonical-cache>
python scripts/analysis/summarize_pm_temporal_diagnostics.py
```

`temporal_diagnostics_summary.json` assembles the compact tables, protocols, input/source hashes and limitations. The cache location is supplied by CLI; reusable code does not hard-code the workstation path.

Support and observed-versus-surrogate structure motivated the now-completed [matched H10–H120 experiment](../pm_temporal_predictability_v1/README.md) (175/175) and [H120 order control](../pm_temporal_predictability_v1/order_control_h120/README.md) (25/25). Diagnostics alone do not establish PM predictability. The separate predictive evidence shows H120 gains for 5/7 PM after BH across 28 contrasts, and order effects for 4/5 gated PM after BH across five contrasts; stress order evidence is inconclusive. This supports one bounded H120 sequence comparison, planned and not yet executed. No reliable embedding dimension or physical attractor was established.

Methodological references: [Bandt and Pompe, PE](https://doi.org/10.1103/PhysRevLett.88.174102); [Kennel et al., FNN](https://doi.org/10.1103/PhysRevA.45.3403). Neither diagnostic establishes physiological causation for proprietary PM proxies.
