# Reviewed permutation entropy

Diagnostic on longest exact-contiguous 10-second segment per record (>=100 windows), 371 canonical features. Orders 3/4/5, tau=1, H/log(m!), 10 within-segment surrogates, seed 42. Stable argsort breaks exact ties by time index; base-m permutation codes are injective. Missing positions stay fixed in surrogates. Feature medians are calculated within record/family and then medians across equally weighted records (not independent participants).

Review changed the minimum valid pattern count to max(50, 5*m!): 50/120/600. This conservative heuristic reduces finite-sample bias but does not eliminate it. Earlier CSVs in the parent directory are preserved as preliminary results, not overwritten.

| Order | Records | Observed PE | Surrogate PE | Paired structure delta |
|---|---:|---:|---:|---:|
| 3 | 78 | 0.9953 | 0.9975 | 0.0015 |
| 4 | 77 | 0.9859 | 0.9898 | 0.0032 |
| 5 | 8 | 0.9760 | 0.9816 | 0.0054 |

Delta is the median paired surrogate-minus-observed difference, not the difference of displayed medians. Positive all-feature record-median deltas occur in all eligible records. This is a small ordinal-order effect, not evidence of a deterministic low-dimensional system or predictive benefit. Order 5 covers only eight long records; differences between orders are confounded by support and finite-sample bias. Ties and valid counts are reported in the detailed artifacts. No significance test is claimed.

Definition: [Bandt and Pompe (2002)](https://doi.org/10.1103/PhysRevLett.88.174102).

Reproduce: `python scripts/analysis/audit_pm_temporal_permutation_entropy.py --feature-cache-dir <canonical-cache>`.
Validation: five targeted synthetic tests passed (entropy limits, all m! codes, gaps/boundaries, length and deterministic missing-mask-preserving surrogates).
