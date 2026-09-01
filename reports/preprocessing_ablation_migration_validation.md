# Preprocessing ablation migration validation

## Result

All eight seed-42 trials were found through semantic benchmark-config matching.
For each trial, the resolved canonical config hash equals the hash recomputed
from the standard timestamped run's `config.yaml`. The only top-level config
difference is `output_dir`, which is execution placement and is intentionally
excluded from the scientific config hash.

| Trial | Legacy ablation result | Standard benchmark run | Semantic match | Config differences | Reuse status |
|---|---|---|---|---|---|
| A | `full/trial_A/seed_42` | `20260716_131347` | `d440b97fea081930…` | `output_dir` only | reusable |
| B | `full/trial_B/seed_42` | `20260716_132542` | `86ebf3a5759bbf63…` | `output_dir` only | reusable |
| C | `full/trial_C/seed_42` | `20260716_133850` | `d1c06057b545d275…` | `output_dir` only | reusable |
| D | `full/trial_D/seed_42` | `20260716_135109` | `32a59434ea1285fa…` | `output_dir` only | reusable |
| E | `full/trial_E/seed_42` | `20260716_140244` | `2d28a8462e3ff2b5…` | `output_dir` only | reusable |
| F | `full/trial_F/seed_42` | `20260716_141608` | `8e0f89ddbde6da80…` | `output_dir` only | reusable |
| G | `full/trial_G/seed_42` | `20260716_142826` | `2d0ad68ee2265002…` | `output_dir` only | reusable |
| H | `full/trial_H/seed_42` | `20260716_144142` | `ee9dc970eb858b72…` | `output_dir` only | reusable |

Validation of each standard timestamped run required:

- saved config hash equal to the expected resolved hash;
- complete five-fold GroupKFold result;
- every artifact path recorded by every fold to exist;
- unified predictions to exist;
- unified prediction count to equal the sum of fold test counts;
- no duplicate observation IDs;
- benchmark result JSON to exist.

## Legacy versus canonical layout

The timestamped subdirectory inside each old seed-42 trial is a valid standard
`BenchmarkRunner` run containing the canonical config, metrics, unified
predictions and fold artifacts. The old trial root additionally contains
ablation-owned copies such as `benchmark_results.json`, `summary.csv`,
`predictions.parquet`, and `trial_status.json`. Those copies are therefore
classified as legacy output schema and are not used as the completion truth.
They were not deleted because they support the completed scientific report.

New `trial_reference.json` files point directly to the validated timestamped
standard runs. Each reference is accompanied only by `resolved_trial.yaml`.
Seed 42 was not retrained.

## Resume separation

Cache resume continues to validate the preprocessing hash, source identity,
record shard config hashes, shapes, dtype and indexed sample IDs. Benchmark
resume independently validates a standard run using the resolved benchmark
config hash and its standard result artifacts. A reusable cache with no
matching standard run remains `reuse_cache_and_run`, not `skip_completed`.

New runs write `run_manifest.json` in the standard timestamped benchmark run
directory. Legacy seed-42 runs have no manifest, so the read-only fallback
validates their `config.yaml`, result JSON and referenced artifacts directly.
