# Canonical configuration validation patch

## Changes

- Replaced the incomplete root `configs.yaml` with a runnable five-class
  diagnostic configuration.
- Updated the explicit `--test` configuration from the obsolete three-class
  task to `cognitive_load_5class`.
- Added validation of registered tasks, model types, dataset mappings and data
  paths.
- Added an explicit task/model type compatibility check.
- Added an explicit error for Torch regression until regression adapters are
  implemented.
- Removed the implicit `cognitive_load_3class` fallback from
  `BenchmarkRunner.run_for_dataset`.
- Added focused tests for the canonical validation contract.

## Scope boundary

This patch does not implement:

- direct PM regression;
- latent proxy targets;
- Torch regression;
- multi-output regression;
- experimental domain-adaptation mixins.

Those changes belong to the next dedicated implementation stage.
