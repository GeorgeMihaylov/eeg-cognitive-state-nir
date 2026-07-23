# Review of WESAD and metrics integration

## Scope

Compared:

- `origin/feature/auxiliary-corn-transformer`
- `origin/feature/benchmarking`

Files:

- `bench/datasets/wesad_loader.py`
- `bench/tasks/wesad_task.py`
- `bench/validation/metrics.py`

## WESAD

| File | Comparison | Decision |
|---|---|---|
| `bench/datasets/wesad_loader.py` | different | manual review required |
| `bench/tasks/wesad_task.py` | different | manual review required |

Conclusion: no WESAD code needs to be copied from the colleague branch in this block. The files must remain unchanged in the integration branch.

## Metrics

The metrics module in `feature/auxiliary-corn-transformer` is retained.

It already includes functionality absent from the colleague version:

- balanced accuracy;
- per-class precision, recall and F1;
- regression metrics: MAE, RMSE, R2, Pearson and Spearman correlations;
- quadratic weighted kappa;
- ordinal MAE;
- adjacent accuracy;
- severe error rate;
- expected-rank metrics;
- backward-compatible metric keys.

Replacing it with the colleague version would remove functionality required by completed Focus, CORAL, CORN and regression experiments.

## Integration decision

- WESAD loader: keep the current implementation.
- WESAD task: keep the current implementation.
- Metrics: keep the current implementation.
- No production code changes are required for this integration block.
- Proceed to review `configs.yaml`, task registries, model factory, runner and CLI.

## Verification

Branch: `integration/benchmark-unification`

WESAD files compared by Git tree diff.

Metrics differ between branches: `True`.
