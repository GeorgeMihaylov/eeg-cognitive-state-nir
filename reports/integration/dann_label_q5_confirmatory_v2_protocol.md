# DANN confirmatory-v2 analysis protocol

- Branch/HEAD: `integration/benchmark-unification` / `e0dc3ff`.
- Diagnostic discovery: Old_EEG to gpn_data, fold 1, seed 42, status `proceed`; mean participant macro-F1 delta +0.013364, but its bootstrap interval crosses zero.
- Fold 1 / seed 42 was already inspected and is retained as discovery evidence, not independent confirmation.
- Primary confirmation uses only new seeds 123 and 2026 across all five folds: 20 planned model runs.
- Secondary sensitivity uses seed 42 on folds 2-5: 8 new runs. The two fold-1/seed-42 mode results are referenced, not retrained.
- Total result cells: 30; new training runs: 28. Training has not started.

## Run groups

| analysis group | folds | seeds | modes | results | status |
|---|---|---|---|---:|---|
| primary_confirmatory | 1-5 | 123, 2026 | source_only_matched, dann | 20 | planned_disabled |
| secondary_sensitivity | 2-5 | 42 | source_only_matched, dann | 8 | planned_disabled |
| previously_observed_diagnostic | 1 | 42 | source_only_matched, dann | 2 | already_completed |

All five task-8E outer-fold hashes, fold partitions, strict shared-participant policy, source-validation splits, target-label firewall, architecture, hyperparameters, and matched budgets are unchanged.

## Matched update budgets

| fold | source natural | target natural | matched per epoch | max epochs |
|---:|---:|---:|---:|---:|
| 1 | 139 | 580 | 580 | 12 |
| 2 | 149 | 602 | 602 | 12 |
| 3 | 157 | 569 | 569 | 12 |
| 4 | 143 | 606 | 606 | 12 |
| 5 | 150 | 586 | 586 | 12 |

## Analysis and target-test locks

Primary mode differences are paired within fold/seed/participant, averaged within each participant across seeds 123 and 2026, then aggregated with equal participant weight. Fold and seed summaries are reported separately; no best seed is selected.

The primary decision uses only `primary_confirmatory`. The combined three-seed sensitivity analysis is reported separately and cannot change the primary status.

Each of 14 new fold/seed pairs has its own locked unlock contract. Target test remains unavailable until both checkpoint hashes, best epochs, source-validation metrics, batch-sequence hashes, and v2 protocol/preregistration hashes are fixed. The diagnostic unlock is not reusable.

The primary `confirmed` rule additionally requires both primary seeds and at least four folds to have nonnegative mean participant macro-F1 deltas, alongside the preregistered effect, median, balanced-accuracy, and win-fraction thresholds.

Task-8E protocol/preregistration: `a261d6081b4924af82752021fa24bbd50a75ed83ac3672db1e691709ad2cad71` / `f4862dbf09d6eccd04438eebd5bbd99899dc4f1530ba3554c87a45a13dba59c4`.
V2 protocol/preregistration: `1ce582a3d73a7ae4393e77cc2f3b2cb7749ddbb30c1cb8fcad0056c6d326c368` / `6fba1eb76133884f0d5984ec1ceedc49234f252846040122310ac45a99ad3d7e`.
Readiness: **confirmatory_v2_protocol_ready**; execution enabled: `false`.

Limitations: this is a disabled protocol, not a scientific result. No new DANN/source-only model, inference, target-test access, or statistical analysis was run.
