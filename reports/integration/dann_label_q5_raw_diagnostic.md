# Diagnostic DANN: Old_EEG → gpn_data

- Branch/HEAD at execution: `integration/benchmark-unification` / `6e54ff9`.
- Scientific hypothesis: unlabeled gpn_data domain training improves direct label_q5 transfer over a matched source-only EEGNet.
- Status: **proceed** (`diagnostic`, not a final scientific result).
- Protocol / candidate / executable preregistration: `7f5642109e1ed26dd6de96aa88fe0711bfa08e8f3a58422b17364301d693f7c5` / `a47141952a1a517555a32d3c6b091bf159e2c6f254f1ab63be99f8a78e5a3551` / `f5e7cd962cc361b36e74b073c8532e2af5f4a94a36831f21531d6db36b54a817`.
- Direction: Old_EEG labeled train → gpn_data unseen participants; strict subject-disjoint fold 1, seed 42.
- Partitions: {'source_train': 3753, 'source_validation': 1456, 'target_test': 4973, 'target_train': 18555}.
- Production EEGNet: 8,501 task parameters, latent 1,280; fixed discriminator 172,354; total DANN 180,855.
- Matched budget: 6960 source optimizer updates per mode over 12 epochs; 580 steps/epoch.
- Best source-validation epochs: source-only 11, DANN 2.
- Source-validation macro F1: source-only 0.221576, DANN 0.223941.
- GRL: `2/(1+exp(-10*p))-1`; domain lambda: constant 1.0. Domain accuracy is diagnostic and never selected a checkpoint.
- GRL alpha range: 0.000000 → 0.999909; mean/final domain loss: 0.644240/0.600744.
- Mean source/target/combined domain accuracy: 0.639801/0.612733/0.626376.
- Mean encoder/task-head/domain-head gradient norms: 2.248087/2.046460/0.622407.
- Gradient decomposition mean domain/task encoder ratio: 0.484999 (maximum 0.943072); all finite and state-preserving: True.
- Checkpoints: source-only `5b371b8da06088f0f386aa82c6848cd9d473e7845b2f1655339587acc72e11f3`, DANN `0fa4900e166c2ce2a6f51bbd3c79871409ba47d20df4e649146684d925e1ada2`.
- Target-test unlock hash: `4a0bbf7d000b3a1155162078a53bb3394d3d9f6a0441d1e2d0870b4521be1054`; checkpoints remained immutable: True.

## Target-test participant-level aggregate

| mode | mean macro F1 | mean balanced accuracy | median macro F1 | mean ordinal MAE |
|---|---:|---:|---:|---:|
| source_only_matched | 0.184363 | 0.203257 | 0.180584 | 1.438685 |
| dann | 0.197728 | 0.222335 | 0.202590 | 1.369355 |

## Paired result

DANN − source-only mean participant Δmacro F1 = +0.013364; Δbalanced accuracy = +0.019079; Δordinal MAE = -0.069330.
Macro-F1 wins/losses/ties: 6/2/0 across eight participants.
The participant bootstrap is descriptive only; eight target participants and three source-validation participants provide low statistical power.

## Confusion matrices (rows=true, columns=predicted)

```json
{
  "dann": [
    [
      312,
      164,
      390,
      241,
      149
    ],
    [
      237,
      117,
      381,
      198,
      135
    ],
    [
      213,
      98,
      342,
      185,
      168
    ],
    [
      155,
      76,
      230,
      179,
      228
    ],
    [
      84,
      52,
      151,
      175,
      313
    ]
  ],
  "source_only_matched": [
    [
      231,
      197,
      273,
      343,
      212
    ],
    [
      163,
      162,
      270,
      285,
      188
    ],
    [
      177,
      136,
      210,
      258,
      225
    ],
    [
      90,
      103,
      154,
      206,
      315
    ],
    [
      72,
      53,
      105,
      176,
      369
    ]
  ]
}
```

## Participant metrics

| mode | subject | n | macro F1 | balanced accuracy | ordinal MAE |
|---|---|---:|---:|---:|---:|
| dann | 3110e0c7 | 442 | 0.245587 | 0.255551 | 1.187783 |
| dann | 40f0714a | 155 | 0.154751 | 0.192111 | 1.400000 |
| dann | 7150e10a | 864 | 0.186351 | 0.218799 | 1.303241 |
| dann | 81f1f0fe | 581 | 0.259269 | 0.270701 | 0.965577 |
| dann | a1721173 | 295 | 0.131040 | 0.176342 | 1.705085 |
| dann | c060c06a | 468 | 0.162735 | 0.198681 | 1.521368 |
| dann | c112918e | 843 | 0.223263 | 0.242468 | 1.406880 |
| dann | d111e017 | 1325 | 0.218828 | 0.224030 | 1.464906 |
| source_only_matched | 3110e0c7 | 442 | 0.240693 | 0.248735 | 1.162896 |
| source_only_matched | 40f0714a | 155 | 0.149927 | 0.161242 | 1.509677 |
| source_only_matched | 7150e10a | 864 | 0.177828 | 0.205977 | 1.380787 |
| source_only_matched | 81f1f0fe | 581 | 0.183339 | 0.200854 | 1.058520 |
| source_only_matched | a1721173 | 295 | 0.156489 | 0.163276 | 1.474576 |
| source_only_matched | c060c06a | 468 | 0.147851 | 0.194199 | 1.916667 |
| source_only_matched | c112918e | 843 | 0.197982 | 0.224703 | 1.469751 |
| source_only_matched | d111e017 | 1325 | 0.220798 | 0.227067 | 1.536604 |

## Integrity and interpretation

All subject, sample, and logical-record overlaps are zero. Target-train batches contain no task-label field. Target-test tensors were opened only after both checkpoint hashes, best epochs, schedules, metrics, and the decision rule were fixed. Gradient decomposition was diagnostic and performed no optimizer step; gradient clipping was applied during normal updates.

This is one fold and one seed, so it cannot establish robustness or statistical significance. Reverse direction, additional folds/seeds, other models, target-supervised bounds, and hyperparameter search were not run. Any follow-up requires a separate approved question; no automatic next experiment is selected.
