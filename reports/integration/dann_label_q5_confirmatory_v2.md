# Confirmatory multi-block DANN experiment

- Branch/HEAD: `integration/benchmark-unification` / `402499b`.
- Hypothesis: DANN improves Old_EEG-to-gpn_data label_q5 transfer over source-update-matched EEGNet.
- Protocol: `1ce582a3d73a7ae4393e77cc2f3b2cb7749ddbb30c1cb8fcad0056c6d326c368`; execution preregistration: `033ecd4da3d6b68966489dae3db9a90be1b2717653c8e654e9a235dae4c3df11`.
- Primary confirmation uses seeds 123/2026 across five folds. Seed 42 is sensitivity-only; fold-1/seed-42 is referenced diagnostic evidence and was not retrained.
- New runs: 28/28 complete in 14 matched pairs; pair attempts 15; pair-level technical restarts 1.
- Fixed training: AdamW, lr 0.001, weight decay 0.0001, batches 32/32, maximum 12 epochs, patience 3, clipping 5.0, logistic GRL, lambda 1.0.
- Matched steps by fold: 580, 602, 569, 606, 586. Source batch hashes and optimizer-update counts match within every pair.
- Fixed data: 30,958 raw-deduplicated windows, 54 participants, 86 logical records, `[1, 14, 2560]`, 256 Hz, 10 s, five-class `label_q5`.
- Direction: Old_EEG source to gpn_data target. The five byte-locked subject-disjoint outer folds and source-validation partitions were reused without rebuilding the raw cache.
- Total paired training time recorded by the 14 pair summaries: 4960.8 s.

## Primary result

- Decision: **partially_confirmed**.
- Mean/median participant macro-F1 delta: 0.008048 / 0.001954.
- Mean balanced-accuracy delta: 0.008332; win fraction: 0.548.
- Participant bootstrap 95% interval: [-0.001672, 0.017882]. No standalone significance claim is made.
- Primary result lock: `f0955136a305048e4364dfafb77c9ef928826b58c5b2a29c87966ab791b3b8dd`.

### Absolute participant-level primary metrics

| mode | accuracy | balanced accuracy | macro F1 | weighted F1 | kappa | ordinal MAE | quadratic kappa |
|---|---:|---:|---:|---:|---:|---:|---:|
| source_only_matched | 0.241361 | 0.223680 | 0.193767 | 0.233038 | 0.032227 | 1.446077 | 0.100019 |
| dann | 0.253282 | 0.232012 | 0.201815 | 0.247595 | 0.035711 | 1.412069 | 0.113800 |

### Fold-level primary deltas

| fold | participants | mean macro F1 delta | median | balanced accuracy delta | ordinal MAE delta | W/L/T |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | -0.007736 | -0.012311 | 0.001145 | -0.046373 | 3/5/0 |
| 2 | 9 | 0.006867 | -0.006078 | 0.018025 | -0.004842 | 4/5/0 |
| 3 | 10 | 0.017168 | 0.015389 | 0.011142 | -0.141634 | 7/3/0 |
| 4 | 7 | 0.009027 | 0.020736 | -0.000874 | -0.039677 | 4/3/0 |
| 5 | 8 | 0.012906 | 0.005760 | 0.009156 | 0.085036 | 5/3/0 |

### Seed-level primary deltas

| seed | participants | mean macro F1 delta | balanced accuracy delta | W/L/T |
|---:|---:|---:|---:|---:|
| 123 | 42 | 0.009895 | 0.009767 | 25/16/1 |
| 2026 | 42 | 0.006202 | 0.006897 | 22/20/0 |

### Checkpoint selection and training

| phase | fold | seed | source best epoch | DANN best epoch | source val macro F1 | DANN val macro F1 | pair seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| primary | 1 | 123 | 4 | 2 | 0.240741 | 0.218588 | 284.4 |
| primary | 1 | 2026 | 3 | 2 | 0.224020 | 0.224739 | 245.8 |
| primary | 2 | 123 | 5 | 3 | 0.229558 | 0.233778 | 337.5 |
| primary | 2 | 2026 | 9 | 10 | 0.235224 | 0.239002 | 503.7 |
| primary | 3 | 123 | 4 | 7 | 0.185201 | 0.218868 | 401.4 |
| primary | 3 | 2026 | 7 | 7 | 0.171295 | 0.221412 | 400.9 |
| primary | 4 | 123 | 1 | 1 | 0.206787 | 0.203318 | 170.9 |
| primary | 4 | 2026 | 4 | 1 | 0.217251 | 0.200862 | 299.2 |
| primary | 5 | 123 | 9 | 12 | 0.234723 | 0.208654 | 494.1 |
| primary | 5 | 2026 | 12 | 10 | 0.245956 | 0.207790 | 499.5 |
| secondary | 2 | 42 | 4 | 3 | 0.233620 | 0.225666 | 301.8 |
| secondary | 3 | 42 | 5 | 3 | 0.171416 | 0.207211 | 329.9 |
| secondary | 4 | 42 | 2 | 6 | 0.214761 | 0.236572 | 393.3 |
| secondary | 5 | 42 | 4 | 4 | 0.216861 | 0.216799 | 298.2 |

## Sensitivity and audits

- Secondary status: **robust_positive_sensitivity**; three-seed mean macro-F1 delta 0.010138.
- Participants with seed-dependent effect sign: 24.
- The primary decision was loaded from the immutable primary lock and was not changed by sensitivity results.
- Diagnostic protocol, preregistration, checkpoints, predictions, participant metrics, and summary hashes were verified without modifying their runtime.
- Global leakage audit passed: True; checkpoints immutable after target test: True.
- Gradient decomposition: all finite `True` and state immutable `True` across all 14 new pairs.

### Three-seed sensitivity

| seed | participants | mean macro F1 delta | balanced accuracy delta | ordinal MAE delta | W/L/T |
|---:|---:|---:|---:|---:|---:|
| 42 | 42 | 0.014316 | 0.021081 | -0.052035 | 27/15/0 |
| 123 | 42 | 0.009895 | 0.009767 | -0.032355 | 25/16/1 |
| 2026 | 42 | 0.006202 | 0.006897 | -0.035662 | 22/20/0 |

## Execution notes and artifacts

- One pair-level technical restart occurred before any gradient step because the execution wrapper initially addressed the wrong manifest field. Resume reused the same scientific run specification; all other pairs completed on their first attempt.
- Two post-training interruptions were limited to initially missing aggregation and report directories. Both were resumed without model retraining.
- Runtime root: `benchmark_results/domain_adaptation_dann_confirmatory_v2/`; it contains the immutable execution preregistration, deterministic registry, pair artifacts, aggregations, audits, and final report and is not tracked by Git.
- Result status: `final` for this preregistered confirmation. The hypothesis is only partially confirmed because the mean effect is below +0.01, participant wins are below 60%, and the participant bootstrap interval includes zero.

### Checkpoint and target-test unlock hashes

```text
primary fold=1 seed=123 source=df5f260084a783803ba54b07556ee48bea25cdc18e543809d5c8752c225b6c0e dann=a66c51c58792667de7cf7f77be80caf49ea43e23a33961072b39a2bd8accaad0 unlock=7a026508916b385d43e336ff6de7b4cc0d5a0097b979bf2c690ab78f93b76885
primary fold=1 seed=2026 source=d2df90a15f8ac338d650f7e43995aaf4839d9971d7a8bbb660b4744ebf84b4f3 dann=377ba69312ccab7f4037e8183d74321f7bcd8a33ede28faf243c57d4c0b1d592 unlock=035605b30c7af5233e87c3179d5a0dea06cb9e6999f80a903e91791e2a982d76
primary fold=2 seed=123 source=7c0f052cbdc6bf93b3e092ba6fbd05439da377b26a8d3c44d9a0eac53753f8d9 dann=86134231f9393a49dbf0f695bcdff3677be9c63edd8ff897b6a78e37956d0cdd unlock=dd64452d72811f655ea12a456b7784248711555ba70da8ee798b025db66fcb4a
primary fold=2 seed=2026 source=cafe3e47715d456c0799d2cf1a5396008899d2b4502af3e04a8753d141428f69 dann=195d335b128566cef202d8a6c32fe35c73612e3131e655cf3800f0085756a990 unlock=b4b91668f89eaa7aa96485ea1183a6c58fc1e8c2c4ca6168a82d81bb5b7f4e37
primary fold=3 seed=123 source=686e0960c9b0a16653336cb9631f548a1ecb35c8bc1731d060187629940f1b7c dann=b25eaa300acc141fac512c89b75d8896059b9e279527ad188c4972c79ad57832 unlock=90a1af4a195a9dcb967f5318817bbe23422776bd027cd71267f033f7b4752958
primary fold=3 seed=2026 source=d81f55372547a876bb60298d633c95ec02cd874dfa2657081a679a31a9ce106f dann=70d309c51660ff3addc17d2d87548c3147a93383b6209f1572ef2de264cfbed0 unlock=185f68490f1f152c2631f2cf766480e73eae67f529a04f80729ae68a12c26716
primary fold=4 seed=123 source=b66f8c610dcd66e922db9be3b756124d6e15626b4e796b1ee5f97789f1f353d6 dann=f42ab95f33e1733d5ed6d6474c70ac85deda5544ef77ce34a1490e510f769c56 unlock=00205b2df28fc9038901eddaad826a9996b0534ccda9a584a96daa4363b389d9
primary fold=4 seed=2026 source=26ff291a56db43222dea433dabd4e6d1bfb4de26ab1f14ca0d2f0edace8f4142 dann=41aae250f3bd6c29a4631825f5d3fe505a5cca1ebeee6dbdf3245462a69e94ce unlock=dda7f54d8a59c89c8248aa85257ccb3713829a9cc8a4fdd47488815c0f765992
primary fold=5 seed=123 source=9a27991008e3f71cf20cf87925ed8a7400f96be54f81a7fabf14a6547a0fa997 dann=b7b1518c7a239fce542f45ce09212058137bfb3b368ef9dc1c53a748022f617b unlock=b43665bc6084c50bd78274c0ed6d540144677add6a4832c21b7029ba708fd5d6
primary fold=5 seed=2026 source=6f4ce93a461832392a1c2802af828521b1e74476ac2db02d0ae0c4bed86da7f4 dann=3dc212d9597cff8ca75e52fdd1741a20ca17d96c4192fdc5ada6440466de7534 unlock=24ea75baf498766019235e6eaec09b21dddb44d46f714b382edb4c7913905b90
secondary fold=2 seed=42 source=3bd1311cf944b152b3b02b0d5aca686012e3d0267f1a1a7c255a1970cacd45cc dann=88b9075b1086eef7b56e908c1bf3984c0b1e56fd7b6992ac1392fffa70ec9627 unlock=e2beb734fca4f953115d711bf96b7f3d4dc0481c0e45d5b19266cf750468577d
secondary fold=3 seed=42 source=4f5d108ecc4ad024ea32e643f2ffd37a84d26c35a0e7de4e4d7f771615bc288d dann=b68ccd2ed6e8940a7ace103330e20d5411425eb733425c390c3ab0787325043e unlock=8b419930e665b2ab821eeafef78bbd5f19c4a7646a90cb28adb86cd15e4a6e3a
secondary fold=4 seed=42 source=b5bb4f20f0c16357e12c6416d582e8e1bd9b141dc84c14b0fb98c64f548a267f dann=075ae0fa9ddb45e590e56e907a6c8906903a022f3a68f026a3157e8e1f323657 unlock=9462708e64aa69e440ef4ab27fd2db2c59a9d368bfee5bb0d352f7fef31ba169
secondary fold=5 seed=42 source=4334751c3d11111ade3fdd03353b8e285a0517e0efee9a023d6ad2813897f9a6 dann=ae8fc4b423a97dc8a7fae8ffe3aebba73cb1b7159c15a30e84dec4b6f27fe49c unlock=635f201de2df3b98bc927d249b8d1660792563329cec030963d13fef39b517d1
```

Limitations: gpn_data and Old_EEG are source/organization domains from the same general Emotiv-class acquisition family, not automatically different devices. The bootstrap uses unique participants after primary-seed averaging; windows are never treated as independent observations.
