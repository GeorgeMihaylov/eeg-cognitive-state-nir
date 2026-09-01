# Raw-deduplicated FOMAML protocol for `label_q5`

## Scope and status

- Branch: `integration/benchmark-unification`.
- Base HEAD: `5340dc3` (`feat(meta): add guarded FOMAML diagnostic runner`).
- Protocol ID: `fomaml_label_q5_raw_deduplicated_v2`.
- Readiness decision: `raw_protocol_ready`.
- Execution flag: `false`.
- Training, gradient steps, optimizers, CUDA tensors, inference, policy
  selection, and checkpoints were not created or run.

This is a new protocol, not a repaired version of the blocked task-8X
protocol. The blocked experiment used feature-level episode IDs from a
45,384-window universe, while the production EEGNet diagnostic requires the
30,958-window raw-deduplicated universe. Of its 3,840 support/query references,
901 were absent from the raw cache. The blocked result and its artifacts were
not changed.

## Immutable predecessor

- Blocked status: `blocked_protocol_raw_sample_mismatch`.
- Old protocol hash:
  `a3e6ff5ee2dbfa1638ffee9180ddff582dbab8aa6186e164320dd92f082871e8`.
- Old preregistration SHA-256:
  `54f21e907ff1a414d45c1594e422c4caede0a449ca9acf02374bb50502122754`.
- Existing production-contract and blocked-diagnostic runtime directories were
  hashed before and after materialization and remained byte-identical.

No old sample or episode ID was remapped. Old IDs appear only in the
analytical comparison table.

## Raw-deduplicated universe

The protocol reads metadata from the canonical raw-window manifest and the
existing logical-recording selection map. It does not build a cache and does
not load all EEG into memory.

| property | value |
|---|---:|
| windows | 30,958 |
| subjects | 54 |
| selected source records | 86 |
| logical records | 86 |
| tensor shape | `[14,2560]` |
| sampling rate | 256 Hz |
| window duration | 10 s |
| class counts | 6,539 / 6,285 / 6,143 / 6,034 / 5,957 |
| duplicate sample IDs | 0 |
| invalid or missing labels | 0 |

All 30,958 `cache_file + cache_offset` references resolve inside 86 existing
float32 mmap shards. Every shard has the expected channel/time shape; 172
boundary windows were sampled for finite-value verification. Cache building
was not invoked.

- Cache ID: `raw-2251ca950a467267`.
- Preprocessing hash:
  `2251ca950a467267dcccc1c5b83157f26e02768f46c6073d33f5dc16225bda84`.
- Sample-ID hash:
  `17b2a1b77d4dbb38370a21ad5a63817ce8ca7433358a04fe7fc689e93fcfd6c5`.
- Metadata hash:
  `a19a833beeee2c95a5390a10dc6a5e37954d3b0a6fffc7279017e4608fd3eac4`.
- Raw-universe hash:
  `308fdd96523565417d0fc2f3b3bfdea74639224ce1ee44f5e87af6cc0b7e94cf`.

The logical-record selection remains the existing deterministic ranking by
accepted fraction, available samples, missingness, source priority, and
lexical record ID.

## Outer fold

Fold 1 is reused without regeneration:

- outer train: 43 subjects;
- protected outer test: 11 subjects;
- subject overlap: 0;
- source fold-artifact SHA-256:
  `41ec5a244e11b5dd4ff25faa7361f2bca302dd719612fea8cbc54a55b6ff3341`;
- semantic outer-split hash:
  `b8591f6a0ff5a8abc2f99a9358629117583e1c219662ddc444a99cba473a6041`.

The raw subject/fold mapping exactly matches the existing GroupKFold
artifact and the old protocol's 43/11 subject lists.

## Eligibility, support budget, and class policy

The primary split is record-disjoint personalization. The complete earliest
record is support and all complete later records are query. A record is never
split between partitions; there is no window-level fallback, replacement, or
oversampling. All materialized episodes have a verified strict temporal
boundary from support to query.

The budget was audited on outer-train only:

- fixed support budget: one complete early record;
- query budget: all remaining complete records;
- minimum acceptable support/query windows: 32/64;
- outer-train support windows: 128–622, median 400.5;
- outer-train query windows: 97–427, median 200;
- no fixed window cap, because applying one would split a record.

Candidate class-policy eligibility was:

| policy | outer-train eligible | outer-test eligible |
|---|---:|---:|
| `none` | 20 | 6 |
| `at_least_one_per_class` | 16 | 5 |
| `require_all_classes` | 16 | 5 |

`at_least_one_per_class` was retained from the preregistered candidate set
after checking outer-train feasibility only. It requires each support and
query partition to contain all labels 0–4. No automatic weakening occurred,
and outer-test was not used to choose the policy or budget.

Thirty-three of 54 subjects are explicitly ineligible: 27 have fewer than two
independent records, three queries lack class 4, two supports lack class 4,
and one support has only 28 windows and lacks class 4. The final eligible
counts are 16 outer-train and five outer-test subjects.

## Nested meta split

Five of the 16 eligible outer-train subjects are selected for meta-validation
by exhaustive minimization of the class-proportion deviation, with seed-42
hash ordering as the deterministic tie-break. Outer-test is not an input.

- Meta-train (11): `0012905a`, `0110f12e`, `2182c1cd`, `40009139`,
  `5001d09a`, `50c02189`, `517001af`, `7072a0e0`, `81e150c1`,
  `b0700166`, `f0f2a1e1`.
- Meta-validation (5): `0182e16c`, `30908049`, `71e10186`, `a02151ac`,
  `d0e2d025`.
- Eligible protected outer-test (5): `3110e0c7`, `7150e10a`, `71f0603f`,
  `c112918e`, `d111e017`.
- Meta-split hash:
  `255844005f38e46e561222e5eb9add6c4b6b003934896f3f323c1075dc186cc1`.

## Episodes and leakage audit

The protocol contains 21 new episodes: 11 meta-train, five meta-validation,
and five protected outer-test. New episode IDs include the raw dataset/cache
identity, raw-universe hash, fold, meta split, scope, participant, complete
support/query sample and record IDs, episode specification, and seed.

All checks pass:

- missing raw IDs: 0;
- duplicate episode IDs: 0;
- duplicate sample references across episodes: 0;
- support/query sample overlap: 0;
- support/query logical-record overlap: 0;
- meta-train/meta-validation/outer-test subject overlap: 0;
- episode subject mismatch: 0;
- chronology failures: 0;
- within-record fallbacks: 0.

Hashes:

- Episode specification:
  `ab1caa365614b9c369195b507395e76df8b7f9106bbe99a91379f069e2620b0b`.
- Episode manifest:
  `ee5e41d21e1c9bc72c7e09ae204b4ca35ed9ee09289ade7552dfa122607151ba`.
- Protocol:
  `e73703a443aea3b34f62606efa76bd592ff70099a30cdca80d292f1d76a1fd60`.

## Comparison with the blocked feature-level protocol

The old protocol had 40 materialized episodes. Twenty-three were fully
present in the raw universe, but the new eligibility and complete-record
contract yield 21 episodes. All 21 common participants have new support/query
sets and new episode IDs. No old episode ID was reused, and no remapping was
performed. The old comparison still accounts for exactly 901 missing raw
references.

## Disabled preregistration and future launch contract

The new preregistration was written only after the universe, outer split,
episodes, leakage, determinism, and predecessor-immutability audits passed.

- Production EEGNet architecture signature:
  `248d244e050af2b9e5bd1de0b706665efd3cb13a642044a965fc85602bbe23a7`.
- New preregistration SHA-256:
  `dc998ca72142678394e6f85e10d4b89b1fd0205a6a87be5cae2ba26c37c98692`.
- Seed: 42.
- Future device selection: `auto`.
- Execution enabled: `false`.
- BatchNorm candidates: `frozen_global`, `support_local`, selected only on
  meta-validation.
- Future comparison modes: `zero_shot_supervised`,
  `supervised_full_model`, `fomaml_frozen_global`, and
  `fomaml_support_local`.

The scientific hypothesis is unchanged: FOMAML initialization should improve
personal EEGNet adaptation relative to supervised initialization under
identical support/query data and adaptation budget. Before any future run,
the disabled flag must be explicitly authorized, checkpoints and policy must
be fixed using meta-validation only, and outer-test must remain unopened until
all selection is complete.

## Runtime artifacts

Ignored runtime outputs are under
`benchmark_results/meta_learning_fomaml_label_q5_raw_protocol/` and include
the raw inventory, fold and eligibility audits, policy/budget tables, meta
split, episode specification/index/manifest/balance, leakage audit,
old-versus-new comparison, protocol manifest/hash, disabled preregistration,
readiness decision, errors, and runtime report. They contain no absolute local
paths and are not intended for Git.

## Verification

- New protocol tests: 12 passed, one existing pytest-config warning.
- Related meta/FOMAML regression tests: 63 passed, one existing warning.
- Full `tests` suite: 1,087 passed, 13 warnings.
- Full repository suite: 1,087 passed, 13 warnings.
- A second metadata-only build produced byte-identical protocol and
  preregistration files.
- Python compilation and `git diff --check`: passed.
