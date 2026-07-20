# Auxiliary-CORN Transformer technical smoke

> These results use one outer fold and at most three epochs. They are technical pipeline checks, not a lambda selection or a scientific comparison.

## Protocol

- Feature group: `eeg_pow`.
- Input: `[8, 448]`.
- Seed: `42`.
- Outer fold: `1`.
- Auxiliary weights: `[0.25, 0.5, 1.0]`.
- Maximum epochs: `3`.
- Early stopping monitor: `validation_categorical_loss`.
- Lambda selection performed: `false`.

## Technical outcomes

| Lambda | Epochs/best | Validation categorical loss | BA | Macro F1 | Ordinal MAE | Severe error | Aux ordinal MAE | Aux severe error | Head agreement |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 3/3 | 1.215078 | 0.375622 | 0.369047 | 0.962614 | 0.243295 | 0.915909 | 0.215227 | 0.815227 |
| 0.50 | 3/3 | 1.199900 | 0.376653 | 0.366882 | 0.960000 | 0.235909 | 0.914773 | 0.215455 | 0.830227 |
| 1.00 | 3/3 | 1.227571 | 0.371293 | 0.366639 | 0.926364 | 0.210000 | 0.895455 | 0.198295 | 0.826364 |

## Audit conclusion

- Status: `completed`.
- Exact sequence alignment is required across all three weights.
- Inner splits and normalization statistics are required to be identical across weights.
- Primary and auxiliary probabilities, loss decomposition, and strict checkpoint reload are audited per trial.
- Ready for the nested lambda experiment: `True`.

No preferred lambda is selected from this smoke run.
