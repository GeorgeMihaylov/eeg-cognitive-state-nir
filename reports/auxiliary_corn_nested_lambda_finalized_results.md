# Finalized auxiliary-CORN selection policy

> Protocol amendment: the original aborted units are preserved in the source report. This finalization adds the already trained paired categorical baseline as a safe fallback. No outer-test result was used to make a selection decision.

- Selection units completed: 30/30.
- Joint auxiliary-CORN selections: 25.
- Categorical fallbacks: 5.
- Candidate fold fits audited: 90/90.
- Corrected source candidate counter by: 0.
- Selected joint lambda counts: {'0.25': 16, '0.5': 5, '1.0': 4}.
- Model training performed during finalization: false.
- Ready for subject-level analysis: true.

## Fallback units

| Selection unit | Original guard outcome |
| --- | --- |
| `eeg_only_seed123_fold01` | No auxiliary-CORN lambda satisfies the inner-validation BA guard: baseline=0.451816, tolerance=0.010000, minimum=0.441816 |
| `eeg_only_seed42_fold02` | No auxiliary-CORN lambda satisfies the inner-validation BA guard: baseline=0.532180, tolerance=0.010000, minimum=0.522180 |
| `eeg_only_seed7_fold01` | No auxiliary-CORN lambda satisfies the inner-validation BA guard: baseline=0.496687, tolerance=0.010000, minimum=0.486687 |
| `eeg_pow_seed123_fold05` | No auxiliary-CORN lambda satisfies the inner-validation BA guard: baseline=0.554107, tolerance=0.010000, minimum=0.544107 |
| `eeg_pow_seed42_fold01` | No auxiliary-CORN lambda satisfies the inner-validation BA guard: baseline=0.570077, tolerance=0.010000, minimum=0.560077 |
