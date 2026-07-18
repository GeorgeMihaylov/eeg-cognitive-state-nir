# Ordinal Transformer multiseed decision

Selected decision: **A**.
Selected head: **corn**.
Primary feature group: eeg_pow; control: eeg_only.

## Evidence for

- coral on eeg_pow: supported primary metrics ordinal_mae, severe_error_rate; confirmed BA/F1 loss=False.
- corn on eeg_pow: supported primary metrics ordinal_mae, severe_error_rate; confirmed BA/F1 loss=False.
- coral on eeg_only: supported primary metrics ordinal_mae, severe_error_rate; confirmed BA/F1 loss=False.

## Evidence against

- ordinal_mae changes sign across seeds (2/3 positive).
- Mean balanced_accuracy change is negative (-0.01114) although it is not Holm-confirmed.
- Mean macro_f1 change is negative (-0.00168) although it is not Holm-confirmed.

## Remaining uncertainty

- Only three initialization seeds were evaluated.
- All inference is internal to the same 53 subjects and one benchmark dataset.

## Next experiment

Confirm the selected pure ordinal head on an external or nested-validation cohort.

Runs not recommended:
- Additional pure ordinal-head seeds before the selected next experiment
- New preprocessing variants as a response to this head comparison
