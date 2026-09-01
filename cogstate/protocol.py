"""Immutable contract of the canonical EEG/PM benchmark."""

PM_METRICS = (
    "attention", "engagement", "excitement", "stress",
    "relaxation", "interest", "focus",
)
EEG_CHANNELS = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
)
SAMPLE_RATE = 256
WINDOW_SECONDS = 10
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
N_PM_CLASSES = 3
