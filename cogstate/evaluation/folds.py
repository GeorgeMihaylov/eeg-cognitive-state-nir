"""Validation guards for fixed participant-disjoint external folds."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class ExternalFold:
    fold_id: int
    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    def __post_init__(self):
        if set(self.train_subjects) & set(self.test_subjects): raise ValueError("External fold leaks participants between train and test")
    def masks(self, subject_ids):
        ids = np.asarray(subject_ids).astype(str)
        return np.isin(ids, self.train_subjects), np.isin(ids, self.test_subjects)


def validate_external_folds(folds: Iterable[ExternalFold], subject_ids):
    known = set(np.asarray(subject_ids).astype(str))
    folds = tuple(folds)
    if len(folds) != 5: raise ValueError("Canonical benchmark requires five fixed external folds")
    for fold in folds:
        if not set(fold.train_subjects) | set(fold.test_subjects) <= known: raise ValueError(f"Fold {fold.fold_id} contains an unknown participant")
    return folds
