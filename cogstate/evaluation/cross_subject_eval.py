import numpy as np
from .metrics import classification_metrics
def leave_one_subject_out(model_factory, X, y, subject_ids):
    X, y, ids = np.asarray(X), np.asarray(y), np.asarray(subject_ids)
    results = {}
    for subject in np.unique(ids):
        test = ids == subject; model = model_factory(); model.fit(X[~test], y[~test])
        results[str(subject)] = classification_metrics(y[test], model.predict(X[test]))
    return results
