import numpy as np
def late_fuse(probabilities, weights=None):
    probs = np.asarray(probabilities, float)
    weights = np.ones(len(probs)) if weights is None else np.asarray(weights, float)
    if probs.ndim != 2 or len(weights) != len(probs): raise ValueError("probabilities must be [models, classes] with matching weights")
    result = np.average(probs, axis=0, weights=weights)
    return result / result.sum()
