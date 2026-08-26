import numpy as np
def early_fuse(*feature_blocks):
    arrays = [np.asarray(x) for x in feature_blocks]
    if not arrays or len({a.shape[0] for a in arrays}) != 1: raise ValueError("All feature blocks must share sample count")
    return np.concatenate(arrays, axis=1)
