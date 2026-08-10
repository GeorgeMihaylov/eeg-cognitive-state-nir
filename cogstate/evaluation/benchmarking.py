import time
import numpy as np
def benchmark_predict(model, X, repeats=3):
    timings = []
    for _ in range(repeats):
        start = time.perf_counter(); model.predict(X); timings.append((time.perf_counter()-start)*1000)
    return {"mean_ms": float(np.mean(timings)), "per_sample_ms": float(np.mean(timings) / len(X))}
