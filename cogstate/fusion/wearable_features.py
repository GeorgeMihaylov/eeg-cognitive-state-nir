import numpy as np
def extract_hrv_features(rr_intervals_ms):
    rr = np.asarray(rr_intervals_ms, float)
    if len(rr) < 2: raise ValueError("At least two RR intervals are required")
    diff = np.diff(rr)
    return {"mean_rr_ms": float(rr.mean()), "sdnn_ms": float(rr.std(ddof=1)), "rmssd_ms": float(np.sqrt(np.mean(diff ** 2))), "mean_hr_bpm": float(60000 / rr.mean())}
