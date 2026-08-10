import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
def classification_metrics(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)), "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}
def latency_metrics(latencies_ms):
    x = np.asarray(latencies_ms, float); return {"mean_ms": float(x.mean()), "p95_ms": float(np.percentile(x, 95)), "max_ms": float(x.max())}
