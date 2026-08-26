import numpy as np

class CORALAdapter:
    """Covariance alignment between source and target feature domains."""
    def fit(self, source, target):
        source, target = np.asarray(source, float), np.asarray(target, float)
        self.source_mean_, self.target_mean_ = source.mean(0), target.mean(0)
        cs, ct = np.cov(source, rowvar=False) + np.eye(source.shape[1]) * 1e-6, np.cov(target, rowvar=False) + np.eye(target.shape[1]) * 1e-6
        es, vs = np.linalg.eigh(cs); et, vt = np.linalg.eigh(ct)
        self.transform_ = vs @ np.diag(1 / np.sqrt(np.maximum(es, 1e-12))) @ vs.T @ vt @ np.diag(np.sqrt(np.maximum(et, 1e-12))) @ vt.T
        return self
    def transform(self, X): return (np.asarray(X) - self.source_mean_) @ self.transform_ + self.target_mean_
