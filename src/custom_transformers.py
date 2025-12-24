# src/custom_transformers.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = list(columns)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols = [c for c in self.columns if c in X.columns]
        return X.drop(columns=cols, errors="ignore")

def log_eps(X, eps: float = 1e-7):
    X = np.asarray(X, dtype=np.float64)
    return np.log(X + eps)
