"""Classifiers shared by the experiments."""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class PLSDA:
    """PLS-DA: PLS regression onto one-hot class labels, predict the largest output."""

    def __init__(self, classes, n_components=2):
        self.classes_ = np.asarray(classes)
        self.pls = make_pipeline(StandardScaler(), PLSRegression(n_components=n_components))

    def fit(self, X, y):
        self.pls.fit(X, (np.asarray(y)[:, None] == self.classes_[None]).astype(float))
        return self

    def scores(self, X):
        """One score per class; higher = more like that class."""
        return self.pls.predict(X)

    def predict(self, X):
        return self.classes_[self.scores(X).argmax(1)]


def band_features(pixels: np.ndarray) -> np.ndarray:
    """Whole-plant summary of N x 3 pixels: mean and spread of each band."""
    return np.r_[pixels.mean(0), pixels.std(0)]
