"""Baseline: average the three bands over each plant (no spatial information) and classify.

This mirrors the Raman / averaged-spectrum analysis: if the plant-average signal already separates
the light conditions and time points, the bands carry the information on their own.

Two evaluation schemes:
  * leave-one-plant-out (LOPO): the main test.
  * leave-one-time-point-out (LOTO) for light condition: every image of one time point is held out,
    so each test image comes from an imaging session the model never saw. If accuracy collapses here,
    the LOPO score is being driven by session-level differences (brightness, scaling) rather than
    by a light-stress signature that holds across time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.data import LIGHTS, load  # noqa: E402
from stress.evaluate import hours_metrics, leave_one_plant_out, summarise  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def band_features(samples, kind: str) -> np.ndarray:
    rows = []
    for s in samples:
        px = s.image()[s.mask()]
        if kind == "raw":           # absolute band values: mean and spread
            rows.append(np.r_[px.mean(0), px.std(0)])
        elif kind == "chromatic":   # each band as a fraction of the total: removes overall brightness
            chrom = px / np.clip(px.sum(1, keepdims=True), 1, None)
            rows.append(np.r_[chrom.mean(0), chrom.std(0)])
    return np.array(rows)


class PLSDA:
    """PLS-DA: PLS regression onto one-hot class labels, predict the largest output."""

    def __init__(self, n_components=2):
        self.pls = make_pipeline(StandardScaler(), PLSRegression(n_components=n_components))

    def fit(self, X, y):
        self.classes_ = np.array(LIGHTS)
        self.pls.fit(X, (y[:, None] == self.classes_[None]).astype(float))
        return self

    def predict(self, X):
        return self.classes_[self.pls.predict(X).argmax(1)]


MODELS = {
    "PLS-DA": lambda: PLSDA(2),
    "SVM": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
}
REGRESSORS = {
    "PLSR": lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=2)),
    "SVR": lambda: make_pipeline(StandardScaler(), SVR(C=100, epsilon=4)),
}


def run(features: str) -> dict:
    samples = load()
    X = band_features(samples, features)
    light = np.array([s.light for s in samples])
    hours = np.array([s.hours for s in samples])
    plants = np.array([s.plant for s in samples])
    sessions = np.array([s.session for s in samples])
    out = {}

    for name, make in MODELS.items():
        pred = np.empty_like(light)
        for _, tr, te in leave_one_plant_out(plants):
            pred[te] = make().fit(X[tr], light[tr]).predict(X[te])
        out[f"light/{name}/LOPO"] = summarise(light, pred, hours, None)

        pred = np.empty_like(light)
        for h in np.unique(hours):
            tr, te = np.flatnonzero(hours != h), np.flatnonzero(hours == h)
            pred[te] = make().fit(X[tr], light[tr]).predict(X[te])
        out[f"light/{name}/leave-one-time-point-out"] = summarise(light, pred, hours, None)

    # Can the plant average identify the imaging session? (12 sessions, 3 plants each.)
    pred = np.empty_like(sessions)
    for _, tr, te in leave_one_plant_out(plants):
        clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10)).fit(X[tr], sessions[tr])
        pred[te] = clf.predict(X[te])
    out["session_identification_accuracy_LOPO"] = float((pred == sessions).mean())

    for name, make in REGRESSORS.items():
        pred = np.zeros(len(hours), dtype=float)
        for _, tr, te in leave_one_plant_out(plants):
            pred[te] = np.ravel(make().fit(X[tr], hours[tr]).predict(X[te]))
        out[f"hours/{name}/LOPO"] = hours_metrics(hours, pred)
    return out


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    results = {f: run(f) for f in ["raw", "chromatic"]}
    (RESULTS / "01_baseline_bands.json").write_text(json.dumps(results, indent=2))
    for feat, res in results.items():
        print(f"\n== features: {feat}")
        for k, v in res.items():
            if isinstance(v, dict):
                main = {m: (round(x, 3) if isinstance(x, float) else x) for m, x in v.items() if m != "light_confusion"}
                print(f"{k}: {main}")
            else:
                print(f"{k}: {v:.3f}")
