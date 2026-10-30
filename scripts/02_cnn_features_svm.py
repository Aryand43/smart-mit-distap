"""Pretrained CNN features + SVM.

Tile features from a frozen ImageNet ResNet-50 are averaged per image and classified with an SVM.
Same evaluation as the band baseline (leave-one-plant-out, plus leave-one-time-point-out for light),
so the two can be compared directly: does keeping the spatial information help?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.cnn import tile_features  # noqa: E402
from stress.data import load  # noqa: E402
from stress.evaluate import hours_metrics, leave_one_plant_out, summarise  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def classifier():
    return make_pipeline(StandardScaler(), PCA(n_components=10), SVC(kernel="linear", C=0.1))


def regressor():
    return make_pipeline(StandardScaler(), PCA(n_components=10), SVR(kernel="linear", C=1, epsilon=4))


def main():
    samples = load()
    X = np.stack([f.mean(0) for f in tile_features(samples)])
    light = np.array([s.light for s in samples])
    hours = np.array([s.hours for s in samples])
    plants = np.array([s.plant for s in samples])
    out = {"n_tiles_per_image": [len(f) for f in tile_features(samples)]}

    pred = np.empty_like(light)
    for _, tr, te in leave_one_plant_out(plants):
        pred[te] = classifier().fit(X[tr], light[tr]).predict(X[te])
    out["light/LOPO"] = summarise(light, pred, hours, None)

    pred = np.empty_like(light)
    for h in np.unique(hours):
        tr, te = np.flatnonzero(hours != h), np.flatnonzero(hours == h)
        pred[te] = classifier().fit(X[tr], light[tr]).predict(X[te])
    out["light/leave-one-time-point-out"] = summarise(light, pred, hours, None)

    hp = np.zeros(len(hours))
    for _, tr, te in leave_one_plant_out(plants):
        hp[te] = regressor().fit(X[tr], hours[tr]).predict(X[te])
    out["hours/LOPO"] = hours_metrics(hours, hp)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "02_cnn_features_svm.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        if isinstance(v, dict):
            print(k, {m: (round(x, 3) if isinstance(x, float) else x) for m, x in v.items() if m != "light_confusion"})
        else:
            print(k, v)


if __name__ == "__main__":
    main()
