"""Classic classifiers on frozen ResNet-50 patch features, shared by scripts 14 to 17.

  logreg  logistic regression on standardised features (setting: C)
  svm     support vector machine on standardised features (settings: kernel linear or rbf, C, and gamma
          for rbf). The linear kernel is fitted on all patches; the rbf kernel on at most SVM_PATCHES
          (150) evenly spaced patches per training image, because its cost grows with the square of
          the number of patches.
  plsda   PLS-DA: PLS regression onto the one-hot classes (setting: number of components)

One setting per classifier for the whole experiment: script 14 finds it with Bayesian optimisation on
the 6 inner splits and saves it to results/14_classic_settings.json; scripts 16 and 17 load it.
The patches of an image vote (mean score over its patches).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from .data import CONTROL, gains, white_reference
from .evaluate import report
from .patches import features

RESULTS = Path(__file__).resolve().parents[1] / "results"
SETTINGS = RESULTS / "14_classic_settings.json"
PATCH, STRIDE = int(os.environ.get("PATCH", 64)), int(os.environ.get("STRIDE", 8))
MAX_PATCHES = int(os.environ.get("MAX_PATCHES", 600))
SVM_PATCHES = int(os.environ.get("SVM_PATCHES", 150))
FALLBACK = {"logreg": {"C": 1e-3}, "svm": {"kernel": "linear", "C": 1e-4}, "plsda": {"n_components": 20}}


def fold_features(samples, folds, tag="", offsets=None, net="resnet50"):
    """{fold: patch features per image}, white-light normalised with the training plants of that fold.
    offsets: optional {fold: per-image band offsets} (image-level mean-centring)."""
    out = {}
    for fold, tr, te in folds:
        g = [gi * 0.5 for gi in gains(samples, white_reference(samples, tr))]
        out[fold] = features(samples, g, PATCH, STRIDE, MAX_PATCHES, f"white-{fold}{tag}", net,
                             offsets=None if offsets is None else offsets[fold])
    return out


def stack(feats, idx, labels):
    """(patches, patch labels, image of every patch) for the images idx."""
    return (np.concatenate([feats[i] for i in idx]),
            np.concatenate([[labels[i]] * len(feats[i]) for i in idx]),
            np.concatenate([[i] * len(feats[i]) for i in idx]))


def fit(kind, p, X, y, owner, n_classes):
    """Fitted classifier as a function patch features -> class scores (n x n_classes)."""
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=3000)).fit(X, y).predict_proba
    if kind == "svm":
        if p["kernel"] == "linear":
            return make_pipeline(StandardScaler(), LinearSVC(C=p["C"], dual=False, max_iter=2000)).fit(X, y).decision_function
        keep = np.concatenate([np.flatnonzero(owner == i)[np.unique(np.linspace(0, (owner == i).sum() - 1, SVM_PATCHES).round().astype(int))]
                               for i in np.unique(owner)])
        m = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=p["C"], gamma=p["gamma"]))
        return m.fit(X[keep], y[keep]).decision_function
    if kind == "plsda":
        return PLSRegression(n_components=p["n_components"]).fit(X, np.eye(n_classes)[y]).predict
    raise ValueError(kind)


def vote(scores, feats, idx):
    """Each image's class: the highest mean score over its patches."""
    return np.array([int(scores(feats[i]).mean(0).argmax()) for i in idx])


def settings(kinds):
    """The saved setting of every classifier (script 14), or the 22 Nov grid choice if 14 has not run."""
    saved = json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}
    out = {}
    for k in kinds:
        if k in saved:
            out[k] = saved[k]["params"]
        else:
            print(f"  note: no saved setting for {k} (run scripts/14_optuna_classic.py); using {FALLBACK[k]}", flush=True)
            out[k] = FALLBACK[k]
    return out


def test(kind, p, feats, folds, labels, hours, classes):
    """Leave-one-plant-out test of one fixed setting; feats = {fold: features per image}."""
    true, pred, hrs, run_acc, cal_acc, by_plant = [], [], [], [], [], {}
    for fold, tr, te in folds:
        X, y, o = stack(feats[fold], tr, labels)
        sc = fit(kind, p, X, y, o, len(classes))
        p_te, p_tr = vote(sc, feats[fold], te), vote(sc, feats[fold], tr)
        run_acc.append(float(np.mean(p_te == labels[te])))
        cal_acc.append(float(np.mean(p_tr == labels[tr])))
        true += [classes[i] for i in labels[te]]
        pred += [classes[i] for i in p_te]
        hrs += hours[te].tolist()
        h = hours[te]
        by_plant[fold] = {f"{x}h": float(np.mean(p_te[h == x] == labels[te][h == x])) for x in sorted(set(h.tolist()))}
    return {"setting": p, "accuracy_by_test_plant_and_time": by_plant,
            **report(true, pred, hrs, run_acc, classes, CONTROL, cal_acc)}


def line(name, r):
    a, c = r["accuracy_over_runs"], r["calibration_accuracy_over_runs"]
    return (f"  {name}: test {a['median']:.2f} (mean {a['mean']:.2f}), healthy vs stressed "
            f"{r['plant level, stressed vs control']['accuracy']:.2f}, calibration {c['median']:.2f}")


def row(r):
    """Numbers for a csv row: test median, mean, healthy vs stressed, calibration, then per time point."""
    return [f"{r['accuracy_over_runs']['median']:.2f}", f"{r['accuracy_over_runs']['mean']:.2f}",
            f"{r['plant level, stressed vs control']['accuracy']:.2f}", f"{r['calibration_accuracy_over_runs']['median']:.2f}",
            *(f"{r['by_time_point'][h]['accuracy']:.2f}" for h in ("4h", "24h", "48h", "72h"))]


CSV_HEAD = ["Test accuracy (median)", "Test accuracy (mean)", "Healthy vs stressed", "Calibration accuracy",
            "4 h", "24 h", "48 h", "72 h"]
