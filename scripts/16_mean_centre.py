"""Per-plant mean-centring before the classic classifiers (Dr Praveen, 26 Nov).

Each plant has its own overall look (a little brighter, larger, imaged on another day). Mean-centring
removes that offset, so the classifier sees only how an image differs from its own plant. No labels are
used, so it is fair on the test plant too. Variants (VARIANTS):

  none             as before (white-light normalised features)
  feature          every patch feature minus its plant's mean patch feature (over all 16 images of the
                   plant), plus the mean over the three plants
  image            every image's plant pixels shifted, band by band, by (mean over the three plants -
                   its plant's mean), before ResNet-50
  feature_outlier  as feature, but only the outlier plant is moved, to the mean of the other two; the
  image_outlier    as image, same rule. Good plants are left as they are, so they are not pulled
                   towards a mean that the outlier has skewed (Dr Praveen: distances 1, 2 and 20).
  feature_farthest as feature_outlier, but the farthest plant is always moved (even below OUTLIER_RATIO).

Outlier: every image is one point (mean of its patch features, standardised, first NPC (10) principal
components, as in scripts/15); for each plant, the distance of its mean point from the mean of the other
two plants. The largest is an outlier if it is at least OUTLIER_RATIO (1.5) times the second largest;
otherwise no plant is moved and the _outlier variants equal none. OUTLIER=P2 forces a plant.
White-light normalisation, patch 64, stride 8, at most 600 patches, all bands, leave one plant out,
patch vote; classifier settings from scripts/14 (results/14_classic_settings.json).

    python scripts/16_mean_centre.py
    VARIANTS="none feature" CLASSIFIERS="logreg" python scripts/16_mean_centre.py

Writes results/16_mean_centre.json and .csv.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.classic import CSV_HEAD, RESULTS, fold_features, line, row, settings, test  # noqa: E402
from stress.data import conditions, gains, load, white_reference  # noqa: E402
from stress.evaluate import leave_one_plant_out  # noqa: E402

VARIANTS = os.environ.get("VARIANTS", "none feature image feature_outlier image_outlier feature_farthest").split()
KINDS = os.environ.get("CLASSIFIERS", "logreg plsda").split()
RATIO = float(os.environ.get("OUTLIER_RATIO", 1.5))
FORCE = os.environ.get("OUTLIER")
NPC = int(os.environ.get("NPC", 10))


def outlier(feats, plant):
    """(outlier plant or None, {plant: distance from the mean of the other two})."""
    m = PCA(n_components=NPC).fit_transform(StandardScaler().fit_transform(np.stack([f.mean(0) for f in feats])))
    ps = np.unique(plant)
    centre = {p: m[plant == p].mean(0) for p in ps}
    d = {int(p): float(np.linalg.norm(centre[p] - np.mean([centre[q] for q in ps if q != p], 0))) for p in ps}
    if FORCE:
        return (None if FORCE == "none" else int(FORCE.lstrip("P"))), d
    top = sorted(d.values(), reverse=True)
    return (max(d, key=d.get) if top[0] >= RATIO * top[1] else None), d


def shifts(means, plant, moved):
    """{plant: vector to add}: all plants to the mean of all, or only `moved` to the mean of the others."""
    ps = list(means)
    if moved == "all":
        target = np.mean([means[p] for p in ps], 0)
        return {p: target - means[p] for p in ps}
    out = {p: np.zeros_like(means[p]) for p in ps}
    if moved is not None:
        out[moved] = np.mean([means[p] for p in ps if p != moved], 0) - means[moved]
    return out


def main():
    samples = load()
    classes = conditions(samples)
    labels = np.array([classes.index(s.light) for s in samples])
    hours = np.array([s.hours for s in samples])
    plant = np.array([s.plant for s in samples])
    folds = list(leave_one_plant_out(plant))
    params = settings(KINDS)
    plain = fold_features(samples, folds)
    found = {fold: outlier(plain[fold], plant) for fold, _, _ in folds}
    for fold, (o, d) in found.items():
        print(f"  test {fold}: distance of each plant from the other two {d} -> outlier "
              f"{'P%d' % o if o else 'none'}", flush=True)
    out, rows = {"outlier": {f: {"plant": o, "distances": d} for f, (o, d) in found.items()}, "settings": params}, []
    for v in VARIANTS:
        moved = lambda fold: ("all" if v in ("feature", "image") else found[fold][0] if v.endswith("_outlier")
                              else max(found[fold][1], key=found[fold][1].get))
        if v == "none":
            feats = plain
        elif v.startswith("feature"):
            feats = {}
            for fold, _, _ in folds:
                means = {p: np.concatenate([plain[fold][i] for i in np.flatnonzero(plant == p)]).mean(0) for p in np.unique(plant)}
                sh = shifts(means, plant, moved(fold))
                feats[fold] = [f + sh[plant[i]] for i, f in enumerate(plain[fold])]
        else:
            offsets, tags = {}, set()
            for fold, tr, te in folds:
                g = [gi * 0.5 for gi in gains(samples, white_reference(samples, tr))]
                px = [np.clip(s.pixels() * gi, 0, 1) for s, gi in zip(samples, g)]
                means = {p: np.concatenate([px[i] for i in np.flatnonzero(plant == p)]).mean(0) for p in np.unique(plant)}
                sh = shifts(means, plant, moved(fold))
                offsets[fold] = [sh[plant[i]] for i in range(len(samples))]
                tags.add(moved(fold))
                print(f"  {v}, test {fold}: band shifts " + ", ".join(f"P{p} {np.round(s, 3).tolist()}" for p, s in sh.items()), flush=True)
            if v == "image_outlier" and tags == {None}:
                feats = plain
            else:
                tag = "_imgc" if v == "image" else f"_imgc-out{RATIO:g}{FORCE or ''}"
                feats = fold_features(samples, folds, tag, offsets)
        out[v] = {}
        for kind in KINDS:
            r = test(kind, params[kind], feats, folds, labels, hours, classes)
            out[v][kind] = r
            print(line(f"{v} {kind}", r), flush=True)
            rows.append([v, kind, json.dumps(params[kind]), *row(r)])
        (RESULTS / "16_mean_centre.json").write_text(json.dumps(out, indent=2))
    with open(RESULTS / "16_mean_centre.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Mean-centring", "Classifier", "Setting", *CSV_HEAD])
        w.writerows(rows)
    print("-> results/16_mean_centre.json, 16_mean_centre.csv")


if __name__ == "__main__":
    main()
