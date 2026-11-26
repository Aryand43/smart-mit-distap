"""Band ablation again, with the classic classifiers (Dr Praveen, 26 Nov).

The 20 to 22 Nov ablation used the softmax head. Here the same bands are tested with logistic regression
and PLS-DA on frozen ResNet-50 features, with the settings saved by scripts/14 (not re-tuned per band, so
only the bands change). Bands as in stress/patches.py: one band (copied into all three channels), two
bands (third channel their mean), "mean" (average of the three) and "012" (all).

    python scripts/17_bands_classic.py
    BAND_SETS="0 1 2" python scripts/17_bands_classic.py

Writes results/17_bands_classic.json and .csv.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stress.patches as patches  # noqa: E402
from stress.classic import CSV_HEAD, RESULTS, fold_features, line, row, settings, test  # noqa: E402
from stress.data import conditions, load  # noqa: E402
from stress.evaluate import leave_one_plant_out  # noqa: E402

BAND_SETS = os.environ.get("BAND_SETS", "012 0 1 2 01 02 12 mean").split()
KINDS = os.environ.get("CLASSIFIERS", "logreg plsda").split()


def main():
    samples = load()
    classes = conditions(samples)
    labels = np.array([classes.index(s.light) for s in samples])
    hours = np.array([s.hours for s in samples])
    folds = list(leave_one_plant_out(np.array([s.plant for s in samples])))
    params = settings(KINDS)
    out, rows = {"settings": params}, []
    for b in BAND_SETS:
        patches.BANDS = b                       # read by select_bands and the cache name in features()
        feats = fold_features(samples, folds)
        out[b] = {}
        for kind in KINDS:
            r = test(kind, params[kind], feats, folds, labels, hours, classes)
            out[b][kind] = r
            print(line(f"bands {b} {kind}", r), flush=True)
            rows.append([b, kind, *row(r)])
        (RESULTS / "17_bands_classic.json").write_text(json.dumps(out, indent=2))
    with open(RESULTS / "17_bands_classic.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Bands", "Classifier", *CSV_HEAD])
        w.writerows(rows)
    print("-> results/17_bands_classic.json, 17_bands_classic.csv")


if __name__ == "__main__":
    main()
