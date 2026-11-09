"""Confusion matrix for each time point (4 h, day 1, day 2, day 3) from the saved dense-patch predictions.

Reads results/08_dense_patch_predictions.npz (or the file given as the first argument), so nothing is
rerun. Each test image's patches vote once per seed, exactly as in 08.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.data import HOURS  # noqa: E402
from stress.evaluate import classification_metrics  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
_NAME = re.compile(r"^(HRS_4|D\d)_([A-Z]+)_")


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "08_dense_patch_predictions.npz"
    z = np.load(src)
    classes = [str(c) for c in z["classes"]]
    true, pred, hours = [], [], []
    for name in z.files:
        if name == "classes" or name.endswith("|pos"):
            continue
        t, light = _NAME.match(name).groups()
        p = z[name]
        for k in range(p.shape[1]):
            true.append(light)
            pred.append(classes[int(np.bincount(p[:, k], minlength=len(classes)).argmax())])
            hours.append(HOURS[t])
    true, pred, hours = map(np.array, (true, pred, hours))
    binary = lambda ys: np.where(ys == classes[0], "control", "stressed")

    out = {"source": src.name}
    for h in [None, *sorted(set(hours))]:
        sel = np.ones(len(true), bool) if h is None else hours == h
        m = classification_metrics(true[sel], pred[sel], classes)
        m["stressed_vs_control_accuracy"] = classification_metrics(
            binary(true[sel]), binary(pred[sel]), ["control", "stressed"])["accuracy"]
        out["all" if h is None else f"{h}h"] = m
        print(f"{'all' if h is None else f'{h} h':>5}: accuracy {m['accuracy']:.2f}, "
              f"stressed vs control {m['stressed_vs_control_accuracy']:.2f}, {m['confusion']['rows_true_cols_pred']}")
    (RESULTS / "08b_confusion_by_time.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
