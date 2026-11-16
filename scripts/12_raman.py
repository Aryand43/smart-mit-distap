"""Raman only: the same leave-one-plant-out test on Dr Praveen's preprocessed Raman spectra.

One file per plant and time point, normalised to the peak ("NMBK"), named crop, condition, time and
plant, for example CS_HL_1D1_NMBK.csv: CS = choy sum, HL = high light, 1D = day 1, 1 = plant 1
(4H = 4 hours; 1D, 2D, 3D = days 1 to 3). Separators between the parts are optional.

A file may hold a single row of intensities, or two columns (Raman shift, intensity); several rows of
one spectrum are averaged. PLS-DA, SVM and gradient boosting run on the standardised spectra, tested
the same way as the image models in 11. Files go in data/RAMAN/<CROP>/ (or set RAMAN_DIR).
Writes results/12_raman.json and results/12_raman.md.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.data import CONTROL, ORDER  # noqa: E402
from stress.evaluate import leave_one_plant_out, report  # noqa: E402
from stress.models import PLSDA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CROP = os.environ.get("CROP", "CHOYSUM")
RAMAN_DIR = Path(os.environ.get("RAMAN_DIR", ROOT / "data" / "RAMAN" / CROP))
SEEDS = 5
TIMES = {"4H": 4, "1D": 24, "2D": 48, "3D": 72}
_NAME = re.compile(r"^([A-Z]{2})[ _-]*(HL|LL|SH|WL)[ _-]*(4H|1D|2D|3D)[ _-]*(\d)[ _-]*NMBK", re.IGNORECASE)

MODELS = {
    "PLS-DA": (lambda classes, seed: PLSDA(classes, 3), False),
    "SVM": (lambda classes, seed: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")), False),
    "Gradient boosting": (lambda classes, seed: GradientBoostingClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=seed), True),
}


def read_spectrum(path: Path) -> np.ndarray:
    """Intensities from a single-row file or a (shift, intensity) two-column file."""
    for delim in (",", None, "\t", ";"):
        a = np.genfromtxt(path, delimiter=delim)
        a = np.atleast_2d(a)
        a = a[~np.isnan(a).all(1)][:, ~np.isnan(a).all(0)]        # drop header rows and empty columns
        if a.size > 1:
            break
    else:
        raise ValueError(f"No numbers found in {path.name}")
    if a.shape[0] == 1:
        return a[0]
    if a.shape[1] == 2:                                          # Raman shift, intensity
        return a[:, 1]
    return a.mean(0)


def load_raman(folder: Path):
    files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))
    rows, skipped = [], []
    for p in files:
        m = _NAME.match(p.stem)
        if not m:
            skipped.append(p.name)
            continue
        _, cond, t, plant = m.groups()
        rows.append((cond.upper(), TIMES[t.upper()], int(plant), read_spectrum(p), p.name))
    if not rows:
        raise SystemExit(f"No NMBK files found in {folder}")
    lengths = {len(r[3]) for r in rows}
    if len(lengths) > 1:
        raise SystemExit(f"Spectra have different lengths {sorted(lengths)}; check the files")
    if skipped:
        print(f"Skipped {len(skipped)} files with unrecognised names: {skipped[:5]}")
    return rows


def main():
    rows = load_raman(RAMAN_DIR)
    light = np.array([r[0] for r in rows])
    hours = np.array([r[1] for r in rows])
    plants = np.array([r[2] for r in rows])
    X = np.stack([r[3] for r in rows])
    present = set(light.tolist())
    classes = [c for c in ORDER if c in present] + sorted(present - set(ORDER))
    print(f"{len(rows)} spectra of length {X.shape[1]}, conditions {classes}, plants {sorted(set(plants.tolist()))}")

    results = {}
    for name, (make, seeded) in MODELS.items():
        true, pred, hrs, run_acc = [], [], [], []
        for fold, tr, te in leave_one_plant_out(plants):
            for seed in range(SEEDS if seeded else 1):
                p = make(classes, seed).fit(X[tr], light[tr]).predict(X[te])
                run_acc.append(float(np.mean(p == light[te])))
                true += light[te].tolist()
                pred += list(p)
                hrs += hours[te].tolist()
        results[name] = report(true, pred, hrs, run_acc, classes, CONTROL)

    out = {"crop": CROP, "n_spectra": len(rows), "spectrum_length": int(X.shape[1]), "models": results}
    (RESULTS / f"12_raman{'' if CROP == 'CHOYSUM' else '_' + CROP.lower()}.json").write_text(json.dumps(out, indent=2))
    lines = ["| Model | Accuracy, median (mean) | Precision | Recall | F1 | At 4 h | Healthy vs stressed |",
             "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        a, p = r["accuracy_over_runs"], r["plant level"]
        at4 = p["accuracy_by_hour"].get(4)
        lines.append(f"| {name} | {a['median']:.2f} ({a['mean']:.2f}) | {p['precision_macro']:.2f} | "
                     f"{p['recall_macro']:.2f} | {p['f1_macro']:.2f} | {'' if at4 is None else f'{at4:.2f}'} | "
                     f"{r['plant level, stressed vs control']['accuracy']:.2f} |")
    md = "\n".join(lines)
    (RESULTS / f"12_raman{'' if CROP == 'CHOYSUM' else '_' + CROP.lower()}.md").write_text(
        f"# Raman only, {CROP.lower()}\n\n" + md + "\n")
    print(md)


if __name__ == "__main__":
    main()
