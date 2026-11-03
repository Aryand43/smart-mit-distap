"""Classification metrics for every technique, with and without white-light normalisation.

Following Dr Praveen's feedback (30 Oct):
  * Two loops. Outer: rotate the test plant (train on two plants, test on the third, 3 rounds).
    Inner, for the pixel-based PLS-DA: 20 different sets of 5,000 pixels per training image,
    non-overlapping where the plant is large enough. That gives 60 runs per technique.
  * Accuracy counts plants, not pixels: each test plant's pixels (or patches) vote, the most-voted
    condition becomes its label, so in each run every test plant is either right or wrong.
  * The headline accuracy is the median over runs, so a run that drew unlucky pixels does not pull
    the result down. The confusion matrix, TP / FP / FN / TN, precision, recall and F1 are pooled
    over all runs.
  * White-light normalisation: every image is divided by the mean white-light band values of its
    time point (training plants only), so white light is 1 on every day.
  * Machine learning works on the whole plant (or its pixels); deep learning works on patches and is
    reported at patch level and at plant level after the patches vote.
  * Augmentation (flips and rotations) is applied to training patches only, never the test plant.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.cnn import tile_features, train_softmax  # noqa: E402
from stress.data import conditions, gains, load, white_reference  # noqa: E402
from stress.evaluate import accuracy_by_hour, classification_metrics, leave_one_plant_out, spread  # noqa: E402
from stress.models import PLSDA, band_features  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_DRAWS, N_PIXELS, SEEDS = 20, 5000, 5
CNN_WHITE_LEVEL = 0.5          # white light sits at 0.5 in the CNN input after normalisation


def vote(pred: np.ndarray, classes) -> np.ndarray:
    """Fraction of votes for each class."""
    return np.array([(pred == c).mean() for c in classes])


def run_norm(samples, norm: str) -> dict:
    classes = conditions(samples)
    light = np.array([s.light for s in samples])
    hours = np.array([s.hours for s in samples])
    plants = np.array([s.plant for s in samples])
    rng = np.random.default_rng(0)
    perms = [rng.permutation(len(s.pixels())) for s in samples]   # fixed order, so draws do not overlap

    runs = {k: {"acc": [], "true": [], "pred": [], "hours": []} for k in
            ["PLS-DA (whole plant)", "SVM (whole plant)", "PLS-DA (pixels, majority vote)",
             "ResNet-50 + SVM (whole plant)", "ResNet-50 patches + softmax (plant vote)"]}
    patch_true, patch_pred, pixel_acc, votes = [], [], [], {s.path.name: [] for s in samples}

    def record(name, te, pred):
        runs[name]["acc"].append(float((pred == light[te]).mean()))
        runs[name]["true"] += light[te].tolist()
        runs[name]["pred"] += list(pred)
        runs[name]["hours"] += hours[te].tolist()

    for fold, tr, te in leave_one_plant_out(plants):
        ref = white_reference(samples, tr) if norm == "white" else None
        g = gains(samples, ref)
        px = [s.pixels() * gi for s, gi in zip(samples, g)]

        # Machine learning on the whole plant: mean and spread of each band.
        X = np.stack([band_features(p) for p in px])
        record("PLS-DA (whole plant)", te, PLSDA(classes, 2).fit(X[tr], light[tr]).predict(X[te]))
        svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))
        record("SVM (whole plant)", te, svm.fit(X[tr], light[tr]).predict(X[te]))

        # Machine learning on pixels: inner loop over pixel draws, plant label by majority vote.
        for k in range(N_DRAWS):
            take = [perms[i][np.arange(k * N_PIXELS, (k + 1) * N_PIXELS) % len(perms[i])] for i in tr]
            Xp = np.concatenate([px[i][t] for i, t in zip(tr, take)])
            yp = np.repeat(light[tr], N_PIXELS)
            model = PLSDA(classes, 2).fit(Xp, yp)
            pred = []
            for i in te:
                pix = model.predict(px[i])
                frac = vote(pix, classes)
                votes[samples[i].path.name].append(frac.tolist())
                pixel_acc.append(float((pix == light[i]).mean()))
                pred.append(classes[int(frac.argmax())])
            record("PLS-DA (pixels, majority vote)", te, np.array(pred))

        # Deep learning features: ResNet-50 (last block), per-fold normalisation gets its own cache.
        tag = "raw" if norm == "raw" else f"white-{fold}"
        cg = None if norm == "raw" else [gi * CNN_WHITE_LEVEL for gi in g]
        plain = tile_features(samples, False, ("layer4",), cg, tag)
        Xc = np.stack([f.mean(0) for f in plain])
        clf = make_pipeline(StandardScaler(), PCA(n_components=10), SVC(kernel="linear", C=0.1))
        record("ResNet-50 + SVM (whole plant)", te, clf.fit(Xc[tr], light[tr]).predict(Xc[te]))

        aug = tile_features(samples, True, ("layer4",), cg, tag)
        Xa = np.concatenate([aug[i] for i in tr])
        ya = np.concatenate([[classes.index(light[i])] * len(aug[i]) for i in tr])
        for seed in range(SEEDS):
            proba = train_softmax(Xa, ya, len(classes), seed)
            pred = []
            for i in te:
                tile_pred = np.array(classes)[proba(plain[i]).argmax(1)]
                patch_true += [light[i]] * len(tile_pred)
                patch_pred += tile_pred.tolist()
                pred.append(classes[int(vote(tile_pred, classes).argmax())])
            record("ResNet-50 patches + softmax (plant vote)", te, np.array(pred))
        print(f"  {norm} {fold} done", flush=True)

    out = {}
    for name, r in runs.items():
        out[name] = {
            "level": "plant",
            "accuracy_over_runs": spread(r["acc"]),
            "accuracy_by_hour_pooled": accuracy_by_hour(r["true"], r["pred"], r["hours"]),
            **classification_metrics(r["true"], r["pred"], classes),
        }
    out["ResNet-50 patches + softmax (patch level)"] = {
        "level": "patch", **classification_metrics(patch_true, patch_pred, classes)}
    out["_pixel_accuracy_over_runs"] = spread(pixel_acc)
    out["_pixel_vote_fractions_median"] = {
        name: dict(zip(classes, np.median(v, 0).round(3).tolist())) for name, v in votes.items()}
    return out


def table(results: dict, classes) -> str:
    lines = ["| Technique | Normalisation | Level | Accuracy, median (IQR) over runs | Runs | Precision | Recall | F1 | Accuracy at 4 h |",
             "|---|---|---|---|---|---|---|---|---|"]
    for norm, res in results.items():
        for name, m in res.items():
            if name.startswith("_"):
                continue
            a = m.get("accuracy_over_runs")
            acc = f"{a['median']:.0%} ({a['q1']:.0%} to {a['q3']:.0%})" if a else f"{m['accuracy']:.0%} (pooled)"
            runs = a["n_runs"] if a else "pooled"
            h4 = m.get("accuracy_by_hour_pooled", {}).get(4)
            lines.append(f"| {name} | {norm} | {m['level']} | {acc} | {runs} | {m['precision_macro']:.2f} | "
                         f"{m['recall_macro']:.2f} | {m['f1_macro']:.2f} | {'' if h4 is None else f'{h4:.0%}'} |")
    lines += ["", f"Precision, recall and F1 are macro averages over {', '.join(classes)}, pooled over all runs.", ""]
    for norm, res in results.items():
        lines.append(f"### Per class, {norm}")
        lines += ["", "| Technique | Class | TP | FP | FN | TN | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|---|---|"]
        for name, m in res.items():
            if name.startswith("_"):
                continue
            for c, pc in m["per_class"].items():
                lines.append(f"| {name} | {c} | {pc['TP']} | {pc['FP']} | {pc['FN']} | {pc['TN']} | "
                             f"{pc['precision']:.2f} | {pc['recall']:.2f} | {pc['f1']:.2f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    samples = load()
    classes = conditions(samples)
    results = {}
    for norm in ["raw", "white"]:
        results[norm] = run_norm(samples, norm)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "05_metrics.json").write_text(json.dumps(results, indent=2))
    md = table(results, classes)
    (RESULTS / "05_metrics_table.md").write_text("# Classification metrics, choy sum\n\n" + md)
    print(md)


if __name__ == "__main__":
    main()
