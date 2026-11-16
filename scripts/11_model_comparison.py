"""Five-model comparison on the three-band images, all tested the same way (Dr Praveen, 14 and 16 Nov).

  * PLS-DA, SVM and gradient boosting on pixels: each training image gives 5,000 random plant pixels
    (3 band values each); this is repeated for 20 draws that do not overlap, so the draws cover
    different regions of every plant. Each test plant is labelled by the majority vote of its pixels.
    For speed, the SVM's RBF kernel is approximated (Nystroem, 100 components; linear SVM fitted by stochastic
    gradient descent) and gradient boosting is
    the histogram version; each vote uses a fixed sample of 20,000 pixels per image.
  * ResNet-50 and a vision transformer (ViT-B/16) as frozen ImageNet backbones on patches, each with
    its own setting from Bayesian optimisation (09 then 10, run per backbone). Every patch is resized
    to the 224 x 224 network input.

Leave-one-plant-out, white-light normalised. Pixel models: 20 draws per test plant (60 runs). Patch
models: 5 seeds per test plant (15 runs). Calibration accuracy is each model scored on its training plants.
Reads results/10_final_resnet50.json and results/10_final_vit_b_16.json.
Writes results/11_model_comparison.json and results/11_model_comparison.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.data import CONTROL, conditions, gains, load, white_reference  # noqa: E402
from stress.evaluate import leave_one_plant_out, report  # noqa: E402
from stress.models import PLSDA  # noqa: E402
from stress.patches import evaluate  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_DRAWS, N_PIXELS, N_VOTE = 20, 5000, 20000

CLASSIC = {
    "PLS-DA": lambda classes, k: PLSDA(classes, 2),
    "SVM": lambda classes, k: make_pipeline(StandardScaler(), Nystroem(gamma=1 / 3, n_components=100, random_state=k),
                                            SGDClassifier(loss="hinge", alpha=1e-5, random_state=k)),
    "Gradient boosting": lambda classes, k: HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, learning_rate=0.05, random_state=k),
}
DEEP = [("ResNet-50", "resnet50"), ("Vision transformer (ViT-B/16)", "vit_b_16")]


def run_pixels(samples, make):
    classes = conditions(samples)
    light = np.array([s.light for s in samples])
    hours = np.array([s.hours for s in samples])
    plants = np.array([s.plant for s in samples])
    rng = np.random.default_rng(0)
    perms = [rng.permutation(len(s.pixels())) for s in samples]   # fixed order, so draws do not overlap
    voters = [p[-N_VOTE:] for p in perms]                           # fixed pixels used for each image's vote

    def plant_vote(model, px, idx):
        return np.array([max(classes, key=list(model.predict(px[i][voters[i]])).count) for i in idx])

    true, pred, hrs, run_acc, cal_acc = [], [], [], [], []
    for fold, tr, te in leave_one_plant_out(plants):
        g = gains(samples, white_reference(samples, tr))
        px = [s.pixels() * gi for s, gi in zip(samples, g)]
        for k in range(N_DRAWS):
            take = [perms[i][np.arange(k * N_PIXELS, (k + 1) * N_PIXELS) % len(perms[i])] for i in tr]
            model = make(classes, k).fit(np.concatenate([px[i][t] for i, t in zip(tr, take)]),
                                         np.repeat(light[tr], N_PIXELS))
            p = plant_vote(model, px, te)
            run_acc.append(float(np.mean(p == light[te])))
            cal_acc.append(float(np.mean(plant_vote(model, px, tr) == light[tr])))
            true += light[te].tolist()
            pred += list(p)
            hrs += hours[te].tolist()
        print(f"  {fold}: test accuracy {np.median(run_acc[-N_DRAWS:]):.2f}", flush=True)
    return report(true, pred, hrs, run_acc, classes, CONTROL, cal_acc)


def table(results) -> str:
    rows = ["| Model | Network input | Patch, stride | Calibration | Test accuracy, median (mean) | Precision | "
            "Recall | F1 | At 4 h | Healthy vs stressed |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        a, c, p = r["accuracy_over_runs"], r["calibration_accuracy_over_runs"], r["plant level"]
        rows.append(f"| {name} | {r['input']} | {r['patch']} | {c['median']:.2f} | {a['median']:.2f} ({a['mean']:.2f}) | "
                    f"{p['precision_macro']:.2f} | {p['recall_macro']:.2f} | {p['f1_macro']:.2f} | "
                    f"{p['accuracy_by_hour'][4]:.2f} | {r['plant level, stressed vs control']['accuracy']:.2f} |")
    return "\n".join(rows)


def main():
    samples = load()
    locked = {}
    for _, net in DEEP:
        path = RESULTS / f"10_final_{net}.json"
        if not path.exists():
            raise SystemExit(f"Missing {path.name}: run 09 then 10 with BACKBONE={net} first")
        locked[net] = json.loads(path.read_text())

    results = {}
    for name, make in CLASSIC.items():
        print(name, flush=True)
        results[name] = {"input": f"{N_PIXELS:,} pixels per image x {N_DRAWS} draws", "patch": "none",
                         **run_pixels(samples, make)}
    for name, net in DEEP:
        s = locked[net]["selected_setting"]
        results[name] = {"input": "224 x 224", "patch": f"{s['patch']} px, stride {s['stride']}", "setting": s,
                         **evaluate(samples, s, locked[net]["max_patches"], net)}
        print(f"{name}: {results[name]['accuracy_over_runs']['median']:.2f}", flush=True)

    (RESULTS / "11_model_comparison.json").write_text(json.dumps({"models": results}, indent=2))
    md = table(results)
    (RESULTS / "11_model_comparison.md").write_text(
        "# Five-model comparison, choy sum three-band images\n\n" + md +
        "\n\nFour conditions, chance 0.25. Leave-one-plant-out, white-light normalised. Test accuracy is the "
        "median (mean) over runs; calibration is the median accuracy on the training plants; precision, recall "
        "and F1 are macro averages pooled over runs.\n")
    print(md)


if __name__ == "__main__":
    main()
