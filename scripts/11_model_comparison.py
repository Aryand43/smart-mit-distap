"""Five-model comparison on the three-band images, all tested the same way (Dr Praveen, 14 Nov).

  * PLS-DA, SVM and gradient boosting on whole-plant band features (mean and spread of each band)
  * ResNet-50 and a vision transformer (ViT-B/16) as frozen ImageNet backbones on patches, each with
    the same softmax head and plant vote, using the one setting locked by 10 (patch size, stride,
    learning rate, optimiser, batch size; epochs from early stopping)

Leave-one-plant-out, white-light normalised. Models with randomness (gradient boosting, the patch
models) are run with 5 seeds per test plant, so 15 runs; PLS-DA and SVM are deterministic (3 runs).
Writes results/11_model_comparison.json and results/11_model_comparison.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.data import CONTROL, conditions, gains, load, white_reference  # noqa: E402
from stress.evaluate import leave_one_plant_out, report  # noqa: E402
from stress.models import PLSDA, band_features  # noqa: E402
from stress.patches import evaluate  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS = 5

CLASSIC = {
    "PLS-DA": (lambda classes, seed: PLSDA(classes, 2), False),
    "SVM": (lambda classes, seed: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")), False),
    "Gradient boosting": (lambda classes, seed: GradientBoostingClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=seed), True),
}


def run_classic(samples, make, seeded):
    classes = conditions(samples)
    light = np.array([s.light for s in samples])
    hours = np.array([s.hours for s in samples])
    plants = np.array([s.plant for s in samples])
    true, pred, hrs, run_acc = [], [], [], []
    for fold, tr, te in leave_one_plant_out(plants):
        g = gains(samples, white_reference(samples, tr))
        X = np.stack([band_features(s.pixels() * gi) for s, gi in zip(samples, g)])
        for seed in range(SEEDS if seeded else 1):
            p = make(classes, seed).fit(X[tr], light[tr]).predict(X[te])
            run_acc.append(float(np.mean(p == light[te])))
            true += light[te].tolist()
            pred += list(p)
            hrs += hours[te].tolist()
    return report(true, pred, hrs, run_acc, classes, CONTROL)


def table(results) -> str:
    rows = ["| Model | Input | Accuracy, median (mean) | Precision | Recall | F1 | At 4 h | Healthy vs stressed |",
            "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        a, p = r["accuracy_over_runs"], r["plant level"]
        rows.append(f"| {name} | {r['input']} | {a['median']:.2f} ({a['mean']:.2f}) | {p['precision_macro']:.2f} | "
                    f"{p['recall_macro']:.2f} | {p['f1_macro']:.2f} | {p['accuracy_by_hour'][4]:.2f} | "
                    f"{r['plant level, stressed vs control']['accuracy']:.2f} |")
    return "\n".join(rows)


def main():
    samples = load()
    locked = json.loads((RESULTS / "10_final_patch_model.json").read_text())
    setting, max_patches = locked["selected_setting"], locked["max_patches"]
    patch_input = f"{setting['patch']} px patches, stride {setting['stride']}"

    results = {}
    for name, (make, seeded) in CLASSIC.items():
        results[name] = {"input": "whole-plant band features", **run_classic(samples, make, seeded)}
        print(f"{name}: {results[name]['accuracy_over_runs']['median']:.2f}", flush=True)
    for name, net in [("ResNet-50", "resnet50"), ("Vision transformer (ViT-B/16)", "vit_b_16")]:
        results[name] = {"input": patch_input, **evaluate(samples, setting, max_patches, net, seeds=SEEDS)}
        print(f"{name}: {results[name]['accuracy_over_runs']['median']:.2f}", flush=True)

    out = {"locked_setting": setting, "models": results}
    (RESULTS / "11_model_comparison.json").write_text(json.dumps(out, indent=2))
    md = table(results)
    (RESULTS / "11_model_comparison.md").write_text(
        "# Five-model comparison, choy sum three-band images\n\n" + md +
        "\n\nFour conditions, chance 0.25. Leave-one-plant-out, white-light normalised. Accuracy is the median "
        "(mean) over runs; precision, recall and F1 are macro averages pooled over runs.\n")
    print(md)


if __name__ == "__main__":
    main()
