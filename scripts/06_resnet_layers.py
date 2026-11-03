"""Which ResNet-50 layer carries the stress signal? Trial and error over layers and combinations.

Lower blocks hold colour and fine texture, the last block holds object-level ImageNet patterns, so a
middle layer may suit stress better than the final one. Each layer's tile features are averaged per
plant, reduced to 10 PCA components inside each fold (training plants only), and combinations are
built by joining the reduced layers side by side, so every layer gets equal weight. The same linear
SVM classifies, with the test plant rotated as before.

Element-wise combinations (adding, multiplying) need features of the same size that line up; they
are left for a learned projection once the per-layer picture is clear.
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
from stress.cnn import LAYERS, tile_features  # noqa: E402
from stress.data import conditions, gains, load, white_reference  # noqa: E402
from stress.evaluate import accuracy_by_hour, classification_metrics, leave_one_plant_out, spread  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
CNN_WHITE_LEVEL = 0.5
COMBOS = [["layer1"], ["layer2"], ["layer3"], ["layer4"], ["layer1", "layer2"], ["layer2", "layer3"],
          ["layer3", "layer4"], ["layer2", "layer3", "layer4"], list(LAYERS)]


def reduce(X, tr, k=10):
    p = make_pipeline(StandardScaler(), PCA(n_components=k)).fit(X[tr])
    return p.transform(X)


def main():
    samples = load()
    classes = conditions(samples)
    light = np.array([s.light for s in samples])
    hours = np.array([s.hours for s in samples])
    plants = np.array([s.plant for s in samples])
    results = {}
    for norm in ["raw", "white"]:
        runs = {"+".join(c): {"acc": [], "pred": np.empty_like(light)} for c in COMBOS}
        for fold, tr, te in leave_one_plant_out(plants):
            tag, cg = "raw", None
            if norm == "white":
                tag = f"white-{fold}"
                cg = [g * CNN_WHITE_LEVEL for g in gains(samples, white_reference(samples, tr))]
            feats = tile_features(samples, False, tuple(LAYERS), cg, tag)
            widths = np.cumsum([0] + list(LAYERS.values()))
            per_layer = {l: reduce(np.stack([f[:, widths[j]:widths[j + 1]].mean(0) for f in feats]), tr)
                         for j, l in enumerate(LAYERS)}
            for combo in COMBOS:
                X = np.concatenate([per_layer[l] for l in combo], 1)
                clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.1)).fit(X[tr], light[tr])
                pred = clf.predict(X[te])
                r = runs["+".join(combo)]
                r["acc"].append(float((pred == light[te]).mean()))
                r["pred"][te] = pred
        results[norm] = {name: {"accuracy_over_folds": spread(r["acc"]),
                                "accuracy_by_hour": accuracy_by_hour(light, r["pred"], hours),
                                **classification_metrics(light, r["pred"], classes)}
                         for name, r in runs.items()}

    (RESULTS / "06_resnet_layers.json").write_text(json.dumps(results, indent=2))
    lines = ["| Layers | Normalisation | Accuracy | F1 (macro) | Accuracy at 4 h |", "|---|---|---|---|---|"]
    for norm, res in results.items():
        for name, m in res.items():
            lines.append(f"| {name} | {norm} | {m['accuracy']:.0%} | {m['f1_macro']:.2f} | {m['accuracy_by_hour'][4]:.0%} |")
    md = "\n".join(lines)
    (RESULTS / "06_resnet_layers.md").write_text("# ResNet-50 layer comparison (+ linear SVM), choy sum\n\n" + md + "\n")
    print(md)


if __name__ == "__main__":
    main()
