"""Simpler classifiers on frozen backbone features (Dr Praveen, 22 Nov).

The softmax head of 09 / 10 (about 130,000 parameters) memorises the two training plants: the validation
loss is lowest after one epoch and then rises. Here the backbone stays frozen and the classifier on the
pooled patch features is made as simple as possible, for three backbones:

  backbones  ResNet-50 (2048 features), ResNet-18 (512), MobileNet-V3-small (576)
  classifiers
    linear   one linear layer, dropout DROP (0.5) before it, label smoothing LS (0.15), AdamW with a low
             learning rate LR (1e-4) and weight decay WD (1e-2); epochs from the averaged validation curve
    logreg   logistic regression (C picked on the inner splits)
    svm      linear SVM (C picked on the inner splits)
    plsda    PLS-DA (number of components picked on the inner splits)

Same test as before: leave one plant out, inside each round the two training plants validate each other
(6 inner splits in total), the patches of a test image vote (mean score over its patches). One setting
and one epoch count for all three test plants. Patch 64, stride 8, at most 600 patches per image, all
three bands, no augmentation (the ResNet-50 features are reused from the cache of 09 / 10).

    python scripts/12_frozen_heads.py
    BACKBONES_RUN="resnet50" HEADS="linear logreg" python scripts/12_frozen_heads.py

Writes results/12_frozen_heads.json, results/12_frozen_heads.csv and results/12_curves_<backbone>.png.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.data import CONTROL, conditions, gains, load, white_reference  # noqa: E402
from stress.evaluate import leave_one_plant_out, report  # noqa: E402
from stress.patches import features, inner_splits, stack  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
PATCH, STRIDE = int(os.environ.get("PATCH", 64)), int(os.environ.get("STRIDE", 8))
MAX_PATCHES = int(os.environ.get("MAX_PATCHES", 600))
RUN_BACKBONES = os.environ.get("BACKBONES_RUN", "resnet50 resnet18 mobilenet_v3_small").split()
HEADS = os.environ.get("HEADS", "linear logreg svm plsda").split()
DROP, LS = float(os.environ.get("DROP", 0.5)), float(os.environ.get("LS", 0.15))
LR, WD = float(os.environ.get("LR", 1e-4)), float(os.environ.get("WD", 1e-2))
MAX_EPOCHS, HEAD_BATCH = int(os.environ.get("MAX_EPOCHS", 60)), int(os.environ.get("HEAD_BATCH", 64))
SEEDS = int(os.environ.get("SEEDS", 5))
GRID = {"logreg": [1e-4, 1e-3, 1e-2, 1e-1], "svm": [1e-4, 1e-3, 1e-2], "plsda": [2, 5, 10, 20]}


def linear(X, y, Xv, yv, epochs, seed, n_classes):
    """Dropout + one linear layer on standardised features. Returns (scores, curve); curve[k] =
    (training loss, validation loss, validation accuracy) after k epochs, k = 0 before training.
    Both losses are the training objective (cross-entropy with label smoothing)."""
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    f = lambda A: torch.from_numpy((A - mu) / sd).float()
    Xt, yt = f(X), torch.from_numpy(y).long()
    net = torch.nn.Sequential(torch.nn.Dropout(DROP), torch.nn.Linear(X.shape[1], n_classes))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    loss = lambda o, t: torch.nn.functional.cross_entropy(o, t, label_smoothing=LS)
    Xvt, yvt = (f(Xv), torch.from_numpy(yv).long()) if Xv is not None else (None, None)
    curve = []

    def record(train_loss):
        if Xvt is None:
            return
        net.eval()
        with torch.no_grad():
            o = net(Xvt)
            curve.append((train_loss, loss(o, yvt).item(), (o.argmax(-1) == yvt).float().mean().item()))

    if Xvt is not None:
        net.eval()
        with torch.no_grad():
            record(loss(net(Xt), yt).item())
    for _ in range(epochs):
        net.train()
        total = 0.0
        for b in torch.randperm(len(Xt)).split(HEAD_BATCH):
            l = loss(net(Xt[b]), yt[b])
            opt.zero_grad()
            l.backward()
            opt.step()
            total += l.item() * len(b)
        record(total / len(Xt))
    net.eval()

    def scores(A):
        with torch.no_grad():
            return torch.softmax(net(f(A)), -1).numpy()
    return scores, curve


def sklearn_model(kind, param, X, y, n_classes):
    """Fitted classifier as a function patch features -> class scores."""
    if kind == "logreg":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=param, max_iter=3000)).fit(X, y)
        return m.predict_proba
    if kind == "svm":
        m = make_pipeline(StandardScaler(), LinearSVC(C=param, dual=False, max_iter=5000)).fit(X, y)
        return m.decision_function
    m = PLSRegression(n_components=param).fit(X, np.eye(n_classes)[y])
    return m.predict


def vote(scores, feats, idx):
    """Each image's class: the highest mean score over its patches."""
    return np.array([int(scores(feats[i]).mean(0).argmax()) for i in idx])


def summary(rounds, labels, hours, classes, runs):
    """report() plus accuracy per test plant and time point, from runs = [(fold, tr, te, pred_te, pred_tr)]."""
    true, pred, hrs, run_acc, cal_acc, by_plant = [], [], [], [], [], {}
    for fold, tr, te, p_te, p_tr in runs:
        run_acc.append(float(np.mean(p_te == labels[te])))
        cal_acc.append(float(np.mean(p_tr == labels[tr])))
        true += [classes[i] for i in labels[te]]
        pred += [classes[i] for i in p_te]
        hrs += hours[te].tolist()
    t, q, h = map(np.array, (true, pred, hrs))
    for fold in sorted({r[0] for r in runs}):
        sel = np.concatenate([np.full(len(r[2]), r[0] == fold) for r in runs])
        tf, qf, hf = t[sel], q[sel], h[sel]
        by_plant[fold] = {f"{x}h": float(np.mean(tf[hf == x] == qf[hf == x])) for x in sorted(set(hf.tolist()))}
    return {"accuracy_by_test_plant_and_time": by_plant,
            **report(true, pred, hrs, run_acc, classes, CONTROL, cal_acc)}


def plot(name, curves, mean, n_epochs):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ep = np.arange(len(mean))
    for c in curves:
        ax.plot(ep[:len(c)], [v[1] for v in c], color="#9D192B", alpha=0.15, lw=1)
    ax.plot(ep, [m[0] for m in mean], color="#7B7B79", lw=2, label="training loss (mean)")
    ax.plot(ep, [m[1] for m in mean], color="#9D192B", lw=2.5, label="validation loss (mean)")
    ax.axvline(n_epochs, ls="--", color="k", lw=1, label=f"epochs used: {n_epochs}")
    ax2 = ax.twinx()
    ax2.plot(ep, [m[2] for m in mean], color="#2B4BA0", lw=1.5, label="validation accuracy")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("validation accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (cross-entropy, label smoothing)")
    ax.set_title(f"{name}, linear head: mean over 6 validation splits")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / f"12_curves_{name}.png", dpi=130)
    plt.close(fig)


def main():
    samples = load()
    classes = conditions(samples)
    labels = np.array([classes.index(s.light) for s in samples])
    hours = np.array([s.hours for s in samples])
    plants = np.array([s.plant for s in samples])
    folds = list(leave_one_plant_out(plants))
    out_path = RESULTS / "12_frozen_heads.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    rows = []
    for name in RUN_BACKBONES:
        print(f"\n{name}: patch {PATCH}, stride {STRIDE}, at most {MAX_PATCHES} patches per image", flush=True)
        feats = {}
        for fold, tr, te in folds:
            g = [gi * 0.5 for gi in gains(samples, white_reference(samples, tr))]
            feats[fold] = features(samples, g, PATCH, STRIDE, MAX_PATCHES, f"white-{fold}", name)
        out.setdefault(name, {})
        for head in HEADS:
            if head == "linear":
                curves = [linear(*stack(feats[fold], itr, labels), *stack(feats[fold], ival, labels),
                                 MAX_EPOCHS, 0, len(classes))[1]
                          for fold, tr, te in folds for itr, ival in inner_splits(plants, tr)]
                mean = [tuple(float(np.mean([c[e][k] for c in curves])) for k in range(3))
                        for e in range(MAX_EPOCHS + 1)]
                n_epochs = max(1, min(range(1, MAX_EPOCHS + 1), key=lambda e: mean[e][1]))
                plot(name, curves, mean, n_epochs)
                runs = []
                for fold, tr, te in folds:
                    X, y = stack(feats[fold], tr, labels)
                    for seed in range(SEEDS):
                        sc, _ = linear(X, y, None, None, n_epochs, seed, len(classes))
                        runs.append((fold, tr, te, vote(sc, feats[fold], te), vote(sc, feats[fold], tr)))
                setting = {"epochs": n_epochs, "lr": LR, "weight_decay": WD, "dropout": DROP,
                           "label_smoothing": LS, "batch_size": HEAD_BATCH}
                extra = {"validation_curve_mean": mean, "curve_from_epoch": 0}
                note = f"epochs {n_epochs} (minimum of the mean validation loss; last tried {MAX_EPOCHS})"
            else:
                val = {}
                for param in GRID[head]:
                    accs = []
                    for fold, tr, te in folds:
                        for itr, ival in inner_splits(plants, tr):
                            sc = sklearn_model(head, param, *stack(feats[fold], itr, labels), len(classes))
                            Xv, yv = stack(feats[fold], ival, labels)
                            accs.append(float(np.mean(sc(Xv).argmax(1) == yv)))
                    val[param] = float(np.mean(accs))
                    print(f"  {head} {param}: inner validation accuracy (patches) {val[param]:.2f}", flush=True)
                best = max(val, key=val.get)
                runs = []
                for fold, tr, te in folds:
                    sc = sklearn_model(head, best, *stack(feats[fold], tr, labels), len(classes))
                    runs.append((fold, tr, te, vote(sc, feats[fold], te), vote(sc, feats[fold], tr)))
                setting = {"C" if head != "plsda" else "components": best}
                extra = {"inner_validation_accuracy": {str(k): v for k, v in val.items()}}
                note = f"{list(setting)[0]} {best}"
            res = {"setting": setting, **extra, **summary(folds, labels, hours, classes, runs)}
            out[name][head] = res
            a, c = res["accuracy_over_runs"], res["calibration_accuracy_over_runs"]
            hs = res["plant level, stressed vs control"]["accuracy"]
            print(f"  {name} {head}: test {a['median']:.2f} (mean {a['mean']:.2f}), healthy vs stressed "
                  f"{hs:.2f}, calibration {c['median']:.2f}; {note}", flush=True)
            out_path.write_text(json.dumps(out, indent=2))      # after every model, so a crash keeps the rest
    for name, heads in out.items():
        for head, r in heads.items():
            rows.append([name, head, f"{r['accuracy_over_runs']['median']:.2f}", f"{r['accuracy_over_runs']['mean']:.2f}",
                         f"{r['plant level, stressed vs control']['accuracy']:.2f}",
                         f"{r['calibration_accuracy_over_runs']['median']:.2f}",
                         *(f"{r['by_time_point'][h]['accuracy']:.2f}" for h in ("4h", "24h", "48h", "72h")),
                         json.dumps(r["setting"])])
    with open(RESULTS / "12_frozen_heads.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Backbone", "Classifier", "Test accuracy (median)", "Test accuracy (mean)", "Healthy vs stressed",
                    "Calibration accuracy", "4 h", "24 h", "48 h", "72 h", "Setting"])
        w.writerows(rows)
    print(f"\n-> {out_path}, results/12_frozen_heads.csv")


if __name__ == "__main__":
    main()
