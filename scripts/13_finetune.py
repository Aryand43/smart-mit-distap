"""Two-phase fine-tuning of ResNet-50 (Dr Praveen, 22 Nov).

  Phase 1  backbone frozen; train only the head (dropout DROP 0.5 + one linear layer on the 2048 pooled
           features), learning rate LR1 (1e-3), P1 epochs (5).
  Phase 2  unfreeze the convolution weights of the last block (layer 4) only. Batch normalisation stays
           frozen (evaluation mode, ImageNet statistics, weights not trained), because the batches are
           small. Learning rate LR_L4 (1e-5) for layer 4 and LR2 (1e-4) for the head, up to P2 epochs (20).
  Both     AdamW, weight decay WD (1e-2), cross-entropy with label smoothing LS (0.15), random flips and
           90 degree rotations of the training patches. Batch sizes 8, 16 and 32 in phase 2 (BATCHES);
           phase 1 uses batch 64.

Epochs: every inner split (each round, one training plant trains and the other validates: 6 splits) is
run for the full P1 + P2 epochs and the training and validation loss are recorded after every epoch.
The validation curves are averaged and the phase 2 epoch count is the minimum of the mean validation
loss (0 = no fine-tuning), one count for all test plants (early stopping on the averaged curve). The
final model for each test plant is trained on both training plants with that count; it is scored after
phase 1 and after phase 2, so the two phases can be compared. The patches of a test image vote (mean
softmax).

Patch 64, stride 8, at most MAX_PATCHES (200) patches per image, all three bands. Fine-tuning cannot
reuse cached features, so this needs the GPU (about 1 to 3 hours for the three batch sizes).

    python scripts/13_finetune.py
    BATCHES="16" SEEDS=2 python scripts/13_finetune.py

Writes results/13_finetune_bs<b>.json and results/13_curves_bs<b>.png per batch size, and prints a table.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision.models import ResNet50_Weights, resnet50  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.cnn import _MEAN, _STD  # noqa: E402
from stress.data import CONTROL, conditions, gains, load, white_reference  # noqa: E402
from stress.evaluate import leave_one_plant_out, report  # noqa: E402
from stress.patches import inner_splits, positions, select_bands  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
E = os.environ.get
PATCH, STRIDE, MAX_PATCHES = int(E("PATCH", 64)), int(E("STRIDE", 8)), int(E("MAX_PATCHES", 200))
BATCHES = [int(b) for b in E("BATCHES", "8 16 32").split()]
P1, P2, P1_BATCH = int(E("P1", 5)), int(E("P2", 20)), int(E("P1_BATCH", 64))
LR1, LR2, LR_L4 = float(E("LR1", 1e-3)), float(E("LR2", 1e-4)), float(E("LR_L4", 1e-5))
WD, DROP, LS = float(E("WD", 1e-2)), float(E("DROP", 0.5)), float(E("LS", 0.15))
SEEDS = int(E("SEEDS", 1))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # Apple MPS cannot run this backward pass
AMP = DEV.type == "cuda"


class Net(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.body = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.body.fc = torch.nn.Identity()
        for p in self.body.parameters():
            p.requires_grad = False
        self.head = torch.nn.Sequential(torch.nn.Dropout(DROP), torch.nn.Linear(2048, n_classes))

    def train(self, mode=True):                     # the backbone, batch normalisation included, never leaves eval mode
        super().train(mode)
        self.body.eval()
        return self

    def unfreeze_layer4(self):
        """Phase 2: convolution weights of layer 4 learn; batch normalisation weights stay frozen."""
        params = []
        for m in self.body.layer4.modules():
            if isinstance(m, torch.nn.Conv2d):
                for p in m.parameters():
                    p.requires_grad = True
                    params.append(p)
        return params

    def forward(self, x, tune_l4=False):
        b = self.body
        with torch.no_grad():
            x = b.layer3(b.layer2(b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(x)))))))
        with torch.set_grad_enabled(tune_l4 and torch.is_grad_enabled()):
            x = b.layer4(x)
        return self.head(x.mean((2, 3)))


def patch_tensor(samples, g):
    """All patches of all images as one float16 tensor on the device (n, 3, PATCH, PATCH), ImageNet
    normalised, and the image index of every patch."""
    xs, owner = [], []
    for k, (s, gi) in enumerate(zip(samples, g)):
        img = select_bands(np.clip(s.image() * gi, 0, 1).astype(np.float32))
        pos = positions(s.mask(), PATCH, STRIDE, MAX_PATCHES)
        x = (np.stack([img[r:r + PATCH, c:c + PATCH] for r, c in pos]) - _MEAN) / _STD
        xs.append(torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).half())
        owner += [k] * len(pos)
    return torch.cat(xs).to(DEV), np.array(owner)


def prep(x, augment=False):
    x = x.float()
    if augment:                                     # one random flip / rotation per batch
        x = torch.rot90(x, int(torch.randint(4, ())), (2, 3))
        if torch.rand(()) < 0.5:
            x = x.flip(3)
    return torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False) if PATCH != 224 else x


def loss_fn(out, y):
    return torch.nn.functional.cross_entropy(out.float(), y, label_smoothing=LS)


@torch.no_grad()
def scores(model, X, idx):
    model.eval()
    out = []
    for b in torch.arange(len(idx)).split(256):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            out.append(model(prep(X[idx[b]])).float())
    return torch.cat(out)


def evaluate_loss(model, X, idx, y):
    o = scores(model, X, idx)
    return loss_fn(o, y).item(), (o.argmax(-1) == y).float().mean().item()


def epoch(model, opt, X, idx, y, batch, tune_l4):
    model.train()
    total = 0.0
    for b in torch.randperm(len(idx)).split(batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            l = loss_fn(model(prep(X[idx[b]], augment=True), tune_l4), y[b])
        opt.zero_grad()
        l.backward()
        opt.step()
        total += l.item() * len(b)
    return total / len(idx)


def train(X, owner, labels, tr_imgs, batch, n2, seed, val_imgs=None, n_classes=4, after_phase1=None):
    """Phase 1 (P1 epochs) then phase 2 (n2 epochs). With val_imgs, returns the curve: entry k =
    (training loss, validation loss, validation accuracy) after k epochs in total, k = 0 before training."""
    torch.manual_seed(seed)
    model = Net(n_classes).to(DEV)
    idx = torch.from_numpy(np.flatnonzero(np.isin(owner, tr_imgs))).to(DEV)
    y = torch.from_numpy(labels[owner[idx.cpu().numpy()]]).long().to(DEV)
    if val_imgs is not None:
        vidx = torch.from_numpy(np.flatnonzero(np.isin(owner, val_imgs))).to(DEV)
        vy = torch.from_numpy(labels[owner[vidx.cpu().numpy()]]).long().to(DEV)
    curve = []
    record = lambda tl: curve.append((tl, *evaluate_loss(model, X, vidx, vy))) if val_imgs is not None else None
    if val_imgs is not None:
        record(evaluate_loss(model, X, idx, y)[0])
    opt = torch.optim.AdamW(model.head.parameters(), lr=LR1, weight_decay=WD)
    for _ in range(P1):
        record(epoch(model, opt, X, idx, y, P1_BATCH, False))
    if after_phase1 is not None:
        after_phase1(model)
    opt = torch.optim.AdamW([{"params": model.unfreeze_layer4(), "lr": LR_L4},
                             {"params": model.head.parameters(), "lr": LR2}], weight_decay=WD)
    for _ in range(n2):
        record(epoch(model, opt, X, idx, y, batch, True))
    return model, curve


def vote(model, X, owner, imgs):
    """Each image's class: highest mean softmax over its patches."""
    out = []
    for i in imgs:
        p = torch.softmax(scores(model, X, torch.from_numpy(np.flatnonzero(owner == i)).to(DEV)), -1)
        out.append(int(p.mean(0).argmax()))
    return np.array(out)


def summarise(runs, labels, hours, classes):
    true, pred, hrs, run_acc, cal_acc, by_plant = [], [], [], [], [], {}
    for fold, tr, te, p_te, p_tr in runs:
        run_acc.append(float(np.mean(p_te == labels[te])))
        cal_acc.append(float(np.mean(p_tr == labels[tr])))
        t, q, h = labels[te], p_te, hours[te]
        by_plant.setdefault(fold, []).append((t, q, h))
        true += [classes[i] for i in t]
        pred += [classes[i] for i in q]
        hrs += h.tolist()
    per_plant = {}
    for fold, parts in by_plant.items():
        t, q, h = (np.concatenate([p[k] for p in parts]) for k in range(3))
        per_plant[fold] = {f"{x}h": float(np.mean(t[h == x] == q[h == x])) for x in sorted(set(h.tolist()))}
    return {"accuracy_by_test_plant_and_time": per_plant, **report(true, pred, hrs, run_acc, classes, CONTROL, cal_acc)}


def plot(batch, curves, mean, n2):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ep = np.arange(len(mean))
    for c in curves:
        ax.plot(ep, [v[1] for v in c], color="#9D192B", alpha=0.15, lw=1)
    ax.plot(ep, [m[0] for m in mean], color="#7B7B79", lw=2, label="training loss (mean)")
    ax.plot(ep, [m[1] for m in mean], color="#9D192B", lw=2.5, label="validation loss (mean)")
    ax.axvspan(0, P1, color="#7B7B79", alpha=0.08, label="phase 1: head only")
    ax.axvline(P1 + n2, ls="--", color="k", lw=1, label=f"epochs used: {P1} + {n2}")
    ax2 = ax.twinx()
    ax2.plot(ep, [m[2] for m in mean], color="#2B4BA0", lw=1.5, label="validation accuracy")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("validation accuracy")
    ax.set_xlabel("epoch (phase 1, then phase 2: layer 4 fine-tuned)")
    ax.set_ylabel("loss (cross-entropy, label smoothing)")
    ax.set_title(f"ResNet-50 two-phase fine-tuning, batch {batch}: mean over 6 validation splits")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / f"13_curves_bs{batch}.png", dpi=130)
    plt.close(fig)


def main():
    samples = load()
    classes = conditions(samples)
    labels = np.array([classes.index(s.light) for s in samples])
    hours = np.array([s.hours for s in samples])
    plants = np.array([s.plant for s in samples])
    folds = list(leave_one_plant_out(plants))
    print(f"ResNet-50 two-phase fine-tuning on {DEV}: patch {PATCH}, stride {STRIDE}, at most {MAX_PATCHES} "
          f"patches per image; phase 1 {P1} epochs (lr {LR1}), phase 2 up to {P2} epochs (layer 4 lr {LR_L4}, "
          f"head lr {LR2}); label smoothing {LS}, dropout {DROP}, weight decay {WD}", flush=True)
    data = {}
    for fold, tr, te in folds:
        g = [gi * 0.5 for gi in gains(samples, white_reference(samples, tr))]
        data[fold] = patch_tensor(samples, g)
    table = []
    for batch in BATCHES:
        t0, curves = time.time(), []
        for fold, tr, te in folds:
            X, owner = data[fold]
            for itr, ival in inner_splits(plants, tr):
                _, c = train(X, owner, labels, itr, batch, P2, 0, val_imgs=ival, n_classes=len(classes))
                curves.append(c)
                print(f"  batch {batch}, {fold}, validate on plant {plants[ival][0]}: validation loss "
                      + " ".join(f"{v[1]:.2f}" for v in c) + f"  ({time.time() - t0:.0f} s)", flush=True)
        mean = [tuple(float(np.mean([c[e][k] for c in curves])) for k in range(3)) for e in range(P1 + P2 + 1)]
        n2 = min(range(P2 + 1), key=lambda e: mean[P1 + e][1])
        print(f"  batch {batch}: phase 2 epochs {n2} (minimum of the mean validation loss)", flush=True)
        plot(batch, curves, mean, n2)
        runs1, runs2 = [], []
        for fold, tr, te in folds:
            X, owner = data[fold]
            for seed in range(SEEDS):
                keep = lambda m: runs1.append((fold, tr, te, vote(m, X, owner, te), vote(m, X, owner, tr)))
                model, _ = train(X, owner, labels, tr, batch, n2, seed, n_classes=len(classes), after_phase1=keep)
                runs2.append((fold, tr, te, vote(model, X, owner, te), vote(model, X, owner, tr)))
            print(f"  batch {batch}, test {fold}: phase 1 {np.mean(runs1[-1][3] == labels[te]):.2f}, "
                  f"phase 1 + 2 {np.mean(runs2[-1][3] == labels[te]):.2f}", flush=True)
        out = {"batch_size": batch, "phase1_epochs": P1, "phase2_epochs": n2,
               "settings": {"patch": PATCH, "stride": STRIDE, "max_patches": MAX_PATCHES, "lr_phase1_head": LR1,
                            "lr_phase2_layer4": LR_L4, "lr_phase2_head": LR2, "weight_decay": WD, "dropout": DROP,
                            "label_smoothing": LS, "batch_norm": "frozen", "seeds": SEEDS},
               "validation_curve_mean": mean, "validation_curves": curves, "curve_from_epoch": 0,
               "phase 1 only": summarise(runs1, labels, hours, classes),
               "phase 1 + 2": summarise(runs2, labels, hours, classes), "minutes": (time.time() - t0) / 60}
        (RESULTS / f"13_finetune_bs{batch}.json").write_text(json.dumps(out, indent=2))
        for phase in ("phase 1 only", "phase 1 + 2"):
            r = out[phase]
            table.append((batch, phase, r["accuracy_over_runs"]["median"], r["accuracy_over_runs"]["mean"],
                          r["plant level, stressed vs control"]["accuracy"], r["calibration_accuracy_over_runs"]["median"]))
    print("\nbatch  phase          test (median / mean)  healthy vs stressed  calibration")
    for b, ph, med, mn, hs, cal in table:
        print(f"{b:5d}  {ph:13s}  {med:.2f} / {mn:.2f}           {hs:.2f}                 {cal:.2f}")


if __name__ == "__main__":
    main()
