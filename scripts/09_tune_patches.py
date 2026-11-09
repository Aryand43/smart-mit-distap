"""Bayesian optimisation (Optuna, TPE) of patch size, stride and training settings for the patch model.

Dr Praveen's points (9 Nov): 224-pixel patches are too large; patch size and stride should be tuned
together with the number of epochs (with early stopping), learning rate and its decay, optimiser and
batch size, instead of being fixed by hand.

Search space:
  * patch size 8, 16, 32, 64 or 224 pixels (smaller patches are enlarged to 224 for ResNet-50)
  * stride 2 or 4 pixels
  * up to 10 to 120 epochs, stopped early when the validation loss stops improving
  * learning rate 1e-4 to 1e-2, exponential decay per epoch 0.90 to 1.00
  * optimiser Adam, AdamW or SGD with momentum; weight decay 1e-4 to 1e-1; batch size 64 to 512

Nested test, so the test plant never influences the choice:
  outer loop: each plant in turn is the test plant;
  inner loop: of the two training plants, one trains and the other validates, then they swap.
The best settings are refit on both training plants for the median epoch at which the inner runs
stopped, trained with 5 seeds, and scored once on the test plant (plant vote, metrics per time point).

To keep the cost bounded, each image contributes at most MAX_PATCHES patches, evenly spaced over all
positions on the stride grid. White-light normalised as in 08. Features are cached per setting.
Run on the GPU: TRIALS=40 MAX_PATCHES=600 python scripts/09_tune_patches.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.cnn import CACHE, MIN_PLANT, _MEAN, _STD, _pooled, backbone, device  # noqa: E402
from stress.data import CONTROL, conditions, gains, load, white_reference  # noqa: E402
from stress.evaluate import accuracy_by_hour, classification_metrics, leave_one_plant_out, spread  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
PATCHES, STRIDES = [8, 16, 32, 64, 224], [2, 4]
TRIALS = int(os.environ.get("TRIALS", 40))
MAX_PATCHES = int(os.environ.get("MAX_PATCHES", 600))
SEEDS, PATIENCE, CNN_WHITE_LEVEL, BATCH = 5, 10, 0.5, 128
optuna.logging.set_verbosity(optuna.logging.WARNING)


def positions(mask: np.ndarray, size: int, stride: int) -> np.ndarray:
    """(row, col) of every size x size window with at least MIN_PLANT plant pixels, at most MAX_PATCHES."""
    ii = np.pad(mask.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    R, C = np.meshgrid(np.arange(0, mask.shape[0] - size + 1, stride),
                       np.arange(0, mask.shape[1] - size + 1, stride), indexing="ij")
    frac = (ii[R + size, C + size] - ii[R, C + size] - ii[R + size, C] + ii[R, C]) / size ** 2
    pos = np.stack([R[frac >= MIN_PLANT], C[frac >= MIN_PLANT]], 1)
    if len(pos) > MAX_PATCHES:
        pos = pos[np.linspace(0, len(pos) - 1, MAX_PATCHES).round().astype(int)]
    return pos


_NET = None


@torch.no_grad()
def features(samples, g, size: int, stride: int, tag: str) -> list[np.ndarray]:
    """Layer-4 ResNet-50 features (n_patches x 2048) per image, cached."""
    global _NET
    path = CACHE / f"tune_{samples[0].crop}_p{size}_s{stride}_n{MAX_PATCHES}_{tag}.npz"
    if not path.exists():
        _NET = _NET or backbone()
        out = {}
        for s, gi in zip(samples, g):
            img = np.clip(s.image() * gi, 0, 1).astype(np.float32)
            pos = positions(s.mask(), size, stride)
            feats = []
            for i in range(0, len(pos), BATCH):
                x = np.stack([img[r:r + size, c:c + size] for r, c in pos[i:i + BATCH]])
                x = torch.from_numpy(np.ascontiguousarray(((x - _MEAN) / _STD).transpose(0, 3, 1, 2))).float()
                x = x.to(device())
                if size != 224:
                    x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
                feats.append(_pooled(_NET, x)["layer4"].cpu().numpy())
            out[s.path.name] = np.concatenate(feats) if feats else np.zeros((0, 2048), np.float32)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)
    data = np.load(path)
    return [data[s.path.name] for s in samples]


def train_head(X, y, p, seed, Xval=None, yval=None, epochs=None, n_classes=4):
    """Softmax head on patch features. With validation data, stops early on validation loss.

    Returns (predict_proba, epochs_used, loss curve [(train, val), ...]).
    """
    torch.manual_seed(seed)
    Xt, yt = torch.from_numpy(X).float(), torch.from_numpy(y).long()
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(),
                              torch.nn.Dropout(0.3), torch.nn.Linear(64, n_classes))
    params = dict(lr=p["lr"], weight_decay=p["weight_decay"])
    opt = {"adam": lambda: torch.optim.Adam(net.parameters(), **params),
           "adamw": lambda: torch.optim.AdamW(net.parameters(), **params),
           "sgd": lambda: torch.optim.SGD(net.parameters(), momentum=0.9, **params)}[p["optimizer"]]()
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=p["lr_decay"])
    loss_fn = torch.nn.functional.cross_entropy
    if Xval is not None:
        Xv, yv = (torch.from_numpy(Xval).float() - mu) / sd, torch.from_numpy(yval).long()
    best, best_ep, best_state, curve = np.inf, 0, None, []
    for ep in range(epochs or p["max_epochs"]):
        net.train()
        total = 0.0
        for b in torch.randperm(len(Xt)).split(p["batch_size"]):
            loss = loss_fn(net((Xt[b] - mu) / sd), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        if Xval is None:
            continue
        net.eval()
        with torch.no_grad():
            val = loss_fn(net(Xv), yv).item()
        curve.append((total / len(Xt), val))
        if val < best:
            best, best_ep, best_state = val, ep, copy.deepcopy(net.state_dict())
        elif ep - best_ep >= PATIENCE:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    @torch.no_grad()
    def predict(Xnew):
        return net((torch.from_numpy(Xnew).float() - mu) / sd).argmax(-1).numpy()

    return predict, (best_ep + 1 if Xval is not None else epochs), curve


def stack(feats, idx, labels):
    return (np.concatenate([feats[i] for i in idx]),
            np.concatenate([[labels[i]] * len(feats[i]) for i in idx]))


def vote(predict, feats, idx, n_classes):
    return [int(np.bincount(predict(feats[i]), minlength=n_classes).argmax()) if len(feats[i]) else 0
            for i in idx]


def main():
    samples = load()
    classes = conditions(samples)
    labels = np.array([classes.index(s.light) for s in samples])
    hours = np.array([s.hours for s in samples])
    plants = np.array([s.plant for s in samples])
    print(f"device {device()}, {TRIALS} trials per test plant, at most {MAX_PATCHES} patches per image", flush=True)

    folds, all_true, all_pred, all_hours = {}, [], [], []
    for fold, tr, te in leave_one_plant_out(plants):
        g = [gi * CNN_WHITE_LEVEL for gi in gains(samples, white_reference(samples, tr))]
        stopped = {}

        def objective(t):
            p = {"patch": t.suggest_categorical("patch", PATCHES),
                 "stride": t.suggest_categorical("stride", STRIDES),
                 "max_epochs": t.suggest_int("max_epochs", 10, 120, step=10),
                 "lr": t.suggest_float("lr", 1e-4, 1e-2, log=True),
                 "lr_decay": t.suggest_float("lr_decay", 0.90, 1.0),
                 "optimizer": t.suggest_categorical("optimizer", ["adam", "adamw", "sgd"]),
                 "weight_decay": t.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
                 "batch_size": t.suggest_categorical("batch_size", [64, 128, 256, 512])}
            feats = features(samples, g, p["patch"], p["stride"], f"white-{fold}")
            accs, eps = [], []
            for val_plant in np.unique(plants[tr]):
                itr, ival = tr[plants[tr] != val_plant], tr[plants[tr] == val_plant]
                X, y = stack(feats, itr, labels)
                Xv, yv = stack(feats, ival, labels)
                predict, ep, _ = train_head(X, y, p, 0, Xv, yv, n_classes=len(classes))
                accs.append(np.mean(np.array(vote(predict, feats, ival, len(classes))) == labels[ival]))
                eps.append(ep)
            stopped[t.number] = int(np.median(eps))
            return float(np.mean(accs))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(objective, n_trials=TRIALS)
        best = dict(study.best_params)
        best["epochs_used"] = stopped[study.best_trial.number]

        feats = features(samples, g, best["patch"], best["stride"], f"white-{fold}")
        X, y = stack(feats, tr, labels)
        preds = []
        for seed in range(SEEDS):
            predict, _, _ = train_head(X, y, best, seed, epochs=best["epochs_used"], n_classes=len(classes))
            preds.append(vote(predict, feats, te, len(classes)))
        # loss curve of the best settings on one inner split, for the training-curve plot
        itr, ival = tr[plants[tr] == np.unique(plants[tr])[0]], tr[plants[tr] == np.unique(plants[tr])[1]]
        _, _, curve = train_head(*stack(feats, itr, labels), best, 0, *stack(feats, ival, labels),
                                 n_classes=len(classes))

        accs = [float(np.mean(np.array(pr) == labels[te])) for pr in preds]
        folds[fold] = {"best_params": best, "inner_accuracy": study.best_value,
                       "test_accuracy_over_seeds": spread(accs), "loss_curve_train_val": curve,
                       "trials": [{"params": t.params, "value": t.value} for t in study.trials]}
        for pr in preds:
            all_true += [classes[i] for i in labels[te]]
            all_pred += [classes[i] for i in pr]
            all_hours += hours[te].tolist()
        print(f"{fold}: best {best}, inner {study.best_value:.2f}, test median {np.median(accs):.2f}", flush=True)

    all_true, all_pred, all_hours = map(np.array, (all_true, all_pred, all_hours))
    binary = lambda ys: np.where(ys == CONTROL, "control", "stressed")
    by_time = {}
    for h in sorted(set(all_hours)):
        sel = all_hours == h
        by_time[f"{h}h"] = classification_metrics(all_true[sel], all_pred[sel], classes)
    out = {
        "search": {"patch": PATCHES, "stride": STRIDES, "trials_per_fold": TRIALS, "max_patches": MAX_PATCHES},
        "per_test_plant": folds,
        "plant level (patch vote)": {**classification_metrics(all_true, all_pred, classes),
                                     "accuracy_by_hour": accuracy_by_hour(all_true, all_pred, all_hours)},
        "plant level, stressed vs control": classification_metrics(binary(all_true), binary(all_pred),
                                                                    ["control", "stressed"]),
        "by_time_point": by_time,
    }
    (RESULTS / "09_tune_patches.json").write_text(json.dumps(out, indent=2))
    p = out["plant level (patch vote)"]
    print(f"tuned: accuracy {p['accuracy']:.2f}, F1 {p['f1_macro']:.2f}, "
          f"stressed vs control {out['plant level, stressed vs control']['accuracy']:.2f}")
    print(p["confusion"])


if __name__ == "__main__":
    main()
