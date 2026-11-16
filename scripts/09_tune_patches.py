"""Bayesian optimisation (Optuna, TPE) of patch size, stride and training settings for the patch model.

Dr Praveen's points (9 Nov): 224-pixel patches are too large; patch size and stride should be tuned
together with the number of epochs (with early stopping), learning rate and its decay, optimiser and
batch size, instead of being fixed by hand.

16 Nov: patch sizes are powers of two, and each backbone (ResNet-50 or the vision transformer) gets its
own search, chosen with BACKBONE=resnet50 (default) or BACKBONE=vit_b_16.

Search space:
  * patch size 8, 16, 32, 64, 128 or 256 pixels (every patch is resized to the network's 224 x 224 input)
  * stride 2, 4 or 8 pixels
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
Run on the GPU, once per backbone:
    TRIALS=40 MAX_PATCHES=600 python scripts/09_tune_patches.py
    BACKBONE=vit_b_16 TRIALS=40 MAX_PATCHES=600 python scripts/09_tune_patches.py
Writes results/09_tune_<backbone>.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.cnn import device  # noqa: E402
from stress.data import CONTROL, conditions, gains, load, white_reference  # noqa: E402
from stress.evaluate import accuracy_by_hour, classification_metrics, leave_one_plant_out, spread  # noqa: E402
from stress.patches import BACKBONES, features, inner_splits, stack, train_head, vote  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
PATCHES, STRIDES = [8, 16, 32, 64, 128, 256], [2, 4, 8]
BACKBONE = os.environ.get("BACKBONE", "resnet50")
TRIALS = int(os.environ.get("TRIALS", 40))
MAX_PATCHES = int(os.environ.get("MAX_PATCHES", 600))
SEEDS, CNN_WHITE_LEVEL = 5, 0.5
optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    if BACKBONE not in BACKBONES:
        raise SystemExit(f"BACKBONE must be one of {BACKBONES}")
    samples = load()
    classes = conditions(samples)
    labels = np.array([classes.index(s.light) for s in samples])
    hours = np.array([s.hours for s in samples])
    plants = np.array([s.plant for s in samples])
    print(f"{BACKBONE} on {device()}, {TRIALS} trials per test plant, at most {MAX_PATCHES} patches per image", flush=True)

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
            feats = features(samples, g, p["patch"], p["stride"], MAX_PATCHES, f"white-{fold}", BACKBONE)
            accs, eps = [], []
            for itr, ival in inner_splits(plants, tr):
                X, y = stack(feats, itr, labels)
                Xv, yv = stack(feats, ival, labels)
                predict, ep, _ = train_head(X, y, p, 0, Xv, yv, n_classes=len(classes))
                accs.append(np.mean(np.array(vote(predict, feats, ival, len(classes))) == labels[ival]))
                eps.append(ep)
            stopped[t.number] = int(np.median(eps))
            t.set_user_attr("epochs_used", stopped[t.number])
            return float(np.mean(accs))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(objective, n_trials=TRIALS)
        best = dict(study.best_params)
        best["epochs_used"] = stopped[study.best_trial.number]

        feats = features(samples, g, best["patch"], best["stride"], MAX_PATCHES, f"white-{fold}", BACKBONE)
        X, y = stack(feats, tr, labels)
        preds = []
        for seed in range(SEEDS):
            predict, _, _ = train_head(X, y, best, seed, epochs=best["epochs_used"], n_classes=len(classes))
            preds.append(vote(predict, feats, te, len(classes)))
        # loss curve of the best settings on one inner split, for the training-curve plot
        itr, ival = inner_splits(plants, tr)[0]
        _, _, curve = train_head(*stack(feats, itr, labels), best, 0, *stack(feats, ival, labels),
                                 n_classes=len(classes))

        accs = [float(np.mean(np.array(pr) == labels[te])) for pr in preds]
        folds[fold] = {"best_params": best, "inner_accuracy": study.best_value,
                       "test_accuracy_over_seeds": spread(accs), "loss_curve_train_val": curve,
                       "trials": [{"params": t.params, "value": t.value, **t.user_attrs} for t in study.trials]}
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
        "backbone": BACKBONE,
        "search": {"patch": PATCHES, "stride": STRIDES, "trials_per_fold": TRIALS, "max_patches": MAX_PATCHES},
        "per_test_plant": folds,
        "plant level (patch vote)": {**classification_metrics(all_true, all_pred, classes),
                                     "accuracy_by_hour": accuracy_by_hour(all_true, all_pred, all_hours)},
        "plant level, stressed vs control": classification_metrics(binary(all_true), binary(all_pred),
                                                                    ["control", "stressed"]),
        "by_time_point": by_time,
    }
    (RESULTS / f"09_tune_{BACKBONE}.json").write_text(json.dumps(out, indent=2))
    p = out["plant level (patch vote)"]
    print(f"tuned: accuracy {p['accuracy']:.2f}, F1 {p['f1_macro']:.2f}, "
          f"stressed vs control {out['plant level, stressed vs control']['accuracy']:.2f}")
    print(p["confusion"])


if __name__ == "__main__":
    main()
