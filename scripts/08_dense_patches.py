"""Dense patches on the test plant: stride 4, so every small region of the plant is covered.

Dr Praveen's point (2 Nov): with a large stride a patch spans a wide area and small stressed regions
get diluted; a very small stride lets overlapping patches cover every region. Every patch is
classified, then all patches of a plant are combined into one decision for that exact plant, and the
classification metrics are computed on plants.

To keep it to about an hour on the laptop:
  * training uses the stride-56 patches of the two training plants, with flips and rotations
    (unchanged from 05), so only the test plant is cut at stride 4;
  * each test patch is classified on the fly and only its predictions are stored
    (results/08_dense_patch_predictions.npz), so the analysis can be redone without the network.
White-light normalised throughout, same three rounds (test plant rotated) and 5 seeds as before.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.cnn import TILE, _MEAN, _STD, _pooled, backbone, device, tile_features, tile_positions, train_softmax  # noqa: E402
from stress.data import CONTROL, conditions, gains, load, white_reference  # noqa: E402
from stress.evaluate import accuracy_by_hour, classification_metrics, leave_one_plant_out, spread  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
TEST_STRIDE = int(os.environ.get("TEST_STRIDE", 4))
SEEDS, CNN_WHITE_LEVEL, BATCH = 5, 0.5, 128


@torch.no_grad()
def dense_features(net, img: np.ndarray, pos: np.ndarray):
    """Yield layer-4 features for the tiles at pos, one batch at a time."""
    for i in range(0, len(pos), BATCH):
        batch = np.stack([img[r:r + TILE, c:c + TILE] for r, c in pos[i:i + BATCH]])
        x = torch.from_numpy(np.ascontiguousarray(((batch - _MEAN) / _STD).transpose(0, 3, 1, 2))).float()
        yield _pooled(net, x.to(device()))["layer4"].cpu().numpy()


def main():
    samples = load()
    classes = conditions(samples)
    light = np.array([s.light for s in samples])
    hours = np.array([s.hours for s in samples])
    plants = np.array([s.plant for s in samples])
    net = backbone()

    stored = {}
    plant_true, plant_pred, plant_hours, run_acc = [], {k: [] for k in range(SEEDS)}, [], {k: [] for k in range(SEEDS)}
    patch_true, patch_pred = [], []
    for fold, tr, te in leave_one_plant_out(plants):
        g = [gi * CNN_WHITE_LEVEL for gi in gains(samples, white_reference(samples, tr))]
        aug = tile_features(samples, True, ("layer4",), g, f"white-{fold}")
        Xa = np.concatenate([aug[i] for i in tr])
        ya = np.concatenate([[classes.index(light[i])] * len(aug[i]) for i in tr])
        heads = [train_softmax(Xa, ya, len(classes), seed) for seed in range(SEEDS)]

        for i in te:
            s = samples[i]
            img = np.clip(s.image() * g[i], 0, 1).astype(np.float32)
            pos = tile_positions(s.mask(), TEST_STRIDE)
            preds = np.concatenate([np.stack([h(f).argmax(1) for h in heads], 1)
                                    for f in dense_features(net, img, pos)])        # n_tiles x seeds
            stored[s.path.name] = preds.astype(np.int8)
            stored[s.path.name + "|pos"] = pos.astype(np.int16)
            plant_true.append(light[i])
            plant_hours.append(hours[i])
            for k in range(SEEDS):
                votes = np.bincount(preds[:, k], minlength=len(classes))
                plant_pred[k].append(classes[int(votes.argmax())])
            patch_true += [light[i]] * preds.size
            patch_pred += np.array(classes)[preds.ravel()].tolist()
            print(f"  {fold} {s.path.name}: {len(pos)} patches", flush=True)
        for k in range(SEEDS):
            n = len(te)
            run_acc[k].append(float((np.array(plant_pred[k][-n:]) == light[te]).mean()))

    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / "08_dense_patch_predictions.npz", classes=np.array(classes), **stored)

    all_true = plant_true * SEEDS
    all_pred = sum((plant_pred[k] for k in range(SEEDS)), [])
    all_hours = plant_hours * SEEDS
    binary = lambda ys: ["stressed" if y != CONTROL else "control" for y in ys]
    out = {
        "test_stride": TEST_STRIDE,
        "patches_per_plant": spread([len(v) for k, v in stored.items() if not k.endswith("|pos")]),
        "plant level (patch vote)": {
            "accuracy_over_runs": spread(sum(run_acc.values(), [])),
            "accuracy_by_hour_pooled": accuracy_by_hour(all_true, all_pred, all_hours),
            **classification_metrics(all_true, all_pred, classes)},
        "plant level, stressed vs control": classification_metrics(binary(all_true), binary(all_pred),
                                                                    ["control", "stressed"]),
        "patch level": classification_metrics(patch_true, patch_pred, classes),
    }
    (RESULTS / "08_dense_patches.json").write_text(json.dumps(out, indent=2))
    p, b, t = out["plant level (patch vote)"], out["plant level, stressed vs control"], out["patch level"]
    print(f"plant: median {p['accuracy_over_runs']['median']:.0%}, F1 {p['f1_macro']:.2f}, 4 h {p['accuracy_by_hour_pooled'][4]:.0%}")
    print(f"stressed vs control: {b['accuracy']:.0%}, F1 {b['f1_macro']:.2f}")
    print(f"patch: {t['accuracy']:.0%}, F1 {t['f1_macro']:.2f}")
    print(p["confusion"])


if __name__ == "__main__":
    main()
