"""First version of the two-output model: light condition (classification) + hours of stress (regression).

A small shared network sits on top of the frozen ResNet-50 tile features and splits into two heads:
  * light head: HL / LL / WL (cross-entropy)
  * hours head: hours since stress onset as a number (mean squared error on hours / 72)
It is trained on tiles (with flips and rotations) and an image's prediction is the average over its
tiles. Evaluated leave-one-plant-out, averaged over several random seeds.

With 24 training images per fold this is a pipeline check, not a final model: the CNN-LSTM and
CNN-transformer backbones plug in here once more data arrives.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.cnn import tile_features  # noqa: E402
from stress.data import LIGHTS, load  # noqa: E402
from stress.evaluate import hours_metrics, leave_one_plant_out, summarise  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
SEEDS, EPOCHS, HOUR_SCALE, HOUR_WEIGHT = 5, 60, 72.0, 1.0


class TwoOutput(nn.Module):
    def __init__(self, d_in=2048, d_hidden=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Dropout(0.5), nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(0.3))
        self.light = nn.Linear(d_hidden, len(LIGHTS))
        self.hours = nn.Linear(d_hidden, 1)

    def forward(self, x):
        z = self.shared(x)
        return self.light(z), self.hours(z).squeeze(-1)


def train_fold(train_feats, test_feats, light_idx, hours, train, test, seed):
    torch.manual_seed(seed)
    X = torch.from_numpy(np.concatenate([train_feats[i] for i in train])).float()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    yl = torch.from_numpy(np.concatenate([[light_idx[i]] * len(train_feats[i]) for i in train]))
    yh = torch.from_numpy(np.concatenate([[hours[i] / HOUR_SCALE] * len(train_feats[i]) for i in train])).float()

    model = TwoOutput()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(EPOCHS):
        model.train()
        for b in torch.randperm(len(X)).split(256):
            pl, ph = model((X[b] - mu) / sd)
            loss = nn.functional.cross_entropy(pl, yl[b]) + HOUR_WEIGHT * nn.functional.mse_loss(ph, yh[b])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    probs, hrs = [], []
    with torch.no_grad():
        for i in test:
            pl, ph = model((torch.from_numpy(test_feats[i]).float() - mu) / sd)
            probs.append(pl.softmax(-1).mean(0).numpy())
            hrs.append(float(ph.mean()) * HOUR_SCALE)
    return np.array(probs), np.array(hrs)


def main():
    samples = load()
    # Training tiles include 8 orientations each; held-out images are scored on their original tiles.
    train_feats = tile_features(samples, augment=True)
    test_feats = tile_features(samples, augment=False)
    light = np.array([s.light for s in samples])
    light_idx = np.array([LIGHTS.index(l) for l in light])
    hours = np.array([s.hours for s in samples])
    plants = np.array([s.plant for s in samples])

    probs = np.zeros((SEEDS, len(samples), len(LIGHTS)))
    hrs = np.zeros((SEEDS, len(samples)))
    for seed in range(SEEDS):
        for _, tr, te in leave_one_plant_out(plants):
            probs[seed, te], hrs[seed, te] = train_fold(train_feats, test_feats, light_idx, hours, tr, te, seed)

    per_seed = [float((np.array(LIGHTS)[probs[s].argmax(1)] == light).mean()) for s in range(SEEDS)]
    pred_light = np.array(LIGHTS)[probs.mean(0).argmax(1)]
    out = {
        "light/LOPO (seed ensemble)": summarise(light, pred_light, hours, None),
        "light_accuracy_per_seed": per_seed,
        "hours/LOPO (seed ensemble)": hours_metrics(hours, hrs.mean(0)),
        "per_image": [
            {"file": s.path.name, "light": s.light, "pred_light": pl, "hours": int(s.hours), "pred_hours": round(float(h), 1)}
            for s, pl, h in zip(samples, pred_light, hrs.mean(0))
        ],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "03_two_output_model.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        if k == "per_image":
            continue
        if isinstance(v, dict):
            print(k, {m: (round(x, 3) if isinstance(x, float) else x) for m, x in v.items() if m != "light_confusion"})
        else:
            print(k, [round(x, 3) for x in v])


if __name__ == "__main__":
    main()
