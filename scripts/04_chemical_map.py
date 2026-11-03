"""Chemical maps: where on the plant each stress shows up, and how it progresses over time.

Pixel-level PLS-DA on the three white-light-normalised bands. For each test plant in turn (the other
two plants train the model), every pixel gets a score per condition, and the maps show:
  * stress score = strongest stress-class score minus the white-light score (above 0 = less like control)
  * the share of the plant's pixels voting for each condition (the plant's label is the majority)
One figure per test plant: rows are conditions, columns are 4 h, 1 day, 2 days, 3 days.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.data import CONTROL, conditions, gains, load, white_reference  # noqa: E402
from stress.evaluate import leave_one_plant_out  # noqa: E402
from stress.models import PLSDA  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
PIXELS_PER_IMAGE = 5000
TIME_LABEL = {4: "4 h", 24: "1 day", 48: "2 days", 72: "3 days"}


def main():
    rng = np.random.default_rng(0)
    samples = load()
    classes = conditions(samples)
    stress_cols = [i for i, c in enumerate(classes) if c != CONTROL]
    ctrl = classes.index(CONTROL)
    light = np.array([s.light for s in samples])
    plants = np.array([s.plant for s in samples])
    times = sorted({s.hours for s in samples})
    out = {}

    for fold, tr, te in leave_one_plant_out(plants):
        g = gains(samples, white_reference(samples, tr))
        X, y = [], []
        for i in tr:
            px = samples[i].pixels() * g[i]
            X.append(px[rng.choice(len(px), min(PIXELS_PER_IMAGE, len(px)), replace=False)])
            y += [light[i]] * len(X[-1])
        model = PLSDA(classes, 2).fit(np.concatenate(X), np.array(y))

        fig, axes = plt.subplots(len(classes), len(times), figsize=(3.2 * len(times), 2.6 * len(classes)),
                                 constrained_layout=True, squeeze=False)
        out[fold] = {}
        for i in te:
            s = samples[i]
            mask = s.mask()
            scores = model.scores(s.pixels() * g[i])
            pred = np.array(classes)[scores.argmax(1)]
            share = {c: float((pred == c).mean()) for c in classes}
            stress = np.full(mask.shape, np.nan)
            stress[mask] = scores[:, stress_cols].max(1) - scores[:, ctrl]
            ax = axes[classes.index(s.light), times.index(s.hours)]
            im = ax.imshow(stress, cmap="magma", vmin=-1, vmax=1)
            label = max(share, key=share.get)
            ax.set_title(f"{s.light}, {TIME_LABEL[s.hours]}: voted {label} ({share[label]:.0%})", fontsize=9,
                         color="black" if label == s.light else "#9D192B")
            out[fold][s.path.name] = {"true": s.light, "voted": label, "pixel_share": share}
        for ax in axes.ravel():
            ax.axis("off")
        fig.colorbar(im, ax=axes, fraction=0.02, label="stress score (above 0 = less like white light)")
        fig.suptitle(f"Test plant {fold}, model trained on the other two plants (white-light normalised)")
        fig.savefig(RESULTS / f"04_stress_progression_{fold}.png", dpi=100)
        plt.close(fig)

    beta = model.pls[-1].coef_
    out["beta_coefficients_last_fold (rows = classes, cols = band 1-3)"] = np.asarray(beta).round(3).tolist()
    (RESULTS / "04_chemical_map.json").write_text(json.dumps(out, indent=2))
    for fold in [k for k in out if k.startswith("P")]:
        right = sum(v["true"] == v["voted"] for v in out[fold].values())
        print(f"{fold}: {right}/{len(out[fold])} plants voted correctly")


if __name__ == "__main__":
    main()
