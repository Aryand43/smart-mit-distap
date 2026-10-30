"""First chemical map: pixel-level PLS-DA on the three bands, applied to the 4-hour images.

Pixels from plants 1 and 2 (all time points) train a PLS-DA model that separates high, low and white
light from the three band values of each pixel. The model's beta coefficients are then applied to
every pixel of plant 3, which the model never saw, to show *where* on the plant the light-stress
signature appears at 4 hours.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.data import LIGHTS, load  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
TEST_PLANT, PIXELS_PER_IMAGE = 3, 5000
CLASS_COLOURS = ["#C8553D", "#2E86AB", "#BDBDBD"]   # HL, LL, WL


def main():
    rng = np.random.default_rng(0)
    samples = load()
    X, y = [], []
    for s in samples:
        if s.plant == TEST_PLANT:
            continue
        px = s.image()[s.mask()]
        px = px[rng.choice(len(px), min(PIXELS_PER_IMAGE, len(px)), replace=False)]
        X.append(px)
        y += [s.light] * len(px)
    X, y = np.concatenate(X), np.array(y)
    scaler = StandardScaler().fit(X)
    pls = PLSRegression(n_components=2).fit(scaler.transform(X), (y[:, None] == np.array(LIGHTS)[None]).astype(float))

    test = sorted((s for s in samples if s.plant == TEST_PLANT and s.hours == 4), key=lambda s: LIGHTS.index(s.light))
    fig, axes = plt.subplots(len(test), 3, figsize=(12, 3.2 * len(test)), constrained_layout=True)
    pixel_acc = {}
    for row, s in zip(axes, test):
        img, mask = s.image(), s.mask()
        scores = pls.predict(scaler.transform(img[mask]))
        cls = scores.argmax(1)
        pixel_acc[s.light] = float((cls == LIGHTS.index(s.light)).mean())

        class_map = np.full(mask.shape, np.nan)
        class_map[mask] = cls
        stress = np.full(mask.shape, np.nan)       # stress score: strongest stress class minus white light
        stress[mask] = scores[:, :2].max(1) - scores[:, 2]

        row[0].imshow(np.clip(img / np.percentile(img[mask], 99), 0, 1))
        row[0].set_title(f"{s.light}, 4 h, plant {s.plant}: three-band image")
        row[1].imshow(class_map, cmap=ListedColormap(CLASS_COLOURS), vmin=0, vmax=2, interpolation="nearest")
        row[1].set_title(f"Pixel class (HL / LL / WL), {pixel_acc[s.light]:.0%} correct")
        im = row[2].imshow(stress, cmap="magma", vmin=-1, vmax=1)
        row[2].set_title("Stress score (higher = less like white light)")
        fig.colorbar(im, ax=row[2], fraction=0.035)
        for ax in row:
            ax.axis("off")

    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "04_chemical_map_4h.png", dpi=110)
    beta = pls.coef_.T if pls.coef_.shape[0] == 3 else pls.coef_
    out = {
        "trained_on_plants": [1, 2],
        "mapped_plant": TEST_PLANT,
        "pixel_accuracy_4h": pixel_acc,
        "beta_coefficients (rows = band 1-3, cols = HL, LL, WL)": np.asarray(beta).reshape(3, 3).round(3).tolist(),
    }
    (RESULTS / "04_chemical_map.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
