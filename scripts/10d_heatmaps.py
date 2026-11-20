"""Heat maps (Grad-CAM): which parts of each test plant drive the ResNet-50 patch model's decision.

Trains the locked setting from 10 on the two training plants (seed 0), then for every image of the
test plant computes Grad-CAM on ResNet-50's last block for every patch: how much each location pushed
the patch towards the class the plant was finally voted. The patch maps are pasted back into the
image (overlaps averaged), giving one heat map per plant image.

It also measures where the attention falls: the share of the heat map on leaf interior, on the leaf
edge (a band about 4 pixels wide along the plant outline) and on background, next to the share of
pixels each region covers. Attention far above its area share on edge or background means the model
leans on outline or background rather than on the leaf itself.

    TEST_PLANT=P1 python scripts/10d_heatmaps.py          # uses results/10_final_resnet50.json
    RUN=resnet50_aug2n0.02 AUG=2 NOISE=0.02 TEST_PLANT=P3 python scripts/10d_heatmaps.py

ResNet-50 only. Set the same BANDS / AUG / NOISE / PLANT_FRAC as the run. Writes
results/10d_heatmaps_<run>_<plant>.png and results/10d_attention_<run>_<plant>.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.cnn import _MEAN, _STD, backbone, device  # noqa: E402
from stress.data import conditions, gains, load, white_reference  # noqa: E402
from stress.evaluate import leave_one_plant_out  # noqa: E402
from stress.patches import (BATCH, patch_sets, positions, run_name, select_bands, train_head,  # noqa: E402
                            train_set, vote)

RESULTS = Path(__file__).resolve().parents[1] / "results"
EDGE = 4                                   # width of the leaf-edge band, pixels


def cam_map(net, head, mu, sd, img, pos, size, cls, dev):
    """Image-sized Grad-CAM map for class cls, averaged over overlapping patches."""
    acc, cnt = np.zeros(img.shape[:2], np.float32), np.zeros(img.shape[:2], np.float32)
    for i in range(0, len(pos), BATCH):
        p = pos[i:i + BATCH]
        x = np.stack([img[r:r + size, c:c + size] for r, c in p])
        x = torch.from_numpy(np.ascontiguousarray(((x - _MEAN) / _STD).transpose(0, 3, 1, 2))).float().to(dev)
        if size != 224:
            x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
        with torch.no_grad():
            a = net.layer3(net.layer2(net.layer1(net.maxpool(net.relu(net.bn1(net.conv1(x)))))))
        a = net.layer4(a).detach().requires_grad_(True)            # (n, 2048, 7, 7)
        score = head((a.mean((2, 3)) - mu) / sd)[:, cls].sum()
        g, = torch.autograd.grad(score, a)
        cam = torch.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)
        for (r, c), m in zip(p, cam[:, 0].detach().cpu().numpy()):
            acc[r:r + size, c:c + size] += m
            cnt[r:r + size, c:c + size] += 1
    return np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0), cnt > 0


def main():
    run = os.environ.get("RUN", run_name("resnet50"))
    test_plant = os.environ.get("TEST_PLANT", "P1")
    final = json.loads((RESULTS / f"10_final_{run}.json").read_text())
    setting, size = final["selected_setting"], final["selected_setting"]["patch"]
    samples = load()
    classes = conditions(samples)
    labels = np.array([classes.index(s.light) for s in samples])
    plants = np.array([s.plant for s in samples])
    fold, tr, te = next(f for f in leave_one_plant_out(plants) if f[0] == test_plant)

    g = [gi * 0.5 for gi in gains(samples, white_reference(samples, tr))]
    feats, aug = patch_sets(samples, g, size, setting["stride"], final["max_patches"], f"white-{fold}")
    predict, _, _ = train_head(*train_set(feats, aug, tr, labels), setting, 0, epochs=final["epochs"],
                               n_classes=len(classes))
    voted = vote(predict, feats, te, len(classes))

    dev = device() if device().type == "cuda" else torch.device("cpu")   # Apple MPS cannot run this backward pass
    net = backbone().to(dev)
    head = predict.net.to(dev).eval()
    mu, sd = predict.mu.to(dev), predict.sd.to(dev)

    hours = sorted({samples[i].hours for i in te})
    fig, axes = plt.subplots(len(classes), len(hours), figsize=(3.2 * len(hours), 2.3 * len(classes)), squeeze=False)
    shares = {"interior": [], "edge": [], "background": []}
    areas = {"interior": [], "edge": [], "background": []}
    for i, v in zip(te, voted):
        s = samples[i]
        img = select_bands(np.clip(s.image() * g[i], 0, 1).astype(np.float32))
        mask = s.mask()
        pos = positions(mask, size, setting["stride"], final["max_patches"])
        cam, covered = cam_map(net, head, mu, sd, img, pos, size, v, dev)
        interior = ndimage.binary_erosion(mask, iterations=EDGE)
        regions = {"interior": interior, "edge": mask & ~interior, "background": ~mask}
        total = cam[covered].sum() + 1e-9
        for k, reg in regions.items():
            shares[k].append(float(cam[covered & reg].sum() / total))
            areas[k].append(float((covered & reg).sum() / covered.sum()))
        ax = axes[classes.index(s.light)][hours.index(s.hours)]
        base = s.image()
        base = np.clip(base / (np.percentile(base[mask], 99) + 1e-9), 0, 1)     # brightened for display
        ax.imshow(base.mean(2), cmap="gray", vmin=0, vmax=1)
        heat = cam / (np.percentile(cam[covered], 99) + 1e-9) if covered.any() else cam
        ax.imshow(np.ma.masked_where(~covered | (heat < 0.15), np.clip(heat, 0, 1)), cmap="turbo",
                  vmin=0, vmax=1, alpha=0.6)
        ax.set_title(f"{s.light}, {s.hours} h -> {classes[v]}", fontsize=9,
                     color="#1F1F1E" if classes[v] == s.light else "#9D192B")
        ax.axis("off")
    fig.suptitle(f"{run}, test plant {test_plant}: Grad-CAM for the voted class (red title = wrong)")
    fig.tight_layout()
    out_png = RESULTS / f"10d_heatmaps_{run}_{test_plant}.png"
    fig.savefig(out_png, dpi=130)

    summary = {k: {"attention_share": float(np.mean(shares[k])), "area_share": float(np.mean(areas[k]))}
               for k in shares}
    (RESULTS / f"10d_attention_{run}_{test_plant}.json").write_text(json.dumps(summary, indent=2))
    print(f"-> {out_png}\nwhere the attention falls (share of heat map vs share of pixels in the patches):")
    for k, v in summary.items():
        print(f"  {k:10s} attention {v['attention_share']:.2f}   area {v['area_share']:.2f}")


if __name__ == "__main__":
    main()
