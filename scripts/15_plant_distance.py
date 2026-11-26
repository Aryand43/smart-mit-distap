"""How far each plant sits from the others, per stress condition (Dr Praveen, 26 Nov). For checking only,
not for the paper.

Every image becomes one point: the mean of its frozen ResNet-50 patch features (patch 64, stride 8, at
most 600 patches, all bands, white-light normalised with all three plants), standardised, then PCA.
  small faint dots  single images
  circles           one plant under one condition (mean of its images), labelled P1 / P2 / P3
  big squares       the condition centre (mean of the three plants)
Colours: shade black, high light red, white light green, low light blue.

Distances of every plant from its condition centre, in the first 2 principal components (what the plot
shows) and in the first NPC (10):
  Euclidean    straight-line distance
  Mahalanobis  distance in units of the natural spread of the images, from the pooled covariance of the
               images around their own plant-and-condition mean
A plant whose distance is much larger than the other two is a candidate outlier.

    python scripts/15_plant_distance.py

Writes results/15_plant_pca.png, results/15_plant_pca_by_time.png, results/15_plant_distance.csv / .json.
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
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.classic import MAX_PATCHES, PATCH, RESULTS, STRIDE  # noqa: E402
from stress.data import conditions, gains, load, white_reference  # noqa: E402
from stress.patches import features  # noqa: E402

NPC = int(os.environ.get("NPC", 10))
COLOUR = {"SH": "black", "HL": "#D62728", "WL": "#2CA02C", "LL": "#1F77B4"}
NAME = {"SH": "shade", "HL": "high light", "WL": "white light", "LL": "low light"}


def mahalanobis(points, centre, cov):
    inv = np.linalg.pinv(cov)
    d = points - centre
    return np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d))


def spread(Z, light, plant, k):
    """Pooled covariance (first k components) of all images around their own plant-and-condition mean."""
    groups = {(c, p) for c, p in zip(light, plant)}
    resid = np.concatenate([Z[(light == c) & (plant == p), :k] - Z[(light == c) & (plant == p), :k].mean(0) for c, p in groups])
    return np.atleast_2d(np.cov(resid.T, ddof=len(groups)))


def distances(Z, light, plant, sel, classes, k, cov):
    """Rows (condition, plant, euclidean, mahalanobis) in the first k components, images sel only."""
    Zk, out = Z[sel, :k], []
    li, pl = light[sel], plant[sel]
    groups = [(c, p) for c in classes for p in np.unique(pl) if ((li == c) & (pl == p)).any()]
    means = {g: Zk[(li == g[0]) & (pl == g[1])].mean(0) for g in groups}
    for c in classes:
        ps = [p for (cc, p) in groups if cc == c]
        centre = np.mean([means[(c, p)] for p in ps], 0)
        pts = np.array([means[(c, p)] for p in ps])
        for p, e, m in zip(ps, np.linalg.norm(pts - centre, axis=1), mahalanobis(pts, centre, cov)):
            out.append((c, int(p), float(e), float(m)))
    return out


def panel(ax, Z, light, plant, sel, classes, title):
    for c in classes:
        m = sel & (light == c)
        ax.scatter(Z[m, 0], Z[m, 1], s=10, color=COLOUR.get(c, "grey"), alpha=0.2, lw=0)
        pts = []
        for p in np.unique(plant[m]):
            q = Z[m & (plant == p), :2].mean(0)
            pts.append(q)
            ax.scatter(*q, s=70, color=COLOUR.get(c, "grey"), edgecolor="white", lw=0.8, zorder=3)
            ax.annotate(f"P{p}", q, xytext=(4, 4), textcoords="offset points", fontsize=7, color=COLOUR.get(c, "grey"))
        centre = np.mean(pts, 0)
        ax.scatter(*centre, s=220, marker="s", color=COLOUR.get(c, "grey"), edgecolor="white", lw=1.2, zorder=4,
                   label=NAME.get(c, c))
        for q in pts:
            ax.plot([centre[0], q[0]], [centre[1], q[1]], color=COLOUR.get(c, "grey"), lw=0.8, alpha=0.6)
    ax.axhline(0, color="#7B7B79", lw=0.6)
    ax.axvline(0, color="#7B7B79", lw=0.6)
    ax.set_title(title, fontsize=10)


def main():
    samples = load()
    classes = conditions(samples)
    light = np.array([s.light for s in samples])
    plant = np.array([s.plant for s in samples])
    hours = np.array([s.hours for s in samples])
    g = [gi * 0.5 for gi in gains(samples, white_reference(samples, np.arange(len(samples))))]
    feats = features(samples, g, PATCH, STRIDE, MAX_PATCHES, "white-all")
    X = StandardScaler().fit_transform(np.stack([f.mean(0) for f in feats]))
    pca = PCA(n_components=min(NPC, len(X) - 1)).fit(X)
    Z = pca.transform(X)
    ev = pca.explained_variance_ratio_
    lab = lambda ax: (ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)"), ax.set_ylabel(f"PC2 ({ev[1]:.0%})"))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    panel(ax, Z, light, plant, np.ones(len(Z), bool), classes, "Each plant per condition (circles) and the condition centre (squares), all time points")
    lab(ax)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(RESULTS / "15_plant_pca.png", dpi=150)
    plt.close(fig)

    times = sorted(set(hours.tolist()))
    fig, axes = plt.subplots(1, len(times), figsize=(4.2 * len(times), 4.2), sharex=True, sharey=True)
    for ax, h in zip(np.atleast_1d(axes), times):
        panel(ax, Z, light, plant, hours == h, classes, f"{h} h")
        lab(ax)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(RESULTS / "15_plant_pca_by_time.png", dpi=150)
    plt.close(fig)

    rows, out = [], {"explained_variance_ratio": ev.tolist(), "distances": []}
    cov = {k: spread(Z, light, plant, k) for k in (2, Z.shape[1])}
    for when, sel in [("all", np.ones(len(Z), bool))] + [(f"{h}h", hours == h) for h in times]:
        for k in (2, Z.shape[1]):
            for c, p, e, m in distances(Z, light, plant, sel, classes, k, cov[k]):
                rows.append([when, f"{k} PCs", NAME.get(c, c), f"P{p}", f"{e:.2f}", f"{m:.2f}"])
                out["distances"].append({"time": when, "components": k, "condition": c, "plant": p,
                                         "euclidean": e, "mahalanobis": m})
    # whole plant (all conditions and times together) from the centre of the three plants
    for k in (2, Z.shape[1]):
        means = {p: Z[plant == p, :k].mean(0) for p in np.unique(plant)}
        centre = np.mean(list(means.values()), 0)
        for p, q in means.items():
            others = np.mean([v for pp, v in means.items() if pp != p], 0)
            rows.append(["all", f"{k} PCs", "whole plant", f"P{p}", f"{np.linalg.norm(q - centre):.2f}",
                         f"(from the other two: {np.linalg.norm(q - others):.2f})"])
    with open(RESULTS / "15_plant_distance.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Time", "Space", "Condition", "Plant", "Euclidean", "Mahalanobis"])
        w.writerows(rows)
    (RESULTS / "15_plant_distance.json").write_text(json.dumps(out, indent=2))
    print("time  space   condition     plant  euclidean  mahalanobis")
    for r in rows:
        if r[0] == "all":
            print(f"{r[0]:5s} {r[1]:7s} {r[2]:13s} {r[3]:5s}  {r[4]:>9s}  {r[5]}")
    print("-> results/15_plant_pca.png, 15_plant_pca_by_time.png, 15_plant_distance.csv")


if __name__ == "__main__":
    main()
