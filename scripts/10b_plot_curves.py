"""Training and validation curves of the best setting for each test plant, from a 09 run.

For each test plant: training loss and validation loss (left axis) and validation accuracy (right
axis) per epoch. The minimum of the validation loss is marked and labelled, and the part after it,
where validation loss rises while training loss keeps falling (overfitting), is drawn thicker and shaded; on one inner split (train on one training plant, validate on the other). The dashed
line is the epoch kept by early stopping in that split.

    python scripts/10b_plot_curves.py                  # results/09_tune_resnet50.json
    RUN=resnet50_aug2n0.02 python scripts/10b_plot_curves.py

Writes results/10b_curves_<run>.png and, once 10 has run, results/10b_mean_curve_<run>.png: the
validation curve averaged over all 6 inner splits, with the one epoch count used for every test plant.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
RUN = os.environ.get("RUN", "resnet50")


def validation_loss(ax, ep, val):
    """Validation loss with its minimum labelled and the rising part after it (overfitting) highlighted."""
    ep = list(ep)
    i = min(range(len(val)), key=val.__getitem__)
    ax.plot(ep[:i + 1], val[:i + 1], color="#9D192B", label="validation loss")
    if i < len(val) - 1:
        ax.plot(ep[i:], val[i:], color="#9D192B", linewidth=3, label="validation loss rising")
        ax.axvspan(ep[i], ep[-1], color="#9D192B", alpha=0.06, linewidth=0)
    ax.plot(ep[i], val[i], "o", color="#9D192B", markersize=7, zorder=5)
    left = i < 0.2 * len(val)                          # minimum near the start: put the label to its right
    ax.annotate(f"minimum: epoch {ep[i]}\nloss {val[i]:.3f}", (ep[i], val[i]),
                xytext=(14 if left else 0, 26), textcoords="offset points", ha="left" if left else "center",
                fontsize=9, color="#9D192B", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#9D192B", lw=0.8),
                arrowprops=dict(arrowstyle="-", color="#9D192B", lw=0.8))
    ax.xaxis.get_major_locator().set_params(integer=True)


def main():
    tuning = json.loads((RESULTS / f"09_tune_{RUN}.json").read_text())
    folds = tuning["per_test_plant"]
    start = tuning.get("curve_from_epoch", 1)          # runs from 20 Nov include epoch 0 (before training)
    fig, axes = plt.subplots(1, len(folds), figsize=(5 * len(folds), 3.8), squeeze=False)
    for ax, (plant, f) in zip(axes[0], folds.items()):
        curve = f["loss_curve_train_val"]
        ep = range(start, start + len(curve))
        ax.plot(ep, [c[0] for c in curve], color="#7B7B79", label="training loss")
        validation_loss(ax, ep, [c[1] for c in curve])
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        train_p, val_p = sorted(set(folds) - {plant})[::-1]   # 09 draws the curve of the first inner split
        ax.set_title(f"test plant {plant}: trained on {train_p}, validated on {val_p}", fontsize=10)
        if len(curve[0]) > 2:                       # runs from 20 Nov onwards also store validation accuracy
            ax2 = ax.twinx()
            ax2.plot(ep, [c[2] for c in curve], color="#2E3F9E", label="validation accuracy")
            ax2.set_ylabel("validation accuracy")
            ax2.set_ylim(0, 1)
            first = max(0, min(int(os.environ.get("MIN_EPOCHS", 5)) - start, len(curve) - 1))
            best = min(range(first, len(curve)), key=lambda i: curve[i][1])
            ax.axvline(start + best, color="#1F1F1E", linestyle="--", linewidth=1, label="epoch kept")
        handles = ax.get_legend_handles_labels()
        if len(curve[0]) > 2:
            h2 = ax2.get_legend_handles_labels()
            handles = (handles[0] + h2[0], handles[1] + h2[1])
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.3 * (hi - lo))                # room for the minimum label
    fig.legend(*handles, loc="lower center", ncol=len(handles[0]), fontsize=9, frameon=False)
    final = RESULTS / f"10_final_{RUN}.json"
    if final.exists():                                  # the one epoch count used for every test plant
        f10 = json.loads(final.read_text())
        mean, n = f10["validation_curve_mean"], f10["epochs"]
        fig2, ax = plt.subplots(figsize=(6, 3.8))
        s0 = f10.get("curve_from_epoch", 1)
        ep = range(s0, s0 + len(mean))
        ax.plot(ep, [c[0] for c in mean], color="#7B7B79", label="training loss")
        validation_loss(ax, ep, [c[1] for c in mean])
        ax.axvline(n, color="#1F1F1E", linestyle="--", linewidth=1, label=f"epochs used: {n}")
        ax2 = ax.twinx()
        ax2.plot(ep, [c[2] for c in mean], color="#2E3F9E", label="validation accuracy")
        ax2.set_ylim(0, 1)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax2.set_ylabel("validation accuracy")
        h1, h2 = ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.3 * (hi - lo))
        ax.legend(h1[0] + h2[0], h1[1] + h2[1], loc="lower left", fontsize=8, frameon=False)
        ax.set_title(f"{RUN}: mean over all 6 validation splits")
        fig2.tight_layout()
        fig2.savefig(RESULTS / f"10b_mean_curve_{RUN}.png", dpi=150)
        print(f"-> {RESULTS / f'10b_mean_curve_{RUN}.png'}")
    fig.suptitle(f"{RUN}: training and validation curves of the best setting")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = RESULTS / f"10b_curves_{RUN}.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
