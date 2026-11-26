"""Side-by-side validation curves of the two-phase fine-tuning at different phase 2 learning rates
(Dr Praveen, 26 Nov): one panel per rate setting (layer 4 / head), one line per batch size, same y axis.

    python scripts/13b_compare_lr.py

Reads every results/13_finetune_bs*.json and writes results/13_compare_lr.png; prints the test accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
SHADES = {8: "#E8A0A8", 16: "#C4505E", 32: "#9D192B"}


def main():
    runs = {}
    for f in sorted(RESULTS.glob("13_finetune_bs*.json")):
        r = json.loads(f.read_text())
        s = r["settings"]
        runs.setdefault((s["lr_phase2_layer4"], s["lr_phase2_head"]), []).append(r)
    keys = sorted(runs)
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), sharey=True, squeeze=False)
    finite = [v[1] for rs in runs.values() for r in rs for v in r["validation_curve_mean"] if np.isfinite(v[1])]
    print("layer 4 lr  head lr  batch  phase 2 epochs  test phase 1  test phase 1 + 2 (median / mean)  healthy vs stressed")
    for ax, key in zip(axes[0], keys):
        for r in sorted(runs[key], key=lambda r: r["batch_size"]):
            b, m, p1 = r["batch_size"], np.array(r["validation_curve_mean"], float), r["phase1_epochs"]
            ep = np.arange(len(m))
            ax.plot(ep, m[:, 1], color=SHADES.get(b, "k"), lw=2, label=f"batch {b}: validation")
            ax.plot(ep, m[:, 0], color=SHADES.get(b, "k"), lw=1, ls="--", alpha=0.7, label=f"batch {b}: training")
            ax.scatter(p1 + r["phase2_epochs"], m[p1 + r["phase2_epochs"], 1], color=SHADES.get(b, "k"), zorder=3, s=25)
            a1, a2 = r["phase 1 only"]["accuracy_over_runs"], r["phase 1 + 2"]["accuracy_over_runs"]
            hs = r["phase 1 + 2"]["plant level, stressed vs control"]["accuracy"]
            print(f"{key[0]:<10g}  {key[1]:<7g}  {b:5d}  {r['phase2_epochs']:14d}  {a1['median']:.2f}          "
                  f"{a2['median']:.2f} / {a2['mean']:.2f}                       {hs:.2f}")
        ax.axvspan(0, runs[key][0]["phase1_epochs"], color="#7B7B79", alpha=0.08)
        ax.text(0.2, 0.02, "phase 1: head only", transform=ax.get_xaxis_transform(), fontsize=8, color="#7B7B79")
        ax.set_title(f"phase 2: layer 4 lr {key[0]:g}, head lr {key[1]:g}", fontsize=10)
        ax.set_xlabel("epoch (dot: minimum of the mean validation loss)")
    if finite:
        lo, hi = min(finite), max(finite)
        axes[0][0].set_ylim(lo - 0.05 * (hi - lo), min(hi, 3.0) + 0.05 * (hi - lo))
    axes[0][0].set_ylabel("loss, mean over 6 validation splits")
    axes[0][-1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "13_compare_lr.png", dpi=140)
    print("-> results/13_compare_lr.png")


if __name__ == "__main__":
    main()
