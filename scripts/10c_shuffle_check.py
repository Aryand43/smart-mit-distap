"""Sanity check: does the model learn the stress labels, or something else?

Takes the locked setting from 10 and trains it again, but in every round the labels of the training
images are randomly permuted among them (the test labels stay true). Everything else is identical:
same patches, same epochs rule, same 5 seeds, same plant vote. Repeated for SHUFFLES (default 5)
random orders, because with 16 test images per plant one shuffle can land far from chance by luck.

  * If the model learns stress, the shuffled run falls to chance: about 0.25 for four classes.
  * If it stays clearly above chance, the model can predict the condition without the labels being
    right, which points to a shortcut (for example image brightness per imaging session).

    python scripts/10c_shuffle_check.py                    # uses results/10_final_resnet50.json
    RUN=resnet50_aug2n0.02 AUG=2 NOISE=0.02 python scripts/10c_shuffle_check.py

Set the same BANDS / AUG / NOISE / PLANT_FRAC as the run being checked. Writes results/10c_shuffle_<run>.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.data import load  # noqa: E402
from stress.patches import evaluate, run_name  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
BACKBONE = os.environ.get("BACKBONE", "resnet50")
SHUFFLES = int(os.environ.get("SHUFFLES", 5))


def main():
    run = os.environ.get("RUN", run_name(BACKBONE))
    final = json.loads((RESULTS / f"10_final_{run}.json").read_text())
    setting = final["selected_setting"]
    print(f"{run}: real labels gave test accuracy {final['accuracy_over_runs']['median']:.2f}; "
          f"now training on shuffled labels", flush=True)
    samples, runs, binary = load(), [], []
    for k in range(1, SHUFFLES + 1):                # several random orders: one can be lucky with 16 test images
        r = evaluate(samples, setting, final["max_patches"], BACKBONE, shuffle=k)
        runs.append(r["accuracy_over_runs"]["mean"])
        binary.append(r["plant level, stressed vs control"]["accuracy"])
        print(f"  shuffle {k}: test accuracy {runs[-1]:.2f}", flush=True)
    a = {"median": float(np.median(runs)), "mean": float(np.mean(runs)), "min": float(min(runs)),
         "max": float(max(runs)), "per_shuffle": runs}
    out = {"run": run, "setting": setting, "real_labels_test_accuracy": final["accuracy_over_runs"],
           "shuffled_labels_test_accuracy": a, "shuffled_labels_healthy_vs_stressed": float(np.mean(binary)),
           "chance": 0.25}
    (RESULTS / f"10c_shuffle_{run}.json").write_text(json.dumps(out, indent=2))
    verdict = ("close to chance: the model is learning from the labels"
               if a["mean"] <= 0.33 else "clearly above chance: check for a shortcut")
    print(f"\nshuffled labels, {SHUFFLES} random orders: test accuracy mean {a['mean']:.2f} "
          f"(range {a['min']:.2f} to {a['max']:.2f}), chance 0.25 -> {verdict}")


if __name__ == "__main__":
    main()
