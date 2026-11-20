"""Lock one setting for a patch model and test it (Dr Praveen, 14 and 16 Nov).

09 picks the best setting separately for each test plant. Here the selection is automatic and global:
over all trials of all three test plants, the single trial with the highest validation accuracy wins
(ties go to the patch size and stride with the higher mean validation accuracy). That one setting is
then used for every test plant, with one epoch count for all of them: the best epoch of the validation
curve averaged over all 6 inner splits (stress.patches.shared_epochs). The head is trained on both
training plants with 5 seeds, and the patches of each test image vote.
Calibration accuracy is the same model scored on its own training plants.

Note: the validation scores of one round come from plants that are test plants in other rounds, so a
global choice is slightly less strict than 09's per-plant choice.

Run once per backbone, after 09 for that backbone:
    python scripts/10_final_patch_model.py
    BACKBONE=vit_b_16 python scripts/10_final_patch_model.py
To fix the patch size and stride instead, set PATCH and STRIDE (the training settings are then the best
trial within that patch size and stride):
    PATCH=64 STRIDE=4 python scripts/10_final_patch_model.py

Uses the same BANDS / AUG / NOISE settings as the 09 run it follows (see stress/patches.py).
Reads results/09_tune_<run>.json. Writes results/10_final_<run>.json and
results/10_patch_stride_<run>.csv (best, mean and median validation accuracy for every patch size
and stride tried, and for every patch size over all strides).
"""
from __future__ import annotations

import collections
import csv
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.data import load  # noqa: E402
from stress.patches import BACKBONES, evaluate, run_name  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
BACKBONE = os.environ.get("BACKBONE", "resnet50")
SHOWN = ["patch", "stride", "epochs", "lr", "lr_decay", "optimizer", "weight_decay", "batch_size"]


def summary(values, **key) -> dict:
    return {**key, "trials": len(values), "best": max(values), "mean": statistics.mean(values),
            "median": statistics.median(values)}


def select(tuning: dict, patch: int | None = None, stride: int | None = None):
    """Best trial over all test plants (optionally within one patch size and stride), and the
    validation summaries behind the choice."""
    trials = [t for f in tuning["per_test_plant"].values() for t in f["trials"] if t["value"] is not None]
    by_pair, by_patch = collections.defaultdict(list), collections.defaultdict(list)
    for t in trials:
        by_pair[(t["params"]["patch"], t["params"]["stride"])].append(t["value"])
        by_patch[t["params"]["patch"]].append(t["value"])
    pairs = [summary(v, patch=p, stride=s) for (p, s), v in sorted(by_pair.items())]
    patches = [summary(v, patch=p, stride="all") for p, v in sorted(by_patch.items())]
    pair_mean = {(r["patch"], r["stride"]): r["mean"] for r in pairs}
    if patch is not None:
        trials = [t for t in trials if t["params"]["patch"] == patch and t["params"]["stride"] == stride]
        if not trials:
            raise SystemExit(f"No tuning trial used patch {patch}, stride {stride}; see the table in the CSV")
    best = max(trials, key=lambda t: (t["value"], pair_mean[(t["params"]["patch"], t["params"]["stride"])]))
    return {**best["params"], "epochs_used": best.get("epochs_used")}, best["value"], pairs, patches


def main():
    if BACKBONE not in BACKBONES:
        raise SystemExit(f"BACKBONE must be one of {BACKBONES}")
    run = run_name(BACKBONE)
    tuning = json.loads((RESULTS / f"09_tune_{run}.json").read_text())
    patch, stride = os.environ.get("PATCH"), os.environ.get("STRIDE")
    if (patch is None) != (stride is None):
        raise SystemExit("Set both PATCH and STRIDE, or neither")
    setting, val_acc, pairs, patches = select(tuning, *(map(int, (patch, stride)) if patch else (None, None)))
    max_patches = tuning["search"]["max_patches"]

    with open(RESULTS / f"10_patch_stride_{run}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Patch (px)", "Stride (px)", "Trials", "Best validation accuracy",
                    "Mean validation accuracy", "Median validation accuracy"])
        for r in patches + pairs:
            w.writerow([r["patch"], r["stride"], r["trials"], *(f"{r[k]:.2f}" for k in ("best", "mean", "median"))])

    result = evaluate(load(), setting, max_patches, BACKBONE)
    shown = {**setting, "epochs": result["epochs"]}
    print(f"\n{run}: hyperparameters obtained by Bayesian optimisation "
          f"(validation accuracy {val_acc:.2f}); one setting and one epoch count for all test plants")
    for k in SHOWN:
        print(f"  {k:13s} {shown.get(k)}")
    print("  loss          cross-entropy", flush=True)
    out = {"backbone": BACKBONE, "run": run, "selected_setting": shown, "validation_accuracy": val_acc,
           "max_patches": max_patches, "network_input": "224 x 224", "loss": "cross-entropy",
           "patch_size_summary": patches, "patch_stride_table": pairs, **result}
    (RESULTS / f"10_final_{run}.json").write_text(json.dumps(out, indent=2))

    a, c = out["accuracy_over_runs"], out["calibration_accuracy_over_runs"]
    print(f"\ncalibration accuracy {c['median']:.2f}, test accuracy median {a['median']:.2f} (mean {a['mean']:.2f})")
    print("\nAccuracy by test plant and time point (day 3 check)")
    for plant, row in out["accuracy_by_test_plant_and_time"].items():
        print(f"  {plant}: " + ", ".join(f"{k} {v:.2f}" for k, v in row.items()))
    for title, key in [("Four classes", "by_time_point"), ("Healthy vs stressed", "by_time_point, stressed vs control")]:
        print(f"\n{title}\n  time   precision recall  F1    accuracy")
        for h, m in out[key].items():
            print(f"  {h:6s} {m['precision_macro']:.2f}      {m['recall_macro']:.2f}    {m['f1_macro']:.2f}  {m['accuracy']:.2f}")


if __name__ == "__main__":
    main()
