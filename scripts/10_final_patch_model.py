"""Lock one setting for the patch model and test it, as Dr Praveen asked (14 Nov).

09 picks the best setting separately for each test plant. Here the selection is automatic and global:
over all trials of all three test plants, the single trial with the highest validation accuracy wins
(ties go to the patch size and stride with the higher mean validation accuracy). That one setting is
then used for every test plant: the epoch count comes from early stopping on the inner splits, the
head is trained on both training plants with 5 seeds, and the patches of each test image vote.

Note: the validation scores of one round come from plants that are test plants in other rounds, so a
global choice is slightly less strict than 09's per-plant choice. Report both.

Reads results/09_tune_patches.json; writes results/10_final_patch_model.json and
results/10_patch_stride_table.csv (validation accuracy for every patch size and stride tried).
"""
from __future__ import annotations

import collections
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.data import load  # noqa: E402
from stress.patches import evaluate  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def select(tuning: dict):
    """Best trial over all test plants, and the patch size x stride table behind the choice."""
    trials = [t for f in tuning["per_test_plant"].values() for t in f["trials"] if t["value"] is not None]
    groups = collections.defaultdict(list)
    for t in trials:
        groups[(t["params"]["patch"], t["params"]["stride"])].append(t["value"])
    table = [{"patch": p, "stride": s, "trials": len(v), "best": max(v), "mean": statistics.mean(v)}
             for (p, s), v in sorted(groups.items())]
    group_mean = {(r["patch"], r["stride"]): r["mean"] for r in table}
    best = max(trials, key=lambda t: (t["value"], group_mean[(t["params"]["patch"], t["params"]["stride"])]))
    return dict(best["params"]), best["value"], table


def main():
    tuning = json.loads((RESULTS / "09_tune_patches.json").read_text())
    setting, val_acc, table = select(tuning)
    max_patches = tuning["search"]["max_patches"]
    print(f"selected {setting} (validation accuracy {val_acc:.2f})", flush=True)

    with open(RESULTS / "10_patch_stride_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Patch (px)", "Stride (px)", "Trials", "Best validation accuracy", "Mean validation accuracy"])
        for r in table:
            w.writerow([r["patch"], r["stride"], r["trials"], f"{r['best']:.2f}", f"{r['mean']:.2f}"])

    result = evaluate(load(), setting, max_patches)
    out = {"selected_setting": setting, "validation_accuracy": val_acc, "max_patches": max_patches,
           "patch_stride_table": table, **result}
    (RESULTS / "10_final_patch_model.json").write_text(json.dumps(out, indent=2))
    p, a = out["plant level"], out["accuracy_over_runs"]
    print(f"fixed setting: accuracy median {a['median']:.2f} (mean {a['mean']:.2f}), F1 {p['f1_macro']:.2f}, "
          f"stressed vs control {out['plant level, stressed vs control']['accuracy']:.2f}")
    print(p["confusion"])


if __name__ == "__main__":
    main()
