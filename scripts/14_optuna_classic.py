"""Bayesian optimisation of the classic classifiers on frozen ResNet-50 features (Dr Praveen, 26 Nov).

One setting per classifier for the whole experiment (all plants, stress types and durations), so no
setting is picked by hand:

  plsda   number of components 2 to 50 in steps of one: all 49 values are tried (a grid search inside
          Optuna), so the choice is exhaustive, not sampled
  svm     kernel (linear or rbf), C (linear 1e-6 to 1, rbf 1e-2 to 1e3) and, for rbf, gamma (1e-6 to 1e-2)
  logreg  C (1e-6 to 1e1)

The score of a setting is the mean patch accuracy over the 6 inner splits (in each leave-one-plant-out
round the two training plants validate each other); the test plants are never used. Optuna's TPE sampler
(seed 0), N_TRIALS (40) settings for svm and logreg; settings that are clearly worse after a few splits are
stopped early (median pruner). The study is stored in results/14_optuna.db, so a stopped run resumes.

The best settings are saved to results/14_classic_settings.json (reused by 16 and 17) and tested on the
held-out plants: results/14_classic_test.json and .csv.

    python scripts/14_optuna_classic.py
    CLASSIFIERS="plsda" python scripts/14_optuna_classic.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import warnings

import numpy as np
import optuna
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.classic import (CSV_HEAD, MAX_PATCHES, PATCH, RESULTS, SETTINGS, STRIDE, fit, fold_features,  # noqa: E402
                            line, row, stack, test)
from stress.data import conditions, load  # noqa: E402
from stress.evaluate import leave_one_plant_out  # noqa: E402
from stress.patches import inner_splits  # noqa: E402

KINDS = os.environ.get("CLASSIFIERS", "plsda logreg svm").split()
N_TRIALS = int(os.environ.get("N_TRIALS", 40))
PLS_MAX = int(os.environ.get("PLS_MAX", 50))


def space(trial, kind):
    if kind == "plsda":
        return {"n_components": trial.suggest_int("n_components", 2, PLS_MAX)}
    if kind == "logreg":
        return {"C": trial.suggest_float("C", 1e-6, 1e1, log=True)}
    if trial.suggest_categorical("kernel", ["linear", "rbf"]) == "linear":     # large C on the linear kernel does not converge
        return {"kernel": "linear", "C": trial.suggest_float("C_linear", 1e-6, 1e0, log=True)}
    return {"kernel": "rbf", "C": trial.suggest_float("C_rbf", 1e-2, 1e3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-6, 1e-2, log=True)}


def main():
    samples = load()
    classes = conditions(samples)
    labels = np.array([classes.index(s.light) for s in samples])
    hours = np.array([s.hours for s in samples])
    plants = np.array([s.plant for s in samples])
    folds = list(leave_one_plant_out(plants))
    feats = fold_features(samples, folds)
    splits = [(stack(feats[fold], itr, labels), stack(feats[fold], ival, labels)[:2])
              for fold, tr, te in folds for itr, ival in inner_splits(plants, tr)]
    print(f"ResNet-50 frozen features: patch {PATCH}, stride {STRIDE}, at most {MAX_PATCHES} patches per image; "
          f"{len(splits)} inner splits", flush=True)
    storage = f"sqlite:///{RESULTS / '14_optuna.db'}"
    saved = json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}
    tests, rows = {}, []
    for kind in KINDS:
        t0 = time.time()

        def objective(trial):
            p, accs = space(trial, kind), []
            for k, ((X, y, o), (Xv, yv)) in enumerate(splits):
                accs.append(float(np.mean(fit(kind, p, X, y, o, len(classes))(Xv).argmax(1) == yv)))
                trial.report(float(np.mean(accs)), k)
                if kind != "plsda" and trial.should_prune():
                    raise optuna.TrialPruned()
            trial.set_user_attr("split_accuracy", accs)
            return float(np.mean(accs))

        if kind == "plsda":
            sampler, n = optuna.samplers.GridSampler({"n_components": list(range(2, PLS_MAX + 1))}, seed=0), PLS_MAX - 1
        else:
            sampler, n = optuna.samplers.TPESampler(seed=0), N_TRIALS
        study = optuna.create_study(study_name=f"{kind}_p{PATCH}_s{STRIDE}_n{MAX_PATCHES}", storage=storage,
                                    direction="maximize", sampler=sampler, load_if_exists=True,
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
        done = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED) for t in study.trials)
        cb = lambda s, t: print(f"  {kind} trial {t.number}: {t.params} -> {t.value if t.value is not None else 'stopped early'}"
                                f"  ({time.time() - t0:.0f} s)", flush=True)
        if n - done > 0:
            study.optimize(objective, n_trials=n - done, callbacks=[cb])
        best = study.best_trial
        p = space(optuna.trial.FixedTrial(best.params), kind)
        saved[kind] = {"params": p, "validation_patch_accuracy": best.value,
                       "validation_split_accuracy": best.user_attrs.get("split_accuracy"),
                       "n_trials": len(study.trials),
                       "all_trials": [{"params": t.params, "value": t.value, "state": t.state.name} for t in study.trials],
                       "features": f"ResNet-50 frozen, patch {PATCH}, stride {STRIDE}, at most {MAX_PATCHES} patches, all bands"}
        SETTINGS.write_text(json.dumps(saved, indent=2))
        print(f"  {kind}: best {p}, inner validation patch accuracy {best.value:.3f} ({len(study.trials)} trials)", flush=True)
        tests[kind] = test(kind, p, feats, folds, labels, hours, classes)
        print(line(kind, tests[kind]), flush=True)
        rows.append([kind, json.dumps(p), f"{best.value:.3f}", *row(tests[kind])])
        (RESULTS / "14_classic_test.json").write_text(json.dumps(tests, indent=2))
    with open(RESULTS / "14_classic_test.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Classifier", "Setting", "Inner validation (patches)", *CSV_HEAD])
        w.writerows(rows)
    print(f"\n-> {SETTINGS.name}, 14_classic_test.json, 14_classic_test.csv")


if __name__ == "__main__":
    main()
