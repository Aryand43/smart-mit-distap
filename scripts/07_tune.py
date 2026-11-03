"""Bayesian optimisation of model parameters (Optuna, TPE sampler) with nested cross-validation.

Outer loop: rotate the test plant, as everywhere else. Inner loop: the two training plants take turns
as a validation plant, and Optuna searches the parameter ranges below for the best mean validation
accuracy. The best parameters are then refit on both training plants and scored once on the test
plant, so the test plant never influences the choice.

Search ranges:
  * SVM (whole plant): kernel linear / RBF / polynomial, C 0.01 to 1000, gamma 0.001 to 10, degree 2 to 3
  * PLS-DA (whole plant): 1 to 6 components
  * ResNet-50 + SVM: which layer, 2 to 10 PCA components, C 0.001 to 100
All inputs are white-light normalised. For speed, the ResNet features of each outer fold use that
fold's white reference (both training plants) inside the inner loop too.
Epochs, learning rate and loss weighting for the deep models need the lab GPU and come next.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import optuna
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress.cnn import LAYERS, tile_features  # noqa: E402
from stress.data import conditions, gains, load, white_reference  # noqa: E402
from stress.evaluate import classification_metrics, leave_one_plant_out  # noqa: E402
from stress.models import PLSDA, band_features  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
TRIALS, CNN_WHITE_LEVEL = 50, 0.5
optuna.logging.set_verbosity(optuna.logging.WARNING)


def svm_from(p):
    kw = {"kernel": p["kernel"], "C": p["C"]}
    if p["kernel"] != "linear":
        kw["gamma"] = p["gamma"]
    if p["kernel"] == "poly":
        kw["degree"] = p["degree"]
    return make_pipeline(StandardScaler(), SVC(**kw))


def suggest_svm(t):
    p = {"kernel": t.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
         "C": t.suggest_float("C", 1e-2, 1e3, log=True)}
    if p["kernel"] != "linear":
        p["gamma"] = t.suggest_float("gamma", 1e-3, 10, log=True)
    if p["kernel"] == "poly":
        p["degree"] = t.suggest_int("degree", 2, 3)
    return p


def main():
    samples = load()
    classes = conditions(samples)
    light = np.array([s.light for s in samples])
    plants = np.array([s.plant for s in samples])

    def plant_features(ref_idx):
        g = gains(samples, white_reference(samples, ref_idx))
        return np.stack([band_features(s.pixels() * gi) for s, gi in zip(samples, g)])

    def cnn_features(fold, tr, layer):
        cg = [g * CNN_WHITE_LEVEL for g in gains(samples, white_reference(samples, tr))]
        return np.stack([f.mean(0) for f in tile_features(samples, False, (layer,), cg, f"white-{fold}")])

    models = {
        "SVM (whole plant)": (suggest_svm, lambda p, fold, tr, val: (svm_from(p), plant_features(tr))),
        "PLS-DA (whole plant)": (lambda t: {"n_components": t.suggest_int("n_components", 1, 6)},
                                 lambda p, fold, tr, val: (PLSDA(classes, p["n_components"]), plant_features(tr))),
        "ResNet-50 + SVM": (
            lambda t: {"layer": t.suggest_categorical("layer", list(LAYERS)),
                       "pca": t.suggest_int("pca", 2, 10), "C": t.suggest_float("C", 1e-3, 1e2, log=True)},
            lambda p, fold, tr, val: (make_pipeline(StandardScaler(), PCA(n_components=p["pca"]),
                                                    SVC(kernel="linear", C=p["C"])),
                                      cnn_features(fold, tr, p["layer"]))),
    }

    results = {}
    for name, (suggest, build) in models.items():
        pred, chosen = np.empty_like(light), {}
        for fold, tr, te in leave_one_plant_out(plants):
            outer_tr = tr

            def objective(t):
                p = suggest(t)
                accs = []
                for val_plant in np.unique(plants[outer_tr]):
                    itr = outer_tr[plants[outer_tr] != val_plant]
                    ival = outer_tr[plants[outer_tr] == val_plant]
                    ref_idx = itr if name != "ResNet-50 + SVM" else outer_tr
                    model, X = build(p, fold, ref_idx, ival)
                    accs.append((model.fit(X[itr], light[itr]).predict(X[ival]) == light[ival]).mean())
                return float(np.mean(accs))

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
            study.optimize(objective, n_trials=TRIALS)
            best = suggest(optuna.trial.FixedTrial(study.best_params))
            model, X = build(best, fold, tr, te)
            pred[te] = model.fit(X[tr], light[tr]).predict(X[te])
            chosen[fold] = {"params": study.best_params, "inner_accuracy": study.best_value,
                            "test_accuracy": float((pred[te] == light[te]).mean())}
        results[name] = {"per_fold": chosen, **classification_metrics(light, pred, classes)}
        print(f"{name}: tuned accuracy {results[name]['accuracy']:.0%}, F1 {results[name]['f1_macro']:.2f}")
        for f, c in chosen.items():
            print(f"   {f}: {c['params']}  inner {c['inner_accuracy']:.0%}  test {c['test_accuracy']:.0%}")

    (RESULTS / "07_tuning.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
