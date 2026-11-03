"""Leave-one-plant-out evaluation and the classification metrics shared by all experiments."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error

from .data import HOURS


def leave_one_plant_out(plants: np.ndarray):
    """Yield (name, train_idx, test_idx), holding out one plant number at a time."""
    for p in np.unique(plants):
        yield f"P{p}", np.flatnonzero(plants != p), np.flatnonzero(plants == p)


def classification_metrics(y_true, y_pred, labels) -> dict:
    """Confusion matrix, TP / FP / FN / TN per class, precision, recall, F1 and accuracy.

    Per-class counts are one-vs-rest: for class c, a positive is "predicted c". Macro averages
    weight every class equally.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total, per_class = cm.sum(), {}
    for i, c in enumerate(labels):
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                        "precision": precision, "recall": recall, "f1": f1}
    return {
        "accuracy": float(np.trace(cm) / total) if total else 0.0,
        "precision_macro": float(np.mean([m["precision"] for m in per_class.values()])),
        "recall_macro": float(np.mean([m["recall"] for m in per_class.values()])),
        "f1_macro": float(np.mean([m["f1"] for m in per_class.values()])),
        "per_class": per_class,
        "confusion": {"labels": list(labels), "rows_true_cols_pred": cm.tolist()},
    }


def accuracy_by_hour(y_true, y_pred, hours) -> dict:
    y_true, y_pred, hours = map(np.asarray, (y_true, y_pred, hours))
    return {int(h): float((y_true[hours == h] == y_pred[hours == h]).mean()) for h in sorted(set(hours))}


def spread(values) -> dict:
    """Median and spread over repeated runs. Outliers: beyond 1.5 x IQR from the quartiles."""
    v = np.asarray(values, dtype=float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    outliers = int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum())
    return {"median": float(med), "q1": float(q1), "q3": float(q3), "mean": float(v.mean()),
            "min": float(v.min()), "max": float(v.max()), "n_runs": int(v.size), "n_outliers": outliers}


def summarise(light_true, light_pred, labels, hours_true=None) -> dict:
    out = {"light_accuracy": accuracy_score(light_true, light_pred),
           "light_confusion": confusion_matrix(light_true, light_pred, labels=labels).tolist()}
    if hours_true is not None:
        out["light_accuracy_by_hour"] = accuracy_by_hour(light_true, light_pred, hours_true)
    return out


def hours_metrics(hours_true, hours_pred) -> dict:
    hours_true, hours_pred = np.asarray(hours_true), np.asarray(hours_pred)
    levels = np.array(sorted(HOURS.values()))
    snapped = levels[np.abs(hours_pred[:, None] - levels[None]).argmin(1)]
    return {
        "hours_mae": mean_absolute_error(hours_true, hours_pred),
        "hours_accuracy_snapped": accuracy_score(hours_true, snapped),
    }
