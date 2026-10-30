"""Leave-one-plant-out evaluation shared by all experiments."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error

from .data import HOURS, LIGHTS


def leave_one_plant_out(plants: np.ndarray):
    """Yield (name, train_idx, test_idx), holding out one plant number at a time."""
    for p in np.unique(plants):
        yield f"P{p}", np.flatnonzero(plants != p), np.flatnonzero(plants == p)


def summarise(light_true, light_pred, hours_true=None, hours_pred=None) -> dict:
    light_true, light_pred = np.asarray(light_true), np.asarray(light_pred)
    out = {
        "light_accuracy": accuracy_score(light_true, light_pred),
        "light_confusion": confusion_matrix(light_true, light_pred, labels=LIGHTS).tolist(),
    }
    if hours_true is not None:
        hours_true, hours_pred = np.asarray(hours_true), np.asarray(hours_pred)
        out["light_accuracy_by_hour"] = {
            h: accuracy_score(light_true[hours_true == h], light_pred[hours_true == h])
            for h in sorted(HOURS.values())
        }
    return out


def hours_metrics(hours_true, hours_pred) -> dict:
    hours_true, hours_pred = np.asarray(hours_true), np.asarray(hours_pred)
    levels = np.array(sorted(HOURS.values()))
    snapped = levels[np.abs(hours_pred[:, None] - levels[None]).argmin(1)]
    return {
        "hours_mae": mean_absolute_error(hours_true, hours_pred),
        "hours_accuracy_snapped": accuracy_score(hours_true, snapped),
    }
