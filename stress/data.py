"""Loading, labelling and normalising the three-band plant images.

File names look like ``D2_HL_C_P3_ref_3band.png`` or ``HRS_4_WL_C_P1_ref_3band.png``:
time point, stress condition, C, plant number. The background is already masked to black.
Any condition folder is picked up (HL, LL, WL, and shade once it arrives); WL is the control.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

CONTROL = "WL"                                   # white light
ORDER = ["WL", "HL", "LL"]                       # known conditions first; others (shade) follow
HOURS = {"HRS_4": 4, "D1": 24, "D2": 48, "D3": 72}

_NAME = re.compile(r"^(HRS_4|D\d)_([A-Z]+)_C_P(\d)_ref_3band\.png$")


@lru_cache(maxsize=None)
def _image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


@lru_cache(maxsize=None)
def _mask(path: Path, min_size: int = 200) -> np.ndarray:
    m = _image(path).sum(-1) > 0
    m = ndimage.binary_opening(m, iterations=2)
    labels, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, labels, range(1, n + 1))
        m = np.isin(labels, np.flatnonzero(sizes >= min_size) + 1)
    return m


@dataclass
class Sample:
    path: Path
    crop: str
    light: str
    hours: int
    plant: int

    @property
    def session(self) -> str:
        """One imaging session = one condition at one time point."""
        return f"{self.light}_{self.hours}h"

    def image(self) -> np.ndarray:
        """H x W x 3 float array, 0-255."""
        return _image(self.path)

    def mask(self) -> np.ndarray:
        """Plant pixels: non-black, with isolated speckle removed."""
        return _mask(self.path)

    def pixels(self) -> np.ndarray:
        """N x 3 band values of the plant pixels."""
        return self.image()[self.mask()]


def load(crop: str = "CHOYSUM") -> list[Sample]:
    samples = []
    for p in sorted((DATA_ROOT / crop).glob("*/*/*.png")):
        m = _NAME.match(p.name)
        if not m:
            raise ValueError(f"Unexpected file name: {p}")
        t, light, plant = m.groups()
        samples.append(Sample(p, crop, light, HOURS[t], int(plant)))
    return samples


def conditions(samples) -> list[str]:
    """Condition labels present in the data, control first."""
    present = {s.light for s in samples}
    return [c for c in ORDER if c in present] + sorted(present - set(ORDER))


def white_reference(samples, idx) -> dict[int, np.ndarray]:
    """Mean band values of the white-light plant pixels at each time point, from samples[idx] only.

    Dividing every image of a time point by its reference sets white light to 1 on every day, so
    day-to-day differences in the control (plant variation, imaging) are removed and each stress is
    read as a deviation from white light. Only training plants go into the reference, so the test
    plant never influences it.
    """
    ref = {}
    for h in sorted({samples[i].hours for i in idx}):
        px = [samples[i].pixels() for i in idx if samples[i].hours == h and samples[i].light == CONTROL]
        ref[h] = np.concatenate(px).mean(0)
    return ref


def gains(samples, ref: dict[int, np.ndarray] | None) -> list[np.ndarray]:
    """Per-image band multiplier: 1 / white reference of its time point (ones when ref is None)."""
    return [np.ones(3, np.float32) if ref is None else (1.0 / ref[s.hours]).astype(np.float32) for s in samples]
