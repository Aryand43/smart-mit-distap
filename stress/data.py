"""Loading and labelling the three-band plant images.

File names look like ``D2_HL_C_P3_ref_3band.png`` or ``HRS_4_WL_C_P1_ref_3band.png``:
time point, light condition, C, plant number. The background is already masked to black.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

LIGHTS = ["HL", "LL", "WL"]                      # high, low, white light
HOURS = {"HRS_4": 4, "D1": 24, "D2": 48, "D3": 72}

_NAME = re.compile(r"^(HRS_4|D\d)_(HL|LL|WL)_C_P(\d)_ref_3band\.png$")


@dataclass
class Sample:
    path: Path
    crop: str
    light: str
    hours: int
    plant: int

    @property
    def session(self) -> str:
        """One imaging session = one light condition at one time point."""
        return f"{self.light}_{self.hours}h"

    def image(self) -> np.ndarray:
        """H x W x 3 float array, 0-255."""
        return np.asarray(Image.open(self.path).convert("RGB"), dtype=np.float32)

    def mask(self, min_size: int = 200) -> np.ndarray:
        """Plant pixels: non-black, with isolated speckle removed."""
        img = self.image()
        m = img.sum(-1) > 0
        m = ndimage.binary_opening(m, iterations=2)
        labels, n = ndimage.label(m)
        if n:
            sizes = ndimage.sum(m, labels, range(1, n + 1))
            keep = np.isin(labels, np.flatnonzero(sizes >= min_size) + 1)
            m = keep
        return m


def load(crop: str = "CHOYSUM") -> list[Sample]:
    samples = []
    for p in sorted((DATA_ROOT / crop).glob("*/*/*.png")):
        m = _NAME.match(p.name)
        if not m:
            raise ValueError(f"Unexpected file name: {p}")
        t, light, plant = m.groups()
        samples.append(Sample(p, crop, light, HOURS[t], int(plant)))
    return samples
