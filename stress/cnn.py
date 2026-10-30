"""Pretrained CNN features from plant tiles.

Each image (608 x 968) is cut into 224 x 224 tiles with 50% overlap, and tiles that are mostly
background are dropped. A ResNet-50 pretrained on ImageNet, with its classifier removed, turns
each tile into a 2048-number feature vector. Nothing in the network is trained here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision.models import ResNet50_Weights, resnet50

CACHE = Path(__file__).resolve().parents[1] / "results" / "cache"
TILE, STRIDE, MIN_PLANT = 224, 112, 0.4
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def backbone() -> torch.nn.Module:
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()
    return net.eval().to(device())


def tiles(img: np.ndarray, mask: np.ndarray):
    """Yield (row, col, tile) for tiles with at least MIN_PLANT plant pixels."""
    h, w = mask.shape
    for r in range(0, h - TILE + 1, STRIDE):
        for c in range(0, w - TILE + 1, STRIDE):
            if mask[r:r + TILE, c:c + TILE].mean() >= MIN_PLANT:
                yield r, c, img[r:r + TILE, c:c + TILE]


@torch.no_grad()
def tile_features(samples, augment: bool = False) -> list[np.ndarray]:
    """One (n_tiles x 2048) array per sample. With augment, each tile also adds its flips/rotations."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / f"resnet50_{samples[0].crop}_{'aug' if augment else 'plain'}.npz"
    if key.exists():
        data = np.load(key)
        return [data[s.path.name] for s in samples]

    net, out = backbone(), {}
    for s in samples:
        img, mask = s.image() / 255.0, s.mask()
        batch = [t for _, _, t in tiles(img, mask)]
        if augment:
            batch = [np.rot90(t, k) for t in batch for k in range(4)] + \
                    [np.rot90(t[:, ::-1], k) for t in batch for k in range(4)]
        x = (np.stack(batch) - _MEAN) / _STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).float()
        feats = torch.cat([net(x[i:i + 64].to(device())).cpu() for i in range(0, len(x), 64)])
        out[s.path.name] = feats.numpy()
    np.savez_compressed(key, **out)
    return [out[s.path.name] for s in samples]
