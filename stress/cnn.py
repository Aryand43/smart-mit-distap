"""Pretrained CNN features from plant tiles.

Each image (608 x 968) is cut into 224 x 224 tiles with 75% overlap (stride 56), and tiles that are mostly
background are dropped. A ResNet-50 pretrained on ImageNet turns each tile into features. Nothing in
the network is trained here.

Features can be taken from any of the four residual blocks, each average-pooled over space:
lower blocks hold colour and fine texture, the last block holds object-level patterns.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torchvision.models import ResNet50_Weights, resnet50

CACHE = Path(__file__).resolve().parents[1] / "results" / "cache"
TILE, MIN_PLANT = 224, 0.4
STRIDE = int(os.environ.get("STRIDE", 56))      # training tiles; 56 = 75% overlap between neighbours
LAYERS = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def backbone() -> torch.nn.Module:
    return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval().to(device())


def _pooled(net, x) -> dict[str, torch.Tensor]:
    """Average-pooled output of every residual block."""
    x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
    out = {}
    for name in LAYERS:
        x = getattr(net, name)(x)
        out[name] = x.mean((2, 3))
    return out


def tile_positions(mask: np.ndarray, stride: int = STRIDE) -> np.ndarray:
    """(row, col) of every tile with at least MIN_PLANT plant pixels, via an integral image."""
    ii = np.pad(mask.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    r = np.arange(0, mask.shape[0] - TILE + 1, stride)
    c = np.arange(0, mask.shape[1] - TILE + 1, stride)
    R, C = np.meshgrid(r, c, indexing="ij")
    frac = (ii[R + TILE, C + TILE] - ii[R, C + TILE] - ii[R + TILE, C] + ii[R, C]) / TILE ** 2
    keep = frac >= MIN_PLANT
    return np.stack([R[keep], C[keep]], 1)


def tiles(img: np.ndarray, mask: np.ndarray, stride: int = STRIDE):
    """Yield (row, col, tile) for tiles with at least MIN_PLANT plant pixels."""
    for r, c in tile_positions(mask, stride):
        yield r, c, img[r:r + TILE, c:c + TILE]


@torch.no_grad()
def tile_features(samples, augment: bool = False, layers=("layer4",), gains=None, tag: str = "raw"):
    """One (n_tiles x d) array per sample, d = widths of the chosen layers concatenated.

    gains: per-sample band multipliers applied to the 0-255 image so it lands in 0-1 (default 1/255).
    tag names the normalisation in the cache, so each fold's white-light gains get their own file.
    With augment, each tile also adds its flips and rotations (8 orientations).
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    kind = "aug" if augment else "plain"
    path = {l: CACHE / f"resnet50_{samples[0].crop}_s{STRIDE}_{tag}_{kind}_{l}.npz" for l in layers}
    missing = [l for l in layers if not path[l].exists()]
    if missing:
        net, out = backbone(), {l: {} for l in missing}
        gains = gains if gains is not None else [np.full(3, 1 / 255, np.float32)] * len(samples)
        for s, g in zip(samples, gains):
            img = np.clip(s.image() * g, 0, 1)
            batch = [t for _, _, t in tiles(img, s.mask())]
            if augment:
                batch = [np.rot90(t, k) for t in batch for k in range(4)] + \
                        [np.rot90(t[:, ::-1], k) for t in batch for k in range(4)]
            x = (np.stack(batch) - _MEAN) / _STD
            x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).float()
            feats = {l: [] for l in missing}
            for i in range(0, len(x), 64):
                pooled = _pooled(net, x[i:i + 64].to(device()))
                for l in missing:
                    feats[l].append(pooled[l].cpu())
            for l in missing:
                out[l][s.path.name] = torch.cat(feats[l]).numpy()
        for l in missing:
            np.savez_compressed(path[l], **out[l])
    data = {l: np.load(path[l]) for l in layers}
    return [np.concatenate([data[l][s.path.name] for l in layers], 1) for s in samples]


def train_softmax(X: np.ndarray, y: np.ndarray, n_classes: int, seed: int,
                  epochs: int = 60, lr: float = 1e-3, weight_decay: float = 1e-2, hidden: int = 64):
    """Small softmax classifier on tile features (cross-entropy). Returns a function: X -> class probabilities."""
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X).float()
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    yt = torch.from_numpy(y).long()
    net = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
                              torch.nn.Dropout(0.3), torch.nn.Linear(hidden, n_classes))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(Xt)).split(256):
            loss = torch.nn.functional.cross_entropy(net((Xt[b] - mu) / sd), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()

    @torch.no_grad()
    def predict_proba(Xnew: np.ndarray) -> np.ndarray:
        return net((torch.from_numpy(Xnew).float() - mu) / sd).softmax(-1).numpy()

    return predict_proba
