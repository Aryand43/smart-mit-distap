"""Patch model shared by the tuning, final-setting and comparison scripts (09 to 11).

A plant image is cut into size x size patches on a stride grid (patches at least MIN_PLANT plant). A
frozen ImageNet backbone turns each patch into a feature vector, a small softmax head labels each
patch, and the patches of a plant vote. Patches smaller than 224 pixels are enlarged to 224 first.

Two options from Prof Ram (18 Nov), set by environment variables and applied by 09, 10 and 11:
  BANDS  which of the three bands the network sees: "012" (all, default), one band ("0", "1", "2",
         copied into all three input channels), two bands ("01", "02", "12"; the third channel is
         their mean, so it adds no information), or "mean" (the average of the three bands).
  AUG    number of noisy copies of every training patch (default 0 = off). Each copy is randomly
         flipped / rotated by 90 degrees and gets shot-noise-like Gaussian noise whose standard
         deviation is NOISE (default 0.02) at the white-light level and scales with the square root
         of the signal. Validation and test patches are never augmented. The loss stays plain
         cross-entropy.
  PLANT_FRAC least share of plant pixels in a patch (default 0.4). Higher values keep patches inside
         the leaf, so the model cannot lean on leaf edges and the black background.
  MIN_EPOCHS  shortest training before early stopping may pick an epoch (default 5). The epoch kept
         is the minimum of the validation loss curve.
"""
from __future__ import annotations

import copy
import os
import zipfile

import numpy as np
import torch

from .cnn import CACHE, MIN_PLANT, _MEAN, _STD, _pooled, backbone, device
from .data import CONTROL, conditions, gains, white_reference
from .evaluate import leave_one_plant_out, report

PATIENCE, BATCH = 10, 128
MIN_EPOCHS = int(os.environ.get("MIN_EPOCHS", 5))
PLANT_FRAC = float(os.environ.get("PLANT_FRAC", MIN_PLANT))   # least share of plant pixels in a patch (0.4 = before 20 Nov)   # train at least this long before the best epoch is picked
BANDS = os.environ.get("BANDS", "012")
AUG = int(os.environ.get("AUG", 0))
NOISE = float(os.environ.get("NOISE", 0.02))
WHITE = 0.5                      # white light sits at 0.5 after normalisation


def run_name(net_name: str) -> str:
    """Name of a tuning / final run, so ablations and augmented runs never overwrite each other."""
    parts = ([net_name] + ([f"bands{BANDS}"] if BANDS != "012" else []) + ([f"aug{AUG}n{NOISE:g}"] if AUG else [])
             + ([f"plant{PLANT_FRAC:g}"] if PLANT_FRAC != MIN_PLANT else []))
    return "_".join(parts)


def select_bands(img: np.ndarray) -> np.ndarray:
    """Keep only the bands in BANDS, still as a 3-channel image for the pretrained network."""
    if BANDS == "012":
        return img
    if BANDS == "mean":
        return np.repeat(img.mean(2, keepdims=True), 3, 2)
    chans = [img[..., int(c)] for c in BANDS]
    if len(chans) == 1:
        chans *= 3
    elif len(chans) == 2:
        chans.append((chans[0] + chans[1]) / 2)
    return np.stack(chans, 2)


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One noisy copy of a batch of patches (n x h x w x 3, values 0-1)."""
    x = np.rot90(x, rng.integers(4), axes=(1, 2))
    if rng.random() < 0.5:
        x = x[:, :, ::-1]
    sd = NOISE * np.sqrt(np.clip(x, 0, None) / WHITE)
    return np.clip(x + rng.standard_normal(x.shape).astype(np.float32) * sd, 0, 1).astype(np.float32)
BACKBONES = ("resnet50", "vit_b_16")


def positions(mask: np.ndarray, size: int, stride: int, max_patches: int) -> np.ndarray:
    """(row, col) of every size x size window with at least PLANT_FRAC plant pixels, at most max_patches,
    evenly spaced over all positions on the stride grid. If no window reaches PLANT_FRAC (a large patch
    on a small plant), the 10 windows with the most plant are used, so every image still votes."""
    ii = np.pad(mask.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    R, C = np.meshgrid(np.arange(0, mask.shape[0] - size + 1, stride),
                       np.arange(0, mask.shape[1] - size + 1, stride), indexing="ij")
    frac = (ii[R + size, C + size] - ii[R, C + size] - ii[R + size, C] + ii[R, C]) / size ** 2
    pos = np.stack([R[frac >= PLANT_FRAC], C[frac >= PLANT_FRAC]], 1)
    if len(pos) == 0:
        top = np.argsort(frac, axis=None)[::-1][:10]
        pos = np.stack([R.ravel()[top], C.ravel()[top]], 1)
    if len(pos) > max_patches:
        pos = pos[np.linspace(0, len(pos) - 1, max_patches).round().astype(int)]
    return pos


_NETS = {}


def _device(name: str) -> torch.device:
    """The ViT is not supported by Apple's MPS backend, so on a Mac it runs on the CPU."""
    d = device()
    return torch.device("cpu") if name == "vit_b_16" and d.type == "mps" else d


def _net(name: str):
    """Frozen backbone returning one feature vector per image: ResNet-50 layer 4 (2048) or ViT-B/16 class token (768)."""
    if name not in _NETS:
        if name == "resnet50":
            net = backbone()
            _NETS[name] = lambda x: _pooled(net, x)["layer4"]
        elif name == "vit_b_16":
            from torchvision.models import ViT_B_16_Weights, vit_b_16
            net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            net.heads = torch.nn.Identity()
            net = net.eval().to(_device(name))
            _NETS[name] = net
        else:
            raise ValueError(f"Unknown backbone {name}; choose from {BACKBONES}")
    return _NETS[name]


@torch.no_grad()
def features(samples, g, size: int, stride: int, max_patches: int, tag: str,
             net_name: str = "resnet50", copies: int = 0) -> list[np.ndarray]:
    """Patch features (n_patches x d) per image, cached per backbone and setting.

    g: per-image band multipliers that bring the 0-255 image to 0-1 (white-light normalisation).
    copies: 0 for the clean patches; k > 0 for k noisy copies of every patch (see augment).
    """
    prefix = "tune" if net_name == "resnet50" else net_name       # keeps the ResNet caches from 09 valid
    extra = ((f"_b{BANDS}" if BANDS != "012" else "") + (f"_aug{copies}n{NOISE:g}" if copies else "")
             + (f"_pf{PLANT_FRAC:g}" if PLANT_FRAC != MIN_PLANT else ""))
    path = CACHE / f"{prefix}_{samples[0].crop}_p{size}_s{stride}_n{max_patches}_{tag}{extra}.npz"
    if not path.exists():
        net, out = _net(net_name), {}
        for k, (s, gi) in enumerate(zip(samples, g)):
            img = select_bands(np.clip(s.image() * gi, 0, 1).astype(np.float32))
            pos = positions(s.mask(), size, stride, max_patches)
            rng = np.random.default_rng(k)
            feats = []
            for i in range(0, len(pos), BATCH):
                batch = np.stack([img[r:r + size, c:c + size] for r, c in pos[i:i + BATCH]])
                for xb in ([augment(batch, rng) for _ in range(copies)] if copies else [batch]):
                    x = torch.from_numpy(np.ascontiguousarray(((xb - _MEAN) / _STD).transpose(0, 3, 1, 2))).float()
                    x = x.to(_device(net_name))
                    if size != 224:
                        x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
                    feats.append(net(x).cpu().numpy())
            out[s.path.name] = np.concatenate(feats) if feats else np.zeros((0, 1), np.float32)
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")   # write, then rename: never a half-written cache
        np.savez_compressed(tmp, **out)
        os.replace(tmp, path)
    try:
        data = np.load(path)
        return [data[s.path.name] for s in samples]
    except (zipfile.BadZipFile, KeyError, EOFError, ValueError):
        print(f"  rebuilding damaged cache {path.name}", flush=True)
        path.unlink()
        return features(samples, g, size, stride, max_patches, tag, net_name, copies)


def patch_sets(samples, g, size, stride, max_patches, tag, net_name="resnet50"):
    """(clean features, augmented features or None). Training uses both; validation and test only clean."""
    clean = features(samples, g, size, stride, max_patches, tag, net_name)
    return clean, (features(samples, g, size, stride, max_patches, tag, net_name, AUG) if AUG else None)


def train_set(feats, aug, idx, labels):
    """Training patches of the images idx: the clean ones plus their noisy copies when AUG is on."""
    X, y = stack(feats, idx, labels)
    if aug is not None:
        Xa, ya = stack(aug, idx, labels)
        X, y = np.concatenate([X, Xa]), np.concatenate([y, ya])
    return X, y


def train_head(X, y, p, seed, Xval=None, yval=None, epochs=None, n_classes=4, patience=PATIENCE):
    """Softmax head on patch features. With validation data, keeps the epoch at the minimum of the
    validation loss curve, never before MIN_EPOCHS, and stops after PATIENCE epochs without improvement.

    p: lr, lr_decay, optimizer (adam / adamw / sgd), weight_decay, batch_size, max_epochs.
    Returns (predict, epochs_used, curve): curve[k] = (training loss, validation loss, validation accuracy)
    after k epochs, starting with k = 0 (before training).
    """
    torch.manual_seed(seed)
    Xt, yt = torch.from_numpy(X).float(), torch.from_numpy(y).long()
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(),
                              torch.nn.Dropout(0.3), torch.nn.Linear(64, n_classes))
    params = dict(lr=p["lr"], weight_decay=p["weight_decay"])
    opt = {"adam": lambda: torch.optim.Adam(net.parameters(), **params),
           "adamw": lambda: torch.optim.AdamW(net.parameters(), **params),
           "sgd": lambda: torch.optim.SGD(net.parameters(), momentum=0.9, **params)}[p["optimizer"]]()
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=p["lr_decay"])
    loss_fn = torch.nn.functional.cross_entropy
    if Xval is not None:
        Xv, yv = (torch.from_numpy(Xval).float() - mu) / sd, torch.from_numpy(yval).long()
    n_epochs = epochs or p["max_epochs"]
    first = min(MIN_EPOCHS, n_epochs) - 1          # earliest epoch (0-based) that may be kept
    best, best_ep, best_state, curve = (-np.inf, -np.inf), first, None, []
    if Xval is not None:                            # epoch 0: before any training, so the curve shows the whole descent
        net.eval()
        with torch.no_grad():
            out = net(Xv)
            curve.append((loss_fn(net((Xt - mu) / sd), yt).item(), loss_fn(out, yv).item(),
                          (out.argmax(-1) == yv).float().mean().item()))
    for ep in range(n_epochs):
        net.train()
        total = 0.0
        for b in torch.randperm(len(Xt)).split(p["batch_size"]):
            loss = loss_fn(net((Xt[b] - mu) / sd), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        if Xval is None:
            continue
        net.eval()
        with torch.no_grad():
            out = net(Xv)
            val, acc = loss_fn(out, yv).item(), (out.argmax(-1) == yv).float().mean().item()
        curve.append((total / len(Xt), val, acc))
        if ep < first:
            continue
        if -val > best[0]:                           # minimum of the validation loss curve
            best, best_ep, best_state = (-val, acc), ep, copy.deepcopy(net.state_dict())
        elif patience is not None and ep - best_ep >= patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    @torch.no_grad()
    def predict(Xnew):
        return net((torch.from_numpy(Xnew).float() - mu) / sd).argmax(-1).numpy()

    predict.net, predict.mu, predict.sd = net, mu, sd       # for the heat maps (scripts/10d_heatmaps.py)

    return predict, (best_ep + 1 if Xval is not None else epochs), curve


def stack(feats, idx, labels):
    """Patch features of the images idx, each patch labelled with its image's class."""
    return (np.concatenate([feats[i] for i in idx]),
            np.concatenate([[labels[i]] * len(feats[i]) for i in idx]))


def vote(predict, feats, idx, n_classes):
    """Majority class over each image's patches."""
    return [int(np.bincount(predict(feats[i]), minlength=n_classes).argmax()) if len(feats[i]) else 0
            for i in idx]


def inner_splits(plants, tr):
    """The two (train, validate) splits of the training plants tr: each plant validates once."""
    ps = np.unique(plants[tr])
    return [(tr[plants[tr] != v], tr[plants[tr] == v]) for v in ps]


def early_stopped_epochs(feats, labels, plants, tr, p, n_classes, aug=None):
    """Median epoch at which training stops on the inner splits; used to refit on all training plants."""
    eps = []
    for itr, ival in inner_splits(plants, tr):
        _, ep, _ = train_head(*train_set(feats, aug, itr, labels), p, 0, *stack(feats, ival, labels),
                              n_classes=n_classes)
        eps.append(ep)
    return int(np.median(eps))


def shared_epochs(curves) -> tuple[int, list]:
    """One epoch count for every test plant: average the validation curves of all inner splits and take
    the epoch at the minimum of the mean validation loss, never before MIN_EPOCHS. Returns (epochs, mean curve [(train loss, validation loss, validation accuracy), ...])."""
    n = min(len(c) for c in curves)                 # curve[k] = after k epochs, k = 0 is before training
    mean = [tuple(float(np.mean([c[e][k] for c in curves])) for k in range(3)) for e in range(n)]
    first = min(MIN_EPOCHS, n - 1)
    best = min(range(first, n), key=lambda e: mean[e][1])      # minimum of the mean validation loss
    return best, mean


def evaluate(samples, setting, max_patches, net_name="resnet50", seeds=5, white_level=0.5, shuffle=False):
    """Leave-one-plant-out test of one fixed setting: 5 seeds per test plant, plant vote.

    The number of epochs is the same for every test plant. The setting is first trained for its full
    max_epochs on all inner splits (in each round, one training plant trains and the other validates,
    then they swap: 6 splits in total); the validation curves are averaged and the best epoch of the
    mean curve (see shared_epochs) is used to train the final model for every test plant.

    shuffle (True or a number 1, 2, ... for a different random order) is the sanity check: in every round the labels of the training images are randomly
    permuted among them (test labels stay true). A model that learns stress should then fall to chance
    (0.25); clearly more means it is picking up something other than the labels.
    """
    classes = conditions(samples)
    labels = np.array([classes.index(s.light) for s in samples])
    hours = np.array([s.hours for s in samples])
    plants = np.array([s.plant for s in samples])
    rounds, curves = [], []
    for fold, tr, te in leave_one_plant_out(plants):
        g = [gi * white_level for gi in gains(samples, white_reference(samples, tr))]
        feats, aug = patch_sets(samples, g, setting["patch"], setting["stride"], max_patches, f"white-{fold}", net_name)
        lab = labels.copy()
        if shuffle:
            lab[tr] = np.random.default_rng(1000 * int(shuffle) + int(fold[1:])).permutation(labels[tr])
        rounds.append((fold, tr, te, feats, aug, lab))
        for itr, ival in inner_splits(plants, tr):
            _, _, c = train_head(*train_set(feats, aug, itr, lab), setting, 0, *stack(feats, ival, lab),
                                 epochs=setting["max_epochs"], n_classes=len(classes), patience=None)
            curves.append(c)
    n_epochs, mean_curve = shared_epochs(curves)
    print(f"  {net_name}: {n_epochs} epochs for every test plant (best of the mean validation curve "
          f"over {len(curves)} inner splits, validation accuracy {mean_curve[n_epochs][2]:.2f})", flush=True)
    if n_epochs == len(mean_curve) - 1:
        print(f"  note: the best epoch is the last one tried ({n_epochs}); the curve has not turned yet, "
              f"so a longer run may do better", flush=True)

    true, pred, hrs, run_acc, cal_acc, by_plant = [], [], [], [], [], {}
    for fold, tr, te, feats, aug, lab in rounds:
        X, y = train_set(feats, aug, tr, lab)
        for seed in range(seeds):
            predict, _, _ = train_head(X, y, setting, seed, epochs=n_epochs, n_classes=len(classes))
            p = np.array(vote(predict, feats, te, len(classes)))
            run_acc.append(float(np.mean(p == labels[te])))
            cal_acc.append(float(np.mean(np.array(vote(predict, feats, tr, len(classes))) == lab[tr])))
            true += [classes[i] for i in labels[te]]
            pred += [classes[i] for i in p]
            hrs += hours[te].tolist()
        t, q, h = (np.array(v[-seeds * len(te):]) for v in (true, pred, hrs))
        by_plant[fold] = {f"{x}h": float(np.mean(t[h == x] == q[h == x])) for x in sorted(set(h.tolist()))}
        by_plant[fold]["healthy vs stressed, 72h"] = float(np.mean((t[h == 72] == CONTROL) == (q[h == 72] == CONTROL)))
        print(f"  {net_name} {fold}: test accuracy {np.median(run_acc[-seeds:]):.2f}", flush=True)
    return {"epochs": n_epochs, "validation_curve_mean": mean_curve, "curve_from_epoch": 0, "accuracy_by_test_plant_and_time": by_plant,
            **report(true, pred, hrs, run_acc, classes, CONTROL, cal_acc)}
