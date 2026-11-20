# Meeting with Prof Rajeev Ram and Dr Praveen (18 Nov 2026)

## Prof Ram's feedback on the results (slides 8 and 9)

- **Day 3**: even healthy vs stressed drops on day 3 (0.77), so it is not only low light vs shade.
  Something else is going on. Check the errors plant by plant (script 10 now prints accuracy per
  test plant and time point). Dr Praveen thinks one plant may differ on day 3; check for brightness or
  noise differences in those images.
- **Overfitting**: calibration 1.00 vs test 0.69. Do not keep changing the loss function or tuning
  more: that only fits the model to the data we have. Keep the loss simple (cross-entropy) and make
  the model robust by **adding a little physically meaningful noise** to the training data, several
  noisy copies per patch. In an earlier paper of his group this beat preprocessing and loss changes.
- **Band ablation**: does the spectral information help, or is it mostly leaf shape? Run ResNet-50
  with Bayesian optimisation on 1 band, 2 bands and all 3 (and the mean of the three).
- **More bands**: Dr Praveen's idea, agreed by Prof Ram: all ~200 to 250 hyperspectral bands into a 3D
  ResNet, to see whether accuracy keeps rising or levels off at 3. Needs the full hyperspectral cubes.
- **Focus on ResNet-50 only** for now; no need to repeat everything for every model.
- Order agreed with Dr Praveen: band combinations (ResNet and 3D ResNet) first, then Raman fusion,
  then the other crops. Finish one full pipeline for choy sum before kailan.
- Epoch search 1 to 120 (Dr Praveen, 17 Nov).

## Setup diagram (done 18 Nov)

- Left source blue, 480 nm; right source red, 780 nm.
- Labels: LED to lens 2 to 3.5 cm, lens to filter 0.5 to 1.5 cm, optics to plant about 30 cm, camera to
  sample more than 35 cm, illumination area about 10 x 10 cm ("illumination area", not "field of
  view"). For the January meeting, not the DiSTAP symposium.

## Other

- Symposium slides (Dr Praveen): Aryan Dutt listed in full as visiting student, DiSTAP SMART / NTU.
- Prof Ram: email him to set up a separate time to talk about the visiting student exchange.

## Code changes (18 Nov)

- `stress/patches.py`: `BANDS` ("012", "0", "1", "2", "01", "02", "12", "mean"), `AUG` (noisy copies
  per training patch, random flip / 90 degree rotation, shot-noise-like Gaussian noise of standard
  deviation `NOISE` at the white-light level). Validation and test patches stay clean. Output files
  are named per run, e.g. `09_tune_resnet50_bands0.json`, `10_final_resnet50_aug2n0.02.json`.
- `scripts/09_tune_patches.py`: epochs 1 to 120.
- `scripts/10_final_patch_model.py`: prints accuracy per test plant and time point (day 3 check).

## GPU runs (ResNet-50 only)

    git pull && source .venv/bin/activate

1. Baseline with the new epoch range, then noise augmentation (2 noisy copies, noise 0.02):

       nohup sh -c 'TRIALS=30 MAX_PATCHES=600 python scripts/09_tune_patches.py && python scripts/10_final_patch_model.py && AUG=2 NOISE=0.02 TRIALS=30 MAX_PATCHES=600 python scripts/09_tune_patches.py && AUG=2 NOISE=0.02 python scripts/10_final_patch_model.py' > results/gpu_aug.log 2>&1 &

2. Band ablation (no augmentation), one run per band set:

       nohup sh -c 'for b in 0 1 2 01 02 12 mean; do BANDS=$b TRIALS=30 MAX_PATCHES=600 python scripts/09_tune_patches.py && BANDS=$b python scripts/10_final_patch_model.py; done' > results/gpu_bands.log 2>&1 &

Run 1 and 2 one after the other, not at the same time (one GPU). Each 09 + 10 pair takes roughly
1 to 1.5 hours on the RTX 4070 Ti, so run 2 is most of a day.

## 20 Nov: is the model learning the right thing?

- Epochs: every curve now starts at epoch 0 (before training); the first-epoch drop was hidden before,
  which is why epoch 1 looked best. The epoch kept is the minimum of the validation loss, averaged over
  all 6 inner splits, one count for every plant, never below MIN_EPOCHS (5). Learning rate 1e-5 to 1e-2.
- `scripts/10b_plot_curves.py`: training / validation loss and validation accuracy per epoch, minimum
  labelled, rising part highlighted.
- `scripts/10c_shuffle_check.py`: same model trained on randomly shuffled labels (5 random orders);
  should fall to 0.25. Laptop smoke test: 0.17 to 0.25, mean 0.22.
- `scripts/10d_heatmaps.py`: Grad-CAM heat map per test image and the share of attention on leaf
  interior, leaf edge and background. Laptop check (64 px, test plant 1): interior 0.62 of attention
  (0.60 of area), edge 0.13 (0.09), background 0.25 (0.31): mostly leaf tissue.
- `PLANT_FRAC=0.8`: patches must be at least 80% plant (was 40%), run as a separate variant.
