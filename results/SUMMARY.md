# Choy sum light stress: results

Data: 48 three-band images = 4 conditions (WL white light control, HL high light, LL low light, SH shade) × 4 time points (4, 24, 48, 72 h) × 3 plants. Chance = 25%.

## How it is tested

- **Rotate the test plant:** train on two plants, test on the third, three rounds. The test plant is never seen in training, and augmentation (flips, rotations) is applied to training patches only.
- **Rotate the pixels:** for pixel PLS-DA, 20 non-overlapping sets of 5,000 pixels per training image in each round (60 runs). Headline accuracy is the median over runs.
- **Count plants, not pixels:** each test plant's pixels or patches vote; the majority is its label.
- **White-light normalisation:** each image is divided by the mean white-light band values of its time point (training plants only), so WL = 1 on every day and each stress is read as a deviation from it.
- **Metrics:** confusion matrix, TP / FP / FN / TN, precision, recall, F1 and accuracy for every technique (`05_metrics_table.md`).

## Results (white-light normalised)

| Technique | Accuracy (median) | F1 (macro) | Accuracy at 4 h |
|---|---|---|---|
| ResNet-50 patches + softmax, plant vote | **75%** | 0.68 | 77% |
| SVM, whole plant | 69% | 0.67 | 83% |
| PLS-DA, pixels, majority vote | 62% | 0.50 | 67% |
| PLS-DA, whole plant | 50% | 0.46 | 50% |
| ResNet-50 (last layer) + SVM, whole plant | 44% | 0.50 | 50% |
| ResNet-50 patches + softmax, patch level | 60% | 0.56 | |

Best model confusion matrix (rows true, columns predicted, 60 predictions per class over 5 seeds):

| | WL | HL | LL | SH |
|---|---|---|---|---|
| WL | 60 | 0 | 0 | 0 |
| HL | 1 | 59 | 0 | 0 |
| LL | 5 | 18 | 17 | 20 |
| SH | 0 | 5 | 21 | 34 |

F1 per class: WL 0.95, HL 0.83, SH 0.60, LL 0.35.

## Findings

1. **The patch-based ResNet with softmax is the best model:** 75% across four conditions, nearly perfect on WL and HL.
2. **White-light normalisation helps:** best model 69% → 75%, pixel PLS-DA 38% → 62%. Without it, part of the signal was day-to-day brightness.
3. **Shade is detected better than low light** (F1 0.60 vs 0.35), consistent with shade giving the stronger response. LL and SH are mostly confused with each other.
4. **Middle ResNet layers carry more of the signal:** layers 2 + 3 + 4 reach 71% vs 54% for the last layer alone (`06_resnet_layers.md`).
5. **Bayesian tuning (Optuna, nested) did not beat the defaults** and chose different settings per test plant: three plants per condition is the main limit (`07_tuning.json`).
6. **Stress maps** (`04_stress_progression_P*.png`) show where stress appears from 4 h to 3 days. On day 2, LL and SH plants are often voted HL, which is worth checking in the day-2 images.

## Next steps

- Fine-tune ResNet on the lab GPU, using middle-layer features; tune epochs, learning rate and loss with Bayesian optimisation.
- CNN-LSTM over the time series once it is confirmed P1 to P3 are the same plants every day.
- Same pipeline on kailan and Arabidopsis.

## Reproduce

```
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/04_chemical_map.py      # stress maps
.venv/bin/python scripts/05_metrics_table.py     # metrics for every technique
.venv/bin/python scripts/06_resnet_layers.py     # ResNet layer comparison
.venv/bin/python scripts/07_tune.py              # Bayesian optimisation
```
Scripts 01 to 03 are the first-pass experiments. Images go in `data/CHOYSUM/<condition>/<time point>/`.
