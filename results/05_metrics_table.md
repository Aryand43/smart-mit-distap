# Classification metrics, choy sum

| Technique | Normalisation | Level | Accuracy, median (IQR) over runs | Runs | Precision | Recall | F1 | Accuracy at 4 h |
|---|---|---|---|---|---|---|---|---|
| PLS-DA (whole plant) | raw | plant | 56% (53% to 62%) | 3 | 0.47 | 0.58 | 0.51 | 58% |
| SVM (whole plant) | raw | plant | 69% (59% to 69%) | 3 | 0.59 | 0.62 | 0.60 | 83% |
| PLS-DA (pixels, majority vote) | raw | plant | 38% (38% to 44%) | 60 | 0.36 | 0.40 | 0.34 | 35% |
| ResNet-50 + SVM (whole plant) | raw | plant | 69% (62% to 69%) | 3 | 0.65 | 0.65 | 0.63 | 67% |
| ResNet-50 patches + softmax (plant vote) | raw | plant | 62% (62% to 69%) | 15 | 0.61 | 0.65 | 0.61 | 62% |
| ResNet-50 patches + softmax (patch level) | raw | patch | 63% (pooled) | pooled | 0.56 | 0.58 | 0.57 |  |
| PLS-DA (whole plant) | white | plant | 50% (47% to 59%) | 3 | 0.41 | 0.54 | 0.46 | 50% |
| SVM (whole plant) | white | plant | 69% (62% to 72%) | 3 | 0.67 | 0.67 | 0.67 | 83% |
| PLS-DA (pixels, majority vote) | white | plant | 62% (44% to 69%) | 60 | 0.44 | 0.58 | 0.50 | 67% |
| ResNet-50 + SVM (whole plant) | white | plant | 50% (44% to 56%) | 3 | 0.52 | 0.50 | 0.50 | 42% |
| ResNet-50 patches + softmax (plant vote) | white | plant | 69% (59% to 75%) | 15 | 0.62 | 0.67 | 0.63 | 70% |
| ResNet-50 patches + softmax (patch level) | white | patch | 61% (pooled) | pooled | 0.56 | 0.57 | 0.56 |  |

Precision, recall and F1 are macro averages over WL, HL, LL, SH, pooled over all runs.

### Per class, raw

| Technique | Class | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| PLS-DA (whole plant) | WL | 11 | 1 | 1 | 35 | 0.92 | 0.92 | 0.92 |
| PLS-DA (whole plant) | HL | 7 | 6 | 5 | 30 | 0.54 | 0.58 | 0.56 |
| PLS-DA (whole plant) | LL | 0 | 0 | 12 | 36 | 0.00 | 0.00 | 0.00 |
| PLS-DA (whole plant) | SH | 10 | 13 | 2 | 23 | 0.43 | 0.83 | 0.57 |
| SVM (whole plant) | WL | 12 | 0 | 0 | 36 | 1.00 | 1.00 | 1.00 |
| SVM (whole plant) | HL | 8 | 3 | 4 | 33 | 0.73 | 0.67 | 0.70 |
| SVM (whole plant) | LL | 1 | 6 | 11 | 30 | 0.14 | 0.08 | 0.11 |
| SVM (whole plant) | SH | 9 | 9 | 3 | 27 | 0.50 | 0.75 | 0.60 |
| PLS-DA (pixels, majority vote) | WL | 97 | 40 | 143 | 680 | 0.71 | 0.40 | 0.51 |
| PLS-DA (pixels, majority vote) | HL | 185 | 383 | 55 | 337 | 0.33 | 0.77 | 0.46 |
| PLS-DA (pixels, majority vote) | LL | 0 | 0 | 240 | 720 | 0.00 | 0.00 | 0.00 |
| PLS-DA (pixels, majority vote) | SH | 100 | 155 | 140 | 565 | 0.39 | 0.42 | 0.40 |
| ResNet-50 + SVM (whole plant) | WL | 10 | 1 | 2 | 35 | 0.91 | 0.83 | 0.87 |
| ResNet-50 + SVM (whole plant) | HL | 9 | 4 | 3 | 32 | 0.69 | 0.75 | 0.72 |
| ResNet-50 + SVM (whole plant) | LL | 3 | 3 | 9 | 33 | 0.50 | 0.25 | 0.33 |
| ResNet-50 + SVM (whole plant) | SH | 9 | 9 | 3 | 27 | 0.50 | 0.75 | 0.60 |
| ResNet-50 patches + softmax (plant vote) | WL | 60 | 8 | 0 | 172 | 0.88 | 1.00 | 0.94 |
| ResNet-50 patches + softmax (plant vote) | HL | 56 | 27 | 4 | 153 | 0.67 | 0.93 | 0.78 |
| ResNet-50 patches + softmax (plant vote) | LL | 10 | 14 | 50 | 166 | 0.42 | 0.17 | 0.24 |
| ResNet-50 patches + softmax (plant vote) | SH | 31 | 34 | 29 | 146 | 0.48 | 0.52 | 0.50 |
| ResNet-50 patches + softmax (patch level) | WL | 3067 | 810 | 568 | 7715 | 0.79 | 0.84 | 0.82 |
| ResNet-50 patches + softmax (patch level) | HL | 3055 | 1328 | 840 | 6937 | 0.70 | 0.78 | 0.74 |
| ResNet-50 patches + softmax (patch level) | LL | 535 | 1223 | 2055 | 8347 | 0.30 | 0.21 | 0.25 |
| ResNet-50 patches + softmax (patch level) | SH | 968 | 1174 | 1072 | 8946 | 0.45 | 0.47 | 0.46 |

### Per class, white

| Technique | Class | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| PLS-DA (whole plant) | WL | 11 | 10 | 1 | 26 | 0.52 | 0.92 | 0.67 |
| PLS-DA (whole plant) | HL | 6 | 5 | 6 | 31 | 0.55 | 0.50 | 0.52 |
| PLS-DA (whole plant) | LL | 0 | 0 | 12 | 36 | 0.00 | 0.00 | 0.00 |
| PLS-DA (whole plant) | SH | 9 | 7 | 3 | 29 | 0.56 | 0.75 | 0.64 |
| SVM (whole plant) | WL | 11 | 0 | 1 | 36 | 1.00 | 0.92 | 0.96 |
| SVM (whole plant) | HL | 9 | 5 | 3 | 31 | 0.64 | 0.75 | 0.69 |
| SVM (whole plant) | LL | 4 | 7 | 8 | 29 | 0.36 | 0.33 | 0.35 |
| SVM (whole plant) | SH | 8 | 4 | 4 | 32 | 0.67 | 0.67 | 0.67 |
| PLS-DA (pixels, majority vote) | WL | 180 | 99 | 60 | 621 | 0.65 | 0.75 | 0.69 |
| PLS-DA (pixels, majority vote) | HL | 220 | 155 | 20 | 565 | 0.59 | 0.92 | 0.72 |
| PLS-DA (pixels, majority vote) | LL | 0 | 0 | 240 | 720 | 0.00 | 0.00 | 0.00 |
| PLS-DA (pixels, majority vote) | SH | 160 | 146 | 80 | 574 | 0.52 | 0.67 | 0.59 |
| ResNet-50 + SVM (whole plant) | WL | 8 | 2 | 4 | 34 | 0.80 | 0.67 | 0.73 |
| ResNet-50 + SVM (whole plant) | HL | 7 | 9 | 5 | 27 | 0.44 | 0.58 | 0.50 |
| ResNet-50 + SVM (whole plant) | LL | 4 | 8 | 8 | 28 | 0.33 | 0.33 | 0.33 |
| ResNet-50 + SVM (whole plant) | SH | 5 | 5 | 7 | 31 | 0.50 | 0.42 | 0.45 |
| ResNet-50 patches + softmax (plant vote) | WL | 59 | 9 | 1 | 171 | 0.87 | 0.98 | 0.92 |
| ResNet-50 patches + softmax (plant vote) | HL | 59 | 24 | 1 | 156 | 0.71 | 0.98 | 0.83 |
| ResNet-50 patches + softmax (plant vote) | LL | 12 | 20 | 48 | 160 | 0.38 | 0.20 | 0.26 |
| ResNet-50 patches + softmax (plant vote) | SH | 30 | 27 | 30 | 153 | 0.53 | 0.50 | 0.51 |
| ResNet-50 patches + softmax (patch level) | WL | 2929 | 1032 | 706 | 7493 | 0.74 | 0.81 | 0.77 |
| ResNet-50 patches + softmax (patch level) | HL | 2787 | 1347 | 1108 | 6918 | 0.67 | 0.72 | 0.69 |
| ResNet-50 patches + softmax (patch level) | LL | 615 | 1332 | 1975 | 8238 | 0.32 | 0.24 | 0.27 |
| ResNet-50 patches + softmax (patch level) | SH | 1073 | 1045 | 967 | 9075 | 0.51 | 0.53 | 0.52 |
