# Dr Praveen's feedback on the results deck (16 Nov 2026)

Status: to do.

## Tuning slide (slide 8): rebuild as two tables

1. **Best hyperparameters**: the single combination found by Bayesian optimisation, one value each.
   - Patch size 64 px. State that 8, 16, 32, 64 and 224 were tested, with 64 in bold.
   - Stride, learning rate, number of epochs, optimiser, batch size and the other tuned settings.
   - No search ranges on the slide; the paper says "chosen by Bayesian optimisation".
2. **Results with that one combination**: overall accuracy, precision, recall and F1.

## Hyperparameter table details

- Title along the lines of "Hyperparameters obtained by Bayesian optimisation". Two columns: parameter, value.
- Rows: patch size, stride, learning rate, number of epochs, optimiser, loss function (cross-entropy), batch size.
- Learning-rate decay can be left out (his call; it was tuned, so consider keeping it for reproducibility).
- Later: one such table per technique (ResNet-50, vision transformer, SVM, ...).

## Results by stress duration: two tables

Rows: 4 h, day 1, day 2, day 3. Columns: precision, recall, F1, accuracy.
1. Four classes (WL, HL, LL, SH).
2. Healthy vs stressed.

## Calibration and test accuracy

Report calibration (training) accuracy alongside test accuracy; validation can be shown too.
Justify the 64 px choice on validation, not test (64 x 4 had the best mean validation score, 0.68).

## Patch size and stride search

- Use powers of two: patch sizes 8, 16, 32, 64, 128, 256 (replace 224). Strides 2, 4, and possibly 6, 8.
- Expected conclusion: 64 px is the best patch size. Report best, mean and median.
- Keep the patch/stride table (slide 9) but take it off the main deck: supplementary material for reviewers.
- Be ready to state the search ranges if the professor or a reviewer asks.

## Five-model comparison (slide 10)

- Rerun with 64 px fixed for both ResNet-50 and the ViT (it currently uses 32 px, stride 2).
- Do not set the ViT's settings by hand: give the ViT its own Bayesian optimisation over the same search
  (patch sizes 8, 16, 32, 64, 128, 256; strides; training settings), print the best combination, and
  test with it. Same for ResNet-50. This means script 09 needs a backbone option (ResNet-50 or ViT).
- The "Input" column on slide 10 says "32 px patches, stride 2" for the deep models. Change it to each
  model's own patch size and stride from its Bayesian optimisation (ResNet-50 and ViT listed separately).
- Promised Dr Praveen an update by 4 PM.

## Classic models: use pixels, not six summary numbers

Dr Praveen questioned the whole-plant features (mean and SD of 3 bands = 6 numbers per image): too much
information is thrown away. Pixel-level PLS-DA (5,000 random pixels per training image, 20 draws, plant
vote) already does better: 0.62 vs 0.50. Proposed: run SVM and gradient boosting at pixel level too, with
the same draws and plant vote, and use the pixel versions in the five-model comparison.

## Remove

- The laptop column: report the GPU run only.
- The per-plant patch row (P1, P2, P3 = 32, 64, 64): one general setting for all plants.
- The per-stress F1 rows (shade, low light).

## Before rebuilding: rerun on the GPU with 64 px fixed

```
git pull && PATCH=64 STRIDE=4 .venv/bin/python scripts/10_final_patch_model.py
nohup .venv/bin/python scripts/11_model_comparison.py > results/11_model_comparison.log 2>&1 &
```

Script 10 picks the best training settings within 64 px, stride 4 and tests all three plants with
them; its output fills both tables. Rerunning 11 updates the five-model comparison to 64 px too.
Then update slides 8 to 13 (the tuning slide, the patch table, the comparison slide, the pipeline
diagram and the appendix all mention 32 px, stride 2).

## Already answered in the meeting

- High light F1 0.84, white light F1 0.91 (GPU tuning run).
