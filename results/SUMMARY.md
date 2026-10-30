# First results: choy sum light stress

Data: 36 three-band images of choy sum = 3 light conditions (HL, LL, WL) × 4 time points (4, 24, 48, 72 h) × 3 plants.
Evaluation: leave-one-plant-out (train on two plants, test on the third, repeat for each plant) unless stated.
Chance level: 33% for light condition, 25% for time point. Each time point has only 9 test images, so one image = 11 percentage points.

## Results

| Approach | Light accuracy | Light accuracy at 4 h | Light accuracy, unseen time point | Hours error (MAE) | Hours exactly right |
|---|---|---|---|---|---|
| Averaged bands, raw values + SVM | 83% | 100% | 61% | 16 h | 44% |
| Averaged bands, raw values + PLS-DA | 69% | 56% | 50% | 19 h | 31% |
| Averaged bands, brightness removed + SVM | 50% | 89% | 31% | 14 h | 44% |
| Pretrained CNN (ResNet-50) features + SVM | 64% | 56% | 61% | 18 h | 22% |
| Two-output model on CNN features | 81% (78 to 83% over 5 seeds) | 89% | not run | 16 h | 39 to 44% |

"Unseen time point" holds out every image from one time point, so the model is tested on an imaging session it never saw.

## What this tells us

1. **The light condition can be detected at 4 hours, above chance.** The averaged-band SVM got all 9 four-hour images right and the two-output model got 8 of 9. With 9 test images this is encouraging, not conclusive.
2. **Most of the signal is in overall brightness, and part of it may be the imaging session.** The three plants in each condition and time point have almost identical band averages, while averages jump between time points. Removing brightness drops accuracy to 50%, and testing on an unseen time point drops it to about 60%. The model may be learning session-level differences (lighting, exposure, scaling) as well as the plant's response.
3. **CNN features do not beat averaged bands yet.** With an SVM they reach 64%. A small two-output network on the same features reaches 81%, close to the best averaged-band model. On an unseen time point, CNN features + SVM score 61%, the same as raw averaged bands.
4. **Predicting hours of stress is weak so far.** Errors are around 16 hours and about 40% exact, against 25% chance. Four time points and three plants are not enough to learn a time trend.
5. **The chemical map works as a pipeline but is not yet reliable.** On the 4-hour high-light plant, the pixel-level PLS-DA marks the leaf blades as stressed. It fails for low light (0% of pixels correct) because three raw bands with two PLS components cannot separate low light from white light at the pixel level. See `04_chemical_map_4h.png`.

## Questions for Dr Praveen

1. Are P1 to P3 the same physical plants at every time point? This decides whether the CNN-LSTM can use real sequences.
2. How was reflectance scaled to 8-bit: one scale for all images, or per image or session? Was a white reference imaged in each session? This decides whether the brightness differences are biological.
3. Is WL the control condition, and does "C" in the file names mean control?
4. Which three bands (wavelengths) are in the images?

## Next steps

- Normalise brightness per session (using a reference if one exists) and rerun, to separate session effects from plant response.
- Add the CNN-LSTM once sequences are confirmed, and the CNN-vision transformer.
- Improve the chemical map: per-pixel normalisation, more PLS components, and CNN saliency maps to compare against.
- Run the same pipeline on kailan and Arabidopsis when they arrive.

## How to reproduce

```
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/01_baseline_bands.py
.venv/bin/python scripts/02_cnn_features_svm.py
.venv/bin/python scripts/03_two_output_model.py
.venv/bin/python scripts/04_chemical_map.py
```
