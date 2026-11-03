# ResNet-50 layer comparison (+ linear SVM), choy sum

| Layers | Normalisation | Accuracy | F1 (macro) | Accuracy at 4 h |
|---|---|---|---|---|
| layer1 | raw | 52% | 0.52 | 58% |
| layer2 | raw | 56% | 0.55 | 50% |
| layer3 | raw | 54% | 0.54 | 42% |
| layer4 | raw | 58% | 0.55 | 58% |
| layer1+layer2 | raw | 56% | 0.54 | 58% |
| layer2+layer3 | raw | 54% | 0.54 | 50% |
| layer3+layer4 | raw | 71% | 0.70 | 75% |
| layer2+layer3+layer4 | raw | 65% | 0.64 | 67% |
| layer1+layer2+layer3+layer4 | raw | 60% | 0.59 | 58% |
| layer1 | white | 38% | 0.39 | 50% |
| layer2 | white | 56% | 0.56 | 58% |
| layer3 | white | 54% | 0.55 | 50% |
| layer4 | white | 56% | 0.57 | 50% |
| layer1+layer2 | white | 42% | 0.41 | 58% |
| layer2+layer3 | white | 65% | 0.64 | 58% |
| layer3+layer4 | white | 60% | 0.60 | 58% |
| layer2+layer3+layer4 | white | 69% | 0.68 | 58% |
| layer1+layer2+layer3+layer4 | white | 65% | 0.64 | 58% |
