# DL-01 — Medical Image Classification (Chest X-Ray)

> CNN transfer learning for pneumonia detection. Grad-CAM explainability. Calibration that actually matters.

## Problem
Pneumonia kills 2.5 million people per year. Radiologist shortages in developing countries mean AI-assisted diagnosis can save lives — but only if the model is calibrated, explainable, and honest about its uncertainty. A model that says 90% confidence should be right 90% of the time.

## Datasets
**Starter:** Kaggle Chest X-Ray Images (Pneumonia) — 5,856 images, binary labels (Normal / Pneumonia).
Download: [kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
License: CC BY 4.0. Note: re-split val set (only 16 images in original).

**Advanced:** NIH ChestX-ray14 — 112,120 images, 14 disease labels.
Download: [kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data)
License: NIH public data, no restrictions. Note: labels are NLP-extracted, not radiologist-verified.

## Deliverables
- Trained CNN (from-scratch vs. fine-tuned ResNet/EfficientNet comparison)
- Grad-CAM visualiser (which pixels drove the prediction)
- Confidence calibration report (reliability diagrams, ECE score)

## Stack
Python 3.11 · PyTorch · torchvision · timm · albumentations · grad-cam · matplotlib

## How to Run
```bash
uv sync --dev
uv run jupyter notebook notebooks/
```

## Status
- [ ] Theory session (CNNs, transfer learning, calibration)
- [ ] Data exploration + re-splitting val set
- [ ] Baseline (from-scratch small CNN)
- [ ] Transfer learning (ResNet50 / EfficientNet-B0)
- [ ] Data augmentation
- [ ] Grad-CAM integration
- [ ] Calibration analysis
- [ ] Final report
