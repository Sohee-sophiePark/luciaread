# Design — DL-01 Medical Image Classification (Chest X-Ray)

## Problem Framing

Binary classification: Normal vs. Pneumonia from chest X-ray images.

In medical imaging, the cost of errors is asymmetric and severe:
- False negative (miss pneumonia): patient goes untreated — highest cost
- False positive (flag healthy as pneumonia): unnecessary treatment, anxiety, cost

The model must be calibrated: "90% confidence" must mean correct 90% of the time. An uncalibrated model in a clinical setting is dangerous regardless of its AUC.

## Architecture Decisions

### From-scratch CNN vs. Transfer Learning
Both are built and compared:

**From-scratch (Phase 3):** Small CNN (3–4 conv blocks). Establishes a lower bound. Shows why transfer learning is worth it on limited medical data.

**Transfer learning (Phase 4):** ResNet50 or EfficientNet-B0, pretrained on ImageNet. Why it works: ImageNet features (edges, textures, shapes) generalise to medical images even though the domains differ. Fine-tune last 2 blocks + classifier head.

`timm` library for model selection — 600+ pretrained models with consistent API.

### Data Augmentation
Medical imaging augmentation must be domain-appropriate:
- **Yes:** horizontal flip, rotation (±10°), brightness/contrast jitter
- **No:** vertical flip (lungs have orientation), aggressive colour jitter, cutout
- Use `albumentations` (faster and more flexible than torchvision transforms)

### Validation Strategy
The original dataset has only 16 validation images — unusable. Re-split:
- Train: 70% of original train set
- Val: 30% of original train set (stratified)
- Test: original test set (untouched until final evaluation)

### Calibration
Post-hoc calibration on val set:
- Temperature scaling (single parameter — preferred for neural nets)
- Plot reliability diagram before and after
- Report Expected Calibration Error (ECE)

### Grad-CAM
`grad-cam` library (jacobgil/pytorch-grad-cam). Apply to the last conv layer.
Overlay heatmap on original X-ray to show which regions drove the prediction.
Sanity check: heatmap should highlight lung regions, not image borders or labels.

## Evaluation Metrics
- **Primary:** AUC-ROC (balanced classes after re-split), F1
- **Calibration:** ECE, reliability diagram
- **Clinical framing:** sensitivity (recall) and specificity at operating threshold

## Trade-offs Considered

| Decision | Chosen | Rejected | Reason |
|----------|--------|----------|--------|
| Framework | PyTorch | TensorFlow/Keras | Ecosystem, research-standard, timm |
| Base model | EfficientNet-B0 | VGG16 | Better accuracy/parameter ratio |
| Augmentation | albumentations | torchvision | Speed, flexibility, medical-safe transforms |
| Calibration | Temperature scaling | Platt scaling | Simpler, fewer parameters, works well for NNs |
| Explainability | Grad-CAM | LIME | LIME too slow for images; Grad-CAM is standard |
