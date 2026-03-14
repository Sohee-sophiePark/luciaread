# Development Cycle — DL-01 Medical Image Classification

## Phase 0 — Theory
- [ ] CNN architecture: conv layers, pooling, receptive field, why depth matters
- [ ] Transfer learning: why ImageNet features generalise to medical images
- [ ] Data augmentation: what to do and what NOT to do for medical images
- [ ] Calibration: reliability diagrams, ECE, temperature scaling
- [ ] Grad-CAM: gradient-weighted class activation mapping — the math

Output: notes in `notebooks/00_theory.md`

## Phase 1 — Data Setup
- [ ] Download Kaggle dataset (see `data/README.md`)
- [ ] Re-split validation set (original has only 16 images)
- [ ] Verify class distribution in each split
- [ ] Visualise sample images from each class
- [ ] Check for label noise / corrupted images

Output: `notebooks/01_data_setup.ipynb`

## Phase 2 — Data Pipeline
- [ ] PyTorch Dataset class for X-ray images
- [ ] DataLoader with appropriate batch size (32–64)
- [ ] Augmentation pipeline (albumentations, medical-safe)
- [ ] Normalisation (ImageNet stats for transfer learning)
- [ ] Visualise augmented samples — sanity check

Output: `src/dataset.py`, `notebooks/02_data_pipeline.ipynb`

## Phase 3 — Baseline CNN (From Scratch)
- [ ] Small CNN: 3 conv blocks → global avg pool → classifier
- [ ] Train for 20 epochs
- [ ] Learning rate: 1e-3 with cosine decay
- [ ] Log: train/val loss, accuracy, AUC
- [ ] Plot training curves
- [ ] Record baseline: val AUC, F1

Output: `notebooks/03_baseline_cnn.ipynb`

## Phase 4 — Transfer Learning
- [ ] Load EfficientNet-B0 (timm, ImageNet pretrained)
- [ ] Freeze all layers, train classifier head (5 epochs)
- [ ] Unfreeze last 2 blocks, fine-tune (10 epochs, lower LR: 1e-4)
- [ ] Compare vs. from-scratch baseline
- [ ] Try ResNet50 as second option

Output: `notebooks/04_transfer_learning.ipynb`

## Phase 5 — Grad-CAM
- [ ] Apply GradCAM to best model
- [ ] Generate heatmaps for 10 correct predictions (5 per class)
- [ ] Generate heatmaps for 10 misclassifications
- [ ] Overlay on original X-rays
- [ ] Sanity check: do heatmaps highlight lungs?

Output: `notebooks/05_gradcam.ipynb`

## Phase 6 — Calibration
- [ ] Plot reliability diagram for uncalibrated model
- [ ] Apply temperature scaling on val set
- [ ] Plot reliability diagram after calibration
- [ ] Report ECE before and after
- [ ] Choose operating threshold for clinical use case

Output: `notebooks/06_calibration.ipynb`

## Phase 7 — Production Code + Report
- [ ] Promote pipeline to `src/`: dataset.py, model.py, train.py, evaluate.py
- [ ] Unit tests for dataset and model
- [ ] Final report: methods, results, calibration analysis, Grad-CAM insights
- [ ] Save model checkpoint

Output: `src/`, `tests/`, `docs/REPORT.md`

## Definition of Done
- Val AUC significantly above from-scratch baseline
- ECE < 0.05 after calibration
- Grad-CAM heatmaps highlight clinically relevant regions
- All src code tested and ruff-clean
