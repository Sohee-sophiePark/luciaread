# Tech Stack — DL-01 Medical Image Classification

## Core Dependencies

### PyTorch `>=2.2` + torchvision `>=0.17`
Framework + standard image transforms and pretrained model loading.
Why PyTorch over TensorFlow: research-standard, dynamic computation graph, better debugging, `timm` ecosystem.

### timm `>=0.9`
`pytorch-image-models` — 600+ pretrained vision models with a consistent API. Used to load EfficientNet-B0 / ResNet50 with ImageNet weights.
`timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)`

### albumentations `>=1.4`
Fast image augmentation library. Medical-safe transform pipeline.
Why over torchvision transforms: 10–100x faster (NumPy/OpenCV backend), more augmentation types, composable with `A.Compose`.

### grad-cam `>=1.4` (pytorch-grad-cam)
Gradient-weighted Class Activation Mapping. Visualises which regions of the image influenced the model's prediction.
`pip install grad-cam` → `from pytorch_grad_cam import GradCAM`

### Pillow `>=10.0`
Image loading and basic manipulation.

### matplotlib `>=3.8`
Visualisation: training curves, confusion matrix, reliability diagrams, Grad-CAM overlays.

### scikit-learn `>=1.4`
Calibration (temperature scaling implemented manually, but sklearn used for metrics: AUC, F1, classification report).

## Dev Dependencies

### jupyter + ipykernel
Notebooks for EDA and experiment tracking.

### pytest `>=8.0`
Unit tests for dataset class, transforms, model forward pass.

### ruff `>=0.4`
Linting and formatting.

## Hardware Notes
- CPU training is feasible for the Kaggle dataset (5,856 images) but slow
- Recommended: Apple Silicon MPS (`device = torch.device("mps")`) or CUDA
- EfficientNet-B0 fine-tuning: ~10 min on M-series Mac

## Python Version
3.11
