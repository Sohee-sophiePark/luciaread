# Data — DL-01 Medical Image Classification

Data files are gitignored. Download manually before running notebooks.

## Starter Dataset: Kaggle Chest X-Ray Images (Pneumonia)

**Source:** [kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
**License:** CC BY 4.0
**Size:** ~1.2 GB — images in JPEG format

### Download via Kaggle CLI
```bash
kaggle datasets download paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

### Expected Structure After Unzip
```
data/chest_xray/
    train/
        NORMAL/     (~1,341 images)
        PNEUMONIA/  (~3,875 images)
    val/
        NORMAL/     (8 images — re-split this)
        PNEUMONIA/  (8 images — re-split this)
    test/
        NORMAL/     (~234 images)
        PNEUMONIA/  (~390 images)
```

**Important:** Re-split the validation set. The original val set has only 16 images — insufficient for evaluation.
Use `notebooks/01_data_setup.ipynb` to regenerate a proper 70/30 train/val split from the training data.

## Advanced Dataset: NIH ChestX-ray14

**Source:** [kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data)
**License:** NIH public data — no restrictions
**Size:** ~42 GB — 112,120 images
**Note:** Labels are NLP-extracted from radiology reports, not radiologist-verified. Expected accuracy >90%.

Start with the Kaggle dataset. Move to NIH only when the basic pipeline is working.
