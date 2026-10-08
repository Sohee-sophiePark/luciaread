# Data

Source: Kermany D, Zhang K, Goldbaum M (2018), "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray
Images for Classification", Mendeley Data V2, doi:10.17632/rscbjbr9sj.2, licensed CC BY 4.0.

Raw images are not in the repo. Only the split manifest (`splits/manifest.csv`) and the demo samples
(`../samples/`) are committed.

## Download (about 7 GB)

```bash
mkdir -p data/raw/cxr data/raw/oct
curl -fL -o data/raw/cxr/ChestXRay2017.zip "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded"
curl -fL -o data/raw/oct/OCT2017.tar.gz "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/5699a1d8-d1b6-45db-bb92-b61051445347/file_downloaded"
(cd data/raw/cxr && unzip -q ChestXRay2017.zip 'chest_xray/*.jpeg')
(cd data/raw/oct && tar -xzf OCT2017.tar.gz --exclude='._*')
```

Expected sizes: `ChestXRay2017.zip` 1,235,512,464 bytes; `OCT2017.tar.gz` 5,793,183,169 bytes.

## Patient-level split

The official folders share patients between train and test (OCT: 566 of 633 test patients also appear in
train; chest X-ray: 170 pneumonia patient IDs appear in both). `make split` pools all images and re-splits
70/15/15 by patient ID parsed from the filename, stratified by each patient's majority label. OCT is sampled
to 2,000 / 300 / 500 images per class for train / val / test. A test asserts that no patient appears in two
splits.

| Modality | Train | Val | Test |
|---|---|---|---|
| Chest X-ray (NORMAL / PNEUMONIA) | 1,102 / 3,036 | 244 / 636 | 237 / 601 |
| Retinal OCT (CNV, DME, DRUSEN, NORMAL) | 2,000 each | 300 each | 500 each |
