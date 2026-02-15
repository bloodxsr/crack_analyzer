# Crack Analyzer (Training)

This project trains a binary crack-segmentation UNet using the Crack500-style dataset layout.

## Dataset layout

The training code expects paired files with this naming pattern:

- Images: `output/images/<name>.jpg|png|jpeg`
- Masks: `output/masks/<name>_mask.png`

Example:

- `output/images/IMG_1234.jpg`
- `output/masks/IMG_1234_mask.png`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m training.train_segmentation \
  --img-dir output/images \
  --mask-dir output/masks \
  --epochs 30 \
  --batch-size 4 \
  --lr 1e-4 \
  --val-split 0.1 \
  --save-path models/unet.pth
```

## Useful improvements already included

- Configurable training through CLI arguments (no hardcoded Windows paths)
- Reproducible dataset split with fixed random seed
- Optional validation split and best-model checkpointing
- Safer dataset loading with directory checks and clearer error messages
