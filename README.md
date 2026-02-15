# Crack Analyzer

A deep-learning–based crack analysis system that detects cracks in surface images using **UNet-based semantic segmentation**, with optional post-processing for feature extraction and severity assessment.

---

## Project Overview

This project focuses on **crack detection and analysis** in images of concrete, roads, walls, or similar surfaces.

The system works in three conceptual stages:

1. **Crack Segmentation (Deep Learning)**
2. **Feature Extraction (Image Processing)**
3. **Logic-Based Assessment (Rule Engine)**

Currently, the core **segmentation model is fully implemented and trained**.

---

## How It Works (High Level)

- Input Image  
  ↓  
- Preprocessing (grayscale, resize, normalize)  
  ↓  
- UNet Model  
  ↓  
- Predicted Crack Mask  
  ↓  
- (Optional) Feature Extraction  
  ↓  
- (Optional) Severity Assessment


## Useful improvements already included

- Configurable training through CLI arguments (no hardcoded Windows paths)
- Reproducible dataset split with fixed random seed
- Optional validation split and best-model checkpointing
- Safer dataset loading with directory checks and clearer error messages


---

## Dataset Format

Each image **must have a corresponding mask**.

### Images
- Grayscale or RGB (converted internally)
- Any resolution (resized during loading)

### Masks
- Binary PNG images
- White (255) = crack
- Black (0) = background
- Same filename as image with `_mask.png` or `.png`

Example:
image_001.jpg
image_001.png ← mask


---

## Model Architecture

- **Model**: UNet (Encoder–Decoder with skip connections)
- **Input**: 1 × 512 × 512 grayscale image
- **Output**: 1 × 512 × 512 probability mask
- **Activation**: Sigmoid (binary segmentation)

The model predicts a probability for each pixel being part of a crack.

---

## Training the Model

### Install dependencies
```bash
pip install -r requirements.txt

for training:-

python -m training.train_segmentation \
  --img-dir output/images \
  --mask-dir output/masks \
  --epochs 30 \
  --batch-size 4 \
  --lr 1e-4 \
  --val-split 0.1 \
  --save-path models/unet.pth

supported version : python - 3.10.11
