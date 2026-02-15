from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, image_size=512):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = int(image_size)

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.img_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory does not exist: {self.mask_dir}")

        self.samples = []
        for img_path in sorted(self.img_dir.iterdir()):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue

            mask_name = f"{img_path.stem}_mask.png"
            mask_path = self.mask_dir / mask_name
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        if not self.samples:
            raise RuntimeError(
                f"No valid image-mask pairs found in {self.img_dir} and {self.mask_dir}"
            )

        print(f"Loaded {len(self.samples)} valid image-mask pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {mask_path}")

        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        mask = (mask > 0).float()

        return image, mask
