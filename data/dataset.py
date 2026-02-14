import os
import cv2
import torch
from torch.utils.data import Dataset

class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Collect only valid image–mask pairs
        self.samples = []

        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            name = os.path.splitext(img_name)[0]
            mask_name = f"{name}_mask.png"
            mask_path = os.path.join(mask_dir, mask_name)

            if os.path.exists(mask_path):
                self.samples.append((img_name, mask_name))

        if len(self.samples) == 0:
            raise RuntimeError("❌ No valid image–mask pairs found")

        print(f"✅ Loaded {len(self.samples)} valid image–mask pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, mask_name = self.samples[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError("Failed to load image or mask")
        
        image = cv2.resize(image, (512, 512))
        mask  = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        mask  = torch.from_numpy(mask).float().unsqueeze(0)
        mask  = (mask > 0).float()
        
        return image, mask

