import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data.dataset import CrackDataset
from models.unet import UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet on crack segmentation masks")
    parser.add_argument("--img-dir", default="output/images", help="Directory with input images")
    parser.add_argument("--mask-dir", default="output/masks", help="Directory with mask images")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default="models/unet.pth")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for imgs, masks in tqdm(loader, leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / max(1, len(loader))


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = CrackDataset(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        image_size=args.image_size,
    )

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    if train_size <= 0:
        raise ValueError("Validation split is too large and leaves no training samples")

    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False) if val_size > 0 else None

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer=optimizer)

        if val_loader is not None:
            val_loss = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
            print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
        else:
            print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f}")
            torch.save(model.state_dict(), save_path)

    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
