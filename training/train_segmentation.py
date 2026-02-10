import torch
from torch.utils.data import DataLoader
from data.dataset import CrackDataset
from models.unet import UNet
import torch.nn as nn
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CrackDataset(
    img_dir="E:\\CRACK_AI\\output\\images",
    mask_dir="E:\\CRACK_AI\\output\\masks"
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCELoss()

for epoch in range(30):
    model.train()
    epoch_loss = 0

    for imgs, masks in tqdm(loader):
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(loader)}")

torch.save(model.state_dict(), "models/crack_unet.pth")
