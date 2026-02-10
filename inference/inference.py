import cv2, torch
from models.unet import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
model.load_state_dict(torch.load("models/crack_unet.pth", map_location=device))
model.eval()

img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
assert img is not None, "Failed to load image"
img = cv2.resize(img, (512,512))/255.0

t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(t)[0][0].cpu().numpy()

mask = (pred > 0.5).astype("uint8")*255
cv2.imwrite("E:\\CRACK_AI\\output\\images\\test.png", mask)
