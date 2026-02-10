import cv2

def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image from {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced, (3,3), 0)
    return blur
