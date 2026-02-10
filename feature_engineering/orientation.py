import numpy as np
import cv2

def crack_orientation(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return "undefined"

    coords = np.column_stack((xs, ys))
    mean = np.mean(coords, axis=0)
    centered = coords - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(cov)

    principal = eigvecs[:, np.argmax(eigvals)]
    angle = np.degrees(np.arctan2(principal[1], principal[0]))

    if abs(angle) < 30:
        return "horizontal"
    elif abs(angle) > 60:
        return "vertical"
    else:
        return "diagonal"
