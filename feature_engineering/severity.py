import cv2
import numpy as np

def crack_width(mask):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_w = dist.max()*2
    avg_w = dist[dist>0].mean()*2
    return avg_w, max_w

def severity_score(length, width, density):
    score = (length*0.4) + (width*0.4) + (density*0.2)
    if score < 50:
        return "low"
    elif score < 150:
        return "medium"
    else:
        return "high"
