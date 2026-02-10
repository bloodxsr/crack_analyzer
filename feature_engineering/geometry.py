import cv2
import numpy as np

def crack_length(skeleton):
    return cv2.countNonZero(skeleton)

def crack_area(mask):
    return cv2.countNonZero(mask)

def crack_density(mask):
    h, w = mask.shape
    return cv2.countNonZero(mask) / (h*w)
