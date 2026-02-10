import cv2
from feature_engineering.skeleton import skeletonize
from feature_engineering.geometry import crack_length, crack_area, crack_density
from feature_engineering.orientation import crack_orientation
from feature_engineering.severity import crack_width, severity_score

def extract_features(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load image from {mask_path}")
    mask = (mask > 0).astype("uint8")*255

    skel = skeletonize(mask)

    length = crack_length(skel)
    area = crack_area(mask)
    density = crack_density(mask)
    orientation = crack_orientation(mask)
    avg_w, max_w = crack_width(mask)
    severity = severity_score(length, avg_w, density)

    return {
        "length": float(length),
        "area": float(area),
        "density": float(density),
        "orientation": orientation,
        "avg_width": float(avg_w),
        "max_width": float(max_w),
        "severity": severity
    }
