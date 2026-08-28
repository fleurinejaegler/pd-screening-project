"""
Classical (non-deep-learning) feature extraction for spiral/wave drawings.

These are hand-crafted features meant to capture the visible tremor Parkinson's produces
in fine motor control: shaky, uneven line thickness and a wobbly contour instead of a smooth
one. Shared by src/drawings_pipeline.py (training) and src/app.py (live prediction on a
user's drawing), so both use *exactly* the same feature definitions.
"""

from __future__ import annotations

import cv2
import numpy as np

IMG_SIZE = 256  # every image is resized to IMG_SIZE x IMG_SIZE before feature extraction


def load_and_binarize(image_path_or_array) -> np.ndarray:
    """Load an image (path or already-loaded array), resize, and binarize (ink vs. background)."""
    if isinstance(image_path_or_array, (str, bytes)):
        img = cv2.imread(str(image_path_or_array), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path_or_array}")
    else:
        img = image_path_or_array
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Otsu thresholding: works whether the drawing is dark-on-light or light-on-dark scans.
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def extract_features(image_path_or_array) -> dict[str, float]:
    """Turn one spiral/wave image into a small vector of tremor-related numbers."""
    binary = load_and_binarize(image_path_or_array)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        # Blank/unreadable image -- return zeros rather than crashing.
        return _empty_features()

    # Use the largest contour (the drawn line itself).
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, closed=False)
    area = cv2.contourArea(contour)

    # 1) Contour smoothness: ratio of perimeter^2 to area. A shaky, tremor-heavy line has a
    #    much longer perimeter relative to the area it encloses than a smooth one.
    smoothness = (perimeter**2 / area) if area > 0 else 0.0

    # 2) Stroke width variance: distance-transform gives, at each ink pixel, the distance to
    #    the nearest background pixel -- roughly half the local stroke width. A shaky hand
    #    produces uneven pressure/width, so its variance is higher.
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_widths = dist_transform[binary > 0]
    stroke_width_mean = float(stroke_widths.mean()) if stroke_widths.size else 0.0
    stroke_width_std = float(stroke_widths.std()) if stroke_widths.size else 0.0

    # 3) Tremor-like direction changes: walk along the contour and count how often the local
    #    direction reverses sign -- a smooth spiral/wave has few reversals, a tremor-heavy one
    #    has many small back-and-forth wobbles.
    direction_changes = _count_direction_changes(contour)

    # 4) Number of ink pixels (roughly, how much was drawn) -- not tremor-specific by itself,
    #    but helps normalize the other features across drawings of different sizes/pressure.
    ink_pixel_count = int((binary > 0).sum())

    return {
        "contour_smoothness": float(smoothness),
        "stroke_width_mean": stroke_width_mean,
        "stroke_width_std": stroke_width_std,
        "direction_changes": float(direction_changes),
        "ink_pixel_count": float(ink_pixel_count),
        "perimeter": float(perimeter),
        "area": float(area),
    }


def _count_direction_changes(contour: np.ndarray, step: int = 5) -> int:
    points = contour.reshape(-1, 2)
    if len(points) < 2 * step:
        return 0

    angles = []
    for i in range(0, len(points) - step, step):
        dx, dy = points[i + step] - points[i]
        angles.append(np.arctan2(dy, dx))

    changes = 0
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i - 1]
        # wrap to [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        if abs(diff) > np.pi / 4:  # a "sharp" wobble, not just gradual curvature
            changes += 1
    return changes


def _empty_features() -> dict[str, float]:
    return {
        "contour_smoothness": 0.0,
        "stroke_width_mean": 0.0,
        "stroke_width_std": 0.0,
        "direction_changes": 0.0,
        "ink_pixel_count": 0.0,
        "perimeter": 0.0,
        "area": 0.0,
    }


FEATURE_NAMES = list(_empty_features().keys())
