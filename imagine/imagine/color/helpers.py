import cv2
import numpy as np


def generate_distinct_colors(k):
    if k == 0:
        return np.array([[]], dtype=np.uint8)
    colors = np.array([[[int(x * 179 / k), 128, 128] for x in range(k)]], dtype=np.uint8)
    return np.atleast_2d(cv2.cvtColor(colors, cv2.COLOR_HSV2RGB).squeeze())


def recolor(img, mask, color, alpha):
    img = np.copy(img)
    img[mask] = alpha * img[mask] + (1 - alpha) * np.asarray(color)
    return img
