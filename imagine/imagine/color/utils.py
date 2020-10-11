import cv2
import numpy as np

from imagine.functional.functional import ImageOperation


def generate_distinct_colors(k):
    """
    Generate k distinct colors

    Returns:
        numpy array of shape (k, 3) with generated colors
    """
    if k == 0:
        return np.array([[]], dtype=np.uint8)
    colors = np.array([[[int(x * 179 / k), 128, 128] for x in range(k)]], dtype=np.uint8)
    return np.atleast_2d(cv2.cvtColor(colors, cv2.COLOR_HSV2RGB).squeeze())


def recolor(img, mask, color, alpha):
    """
    Apply color on image

    Args:
        img: numpy array of shape (height, width, channels) in any type with image data
        mask: numpy array of shape (height, width) with True in places of pixels to recolor
        color: numpy array of shape (channels,) or tuple of channels length with color data
        alpha: number between 0.0 and 1.0 controlling color transparency (1.0 means fully transparent)

    Returns:
        numpy array of the same shape as img with recolored pixels
    """
    img = np.copy(img)
    img[mask] = alpha * img[mask] + (1 - alpha) * np.asarray(color)
    return img


# Functional Interface

class Recolor(ImageOperation):
    """Apply color on image"""

    def __init__(self, color, alpha):
        """
        Args:
            color: numpy array of shape (channels,) or tuple of channels length with color data
            alpha: number between 0.0 and 1.0 controlling color transparency (1.0 means fully transparent)
        """
        super().__init__()
        self.color = color
        self.alpha = alpha

    def perform(self, img, mask=None, **kwargs):
        """
        Args:
            img: numpy array of shape (height, width, channels) in any type with image data
            mask: numpy array of shape (height, width) with True in places of pixels to recolor

        Returns:
            numpy array of the same shape as img with recolored pixels
        """
        if mask is None:
            mask = np.zeros(img.shape, dtype=bool)
        return recolor(img, mask, self.color, self.alpha)
