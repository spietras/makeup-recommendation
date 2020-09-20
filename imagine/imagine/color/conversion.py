import cv2
import numpy as np


class ColorConverter:
    def __init__(self, mode):
        """
        Args:
            mode: cv2 color conversion mode
        """
        super().__init__()
        self.mode = mode

    def convert(self, imgs):
        """
        Convert values to another colorspace

        Args:
            imgs: numpy array of shape (N, width, height, 3) or (width, height, 3) or (N, 3) or (3,)

        Returns:
            numpy array of the same shape as img
        """

        if imgs.ndim < 4:
            return self._convert_single(imgs)
        return np.stack([self._convert_single(i) for i in imgs])

    def _convert_single(self, img):
        return cv2.cvtColor(np.array(img, ndmin=3), self.mode).reshape(img.shape)


RgbToLab = ColorConverter(cv2.COLOR_RGB2LAB)
LabToRgb = ColorConverter(cv2.COLOR_Lab2RGB)
