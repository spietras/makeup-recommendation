import cv2

from imagine.functional.functional import ImageOperation


class ColorConverter(ImageOperation):
    def __init__(self, mode):
        """
        Args:
            mode: cv2 color conversion mode
        """
        super().__init__()
        self.mode = mode

    def perform(self, img, **kwargs):
        return cv2.cvtColor(img, self.mode)


RgbToBgr = ColorConverter(cv2.COLOR_RGB2BGR)
BgrToRgb = ColorConverter(cv2.COLOR_BGR2RGB)

RgbToLab = ColorConverter(cv2.COLOR_RGB2LAB)
LabToRgb = ColorConverter(cv2.COLOR_Lab2RGB)


