import cv2

from imagine.functional.functional import ImageOperation


class ColorConverter(ImageOperation):
    """
    Convert images from one color space to another

    Attributes:
        mode - cv2 color conversion mode
    """

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def perform(self, img, **kwargs):
        """
        Perform conversion

        Args:
            img: numpy array of shape (height, width, channels) in uint8 with image data

        Returns:
            numpy array of the same shape as img with converted colors
        """
        return cv2.cvtColor(img, self.mode)


RgbToBgr = ColorConverter(cv2.COLOR_RGB2BGR)
BgrToRgb = ColorConverter(cv2.COLOR_BGR2RGB)

RgbToLab = ColorConverter(cv2.COLOR_RGB2LAB)
LabToRgb = ColorConverter(cv2.COLOR_Lab2RGB)


