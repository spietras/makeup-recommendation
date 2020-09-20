import unittest

import cv2
import numpy as np

from imagine.color import conversion


class ConversionTestCase(unittest.TestCase):
    converter = conversion.RgbToLab

    def test_converter_succeeds_for_single_image(self):
        img = np.random.randint(0, 256, size=(30, 30, 3), dtype=np.uint8)
        converted = self.converter.convert(img)
        self.assertEquals(converted.shape, img.shape)

    def test_converter_succeeds_for_multiple_pixels(self):
        img = np.random.randint(0, 256, size=(30, 3), dtype=np.uint8)
        converted = self.converter.convert(img)
        self.assertEquals(converted.shape, img.shape)

    def test_converter_succeeds_for_single_pixel(self):
        img = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
        converted = self.converter.convert(img)
        self.assertEquals(converted.shape, img.shape)

    def test_converter_succeeds_for_batch(self):
        img = np.random.randint(0, 256, size=(2, 30, 30, 3), dtype=np.uint8)
        converted = self.converter.convert(img)
        self.assertEquals(converted.shape, img.shape)

    def test_converter_fails_for_wrong_data_type(self):
        img = np.random.rand(30, 30, 3)
        self.assertRaises(cv2.error, self.converter.convert, img)

    def test_converter_fails_for_wrong_channels(self):
        img = np.random.randint(0, 256, size=(30, 30, 1), dtype=np.uint8)
        self.assertRaises(cv2.error, self.converter.convert, img)


if __name__ == '__main__':
    unittest.main()
