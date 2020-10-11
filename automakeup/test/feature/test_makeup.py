import importlib.resources as pkg_resources
import unittest

import cv2
import numpy as np

from automakeup.feature.makeup import EyeshadowShapeExtractor, EyeshadowColorExtractor, LipstickColorExtractor
from faceparsing import FaceParser
from imagine.color import conversion
from imagine.shape.segment import ParsingSegmenter

parser = FaceParser()


class EyeshadowShapeExtractorTestCase(unittest.TestCase):
    extractor = EyeshadowShapeExtractor()

    def test_eyeshadow_shape_extractor_returns_correct_shape(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        segmented = ParsingSegmenter(parser, parts_map={"skin": 1, "l_eye": 2, "r_eye": 2})(img)
        eyeshadow_mask = self.extractor.extract(img, segmented == 1, segmented == 2)
        self.assertEqual(eyeshadow_mask.shape[:2], img.shape[:2])

    def test_eyeshadow_shape_extractor_returns_correct_type(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        segmented = ParsingSegmenter(parser, parts_map={"skin": 1, "l_eye": 2, "r_eye": 2})(img)
        eyeshadow_mask = self.extractor.extract(img, segmented == 1, segmented == 2)
        self.assertTrue(np.issubdtype(eyeshadow_mask.dtype, np.bool))

    def test_eyeshadow_shape_extractor_can_find_the_eyeshadow(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        segmented = ParsingSegmenter(parser, parts_map={"skin": 1, "l_eye": 2, "r_eye": 2})(img)
        eyeshadow_mask = self.extractor.extract(img, segmented == 1, segmented == 2)
        self.assertTrue(np.any(eyeshadow_mask))

    def test_eyeshadow_shape_extractor_returns_empty_mask_for_no_eyes(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        skin_mask = np.ones(img.shape[:2], dtype=np.bool)
        eyes_mask = np.zeros(img.shape[:2], dtype=np.bool)
        eyeshadow_mask = self.extractor.extract(img, skin_mask, eyes_mask)
        self.assertTrue((~eyeshadow_mask).all())


class EyeshadowColorExtractorTestCase(unittest.TestCase):
    extractor = EyeshadowColorExtractor()

    def test_extract_returns_correct_shape(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        segmented = ParsingSegmenter(parser, parts_map={"skin": 1, "l_eye": 2, "r_eye": 2})(img)
        colors = self.extractor.extract(img, segmented == 1, segmented == 2)
        self.assertEqual(colors.shape, (3, 3))

    def test_extract_returns_correct_type(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        segmented = ParsingSegmenter(parser, parts_map={"skin": 1, "l_eye": 2, "r_eye": 2})(img)
        colors = self.extractor.extract(img, segmented == 1, segmented == 2)
        self.assertTrue(np.issubdtype(colors.dtype, np.integer))

    def test_extract_returns_none_with_no_eyes(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        segmented = ParsingSegmenter(parser, parts_map={"skin": 1})(img)
        colors = self.extractor.extract(img, segmented == 1, np.zeros(img.shape[:2], dtype=np.bool))
        self.assertEqual(colors, None)


class LipstickColorExtractorTestCase(unittest.TestCase):
    extractor = LipstickColorExtractor()

    def test_extract_returns_correct_shape(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        lips_mask = ParsingSegmenter(parser, parts_map={"u_lip": 1, "l_lip": 1})(img) == 1
        colors = self.extractor.extract(img, lips_mask)
        self.assertEqual(colors.shape, (1, 3))

    def test_extract_returns_correct_type(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        lips_mask = ParsingSegmenter(parser, parts_map={"u_lip": 1, "l_lip": 1})(img) == 1
        colors = self.extractor.extract(img, lips_mask)
        self.assertTrue(np.issubdtype(colors.dtype, np.integer))

    def test_extract_returns_none_with_no_lips(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        colors = self.extractor.extract(img, np.zeros(img.shape[:2], dtype=np.bool))
        self.assertEqual(colors, None)


if __name__ == '__main__':
    unittest.main()
