import importlib.resources as pkg_resources
import unittest

import cv2
import dlib
import numpy as np

from automakeup import dlib_predictor_path
from automakeup.face.bounding import DlibBoundingBoxFinder
from automakeup.face.extract import SimpleFaceExtractor, AligningDlibFaceExtractor
from imagine.color import conversion
from imagine.shape.figures import Rect


class SimpleFaceExtractorTestCase(unittest.TestCase):

    def test_simple_face_extractor_returns_correct_shape(self):
        img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        size = 512
        extractor = SimpleFaceExtractor(size)
        face = extractor.extract(img, Rect(10, 90, 10, 90))
        self.assertEqual(face.shape, (512, 512, 3))

    def test_simple_face_extractor_works_with_bounding_box_bigger_than_image(self):
        img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        size = 512
        extractor = SimpleFaceExtractor(size)
        face = extractor.extract(img, Rect(-10, 110, -10, 110))
        self.assertEqual(face.shape, (512, 512, 3))

    def test_simple_face_extractor_returns_correct_type(self):
        img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        extractor = SimpleFaceExtractor(512)
        face = extractor.extract(img, Rect(10, 90, 10, 90))
        self.assertTrue(np.issubdtype(face.dtype, np.uint8))


class AligningDlibFaceExtractorTestCase(unittest.TestCase):
    with dlib_predictor_path() as p:
        predictor = dlib.shape_predictor(str(p))
    bb_finder = DlibBoundingBoxFinder()

    def test_aligning_dlib_face_extractor_returns_correct_shape(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        size = 512
        extractor = AligningDlibFaceExtractor(size, self.predictor)
        face = extractor.extract(img, self.bb_finder.find(img))
        self.assertEqual(face.shape, (size, size, 3))

    def test_aligning_dlib_face_extractor_returns_correct_type(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        extractor = AligningDlibFaceExtractor(512, self.predictor)
        face = extractor.extract(img, Rect(10, 90, 10, 90))
        self.assertTrue(np.issubdtype(face.dtype, np.uint8))

    def test_aligning_dlib_face_extractor_works_for_no_face(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        extractor = AligningDlibFaceExtractor(512, self.predictor)
        face = extractor.extract(img, Rect(10, 90, 10, 90))
        self.assertEqual(face.shape, (512, 512, 3))


if __name__ == '__main__':
    unittest.main()
