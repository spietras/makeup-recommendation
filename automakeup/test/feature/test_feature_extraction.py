import importlib.resources as pkg_resources
import unittest

import cv2
import dlib
import numpy as np

from automakeup import dlib_predictor_path
from automakeup.face.bounding import DlibBoundingBoxFinder
from automakeup.face.face_extraction import SimpleFaceExtractor, AligningDlibFaceExtractor
from automakeup.feature.feature_extraction import ColorsFeatureExtractor, FacenetFeatureExtractor
from facenet import Facenet
from faceparsing import FaceParser
from imagine.color import conversion
from imagine.shape.figures import Rect


class ColorsFeatureExtractorTestCase(unittest.TestCase):
    face_extractor = SimpleFaceExtractor(512)
    bb_finder = DlibBoundingBoxFinder()
    parser = FaceParser()
    color_extractor = ColorsFeatureExtractor(parser)

    def test_colors_feature_extractor_returns_correct_shape(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        bb = self.bb_finder.find(img)
        face = self.face_extractor.extract(img, bb)
        features = self.color_extractor(face)
        self.assertEqual(features.shape, (12,))

    def test_colors_feature_extractor_returns_correct_type(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        bb = self.bb_finder.find(img)
        face = self.face_extractor.extract(img, bb)
        features = self.color_extractor(face)
        self.assertTrue(np.issubdtype(features.dtype, np.uint8))

    def test_colors_feature_extractor_works_on_batches(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        bb = self.bb_finder.find(img)
        face = self.face_extractor.extract(img, bb)
        features = self.color_extractor(np.array([face, face]))
        self.assertEqual(features.shape, (2, 12))

    def test_colors_feature_extractor_works_on_empty_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bb = Rect(10, 90, 10, 90)
        face = self.face_extractor.extract(img, bb)
        features = self.color_extractor(face)
        self.assertEqual(features.shape, (12,))


class FacenetFeatureExtractorTestCase(unittest.TestCase):
    with dlib_predictor_path() as p:
        predictor = dlib.shape_predictor(str(p))
    face_extractor = AligningDlibFaceExtractor(512, predictor)
    bb_finder = DlibBoundingBoxFinder()
    facenet = Facenet()
    color_extractor = FacenetFeatureExtractor(facenet)

    def test_facenet_feature_extractor_returns_correct_shape(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        bb = self.bb_finder.find(img)
        face = self.face_extractor.extract(img, bb)
        features = self.color_extractor(face)
        self.assertEqual(features.shape, (512,))

    def test_facenet_feature_extractor_returns_correct_type(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        bb = self.bb_finder.find(img)
        face = self.face_extractor.extract(img, bb)
        features = self.color_extractor(face)
        self.assertTrue(np.issubdtype(features.dtype, np.floating))

    def test_facenet_feature_extractor_works_on_batches(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        bb = self.bb_finder.find(img)
        face = self.face_extractor.extract(img, bb)
        features = self.color_extractor(np.array([face, face]))
        self.assertEqual(features.shape, (2, 512))

    def test_facenet_feature_extractor_works_on_empty_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bb = Rect(10, 90, 10, 90)
        face = self.face_extractor.extract(img, bb)
        features = self.color_extractor(face)
        self.assertEqual(features.shape, (512,))


if __name__ == '__main__':
    unittest.main()
