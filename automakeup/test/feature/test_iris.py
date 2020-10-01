import importlib.resources as pkg_resources
import unittest

import cv2
import numpy as np

from automakeup.feature.iris import ClusteringIrisShapeExtractor, HoughCircleIrisShapeExtractor
from faceparsing import FaceParser
from imagine.color import conversion
from imagine.shape import operations
from imagine.shape.segment import ParsingSegmenter

parser = FaceParser()


def get_eye(img_path, parser):
    with pkg_resources.path("resources", img_path) as p:
        img = conversion.BgrToRgb(cv2.imread(str(p)))
    eyes_mask = ParsingSegmenter(parser, parts_map={"l_eye": 1, "r_eye": 1})(img) == 1
    biggest_eye_contour = operations.biggest_contour(eyes_mask)
    eye_rect = operations.bounding_rect(biggest_eye_contour)
    eye_rect_square = operations.safe_rect(operations.squarisize(eye_rect), img.shape, allow_scaling=True)
    crop = operations.Crop(eye_rect_square)
    img_cropped = crop(img)
    eye_mask_cropped = crop(np.array(eyes_mask, dtype=np.uint8))
    eye_mask_cropped = operations.Erode(round(0.1 * eye_rect.height()))(eye_mask_cropped)
    return img_cropped, eye_mask_cropped != 0


class ClusteringIrisShapeExtractorTestCase(unittest.TestCase):
    extractor = ClusteringIrisShapeExtractor()

    def test_clustering_iris_shape_extractor_returns_correct_shape(self):
        img, mask = get_eye("face.jpg", parser)
        iris_mask = self.extractor.extract(img, mask)
        self.assertEqual(iris_mask.shape[:2], img.shape[:2])

    def test_clustering_iris_shape_extractor_returns_correct_type(self):
        img, mask = get_eye("face.jpg", parser)
        iris_mask = self.extractor.extract(img, mask)
        self.assertTrue(np.issubdtype(iris_mask.dtype, np.bool))

    def test_clustering_iris_shape_extractor_can_find_the_iris(self):
        img, mask = get_eye("face.jpg", parser)
        iris_mask = self.extractor.extract(img, mask)
        self.assertTrue(np.any(iris_mask))

    def test_clustering_iris_shape_extractor_returns_empty_mask_for_no_iris(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.ones(img.shape[:2], dtype=np.bool)
        iris_mask = self.extractor.extract(img, mask)
        self.assertTrue((~iris_mask).all())


class HoughCircleIrisShapeExtractorTestCase(unittest.TestCase):
    extractor = HoughCircleIrisShapeExtractor()

    def test_hough_circle_iris_shape_extractor_returns_correct_shape(self):
        img, mask = get_eye("face.jpg", parser)
        iris_mask = self.extractor.extract(img, mask)
        self.assertEqual(iris_mask.shape[:2], img.shape[:2])

    def test_hough_circle_iris_shape_extractor_returns_correct_type(self):
        img, mask = get_eye("face.jpg", parser)
        iris_mask = self.extractor.extract(img, mask)
        self.assertTrue(np.issubdtype(iris_mask.dtype, np.bool))

    def test_hough_circle_iris_shape_extractor_can_find_the_iris(self):
        img, mask = get_eye("face.jpg", parser)
        iris_mask = self.extractor.extract(img, mask)
        self.assertTrue(np.any(iris_mask))

    def test_hough_circle_iris_shape_extractor_returns_empty_mask_for_no_iris(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = img == 0
        iris_mask = self.extractor.extract(img, mask)
        self.assertTrue((~iris_mask).all())


if __name__ == '__main__':
    unittest.main()
