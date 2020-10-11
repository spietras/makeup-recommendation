import importlib.resources as pkg_resources
import unittest

import cv2
import numpy as np

from automakeup.face.bounding import DlibBoundingBoxFinder, MTCNNBoundingBoxFinder
from imagine.color import conversion
from imagine.shape.figures import Rect
from mtcnn import MTCNN


class DlibBoundingBoxFinderTestCase(unittest.TestCase):
    finder = DlibBoundingBoxFinder()

    def test_dlib_finder_returns_rect(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        bb = self.finder.find(img)
        self.assertIsInstance(bb, Rect)

    def test_dlib_finder_returns_rect_inside_image(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        bb = self.finder.find(img)
        self.assertTrue(bb is not None)
        self.assertTrue(0 <= bb.top <= img.shape[0])
        self.assertTrue(0 <= bb.bottom <= img.shape[0])
        self.assertTrue(0 <= bb.left <= img.shape[1])
        self.assertTrue(0 <= bb.right <= img.shape[1])

    def test_dlib_finder_returns_none_for_no_face(self):
        img = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
        bb = self.finder.find(img)
        self.assertTrue(bb is None)


class MTCNNBoundingBoxFinderTestCase(unittest.TestCase):
    mtcnn = MTCNN()
    finder = MTCNNBoundingBoxFinder(mtcnn)

    def test_mtcnn_finder_returns_rect(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        bb = self.finder.find(img)
        self.assertIsInstance(bb, Rect)

    def test_mtcnn_finder_returns_rect_inside_image(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = conversion.BgrToRgb(cv2.imread(str(p)))
        bb = self.finder.find(img)
        self.assertTrue(bb is not None)
        self.assertTrue(0 <= bb.top <= img.shape[0])
        self.assertTrue(0 <= bb.bottom <= img.shape[0])
        self.assertTrue(0 <= bb.left <= img.shape[1])
        self.assertTrue(0 <= bb.right <= img.shape[1])

    def test_mtcnn_finder_returns_none_for_no_face(self):
        img = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
        bb = self.finder.find(img)
        self.assertTrue(bb is None)


if __name__ == '__main__':
    unittest.main()
