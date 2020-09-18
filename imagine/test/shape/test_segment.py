import importlib.resources as pkg_resources
import unittest

import cv2
import numpy as np

from faceparsing.parser import FaceParser
from imagine.shape.segment import FaceParsingSegmenter


class FaceParsingSegmenterTestCase(unittest.TestCase):
    segmenter = FaceParsingSegmenter(FaceParser())

    def test_segment_returns_same_width_and_height_as_img(self):
        img = np.random.randint(0, 256, size=(1, 30, 30, 3), dtype=np.uint8)
        segmented = self.segmenter.segment(img)
        self.assertEqual(segmented.shape[1:3], img.shape[1:3])

    def test_segment_runs_correctly_on_batch(self):
        img = np.random.randint(0, 256, size=(2, 30, 30, 3), dtype=np.uint8)
        segmented = self.segmenter.segment(img)
        self.assertEqual(segmented.shape[0], img.shape[0])

    def test_segment_returns_two_dimensions(self):
        img = np.random.randint(0, 256, size=(1, 30, 30, 3), dtype=np.uint8)
        segmented = self.segmenter.segment(img)
        self.assertEqual(len(segmented.shape), 3)

    def test_segment_returns_correct_default_background_code(self):
        img = np.zeros((1, 30, 30, 3), dtype=np.uint8)
        segmented = self.segmenter.segment(img)
        self.assertTrue(0 in segmented)

    def test_segment_sets_correct_background_code(self):
        img = np.zeros((1, 30, 30, 3), dtype=np.uint8)
        segmented = self.segmenter.segment(img, bg_code=255)
        self.assertTrue(255 in segmented)

    def test_segment_returns_correct_face_code(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        segmented = self.segmenter.segment(np.expand_dims(img, 0), parts_map={"face": 255})
        self.assertTrue(255 in segmented)

    def test_segment_returns_correct_lips_code(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        segmented = self.segmenter.segment(np.expand_dims(img, 0), parts_map={"lips": 255})
        self.assertTrue(255 in segmented)

    def test_segment_returns_correct_left_eye_code(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        segmented = self.segmenter.segment(np.expand_dims(img, 0), parts_map={"left_eye": 255})
        self.assertTrue(255 in segmented)

    def test_segment_returns_correct_right_eye_code(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        segmented = self.segmenter.segment(np.expand_dims(img, 0), parts_map={"right_eye": 255})
        self.assertTrue(255 in segmented)

    def test_segment_returns_correct_hair_code(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        segmented = self.segmenter.segment(np.expand_dims(img, 0), parts_map={"hair": 255})
        self.assertTrue(255 in segmented)


if __name__ == '__main__':
    unittest.main()
