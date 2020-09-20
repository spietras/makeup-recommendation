import importlib.resources as pkg_resources
import unittest

import cv2
import numpy as np
from sklearn.cluster import KMeans

from faceparsing.parser import FaceParser
from imagine.shape.segment import ParsingSegmenter, ClusteringSegmenter


class ParsingSegmenterTestCase(unittest.TestCase):
    segmenter = ParsingSegmenter(FaceParser())

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

    def test_segment_uses_parser_map_by_default(self):
        img = np.zeros((1, 30, 30, 3), dtype=np.uint8)
        segmented = self.segmenter.segment(img)
        self.assertTrue(np.isin(segmented, list(self.segmenter.parser.codes.keys())).all())

    def test_segment_returns_correct_single_mapped_code(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        segmented = self.segmenter.segment(img, parts_map={"skin": 255})
        self.assertTrue((np.unique(segmented) == np.array([0, 255])).all())

    def test_segment_returns_correct_multi_mapped_code(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        segmented = self.segmenter.segment(img, parts_map={"u_lip": 255, "l_lip": 255})
        self.assertTrue((np.unique(segmented) == np.array([0, 255])).all())

    def test_segment_works_with_mask(self):
        with pkg_resources.path("resources", "face.jpg") as p:
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        mask = np.zeros(img.shape[:-1], dtype=np.uint8)
        segmented = self.segmenter.segment(img, bg_code=100, parts_map={"skin": 255}, masks=mask)
        self.assertTrue((np.unique(segmented) == np.array([100])).all())


class ClusteringSegmenterTestCase(unittest.TestCase):
    k = 3
    segmenter = ClusteringSegmenter(KMeans(n_clusters=k))

    def test_segment_returns_same_width_and_height_as_img(self):
        img = np.random.rand(1, 30, 30, 3)
        segmented = self.segmenter.segment(img)
        self.assertEqual(segmented.shape[1:3], img.shape[1:3])

    def test_segment_runs_correctly_on_batch(self):
        img = np.random.rand(2, 30, 30, 3)
        segmented = self.segmenter.segment(img)
        self.assertEqual(segmented.shape[0], img.shape[0])

    def test_segment_returns_two_dimensions(self):
        img = np.random.rand(1, 30, 30, 3)
        segmented = self.segmenter.segment(img)
        self.assertEqual(len(segmented.shape), 3)

    def test_segment_uses_clustering_map_by_default(self):
        img = np.random.rand(1, 30, 30, 3)
        segmented = self.segmenter.segment(img)
        self.assertTrue(np.isin(segmented, list(range(self.k))).all())

    def test_segment_returns_correct_single_mapped_code(self):
        img = np.random.rand(1, 30, 30, 3)
        segmented = self.segmenter.segment(img, parts_map={1: 7})
        self.assertTrue((np.unique(segmented) == np.array([0, 7])).all())

    def test_segment_returns_correct_multi_mapped_code(self):
        img = np.random.rand(1, 30, 30, 3)
        segmented = self.segmenter.segment(img, parts_map={1: 255, 2: 255})
        self.assertTrue((np.unique(segmented) == np.array([0, 255])).all())

    def test_segment_works_with_mask(self):
        img = np.random.rand(1, 30, 30, 3)
        mask = np.zeros(img.shape[:-1], dtype=np.uint8)
        segmented = self.segmenter.segment(img, bg_code=100, parts_map={1: 7}, masks=mask)
        self.assertTrue((np.unique(segmented) == np.array([100])).all())

    def test_segment_works_with_custom_ordering(self):
        img = np.random.rand(1, 30, 30, 3)
        segmenter = ClusteringSegmenter(KMeans(n_clusters=self.k),
                                        ordering=lambda labels, _: np.random.permutation(np.unique(labels)))
        segmented = segmenter.segment(img)
        self.assertTrue((np.unique(segmented) == np.array(range(self.k))).all())


if __name__ == '__main__':
    unittest.main()
