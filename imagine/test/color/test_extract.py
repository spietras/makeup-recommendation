import unittest

import cv2
import numpy as np
from sklearn.cluster import KMeans

from imagine.color import extract


class MeanColorExtractorTestCase(unittest.TestCase):
    extractor = extract.MeanColorExtractor()

    def test_extract_returns_correct_shape(self):
        img = np.array([[[200, 200, 200]]])
        mask = np.array([[1]])
        self.assertEqual(self.extractor.extract(img, mask).shape, (1, 3))

    def test_extract_returns_correct_type(self):
        img = np.array([[[200, 200, 200]]])
        mask = np.array([[1]])
        self.assertTrue(np.issubdtype(self.extractor.extract(img, mask).dtype, np.integer))

    def test_extract_correctly_extracts_color_from_whole_image(self):
        img = cv2.cvtColor(np.array([
            [[255, 128, 128], [255, 128, 128]],
            [[255, 128, 128], [0, 128, 128]]
        ], dtype=np.uint8), cv2.COLOR_Lab2RGB)
        mask = np.array([[1, 1],
                         [1, 1]])
        expected = cv2.cvtColor(np.array([[[191, 128, 128]]], dtype=np.uint8), cv2.COLOR_Lab2RGB).reshape(1, 3)
        self.assertTrue((self.extractor.extract(img, mask) == expected).all())

    def test_extract_returns_none_with_empty_mask(self):
        img = np.array([
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ])
        mask = np.array([[0, 0],
                         [0, 0]])
        self.assertEqual(self.extractor.extract(img, mask), None)


class MedianColorExtractorTestCase(unittest.TestCase):
    extractor = extract.MedianColorExtractor()

    def test_extract_returns_correct_shape(self):
        img = np.array([[[200, 200, 200]]])
        mask = np.array([[1]])
        self.assertEqual(self.extractor.extract(img, mask).shape, (1, 3))

    def test_extract_returns_correct_type(self):
        img = np.array([[[200, 200, 200]]])
        mask = np.array([[1]])
        self.assertTrue(np.issubdtype(self.extractor.extract(img, mask).dtype, np.integer))

    def test_extract_correctly_extracts_color_from_whole_image(self):
        img = cv2.cvtColor(np.array([
            [[255, 128, 128], [255, 128, 128]],
            [[255, 128, 128], [0, 128, 128]]
        ], dtype=np.uint8), cv2.COLOR_Lab2RGB)
        mask = np.array([[1, 1],
                         [1, 1]])
        expected = cv2.cvtColor(np.array([[[255, 128, 128]]], dtype=np.uint8), cv2.COLOR_Lab2RGB).reshape(1, 3)
        self.assertTrue((self.extractor.extract(img, mask) == expected).all())

    def test_extract_returns_none_with_empty_mask(self):
        img = np.array([
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ])
        mask = np.array([[0, 0],
                         [0, 0]])
        self.assertEqual(self.extractor.extract(img, mask), None)


class ClusteringColorExtractorTestCase(unittest.TestCase):

    def test_extract_returns_correct_shape_with_single_cluster(self):
        extractor = extract.ClusteringColorExtractor(clustering=KMeans(n_clusters=1))
        img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]])
        mask = np.array([[1, 1, 1]])
        self.assertEqual(extractor.extract(img, mask).shape, (1, 3))

    def test_extract_returns_correct_shape_with_multiple_clusters(self):
        extractor = extract.ClusteringColorExtractor(clustering=KMeans(n_clusters=3))
        img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]])
        mask = np.array([[1, 1, 1]])
        self.assertEqual(extractor.extract(img, mask).shape, (3, 3))

    def test_extract_returns_correct_type(self):
        extractor = extract.ClusteringColorExtractor(clustering=KMeans(n_clusters=1))
        img = np.array([[[200, 200, 200]]])
        mask = np.array([[1]])
        self.assertTrue(np.issubdtype(extractor.extract(img, mask).dtype, np.integer))

    def test_extract_returns_none_with_empty_mask(self):
        extractor = extract.ClusteringColorExtractor(clustering=KMeans(n_clusters=1))
        img = np.array([
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ])
        mask = np.array([[0, 0],
                         [0, 0]])
        self.assertEqual(extractor.extract(img, mask), None)


class MeanClusteringColorExtractorTestCase(unittest.TestCase):
    extractor = extract.MeanClusteringColorExtractor(clustering=KMeans(n_clusters=3))

    def test_extract_returns_correct_shape(self):
        img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]])
        mask = np.array([[1, 1, 1]])
        self.assertEqual(self.extractor.extract(img, mask).shape, (1, 3))

    def test_extract_returns_correct_type(self):
        img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]])
        mask = np.array([[1, 1, 1]])
        self.assertTrue(np.issubdtype(self.extractor.extract(img, mask).dtype, np.integer))

    def test_extract_returns_none_with_empty_mask(self):
        img = np.array([
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]]
        ])
        mask = np.array([[0, 0],
                         [0, 0]])
        self.assertEqual(self.extractor.extract(img, mask), None)


if __name__ == '__main__':
    unittest.main()
