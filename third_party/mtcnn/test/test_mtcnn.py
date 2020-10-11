import importlib.resources as pkg_resources
import unittest

import cv2
import numpy as np

from imagine.color import conversion
from mtcnn import MTCNN


def load_image():
    with pkg_resources.path("resources", "face.jpg") as p:
        img = conversion.BgrToRgb(cv2.imread(str(p)))
    return img


class MTCNNTestCase(unittest.TestCase):

    img = np.expand_dims(load_image(), 0)
    mtcnn = MTCNN()

    def test_mtcnn_returns_correct_shape(self):
        bb, _ = self.mtcnn.find(self.img)
        self.assertEqual(bb[0].shape, (1, 4))

    def test_mtcnn_returns_correct_type(self):
        bb, _ = self.mtcnn.find(self.img)
        self.assertTrue(np.issubdtype(bb[0].dtype, np.integer))

    def test_mtcnn_returns_probability(self):
        _, prob = self.mtcnn.find(self.img)
        self.assertEqual(len(prob), 1)
        self.assertTrue(prob[0] > 0)


if __name__ == '__main__':
    unittest.main()
