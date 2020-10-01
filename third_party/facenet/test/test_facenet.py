import unittest

import numpy as np

from facenet.facenet import Facenet


class FacenetTestCase(unittest.TestCase):

    facenet = Facenet()

    def test_facenet_returns_correct_shape(self):
        img = np.random.randint(0, 256, size=(1, 160, 160, 3), dtype=np.uint8)
        embedded = self.facenet.embed(img)
        self.assertEqual(embedded.shape, (1, 512))

    def test_facenet_returns_correct_type(self):
        img = np.random.randint(0, 256, size=(1, 160, 160, 3), dtype=np.uint8)
        embedded = self.facenet.embed(img)
        self.assertTrue(np.issubdtype(embedded.dtype, np.floating))


if __name__ == '__main__':
    unittest.main()
