import unittest

import numpy as np

from faceparsing.parser import FaceParser


class FaceParserTestCase(unittest.TestCase):

    parser = FaceParser()

    def test_parse_returns_same_width_and_height_as_img(self):
        img = np.random.randint(0, 256, size=(1, 30, 30, 3), dtype=np.uint8)
        parsed = self.parser.parse(img)
        self.assertEqual(parsed.shape[1:3], img.shape[1:3])

    def test_parse_runs_correctly_on_batch(self):
        img = np.random.randint(0, 256, size=(2, 30, 30, 3), dtype=np.uint8)
        parsed = self.parser.parse(img)
        self.assertEqual(parsed.shape[0], img.shape[0])

    def test_parse_returns_three_dimensions(self):
        img = np.random.randint(0, 256, size=(1, 30, 30, 3), dtype=np.uint8)
        parsed = self.parser.parse(img)
        self.assertEqual(len(parsed.shape), 3)

    def test_parse_returns_int_type(self):
        img = np.random.randint(0, 256, size=(1, 30, 30, 3), dtype=np.uint8)
        parsed = self.parser.parse(img)
        self.assertTrue(np.issubdtype(parsed.dtype, np.integer))


if __name__ == '__main__':
    unittest.main()
