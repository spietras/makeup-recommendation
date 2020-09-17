import unittest

import numpy as np

from faceparsing.parser import FaceParser


class FaceParserTestCase(unittest.TestCase):

    parser = FaceParser()

    def test_parse_returns_same_width_and_height_as_img(self):
        img = np.random.randint(0, 255, size=(30, 30, 3), dtype=np.uint8)
        parsed = self.parser.parse(img)
        self.assertEqual(parsed.shape[:2], img.shape[:2])

    def test_parse_returns_two_dimensions(self):
        img = np.random.randint(0, 255, size=(30, 30, 3), dtype=np.uint8)
        parsed = self.parser.parse(img)
        self.assertEqual(len(parsed.shape), 2)

    def test_parse_returns_int_type(self):
        img = np.random.randint(0, 255, size=(30, 30, 3), dtype=np.uint8)
        parsed = self.parser.parse(img)
        self.assertTrue(np.issubdtype(parsed.dtype, np.integer))


if __name__ == '__main__':
    unittest.main()
