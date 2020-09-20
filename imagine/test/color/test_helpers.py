import unittest

import numpy as np

from imagine.color import helpers


class RecolorTestCase(unittest.TestCase):

    def test_recolor_returns_correct_shape(self):
        img = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, img.shape[:2])
        recolored = helpers.recolor(img, mask, [255, 255, 255], 0.5)
        self.assertEqual(recolored.shape, img.shape)

    def test_recolor_works_with_one_channel(self):
        img = np.random.randint(0, 256, (30, 30, 1), dtype=np.uint8)
        mask = np.random.randint(0, 2, img.shape[:2])
        recolored = helpers.recolor(img, mask, [255], 0.5)
        self.assertEqual(recolored.shape, img.shape)

    def test_recolor_works_on_flat_image(self):
        img = np.random.randint(0, 256, (30, 30), dtype=np.uint8)
        mask = np.random.randint(0, 2, img.shape[:2])
        recolored = helpers.recolor(img, mask, [255], 0.5)
        self.assertEqual(recolored.shape, img.shape)

    def test_recolor_returns_correct_data_type(self):
        img = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, img.shape[:2])
        recolored = helpers.recolor(img, mask, [255, 255, 255], 0.5)
        self.assertEqual(recolored.dtype, img.dtype)

    def test_recolor_works_with_tuple_as_color(self):
        img = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, img.shape[:2])
        recolored = helpers.recolor(img, mask, (255, 255, 255), 0.5)
        self.assertEqual(recolored.shape, img.shape)

    def test_recolor_works_with_array_as_color(self):
        img = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, img.shape[:2])
        recolored = helpers.recolor(img, mask, np.array([255, 255, 255]), 0.5)
        self.assertEqual(recolored.shape, img.shape)


class GenerationTestCase(unittest.TestCase):

    def test_generate_distinct_colors_returns_given_number_of_colors(self):
        k = 10
        generated = helpers.generate_distinct_colors(k)
        self.assertEqual(generated.shape[0], k)

    def test_generate_distinct_colors_returns_correct_data_type(self):
        k = 10
        generated = helpers.generate_distinct_colors(k)
        self.assertTrue(np.issubdtype(generated.dtype, np.uint8))

    def test_generate_distinct_colors_returns_succeeds_for_one_color(self):
        k = 1
        generated = helpers.generate_distinct_colors(k)
        self.assertEqual(generated.shape[0], k)

    def test_generate_distinct_colors_returns_succeeds_for_more_than_256_colors(self):
        k = 1000
        generated = helpers.generate_distinct_colors(k)
        self.assertEqual(generated.shape[0], k)

    def test_generate_distinct_returns_empty_array_for_zero_colors(self):
        k = 0
        generated = helpers.generate_distinct_colors(k)
        self.assertEqual(generated.size, 0)


if __name__ == '__main__':
    unittest.main()
