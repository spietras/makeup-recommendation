import unittest

import numpy as np

from imagine import helpers


class NormalizationTestCase(unittest.TestCase):

    def test_normalize_images_converts_float_array_to_uint8(self):
        img = np.random.rand(1, 30, 30, 3)
        normalized = helpers.normalize_images(img)
        self.assertTrue(np.issubdtype(normalized.dtype, np.uint8))

    def test_normalize_images_converts_int_array_to_uint8(self):
        img = np.random.randint(0, 256, size=(1, 30, 30, 3), dtype=np.int)
        normalized = helpers.normalize_images(img)
        self.assertTrue(np.issubdtype(normalized.dtype, np.uint8))

    def test_normalize_images_succeeds_for_single_image(self):
        img = np.random.rand(30, 30, 3)
        normalized = helpers.normalize_images(img)
        self.assertEquals(normalized.shape, img.shape)

    def test_normalize_images_succeeds_for_batch(self):
        img = np.random.rand(1, 30, 30, 3)
        normalized = helpers.normalize_images(img)
        self.assertEquals(normalized.shape, img.shape)

    def test_normalize_images_fails_for_wrong_dimensions(self):
        img = np.random.rand(1, 1, 30, 30, 3)
        self.assertRaises(ValueError, helpers.normalize_images, img)

    def test_normalize_images_succeeds_for_one_channel(self):
        img = np.random.rand(30, 30, 1)
        normalized = helpers.normalize_images(img)
        self.assertEquals(normalized.shape, img.shape)

    def test_normalize_images_succeeds_for_three_channels(self):
        img = np.random.rand(30, 30, 3)
        normalized = helpers.normalize_images(img)
        self.assertEquals(normalized.shape, img.shape)

    def test_normalize_images_fails_for_wrong_channels_number(self):
        img = np.random.rand(30, 30, 2)
        self.assertRaises(ValueError, helpers.normalize_images, img)

    def test_normalize_images_fails_for_wrong_data_type(self):
        img = np.full((1, 30, 30, 3), "string")
        self.assertRaises(ValueError, helpers.normalize_images, img)

    def test_normalize_range_correctly_changes_values_range(self):
        img = np.array([0, 255], dtype=np.uint8)
        normalized = helpers.normalize_range(img)
        self.assertEqual(normalized.min(), 0.0)
        self.assertEqual(normalized.max(), 1.0)

    def test_normalize_range_correctly_changes_data_type(self):
        img = np.random.randint(0, 256, size=(30, 30, 3), dtype=np.uint8)
        normalized = helpers.normalize_range(img)
        self.assertTrue(np.issubdtype(normalized.dtype, np.floating))

    def test_normalize_range_succeeds_for_single_image(self):
        img = np.random.randint(0, 256, size=(30, 30, 3), dtype=np.uint8)
        converted = helpers.normalize_range(img)
        self.assertEquals(converted.shape, img.shape)

    def test_normalize_range_succeeds_for_batch(self):
        img = np.random.randint(0, 256, size=(1, 30, 30, 3), dtype=np.uint8)
        converted = helpers.normalize_range(img)
        self.assertEquals(converted.shape, img.shape)

    def test_normalize_range_succeeds_for_multiple_pixels(self):
        img = np.random.randint(0, 256, size=(30, 3), dtype=np.uint8)
        converted = helpers.normalize_range(img)
        self.assertEquals(converted.shape, img.shape)

    def test_normalize_range_succeeds_for_single_pixel(self):
        img = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
        converted = helpers.normalize_range(img)
        self.assertEquals(converted.shape, img.shape)

    def test_denormalize_range_correctly_changes_values_range(self):
        img = np.array([0.0, 1.0], dtype=np.float)
        normalized = helpers.denormalize_range(img)
        self.assertEqual(normalized.min(), 0)
        self.assertEqual(normalized.max(), 255)

    def test_denormalize_range_correctly_changes_data_type(self):
        img = np.random.rand(30, 30, 3)
        normalized = helpers.denormalize_range(img)
        self.assertTrue(np.issubdtype(normalized.dtype, np.uint8))

    def test_denormalize_range_succeeds_for_single_image(self):
        img = np.random.rand(30, 30, 3)
        converted = helpers.denormalize_range(img)
        self.assertEquals(converted.shape, img.shape)

    def test_denormalize_range_succeeds_for_batch(self):
        img = np.random.rand(1, 30, 30, 3)
        converted = helpers.denormalize_range(img)
        self.assertEquals(converted.shape, img.shape)

    def test_denormalize_range_succeeds_for_multiple_pixels(self):
        img = np.random.rand(30, 3)
        converted = helpers.denormalize_range(img)
        self.assertEquals(converted.shape, img.shape)

    def test_denormalize_range_succeeds_for_single_pixel(self):
        img = np.random.rand(3, )
        converted = helpers.denormalize_range(img)
        self.assertEquals(converted.shape, img.shape)


if __name__ == '__main__':
    unittest.main()
