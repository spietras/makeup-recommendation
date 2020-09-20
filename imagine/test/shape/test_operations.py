import unittest

import cv2
import numpy as np

from imagine.shape import operations


def generate_binary_square(square_value=1, size=1, padding_value=0, padding=1):
    return np.pad(np.pad(np.array([[square_value]]),
                         size - 1,
                         constant_values=square_value),
                  padding,
                  constant_values=padding_value)


class BiggestContourTestCase(unittest.TestCase):

    def test_biggest_contour_finds_contour_in_binary_mask_without_channel(self):
        img = generate_binary_square()
        contour = operations.biggest_contour(img)
        self.assertEqual(contour.ndim, 2)
        self.assertNotEqual(contour.shape[0], 0)

    def test_biggest_contour_finds_contour_in_binary_mask_with_channel(self):
        img = np.expand_dims(generate_binary_square(), 2)
        contour = operations.biggest_contour(img)
        self.assertEqual(contour.ndim, 2)
        self.assertNotEqual(contour.shape[0], 0)

    def test_biggest_contour_fails_for_three_channel_image(self):
        img = np.expand_dims(generate_binary_square(), 2)
        img = np.dstack([img, img, img])
        self.assertRaises(cv2.error, operations.biggest_contour, img)

    def test_biggest_contour_returns_correct_data_type(self):
        img = generate_binary_square()
        contour = operations.biggest_contour(img)
        self.assertTrue(np.issubdtype(contour.dtype, np.integer))

    def test_biggest_contour_only_one_contour(self):
        s1 = generate_binary_square(size=2, padding=1)
        s2 = generate_binary_square(square_value=0, size=2, padding=1)
        s3 = generate_binary_square(size=1, padding=2)
        img = np.hstack([s1, s2, s3])
        contour = operations.biggest_contour(img)
        self.assertEqual(contour.ndim, 2)
        self.assertNotEqual(contour.shape[0], 0)

    def test_biggest_contour_finds_bigger_contour(self):
        s1 = generate_binary_square(size=2, padding=1)
        s2 = generate_binary_square(square_value=0, size=2, padding=1)
        s3 = generate_binary_square(size=1, padding=2)
        img = np.hstack([s1, s2, s3])
        contour = operations.biggest_contour(img)
        center = operations.mass_center(contour)
        self.assertTrue((center == np.array([2.0, 2.0])).all())

    def test_biggest_contour_returns_none_on_no_contour(self):
        img = generate_binary_square(square_value=0)
        contour = operations.biggest_contour(img)
        self.assertEqual(contour.size, 0)


class MassCenterTestCase(unittest.TestCase):

    def test_mass_center_finds_correct_center(self):
        img = generate_binary_square(size=2, padding=1)
        contour = operations.biggest_contour(img)
        center = operations.mass_center(contour)
        self.assertTrue((center == np.array([2.0, 2.0])).all())

    def test_mass_center_return_correct_shape(self):
        img = generate_binary_square(size=2, padding=1)
        contour = operations.biggest_contour(img)
        center = operations.mass_center(contour)
        self.assertEqual(center.shape, (2,))

    def test_mass_center_return_correct_data_type(self):
        img = generate_binary_square(size=2, padding=1)
        contour = operations.biggest_contour(img)
        center = operations.mass_center(contour)
        self.assertTrue(np.issubdtype(center.dtype, np.float))

    def test_mass_center_return_none_with_empty_contour(self):
        contour = np.array([[]], dtype=np.int)
        center = operations.mass_center(contour)
        self.assertTrue(center is None)


class BoundingRectTestCase(unittest.TestCase):

    def test_bounding_rect_returns_correct_values(self):
        img = generate_binary_square(size=2, padding=1)
        contour = operations.biggest_contour(img)
        x, y, w, h = operations.bounding_rect(contour)
        self.assertEqual(x, 1)
        self.assertEqual(y, 1)
        self.assertEqual(w, 3)
        self.assertEqual(h, 3)

    def test_bounding_rect_returns_none_with_empty_contour(self):
        contour = np.array([[]], dtype=np.int)
        rect = operations.bounding_rect(contour)
        self.assertTrue(rect is None)


class CropTestCase(unittest.TestCase):

    def test_crop_returns_correct_values(self):
        img = generate_binary_square(square_value=1, size=2, padding=1)
        cropped = operations.crop(img, (1, 1, 3, 3))
        self.assertTrue((cropped == np.ones((3, 3))).all())

    def test_crop_succeeds_on_whole_image_rect(self):
        img = np.ones((30, 30, 3))
        cropped = operations.crop(img, (0, 0, 30, 30))
        self.assertTrue((cropped == img).all())

    def test_crop_returns_correct_shape(self):
        img = np.ones((30, 30, 3))
        cropped = operations.crop(img, (0, 0, 30, 30))
        self.assertEqual(cropped.shape, img.shape)

    def test_crop_works_on_flat_image(self):
        img = np.ones((30, 30))
        cropped = operations.crop(img, (0, 0, 30, 30))
        self.assertEqual(cropped.shape, img.shape)

    def test_crop_returns_correct_data_type(self):
        img = np.ones((30, 30, 3))
        cropped = operations.crop(img, (0, 0, 30, 30))
        self.assertEqual(cropped.dtype, img.dtype)

    def test_crop_works_on_rect_besides_edges(self):
        img = np.ones((30, 30, 3))
        cropped = operations.crop(img, (0, 0, 40, 20))
        expected = operations.crop(img, (0, 0, 30, 20))
        self.assertEqual(cropped.shape, expected.shape)


class ErodeTestCase(unittest.TestCase):

    def test_erode_returns_correct_shape(self):
        img = np.ones((30, 30, 1))
        eroded = operations.erode(img, 1)
        self.assertEqual(eroded.shape, img.shape)

    def test_erode_returns_correct_shape_with_three_channels(self):
        img = np.ones((30, 30, 3))
        eroded = operations.erode(img, 1)
        self.assertEqual(eroded.shape, img.shape)

    def test_erode_works_on_flat_image(self):
        img = np.ones((30, 30))
        eroded = operations.erode(img, 1)
        self.assertEqual(eroded.shape, img.shape)

    def test_erode_returns_correct_data_type(self):
        img = np.ones((30, 30, 3))
        eroded = operations.erode(img, 1)
        self.assertEqual(eroded.dtype, img.dtype)


class SquarisizeTestCase(unittest.TestCase):

    def test_squarisize_returns_square(self):
        _, _, w, h = operations.squarisize((0, 0, 5, 10))
        self.assertEqual(w, h)

    def test_squarisize_returns_bigger_dimension(self):
        _, _, w, h = operations.squarisize((0, 0, 5, 10))
        self.assertEqual(w, 10)
        self.assertEqual(h, 10)


class SafeRectTestCase(unittest.TestCase):

    def test_safe_rect_returns_the_same_rect_when_inside_bounds(self):
        rect = (1, 1, 5, 5)
        safe = operations.safe_rect(rect, 10, 10)
        self.assertEqual(rect, safe)

    def test_safe_rect_moves_rect_correctly_when_outside_bounds(self):
        rect = (9, 9, 2, 2)
        x, y, _, _ = operations.safe_rect(rect, 10, 10)
        self.assertEqual(x, 8)
        self.assertEqual(y, 8)

    def test_safe_rect_fails_when_rect_is_too_big(self):
        rect = (0, 0, 11, 1)
        self.assertRaises(ValueError, operations.safe_rect, rect, 10, 10)


class CircleMaskTestCase(unittest.TestCase):

    def test_circle_mask_returns_correct_shape(self):
        shape = (10, 10, 1)
        mask = operations.circle_mask(shape, (0, 0), 3)
        self.assertEqual(mask.shape, shape[:2])

    def test_circle_mask_returns_correct_shape_with_three_channels(self):
        shape = (10, 10, 3)
        mask = operations.circle_mask(shape, (0, 0), 3)
        self.assertEqual(mask.shape, shape[:2])

    def test_circle_mask_returns_correct_shape_with_flat_image(self):
        shape = (10, 10)
        mask = operations.circle_mask(shape, (0, 0), 3)
        self.assertEqual(mask.shape, shape[:2])

    def test_circle_mask_returns_correct_data_type(self):
        shape = (10, 10, 1)
        mask = operations.circle_mask(shape, (0, 0), 3)
        self.assertEqual(mask.dtype, np.bool)

    def test_circle_mask_works_with_circles_besides_bounds(self):
        shape = (10, 10, 1)
        mask = operations.circle_mask(shape, (100, 100), 3)
        self.assertEqual(mask.shape, shape[:2])


if __name__ == '__main__':
    unittest.main()
