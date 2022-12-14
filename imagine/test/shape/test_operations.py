import unittest

import cv2
import numpy as np

from imagine.functional import functional as f
from imagine.shape import operations
from imagine.shape.figures import Rect


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
        img = f.Rearrange("h w -> h w 1")(generate_binary_square())
        contour = operations.biggest_contour(img)
        self.assertEqual(contour.ndim, 2)
        self.assertNotEqual(contour.shape[0], 0)

    def test_biggest_contour_fails_for_three_channel_image(self):
        img = f.Rearrange("h w -> h w 1")(generate_binary_square())
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


class FillContourTestCase(unittest.TestCase):

    def test_fill_contour_returns_correct_shape(self):
        shape = (10, 10, 1)
        contour = np.array([[1, 1], [1, 9], [9, 9], [9, 1]])
        mask = operations.fill_contour(contour, shape)
        self.assertEqual(mask.shape, shape[:2])

    def test_fill_contour_returns_correct_shape_with_three_channels(self):
        shape = (10, 10, 3)
        contour = np.array([[1, 1], [1, 9], [9, 9], [9, 1]])
        mask = operations.fill_contour(contour, shape)
        self.assertEqual(mask.shape, shape[:2])

    def test_fill_contour_returns_correct_shape_with_flat_image(self):
        shape = (10, 10)
        contour = np.array([[1, 1], [1, 9], [9, 9], [9, 1]])
        mask = operations.fill_contour(contour, shape)
        self.assertEqual(mask.shape, shape[:2])

    def test_fill_contour_returns_correct_data_type(self):
        shape = (10, 10, 1)
        contour = np.array([[1, 1], [1, 9], [9, 9], [9, 1]])
        mask = operations.fill_contour(contour, shape)
        self.assertEqual(mask.dtype, np.bool)

    def test_fill_contour_works_with_empty_contour(self):
        shape = (10, 10, 1)
        contour = np.array([], dtype=np.int)
        mask = operations.fill_contour(contour, shape)
        self.assertEqual(mask.shape, shape[:2])


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
        x, y, w, h = operations.bounding_rect(contour).to_cv()
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
        cropped = operations.crop(img, Rect.from_cv((1, 1, 3, 3)))
        self.assertTrue((cropped == np.ones((3, 3))).all())

    def test_crop_succeeds_on_whole_image_rect(self):
        img = np.ones((30, 30, 3))
        cropped = operations.crop(img, Rect.from_cv((0, 0, 30, 30)))
        self.assertTrue((cropped == img).all())

    def test_crop_returns_correct_shape(self):
        img = np.ones((30, 30, 3))
        cropped = operations.crop(img, Rect.from_cv((0, 0, 30, 30)))
        self.assertEqual(cropped.shape, img.shape)

    def test_crop_works_on_flat_image(self):
        img = np.ones((30, 30))
        cropped = operations.crop(img, Rect.from_cv((0, 0, 30, 30)))
        self.assertEqual(cropped.shape, img.shape)

    def test_crop_returns_correct_data_type(self):
        img = np.ones((30, 30, 3))
        cropped = operations.crop(img, Rect.from_cv((0, 0, 30, 30)))
        self.assertEqual(cropped.dtype, img.dtype)

    def test_crop_works_on_rect_besides_edges(self):
        img = np.ones((30, 30, 3))
        cropped = operations.crop(img, Rect.from_cv((0, 0, 40, 20)))
        expected = operations.crop(img, Rect.from_cv((0, 0, 30, 20)))
        self.assertEqual(cropped.shape, expected.shape)


class ErodeTestCase(unittest.TestCase):

    def test_erode_returns_correct_shape(self):
        img = np.ones((30, 30, 1))
        eroded = operations.erode(img, (1, 1))
        self.assertEqual(eroded.shape, img.shape)

    def test_erode_returns_correct_shape_with_three_channels(self):
        img = np.ones((30, 30, 3))
        eroded = operations.erode(img, (1, 1))
        self.assertEqual(eroded.shape, img.shape)

    def test_erode_works_on_flat_image(self):
        img = np.ones((30, 30))
        eroded = operations.erode(img, (1, 1))
        self.assertEqual(eroded.shape, img.shape)

    def test_erode_returns_correct_data_type(self):
        img = np.ones((30, 30, 3))
        eroded = operations.erode(img, (1, 1))
        self.assertEqual(eroded.dtype, img.dtype)

    def test_erode_works_with_single_value_kernel(self):
        img = np.ones((30, 30, 1))
        eroded = operations.erode(img, 1)
        self.assertEqual(eroded.shape, img.shape)


class DilateTestCase(unittest.TestCase):

    def test_dilate_returns_correct_shape(self):
        img = np.ones((30, 30, 1))
        dilated = operations.dilate(img, (1, 1))
        self.assertEqual(dilated.shape, img.shape)

    def test_dilate_returns_correct_shape_with_three_channels(self):
        img = np.ones((30, 30, 3))
        dilated = operations.dilate(img, (1, 1))
        self.assertEqual(dilated.shape, img.shape)

    def test_dilate_works_on_flat_image(self):
        img = np.ones((30, 30))
        dilated = operations.dilate(img, (1, 1))
        self.assertEqual(dilated.shape, img.shape)

    def test_dilate_returns_correct_data_type(self):
        img = np.ones((30, 30, 3))
        dilated = operations.dilate(img, (1, 1))
        self.assertEqual(dilated.dtype, img.dtype)

    def test_dilate_works_with_single_value_kernel(self):
        img = np.ones((30, 30, 1))
        dilated = operations.dilate(img, 1)
        self.assertEqual(dilated.shape, img.shape)


class SquarisizeTestCase(unittest.TestCase):

    def test_squarisize_returns_square(self):
        _, _, w, h = operations.squarisize(Rect.from_cv((0, 0, 5, 10))).to_cv()
        self.assertEqual(w, h)

    def test_squarisize_returns_bigger_dimension(self):
        _, _, w, h = operations.squarisize(Rect.from_cv((0, 0, 5, 10))).to_cv()
        self.assertEqual(w, 10)
        self.assertEqual(h, 10)


class SafeRectTestCase(unittest.TestCase):

    def test_safe_rect_returns_the_same_rect_when_inside_bounds(self):
        rect = Rect.from_cv((1, 1, 5, 5))
        safe = operations.safe_rect(rect, (10, 10))
        self.assertEqual(rect, safe)

    def test_safe_rect_moves_rect_correctly_when_outside_bounds(self):
        rect = Rect.from_cv((9, 9, 2, 2))
        x, y, _, _ = operations.safe_rect(rect, (10, 10)).to_cv()
        self.assertEqual(x, 8)
        self.assertEqual(y, 8)

    def test_safe_rect_fails_when_rect_is_too_big(self):
        rect = Rect.from_cv((0, 0, 11, 1))
        self.assertRaises(ValueError, operations.safe_rect, rect, (10, 10))

    def test_safe_rect_scales_rect_when_rect_is_too_big_and_allow_scale_is_true(self):
        rect = Rect.from_cv((0, 0, 4, 16))
        x, y, w, h = operations.safe_rect(rect, (8, 8), allow_scaling=True).to_cv()
        self.assertEqual(x, 1)
        self.assertEqual(y, 0)
        self.assertEqual(w, 2)
        self.assertEqual(h, 8)


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


class ResizeTestCase(unittest.TestCase):

    def test_resize_returns_correct_shape(self):
        img = np.random.randint(0, 256, size=(30, 30, 3), dtype=np.uint8)
        shape = (10, 10)
        resized = operations.resize(img, shape)
        self.assertEqual(resized.shape[:2], shape)


if __name__ == '__main__':
    unittest.main()