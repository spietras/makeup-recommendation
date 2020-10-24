import unittest

import numpy as np

from imagine.functional import functional as f


class ImageOperationTestCase(unittest.TestCase):

    class Value(f.ImageOperation):
        def perform(self, img, val=None, **kwargs):
            return 0 if val is None else val

    class BatchValue(f.Batchable, f.ImageOperation):
        def perform(self, img, val=None, **kwargs):
            fill = 0 if val is None else val
            return np.full(len(img), fill)

    def test_image_operation_works_on_batches(self):
        imgs = np.random.randint(0, 256, size=(2, 30, 30, 3))
        result = self.Value()(imgs)
        self.assertEqual(len(result), len(imgs))

    def test_image_operation_works_on_single_images(self):
        img = np.random.randint(0, 256, size=(30, 30, 3))
        result = self.Value()(img)
        self.assertEqual(result, 0)

    def test_batchable_image_operation_works_on_batches(self):
        imgs = np.random.randint(0, 256, size=(2, 30, 30, 3))
        result = self.BatchValue()(imgs)
        self.assertEqual(len(result), len(imgs))

    def test_batchable_image_operation_works_on_single_images(self):
        img = np.random.randint(0, 256, size=(30, 30, 3))
        result = self.BatchValue()(img)
        self.assertEqual(result, 0)

    def test_image_operation_works_with_args(self):
        imgs = np.random.randint(0, 256, size=(2, 30, 30, 3))
        constant = 5
        result = self.Value().withArgs(val=f.Constant(constant))(imgs)
        self.assertEqual(len(result), len(imgs))
        self.assertTrue(constant in result)

    def test_batchable_image_operation_works_with_args(self):
        imgs = np.random.randint(0, 256, size=(2, 30, 30, 3))
        constant = 5
        result = self.BatchValue().withArgs(val=f.Constant(constant))(imgs)
        self.assertEqual(len(result), len(imgs))
        self.assertTrue(constant in result)


class IdentityTestCase(unittest.TestCase):

    identity = f.Identity()

    def test_identity_returns_the_same_object_int(self):
        x = 0
        self.assertEqual(self.identity(x), x)

    def test_identity_returns_the_same_object_float(self):
        x = 0.5
        self.assertEqual(self.identity(x), x)

    def test_identity_returns_the_same_object_string(self):
        x = "test"
        self.assertEqual(self.identity(x), x)

    def test_identity_returns_the_same_object_array(self):
        x = np.random.rand(1, 2, 3)
        self.assertTrue((self.identity(x) == x).all())

    def test_identity_returns_the_same_object_object(self):
        x = f.Identity()
        self.assertEqual(self.identity(x), x)


class ConstantTestCase(unittest.TestCase):

    c = 1
    constant = f.Constant(c)

    def test_constant_works_on_single_objects(self):
        self.assertEqual(self.constant("test"), self.c)

    def test_constant_works_on_batches(self):
        self.assertEqual(self.constant(["test1", "test2"]), [self.c]*2)


class LambdaTestCase(unittest.TestCase):

    def test_lambda_works_correctly(self):
        def fun(x):
            return x+5
        lambda_op = f.Lambda(fun)
        self.assertEqual(lambda_op(5), fun(5))


class JoinTestCase(unittest.TestCase):

    def test_join_works_correctly(self):
        join = f.Join([
            f.Lambda(lambda x: x+5),
            f.Lambda(lambda x: x*2),
            f.Identity(),
            f.Lambda(lambda x: str(x))
        ])
        self.assertEqual(join(5), "20")


class RearrangeTestCase(unittest.TestCase):

    def test_rearrange_correctly_changes_dimension_order(self):
        x = np.random.rand(20, 30, 3)
        result = f.Rearrange("x y c -> y x c")(x)
        self.assertEqual(result.shape, (30, 20, 3))

    def test_rearrange_correctly_composes_axes(self):
        x = np.random.rand(20, 30, 3)
        result = f.Rearrange("x y c -> (x y) c")(x)
        self.assertEqual(result.shape, (20*30, 3))

    def test_rearrange_correctly_composes_nonadjacent_axes(self):
        x = np.random.rand(20, 30, 3)
        result = f.Rearrange("x y c -> (x c) y")(x)
        self.assertEqual(result.shape, (20*3, 30))

    def test_rearrange_correctly_composes_all_axes(self):
        x = np.random.rand(20, 30, 3)
        result = f.Rearrange("x y c -> (x y c)")(x)
        self.assertEqual(result.shape, (20*30*3,))

    def test_rearrange_correctly_decomposes_axis(self):
        x = np.random.rand(20, 30, 3)
        result = f.Rearrange("(x x1) y c -> x x1 y c", x1=2)(x)
        self.assertEqual(result.shape, (10, 2, 30, 3))
        
    def test_rearrange_correctly_adds_empty_axis(self):
        x = np.random.rand(20, 30, 3)
        result = f.Rearrange("x y c -> 1 x y c")(x)
        self.assertEqual(result.shape, (1, 20, 30, 3))

    def test_rearrange_correctly_removes_empty_axis(self):
        x = np.random.rand(1, 20, 30, 3)
        result = f.Rearrange("1 x y c -> x y c")(x)
        self.assertEqual(result.shape, (20, 30, 3))

    def test_rearrange_preserves_dtype(self):
        x = np.random.rand(20, 30, 3)
        result = f.Rearrange("x y c -> y x c")(x)
        self.assertEqual(result.dtype, x.dtype)


class ReduceTestCase(unittest.TestCase):

    def test_reduce_correctly_reduces_axis(self):
        x = np.random.rand(10, 20, 30, 3)
        result = f.Reduce("b h w c -> h w c", "mean")(x)
        self.assertEqual(result.shape, (20, 30, 3))

    def test_reduce_correctly_reduces_multiple_axes(self):
        x = np.random.rand(10, 20, 30, 3)
        result = f.Reduce("b (h hs) (w ws) c -> b h w c", "max", hs=2, ws=3)(x)
        self.assertEqual(result.shape, (10, 10, 10, 3))

    def test_reduce_preserves_dtype(self):
        x = np.random.rand(10, 20, 30, 3)
        result = f.Reduce("b h w c -> h w c", "mean")(x)
        self.assertEqual(result.dtype, x.dtype)


class RepeatTestCase(unittest.TestCase):

    def test_repeat_correctly_repeats_axis(self):
        x = np.random.rand(10, 20, 30, 3)
        result = f.Repeat("b h w c -> b (h r) w c", r=2)(x)
        self.assertEqual(result.shape, (10, 40, 30, 3))

    def test_reduce_correctly_reduces_multiple_axes(self):
        x = np.random.rand(10, 20, 30, 3)
        result = f.Repeat("b h w c -> b (h rh) (w rw) c", rh=2, rw=3)(x)
        self.assertEqual(result.shape, (10, 40, 90, 3))

    def test_reduce_preserves_dtype(self):
        x = np.random.rand(10, 20, 30, 3)
        result = f.Repeat("b h w c -> b (h r) w c", r=2)(x)
        self.assertEqual(result.dtype, x.dtype)


if __name__ == '__main__':
    unittest.main()
