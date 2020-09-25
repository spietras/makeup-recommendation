import unittest

import numpy as np

from imagine.functional import functional


class ImageOperationTestCase(unittest.TestCase):

    class Value(functional.ImageOperation):
        def perform(self, img, val=None, **kwargs):
            return 0 if val is None else val

    class BatchValue(functional.Batchable, functional.ImageOperation):
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
        result = self.Value().withArgs(val=functional.Constant(constant))(imgs)
        self.assertEqual(len(result), len(imgs))
        self.assertTrue(constant in result)

    def test_batchable_image_operation_works_with_args(self):
        imgs = np.random.randint(0, 256, size=(2, 30, 30, 3))
        constant = 5
        result = self.BatchValue().withArgs(val=functional.Constant(constant))(imgs)
        self.assertEqual(len(result), len(imgs))
        self.assertTrue(constant in result)


class IdentityTestCase(unittest.TestCase):

    identity = functional.Identity()

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
        x = functional.Identity()
        self.assertEqual(self.identity(x), x)


class ConstantTestCase(unittest.TestCase):

    c = 1
    constant = functional.Constant(c)

    def test_constant_works_on_single_objects(self):
        self.assertEqual(self.constant("test"), self.c)

    def test_constant_works_on_batches(self):
        self.assertEqual(self.constant(["test1", "test2"]), [self.c]*2)


class ChannelizeTestCase(unittest.TestCase):

    channelize = functional.Channelize()
    dechannelize = functional.Dechannelize()

    def test_channelize_correctly_adds_channel(self):
        x = np.random.rand(30, 30)
        self.assertEqual(self.channelize(x).shape, x.shape + (1,))

    def test_dechannelize_correctly_removes_channel(self):
        x = np.random.rand(30, 30, 1)
        self.assertEqual(self.dechannelize(x).shape, x.shape[:-1])


class LambdaTestCase(unittest.TestCase):

    def test_lambda_works_correctly(self):
        def f(x):
            return x+5
        lambda_op = functional.Lambda(f)
        self.assertEqual(lambda_op(5), f(5))


class JoinTestCase(unittest.TestCase):

    def test_join_works_correctly(self):
        join = functional.Join([
            functional.Lambda(lambda x: x+5),
            functional.Lambda(lambda x: x*2),
            functional.Identity(),
            functional.Lambda(lambda x: str(x))
        ])
        self.assertEqual(join(5), "20")


if __name__ == '__main__':
    unittest.main()
