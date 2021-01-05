import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from ganette import Ganette


class GanetteTestCase(unittest.TestCase):

    def test_ganette_can_be_created_with_default_parameters(self):
        Ganette()

    def test_ganette_is_fitted_correctly(self):
        n, xf, yf = 10, 12, 12
        x, y = np.random.rand(n, xf), np.random.rand(n, yf)
        g = Ganette().fit(x, y)
        self.assertTrue(g.logger_.history)

    def test_ganette_is_fitted_correctly_with_different_feature_size(self):
        n, xf, yf = 10, 12, 16
        x, y = np.random.rand(n, xf), np.random.rand(n, yf)
        g = Ganette().fit(x, y)
        self.assertTrue(g.logger_.history)

    def test_ganette_is_fitted_correctly_with_different_batch_size(self):
        n, xf, yf = 10, 12, 12
        x, y = np.random.rand(n, xf), np.random.rand(n, yf)
        g = Ganette(batch_size=10).fit(x, y)
        self.assertTrue(g.logger_.history)

    def test_ganette_is_fitted_correctly_with_batch_size_bigger_than_n(self):
        n, xf, yf = 10, 12, 12
        x, y = np.random.rand(n, xf), np.random.rand(n, yf)
        g = Ganette(batch_size=n + 1).fit(x, y)
        self.assertTrue(g.logger_.history)

    def test_ganette_is_fitted_for_specified_epochs(self):
        epochs = 10
        n, xf, yf = 10, 12, 12
        x, y = np.random.rand(n, xf), np.random.rand(n, yf)
        g = Ganette(epochs=epochs).fit(x, y)
        self.assertEqual(len(next(iter(g.logger_.history.values()))), epochs)

    def test_ganette_fit_fails_with_wrong_input_type(self):
        x, y = 5, "X"
        self.assertRaises(ValueError, Ganette().fit, x, y)

    def test_ganette_fit_fails_with_not_2d_arrays(self):
        x, y = np.random.rand(10), np.random.rand(10, 1, 1)
        self.assertRaises(ValueError, Ganette().fit, x, y)

    def test_ganette_fit_fails_with_unequal_batch_size(self):
        xb, yb = 10, 20
        x, y = np.random.rand(xb, 10), np.random.rand(yb, 10)
        self.assertRaises(ValueError, Ganette().fit, x, y)

    def test_ganette_sample_returns_correct_shape(self):
        n, sn, xf, yf = 10, 20, 12, 12
        x, y, sy = np.random.rand(n, xf), np.random.rand(n, yf), np.random.rand(sn, yf)
        sx = Ganette().fit(x, y).sample(sy)
        self.assertEqual(sx.shape, (sn, xf))

    def test_ganette_sample_returns_correct_shape_with_one_sample(self):
        n, sn, xf, yf = 10, 1, 12, 12
        x, y, sy = np.random.rand(n, xf), np.random.rand(n, yf), np.random.rand(sn, yf)
        sx = Ganette().fit(x, y).sample(sy)
        self.assertEqual(sx.shape, (sn, xf))

    def test_ganette_sample_returns_correct_type(self):
        n, sn, xf, yf = 10, 20, 12, 12
        x, y, sy = np.random.rand(n, xf), np.random.rand(n, yf), np.random.rand(sn, yf)
        sx = Ganette().fit(x, y).sample(sy)
        self.assertTrue(np.issubdtype(sx.dtype, x.dtype))

    def test_ganette_sample_returns_equal_samples_with_equal_states(self):
        n, sn, xf, yf, state = 10, 20, 12, 12, 42
        x, y, sy = np.random.rand(n, xf), np.random.rand(n, yf), np.random.rand(sn, yf)
        g = Ganette().fit(x, y)
        self.assertTrue((g.sample(sy, state=42) == g.sample(sy, state=42)).all())

    def test_ganette_sample_fails_when_model_is_not_fitted(self):
        self.assertRaises(NotFittedError, Ganette().sample, np.random.rand(1, 10))

    def test_ganette_sample_fails_when_y_is_not_an_array(self):
        n, xf, yf = 10, 12, 12
        x, y = np.random.rand(n, xf), np.random.rand(n, yf)
        self.assertRaises(ValueError, Ganette().fit(x, y).sample, "X")

    def test_ganette_sample_fails_when_y_is_not_2d_array(self):
        n, xf, yf = 10, 12, 12
        x, y = np.random.rand(n, xf), np.random.rand(n, yf)
        self.assertRaises(ValueError, Ganette().fit(x, y).sample, np.random.rand(yf))

    def test_ganette_sample_fails_when_y_has_different_features_than_when_trained(self):
        n, sn, xf, yf = 10, 20, 12, 12
        x, y = np.random.rand(n, xf), np.random.rand(n, yf)
        self.assertRaises(ValueError, Ganette().fit(x, y).sample, np.random.rand(sn, yf + 1))

    def test_ganette_score_returns_correct_type(self):
        n, tn, xf, yf = 10, 5, 12, 12
        x, y, tx, ty = np.random.rand(n, xf), np.random.rand(n, yf), np.random.rand(tn, xf), np.random.rand(tn, yf)
        score = Ganette().fit(x, y).score(tx, ty)
        self.assertIsInstance(score, float)

    def test_ganette_score_is_non_positive(self):
        n, tn, xf, yf = 10, 5, 12, 12
        x, y, tx, ty = np.random.rand(n, xf), np.random.rand(n, yf), np.random.rand(tn, xf), np.random.rand(tn, yf)
        score = Ganette().fit(x, y).score(tx, ty)
        self.assertLessEqual(score, 0)

    def test_ganette_score_fails_when_model_is_not_fitted(self):
        self.assertRaises(NotFittedError, Ganette().score, np.random.rand(1, 12), np.random.rand(1, 12))

    def test_ganette_score_fails_when_x_has_different_features_than_when_trained(self):
        n, tn, xf, yf = 10, 20, 12, 12
        x, y = np.random.rand(n, xf), np.random.rand(n, yf)
        self.assertRaises(ValueError, Ganette().fit(x, y).score, np.random.rand(tn, xf + 1), np.random.rand(tn, yf))

    def test_ganette_score_fails_when_y_has_different_features_than_when_trained(self):
        n, tn, xf, yf = 10, 20, 12, 12
        x, y = np.random.rand(n, xf), np.random.rand(n, yf)
        self.assertRaises(ValueError, Ganette().fit(x, y).score, np.random.rand(tn, xf), np.random.rand(tn, yf + 1))


if __name__ == '__main__':
    unittest.main()
