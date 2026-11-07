import unittest
import numpy as np

from agents.learner_agent import training_rows
from common.images import transform_image


class AugmentationTest(unittest.TestCase):
    def test_variants_stay_in_split_and_preserve_mass(self):
        X = np.arange(4, dtype=float).reshape(4, 1)
        variants = np.stack([X + 10, X + 20])
        labels = np.array([1, -1, 1, -1])
        weights = np.array([.1, .2, .3, .4])
        rows, y, w = training_rows(X, variants, labels, weights, np.array([0, 2]))
        np.testing.assert_array_equal(rows.ravel(), [0, 2, 10, 12, 20, 22])
        np.testing.assert_allclose(w.reshape(3, 2).sum(axis=0), [.1, .3])
        np.testing.assert_array_equal(y, [1, 1, 1, 1, 1, 1])

    def test_translation_and_constant_rotation(self):
        image = np.arange(361, dtype=np.uint8).reshape(19, 19)
        shifted = transform_image(image, dx=1)
        np.testing.assert_array_equal(shifted[:, 1:], image[:, :-1])
        const = np.full((19, 19), 70, dtype=np.uint8)
        np.testing.assert_array_equal(transform_image(const, angle=5), const)


if __name__ == "__main__":
    unittest.main()
