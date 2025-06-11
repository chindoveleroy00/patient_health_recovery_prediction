import unittest
import numpy as np
from sklearn.dummy import DummyRegressor

class TestModelTraining(unittest.TestCase):
    def test_dummy_model(self):
        X = np.random.rand(10, 3)
        y = np.random.rand(10)
        model = DummyRegressor()
        model.fit(X, y)
        self.assertTrue(hasattr(model, 'predict'))

if __name__ == "__main__":
    unittest.main()