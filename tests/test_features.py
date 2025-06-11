import unittest
import pandas as pd
import numpy as np

class TestFeatureEngineering(unittest.TestCase):
    def test_feature_shapes(self):
        test_df = pd.DataFrame({
            'age': [25, 30],
            'gender': ['M', 'F'],
            'recovery_days': [10, 15]
        })
        self.assertEqual(test_df.shape, (2, 3))

if __name__ == "__main__":
    unittest.main()