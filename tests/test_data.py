import unittest
import pandas as pd
import os

class TestDataProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data_path = "tests/test_data/sample_patients.csv"
        
    def test_data_loading(self):
        df = pd.read_csv(self.test_data_path)
        self.assertGreater(len(df), 0)
        self.assertIn('age', df.columns)

if __name__ == "__main__":
    unittest.main()