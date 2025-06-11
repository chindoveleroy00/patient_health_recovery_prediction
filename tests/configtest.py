import pytest
import pandas as pd
import os

@pytest.fixture
def sample_patient_data():
    return pd.DataFrame({
        'patient_id': ['P1001', 'P1002'],
        'age': [35, 28],
        'gender': ['M', 'F'],
        'diagnosis': ['Fracture', 'Infection'],
        'recovery_days': [14, 21]
    })

@pytest.fixture
def temp_data_dir(tmp_path):
    dir_path = tmp_path / "test_data"
    dir_path.mkdir()
    return dir_path