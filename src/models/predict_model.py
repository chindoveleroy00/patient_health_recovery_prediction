import pandas as pd
import joblib
import numpy as np
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Union, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the project root and add necessary paths to sys.path
project_root = Path(__file__).resolve().parents[2]
src_path = project_root / "src"
utils_path = project_root / "src" / "utils"
features_path = project_root / "src" / "features"
sys.path.extend([str(project_root), str(src_path), str(utils_path), str(features_path)])

# Import functions needed for preprocessing
from src.utils.utils import bool_to_int

# Define the create_condition_features function here to ensure it's available
def create_condition_features(X):
    """Create features related to patient conditions"""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=['preexisting_condition', 'diagnosis'])
    
    if len(X) == 0:
        return pd.DataFrame(columns=['has_preexisting_condition', 'has_chronic_condition'])
    
    return pd.DataFrame({
        'has_preexisting_condition': (X['preexisting_condition'] != 'None').astype(int),
        'has_chronic_condition': (X['diagnosis'] == 'Chronic').astype(int)
    })

# Define create_temporal_features to ensure it's available
def create_temporal_features(X):
    """Extract temporal features from admission date"""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=['admission_date'])
    
    if len(X) == 0:
        return pd.DataFrame(columns=['admission_dayofyear', 'admission_weekday'])
    
    dates = pd.to_datetime(X['admission_date'])
    return pd.DataFrame({
        'admission_dayofyear': dates.dt.dayofyear,
        'admission_weekday': dates.dt.weekday,
        'admission_month': dates.dt.month
    })

# Try to import all potential preprocessing functions
try:
    from src.utils.preprocessing_utils import (
        calculate_bmi_category,
        calculate_age_group,
        calculate_bp_category,
        handle_preexisting_conditions,
        days_since_admission
    )
except ImportError:
    logger.warning("Could not import preprocessing_utils. Some functions may not be available.")

# Make preprocessing functions available in this module's namespace
# This is necessary for unpickling the preprocessor
all_preprocessing_funcs = {
    'bool_to_int': bool_to_int,
    'create_condition_features': create_condition_features,
    'create_temporal_features': create_temporal_features
}

try:
    all_preprocessing_funcs.update({
        'calculate_bmi_category': calculate_bmi_category,
        'calculate_age_group': calculate_age_group,
        'calculate_bp_category': calculate_bp_category,
        'handle_preexisting_conditions': handle_preexisting_conditions,
        'days_since_admission': days_since_admission
    })
except NameError:
    logger.warning("Some preprocessing functions are not defined")

# Make all preprocessing functions available in the main module namespace
for func_name, func in all_preprocessing_funcs.items():
    globals()[func_name] = func
    try:
        sys.modules['__main__'].__dict__[func_name] = func
    except:
        logger.warning(f"Could not set {func_name} in __main__ namespace")


# Define feature lists
numeric_features = [
    'age', 'bmi', 'blood_pressure', 'heart_rate',
    'procedures_count', 'duration_of_treatment'
]
categorical_features = [
    'gender', 'admission_reason', 'admission_type',
    'ward_type', 'treatment_type', 'medication_given', 'diagnosis'
]
binary_features = ['smoking_status', 'complications']
ordinal_features = ['severity']
target_column = 'recovery_days'

def load_model_artifacts():
    """Load model and preprocessor with error handling"""
    model_path = project_root / "models" / "xgboost_model.joblib"
    preprocessor_path = project_root / "models" / "preprocessor.joblib"
    
    try:
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = joblib.load(f)
        logger.info(f"Successfully loaded model from {model_path}")
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model not found at {model_path}. Please train the model first.")
    
    try:
        logger.info(f"Loading preprocessor from {preprocessor_path}")
        with open(preprocessor_path, 'rb') as f:
            preprocessor = joblib.load(f)
        logger.info(f"Successfully loaded preprocessor")
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Failed to load preprocessor: {e}")
        raise RuntimeError(f"Preprocessor not found at {preprocessor_path}. Please run feature engineering first.")
    except Exception as e:
        logger.error(f"Unexpected error while loading preprocessor: {e}")
        raise RuntimeError(f"Error loading preprocessor: {str(e)}")
    
    return model, preprocessor

# Load model artifacts
model, preprocessor = load_model_artifacts()

# Required input fields
required_columns = (
    numeric_features + 
    categorical_features + 
    binary_features + 
    ordinal_features + 
    ['admission_date', 'preexisting_condition']
)

def validate_input(data: Union[Dict, pd.DataFrame]) -> None:
    """Validate that all required columns are present in the input data"""
    if isinstance(data, dict):
        missing = [col for col in required_columns if col not in data]
    else:  # pandas DataFrame
        missing = [col for col in required_columns if col not in data.columns]
    
    if missing:
        error_msg = f"Missing required fields: {missing}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def predict_single(data: Dict) -> int:
    """Make a prediction for a single patient using the trained model."""
    validate_input(data)
    
    try:
        df = pd.DataFrame([data])
        X = preprocessor.transform(df)
        prediction = model.predict(X)
        result = round(prediction[0])
        logger.info(f"Successfully predicted {result} days for recovery")
        return result
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise RuntimeError(f"Prediction failed: {str(e)}")

def predict_batch(
    csv_path: str, 
    output_path: Optional[str] = None,
    return_dataframe: bool = False
) -> Optional[pd.DataFrame]:
    """Make predictions for a batch of patients from a CSV file."""
    try:
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        validate_input(df)
        
        logger.info("Transforming features and making predictions")
        X = preprocessor.transform(df)
        predictions = model.predict(X)
        rounded_predictions = np.round(predictions).astype(int)
        
        result_df = df.copy()
        result_df['predicted_recovery_days'] = rounded_predictions
        
        if output_path:
            result_df.to_csv(output_path, index=False)
            logger.info(f"Batch predictions saved to: {output_path}")
        
        return result_df if return_dataframe else None
    
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise RuntimeError(f"Batch prediction failed: {str(e)}")

def predict_from_json(json_data: List[Dict]) -> List[Dict]:
    """Make predictions for a list of patient records provided as JSON."""
    try:
        df = pd.DataFrame(json_data)
        validate_input(df)
        
        X = preprocessor.transform(df)
        predictions = model.predict(X)
        
        result = []
        for i, record in enumerate(json_data):
            prediction_record = record.copy()
            prediction_record['predicted_recovery_days'] = round(predictions[i])
            result.append(prediction_record)
        
        logger.info(f"Successfully processed {len(result)} JSON records")
        return result
    
    except Exception as e:
        logger.error(f"Error processing JSON data: {e}")
        raise RuntimeError(f"JSON prediction failed: {str(e)}")

if __name__ == "__main__":
    sample = {
        "age": 42,
        "bmi": 24.5,
        "blood_pressure": 120,
        "heart_rate": 72,
        "procedures_count": 2,
        "duration_of_treatment": 14,
        "gender": "Female",
        "admission_reason": "Emergency",
        "admission_type": "Inpatient",
        "ward_type": "General",
        "treatment_type": "Medication",
        "medication_given": "Antibiotics",
        "diagnosis": "Infection",
        "smoking_status": False,
        "complications": False,
        "severity": "Mild",
        "admission_date": "2025-03-01",
        "preexisting_condition": "None"
    }

    try:
        result = predict_single(sample)
        print(f"Predicted recovery days: {result} days")
    except Exception as e:
        print(f"Error: {e}")