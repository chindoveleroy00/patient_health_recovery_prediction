import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
import joblib
from datetime import datetime

# Define the bool_to_int function directly in this file to avoid import issues
def bool_to_int(x):
    """Convert boolean values to integers."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.astype(int)
    elif isinstance(x, bool):
        return int(x)
    elif isinstance(x, str) and x.lower() in ['true', 'yes', 'y', '1']:
        return 1
    elif isinstance(x, str) and x.lower() in ['false', 'no', 'n', '0']:
        return 0
    elif x is None:
        return 0
    else:
        return x

project_root = Path(__file__).resolve().parents[2]

# Updated feature lists based on our analysis
numeric_features = [
    'age', 
    'bmi',
    'blood_pressure',
    'heart_rate',
    'procedures_count',
    'duration_of_treatment'
]

categorical_features = [
    'gender',
    'admission_reason',
    'admission_type',
    'ward_type',
    'treatment_type',
    'medication_given',
    'diagnosis'
]

binary_features = [
    'smoking_status',
    'complications'
]

ordinal_features = ['severity']  # Ordered from mild to severe

target_column = 'recovery_days'

# Custom transformers
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

# Create feature processing pipeline
def create_feature_pipeline():
    # Numeric features pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical features pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Binary features pipeline
    binary_transformer = Pipeline(steps=[
        ('bool_to_int', FunctionTransformer(bool_to_int)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='if_binary', sparse_output=False))
    ])

    # Ordinal features pipeline
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=[['Mild', 'Moderate', 'Severe']]))
    ])

    # Temporal features pipeline
    temporal_transformer = Pipeline(steps=[
        ('date_features', FunctionTransformer(create_temporal_features)),
        ('scaler', StandardScaler())
    ])

    # Condition features pipeline
    condition_transformer = Pipeline(steps=[
        ('condition_features', FunctionTransformer(create_condition_features)),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('binary', binary_transformer, binary_features),
        ('ord', ordinal_transformer, ordinal_features),
        ('temp', temporal_transformer, ['admission_date']),
        ('cond', condition_transformer, ['preexisting_condition', 'diagnosis'])
    ], remainder='drop')

    return preprocessor

# Main feature-building function
def build_features(input_path, output_dir):
    # Load data
    df = pd.read_csv(input_path)
    
    # Data validation
    required_columns = (numeric_features + categorical_features + 
                       binary_features + ordinal_features + 
                       ['admission_date', 'preexisting_condition', target_column])
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
    
    # Target validation
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    print(f"Target distribution:\n{df[target_column].describe()}")

    # Create and fit the preprocessor
    preprocessor = create_feature_pipeline()
    features = preprocessor.fit_transform(df)
    
    # Get feature names
    feature_names = []
    
    # Numeric features
    feature_names.extend(numeric_features)
    
    # Categorical features
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    feature_names.extend(cat_encoder.get_feature_names_out(categorical_features))
    
    # Binary features
    binary_encoder = preprocessor.named_transformers_['binary'].named_steps['onehot']
    feature_names.extend(binary_encoder.get_feature_names_out(binary_features))
    
    # Ordinal features
    feature_names.extend(ordinal_features)
    
    # Temporal features
    feature_names.extend(['admission_dayofyear', 'admission_weekday', 'admission_month'])
    
    # Condition features
    feature_names.extend(['has_preexisting_condition', 'has_chronic_condition'])
    
    # Create final DataFrame
    features_df = pd.DataFrame(features, columns=feature_names)
    features_df[target_column] = df[target_column].values
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "processed_features.csv"
    features_df.to_csv(output_path, index=False)
    
    # Save preprocessor as preprocessor.joblib in the models directory
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = models_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    
    print(f"✅ Features built successfully")
    print(f"📊 Final feature matrix shape: {features_df.shape}")
    print(f"💾 Saved features to: {output_path}")
    print(f"🔧 Saved preprocessor to: {preprocessor_path}")

if __name__ == "__main__":
    build_features(
        input_path=project_root / "data/processed/cleaned_patient_data.csv",
        output_dir=project_root / "data/processed"
    )