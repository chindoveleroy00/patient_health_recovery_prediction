"""
This module provides all the custom transformation functions needed for the feature preprocessing pipeline.
These functions are required when loading the serialized preprocessor.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

def create_temporal_features(df):
    """
    Extract temporal features from admission date.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing an 'admission_date' column
        
    Returns:
        pd.DataFrame: DataFrame with additional temporal features
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Convert admission_date to datetime if it's not already
    if 'admission_date' in result.columns:
        result['admission_date'] = pd.to_datetime(result['admission_date'], errors='coerce')
        
        # Extract features
        result['admission_day_of_week'] = result['admission_date'].dt.dayofweek
        result['admission_month'] = result['admission_date'].dt.month
        result['admission_is_weekend'] = result['admission_date'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Drop the original admission_date column
        # result = result.drop('admission_date', axis=1)
    
    return result

# More transformation functions can be added here as needed
# For example:

def calculate_bmi_category(df):
    """
    Calculate BMI category based on BMI value.
    """
    result = df.copy()
    
    if 'bmi' in result.columns:
        # Define BMI categories
        result['bmi_category'] = pd.cut(
            result['bmi'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
    
    return result

def calculate_age_group(df):
    """
    Calculate age group based on age value.
    """
    result = df.copy()
    
    if 'age' in result.columns:
        # Define age groups
        result['age_group'] = pd.cut(
            result['age'],
            bins=[0, 18, 35, 50, 65, 120],
            labels=['Child', 'Young Adult', 'Adult', 'Middle Aged', 'Senior']
        )
    
    return result

def calculate_bp_category(df):
    """
    Calculate blood pressure category.
    """
    result = df.copy()
    
    if 'blood_pressure' in result.columns:
        # Define blood pressure categories (systolic)
        result['bp_category'] = pd.cut(
            result['blood_pressure'],
            bins=[0, 90, 120, 140, 160, 300],
            labels=['Low', 'Normal', 'Elevated', 'High', 'Very High']
        )
    
    return result

def handle_preexisting_conditions(df):
    """
    Process preexisting conditions.
    """
    result = df.copy()
    
    if 'preexisting_condition' in result.columns:
        # Convert to lowercase for consistency
        result['preexisting_condition'] = result['preexisting_condition'].str.lower()
        
        # Create binary features for common conditions
        common_conditions = ['diabetes', 'hypertension', 'asthma', 'heart disease', 
                           'cancer', 'copd', 'arthritis', 'none']
        
        for condition in common_conditions:
            col_name = f'condition_{condition.replace(" ", "_")}'
            result[col_name] = result['preexisting_condition'].str.contains(condition, na=False).astype(int)
    
    return result

def days_since_admission(df):
    """
    Calculate days since admission to the current date.
    """
    result = df.copy()
    
    if 'admission_date' in result.columns:
        today = datetime.now().date()
        result['admission_date'] = pd.to_datetime(result['admission_date'], errors='coerce')
        result['days_since_admission'] = (today - result['admission_date'].dt.date).dt.days
    
    return result

# Add any other custom functions that might be used in your preprocessing pipeline