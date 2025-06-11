import pandas as pd
import numpy as np
import os
from typing import Tuple

def clean_data(raw_path: str, output_path: str) -> Tuple[str, dict]:
    """
    Handles missing values, outliers, and data validation
    
    Args:
        raw_path: Path to raw CSV data
        output_path: Directory to save cleaned data
        
    Returns:
        Tuple of (output_file_path, cleaning_report)
    """
    df = pd.read_csv(raw_path)
    report = {'original_rows': len(df)}

    # Handle missing values
    df = df.dropna(subset=['age', 'gender', 'diagnosis', 'admission_date'])
    report['rows_after_dropna'] = len(df)

    # Parse admission_date
    df['admission_date'] = pd.to_datetime(df['admission_date'], errors='coerce')
    df = df.dropna(subset=['admission_date'])  # Drop rows with unparseable dates

    # Feature Engineering: extract date features
    df['admission_dayofyear'] = df['admission_date'].dt.dayofyear
    df['admission_weekday'] = df['admission_date'].dt.weekday

    # Remove unrealistic values
    df = df[(df['age'] > 0) & (df['age'] < 120)]
    df = df[df['heart_rate'].between(40, 200)]
    report['rows_after_filtering'] = len(df)

    # Save cleaned data
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "cleaned_patient_data.csv")
    df.to_csv(output_file, index=False)

    return output_file, report

if __name__ == "__main__":
    import json
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    cleaned_file, report = clean_data(
        raw_path=project_root/"data/raw/synthetic_patient_data.csv",
        output_path=project_root/"data/processed"
    )

    # Save cleaning report
    with open(project_root/"reports/data_cleaning_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Cleaned data saved to: {cleaned_file}")
