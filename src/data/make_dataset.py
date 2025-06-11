import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta


def generate_synthetic_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generates synthetic patient recovery data"""
    np.random.seed(42)

    def random_conditions():
        conditions = ['None', 'Diabetes', 'Hypertension', 'Asthma', 'Cardiac Disease']
        return np.random.choice(conditions, p=[0.4, 0.2, 0.2, 0.1, 0.1])

    def random_diagnosis():
        # Add a random diagnosis that could be related to the conditions
        diagnoses = ['Acute', 'Chronic', 'Infectious', 'Neurological', 'None']
        return np.random.choice(diagnoses, p=[0.3, 0.3, 0.2, 0.1, 0.1])

    def calculate_recovery_time(row):
        base = 5
        if row['severity'] == 'Severe':
            base += 10
        elif row['severity'] == 'Moderate':
            base += 5
        if row['preexisting_condition'] != 'None': base += 3
        if row['treatment_type'] == 'Surgery': base += 7
        if row['complications']: base += 5
        return max(base + np.random.randint(-2, 3), 1)

    def random_admission_date():
        # Generate a random admission date in the past year
        start_date = datetime(2023, 1, 1)  # Adjust the start date if needed
        end_date = datetime(2024, 1, 1)  # Adjust the end date if needed
        delta = end_date - start_date
        random_days = np.random.randint(delta.days)
        admission_date = start_date + timedelta(days=random_days)
        return admission_date.strftime('%Y-%m-%d')  # Format the date as 'YYYY-MM-DD'

    data = pd.DataFrame({
        "patient_id": [f"PT_{i:06d}" for i in range(n_samples)],
        "age": np.random.randint(18, 90, n_samples),
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "bmi": np.round(np.random.normal(25, 5, n_samples), 1),
        "smoking_status": np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
        "preexisting_condition": [random_conditions() for _ in range(n_samples)],
        "diagnosis": [random_diagnosis() for _ in range(n_samples)],  # New 'diagnosis' column
        "admission_reason": np.random.choice(["Infection", "Surgery", "Accident", "Chronic"], n_samples),
        "admission_type": np.random.choice(["Emergency", "Elective", "Urgent", "Newborn"], n_samples),
        "severity": np.random.choice(["Mild", "Moderate", "Severe"], n_samples, p=[0.3, 0.5, 0.2]),
        "ward_type": np.random.choice(["ICU", "General", "Emergency"], n_samples),
        "treatment_type": np.random.choice(["Surgery", "Medication", "Therapy"], n_samples),
        "duration_of_treatment": np.random.randint(1, 30, n_samples),  # in days
        "medication_given": np.random.choice(["Antibiotics", "Painkillers", "Steroids", "None"], n_samples),
        "procedures_count": np.random.randint(0, 5, n_samples),
        "blood_pressure": np.round(np.random.normal(120, 15, n_samples), 1),
        "heart_rate": np.round(np.random.normal(80, 10, n_samples), 1),
        "complications": np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
        "length_of_stay": np.random.randint(1, 20, n_samples),  # in days
        "recovery_days": None,  # Will be calculated
        "admission_date": [random_admission_date() for _ in range(n_samples)]  # New 'admission_date' column
    })

    data["recovery_days"] = data.apply(calculate_recovery_time, axis=1)
    return data


def save_dataset(df: pd.DataFrame, output_dir: Path) -> Path:
    """Saves dataset to CSV with validation"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_patient_data.csv"

    # Basic validation
    assert not df.isnull().any().any(), "Data contains null values"
    assert (df['recovery_days'] > 0).all(), "Invalid recovery times"

    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    try:
        project_root = Path(__file__).parent.parent.parent
        raw_data_path = project_root / "data/raw"

        logging.info("Generating synthetic data...")
        df = generate_synthetic_data(n_samples=100000)

        logging.info("Saving dataset...")
        saved_path = save_dataset(df, raw_data_path)

        logging.info(f"✅ Dataset created ({len(df):,} records)\nSaved to: {saved_path}")
    except Exception as e:
        logging.error(f"❌ Error: {str(e)}")
        sys.exit(1)
