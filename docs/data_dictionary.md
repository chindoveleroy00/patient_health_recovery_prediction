# Data Dictionary

## Patient Data

| Field | Description | Type | Notes |
|-------|-------------|------|-------|
| patient_id | Unique identifier for patient | string | Primary key |
| age | Patient age in years | integer | |
| gender | Patient gender | categorical | Male, Female, Other |
| diagnosis | Primary diagnosis | categorical | |
| comorbidities | Secondary health conditions | list | Comma-separated list |
| admission_date | Date of hospital admission | date | YYYY-MM-DD format |
| discharge_date | Date of hospital discharge | date | YYYY-MM-DD format |
| length_of_stay | Days in hospital | integer | Target variable |
| readmission | Whether patient was readmitted | boolean | Secondary target |

## Treatment Data

| Field | Description | Type | Notes |
|-------|-------------|------|-------|
| treatment_id | Unique identifier for treatment | string | Primary key |
| patient_id | Patient identifier | string | Foreign key |
| treatment_type | Type of treatment | categorical | |
| medication | Medications prescribed | list | Comma-separated list |
| procedure | Medical procedures performed | list | Comma-separated list |
| outcome | Treatment outcome | categorical | Successful, Partial, Failed |
