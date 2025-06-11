# Patient Health Recovery Prediction System

## Overview
The Patient Health Recovery Prediction System is a machine learning-powered application designed to assist healthcare providers at Chitungwiza Central Hospital in Zimbabwe with predicting patient recovery durations. This system transforms underutilized patient data into actionable insights, enabling proactive care planning and resource allocation in resource-constrained environments.

## Key Features
- **Recovery Duration Prediction**: Uses XGBoost machine learning model to estimate patient recovery time in days
- **Comprehensive Patient Assessment**: Analyzes multiple clinical and demographic factors, including:
  - Vital signs (blood pressure, heart rate, BMI)
  - Treatment details (procedures count, duration, medication)
  - Admission circumstances (reason, type, ward)
  - Medical history (preexisting conditions, diagnosis)
- **Risk Stratification**: Identifies patients at risk of prolonged recovery
- **Web Interface**: Intuitive form for healthcare staff to input patient data
- **Data Integration**: Stores predictions and patient data in MySQL database
- **Batch Processing**: Supports CSV file uploads for multiple patient predictions

## Technical Specifications
### Backend
- **Machine Learning Framework**: XGBoost with Scikit-learn preprocessing
- **Feature Engineering**:
  - Temporal feature extraction (admission day, weekday, month)
  - Condition flags (chronic conditions, preexisting issues)
  - Standardized preprocessing pipeline for consistent predictions
- **Web Framework**: Flask with MySQL integration
- **API Endpoint**: Supports single and batch predictions

### Frontend
- **Form Validation**: WTForms with comprehensive input validation
- **Dynamic Dropdowns**: Clinically relevant options for medical fields
- **Result Visualization**: Clear presentation of predicted recovery duration

### Data Requirements
The system requires the following patient data for accurate predictions:
- **Demographics**: Age, gender
- **Clinical Measurements**: BMI, blood pressure, heart rate
- **Treatment Details**: Procedures count, duration, medication
- **Admission Information**: Reason, type, ward, date
- **Medical History**: Diagnosis, preexisting conditions, complications
- **Lifestyle Factors**: Smoking status

## Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up MySQL database:
   - Configure connection in `__init__.py`
   - Create 'patient_recovery' database
   - Set up patient's table with the required fields
4. Run the application:
   ```bash
   python run.py
   ```

## Usage
1. Access the web interface at `http://localhost:5000`
2. Complete the patient assessment form with all required information
3. Submit to receive the predicted recovery duration
4. Review results and recommended care plan

For batch processing:
```bash
POST /predict_batch
Content-Type: multipart/form-data

[Upload CSV file with patient data]
```

## Project Significance
This system addresses critical challenges at Chitungwiza Central Hospital by:
- Replacing inconsistent manual assessments with data-driven predictions
- Enabling early identification of at-risk patients
- Optimizing limited hospital resources through better planning
- Providing standardized recovery monitoring across patient groups
- Leveraging existing patient data that was previously underutilized

## Future Enhancements
- Integration with hospital EHR systems
- Mobile interface for bedside assessments
- Dynamic care recommendations based on predictions
- Model retraining with new patient outcomes
- Expanded prediction capabilities (complication risks, readmission likelihood)


## Acknowledgments
Chitungwiza Central Hospital for providing the clinical context and data requirements that shaped this project's development.
