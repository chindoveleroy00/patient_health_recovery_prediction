# Modeling Strategy

## Problem Framing
- **Target Variable**: `recovery_days` (regression)
- **Secondary Target**: `readmission_risk` (binary classification - future work)

## Feature Groups
1. **Demographics**: age, gender
2. **Vitals**: blood_pressure, heart_rate
3. **Diagnosis**: Encoded medical condition

## Model Selection
1. **Baseline**: Linear Regression (MAE metric)
2. **Primary**: Random Forest (with feature importance)
3. **Advanced**: XGBoost (if non-linear patterns detected)

## Validation
- 80/20 train-test split
- 5-fold cross-validation
- MAE + R² metrics