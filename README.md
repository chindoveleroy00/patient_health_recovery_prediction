# Patient Recovery Prediction

## Project Overview
Predicts patient recovery times using machine learning.

## Workflow
1. `make_dataset.py` - Ingests raw data
2. `build_features.py` - Creates ML features
3. `train_model.py` - Trains prediction model
4. `predict_model.py` - Generates predictions

## Setup
```bash
pip install -r requirements.txt
python src/data/make_dataset.py
python src/models/train_model.py