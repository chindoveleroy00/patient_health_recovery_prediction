#!/usr/bin/env python3

# Add this at the top of your run.py file to ensure custom functions are available
# in the main module namespace for unpickling

import sys
from pathlib import Path

# Define the project root and add necessary paths
project_root = Path(__file__).parent
src_path = project_root / "src"
utils_path = project_root / "src" / "utils"
sys.path.extend([str(project_root), str(src_path), str(utils_path)])

# Import all necessary functions for unpickling
from src.utils.utils import bool_to_int
from src.utils.preprocessing_utils import (
    bool_to_int,
    create_temporal_features,
    calculate_bmi_category,
    calculate_age_group,
    calculate_bp_category,
    handle_preexisting_conditions,
    days_since_admission
)

# Make all preprocessing functions available in the main module namespace
# This is necessary for unpickling the preprocessor
for func_name, func in {
    'bool_to_int': bool_to_int,
    'create_temporal_features': create_temporal_features,
    'calculate_bmi_category': calculate_bmi_category,
    'calculate_age_group': calculate_age_group,
    'calculate_bp_category': calculate_bp_category,
    'handle_preexisting_conditions': handle_preexisting_conditions,
    'days_since_admission': days_since_admission,
}.items():
    sys.modules['__main__'].__dict__[func_name] = func

# Import and run the Flask app
from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)