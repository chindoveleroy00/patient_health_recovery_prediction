import os
from pathlib import Path

class Config:
    SECRET_KEY = 'your-secret-key'
    # SQLite database configuration - pointing to database folder
    BASE_DIR = Path(__file__).resolve().parent
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{BASE_DIR}/database/patient_recovery.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False