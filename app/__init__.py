import os
import sys
from flask import Flask
from pathlib import Path
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

# Initialize SQLAlchemy outside the create_app function
db = SQLAlchemy()


def create_app(test_config=None):
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    # Configure Python path for modular project structure
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    features_path = src_path / "features"
    models_path = src_path / "models"
    utils_path = src_path / "utils"

    for path in [str(src_path), str(features_path), str(models_path), str(utils_path)]:
        if path not in sys.path:
            sys.path.insert(0, path)

    # Basic configuration for SQLite - pointing to database folder
    BASE_DIR = Path(__file__).resolve().parents[1]
    app.config.from_mapping(
        SECRET_KEY='dev',  # Replace with a secure key in production
        SQLALCHEMY_DATABASE_URI=f'sqlite:///{BASE_DIR}/database/patient_recovery.db',
        SQLALCHEMY_TRACK_MODIFICATIONS=False
    )

    # Load additional config from file or test config
    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    # Ensure the database folder exists
    try:
        os.makedirs(BASE_DIR / 'database', exist_ok=True)
    except OSError:
        pass

    # Initialize SQLAlchemy
    db.init_app(app)

    # Create tables if they don't exist
    with app.app_context():
        db.create_all()

        # Create the patients table if it doesn't exist
        db.session.execute(text("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                bmi REAL,
                blood_pressure TEXT,
                heart_rate INTEGER,
                procedures_count INTEGER,
                duration_of_treatment INTEGER,
                gender TEXT,
                admission_reason TEXT,
                admission_type TEXT,
                ward_type TEXT,
                treatment_type TEXT,
                medication_given TEXT,
                diagnosis TEXT,
                smoking_status TEXT,
                complications TEXT,
                severity TEXT,
                admission_date DATE,
                preexisting_condition TEXT,
                predicted_recovery_days INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        db.session.commit()

    # Register your blueprints
    from .routes import main
    app.register_blueprint(main)

    return app