import os
import sys
from flask import Flask
from pathlib import Path
from flask_mysqldb import MySQL

# Initialize MySQL outside the create_app function
mysql = MySQL()

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

    # Basic configuration for XAMPP MySQL (phpMyAdmin)
    app.config.from_mapping(
        SECRET_KEY='dev',  # Replace with a secure key in production
        MYSQL_HOST='localhost',
        MYSQL_USER='root',
        MYSQL_PASSWORD='',  # Set your MySQL password if it's not empty
        MYSQL_DB='patient_recovery',
        MYSQL_CURSORCLASS='DictCursor'  # Optional: for dict-like results
    )

    # Load additional config from file or test config
    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Initialize MySQL
    mysql.init_app(app)

    # Register your blueprints
    from .routes import main
    app.register_blueprint(main)

    return app
