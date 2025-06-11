from setuptools import setup, find_packages

setup(
    name="patient_recovery",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'jupyter',
        'joblib',
        'python-dotenv',
        'xgboost',
        'flask',
        'flask-sqlalchemy',
        'flask-migrate',
        'flask-login',
        'flask-wtf',
        'mysqlclient'
    ],
    python_requires='>=3.8'
)