"""
Configuration file for Credit Risk Calculator
Contains feature definitions, model parameters, and file paths
"""

from pathlib import Path

# ==============================================================
# Dynamic Configuration via environment variables (.env)        
# This module now leverages `pydantic-settings` (Pydantic v2)    
# to allow runtime overrides while koruma existing constants.    
# ==============================================================

from pydantic_settings import BaseSettings, SettingsConfigDict # type: ignore
from typing import List


# ---------------------------
# 1) Settings Definition
# ---------------------------


class Settings(BaseSettings):
    """Project settings that can be overridden via environment variables."""

    # Meta
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown env vars
    )

    # ----- Paths -----
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_DIR: Path = BASE_DIR / "models"
    NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"

    # ----- MLflow -----
    MLFLOW_EXPERIMENT_NAME: str = "credit_risk_scoring"
    MLFLOW_MODEL_NAME: str = "credit_risk_model"

    # ----- API -----
    API_TITLE: str = "Credit Risk Scoring API"
    API_DESCRIPTION: str = "API for predicting credit risk based on customer information"
    API_VERSION: str = "1.0.0"

    # ----- Security -----
    API_KEYS: str | None = None  # Comma separated list of valid API keys

    # ----- Logging -----
    LOG_LEVEL: str = "INFO"


# Singleton settings object
settings = Settings()


# ---------------------------
# 2) Backward-compatible constants
# ---------------------------

# Paths
BASE_DIR: Path = settings.BASE_DIR
DATA_DIR: Path = settings.DATA_DIR
MODEL_DIR: Path = settings.MODEL_DIR
NOTEBOOKS_DIR: Path = settings.NOTEBOOKS_DIR

# Data file path
GERMAN_CREDIT_DATA_PATH: Path = DATA_DIR / "german_credit_data.csv"

# Feature definitions for German Credit Dataset
# Based on UCI German Credit Data documentation (20 attributes)
NUMERIC_FEATURES = [
    'Duration',          # Duration in months
    'Credit_amount',     # Credit amount  
    'Age',              # Age in years
    'Installment_rate', # Installment rate in percentage of disposable income
    'Present_residence', # Present residence since (years)
    'Existing_credits',  # Number of existing credits at this bank
    'Dependents'        # Number of people being liable to provide maintenance for
]

CATEGORICAL_FEATURES = [
    'Checking_account',      # Status of existing checking account
    'Credit_history',        # Credit history
    'Purpose',              # Purpose of credit
    'Savings_account',       # Savings account/bonds
    'Employment',           # Present employment since
    'Personal_status_sex',   # Personal status and sex
    'Other_debtors',        # Other debtors / guarantors
    'Property',             # Property
    'Other_installment_plans', # Other installment plans
    'Housing',              # Housing
    'Job',                  # Job
    'Telephone',            # Telephone
    'Foreign_worker'        # Foreign worker
]

# All features (for validation)
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Target variable
TARGET_COLUMN = 'Risk'  # 1 = Good credit risk, 2 = Bad credit risk

# Model parameters
MODEL_PARAMS = {
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'eval_metric': 'logloss'
    },
    'logistic_regression': {
        'random_state': 42,
        'max_iter': 1000
    }
}

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = settings.MLFLOW_EXPERIMENT_NAME
MLFLOW_MODEL_NAME = settings.MLFLOW_MODEL_NAME

# API configuration
API_TITLE = settings.API_TITLE
API_DESCRIPTION = settings.API_DESCRIPTION
API_VERSION = settings.API_VERSION

# Security
API_KEYS: list[str] | None = (
    settings.API_KEYS.split(",") if settings.API_KEYS else None
)

# Logging
LOG_LEVEL: str = settings.LOG_LEVEL