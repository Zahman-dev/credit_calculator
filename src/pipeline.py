"""
Data preprocessing pipeline for Credit Risk Calculator
Handles both numeric and categorical features with proper scaling and encoding
"""

try:
    import pandas as pd
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
except ImportError as e:
    print(f"Error importing required packages: {e}")
    raise

from .config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN, GERMAN_CREDIT_DATA_PATH
from .data_ingestion import download_and_prepare_german_credit_dataset


class DataPreprocessor:
    """
    Complete data preprocessing pipeline for German Credit Dataset
    """
    
    def __init__(self):
        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        self.categorical_transformer = Pipeline(steps=[ # type: ignore
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer( # type: ignore
            transformers=[
                ('num', self.numeric_transformer, NUMERIC_FEATURES),
                ('cat', self.categorical_transformer, CATEGORICAL_FEATURES)
            ]
        )
        
    def fit(self, X, y=None):
        """Fit the preprocessing pipeline"""
        self.preprocessor.fit(X)
        if y is not None:
            _ = np.where(y == 1, 0, 1)  # type: ignore
        return self
    
    def transform(self, X):
        """Transform the features"""
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X, y=None):
        """Fit and transform the features"""
        return self.fit(X, y).transform(X)
    
    def transform_target(self, y):
        """Transform target variable: 1 (Good) -> 0, 2 (Bad) -> 1"""
        return np.where(y == 1, 0, 1)
    
    def inverse_transform_target(self, y):
        """Inverse transform target variable: 0 -> Good, 1 -> Bad"""
        return np.where(y == 0, "Good", "Bad")
    
    def get_feature_names_out(self):
        """Get feature names after transformation"""
        return self.preprocessor.get_feature_names_out()


def load_german_credit_data(data_path=GERMAN_CREDIT_DATA_PATH):
    """
    Load and prepare German Credit Dataset
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        tuple: (X, y) features and target
    """
    # Ensure dataset exists (automatically download & prepare if necessary)
    if not data_path.exists():
        data_path = download_and_prepare_german_credit_dataset()
    
    # Load data
    df = pd.read_csv(data_path)
    
    # At this stage CSV should have correct header via ingestion module
    
    # Separate features and target
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    
    return X, y


def create_full_pipeline(model):
    """
    Create a complete pipeline with preprocessing and model
    
    Args:
        model: Sklearn-compatible model
        
    Returns:
        Pipeline: Complete pipeline
    """
    preprocessor = DataPreprocessor().preprocessor # type: ignore   
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ]) 