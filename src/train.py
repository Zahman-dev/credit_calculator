"""
Model training script for Credit Risk Calculator
Trains models with MLflow experiment tracking
"""

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import os
import numpy as np

from .config import (
    GERMAN_CREDIT_DATA_PATH, MODEL_PARAMS, MLFLOW_EXPERIMENT_NAME, 
    MLFLOW_MODEL_NAME, MODEL_DIR
)
from .pipeline import load_german_credit_data, create_full_pipeline, DataPreprocessor


def setup_mlflow():
    """Setup MLflow experiment"""
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def train_model(model_type='xgboost', test_size=0.2, random_state=42, save_local: bool | None = None):
    """
    Train a credit risk model with MLflow tracking
    
    Args:
        model_type: Type of model ('xgboost' or 'logistic_regression')
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        save_local: Whether to save the model locally
    """
    
    # Setup MLflow
    setup_mlflow()
    
    # Load and prepare data
    print("Loading data...")
    X, y = load_german_credit_data(GERMAN_CREDIT_DATA_PATH)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Transform target variable (1=Good -> 0, 2=Bad -> 1)
    y_transformed = preprocessor.transform_target(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_transformed, test_size=test_size, random_state=random_state, stratify=y_transformed
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Target distribution - Good: {np.sum(y_train == 0)}, Bad: {np.sum(y_train == 1)}")
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_size_ratio", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        # Initialize model
        if model_type == 'xgboost':
            model = XGBClassifier(**MODEL_PARAMS['xgboost'])
        else:
            model = LogisticRegression(**MODEL_PARAMS['logistic_regression'])
        
        # Log model parameters
        for param_name, param_value in model.get_params().items():
            mlflow.log_param(f"model_{param_name}", param_value)
        
        # Create full pipeline
        pipeline = create_full_pipeline(model)
        
        # Train model
        print(f"Training {model_type} model...")
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        y_train_proba = pipeline.predict_proba(X_train)[:, 1]
        y_test_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_accuracy = (y_train_pred == y_train).mean()
        test_accuracy = (y_test_pred == y_test).mean()
        train_auc = roc_auc_score(y_train, y_train_proba)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Log metrics
        mlflow.log_metric("train_accuracy", float(train_accuracy))
        mlflow.log_metric("test_accuracy", float(test_accuracy))
        mlflow.log_metric("train_auc", float(train_auc))
        mlflow.log_metric("test_auc", float(test_auc))
        mlflow.log_metric("cv_auc_mean", float(cv_mean))
        mlflow.log_metric("cv_auc_std", float(cv_std))
        
        # Print results
        print("\n=== Model Performance ===")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Training AUC: {train_auc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Cross-Validation AUC: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        
        # Classification report
        print("\n=== Classification Report (Test Set) ===")
        target_names = ['Good Credit', 'Bad Credit']
        print(classification_report(y_test, y_test_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        print("\n=== Confusion Matrix ===")
        print(f"True Negatives (Good predicted as Good): {cm[0, 0]}")
        print(f"False Positives (Good predicted as Bad): {cm[0, 1]}")
        print(f"False Negatives (Bad predicted as Good): {cm[1, 0]}")
        print(f"True Positives (Bad predicted as Bad): {cm[1, 1]}")
        
        # Log model
        mlflow.sklearn.log_model(  # type: ignore
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
            input_example=X_train.head() # type: ignore
        )
        
        # Optionally save model locally
        if save_local is None:
            # Fallback to environment variable if CLI arg not provided
            save_local_env = os.getenv("SAVE_LOCAL_MODEL", "false").lower()
            save_local = save_local_env in {"1", "true", "yes"}

        if save_local:
            MODEL_DIR.mkdir(exist_ok=True)
            model_path = MODEL_DIR / f"{model_type}_model.joblib"
            joblib.dump(pipeline, model_path)
            print(f"\nModel saved locally: {model_path}")
        
        # Log run info
        run_id = mlflow.active_run().info.run_id # type: ignore
        print(f"\nMLflow Run ID: {run_id}")
        print(f"MLflow Model URI: runs:/{run_id}/model")
        
        return pipeline, run_id


def compare_models():
    """Train and compare multiple models"""
    print("=== Comparing Models ===")
    
    results = {}
    
    for model_type in ['logistic_regression', 'xgboost']:
        print(f"\n--- Training {model_type} ---")
        try:
            pipeline, run_id = train_model(model_type) # type: ignore
            results[model_type] = {
                'pipeline': pipeline,
                'run_id': run_id
            }
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            results[model_type] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Credit Risk Model")
    parser.add_argument(
        "--model", 
        choices=['xgboost', 'logistic_regression', 'compare'],
        default='xgboost',
        help="Model type to train"
    )
    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2,
        help="Test set size (default: 0.2)"
    )
    parser.add_argument(
        "--save-local",
        action="store_true",
        help="If provided, also save trained model as joblib under models/ directory",
    )
    
    args = parser.parse_args()
    
    if args.model == "compare":
        compare_models()
    else:
        train_model(args.model, test_size=args.test_size, save_local=args.save_local) 