"""
Utility functions for Credit Risk Calculator
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import mlflow
import mlflow.sklearn


def plot_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    model: Any | None = None,
    feature_names: list[str] | None = None,
) -> None:
    """
    Plot comprehensive model performance metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        model_name: Name of the model for plots
        model: Trained model (optional)
        feature_names: List of feature names (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
    axes[0, 0].set_title(f'Confusion Matrix - {model_name}')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    axes[0, 1].plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title(f'ROC Curve - {model_name}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Prediction Distribution
    axes[1, 0].hist(y_proba[y_true == 0], bins=30, alpha=0.7, label='Good Credit', density=True)
    axes[1, 0].hist(y_proba[y_true == 1], bins=30, alpha=0.7, label='Bad Credit', density=True)
    axes[1, 0].set_xlabel('Predicted Probability of Bad Credit')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title(f'Prediction Distribution - {model_name}')
    axes[1, 0].legend()
    
    # --- Feature Importance ---
    if model is not None:
        importance: np.ndarray | None = None

        # scikit-learn compatible estimators expose either `feature_importances_` or `coef_`.
        if hasattr(model, "feature_importances_"):
            importance = getattr(model, "feature_importances_")  # type: ignore[attr-defined]
        elif hasattr(model, "coef_"):
            coef = getattr(model, "coef_")  # type: ignore[attr-defined]
            # For multi-class or binary where coef_ may be 2-D take absolute for magnitude
            importance = np.abs(coef).ravel()

        if importance is not None and len(importance) > 0:
            # Take top 20 features for readability
            top_n = min(20, len(importance))
            indices = np.argsort(importance)[-top_n:][::-1]
            labels = (
                np.array(feature_names)[indices]
                if feature_names is not None and len(feature_names) == len(importance)
                else indices
            )
            sns.barplot(
                x=importance[indices],
                y=labels,
                ax=axes[1, 1],
                orient="h",
            )
            axes[1, 1].set_xlabel("Importance")
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Feature importance not available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Feature importance not provided",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )

    axes[1, 1].set_title(f"Feature Importance - {model_name}")
    
    plt.tight_layout()
    plt.show()


def calculate_business_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              loan_amounts: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate business-relevant metrics for credit risk model
    
    Args:
        y_true: True labels (0=Good, 1=Bad)
        y_pred: Predicted labels (0=Good, 1=Bad)
        loan_amounts: Array of loan amounts (optional)
        
    Returns:
        Dictionary of business metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Basic metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Business metrics
    # False Positive Rate: Good customers rejected
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # False Negative Rate: Bad customers accepted
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    # If loan amounts are provided, calculate financial impact
    if loan_amounts is not None:
        # Assume average loss rate for bad loans
        bad_loan_loss_rate = 0.6  # 60% of loan amount lost on bad loans
        
        # Calculate potential losses
        fn_losses = np.sum(loan_amounts[fn]) * bad_loan_loss_rate  # Losses from accepted bad loans
        fp_opportunity_cost = np.sum(loan_amounts[fp]) * 0.05  # 5% opportunity cost from rejected good loans
        
        metrics.update({
            'estimated_losses_from_bad_loans': fn_losses,
            'opportunity_cost_from_rejected_good_loans': fp_opportunity_cost,
            'total_financial_impact': fn_losses + fp_opportunity_cost
        })
    
    return metrics


def create_model_summary_report(models_results: Dict, test_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary report comparing multiple models
    
    Args:
        models_results: Dictionary with model results
        test_data: Test dataset
        
    Returns:
        DataFrame with model comparison
    """
    summary_data = []
    
    for model_name, results in models_results.items():
        metrics = calculate_business_metrics(
            results['y_true'], 
            results['y_pred']
        )
        
        auc = roc_auc_score(results['y_true'], results['y_proba'])
        
        summary_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Specificity': metrics['specificity'],
            'AUC': auc,
            'False Positive Rate': metrics['false_positive_rate'],
            'False Negative Rate': metrics['false_negative_rate']
        })
    
    return pd.DataFrame(summary_data)


def log_model_to_mlflow(
    model,
    model_name: str,
    metrics: dict[str, float],
    params: dict[str, Any],
) -> str:
    """Log model and metrics to MLflow.
    
    Args:
        model: Trained model
        model_name: Name of the model
        metrics: Model metrics
        params: Model parameters
        
    Returns:
        MLflow run ID
    """
    with mlflow.start_run() as run:
        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(  # type: ignore
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        
        run_id = run.info.run_id

    # `with` block exited – safe to return run_id
    return run_id


def generate_prediction_explanation(prediction: int, probability: float, 
                                  confidence: str, feature_values: Dict) -> str:
    """
    Generate human-readable explanation for prediction
    
    Args:
        prediction: Model prediction (0=Good, 1=Bad)
        probability: Probability of bad credit
        confidence: Confidence level
        feature_values: Dictionary of feature values
        
    Returns:
        Explanation string
    """
    risk_level = "Good Credit" if prediction == 0 else "Bad Credit"
    
    explanation = f"""
    Credit Risk Assessment: {risk_level}
    
    Probability of Default: {probability:.2%}
    Confidence Level: {confidence}
    
    Key Factors:
    - Loan Duration: {feature_values.get('Duration', 'N/A')} months
    - Credit Amount: ${feature_values.get('Credit_amount', 'N/A'):,.2f}
    - Applicant Age: {feature_values.get('Age', 'N/A')} years
    - Credit History: {feature_values.get('Credit_history', 'N/A')}
    - Checking Account: {feature_values.get('Checking_account', 'N/A')}
    
    Recommendation: {'Approve loan' if prediction == 0 else 'Reject loan or require additional review'}
    """
    
    return explanation


def validate_input_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate input data for prediction
    
    Args:
        data: Input DataFrame
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required columns
    from .config import ALL_FEATURES
    missing_features = set(ALL_FEATURES) - set(data.columns)
    if missing_features:
        errors.append(f"Missing required features: {missing_features}")
    
    # Check data types and ranges
    if 'Duration' in data.columns:
        if data['Duration'].min() < 1 or data['Duration'].max() > 72:
            errors.append("Duration must be between 1 and 72 months")
    
    if 'Credit_amount' in data.columns:
        if data['Credit_amount'].min() < 0:
            errors.append("Credit amount must be non-negative")
    
    if 'Age' in data.columns:
        if data['Age'].min() < 18 or data['Age'].max() > 100:
            errors.append("Age must be between 18 and 100 years")
    
    # Check for missing values
    if data.isnull().any().any(): # type: ignore
        null_columns = data.columns[data.isnull().any()].tolist()
        errors.append(f"Missing values found in columns: {null_columns}")
    
    return len(errors) == 0, errors


def create_feature_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary of features in the dataset
    
    Args:
        data: Input DataFrame
        
    Returns:
        DataFrame with feature summary
    """
    summary_data = []
    
    for column in data.columns:
        col_data = data[column]
        
        if col_data.dtype in ['int64', 'float64']:
            summary_data.append({
                'Feature': column,
                'Type': 'Numeric',
                'Min': col_data.min(),
                'Max': col_data.max(),
                'Mean': col_data.mean(),
                'Missing_Values': col_data.isnull().sum(),
                'Unique_Values': col_data.nunique()
            })
        else:
            summary_data.append({
                'Feature': column,
                'Type': 'Categorical',
                'Min': 'N/A',
                'Max': 'N/A',
                'Mean': 'N/A',
                'Missing_Values': col_data.isnull().sum(),
                'Unique_Values': col_data.nunique()
            })
    
    return pd.DataFrame(summary_data) 