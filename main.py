"""
FastAPI application for Credit Risk Scoring
Provides REST API endpoints for credit risk prediction
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, Security, status, Depends
from pydantic import BaseModel, Field
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator
from src.logging_config import setup_logging  # noqa: F401  # side-effect import
from src.config import (
    API_TITLE, API_DESCRIPTION, API_VERSION,
    MLFLOW_MODEL_NAME, ALL_FEATURES, API_KEYS,
)
from starlette.concurrency import run_in_threadpool
from fastapi.security.api_key import APIKeyHeader
from src.observability import init_observability, set_model_version
import joblib

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Attach observability (Prometheus custom metrics & tracing)
init_observability(app)

# Global model variable
model = None

# Prometheus instrumentator (will register /metrics)
Instrumentator().instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")

# Security dependency (disabled if no API_KEYS)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Validate X-API-Key header if API keys are configured."""
    if API_KEYS is None:
        return  # Auth disabled
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

class CreditDataInput(BaseModel):
    """Input schema for credit risk prediction - matches training data features"""
    
    # Numeric features
    Duration: int = Field(..., description="Duration of credit in months", ge=1, le=72)
    Credit_amount: float = Field(..., description="Credit amount", ge=0)
    Age: int = Field(..., description="Age in years", ge=18, le=100)
    Installment_rate: int = Field(..., description="Installment rate in percentage of disposable income", ge=1, le=4)
    Present_residence: int = Field(..., description="Present residence since (years)", ge=1, le=4)
    Existing_credits: int = Field(..., description="Number of existing credits at this bank", ge=1, le=4)
    Dependents: int = Field(..., description="Number of people being liable to provide maintenance for", ge=1, le=2)
    
    # Categorical features
    Checking_account: str = Field(..., description="Status of checking account")
    Credit_history: str = Field(..., description="Credit history")
    Purpose: str = Field(..., description="Purpose of credit")
    Savings_account: str = Field(..., description="Savings account status")
    Employment: str = Field(..., description="Employment status")
    Personal_status_sex: str = Field(..., description="Personal status and sex")
    Other_debtors: str = Field(..., description="Other debtors/guarantors")
    Property: str = Field(..., description="Property ownership")
    Other_installment_plans: str = Field(..., description="Other installment plans")
    Housing: str = Field(..., description="Housing situation")
    Job: str = Field(..., description="Job category")
    Telephone: str = Field(..., description="Telephone availability")
    Foreign_worker: str = Field(..., description="Foreign worker status")

    class Config:
        json_schema_extra = {
            "example": {
                "Duration": 12,
                "Credit_amount": 5000.0,
                "Age": 35,
                "Installment_rate": 3,
                "Present_residence": 2,
                "Existing_credits": 1,
                "Dependents": 1,
                "Checking_account": "A11",
                "Credit_history": "A34",
                "Purpose": "A43",
                "Savings_account": "A61",
                "Employment": "A73",
                "Personal_status_sex": "A93",
                "Other_debtors": "A101",
                "Property": "A121",
                "Other_installment_plans": "A143",
                "Housing": "A152",
                "Job": "A173",
                "Telephone": "A192",
                "Foreign_worker": "A201"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for credit risk prediction"""
    risk_prediction: str = Field(..., description="Predicted risk level: Good or Bad")
    risk_probability: float = Field(..., description="Probability of bad credit risk")
    confidence: str = Field(..., description="Confidence level of prediction")


@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model
    model_loaded = False
    
    # Try local joblib file first (more reliable)
    try:
        model_path = os.getenv("FALLBACK_MODEL_PATH", "models/logistic_regression_model.joblib")
        model = joblib.load(model_path)
        set_model_version("local-joblib")
        print(f"✅ Model loaded from {model_path} (joblib)")
        model_loaded = True
    except Exception as e:
        print(f"⚠️  Local model load failed: {e}")
        
        # Fallback: try MLflow
        try:
            # Try specific version first, then latest  
            model_uri = os.getenv("MODEL_URI", f"models:/{MLFLOW_MODEL_NAME}/8")  # Use latest version 8
            model = mlflow.sklearn.load_model(model_uri)  # type: ignore[attr-defined]
            version_label = model_uri.split(":")[-1] if ":" in model_uri else model_uri
            set_model_version(version_label)
            print(f"✅ Model loaded from {model_uri} (MLflow)")
            model_loaded = True
        except Exception as e2:
            print(f"❌ WARNING: Could not load model from local joblib or MLflow. Errors: {e} | {e2}")
            print("⚠️  API will start without a model loaded. Model can be loaded later.")
            model = None  # Set to None, API will return 503 for predictions


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Credit Risk Scoring API",
        "version": API_VERSION,
        "status": "active" if model is not None else "model_not_loaded",
        "endpoints": {
            "prediction": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "api_version": API_VERSION
    }


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict_credit_risk(data: CreditDataInput):
    """
    Predict credit risk for a given customer profile
    
    Args:
        data: Customer information
        
    Returns:
        PredictionResponse: Risk prediction and probability
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server configuration."
        )
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Ensure all required features are present
        missing_features = set(ALL_FEATURES) - set(input_data.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}"
            )
        
        # Workaround for XGBoost use_label_encoder issue
        def fix_xgboost_model(pipeline):
            """Remove deprecated use_label_encoder attribute from XGBoost models"""
            try:
                from xgboost import XGBClassifier
                if hasattr(pipeline, 'named_steps'):
                    for step_name, step in pipeline.named_steps.items():
                        if isinstance(step, XGBClassifier) and hasattr(step, 'use_label_encoder'):
                            delattr(step, 'use_label_encoder')
                elif isinstance(pipeline, XGBClassifier) and hasattr(pipeline, 'use_label_encoder'):
                    delattr(pipeline, 'use_label_encoder')
            except:
                pass  # Ignore if workaround fails
        
        # Apply workaround
        fix_xgboost_model(model)
        
        # Make prediction in threadpool to avoid blocking
        prediction = (await run_in_threadpool(model.predict, input_data))[0]
        prediction_proba = (await run_in_threadpool(model.predict_proba, input_data))[0]
        
        # Convert prediction to readable format
        risk_prediction = "Good" if prediction == 0 else "Bad"
        risk_probability = float(prediction_proba[1])  # Probability of bad credit
        
        # Determine confidence level
        max_proba = max(prediction_proba)
        if max_proba >= 0.8:
            confidence = "High"
        elif max_proba >= 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return PredictionResponse(
            risk_prediction=risk_prediction,
            risk_probability=risk_probability,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 