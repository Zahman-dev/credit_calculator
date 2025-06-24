"""
Unit tests for the Credit Risk API
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import pandas as pd
import numpy as np

from main import app

client = TestClient(app)


class TestAPI:
    """Test cases for the API endpoints"""
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "api_version" in data
    
    @patch('main.model')
    def test_predict_endpoint_success(self, mock_model):
        """Test successful prediction"""
        # Mock model behavior
        mock_model.predict.return_value = np.array([0])  # Good credit
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])  # 80% good, 20% bad
        
        # Sample input data
        test_data = {
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
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "risk_prediction" in data
        assert "risk_probability" in data
        assert "confidence" in data
        assert data["risk_prediction"] == "Good"
        assert isinstance(data["risk_probability"], float)
        assert data["confidence"] in ["Low", "Medium", "High"]
    
    def test_predict_endpoint_missing_fields(self):
        """Test prediction with missing required fields"""
        incomplete_data = {
            "Duration": 12,
            "Credit_amount": 5000.0
            # Missing other required fields
        }
        
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_invalid_values(self):
        """Test prediction with invalid field values"""
        invalid_data = {
            "Duration": -5,  # Invalid negative duration
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
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    @patch('main.model', None)
    def test_predict_endpoint_no_model(self):
        """Test prediction when model is not loaded"""
        test_data = {
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
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 503  # Service unavailable


class TestDataValidation:
    """Test cases for data validation"""
    
    def test_age_validation(self):
        """Test age field validation"""
        # Test minimum age
        data = self._get_valid_data()
        data["Age"] = 17  # Below minimum
        
        response = client.post("/predict", json=data)
        assert response.status_code == 422
        
        # Test maximum age
        data["Age"] = 101  # Above maximum
        response = client.post("/predict", json=data)
        assert response.status_code == 422
    
    def test_duration_validation(self):
        """Test duration field validation"""
        data = self._get_valid_data()
        data["Duration"] = 0  # Below minimum
        
        response = client.post("/predict", json=data)
        assert response.status_code == 422
        
        data["Duration"] = 73  # Above maximum
        response = client.post("/predict", json=data)
        assert response.status_code == 422
    
    def test_credit_amount_validation(self):
        """Test credit amount field validation"""
        data = self._get_valid_data()
        data["Credit_amount"] = -100  # Negative amount
        
        response = client.post("/predict", json=data)
        assert response.status_code == 422
    
    def _get_valid_data(self):
        """Helper method to get valid test data"""
        return {
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


class TestModelIntegration:
    """Integration tests with actual model components"""
    
    def test_pipeline_integration(self):
        """Test the complete pipeline integration"""
        from src.pipeline import create_full_pipeline
        from sklearn.linear_model import LogisticRegression
        
        # Create a simple pipeline
        model = LogisticRegression(random_state=42)
        pipeline = create_full_pipeline(model)
        
        # Create sample data
        sample_data = pd.DataFrame({
            'Duration': [12, 24],
            'Credit_amount': [5000, 10000],
            'Age': [35, 45],
            'Installment_rate': [2, 4],
            'Present_residence': [1, 3],
            'Existing_credits': [1, 2],
            'Dependents': [1, 1],
            'Checking_account': ['A11', 'A12'],
            'Credit_history': ['A34', 'A33'],
            'Purpose': ['A43', 'A42'],
            'Savings_account': ['A61', 'A62'],
            'Employment': ['A73', 'A74'],
            'Personal_status_sex': ['A93', 'A92'],
            'Other_debtors': ['A101', 'A102'],
            'Property': ['A121', 'A122'],
            'Other_installment_plans': ['A143', 'A141'],
            'Housing': ['A152', 'A151'],
            'Job': ['A173', 'A172'],
            'Telephone': ['A192', 'A191'],
            'Foreign_worker': ['A201', 'A202']
        })
        
        # Mock target data for fitting
        y_mock = [0, 1]  # Good, Bad
        
        # Test that pipeline can fit and predict
        pipeline.fit(sample_data, y_mock)
        predictions = pipeline.predict(sample_data)
        probabilities = pipeline.predict_proba(sample_data)
        
        assert len(predictions) == len(sample_data)
        assert probabilities.shape == (len(sample_data), 2)
        assert all(pred in [0, 1] for pred in predictions)


if __name__ == "__main__":
    pytest.main([__file__]) 