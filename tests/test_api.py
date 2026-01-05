#!/usr/bin/env python3
"""
Unit Tests for API
Task 5: CI/CD Pipeline & Automated Testing
"""

import os
import sys
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    """Create test client fixture."""
    return TestClient(app)


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_200(self, client):
        """Test that root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_returns_message(self, client):
        """Test that root endpoint returns expected message."""
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "Heart Disease" in data["message"]


class TestHealthEndpoint:
    """Tests for health endpoint."""
    
    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_returns_status(self, client):
        """Test that health endpoint returns status."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data


class TestPredictEndpoint:
    """Tests for predict endpoint."""
    
    @pytest.fixture
    def valid_input(self):
        """Valid input data."""
        return {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 1,
            "ca": 0,
            "thal": 6
        }
    
    def test_predict_returns_200(self, client, valid_input):
        """Test that predict endpoint returns 200 for valid input."""
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 200
    
    def test_predict_returns_prediction(self, client, valid_input):
        """Test that predict endpoint returns prediction."""
        response = client.post("/predict", json=valid_input)
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in [0, 1]
    
    def test_predict_returns_confidence(self, client, valid_input):
        """Test that predict endpoint returns confidence."""
        response = client.post("/predict", json=valid_input)
        data = response.json()
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1
    
    def test_predict_returns_risk_level(self, client, valid_input):
        """Test that predict endpoint returns risk level."""
        response = client.post("/predict", json=valid_input)
        data = response.json()
        assert "risk_level" in data
        assert data["risk_level"] in ["Low", "Moderate", "High", "Very High"]
    
    def test_predict_invalid_age(self, client, valid_input):
        """Test that predict endpoint rejects invalid age."""
        valid_input["age"] = -1
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 422
    
    def test_predict_missing_field(self, client, valid_input):
        """Test that predict endpoint rejects missing field."""
        del valid_input["age"]
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 422
    
    def test_predict_invalid_sex(self, client, valid_input):
        """Test that predict endpoint rejects invalid sex value."""
        valid_input["sex"] = 5
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 422


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""
    
    def test_model_info_returns_200(self, client):
        """Test that model info endpoint returns 200."""
        response = client.get("/model/info")
        assert response.status_code == 200
    
    def test_model_info_returns_features(self, client):
        """Test that model info endpoint returns features."""
        response = client.get("/model/info")
        data = response.json()
        assert "features" in data
        assert len(data["features"]) == 13


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""
    
    def test_metrics_returns_200(self, client):
        """Test that metrics endpoint returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200
    
    def test_metrics_contains_prometheus_format(self, client):
        """Test that metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        assert "predictions_total" in response.text or "api_requests_total" in response.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
