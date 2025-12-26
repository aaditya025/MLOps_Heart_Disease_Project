#!/usr/bin/env python3
"""
Unit Tests for Model
Task 5: CI/CD Pipeline & Automated Testing
"""

import os
import sys
import pytest
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'heart_disease.csv')


class TestModelFiles:
    """Tests for model file existence."""
    
    def test_pipeline_exists(self):
        """Test that pipeline file exists."""
        pipeline_path = os.path.join(MODELS_DIR, 'pipeline.pkl')
        assert os.path.exists(pipeline_path), "Pipeline file not found"
    
    def test_scaler_exists(self):
        """Test that scaler file exists."""
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        assert os.path.exists(scaler_path), "Scaler file not found"
    
    def test_feature_names_exists(self):
        """Test that feature names file exists."""
        features_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
        assert os.path.exists(features_path), "Feature names file not found"
    
    def test_best_model_exists(self):
        """Test that best model file exists."""
        model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
        assert os.path.exists(model_path), "Best model file not found"


class TestModelLoading:
    """Tests for model loading."""
    
    def test_pipeline_loads(self):
        """Test that pipeline loads without errors."""
        pipeline_path = os.path.join(MODELS_DIR, 'pipeline.pkl')
        pipeline = joblib.load(pipeline_path)
        assert pipeline is not None
    
    def test_scaler_loads(self):
        """Test that scaler loads without errors."""
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        scaler = joblib.load(scaler_path)
        assert scaler is not None
    
    def test_feature_names_loads(self):
        """Test that feature names load without errors."""
        features_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
        features = joblib.load(features_path)
        assert features is not None
        assert len(features) == 13


class TestModelPrediction:
    """Tests for model prediction."""
    
    @pytest.fixture
    def pipeline(self):
        """Load pipeline fixture."""
        pipeline_path = os.path.join(MODELS_DIR, 'pipeline.pkl')
        return joblib.load(pipeline_path)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample input data."""
        return np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 1, 0, 6]])
    
    def test_predict_returns_binary(self, pipeline, sample_data):
        """Test that prediction returns binary values."""
        prediction = pipeline.predict(sample_data)
        assert prediction[0] in [0, 1]
    
    def test_predict_proba_returns_probabilities(self, pipeline, sample_data):
        """Test that predict_proba returns valid probabilities."""
        proba = pipeline.predict_proba(sample_data)
        assert proba.shape == (1, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all() and (proba <= 1).all()
    
    def test_batch_prediction(self, pipeline):
        """Test batch prediction."""
        batch_data = np.array([
            [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 1, 0, 6],
            [45, 0, 2, 120, 200, 0, 1, 170, 0, 1.0, 2, 0, 3],
            [55, 1, 4, 160, 280, 1, 2, 140, 1, 2.0, 3, 2, 7]
        ])
        predictions = pipeline.predict(batch_data)
        assert len(predictions) == 3
        assert all(p in [0, 1] for p in predictions)


class TestModelPerformance:
    """Tests for model performance."""
    
    @pytest.fixture
    def pipeline(self):
        """Load pipeline fixture."""
        pipeline_path = os.path.join(MODELS_DIR, 'pipeline.pkl')
        return joblib.load(pipeline_path)
    
    @pytest.fixture
    def test_data(self):
        """Load test data."""
        df = pd.read_csv(DATA_PATH)
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        X = df[feature_names].values
        y = df['target'].values
        return X, y
    
    def test_accuracy_threshold(self, pipeline, test_data):
        """Test that model achieves minimum accuracy."""
        X, y = test_data
        predictions = pipeline.predict(X)
        accuracy = accuracy_score(y, predictions)
        assert accuracy >= 0.70, f"Accuracy {accuracy:.4f} below threshold 0.70"
    
    def test_roc_auc_threshold(self, pipeline, test_data):
        """Test that model achieves minimum ROC-AUC."""
        X, y = test_data
        proba = pipeline.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, proba)
        assert roc_auc >= 0.75, f"ROC-AUC {roc_auc:.4f} below threshold 0.75"
    
    def test_no_always_same_prediction(self, pipeline, test_data):
        """Test that model doesn't always predict the same class."""
        X, y = test_data
        predictions = pipeline.predict(X)
        unique_predictions = set(predictions)
        assert len(unique_predictions) > 1, "Model always predicts same class"


class TestModelReproducibility:
    """Tests for model reproducibility."""
    
    def test_consistent_predictions(self):
        """Test that predictions are consistent across loads."""
        pipeline_path = os.path.join(MODELS_DIR, 'pipeline.pkl')
        sample_data = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 1, 0, 6]])
        
        # Load and predict multiple times
        results = []
        for _ in range(3):
            pipeline = joblib.load(pipeline_path)
            pred = pipeline.predict(sample_data)[0]
            proba = pipeline.predict_proba(sample_data)[0, 1]
            results.append((pred, proba))
        
        # Check consistency
        assert all(r[0] == results[0][0] for r in results), "Predictions not consistent"
        assert all(abs(r[1] - results[0][1]) < 1e-6 for r in results), "Probabilities not consistent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
