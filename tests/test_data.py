#!/usr/bin/env python3
"""
Unit Tests for Data Processing
Task 5: CI/CD Pipeline & Automated Testing
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'heart_disease.csv')


class TestDataLoading:
    """Tests for data loading functionality."""
    
    def test_data_file_exists(self):
        """Test that data file exists."""
        assert os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}"
    
    def test_data_loads_correctly(self):
        """Test that data loads without errors."""
        df = pd.read_csv(DATA_PATH)
        assert df is not None
        assert len(df) > 0
    
    def test_data_has_expected_columns(self):
        """Test that data has all expected columns."""
        expected_columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
            'ca', 'thal', 'target'
        ]
        df = pd.read_csv(DATA_PATH)
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_data_shape(self):
        """Test that data has expected shape."""
        df = pd.read_csv(DATA_PATH)
        assert df.shape[1] == 14, f"Expected 14 columns, got {df.shape[1]}"
        assert df.shape[0] > 200, f"Expected > 200 rows, got {df.shape[0]}"


class TestDataQuality:
    """Tests for data quality."""
    
    def test_no_missing_values(self):
        """Test that data has no missing values."""
        df = pd.read_csv(DATA_PATH)
        assert df.isnull().sum().sum() == 0, "Data contains missing values"
    
    def test_target_is_binary(self):
        """Test that target variable is binary."""
        df = pd.read_csv(DATA_PATH)
        unique_targets = df['target'].unique()
        assert set(unique_targets).issubset({0, 1}), f"Target not binary: {unique_targets}"
    
    def test_age_range(self):
        """Test that age values are in valid range."""
        df = pd.read_csv(DATA_PATH)
        assert df['age'].min() >= 0, "Negative age values found"
        assert df['age'].max() <= 120, "Unrealistic age values found"
    
    def test_sex_values(self):
        """Test that sex values are valid."""
        df = pd.read_csv(DATA_PATH)
        assert set(df['sex'].unique()).issubset({0, 1}), "Invalid sex values"
    
    def test_blood_pressure_range(self):
        """Test that blood pressure values are realistic."""
        df = pd.read_csv(DATA_PATH)
        assert df['trestbps'].min() >= 50, "Blood pressure too low"
        assert df['trestbps'].max() <= 250, "Blood pressure too high"
    
    def test_cholesterol_range(self):
        """Test that cholesterol values are realistic."""
        df = pd.read_csv(DATA_PATH)
        assert df['chol'].min() >= 100, "Cholesterol too low"
        assert df['chol'].max() <= 600, "Cholesterol too high"
    
    def test_heart_rate_range(self):
        """Test that max heart rate values are realistic."""
        df = pd.read_csv(DATA_PATH)
        assert df['thalach'].min() >= 50, "Heart rate too low"
        assert df['thalach'].max() <= 250, "Heart rate too high"


class TestDataPreprocessing:
    """Tests for data preprocessing."""
    
    def test_feature_scaling(self):
        """Test that feature scaling works correctly."""
        from sklearn.preprocessing import StandardScaler
        
        df = pd.read_csv(DATA_PATH)
        features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        X = df[features].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Check mean is approximately 0
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        # Check std is approximately 1
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)
    
    def test_train_test_split(self):
        """Test that train/test split works correctly."""
        from sklearn.model_selection import train_test_split
        
        df = pd.read_csv(DATA_PATH)
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Check sizes
        assert len(X_train) == int(len(df) * 0.8)
        assert len(X_test) == len(df) - len(X_train)
        
        # Check stratification
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05


class TestFeatureEngineering:
    """Tests for feature engineering."""
    
    def test_all_features_numeric(self):
        """Test that all features are numeric."""
        df = pd.read_csv(DATA_PATH)
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} is not numeric"
    
    def test_no_infinite_values(self):
        """Test that no infinite values exist."""
        df = pd.read_csv(DATA_PATH)
        assert not np.isinf(df.values).any(), "Infinite values found"
    
    def test_class_balance(self):
        """Test class balance is reasonable."""
        df = pd.read_csv(DATA_PATH)
        class_ratio = df['target'].value_counts()[1] / df['target'].value_counts()[0]
        assert 0.3 < class_ratio < 3.0, f"Severe class imbalance: {class_ratio}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
