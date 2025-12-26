#!/usr/bin/env python3
"""
Data Download Script for Heart Disease UCI Dataset
Processes the dataset from UCI ML Repository
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

def process_dataset():
    """Process Heart Disease dataset from UCI repository files."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Use processed.cleveland.data from the UCI dataset
    raw_path = os.path.join(DATA_DIR, 'processed.cleveland.data')
    clean_path = os.path.join(DATA_DIR, 'heart_disease.csv')
    
    print("Processing Heart Disease UCI Dataset...")
    
    # Load the data
    df = pd.read_csv(raw_path, names=COLUMN_NAMES, na_values='?')
    
    # Handle missing values
    df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
    df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
    
    print(f"Original shape: {df.shape}")
    print(f"Missing values before cleaning:\n{df.isnull().sum()}")
    
    df = df.dropna()
    
    # Convert target to binary (0 = no disease, 1 = disease present)
    # Original target: 0 = no disease, 1-4 = various stages of disease
    df['target'] = (df['target'] > 0).astype(int)
    
    # Convert float columns to appropriate types
    int_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'slope', 'ca', 'thal', 'target']
    for col in int_cols:
        df[col] = df[col].astype(int)
    
    # Save cleaned dataset
    df.to_csv(clean_path, index=False)
    print(f"\nDataset saved to {clean_path}")
    print(f"Final shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df['target'].value_counts())
    print(f"\nClass balance ratio: {df['target'].value_counts()[1]/df['target'].value_counts()[0]:.2f}")
    
    return df

if __name__ == "__main__":
    process_dataset()
