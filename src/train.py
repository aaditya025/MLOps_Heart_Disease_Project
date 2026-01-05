#!/usr/bin/env python3
"""
Heart Disease Prediction - Model Training with MLflow Tracking
Tasks 2 & 3: Feature Engineering, Model Development & Experiment Tracking
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report,
                            confusion_matrix, roc_curve)
import mlflow
import mlflow.sklearn
import joblib
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'heart_disease.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'screenshots')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

class HeartDiseaseModel:
    """Heart Disease Prediction Model Pipeline."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                             'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self):
        """Load and prepare dataset."""
        df = pd.read_csv(DATA_PATH)
        X = df[self.feature_names]
        y = df['target']
        return X, y
    
    def preprocess(self, X_train, X_test):
        """Apply feature scaling."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def get_models(self):
        """Define models with hyperparameter grids."""
        models = {
            'Logistic Regression': {
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }
        }
        return models
    
    def evaluate_model(self, model, X_test, y_test):
        """Calculate evaluation metrics."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        return metrics, y_pred, y_proba
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)
        
        plt.tight_layout()
        save_path = os.path.join(PLOTS_DIR, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_roc_curves(self, results, X_test, y_test):
        """Plot ROC curves for all models."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        for (name, model), color in zip(results.items(), colors):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, color=color, linewidth=2, 
                   label=f'{name} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(PLOTS_DIR, 'roc_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_feature_importance(self, model, model_name):
        """Plot feature importance for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(self.feature_names)))
            
            bars = ax.barh(range(len(indices)), importances[indices], 
                          color=colors[indices], edgecolor='black')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([self.feature_names[i] for i in indices])
            ax.set_xlabel('Feature Importance', fontsize=12)
            ax.set_title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            
            plt.tight_layout()
            save_path = os.path.join(PLOTS_DIR, f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return save_path
        return None
    
    def train_with_mlflow(self):
        """Train models with MLflow experiment tracking."""
        # Set up MLflow
        mlflow.set_tracking_uri(f"file://{os.path.join(BASE_DIR, 'mlruns')}")
        mlflow.set_experiment("heart_disease_prediction")
        
        # Load and split data
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Preprocess
        X_train_scaled, X_test_scaled = self.preprocess(X_train, X_test)
        
        models = self.get_models()
        results = {}
        best_auc = 0
        
        print("=" * 60)
        print("MODEL TRAINING WITH MLFLOW TRACKING")
        print("=" * 60)
        
        for model_name, model_config in models.items():
            print(f"\n{'='*40}")
            print(f"Training: {model_name}")
            print("=" * 40)
            
            with mlflow.start_run(run_name=model_name):
                # Hyperparameter tuning with GridSearchCV
                grid_search = GridSearchCV(
                    model_config['model'],
                    model_config['params'],
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train_scaled, y_train)
                
                best_model = grid_search.best_estimator_
                
                # Cross-validation scores
                cv_scores = cross_val_score(best_model, X_train_scaled, y_train, 
                                           cv=5, scoring='roc_auc')
                
                # Evaluate on test set
                metrics, y_pred, y_proba = self.evaluate_model(best_model, X_test_scaled, y_test)
                
                # Log parameters
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_param("model_type", model_name)
                
                # Log metrics
                mlflow.log_metric("cv_mean_auc", cv_scores.mean())
                mlflow.log_metric("cv_std_auc", cv_scores.std())
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log artifacts (plots)
                cm_path = self.plot_confusion_matrix(y_test, y_pred, model_name)
                mlflow.log_artifact(cm_path)
                
                fi_path = self.plot_feature_importance(best_model, model_name)
                if fi_path:
                    mlflow.log_artifact(fi_path)
                
                # Log model
                mlflow.sklearn.log_model(best_model, f"{model_name.lower().replace(' ', '_')}_model")
                
                # Store results
                results[model_name] = best_model
                
                # Print results
                print(f"\nBest Parameters: {grid_search.best_params_}")
                print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                print(f"\nTest Set Metrics:")
                for metric_name, metric_value in metrics.items():
                    print(f"  {metric_name}: {metric_value:.4f}")
                
                # Track best model
                if metrics['roc_auc'] > best_auc:
                    best_auc = metrics['roc_auc']
                    self.best_model = best_model
                    self.best_model_name = model_name
        
        # Plot ROC curves comparison
        roc_path = self.plot_roc_curves(results, X_test_scaled, y_test)
        
        # Save best model and scaler
        self.save_model()
        
        print("\n" + "=" * 60)
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"ROC-AUC Score: {best_auc:.4f}")
        print("=" * 60)
        
        return results
    
    def save_model(self):
        """Save the best model and preprocessing pipeline."""
        # Save model
        model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
        joblib.dump(self.best_model, model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save feature names
        features_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
        joblib.dump(self.feature_names, features_path)
        print(f"Feature names saved to: {features_path}")
        
        # Create preprocessing pipeline
        pipeline = Pipeline([
            ('scaler', self.scaler),
            ('model', self.best_model)
        ])
        pipeline_path = os.path.join(MODELS_DIR, 'pipeline.pkl')
        joblib.dump(pipeline, pipeline_path)
        print(f"Pipeline saved to: {pipeline_path}")

def main():
    """Main training pipeline."""
    trainer = HeartDiseaseModel()
    results = trainer.train_with_mlflow()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nArtifacts saved:")
    print(f"  - Models: {MODELS_DIR}")
    print(f"  - Plots: {PLOTS_DIR}")
    print(f"  - MLflow: {os.path.join(BASE_DIR, 'mlruns')}")

if __name__ == "__main__":
    main()
