#!/usr/bin/env python3
"""
Heart Disease Prediction API
Task 6: Model Containerization - FastAPI Application
"""

import os
import logging
from datetime import datetime
from typing import Optional
import joblib
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTIONS_TOTAL = Counter('predictions_total', 'Total predictions made', ['result'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Assignment - Heart Disease Risk Prediction Model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

@app.get("/ui", tags=["UI"])
async def ui():
    """Serve the frontend UI."""
    from fastapi.responses import FileResponse
    return FileResponse('src/static/index.html')

# Load model and scaler
MODEL_PATH = os.environ.get('MODEL_PATH', '/app/models/pipeline.pkl')
SCALER_PATH = os.environ.get('SCALER_PATH', '/app/models/scaler.pkl')
FEATURES_PATH = os.environ.get('FEATURES_PATH', '/app/models/feature_names.pkl')

# Try alternate paths for local development
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'pipeline.pkl')
if not os.path.exists(SCALER_PATH):
    SCALER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'scaler.pkl')
if not os.path.exists(FEATURES_PATH):
    FEATURES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'feature_names.pkl')

try:
    pipeline = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    logger.info("Model and preprocessor loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    pipeline = None
    feature_names = None


class PatientData(BaseModel):
    """Input schema for patient health data."""
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=Female, 1=Male)")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: int = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy")
    thal: int = Field(..., ge=3, le=7, description="Thalassemia (3=normal, 6=fixed defect, 7=reversible)")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Output schema for prediction results."""
    prediction: int = Field(..., description="0=No Disease, 1=Disease Present")
    prediction_label: str = Field(..., description="Human-readable prediction")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    risk_level: str = Field(..., description="Risk assessment level")
    timestamp: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    model_loaded: bool
    version: str
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Log startup event."""
    logger.info("Heart Disease Prediction API started")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    REQUEST_COUNT.labels(endpoint="/", method="GET").inc()
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    REQUEST_COUNT.labels(endpoint="/health", method="GET").inc()
    return HealthResponse(
        status="healthy" if pipeline is not None else "unhealthy",
        model_loaded=pipeline is not None,
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientData):
    """
    Predict heart disease risk based on patient health data.
    
    Returns prediction (0/1), confidence score, and risk level.
    """
    REQUEST_COUNT.labels(endpoint="/predict", method="POST").inc()
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_data = np.array([[
            patient.age, patient.sex, patient.cp, patient.trestbps,
            patient.chol, patient.fbs, patient.restecg, patient.thalach,
            patient.exang, patient.oldpeak, patient.slope, patient.ca, patient.thal
        ]])
        
        # Make prediction with latency tracking
        with PREDICTION_LATENCY.time():
            prediction = int(pipeline.predict(input_data)[0])
            probabilities = pipeline.predict_proba(input_data)[0]
            confidence = float(max(probabilities))
        
        # Determine risk level
        disease_prob = probabilities[1]
        if disease_prob < 0.3:
            risk_level = "Low"
        elif disease_prob < 0.6:
            risk_level = "Moderate"
        elif disease_prob < 0.8:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        # Update metrics
        result_label = "disease" if prediction == 1 else "no_disease"
        PREDICTIONS_TOTAL.labels(result=result_label).inc()
        
        # Log prediction
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}, Risk: {risk_level}")
        
        return PredictionResponse(
            prediction=prediction,
            prediction_label="Disease Present" if prediction == 1 else "No Disease",
            confidence=confidence,
            risk_level=risk_level,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    REQUEST_COUNT.labels(endpoint="/metrics", method="GET").inc()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model information."""
    REQUEST_COUNT.labels(endpoint="/model/info", method="GET").inc()
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model = pipeline.named_steps['model']
    model_type = type(model).__name__
    
    return {
        "model_type": model_type,
        "features": feature_names,
        "n_features": len(feature_names),
        "model_path": MODEL_PATH
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
