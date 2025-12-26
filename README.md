# Heart Disease Prediction - MLOps Project

## 📋 Overview

This project implements an end-to-end MLOps pipeline for heart disease prediction using the UCI Heart Disease dataset. The solution includes data exploration, model development, experiment tracking, CI/CD pipelines, containerization, and Kubernetes deployment.

**Course:** MLOps (S1-25_AIMLCZG523)  
**Assignment:** Assignment - I (50 Marks)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline Architecture                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │   Data   │───▶│   EDA    │───▶│  Model   │───▶│ Model Packaging  │   │
│  │ Download │    │ Analysis │    │ Training │    │ (MLflow/Pickle)  │   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────────┘   │
│                                        │                                 │
│                                        ▼                                 │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    GitHub Actions CI/CD Pipeline                  │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │   │
│  │  │  Lint   │─▶│  Test   │─▶│  Train  │─▶│  Build  │─▶│ Deploy  │ │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                        │                                 │
│                                        ▼                                 │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Production Deployment                          │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐   │   │
│  │  │   Docker    │───▶│ Kubernetes  │───▶│  Prometheus/Grafana │   │   │
│  │  │  Container  │    │   Cluster   │    │     Monitoring      │   │   │
│  │  └─────────────┘    └─────────────┘    └─────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
mlops-heart-disease/
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # GitHub Actions CI/CD pipeline
├── data/
│   └── heart_disease.csv       # Cleaned dataset
├── k8s/
│   ├── deployment.yaml         # Kubernetes deployment manifest
│   ├── ingress.yaml           # Ingress configuration
│   └── monitoring.yaml        # Prometheus/Grafana setup
├── models/
│   ├── best_model.pkl         # Trained model
│   ├── scaler.pkl             # Feature scaler
│   ├── feature_names.pkl      # Feature names
│   └── pipeline.pkl           # Complete prediction pipeline
├── notebooks/                  # Jupyter notebooks (EDA, training)
├── screenshots/               # EDA visualizations and results
├── src/
│   ├── app.py                 # FastAPI application
│   ├── download_data.py       # Data download script
│   ├── eda.py                 # Exploratory Data Analysis
│   └── train.py               # Model training with MLflow
├── tests/
│   ├── test_api.py            # API tests
│   ├── test_data.py           # Data tests
│   └── test_model.py          # Model tests
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
├── pytest.ini                 # Pytest configuration
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker
- Kubernetes (Minikube/Docker Desktop) for deployment
- Git

### Local Setup

```bash
# Clone repository
git clone <repository-url>
cd mlops-heart-disease

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run EDA
python src/eda.py

# Train models
python src/train.py

# Run tests
pytest tests/ -v

# Start API server
python src/app.py
```

### Docker Setup

```bash
# Build Docker image
docker build -t heart-disease-api:latest .

# Run container
docker run -d -p 8000:8000 --name hd-api heart-disease-api:latest

# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 1, "ca": 0, "thal": 6
  }'
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/monitoring.yaml

# Check deployment status
kubectl get pods
kubectl get services

# Port forward for local access
kubectl port-forward service/heart-disease-api-service 8000:80
```

## 📊 Dataset

**Source:** UCI Machine Learning Repository - Heart Disease Dataset  
**Features:** 13 clinical attributes  
**Target:** Binary classification (0 = No Disease, 1 = Disease)

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Sex (0 = Female, 1 = Male) |
| cp | Chest pain type (1-4) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results (0-2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia type |

## 🤖 Models

Three classification models were trained and compared:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 91.80% | 92.59% | 89.29% | 90.91% | **99.03%** |
| Random Forest | 95.08% | 93.10% | 96.43% | 94.74% | 97.73% |
| Gradient Boosting | 85.25% | 85.19% | 82.14% | 83.64% | 92.75% |

**Best Model:** Logistic Regression (highest ROC-AUC)

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint |
| `/health` | GET | Health check |
| `/predict` | POST | Make prediction |
| `/metrics` | GET | Prometheus metrics |
| `/model/info` | GET | Model information |
| `/docs` | GET | Swagger documentation |

### Example API Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 160,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 1,
    "thal": 3
  }'
```

### Example Response

```json
{
  "prediction": 1,
  "prediction_label": "Disease Present",
  "confidence": 0.78,
  "risk_level": "High",
  "timestamp": "2024-12-25T12:00:00.000000"
}
```

## 📈 MLflow Experiment Tracking

MLflow tracks all experiments including:
- Model parameters
- Performance metrics
- Artifacts (plots, models)
- Run comparisons

Access MLflow UI:
```bash
cd mlops-heart-disease
mlflow ui --backend-store-uri file:./mlruns
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data.py -v
pytest tests/test_model.py -v
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔄 CI/CD Pipeline

The GitHub Actions pipeline includes:

1. **Lint** - Code quality checks with flake8 and black
2. **Test** - Unit tests for data, model, and API
3. **Train** - Model training with MLflow tracking
4. **Docker** - Build and test container
5. **Security** - Vulnerability scanning with Trivy
6. **Deploy** - Production deployment (manual trigger)

## 📊 Monitoring

### Prometheus Metrics

- `predictions_total` - Total predictions by result
- `prediction_latency_seconds` - Prediction latency histogram
- `api_requests_total` - Total API requests by endpoint

### Grafana Dashboard

Access Grafana at `http://localhost:30030` (in Kubernetes)
- Default credentials: admin/admin123

## 📝 License

This project is for educational purposes as part of the MLOps course assignment.

## 👤 Author - MAHESHWARI ADITYA LALCHAND (2024AA05822)

**Assignment for BITS Pilani - MLOps Course (S1-25_AIMLCZG523)**
