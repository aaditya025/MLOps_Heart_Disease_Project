
import requests
import random
import time
import json

url = "http://localhost:8000/predict"

def generate_patient():
    return {
        "age": random.randint(20, 80),
        "sex": random.randint(0, 1),
        "cp": random.randint(1, 4),
        "trestbps": random.randint(90, 200),
        "chol": random.randint(120, 500),
        "fbs": random.randint(0, 1),
        "restecg": random.randint(0, 2),
        "thalach": random.randint(70, 200),
        "exang": random.randint(0, 1),
        "oldpeak": round(random.uniform(0, 6.0), 1),
        "slope": random.randint(1, 3),
        "ca": random.randint(0, 4),
        "thal": random.choice([3, 6, 7])
    }

print("Generating traffic for Heart Disease API...")
for i in range(1000):
    patient = generate_patient()
    try:
        response = requests.post(url, json=patient)
        if response.status_code == 200:
            result = response.json()
            print(f"Request {i+1}: Prediction={result['prediction_label']}, Risk={result['risk_level']}")
        else:
            print(f"Request {i+1}: Failed with status {response.status_code}")
    except Exception as e:
        print(f"Request {i+1}: Error - {e}")
    time.sleep(0.5)

print("Traffic generation complete.")
