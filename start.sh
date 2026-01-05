#!/bin/bash

# Function to check if Docker Daemon is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "Error: Docker Daemon is not running."
        echo "Please start Docker Desktop and try again."
        exit 1
    fi
}

echo "Step 1: Checking Docker status..."
check_docker

echo "Step 2: Building and Starting Services..."
docker-compose up -d --build

echo "Step 3: Waiting for services to initialize..."
echo "Waiting 15 seconds..."
sleep 15

echo "Step 4: Services Status:"
docker-compose ps

echo "Step 5: Generating Traffic..."
# Kill any existing traffic generator
pkill -f generate_traffic.py || true

# Check for python with requests
if python -c "import requests" &> /dev/null; then
    PYTHON_CMD="python"
elif python3 -c "import requests" &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Installing requests library..."
    # Try to install in the python environment that is available
    if command -v python &> /dev/null; then
         python -m pip install requests || true
         PYTHON_CMD="python"
    elif command -v python3 &> /dev/null; then
         python3 -m pip install requests || true
         PYTHON_CMD="python3"
    fi
fi

if [ -n "$PYTHON_CMD" ]; then
    echo "Starting traffic generator using $PYTHON_CMD..."
    $PYTHON_CMD generate_traffic.py > traffic.log 2>&1 &
    PID=$!
    echo "Traffic generator started with PID $PID"
else
    echo "Error: Could not find Python with 'requests' library. Please install it manually."
fi

echo "=================================================="
echo "Access your services at:"
echo "Application: http://localhost:8000"
echo "Grafana:     http://localhost:30030 (login: admin/admin123)"
echo "             -> Go to Dashboards > Heart Disease AI > Heart Disease Model Metrics"
echo "Prometheus:  http://localhost:30090"
echo "=================================================="
