.PHONY: venv install start-mlflow reset-mlflow run-app simple-tuning custom-tuning save-best-model lint format clean clean-mlflow help docker-build docker-run test docker-compose

# Variables
PYTHON = python3
PIP = $(PYTHON) -m pip
FLASK = $(PYTHON) -m flask
MLFLOW = $(PYTHON) -m mlflow

# Default target
all: help

# Create virtual environment
venv:
	$(PYTHON) -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate (Linux/Mac) or .\\venv\\Scripts\\activate (Windows)"

# Install dependencies
install:
	$(PIP) install -r requirements.txt

# Start MLflow server
start-mlflow:
	$(PYTHON) mlflow_scripts/run_mlflow_server.py

# Reset MLflow data and restart server
reset-mlflow:
	rm -rf mlflow_data
	mkdir -p mlflow_data/artifacts
	$(PYTHON) mlflow_scripts/run_mlflow_server.py

# Run Flask application
run-app:
	$(PYTHON) app.py

# Run simple hyperparameter tuning with default settings
simple-tuning:
	$(PYTHON) tuning_scripts/simple_hyperparam_tuning.py

# Run simple tuning with tiny parameter space
simple-tuning-tiny:
	$(PYTHON) tuning_scripts/simple_hyperparam_tuning.py --space tiny

# Run simple tuning with GRU model
simple-tuning-gru:
	$(PYTHON) tuning_scripts/simple_hyperparam_tuning.py --model gru

# Run simple tuning with larger dataset
simple-tuning-large:
	$(PYTHON) tuning_scripts/simple_hyperparam_tuning.py --samples 2000

# Run custom hyperparameter tuning with example parameters
custom-tuning:
	@echo '{"units": [32, 64, 128], "dropout": [0.1, 0.2, 0.3], "lookback": [30, 60]}' > custom_params.json
	$(PYTHON) tuning_scripts/custom_hyperparam_tuning.py --params-file custom_params.json
	rm custom_params.json

# Save best model from MLflow
save-best-model:
	$(PYTHON) tuning_scripts/save_best_model.py

# Test forecast with best model
test-forecast:
	@echo "Generating sample data and running forecast with best model..."
	curl -X POST -F "forecast_horizon=14" -F "use_sample_data=on" http://localhost:5001/forecast

# Run linting
lint:
	pylint *.py tuning_scripts/*.py mlflow_scripts/*.py

# Format code
format:
	black *.py tuning_scripts/*.py mlflow_scripts/*.py

# Clean up cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

# Clean MLflow data
clean-mlflow:
	rm -rf mlflow_data
	rm -rf tuning_results

# Docker build
docker-build:
	docker build -t time-series-forecasting-platform .

# Docker run
docker-run:
	@if [ -z "$(DOCKER_USERNAME)" ]; then \
		echo "Error: DOCKER_USERNAME environment variable is not set"; \
		echo "Usage: make docker-run DOCKER_USERNAME=your_dockerhub_username"; \
		exit 1; \
	fi
	docker run -p 5001:5001 -p 5002:5002 $(DOCKER_USERNAME)/ddm501-mse:latest

# Docker compose
docker-compose:
	docker-compose up --build
	docker-compose down

# Help message
help:
	@echo "Available targets:"
	@echo "  venv              : Create virtual environment"
	@echo "  install           : Install dependencies"
	@echo "  start-mlflow      : Start MLflow server"
	@echo "  reset-mlflow      : Reset MLflow data and restart server"
	@echo "  run-app           : Run Flask application"
	@echo "  simple-tuning     : Run simple hyperparameter tuning with default settings"
	@echo "  simple-tuning-tiny: Run simple tuning with tiny parameter space"
	@echo "  simple-tuning-gru : Run simple tuning with GRU model"
	@echo "  simple-tuning-large: Run simple tuning with larger dataset"
	@echo "  custom-tuning     : Run custom hyperparameter tuning with example parameters"
	@echo "  save-best-model   : Save best model from MLflow"
	@echo "  test-forecast     : Test forecast with best model"
	@echo "  lint              : Run linting"
	@echo "  format            : Format code"
	@echo "  clean             : Clean up cache files"
	@echo "  clean-mlflow      : Clean MLflow data"
	@echo "  docker-build      : Build Docker image"
	@echo "  docker-run        : Run Docker container"
	@echo "  docker-compose    : Build and start Docker containers" 