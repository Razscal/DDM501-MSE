# Time Series Forecasting Platform

A complete MLOps platform for time series forecasting with LSTM neural networks, hyperparameter tuning, and model tracking.

## Project Structure

```
├── app.py                        # Flask API with web UI
├── mlib.py                       # Core ML library for time series forecasting
├── tuning_scripts/               # Hyperparameter tuning scripts
│   ├── simple_hyperparam_tuning.py   # Simple hyperparameter tuning
│   ├── custom_hyperparam_tuning.py   # Custom hyperparameter tuning
│   └── save_best_model.py            # Save best model from MLflow
├── mlflow_scripts/               # MLflow server and utilities
│   ├── run_mlflow_server.py          # Start MLflow server
├── templates/                    # HTML templates for Flask UI
│   ├── index.html                    # Main application page
│   └── result_detail.html            # Tuning result details page
├── models/                       # Directory for saved models
├── static/                       # Static files (images, CSS, JS)
├── datasets/                     # Time series datasets
├── mlflow_data/                  # MLflow data and artifacts
├── tuning_results/               # Saved tuning results
├── requirements.txt              # Project dependencies
├── Makefile                      # Utility commands
└── README.md                     # Project documentation
```

## System Requirements

* Python 3.9+
* Libraries: flask, tensorflow, scikit-learn, pandas, numpy, joblib, mlflow, matplotlib

## Installation

1. Create a virtual environment:

```bash
make venv
```

2. Activate the virtual environment:

```bash
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:

```bash
make install
```

## Running the Application

1. Start the MLflow server:

```bash
make start-mlflow
```

2. Run the Flask application:

```bash
make run-app
```

3. Open your browser and navigate to: http://localhost:5001

### Using Docker

```bash
make docker-build
make docker-run
```

## Key Features

### Time Series Model

- **LSTM and GRU Models**: Deep learning models for time series forecasting
- **Flexible Preprocessing**: Configurable lookback period and forecast horizon
- **Multi-Step Forecasting**: Generate forecasts for multiple future time steps

### Hyperparameter Tuning

- **Simple Tuning**: Pre-defined parameter spaces of different sizes
- **Custom Tuning**: Define your own hyperparameter grid
- **Tracked Experiments**: All tuning runs are tracked in MLflow

### Web Interface

1. **Model Tuning Tab**:  
   - Select model type (LSTM, GRU)  
   - Choose parameter space (tiny, small, medium)  
   - Set data generation parameters
   - Run custom hyperparameter tuning

2. **Forecasting Tab**:  
   - Generate forecasts using the best trained model
   - Visualize forecast results
   - Download forecast data

3. **Tuning Results Tab**:  
   - View all previous tuning runs
   - Examine detailed metrics and parameters
   - Compare results between runs

4. **Data Management Tab**:  
   - Upload time series datasets
   - Manage available datasets

### MLflow Integration

MLflow is tightly integrated into the project for:

- Experiment tracking
- Model registry
- Artifact storage
- Result comparison

MLflow UI is available at: http://localhost:5002

## Command-Line Interface

The Makefile provides various commands for common operations:

```bash
# Start MLflow server
make start-mlflow

# Run simple hyperparameter tuning
make simple-tuning

# Run simple tuning with tiny parameter space (faster)
make simple-tuning-tiny

# Run tuning with GRU model
make simple-tuning-gru

# Run custom hyperparameter tuning
make custom-tuning

# Save the best model from MLflow
make save-best-model

# Test forecast generation
make test-forecast

# Reset MLflow data
make reset-mlflow

# Clean up caches and temporary files
make clean
```

## API Endpoints

### Health Check
- URL: `/health`
- Method: GET
- Response: Application status and model availability

### Train Model
- URL: `/train`
- Method: POST
- Body: JSON with time series data
- Response: Training metrics

### Generate Forecast
- URL: `/forecast`
- Method: POST
- Form data: Forecast parameters and data source
- Response: Redirects to UI with forecast results

### Get Metrics
- URL: `/metrics`
- Method: GET
- Response: Current model metrics

## Time Series Data Format

For custom datasets, the expected CSV format is:

```
date,value
2020-01-01,10.5
2020-01-02,11.2
...
```

The date column must be in a format that can be parsed by pandas.to_datetime().

## License

MIT 