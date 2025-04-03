#!/usr/bin/env python
"""
Time Series Forecasting Platform
This Flask application provides a web interface and API for time series forecasting
using PyTorch-based LSTM/GRU neural networks with hyperparameter tuning and MLflow integration.
"""

import os
import json
import joblib
import logging
import traceback
import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_from_directory
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import shutil

# Import custom modules
from mlib import TimeSeriesWrapper, DataProcessor, perform_tuning, load_tuning_result

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_key_for_flask_session")

# Constants
TUNING_RESULTS_DIR = Path("tuning_results")
MODELS_DIR = Path("models")
DATASETS_DIR = Path("datasets")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5002")

# Create necessary directories
TUNING_RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)

# Routes
@app.route('/')
def index():
    """Render the home page with list of tuning results."""
    # Get list of tuning results
    results = []
    if TUNING_RESULTS_DIR.exists():
        for file in TUNING_RESULTS_DIR.glob("*.json"):
            try:
                result_data = load_tuning_result(file)
                results.append({
                    'id': file.stem,
                    'model_type': result_data.get('model_type', 'Unknown'),
                    'timestamp': result_data.get('timestamp', 'Unknown'),
                    'status': result_data.get('status', 'Unknown'),
                    'best_score': result_data.get('best_score', 'N/A'),
                    'path': file
                })
            except Exception as e:
                logger.error(f"Error loading tuning result {file}: {e}")
    
    # Get list of saved models
    models = []
    if MODELS_DIR.exists():
        for file in MODELS_DIR.glob("*.pt"):
            models.append({
                'name': file.stem,
                'path': file,
                'created': datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
    
    # Get list of datasets
    datasets = []
    if DATASETS_DIR.exists():
        for file in DATASETS_DIR.glob("*.csv"):
            datasets.append({
                'name': file.stem,
                'path': file
            })
    
    return render_template('index.html', 
                          results=sorted(results, key=lambda x: x.get('timestamp', ''), reverse=True),
                          models=models,
                          datasets=datasets,
                          mlflow_url=MLFLOW_TRACKING_URI)

@app.route('/tuning_result/<result_id>')
def tuning_result_detail(result_id):
    """Show details of a specific tuning result."""
    result_file = TUNING_RESULTS_DIR / f"{result_id}.json"
    
    if not result_file.exists():
        flash(f"Tuning result {result_id} not found", "error")
        return redirect(url_for('index'))
    
    try:
        result_data = load_tuning_result(result_file)
        # Format the JSON data for display
        result_json = json.dumps(result_data, indent=2)
        return render_template('result_detail.html', 
                              result=result_data, 
                              result_id=result_id,
                              result_json=result_json,
                              mlflow_url=MLFLOW_TRACKING_URI)
    except Exception as e:
        flash(f"Error loading tuning result: {e}", "error")
        return redirect(url_for('index'))

@app.route('/tune', methods=['GET', 'POST'])
def tune():
    """Page for initiating hyperparameter tuning."""
    if request.method == 'POST':
        try:
            # Get form data
            model_type = request.form.get('model_type', 'lstm')
            dataset = request.form.get('dataset')
            n_samples = int(request.form.get('n_samples', 10))
            epochs = int(request.form.get('epochs', 50))
            sequence_length = int(request.form.get('sequence_length', 10))
            forecast_horizon = int(request.form.get('forecast_horizon', 5))
            use_mlflow = request.form.get('use_mlflow') == 'on'
            
            # Custom parameters (optional)
            custom_params = {}
            if request.form.get('custom_params'):
                try:
                    custom_params = json.loads(request.form.get('custom_params'))
                except json.JSONDecodeError:
                    flash("Invalid JSON format for custom parameters", "error")
                    return redirect(url_for('tune'))
            
            # Check if dataset exists
            dataset_path = DATASETS_DIR / f"{dataset}.csv"
            if not dataset_path.exists():
                flash(f"Dataset {dataset} not found", "error")
                return redirect(url_for('tune'))
            
            # Set up parameters
            params = {
                'model_type': model_type,
                'dataset': str(dataset_path),
                'n_samples': n_samples,
                'epochs': epochs,
                'sequence_length': sequence_length,
                'forecast_horizon': forecast_horizon,
                'use_mlflow': use_mlflow
            }
            if custom_params:
                params['custom_params'] = custom_params
            
            # Generate result ID
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_id = f"{model_type}_tuning_{timestamp}"
            
            # Perform tuning
            result_file = TUNING_RESULTS_DIR / f"{result_id}.json"
            perform_tuning(params, result_file)
            
            flash(f"Tuning job initiated with ID: {result_id}", "success")
            return redirect(url_for('tuning_result_detail', result_id=result_id))
            
        except Exception as e:
            flash(f"Error starting tuning job: {e}", "error")
            return redirect(url_for('tune'))
    
    # GET request - show tuning form
    datasets = []
    if DATASETS_DIR.exists():
        for file in DATASETS_DIR.glob("*.csv"):
            datasets.append({
                'name': file.stem,
                'path': file
            })
    
    return render_template('tune.html', datasets=datasets)

@app.route('/save_best_model/<result_id>')
def save_best_model(result_id):
    """Save the best model from a tuning result."""
    try:
        result_file = os.path.join(TUNING_RESULTS_DIR, f"{result_id}.json")
        
        if not os.path.exists(result_file):
            flash(f"Tuning result {result_id} not found", "error")
            return redirect(url_for('index'))
        
        result_data = load_tuning_result(result_file)
        
        if result_data.get('status') != 'completed':
            flash("Cannot save model: tuning is not completed", "error")
            return redirect(url_for('tuning_result', result_id=result_id))
        
        # Get the path to the best model
        best_model_path = result_data.get('best_model_path')
        if not best_model_path or not os.path.exists(best_model_path):
            flash("Best model file not found", "error")
            return redirect(url_for('tuning_result', result_id=result_id))
        
        # Create a user-friendly model name (just the ID if nothing else)
        model_name = f"{result_id}_best_model"
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
        
        # Only copy if the paths are different to avoid SameFileError
        if os.path.abspath(best_model_path) != os.path.abspath(model_path):
            shutil.copy(best_model_path, model_path)
            logger.info(f"Best model copied to {model_path}")
        
        # Check if we should use this model for forecasting
        use_for_forecast = request.args.get('use_for_forecast', False)
        if use_for_forecast:
            is_direct = request.args.get('direct', False)
            forecast_url = url_for('forecast_direct' if is_direct else 'forecast')
            flash(f"Best model saved as {model_name}", "success")
            return redirect(f"{forecast_url}?model={model_name}")
        
        flash(f"Best model saved as {model_name}", "success")
        return redirect(url_for('index'))
        
    except Exception as e:
        logger.error(f"Error saving best model: {e}")
        logger.error(traceback.format_exc())
        flash(f"Error saving best model: {e}", "error")
        return redirect(url_for('tuning_result', result_id=result_id))

def get_available_models():
    """Helper function to get available models."""
    models = []
    if MODELS_DIR.exists():
        for file in MODELS_DIR.glob("*.pt"):
            try:
                # Try to load minimal model metadata without loading the full model
                model_info = joblib.load(file, mmap_mode='r')
                model_type = model_info.get('model_type', 'unknown').upper()
                created_at = model_info.get('created_at', '')
                if created_at:
                    try:
                        created_at = datetime.fromisoformat(created_at).strftime('%Y-%m-%d %H:%M')
                    except:
                        pass
            except:
                model_type = "UNKNOWN"
                created_at = ""
            
            models.append({
                'name': file.stem,
                'path': str(file),
                'type': model_type,
                'created': created_at
            })
    return models

def get_available_datasets():
    """Helper function to get available datasets."""
    datasets = []
    if DATASETS_DIR.exists():
        for file in DATASETS_DIR.glob("*.csv"):
            try:
                # Get row count
                processor = DataProcessor()
                row_count = len(processor.load_data(file))
            except:
                row_count = 0
            
            datasets.append({
                'name': file.stem,
                'path': str(file),  # Full path including datasets directory
                'rows': row_count
            })
    return datasets

def generate_forecast(model_path, dataset_path, horizon=5):
    """
    Generate forecast using a PyTorch model and dataset.
    
    Args:
        model_path: Path to the model file
        dataset_path: Path to the dataset file
        horizon: Number of time steps to forecast
        
    Returns:
        dict: Forecast data including dates, values, and original data
        
    Raises:
        ValueError: If the model or dataset doesn't exist or other forecast errors
    """
    import traceback
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.preprocessing import MinMaxScaler
    
    logger.info(f"Generating forecast with model={model_path}, dataset={dataset_path}, horizon={horizon}")
    
    # Load model data
    if not os.path.exists(model_path):
        raise ValueError(f"Model file {model_path} not found")
    
    if not os.path.exists(dataset_path):
        # Try to see if we need to add .csv extension
        csv_path = f"{dataset_path}.csv"
        if os.path.exists(csv_path):
            dataset_path = csv_path
            logger.info(f"Using dataset with added .csv extension: {dataset_path}")
        else:
            raise ValueError(f"Dataset file {dataset_path} not found")
    
    try:
        # Try loading with joblib from an existing tuning result
        model_id = os.path.basename(model_path).replace('.pt', '')
        tuning_result_path = os.path.join(TUNING_RESULTS_DIR, f"{model_id}.json")
        
        if os.path.exists(tuning_result_path):
            logger.info(f"Found tuning result for model {model_id}, attempting to rebuild model")
            result_data = load_tuning_result(tuning_result_path)
            
            # Check if tuning completed
            if result_data.get('status') == 'completed' and result_data.get('best_params'):
                # Recreate model with best parameters
                best_params = result_data.get('best_params')
                model_type = result_data.get('model_type', 'lstm')
                sequence_length = result_data.get('params', {}).get('sequence_length', 10)
                forecast_horizon = result_data.get('params', {}).get('forecast_horizon', 5)
                hidden_size = best_params.get('hidden_size', 50)
                num_layers = best_params.get('num_layers', 1)
                dropout = best_params.get('dropout', 0.2)
                learning_rate = best_params.get('learning_rate', 0.001)
                batch_size = best_params.get('batch_size', 32)
                epochs = result_data.get('params', {}).get('epochs', 50)
                
                # Get dataset path from tuning result
                dataset_path_for_training = result_data.get('params', {}).get('dataset', dataset_path)
                if not os.path.exists(dataset_path_for_training):
                    logger.warning(f"Original dataset not found: {dataset_path_for_training}, using current dataset")
                    dataset_path_for_training = dataset_path
                
                # Create model
                model = TimeSeriesWrapper(
                    model_type=model_type,
                    sequence_length=sequence_length,
                    forecast_horizon=forecast_horizon,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    learning_rate=learning_rate
                )
                
                # Train the model on dataset
                logger.info(f"Training new model with best parameters from {model_id}")
                data_processor = DataProcessor()
                X_train, y_train, X_test, y_test, scaler = data_processor.prepare_data(
                    dataset_path=dataset_path_for_training,
                    sequence_length=sequence_length,
                    forecast_horizon=forecast_horizon
                )
                
                # Train the model
                model.train(
                    X_train=X_train,
                    y_train=y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    patience=10
                )
                
                # Save the model with the new format
                model.save(model_path)
                logger.info(f"Model rebuilt and saved to {model_path}")
                
                # Try loading again
                model_data = joblib.load(model_path)
                logger.info(f"Successfully reloaded rebuilt model")
                return model_data
    except Exception as e:
        logger.warning(f"Could not rebuild model from tuning results: {e}")
    
    # Normal loading process
    try:
        # First try using joblib with extra error handling
        try:
            model_data = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path} using joblib")
        except (pickle.UnpicklingError, UnicodeDecodeError) as e:
            logger.warning(f"Error with joblib loading: {e}, trying PyTorch loading")
            # Try torch loading as fallback with security settings for PyTorch 2.6+
            try:
                # Add numpy scalar type to safe globals
                import torch.serialization
                import numpy as np
                torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
                # Try with safe globals first
                model_data = torch.load(model_path, map_location=torch.device('cpu'))
                logger.info(f"Model loaded from {model_path} using PyTorch with safe globals")
            except Exception as torch_e:
                logger.warning(f"Error with safe globals: {torch_e}, trying with weights_only=False")
                # Fall back to weights_only=False (less secure but more compatible)
                model_data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
                logger.info(f"Model loaded from {model_path} using PyTorch with weights_only=False")
            
            # Convert to expected format
            model_data = {
                'sequence_length': model_data.get('sequence_length', 10),
                'forecast_horizon': model_data.get('forecast_horizon', 5),
                'model_state_dict': model_data.get('model_state_dict')
            }
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Error loading model: {str(e)}")
    
    # Extract model details
    sequence_length = model_data.get('sequence_length', model_data.get('seq_length', 10))
    scaler = model_data.get('scaler')
    
    # If we're dealing with a TensorFlow model in the file, create a PyTorch model instead
    model = model_data.get('pytorch_model')
    if model is None:
        logger.info("No PyTorch model found in file, creating a new one")
        model = create_pytorch_model(model_data)
    
    # Load and prepare dataset
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset loaded: {dataset_path}, shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise ValueError(f"Error loading dataset: {str(e)}")
    
    # Extract date and value columns (assume first column is date, second is value)
    date_col = df.columns[0]
    value_col = df.columns[1]
    
    # Convert dates if needed
    if pd.api.types.is_string_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort by date
    df = df.sort_values(by=date_col)
    values = df[value_col].values.reshape(-1, 1)
    
    # Scale the data
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        values_scaled = scaler.fit_transform(values)
    else:
        values_scaled = scaler.transform(values)
    
    logger.info(f"Data scaled, shape: {values_scaled.shape}")
    
    # Create input sequence
    if values_scaled.shape[0] < sequence_length:
        raise ValueError(f"Not enough data points. Need at least {sequence_length} points.")
    
    # Setup evaluation
    input_sequence = values_scaled[-sequence_length:].reshape(1, sequence_length, 1)
    
    # Check if model is TimeSeriesWrapper or direct PyTorch model
    if hasattr(model, 'predict'):
        # Using TimeSeriesWrapper
        predictions = model.predict(input_sequence)
    else:
        # Using direct PyTorch model
        model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(input_sequence, dtype=torch.float32)
            predictions = model(input_tensor).detach().numpy()
    
    # Make sure predictions have the right shape for inverse transform
    if len(predictions.shape) == 3:
        # If predictions have shape (batch, seq_len, features)
        forecast_scaled = predictions[0, :, 0].reshape(-1, 1)
    elif len(predictions.shape) == 2:
        # If predictions have shape (batch, features)
        forecast_scaled = predictions.reshape(-1, 1)
    else:
        # For any other shape, try to flatten
        forecast_scaled = predictions.reshape(-1, 1)

    # Limit to requested horizon
    forecast_scaled = forecast_scaled[:horizon]
    
    # Inverse transform the forecast
    forecast = scaler.inverse_transform(forecast_scaled)
    
    # Generate forecast dates
    last_date = df[date_col].iloc[-1]
    forecast_dates = []
    
    # Check if last_date is a datetime
    if isinstance(last_date, pd.Timestamp) or isinstance(last_date, datetime):
        for i in range(1, horizon + 1):
            forecast_dates.append(last_date + timedelta(days=i))
    else:
        # For non-datetime indices, use numeric indices
        for i in range(1, horizon + 1):
            forecast_dates.append(i)
    
    # Create forecast response
    forecast_series = []
    for i, date in enumerate(forecast_dates):
        if i < len(forecast):  # Make sure we don't go out of bounds
            if isinstance(date, datetime) or isinstance(date, pd.Timestamp):
                date_str = date.strftime('%Y-%m-%d')
            else:
                date_str = str(date)
            
            forecast_series.append({
                'date': date_str,
                'value': float(forecast[i][0])
            })
    
    # Get historical data
    historical = []
    for i, row in df.iterrows():
        if isinstance(row[date_col], (datetime, pd.Timestamp)):
            date_str = row[date_col].strftime('%Y-%m-%d')
        else:
            date_str = str(row[date_col])
        
        historical.append({
            'date': date_str,
            'value': float(row[value_col])
        })
    
    # Create response
    response = {
        'forecast': forecast_series,
        'historical': historical,
        'model_info': {
            'name': os.path.basename(model_path),
            'type': model_data.get('model_type', 'unknown').upper()
        }
    }
    
    return response

def create_pytorch_model(model_data):
    """
    Create a PyTorch model from saved or tuning parameters.
    
    Args:
        model_data: Dictionary with model parameters
        
    Returns:
        Initialized TimeSeriesWrapper model
    """
    # Extract model parameters
    model_type = model_data.get('model_type', 'lstm')
    sequence_length = model_data.get('sequence_length', 10)
    forecast_horizon = model_data.get('forecast_horizon', 5)
    hidden_size = model_data.get('hidden_size', 50)
    num_layers = model_data.get('num_layers', 1)
    dropout = model_data.get('dropout', 0.2)
    learning_rate = model_data.get('learning_rate', 0.001)
    
    # Create model
    model = TimeSeriesWrapper(
        model_type=model_type,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate
    )
    
    return model

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    """Page for generating forecasts from saved models."""
    if request.method == 'POST':
        try:
            # Get form data
            model_name = request.form.get('model')
            dataset_name = request.form.get('dataset')
            horizon = int(request.form.get('horizon', 5))
            
            # Check if model and dataset exist
            model_path = MODELS_DIR / f"{model_name}.pt"
            dataset_path = DATASETS_DIR / f"{dataset_name}.csv"
            
            if not model_path.exists():
                flash(f"Model {model_name} not found", "error")
                return redirect(url_for('forecast'))
            
            if not dataset_path.exists():
                flash(f"Dataset {dataset_name} not found", "error")
                return redirect(url_for('forecast'))
            
            # Load model
            model = TimeSeriesWrapper()
            model.load(str(model_path))
            
            # Load and prepare data
            data_processor = DataProcessor()
            df = data_processor.load_data(dataset_path)
            
            # Scale data
            scaler = None
            scaler_path = MODELS_DIR / f"{model_name}_scaler.joblib"
            if scaler_path.exists():
                try:
                    scaler = joblib.load(scaler_path)
                except Exception as e:
                    logger.error(f"Error loading scaler: {e}")
                    scaler = None
            
            if scaler is None:
                # Create a new scaler
                scaler = MinMaxScaler()
                scaler.fit(df.values)
                # Save the scaler for future use
                joblib.dump(scaler, scaler_path)
            
            scaled_data = scaler.transform(df.values)
            
            # Create input sequence for prediction
            sequence_length = model.sequence_length
            input_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
            
            # Make prediction
            with torch.no_grad():
                forecast = model.predict(input_sequence)
            
            # Inverse transform the forecast
            forecast = forecast.reshape(-1, 1)
            forecast = scaler.inverse_transform(forecast)
            
            # Generate forecast dates
            last_date = df.index[-1]
            forecast_dates = []
            for i in range(1, len(forecast) + 1):
                forecast_dates.append(last_date + timedelta(days=i))
            
            # Create forecast data for plotting
            forecast_data = []
            for i, date in enumerate(forecast_dates):
                forecast_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'value': float(forecast[i][0])
                })
            
            # Create historical data for plotting
            historical_data = []
            for i in range(min(50, len(df))):
                historical_data.append({
                    'date': df.index[-(i+1)].strftime('%Y-%m-%d'),
                    'value': float(df.values[-(i+1)][0])
                })
            historical_data.reverse()
            
            return render_template('forecast_result.html',
                                 model_name=model_name,
                                 dataset_name=dataset_name,
                                 historical_data=historical_data,
                                 forecast_data=forecast_data,
                                 json_historical=json.dumps(historical_data),
                                 json_forecast=json.dumps(forecast_data))
        
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            logger.error(traceback.format_exc())
            flash(f"Error generating forecast: {e}", "error")
            return redirect(url_for('forecast'))
    
    # GET request - show forecast form
    models = []
    if MODELS_DIR.exists():
        for file in MODELS_DIR.glob("*.pt"):
            models.append({
                'name': file.stem,
                'path': file
            })
    
    datasets = []
    if DATASETS_DIR.exists():
        for file in DATASETS_DIR.glob("*.csv"):
            datasets.append({
                'name': file.stem,
                'path': file
            })
    
    return render_template('forecast.html', models=models, datasets=datasets)

@app.route('/forecast_direct', methods=['GET', 'POST'])
def forecast_direct():
    """Page for direct forecasting (bypassing MLflow)."""
    models = get_available_models()
    datasets = get_available_datasets()
    forecast_data = None
    error_message = None
    
    if request.method == 'POST':
        try:
            model_name = request.form.get('model')
            dataset_name = request.form.get('dataset')
            horizon = int(request.form.get('horizon', 5))
            
            # Get full paths - handle dataset paths correctly
            model_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
            
            # Fix dataset path handling - avoid duplicate 'datasets/' prefix
            if os.path.dirname(dataset_name) == str(DATASETS_DIR):
                # Dataset name already contains the full path
                dataset_path = dataset_name
            else:
                # Just the filename was provided
                dataset_path = os.path.join(DATASETS_DIR, dataset_name)
            
            logger.info(f"Using dataset path: {dataset_path}")
            
            # Generate forecast
            forecast_data = generate_forecast(model_path, dataset_path, horizon)
            
            # Make sure we handle both old and new format for the template
            if isinstance(forecast_data, dict):
                if 'forecast' not in forecast_data and 'dates' in forecast_data and 'values' in forecast_data:
                    # Convert old format to new format
                    forecast_data = {
                        'forecast': [
                            {'date': date, 'value': value}
                            for date, value in zip(forecast_data.get('dates', []), forecast_data.get('values', []))
                        ],
                        'historical': [
                            {'date': date, 'value': value}
                            for date, value in zip(
                                forecast_data.get('original_dates', []), 
                                forecast_data.get('original_values', [])
                            )
                        ],
                        'model_info': {
                            'name': model_name
                        }
                    }
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            error_message = f"Error generating forecast: {e}"
            traceback.print_exc()
    
    # For GET request or after processing POST
    selected_model = request.args.get('model', request.form.get('model', ''))
    selected_dataset = request.args.get('dataset', request.form.get('dataset', ''))
    
    return render_template(
        'forecast.html',
        models=models,
        datasets=datasets,
        forecast_data=forecast_data,
        error=error_message,
        model_name=selected_model,
        dataset_name=selected_dataset,
        mlflow_url=MLFLOW_TRACKING_URI,
        direct_forecast=True
    )

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    """Upload a CSV dataset."""
    if 'dataset' not in request.files:
        flash("No file part", "error")
        return redirect(url_for('index'))
    
    file = request.files['dataset']
    
    if file.filename == '':
        flash("No selected file", "error")
        return redirect(url_for('index'))
    
    if file and file.filename.endswith('.csv'):
        try:
            filename = file.filename
            # Save file
            file_path = DATASETS_DIR / filename
            file.save(file_path)
            flash(f"Dataset {filename} uploaded successfully", "success")
        except Exception as e:
            flash(f"Error uploading dataset: {e}", "error")
    else:
        flash("Only CSV files are supported", "error")
    
    return redirect(url_for('index'))

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

# API Routes
@app.route('/api/health')
def health_check():
    """API health check endpoint."""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/tuning_results')
def api_tuning_results():
    """API endpoint to get all tuning results."""
    results = []
    if TUNING_RESULTS_DIR.exists():
        for file in TUNING_RESULTS_DIR.glob("*.json"):
            try:
                result_data = load_tuning_result(file)
                results.append({
                    'id': file.stem,
                    'model_type': result_data.get('model_type', 'Unknown'),
                    'timestamp': result_data.get('timestamp', 'Unknown'),
                    'status': result_data.get('status', 'Unknown'),
                    'best_score': result_data.get('best_score', 'N/A')
                })
            except Exception as e:
                logger.error(f"Error loading tuning result {file}: {e}")
    
    return jsonify(results)

@app.route('/api/tuning_result/<result_id>')
def api_tuning_result_detail(result_id):
    """API endpoint to get details of a specific tuning result."""
    result_file = TUNING_RESULTS_DIR / f"{result_id}.json"
    
    if not result_file.exists():
        return jsonify({'error': f"Tuning result {result_id} not found"}), 404
    
    try:
        result_data = load_tuning_result(result_file)
        return jsonify(result_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start_tuning', methods=['POST'])
def api_start_tuning():
    """API endpoint to start hyperparameter tuning."""
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['model_type', 'dataset']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f"Missing required field: {field}"}), 400
        
        # Check if dataset exists
        dataset_path = data.get('dataset')
        if not os.path.exists(dataset_path) and not os.path.exists(DATASETS_DIR / f"{dataset_path}"):
            if not os.path.exists(dataset_path):
                dataset_path = DATASETS_DIR / f"{dataset_path}"
                if not os.path.exists(dataset_path):
                    return jsonify({'error': f"Dataset not found: {dataset_path}"}), 404
        
        # Set up parameters
        params = {
            'model_type': data.get('model_type'),
            'dataset': str(dataset_path),
            'n_samples': data.get('n_samples', 10),
            'epochs': data.get('epochs', 50),
            'sequence_length': data.get('sequence_length', 10),
            'forecast_horizon': data.get('forecast_horizon', 5),
            'use_mlflow': data.get('use_mlflow', True)
        }
        
        if 'custom_params' in data:
            params['custom_params'] = data.get('custom_params')
        
        # Generate result ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_id = f"{params['model_type']}_tuning_{timestamp}"
        
        # Perform tuning
        result_file = TUNING_RESULTS_DIR / f"{result_id}.json"
        perform_tuning(params, result_file)
        
        return jsonify({
            'success': True,
            'result_id': result_id,
            'message': f"Tuning job initiated with ID: {result_id}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast', methods=['POST'])
def api_forecast():
    """API endpoint for generating forecasts."""
    try:
        # Get request data
        data = request.json
        model_name = data.get('model')
        dataset_name = data.get('dataset')
        horizon = int(data.get('horizon', 5))
        
        # Check if model and dataset exist
        model_path = MODELS_DIR / f"{model_name}.pt"
        dataset_path = DATASETS_DIR / f"{dataset_name}.csv"
        
        if not model_path.exists():
            return jsonify({'error': f"Model {model_name} not found"}), 404
        
        if not dataset_path.exists():
            return jsonify({'error': f"Dataset {dataset_name} not found"}), 404
        
        # Load model
        model = TimeSeriesWrapper()
        model.load(str(model_path))
        
        # Load and prepare data
        data_processor = DataProcessor()
        df = data_processor.load_data(dataset_path)
        
        # Scale data
        scaler_path = MODELS_DIR / f"{model_name}_scaler.joblib"
        if scaler_path.exists():
            try:
                scaler = joblib.load(scaler_path)
            except Exception as e:
                scaler = MinMaxScaler()
                scaler.fit(df.values)
                joblib.dump(scaler, scaler_path)
        else:
            scaler = MinMaxScaler()
            scaler.fit(df.values)
            joblib.dump(scaler, scaler_path)
        
        scaled_data = scaler.transform(df.values)
        
        # Create input sequence for prediction
        sequence_length = model.sequence_length
        input_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
        
        # Make prediction
        with torch.no_grad():
            forecast = model.predict(input_sequence)
        
        # Inverse transform the forecast
        forecast = forecast.reshape(-1, 1)
        forecast = scaler.inverse_transform(forecast)
        
        # Generate forecast dates
        last_date = df.index[-1]
        forecast_dates = []
        for i in range(1, len(forecast) + 1):
            forecast_dates.append(last_date + timedelta(days=i))
        
        # Create forecast data
        forecast_data = []
        for i, date in enumerate(forecast_dates):
            forecast_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': float(forecast[i][0])
            })
        
        # Create response
        response = {
            'model': model_name,
            'dataset': dataset_name,
            'forecast': forecast_data
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"API error generating forecast: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Function to check if the app is in a debuggable state
def check_app_status():
    """Check if the app is in a debuggable state."""
    try:
        # Check PyTorch
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Check if any models exist
        if MODELS_DIR.exists():
            models = list(MODELS_DIR.glob("*.pt"))
            logger.info(f"Found {len(models)} PyTorch models")
        
        # Check if any datasets exist
        if DATASETS_DIR.exists():
            datasets = list(DATASETS_DIR.glob("*.csv"))
            logger.info(f"Found {len(datasets)} datasets")
        
        return True
    except Exception as e:
        logger.error(f"Error checking app status: {e}")
        return False

if __name__ == '__main__':
    # Check app status
    check_app_status()
    
    # Start the app
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=True) 