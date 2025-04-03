#!/usr/bin/env python
"""
Time Series Forecasting Library
This module provides classes and functions for time series forecasting using PyTorch-based LSTM/GRU models.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
import uuid
from typing import Dict, List, Tuple, Any, Optional, Union
import mlflow
import mlflow.pytorch

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set up MLflow tracking URI
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5002")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data"""
    
    def __init__(self, X, y):
        # Check if X is already a tensor
        if isinstance(X, torch.Tensor):
            self.X = X
        else:
            self.X = torch.tensor(X, dtype=torch.float32)
        
        # Check if y is already a tensor
        if isinstance(y, torch.Tensor):
            self.y = y
        else:
            self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataProcessor:
    """Class for data preprocessing and preparation for time series forecasting."""
    
    def load_data(self, dataset_path: str, date_col: str = "date", value_col: str = "value") -> pd.DataFrame:
        """
        Load time series data from a CSV file.
        
        Args:
            dataset_path: Path to the CSV file
            date_col: Name of the date column
            value_col: Name of the value column
            
        Returns:
            DataFrame with datetime index and values
        """
        df = pd.read_csv(dataset_path)
        
        # Make column names case-insensitive by converting to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Update parameter names to lowercase for consistency
        date_col = date_col.lower()
        value_col = value_col.lower()
        
        # Convert date column to datetime and set as index
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        else:
            # Try to find a date column
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                # Use the first column that looks like a date
                date_col = date_columns[0]
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            else:
                # No date column found, use the first column and assume it's a date
                date_col = df.columns[0]
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df.set_index(date_col, inplace=True)
                except:
                    # If conversion fails, create a synthetic date index
                    logger.warning(f"No date column found in {dataset_path}. Creating synthetic date index.")
                    df = df.reset_index(drop=True)
                    df.index = pd.date_range(start='2022-01-01', periods=len(df), freq='D')
        
        # Keep only the value column if it exists
        if value_col in df.columns:
            df = df[[value_col]]
        else:
            # Try to find a value column
            value_columns = [col for col in df.columns if 'value' in col.lower() or 'price' in col.lower()]
            if value_columns:
                # Use the first column that looks like a value
                value_col = value_columns[0]
                df = df[[value_col]]
            else:
                # If no value column found, use the first non-index column
                if len(df.columns) > 0:
                    value_col = df.columns[0]
                    df = df[[value_col]]
                else:
                    raise ValueError(f"No value column found in {dataset_path}")
            
        return df
    
    def prepare_data(self, dataset_path: str, sequence_length: int = 10, 
                     forecast_horizon: int = 5, train_split: float = 0.8) -> Tuple:
        """
        Prepare data for time series forecasting.
        
        Args:
            dataset_path: Path to the dataset
            sequence_length: Number of time steps to look back
            forecast_horizon: Number of time steps to forecast
            train_split: Proportion of data to use for training
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test, scaler)
        """
        # Load data
        df = self.load_data(dataset_path)
        
        # Scale data using MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        
        # Create sequences for training
        X, y = self._create_sequences(scaled_data, sequence_length, forecast_horizon)
        
        # Split into training and testing sets
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, y_train, X_test, y_test, scaler
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int, 
                          forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input/output sequences for time series forecasting.
        
        Args:
            data: Scaled time series data
            sequence_length: Number of time steps to look back
            forecast_horizon: Number of time steps to forecast
            
        Returns:
            Tuple of (X, y) where X are input sequences and y are target values
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            # X: sequence_length time steps
            X.append(data[i:i+sequence_length])
            
            # y: forecast_horizon time steps
            y_seq = data[i+sequence_length:i+sequence_length+forecast_horizon]
            
            # Flatten forecast horizon dimension to create a vector target
            y.append(y_seq.flatten())
        
        X = np.array(X)
        y = np.array(y)
        
        # Ensure X has shape (samples, sequence_length, features)
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
        return X, y
    
    def generate_forecast_dates(self, last_date: datetime, horizon: int) -> List[datetime]:
        """
        Generate dates for forecast horizon.
        
        Args:
            last_date: Last date in the dataset
            horizon: Number of time steps to forecast
            
        Returns:
            List of dates for the forecast horizon
        """
        dates = []
        for i in range(1, horizon + 1):
            dates.append(last_date + timedelta(days=i))
        return dates

class TimeSeriesModel(nn.Module):
    """PyTorch-based LSTM/GRU time series forecasting model."""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 50, 
                 output_size: int = 5, model_type: str = "lstm", 
                 num_layers: int = 1, dropout: float = 0.2):
        """
        Initialize the model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of units in the recurrent layer
            output_size: Number of time steps to forecast
            model_type: Type of model ("lstm" or "gru")
            num_layers: Number of recurrent layers
            dropout: Dropout rate
        """
        super(TimeSeriesModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type.lower()
        self.output_size = output_size
        
        # RNN layer
        if model_type.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size=input_size, 
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0)
        elif model_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Choose 'lstm' or 'gru'.")
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state
        batch_size = x.size(0)
        
        # Forward propagate through RNN
        out, _ = self.rnn(x)
        
        # Take the output from the last time step
        out = out[:, -1, :]
        
        # Apply dropout and linear layer
        out = self.dropout(out)
        out = self.linear(out)
        
        return out

class TimeSeriesWrapper:
    """Wrapper class for PyTorch model with training and inference functionality."""
    
    def __init__(self, model_type: str = "lstm", sequence_length: int = 10, 
                 forecast_horizon: int = 5, hidden_size: int = 50, num_layers: int = 1,
                 dropout: float = 0.2, learning_rate: float = 0.001):
        """
        Initialize the wrapper.
        
        Args:
            model_type: Type of model ("lstm" or "gru")
            sequence_length: Number of time steps to look back
            forecast_horizon: Number of time steps to forecast
            hidden_size: Number of units in the recurrent layer
            num_layers: Number of recurrent layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
        """
        self.model_type = model_type.lower()
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = TimeSeriesModel(
            input_size=1,
            hidden_size=hidden_size,
            output_size=forecast_horizon,
            model_type=model_type,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.history = {'loss': [], 'val_loss': []}
        self.metrics = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              epochs: int = 50, batch_size: int = 32, 
              validation_split: float = 0.1, patience: int = 10) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            epochs: Number of epochs to train
            batch_size: Batch size
            validation_split: Proportion of training data to use for validation
            patience: Number of epochs to wait for improvement before early stopping
            
        Returns:
            Training history
        """
        # Reshape input data to (batch_size, sequence_length, features)
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        # Reshape target data to match output shape
        if len(y_train.shape) == 3:
            # If target is 3D with shape (batch_size, seq_len, features), flatten to 2D
            y_train = y_train.reshape(y_train.shape[0], -1)
        
        # Split training data into train and validation sets
        val_size = int(len(X_train) * validation_split)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        # Create data loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TimeSeriesDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Ensure outputs and targets have the same shape
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss = train_loss / len(train_loader.dataset)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Ensure outputs and targets have the same shape
                    loss = self.criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss = val_loss / len(val_loader.dataset)
            
            # Store losses
            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Print progress
            logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if patience > 0 and epochs_no_improve >= patience:
                logger.info(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Reshape input data if needed
        if len(X_test.shape) == 2:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Reshape target data to match output shape
        if len(y_test.shape) == 3:
            # If target is 3D with shape (batch_size, seq_len, features), flatten to 2D
            y_test = y_test.reshape(y_test.shape[0], -1)
        
        # Create data loader
        test_dataset = TimeSeriesDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Evaluate
        self.model.eval()
        all_preds = []
        all_targets = []
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Ensure outputs and targets have the same shape
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item() * inputs.size(0)
                
                # Store predictions and targets
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        test_loss = test_loss / len(test_loader.dataset)
        
        # Combine predictions and targets
        predictions = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        
        # If targets were flattened, reshape predictions for metric calculation
        if len(y_test.shape) == 2 and predictions.shape[1] == targets.shape[1]:
            # Both are already 2D and compatible
            pass
        else:
            # Reshape as needed
            predictions = predictions.reshape(-1)
            targets = targets.reshape(-1)
        
        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        self.metrics = {
            'test_loss': test_loss,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        return self.metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        # Reshape input data if needed
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def save(self, model_path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            model_path: Path to save the model
        """
        # Create a standalone dictionary with all model data
        model_data = {
            'pytorch_model': self.model,
            'model_type': self.model_type,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'history': self.history,
            'metrics': self.metrics,
            'created_at': datetime.now().isoformat()
        }
        
        # Use joblib with a specific protocol for better compatibility
        joblib.dump(model_data, model_path, protocol=4)
        logger.info(f"Model saved to {model_path}")
    
    def load(self, model_path: str) -> None:
        """
        Load the model from a file.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            # Try loading with joblib first
            import joblib
            try:
                model_data = joblib.load(model_path)
                
                # Check if this is a joblib-saved dictionary
                if isinstance(model_data, dict) and 'pytorch_model' in model_data:
                    self.model = model_data['pytorch_model']
                    self.model_type = model_data['model_type']
                    self.sequence_length = model_data['sequence_length']
                    self.forecast_horizon = model_data['forecast_horizon']
                    self.hidden_size = model_data['hidden_size']
                    self.num_layers = model_data.get('num_layers', 1)
                    self.dropout = model_data['dropout']
                    self.learning_rate = model_data['learning_rate']
                    self.history = model_data.get('history', {'loss': [], 'val_loss': []})
                    self.metrics = model_data.get('metrics', None)
                    
                    # Recreate optimizer
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
                    
                    logger.info(f"Model loaded from {model_path} using joblib")
                    return
            except Exception as e:
                logger.warning(f"Failed to load with joblib, trying PyTorch: {e}")
                
            # Fallback to PyTorch loading
            try:
                # Add numpy scalar type to safe globals for PyTorch 2.6+
                import torch.serialization
                torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
                checkpoint = torch.load(model_path, map_location=self.device)
                logger.info(f"Model loaded from {model_path} using PyTorch with safe globals")
            except Exception as torch_e:
                logger.warning(f"Error loading with safe globals: {torch_e}, trying with weights_only=False")
                # Try with weights_only=False as a fallback (less secure but more compatible)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                logger.info(f"Model loaded from {model_path} using PyTorch with weights_only=False")
            
            # Update model parameters
            self.model_type = checkpoint['model_type']
            self.sequence_length = checkpoint['sequence_length']
            self.forecast_horizon = checkpoint['forecast_horizon']
            self.hidden_size = checkpoint['hidden_size']
            self.num_layers = checkpoint.get('num_layers', 1)
            self.dropout = checkpoint['dropout']
            self.learning_rate = checkpoint['learning_rate']
            
            # Recreate model with loaded parameters
            self.model = TimeSeriesModel(
                input_size=1,
                hidden_size=self.hidden_size,
                output_size=self.forecast_horizon,
                model_type=self.model_type,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Recreate optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load history and metrics
            self.history = checkpoint.get('history', {'loss': [], 'val_loss': []})
            self.metrics = checkpoint.get('metrics', None)
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

def perform_tuning(params: Dict[str, Any], result_file: Union[str, Path]) -> None:
    """
    Start the hyperparameter tuning process in a separate thread.
    
    Args:
        params: Dictionary of tuning parameters
        result_file: Path to save the tuning results
    """
    # Initialize result file
    with open(result_file, 'w') as f:
        json.dump({
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'model_type': params.get('model_type', 'lstm'),
            'params': params
        }, f, indent=2, cls=NumpyEncoder)
    
    # Start tuning process in a separate thread
    thread = threading.Thread(target=_run_tuning_process, args=(params, result_file))
    thread.daemon = True
    thread.start()
    
    logger.info(f"Started tuning process with ID: {Path(result_file).stem}")

def _run_tuning_process(params: Dict[str, Any], result_file: Union[str, Path]) -> None:
    """
    Run hyperparameter tuning process.
    
    Args:
        params: Dictionary of tuning parameters
        result_file: Path to save the tuning results
    """
    try:
        logger.info(f"Starting hyperparameter tuning with params: {params}")
        
        # Extract parameters
        model_type = params.get('model_type', 'lstm')
        dataset_path = params.get('dataset')
        n_samples = params.get('n_samples', 10)
        epochs = params.get('epochs', 50)
        sequence_length = params.get('sequence_length', 10)
        forecast_horizon = params.get('forecast_horizon', 5)
        use_mlflow = params.get('use_mlflow', True)
        custom_params = params.get('custom_params', {})
        
        # Set up MLflow experiment
        experiment_name = f"{model_type}_tuning_{datetime.now().strftime('%Y%m%d')}"
        
        if use_mlflow:
            mlflow.set_experiment(experiment_name)
        
        # Define parameter space
        param_space = {
            'hidden_size': [32, 50, 64, 100, 128],
            'num_layers': [1, 2, 3],
            'learning_rate': [0.0001, 0.001, 0.01],
            'dropout': [0.0, 0.1, 0.2, 0.3, 0.5],
            'batch_size': [16, 32, 64, 128]
        }
        
        # Override with custom parameters if provided
        if custom_params:
            for key, value in custom_params.items():
                if key in param_space and isinstance(value, list):
                    param_space[key] = value
        
        # Prepare data
        data_processor = DataProcessor()
        X_train, y_train, X_test, y_test, scaler = data_processor.prepare_data(
            dataset_path=dataset_path,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )
        
        # Generate samples from parameter space
        results = []
        best_score = float('inf')
        best_params = None
        best_model = None
        
        for i in range(n_samples):
            # Sample parameters
            trial_params = {
                'hidden_size': random.choice(param_space['hidden_size']),
                'num_layers': random.choice(param_space['num_layers']),
                'learning_rate': random.choice(param_space['learning_rate']),
                'dropout': random.choice(param_space['dropout']),
                'batch_size': random.choice(param_space['batch_size'])
            }
            
            logger.info(f"Trial {i+1}/{n_samples}: {trial_params}")
            
            # Create and train model
            model = TimeSeriesWrapper(
                model_type=model_type,
                sequence_length=sequence_length,
                forecast_horizon=forecast_horizon,
                hidden_size=trial_params['hidden_size'],
                num_layers=trial_params['num_layers'],
                dropout=trial_params['dropout'],
                learning_rate=trial_params['learning_rate']
            )
            
            if use_mlflow:
                with mlflow.start_run(run_name=f"trial_{i+1}"):
                    # Log parameters
                    mlflow.log_param("model_type", model_type)
                    mlflow.log_param("sequence_length", sequence_length)
                    mlflow.log_param("forecast_horizon", forecast_horizon)
                    mlflow.log_param("hidden_size", trial_params['hidden_size'])
                    mlflow.log_param("num_layers", trial_params['num_layers'])
                    mlflow.log_param("learning_rate", trial_params['learning_rate'])
                    mlflow.log_param("dropout", trial_params['dropout'])
                    mlflow.log_param("batch_size", trial_params['batch_size'])
                    
                    # Train model
                    history = model.train(
                        X_train=X_train,
                        y_train=y_train,
                        epochs=epochs,
                        batch_size=trial_params['batch_size'],
                        validation_split=0.2,
                        patience=10
                    )
                    
                    # Evaluate model
                    metrics = model.evaluate(X_test=X_test, y_test=y_test)
                    
                    # Log metrics
                    mlflow.log_metric("rmse", metrics['rmse'])
                    mlflow.log_metric("mse", metrics['mse'])
                    mlflow.log_metric("r2", metrics['r2'])
                    mlflow.log_metric("final_loss", history['loss'][-1])
                    mlflow.log_metric("final_val_loss", history['val_loss'][-1])
                    
                    # Log model
                    mlflow.pytorch.log_model(model.model, "model")
            else:
                # Train and evaluate model without MLflow
                history = model.train(
                    X_train=X_train,
                    y_train=y_train,
                    epochs=epochs,
                    batch_size=trial_params['batch_size'],
                    validation_split=0.2,
                    patience=10
                )
                
                metrics = model.evaluate(X_test=X_test, y_test=y_test)
            
            # Store results
            trial_result = {
                'params': trial_params,
                'metrics': metrics,
                'history': {
                    'loss': history['loss'][-1],
                    'val_loss': history['val_loss'][-1]
                }
            }
            
            results.append(trial_result)
            
            # Update best model
            if metrics['rmse'] < best_score:
                best_score = metrics['rmse']
                best_params = trial_params
                best_model = model
            
            # Update result file with progress
            try:
                with open(result_file, 'r+') as f:
                    try:
                        result_data = json.load(f)
                        result_data['progress'] = {
                            'current': i + 1,
                            'total': n_samples
                        }
                        result_data['current_best'] = {
                            'score': best_score,
                            'params': best_params
                        }
                        result_data['results'] = results
                        f.seek(0)
                        json.dump(result_data, f, indent=2, cls=NumpyEncoder)
                        f.truncate()
                    except json.JSONDecodeError as e:
                        logger.error(f"Error reading progress file: {e}")
                        # File is corrupted, rewrite it completely
                        f.seek(0)
                        new_result_data = {
                            'status': 'running',
                            'timestamp': datetime.now().isoformat(),
                            'model_type': model_type,
                            'params': params,
                            'progress': {
                                'current': i + 1,
                                'total': n_samples
                            },
                            'current_best': {
                                'score': best_score,
                                'params': best_params
                            },
                            'results': results
                        }
                        json.dump(new_result_data, f, indent=2, cls=NumpyEncoder)
                        f.truncate()
            except Exception as e:
                logger.error(f"Failed to update progress file: {e}")
        
        # Save best model
        if best_model:
            model_filename = f"{Path(result_file).stem}_best_model.pt"
            model_path = Path("models") / model_filename
            best_model.save(model_path)
        
        # Update result file with final results
        try:
            with open(result_file, 'r+') as f:
                try:
                    result_data = json.load(f)
                    result_data['status'] = 'completed'
                    result_data['end_timestamp'] = datetime.now().isoformat()
                    result_data['best_score'] = best_score
                    result_data['best_params'] = best_params
                    result_data['best_model_path'] = str(model_path) if best_model else None
                    result_data['results'] = results
                    f.seek(0)
                    json.dump(result_data, f, indent=2, cls=NumpyEncoder)
                    f.truncate()
                except json.JSONDecodeError as e:
                    logger.error(f"Error reading final result file: {e}")
                    # File is corrupted, rewrite it completely
                    f.seek(0)
                    new_result_data = {
                        'status': 'completed',
                        'timestamp': datetime.now().isoformat(),
                        'end_timestamp': datetime.now().isoformat(),
                        'model_type': model_type,
                        'params': params,
                        'best_score': best_score,
                        'best_params': best_params,
                        'best_model_path': str(model_path) if best_model else None,
                        'results': results
                    }
                    json.dump(new_result_data, f, indent=2, cls=NumpyEncoder)
                    f.truncate()
        except Exception as e:
            logger.error(f"Failed to update final result file: {e}")
        
        logger.info(f"Hyperparameter tuning completed. Best score: {best_score}, Best params: {best_params}")
    
    except Exception as e:
        logger.error(f"Error in tuning process: {e}")
        
        # Update result file with error
        try:
            with open(result_file, 'r+') as f:
                try:
                    result_data = json.load(f)
                    result_data['status'] = 'error'
                    result_data['error'] = str(e)
                    f.seek(0)
                    json.dump(result_data, f, indent=2, cls=NumpyEncoder)
                    f.truncate()
                except json.JSONDecodeError:
                    # File is corrupted, rewrite it completely
                    f.seek(0)
                    new_result_data = {
                        'status': 'error',
                        'timestamp': datetime.now().isoformat(),
                        'model_type': model_type if 'model_type' in locals() else 'unknown',
                        'error': str(e)
                    }
                    json.dump(new_result_data, f, indent=2, cls=NumpyEncoder)
                    f.truncate()
        except Exception as err:
            logger.error(f"Failed to update error in result file: {err}")

def load_tuning_result(result_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a tuning result from a file.
    
    Args:
        result_file: Path to the tuning result file
        
    Returns:
        Dictionary containing the tuning results or empty dict if file is corrupted
    """
    try:
        with open(result_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error loading tuning result {result_file}: {e}")
        return {"status": "error", "error": f"Corrupted JSON file: {e}"}
    except FileNotFoundError:
        logger.error(f"Tuning result file not found: {result_file}")
        return {"status": "error", "error": "File not found"}
    except Exception as e:
        logger.error(f"Unexpected error loading tuning result {result_file}: {e}")
        return {"status": "error", "error": str(e)} 