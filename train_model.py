#!/usr/bin/env python
"""
Script to train a PyTorch time series model on a dataset and save it for forecasting.
"""

import os
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import MinMaxScaler

# Import custom modules
from mlib import TimeSeriesWrapper, DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = Path("models")
DATASETS_DIR = Path("datasets")

# Create necessary directories
MODELS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)

def train_model(dataset_name, model_type='lstm', hidden_size=50, num_layers=1, 
                dropout=0.2, learning_rate=0.001, epochs=100, batch_size=32,
                sequence_length=10, forecast_horizon=5):
    """
    Train a PyTorch time series model and save it.
    
    Args:
        dataset_name: Name of the dataset in the datasets directory
        model_type: Type of model ('lstm' or 'gru')
        hidden_size: Number of hidden units
        num_layers: Number of RNN layers
        dropout: Dropout rate
        learning_rate: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size
        sequence_length: Number of time steps to look back
        forecast_horizon: Number of time steps to forecast
    
    Returns:
        Path to the saved model
    """
    # Load dataset
    dataset_path = DATASETS_DIR / f"{dataset_name}.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} not found")
    
    logger.info(f"Training model on dataset: {dataset_name}")
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Prepare data
    X_train, y_train, X_test, y_test, scaler = data_processor.prepare_data(
        dataset_path=str(dataset_path),
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        train_split=0.8
    )
    
    logger.info(f"Data prepared: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    # Initialize model
    model = TimeSeriesWrapper(
        model_type=model_type,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate
    )
    
    # Train model
    logger.info("Training model...")
    model.train(
        X_train=X_train,
        y_train=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        patience=10
    )
    
    # Evaluate model
    metrics = model.evaluate(X_test=X_test, y_test=y_test)
    logger.info(f"Model evaluation: {metrics}")
    
    # Save model
    model_name = f"{dataset_name}_{model_type}_model"
    model_path = MODELS_DIR / f"{model_name}.pt"
    model.save(str(model_path))
    
    # Save scaler
    scaler_path = MODELS_DIR / f"{model_name}_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Scaler saved to {scaler_path}")
    
    return model_path

if __name__ == "__main__":
    # Train a model on the test dataset
    train_model(
        dataset_name="test_data",
        model_type="lstm",
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        epochs=100,
        batch_size=8,
        sequence_length=10,
        forecast_horizon=5
    ) 