#!/usr/bin/env python
"""
Simple Hyperparameter Tuning Script
This script performs hyperparameter tuning with predefined parameter spaces.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlib import perform_tuning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define predefined parameter spaces
PARAMETER_SPACES = {
    "small": {
        "units": [32, 64],
        "dropout": [0.2, 0.3],
        "learning_rate": [0.001, 0.01]
    },
    "medium": {
        "units": [32, 64, 128],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": [0.001, 0.005, 0.01]
    },
    "large": {
        "units": [32, 64, 128, 256],
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
        "learning_rate": [0.0001, 0.001, 0.005, 0.01]
    }
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple hyperparameter tuning for time series forecasting")
    
    parser.add_argument("--model", type=str, default="lstm",
                       help="Model type: 'lstm' or 'gru'")
    
    parser.add_argument("--space", type=str, default="medium",
                       choices=PARAMETER_SPACES.keys(),
                       help="Parameter space size: 'small', 'medium', or 'large'")
    
    parser.add_argument("--dataset", type=str, default="datasets/sample_data.csv",
                       help="Path to the dataset CSV file")
    
    parser.add_argument("--samples", type=int, default=10,
                       help="Number of random samples to try")
    
    parser.add_argument("--epochs", type=int, default=50,
                       help="Maximum number of training epochs")
    
    parser.add_argument("--sequence-length", type=int, default=10,
                       help="Sequence length (lookback period)")
    
    parser.add_argument("--forecast-horizon", type=int, default=5,
                       help="Forecast horizon (number of steps to predict)")
    
    parser.add_argument("--no-mlflow", action="store_true", 
                       help="Disable MLflow logging")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        # Get parameter space
        param_space = PARAMETER_SPACES[args.space]
        logger.info(f"Using {args.space} parameter space: {param_space}")
        
        # Create results directory if it doesn't exist
        results_dir = Path("tuning_results")
        results_dir.mkdir(exist_ok=True)
        
        # Generate result ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_id = f"{args.model}_{args.space}_tuning_{timestamp}"
        result_file = results_dir / f"{result_id}.json"
        
        # Set up parameters
        params = {
            'model_type': args.model,
            'dataset': args.dataset,
            'n_samples': args.samples,
            'epochs': args.epochs,
            'sequence_length': args.sequence_length,
            'forecast_horizon': args.forecast_horizon,
            'use_mlflow': not args.no_mlflow,
            'custom_params': param_space
        }
        
        logger.info(f"Starting simple hyperparameter tuning with ID: {result_id}")
        logger.info(f"Parameters: {params}")
        
        # Perform tuning
        perform_tuning(params, result_file)
        
        logger.info(f"Tuning job initiated. Results will be saved to: {result_file}")
        
    except Exception as e:
        logger.error(f"Error in simple hyperparameter tuning: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 