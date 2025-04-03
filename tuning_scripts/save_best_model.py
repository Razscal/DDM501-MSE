#!/usr/bin/env python
"""
Save Best Model Script
This script identifies the best tuning result and saves it as a model.
"""

import os
import sys
import json
import logging
import joblib
from datetime import datetime
from pathlib import Path
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlib import TimeSeriesModel, DataProcessor, load_tuning_result

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Save the best model from tuning results")
    
    parser.add_argument("--result-id", type=str, default=None,
                       help="Specific tuning result ID to use (if not provided, will use the best result)")
    
    parser.add_argument("--output-name", type=str, default=None,
                       help="Custom name for the saved model (defaults to result_id_best_model)")
    
    return parser.parse_args()

def find_best_tuning_result():
    """Find the best tuning result based on MSE."""
    results_dir = Path("tuning_results")
    if not results_dir.exists() or not list(results_dir.glob("*.json")):
        raise FileNotFoundError("No tuning results found in 'tuning_results' directory")
    
    best_result_file = None
    best_score = float('inf')
    
    # Check all tuning result files
    for result_file in results_dir.glob("*.json"):
        try:
            result_data = load_tuning_result(result_file)
            
            # Skip results that aren't completed
            if result_data.get('status') != 'completed':
                logger.info(f"Skipping incomplete result: {result_file.name}")
                continue
            
            # Get best score (MSE)
            if 'best_score' in result_data and result_data['best_score'] < best_score:
                best_score = result_data['best_score']
                best_result_file = result_file
                
        except Exception as e:
            logger.error(f"Error processing tuning result {result_file}: {e}")
    
    if best_result_file is None:
        raise ValueError("No completed tuning results found with valid scores")
    
    return best_result_file

def save_best_model(result_file, output_name=None):
    """
    Save the best model from a tuning result.
    
    Args:
        result_file: Path to the tuning result file
        output_name: Custom name for the saved model
    
    Returns:
        Path to the saved model
    """
    # Load tuning result
    result_data = load_tuning_result(result_file)
    result_id = result_file.stem
    
    # Check if the tuning is completed
    if result_data.get('status') != 'completed':
        raise ValueError(f"Cannot save model: tuning has not completed for {result_id}")
    
    # Get best parameters
    best_params = result_data.get('best_params')
    if not best_params:
        raise ValueError(f"No best parameters found in tuning result {result_id}")
    
    # Load dataset
    dataset_path = result_data.get('dataset')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Get model configuration
    model_type = result_data.get('model_type')
    sequence_length = result_data.get('sequence_length')
    forecast_horizon = result_data.get('forecast_horizon')
    
    logger.info(f"Loading dataset: {dataset_path}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Best parameters: {best_params}")
    
    # Create and train model with best parameters
    processor = DataProcessor()
    X_train, y_train, X_test, y_test, scaler = processor.prepare_data(
        dataset_path, 
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon
    )
    
    # Initialize model with best parameters
    model = TimeSeriesModel(
        model_type=model_type,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        **best_params
    )
    
    # Train model
    logger.info("Training model with best parameters...")
    model.train(X_train, y_train, epochs=result_data.get('epochs', 50))
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    logger.info(f"Model evaluation metrics: {metrics}")
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    if not output_name:
        output_name = f"{result_id}_best_model"
    
    model_path = models_dir / f"{output_name}.joblib"
    
    # Save model with metadata
    model_data = {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'params': best_params,
        'sequence_length': sequence_length,
        'forecast_horizon': forecast_horizon,
        'model_type': model_type,
        'source_tuning': result_id,
        'created_at': datetime.now().isoformat()
    }
    
    joblib.dump(model_data, model_path)
    logger.info(f"Model saved to: {model_path}")
    
    return model_path

def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        # Get tuning result to use
        if args.result_id:
            result_id = args.result_id
            result_file = Path("tuning_results") / f"{result_id}.json"
            
            if not result_file.exists():
                raise FileNotFoundError(f"Tuning result not found: {result_id}")
                
            logger.info(f"Using specified tuning result: {result_id}")
        else:
            # Find the best tuning result
            result_file = find_best_tuning_result()
            result_id = result_file.stem
            logger.info(f"Found best tuning result: {result_id}")
        
        # Save the model
        model_path = save_best_model(result_file, args.output_name)
        logger.info(f"Successfully saved best model from {result_id} to {model_path}")
        
    except Exception as e:
        logger.error(f"Error saving best model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 