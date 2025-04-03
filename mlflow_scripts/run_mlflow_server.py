#!/usr/bin/env python
"""
MLflow Server Script
This script starts the MLflow tracking server with custom configuration.
"""

import os
import subprocess
import sys
from pathlib import Path

# Define default paths
DEFAULT_PORT = 5002
DEFAULT_MLFLOW_DIR = Path("mlflow_data")
DEFAULT_ARTIFACTS_DIR = DEFAULT_MLFLOW_DIR / "artifacts"

def ensure_directories_exist(mlflow_dir, artifacts_dir):
    """Create the necessary directories if they don't exist."""
    mlflow_dir.mkdir(exist_ok=True, parents=True)
    artifacts_dir.mkdir(exist_ok=True, parents=True)
    print(f"MLflow directories configured: {mlflow_dir}, {artifacts_dir}")

def start_mlflow_server(port, mlflow_dir, artifacts_dir):
    """Start the MLflow tracking server with the specified configuration."""
    ensure_directories_exist(mlflow_dir, artifacts_dir)
    
    # Set environment variable for MLflow tracking URI
    os.environ["MLFLOW_TRACKING_URI"] = f"http://localhost:{port}"
    
    # Build the command for starting MLflow server
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", f"sqlite:///{mlflow_dir}/mlflow.db",
        "--default-artifact-root", f"{artifacts_dir}",
        "--host", "0.0.0.0",
        "--port", str(port)
    ]
    
    print(f"Starting MLflow server: {' '.join(cmd)}")
    
    try:
        # Start the MLflow server
        process = subprocess.Popen(cmd)
        print(f"MLflow server started with process ID: {process.pid}")
        print(f"MLflow UI available at: http://localhost:{port}")
        # Keep the process running
        process.wait()
    except KeyboardInterrupt:
        print("Shutting down MLflow server...")
        process.terminate()
        sys.exit(0)
    except Exception as e:
        print(f"Error starting MLflow server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Get configuration from environment variables or use defaults
    port = int(os.environ.get("MLFLOW_PORT", DEFAULT_PORT))
    mlflow_dir = Path(os.environ.get("MLFLOW_DIR", DEFAULT_MLFLOW_DIR))
    artifacts_dir = Path(os.environ.get("MLFLOW_ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
    
    start_mlflow_server(port, mlflow_dir, artifacts_dir) 