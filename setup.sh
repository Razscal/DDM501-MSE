#!/bin/bash
# Script to clean up and set up a fresh PyTorch-based time series forecasting environment

echo "🧹 Cleaning up old virtual environment..."
rm -rf venv
rm -rf __pycache__
find . -name "*.pyc" -delete

echo "🔧 Creating new virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🔍 Testing PyTorch installation..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('GPU available:', torch.cuda.is_available())"

echo "🌟 Setup complete! You can now activate the environment with:"
echo "source venv/bin/activate" 