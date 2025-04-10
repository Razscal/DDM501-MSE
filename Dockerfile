FROM python:3.9-slim

# Add a label with build date
LABEL build_date="$(date)"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p models static mlflow_data/artifacts tuning_results datasets

# Expose ports for Flask and MLflow
EXPOSE 5001 5002

# Setup environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=app.py
ENV MLFLOW_TRACKING_URI=http://localhost:5002

# Start both servers using script
COPY <<'EOT' /app/start.sh
#!/bin/bash
# Start MLflow server in background
python mlflow_scripts/run_mlflow_server.py &
# Wait for MLflow to start
sleep 5
# Start Flask app
python app.py
EOT

RUN chmod +x /app/start.sh

# Use script as entrypoint
CMD ["/app/start.sh"] 