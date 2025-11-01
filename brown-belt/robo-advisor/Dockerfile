# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and model files
COPY . .

# Make sure the model file exists (should be created by train-script.py)
# If model.joblib doesn't exist, the container build will fail here
RUN test -f model.joblib || (echo "Error: model.joblib not found. Run train-script.py first!" && exit 1)

# Expose port 8000
EXPOSE 8000

# Health check to ensure the API is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]