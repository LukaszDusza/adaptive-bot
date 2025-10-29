# Backend Dockerfile for production deployment

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY run.py .

# Create directories for mounted volumes
RUN mkdir -p /app/logs /app/bot_state /app/models /app/prediction_logs

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Run application
CMD ["python", "run.py", "--prod", "--host", "0.0.0.0", "--port", "8000"]
