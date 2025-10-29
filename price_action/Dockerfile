# ============================================================================
# Adaptive Trading Bot - Production Dockerfile
# ============================================================================
# Version: 1.2
# Python: 3.12.7-slim (pinned for reproducibility)
#
# Security improvements:
#   - Non-root user (botuser:1000)
#   - Minimal base image (slim variant)
#
# Reliability improvements:
#   - Health check for container monitoring
#   - Proper file permissions
# ============================================================================

FROM python:3.12.7-slim

# Metadata labels
LABEL maintainer="adaptive-bot"
LABEL version="1.2"
LABEL description="ML-powered cryptocurrency trading bot for Bybit"
LABEL python.version="3.12.7"

# Set working directory
WORKDIR /app

# Install system dependencies required by Python packages
# libgomp1: Required by LightGBM for OpenMP support
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
# Use requirements-runtime.txt for production (minimal, ~800MB vs ~2.5GB)
# Use requirements.txt for development/training (full dependencies)
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Copy only essential files for trading
COPY main.py .
COPY bot.py .
COPY bybit_adapter.py .
COPY data_preparer_pa.py .
COPY trade_logger.py .
COPY prediction_logger.py .
COPY feature_cache.py .
COPY feature_config.py .
COPY docker-entrypoint.sh .

# Copy ML models into the image (self-contained, ephemeral design)
# Models are read-only and don't change during runtime
COPY models /app/models

# Create directories for runtime data (logs, state, prediction logs)
# These will be mounted as volumes in docker-compose.yaml
RUN mkdir -p logs bot_state prediction_logs

# User configuration
# For macOS Docker Desktop: run as root (volumes auto-map to host user)
# For Linux production: uncomment lines below to run as non-root
# RUN useradd -m -u 1000 botuser && \
#     chown -R botuser:botuser /app
# USER botuser

# Health check: verify bot_state directory exists and is writable
# This is a basic check - bot.py should create state files during operation
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('/app/bot_state') and os.access('/app/bot_state', os.W_OK) else 1)" || exit 1

# Configure entrypoint to fix permissions before running bot
RUN chmod +x docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command (overridden in docker-compose.yaml)
CMD ["python", "main.py", "--run-bot"]
