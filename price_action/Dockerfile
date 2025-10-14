FROM python:3.13

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only essential files for trading
COPY main.py .
COPY bot.py .
COPY bybit_adapter.py .
COPY data_preparer_pa.py .
COPY trade_logger.py .

# Create directories for runtime data
RUN mkdir -p models logs bot_state

CMD ["python", "main.py", "--run-bot"]