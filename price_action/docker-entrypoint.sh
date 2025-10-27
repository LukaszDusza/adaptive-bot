#!/bin/bash
set -e

# Fix permissions for mounted volumes (needed for macOS Docker Desktop)
# Ensure bot can write to logs, bot_state, and prediction_logs
chmod -R 777 /app/logs /app/bot_state /app/prediction_logs 2>/dev/null || true

# Execute the main command (passed as arguments to this script)
exec "$@"