#!/bin/bash
export BYBIT_API_KEY="AQuYHxou38tOrfx5Lw"
export BYBIT_API_SECRET="7R63fmKf2cttDin5rfQRecawJVf86reoQYX5"
export BYBIT_BASE_URL="https://api.bybit.com"

python -m uvicorn app.main:app --port 8000
