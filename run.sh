#!/bin/bash

# --- Configuration ---
SERVICE_PORT=8002

# --- Script Logic ---
echo "▶️ Checking for running process on port ${SERVICE_PORT}..."

# Use fuser to find and kill the process.
# The '-k' flag sends the SIGKILL signal (forceful kill).
# The '-s' flag runs it in silent mode.
fuser -k -s ${SERVICE_PORT}/tcp

echo "🚀 Starting Document & AI Processor..."
# ... (rest of the script is the same)
uvicorn app.main:app --host 0.0.0.0 --port ${SERVICE_PORT} --reload