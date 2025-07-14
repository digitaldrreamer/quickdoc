#!/bin/bash

# Document Text Extraction Service Runner
# This script starts the FastAPI service using uvicorn

echo "🚀 Starting Document Text Extraction Service..."
echo "📄 Service will be available at: http://localhost:8000"
echo "📊 API documentation at: http://localhost:8000/docs"
echo "🔧 Use Ctrl+C to stop the service"
echo ""

# Start the FastAPI service
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
