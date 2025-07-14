#!/bin/bash

# Start the Async Video Audio Text Service with User Authentication
echo "🚀 Starting Async Video Audio Text Service..."
echo "📱 Service will be available at: http://localhost:8000"
echo "📖 API documentation at: http://localhost:8000/docs"
echo "🏠 Home page at: http://localhost:8000"
echo "📊 Dashboard at: http://localhost:8000/dashboard"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your Google OAuth credentials before continuing."
    echo "   Get credentials from: https://console.cloud.google.com/"
    echo ""
    echo "Press Enter to continue with defaults (Google auth will not work)..."
    read
fi

# Ensure output directories exist
mkdir -p outputs/transcripts
mkdir -p outputs/speech
mkdir -p templates
mkdir -p static

echo "🔧 Starting service..."
echo ""

# Run the async service
python async_video_service.py
