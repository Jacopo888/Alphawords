#!/bin/bash

# Start AlphaScrabble Web Server

set -e

echo "🚀 Starting AlphaScrabble Web Server..."

# Check if we're in the right directory
if [ ! -f "web/app.py" ]; then
    echo "❌ Error: web/app.py not found. Please run from project root."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r web/requirements.txt
pip install -e .

# Start the web server
echo "🌐 Starting web server on http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

cd web
python app.py
