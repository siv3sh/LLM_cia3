#!/bin/bash

# Multi-Agent Attrition Analysis System - Installation Script
# This script sets up the environment and installs dependencies

set -e

echo "🚀 Setting up Multi-Agent Attrition Analysis System..."

# Check if Python 3.10+ is installed
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.10+ is required. Current version: $python_version"
    echo "Please install Python 3.10+ and try again."
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install all dependencies
echo "📚 Installing all dependencies..."
pip install -r requirements.txt

# Create .env file from template if it doesn't exist
if [ ! -f .env ]; then
    echo "⚙️ Creating .env file from template..."
    cp env_template.env .env
    echo "⚠️ Please edit .env file with your API keys and configuration"
else
    echo "✅ .env file already exists"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs uploads exports vector_store models

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run the application: streamlit run streamlit_integrated.py --server.port 8502"
echo ""
echo "🔗 Open http://localhost:8502 in your browser to access the application"
echo ""
echo "📚 For more information, see README.md"
