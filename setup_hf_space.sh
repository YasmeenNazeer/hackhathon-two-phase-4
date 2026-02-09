#!/bin/bash
# setup_hf_space.sh - Setup script for Hugging Face Space

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export PORT=7860

# Run the application
python space.py