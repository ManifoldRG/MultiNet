#!/bin/bash

# Set the working directory to the script's directory
cd "$(dirname "$0")"

# Get the path to the conda executable
CONDA_PATH="$(which conda)"
if [ -z "$CONDA_PATH" ]; then
    # If conda is not in PATH, try common locations
    if [ -f "$HOME/miniconda3/bin/conda" ]; then
        CONDA_PATH="$HOME/miniconda3/bin/conda"
    elif [ -f "$HOME/anaconda3/bin/conda" ]; then
        CONDA_PATH="$HOME/anaconda3/bin/conda"
    fi
fi

# Source conda.sh which is the proper way to use conda in scripts
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Could not find conda.sh. Please specify correct conda path in the script."
    exit 1
fi

# Activate the conda environment
conda activate magma

# Print environment information for debugging
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"

# Clean installation approach - first PyTorch, then the rest
cd ../..

# First install PyTorch (needed for flash-attn)
echo "Installing PyTorch first..."
pip install torch torchvision

# Then install the Magma package with all dependencies 
echo "Installing Magma with all dependencies..."
pip install -e ".[server]"

cd server

# Run the FastAPI application
python main.py
