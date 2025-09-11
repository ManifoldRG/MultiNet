#!/bin/bash
#
# This script BUILDS the Docker image and then RUNS the evaluation.
#
# It enforces a naming convention for model adapters: {dataset}_adapter.py
#
# Usage:
#   ./build_and_run_eval_container.sh /path/to/your/models /path/to/your/data DATASET_NAME
#
# Example:
#   ./build_and_run_eval_container.sh /home/user/my_models /home/user/my_data openx
#

# --- Configuration ---
# Exit immediately if a command fails
set -e
IMAGE_NAME="multinet-eval"


# --- Argument Validation ---
# Check if the correct number of arguments (3) are provided.
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 /path/to/your/models /path/to/your/data DATASET_NAME"
    echo "DATASET_NAME_adapter.py (e.g. openx_adapter.py) that implements ModelAdapter must be inside the models folder provided."
    exit 1
fi

# Assign arguments to variables for clarity
MODELS_DIR="$1"
DATA_DIR="$2"
DATASET="$3"

# Check if the provided directories exist
if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: Models directory not found at '$MODELS_DIR'"
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found at '$DATA_DIR'"
    exit 1
fi


# --- Model Adapter Check ---
# Construct the expected model adapter filename based on the dataset
ADAPTER_FILENAME="${DATASET}_adapter.py"
EXPECTED_ADAPTER_PATH="${MODELS_DIR}/${ADAPTER_FILENAME}"

# Check if the required adapter file exists in the models directory
echo "--> Looking for model adapter: $EXPECTED_ADAPTER_PATH"
if [ ! -f "$EXPECTED_ADAPTER_PATH" ]; then
    echo "Error: Model adapter '$ADAPTER_FILENAME' not found in '$MODELS_DIR'"
    exit 1
fi
echo "--> Found model adapter."


# --- Step 1: Build the Docker Image ---
echo "--> Building Docker image: $IMAGE_NAME"
docker build \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  -t "$IMAGE_NAME" .
echo "--> Build complete."


# --- Step 2: Prepare and Run the Container ---
RESULTS_DIR="$(pwd)/eval_results"
echo "--> Ensuring results directory exists at: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

echo "--> Starting evaluation for dataset: $DATASET"
docker run \
  --gpus all \
  --rm \
  -v "$MODELS_DIR":/models \
  -v "$DATA_DIR":/data \
  -v "$RESULTS_DIR":/home/app/multinet/results \
  "$IMAGE_NAME" \
    --dataset "$DATASET" \
    --model_adapter_module_path "/models/$ADAPTER_FILENAME" \
    --output_path /home/app/multinet/results \
    --disk_root_dir /data

echo "--> Evaluation complete. Results are in: $RESULTS_DIR"
