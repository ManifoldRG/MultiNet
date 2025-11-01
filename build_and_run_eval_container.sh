#!/bin/bash
#
# This script BUILDS the Docker image and then RUNS the evaluation.
#
# It uses a key-value configuration file to map datasets to adapter modules.
#
# Usage:
#   ./build_and_run_eval_container.sh DATASET_NAME
#
# Example:
#   ./build_and_run_eval_container.sh odinw
#

# --- Configuration ---
# Exit immediately if a command fails
set -e
IMAGE_NAME="multinet-eval"


# --- Argument Validation ---
# Check if the correct number of arguments (1) are provided.
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 DATASET_NAME"
    echo "All configuration is read from harness_dataset_config.txt"
    exit 1
fi

# Assign argument to variable for clarity
DATASET="$1"

# --- Dataset Configuration Check ---
# Check if dataset configuration file exists
DATASET_CONFIG="harness_dataset_config.txt"
if [ ! -f "$DATASET_CONFIG" ]; then
    echo "Error: Dataset configuration file not found: $DATASET_CONFIG"
    exit 1
fi

# Read global paths from config file
MODELS_DIR_RAW=$(grep "^models_dir=" "$DATASET_CONFIG" | cut -d'=' -f2)
DATA_DIR_RAW=$(grep "^data_dir=" "$DATASET_CONFIG" | cut -d'=' -f2)

# Convert relative paths to absolute paths for Docker
if [[ "$MODELS_DIR_RAW" = /* ]]; then
    MODELS_DIR="$MODELS_DIR_RAW"
else
    MODELS_DIR="$(pwd)/$MODELS_DIR_RAW"
fi

if [[ "$DATA_DIR_RAW" = /* ]]; then
    DATA_DIR="$DATA_DIR_RAW"
else
    DATA_DIR="$(pwd)/$DATA_DIR_RAW"
fi

# Check if the configured directories exist
if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: Models directory not found at '$MODELS_DIR'"
    echo "Please update models_dir in $DATASET_CONFIG"
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found at '$DATA_DIR'"
    echo "Please update data_dir in $DATASET_CONFIG"
    exit 1
fi


# Extract dataset configuration from key-value file using grep and cut
ADAPTER_MODULE=$(grep "^$DATASET\.adapter_module=" "$DATASET_CONFIG" | cut -d'=' -f2)
BATCH_PROCESS=$(grep "^$DATASET\.batch_process=" "$DATASET_CONFIG" | cut -d'=' -f2)
BATCH_SIZE=$(grep "^$DATASET\.batch_size=" "$DATASET_CONFIG" | cut -d'=' -f2)

# Check if dataset exists in configuration
if [ -z "$ADAPTER_MODULE" ]; then
    echo "Error: Dataset '$DATASET' not found in configuration file '$DATASET_CONFIG'"
    echo "Available datasets:"
    grep "^[^#].*\." "$DATASET_CONFIG" | cut -d'.' -f1 | sort -u | sed 's/^/  - /'
    exit 1
fi

# Check if the required adapter file exists in the models directory
EXPECTED_ADAPTER_PATH="${MODELS_DIR}/${ADAPTER_MODULE}"
echo "--> Looking for model adapter: $EXPECTED_ADAPTER_PATH"
if [ ! -f "$EXPECTED_ADAPTER_PATH" ]; then
    echo "Error: Model adapter '$ADAPTER_MODULE' not found in '$MODELS_DIR'"
    exit 1
fi
echo "--> Found model adapter."

# Display batch processing information
if [ "$BATCH_PROCESS" = "true" ]; then
    echo "--> Dataset will be processed in batches"
    echo "--> Using batch size: $BATCH_SIZE"
else
    echo "--> Dataset will be processed in single mode"
fi


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

# Build docker run command with conditional batch size
DOCKER_ARGS="--dataset $DATASET --model_adapter_module_path /models/$ADAPTER_MODULE --output_path /home/app/multinet/results --disk_root_dir /data"

# Add batch size if dataset supports batch processing
if [ "$BATCH_PROCESS" = "true" ]; then
    DOCKER_ARGS="$DOCKER_ARGS --batch_size $BATCH_SIZE"
fi

docker run \
    --gpus all \
    --rm \
    -v "$MODELS_DIR":/models \
    -v "$DATA_DIR":/data \
    -v "$RESULTS_DIR":/home/app/multinet/results \
    "$IMAGE_NAME" \
    $DOCKER_ARGS

echo "--> Evaluation complete. Results are in: $RESULTS_DIR"
