#!/bin/bash
#
# This script BUILDS the Docker image and then RUNS the evaluation on ALL datasets 
# which have data available and are configured in the harness_dataset_config.txt file.
#
# It uses a key-value configuration file to map datasets to adapter modules.
# After building the image and loading the model, it executes evaluation on all
# configured datasets sequentially.
#
# Usage:
#   ./build_and_run_eval_container_for_all_datasets.sh
#
# Configuration is read from harness_dataset_config.txt

# --- Configuration ---
# Exit immediately if a command fails
set -e
IMAGE_NAME="multinet-eval"


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

# Scan data directory to find available datasets
echo "--> Scanning data directory for available datasets: $DATA_DIR"
AVAILABLE_DATASETS=""

# Check each potential dataset directory and map to dataset names
if [ -d "$DATA_DIR/bfcl_v3" ]; then
    AVAILABLE_DATASETS="$AVAILABLE_DATASETS bfcl"
fi
if [ -d "$DATA_DIR/odinw" ]; then
    AVAILABLE_DATASETS="$AVAILABLE_DATASETS odinw"
fi
if [ -d "$DATA_DIR/piqa" ]; then
    AVAILABLE_DATASETS="$AVAILABLE_DATASETS piqa"
fi
if [ -d "$DATA_DIR/overcooked_ai" ]; then
    AVAILABLE_DATASETS="$AVAILABLE_DATASETS overcooked_ai"
fi
if [ -d "$DATA_DIR/sqa3d" ]; then
    AVAILABLE_DATASETS="$AVAILABLE_DATASETS sqa3d"
fi

# Handle OpenX datasets
for openx_dir in "$DATA_DIR"/openx_*; do
    if [ -d "$openx_dir" ]; then
        dirname=$(basename "$openx_dir")
        if [ "$dirname" = "openx_multi_embodiment" ]; then
            # openx_multi_embodiment maps to robot_vqa
            AVAILABLE_DATASETS="$AVAILABLE_DATASETS robot_vqa"
        else
            # Other openx_* directories map directly to dataset names
            dataset_name="$dirname"
            AVAILABLE_DATASETS="$AVAILABLE_DATASETS $dataset_name"
        fi
    fi
done

# Remove leading/trailing whitespace
AVAILABLE_DATASETS=$(echo "$AVAILABLE_DATASETS" | sed 's/^ *//;s/ *$//')

if [ -z "$AVAILABLE_DATASETS" ]; then
    echo "Error: No datasets found in data directory '$DATA_DIR'"
    exit 1
fi

echo "--> Found datasets with available data:"
for DATASET in $AVAILABLE_DATASETS; do
    echo "  - $DATASET"
done

# Filter to only datasets that are both available in data AND configured
CONFIGURED_DATASETS=$(grep "^[^#].*\.adapter_module=" "$DATASET_CONFIG" | cut -d'.' -f1 | sort -u)
DATASETS=""

for DATASET in $AVAILABLE_DATASETS; do
    if echo "$CONFIGURED_DATASETS" | grep -q "^$DATASET$"; then
        DATASETS="$DATASETS $DATASET"
    else
        echo "Warning: Dataset '$DATASET' has data available but is not configured in '$DATASET_CONFIG', skipping..."
    fi
done

# Remove leading/trailing whitespace
DATASETS=$(echo "$DATASETS" | sed 's/^ *//;s/ *$//')

if [ -z "$DATASETS" ]; then
    echo "Error: No configured datasets found with available data"
    exit 1
fi

echo "--> Datasets to evaluate (available data + configured):"
for DATASET in $DATASETS; do
    echo "  - $DATASET"
done


# --- Step 1: Build the Docker Image ---
echo "--> Building Docker image: $IMAGE_NAME"
docker build \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    -t "$IMAGE_NAME" .
echo "--> Build complete."


# --- Step 2: Prepare Results Directory ---
RESULTS_DIR="$(pwd)/eval_results"
echo "--> Ensuring results directory exists at: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"


# --- Step 3: Run Evaluations for All Datasets ---
echo "--> Starting evaluation for all datasets..."

for DATASET in $DATASETS; do
    echo ""
    echo "=================================================================="
    echo "--> Processing dataset: $DATASET"
    echo "=================================================================="

    # Extract dataset configuration
    ADAPTER_MODULE=$(grep "^$DATASET\.adapter_module=" "$DATASET_CONFIG" | cut -d'=' -f2)
    BATCH_PROCESS=$(grep "^$DATASET\.batch_process=" "$DATASET_CONFIG" | cut -d'=' -f2)
    BATCH_SIZE=$(grep "^$DATASET\.batch_size=" "$DATASET_CONFIG" | cut -d'=' -f2)

    # Verify configuration exists
    if [ -z "$ADAPTER_MODULE" ]; then
        echo "Warning: Dataset '$DATASET' missing adapter_module configuration, skipping..."
        continue
    fi

    # Check if the required adapter file exists in the models directory
    EXPECTED_ADAPTER_PATH="${MODELS_DIR}/${ADAPTER_MODULE}"
    echo "--> Looking for model adapter: $EXPECTED_ADAPTER_PATH"
    if [ ! -f "$EXPECTED_ADAPTER_PATH" ]; then
        echo "Error: Model adapter '$ADAPTER_MODULE' not found in '$MODELS_DIR' for dataset '$DATASET', skipping..."
        continue
    fi
    echo "--> Found model adapter."

    # Display batch processing information
    if [ "$BATCH_PROCESS" = "true" ]; then
        echo "--> Dataset will be processed in batches"
        echo "--> Using batch size: $BATCH_SIZE"
    else
        echo "--> Dataset will be processed in single mode"
    fi

    # Build docker run command with conditional batch size
    DOCKER_ARGS="--dataset $DATASET --model_adapter_module_path /models/$ADAPTER_MODULE --output_path /home/app/multinet/results --disk_root_dir /data --max_samples 4"

    # Add batch processing arguments if dataset supports batch processing
    if [ "$BATCH_PROCESS" = "true" ]; then
        DOCKER_ARGS="$DOCKER_ARGS --batch_process --batch_size $BATCH_SIZE"
    fi

    echo "--> Running evaluation for $DATASET..."
    docker run \
        --gpus all \
        --rm \
        -v "$MODELS_DIR":/models \
        -v "$DATA_DIR":/data \
        -v "$RESULTS_DIR":/home/app/multinet/results \
        "$IMAGE_NAME" \
        $DOCKER_ARGS

    echo "--> Completed evaluation for dataset: $DATASET"

done

echo ""
echo "=================================================================="
echo "--> All evaluations complete. Results are in: $RESULTS_DIR"
echo "=================================================================="
