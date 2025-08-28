#!/bin/bash

# Test script for all four OpenX datasets in parallel
# Tests the dynamic clipping framework with 1 shard each

set -e

# Create temp directory for testing
TEMP_DIR="/tmp/openx_parallel_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEMP_DIR"

echo "Testing all OpenX datasets in parallel..."
echo "Results will be stored in: $TEMP_DIR"
echo "Starting tests at $(date)"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Define datasets to test
DATASETS=(
    "openx_single_arm"
    "openx_bimanual"
    "openx_quadrupedal"
    "openx_mobile_manipulation"
)

# Function to run inference on a single dataset
run_dataset_test() {
    local dataset_name="$1"
    local dataset_path="$PROJECT_ROOT/src/v1/processed_datasets/$dataset_name"
    local output_dir="$TEMP_DIR/$dataset_name"
    
    echo "[$dataset_name] Starting test..."
    
    if [ ! -d "$dataset_path" ]; then
        echo "[$dataset_name] ERROR: Dataset directory not found: $dataset_path"
        return 1
    fi
    
    mkdir -p "$output_dir"
    
    # Running with GPU acceleration enabled
    # Using memory-limited settings to avoid GPU OOM
    timeout 300 python "$SCRIPT_DIR/openx_inference.py" \
        --output_dir "$output_dir" \
        --dataset_dir "$dataset_path" \
        --batch_size 1 \
        --num_shards 1 \
        > "$output_dir/stdout.log" 2> "$output_dir/stderr.log"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[$dataset_name] SUCCESS: Test completed"
    elif [ $exit_code -eq 124 ]; then
        echo "[$dataset_name] TIMEOUT: Test timed out after 5 minutes"
    else
        echo "[$dataset_name] ERROR: Test failed with exit code $exit_code"
        echo "[$dataset_name] Check logs at: $output_dir/stderr.log"
    fi
    
    return $exit_code
}

# Run all datasets sequentially
for dataset in "${DATASETS[@]}"; do
    run_dataset_test "$dataset"
done


echo ""
echo "All tests completed at $(date)"
echo ""

# Generate comprehensive summary
generate_summary() {
    echo "Results Summary:"
    echo "================"
    
    local success_count=0
    local total_count=${#DATASETS[@]}
    
    for dataset in "${DATASETS[@]}"; do
        local output_dir="$TEMP_DIR/$dataset"
        local status="UNKNOWN"
        local details=""
        
        if [ -f "$output_dir/stdout.log" ]; then
            # Check for various success indicators
            if grep -q "Processing batch\|Batch.*completed\|finished processing\|SUCCESS" "$output_dir/stdout.log" 2>/dev/null; then
                status="SUCCESS"
                ((success_count++))
            elif [ -f "$output_dir/stderr.log" ]; then
                # Analyze error types
                if grep -q "model loading\|checkpoint\|restore_params" "$output_dir/stderr.log" 2>/dev/null; then
                    status="MODEL_LOAD_ERROR"
                    details="Model checkpoint loading failed"
                elif grep -q "dimension\|shape\|tensor" "$output_dir/stderr.log" 2>/dev/null; then
                    status="DIMENSION_ERROR" 
                    details="Tensor dimension mismatch"
                elif grep -q "action.*dict\|dict.*action" "$output_dir/stderr.log" 2>/dev/null; then
                    status="ACTION_FORMAT_ERROR"
                    details="Action dictionary format issue"
                else
                    status="OTHER_ERROR"
                    details="Check stderr.log for details"
                fi
            fi
        else
            status="NO_OUTPUT"
        fi
        
        # Print status with color coding
        case $status in
            SUCCESS)        echo "✓ $dataset: SUCCESS" ;;
            MODEL_LOAD_ERROR) echo "⚠ $dataset: MODEL_LOAD_ERROR - $details" ;;
            DIMENSION_ERROR)  echo "⚠ $dataset: DIMENSION_ERROR - $details" ;;
            ACTION_FORMAT_ERROR) echo "⚠ $dataset: ACTION_FORMAT_ERROR - $details" ;;
            OTHER_ERROR)    echo "✗ $dataset: ERROR - $details" ;;
            NO_OUTPUT)      echo "✗ $dataset: NO_OUTPUT" ;;
            *)              echo "? $dataset: $status" ;;
        esac
    done
    
    echo ""
    echo "Summary: $success_count/$total_count datasets completed successfully"
    echo ""
}

# Show error details for failed tests
show_error_details() {
    echo "Error Details:"
    echo "=============="
    
    for dataset in "${DATASETS[@]}"; do
        local output_dir="$TEMP_DIR/$dataset"
        if [ -f "$output_dir/stderr.log" ] && [ -s "$output_dir/stderr.log" ]; then
            echo ""
            echo "--- $dataset errors (last 10 lines) ---"
            tail -10 "$output_dir/stderr.log"
        fi
    done
    echo ""
}

generate_summary

# Show error details if any tests failed
if [ "$success_count" -lt "${#DATASETS[@]}" ]; then
    show_error_details
fi

echo "Full results available in: $TEMP_DIR"
echo ""
echo "Quick commands:"
echo "  # View all outputs:           find $TEMP_DIR -name '*.log' -exec echo '=== {} ===' \; -exec cat {} \;"
echo "  # View specific dataset:      cat $TEMP_DIR/{dataset_name}/*.log"
echo "  # View only errors:           find $TEMP_DIR -name 'stderr.log' -exec echo '=== {} ===' \; -exec cat {} \;"