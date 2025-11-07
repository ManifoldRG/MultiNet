#!/usr/bin/env python3
"""
Approximate baseline-relative metrics from aggregate MAE/MSE values.

This is an APPROXIMATION. The exact metric would require per-sample predictions.

The script handles these result file formats:
- Metrics directly under dataset keys (e.g., pi0 results)
- Metrics under a "metrics" subkey (e.g., magma results)
- Metrics under the dataset name as a key

Examples:

# Process a single dataset:
python approx_baseline_rel_err.py \
    --results_file src/v1/results/magma/magma_openx_bimanual_results.json \
    --dataset openx_bimanual \
    --data_split public \
    --disk_root_dir /mnt/disks/mount_dir/MultiNet/src/v1/processed \
    --output approximate_bimanual_metrics.json

# Process all datasets in a multi-dataset results file:
python approx_baseline_rel_err.py \
    --results_file src/v1/results/pi0/pi0_base_openx_results_final.json \
    --data_split public \
    --disk_root_dir /mnt/disks/mount_dir/MultiNet/src/v1/processed \
    --output approximate_all_datasets_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data_utils import find_data_files, get_openx_dataloader
from src.eval_utils import calculate_mae, calculate_mse


def calculate_baseline_metrics_from_dataset(
    dataset_name: str,
    data_split: str,
    disk_root_dir: str,
    num_samples: int = None
):
    """Calculate what baseline MAE/MSE would be for the dataset."""
    
    print(f"Loading dataset: {dataset_name} (split: {data_split})")
    files = find_data_files('openx', disk_root_dir, dataset=dataset_name, split=data_split)
    dataset, data_loader = get_openx_dataloader(files, batch_size=1, dataset_name=dataset_name)
    
    # Get training mean
    training_mean = np.array(dataset.action_stats['mean'])
    print(f"Training mean: {training_mean}")
    
    # Calculate baseline MAE and MSE for each sample
    baseline_maes = []
    baseline_mses = []

    # Use DataLoader to safely iterate through samples
    total_samples = 0
    max_samples = num_samples if num_samples else float('inf')

    print(f"Processing up to {max_samples if max_samples != float('inf') else 'all'} samples...")

    for batch in data_loader:
        if total_samples >= max_samples:
            break

        # Since batch_size=1, batch is a dict with lists containing single elements
        gt_action = batch['action'][0]  # Take the first (only) element

        # Calculate what MAE/MSE would be if we always predicted training_mean
        baseline_mae = calculate_mae(training_mean, gt_action)
        baseline_mse = calculate_mse(training_mean, gt_action)

        baseline_maes.append(baseline_mae)
        baseline_mses.append(baseline_mse)

        total_samples += 1

    samples_processed = len(baseline_maes)
    
    # Average across all samples
    avg_baseline_mae = np.mean(baseline_maes)
    avg_baseline_mse = np.mean(baseline_mses)
    
    return {
        'training_mean': training_mean.tolist(),
        'num_samples': samples_processed,
        'avg_baseline_mae': avg_baseline_mae,
        'avg_baseline_mse': avg_baseline_mse,
        'per_sample_baseline_maes': baseline_maes,
        'per_sample_baseline_mses': baseline_mses,
        'action_stats': dataset.action_stats
    }


def extract_metrics_from_results(model_metrics: dict, dataset_name: str = None):
    """Extract metrics from results, checking multiple possible locations."""
    # Try to get metrics directly first
    metrics_source = model_metrics

    # If metrics aren't directly available, check under 'metrics' key
    if 'avg_dataset_amae' not in model_metrics and 'metrics' in model_metrics:
        metrics_source = model_metrics['metrics']
    # If still not found and dataset_name is provided, check under dataset name as key
    elif 'avg_dataset_amae' not in metrics_source and dataset_name and dataset_name in model_metrics:
        metrics_source = model_metrics[dataset_name]

    # Extract the required metrics
    try:
        model_mae = metrics_source['avg_dataset_amae']
        model_mse = metrics_source['avg_dataset_amse']
        num_timesteps = metrics_source['num_timesteps']
        return model_mae, model_mse, num_timesteps
    except KeyError as e:
        raise KeyError(f"Required metric key '{e}' not found in results. Available keys: {list(metrics_source.keys())}")


def calculate_approximate_relative_metrics(
    model_mae: float,
    model_mse: float,
    baseline_mae: float,
    baseline_mse: float
):
    """Calculate approximate baseline-relative metrics."""

    # APPROXIMATION: Use ratio of averages instead of average of ratios
    approx_baseline_relative_mae = model_mae / baseline_mae
    approx_baseline_relative_mse = model_mse / baseline_mse

    return {
        'approximate_baseline_relative_mae': approx_baseline_relative_mae,
        'approximate_baseline_relative_mse': approx_baseline_relative_mse
    }


def main():
    parser = argparse.ArgumentParser(
        description="Approximate baseline-relative metrics from aggregate values"
    )
    parser.add_argument('--results_file', type=str, required=True,
                        help="Path to results JSON with avg_dataset_amae/amse")
    parser.add_argument('--dataset', type=str, default=None,
                        help="Dataset name (e.g., openx_bimanual). If not specified, process all datasets in results file")
    parser.add_argument('--data_split', type=str, default='public',
                        help="Data split to use")
    parser.add_argument('--disk_root_dir', type=str,
                        default='/mnt/disks/mount_dir/MultiNet/src/v1/processed',
                        help="Root directory containing dataset files")
    parser.add_argument('--num_samples', type=int, default=None,
                        help="Number of samples to process (default: all)")
    parser.add_argument('--output', type=str, default='approximate_metrics.json',
                        help="Output JSON file")
    
    args = parser.parse_args()

    # Load model results
    print(f"Loading model results from: {args.results_file}")
    with open(args.results_file, 'r') as f:
        results = json.load(f)

    # Determine which datasets to process
    if args.dataset:
        # Single dataset mode
        datasets_to_process = [args.dataset]
    else:
        # Multi-dataset mode - process all datasets in results file
        datasets_to_process = list(results.keys())
        print(f"Found {len(datasets_to_process)} datasets in results file: {datasets_to_process}")

    # Process each dataset
    all_results = {}

    for dataset_name in datasets_to_process:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")

        # Extract model MAE/MSE for this dataset
        if dataset_name in results:
            model_metrics = results[dataset_name]
        else:
            print(f"Warning: Dataset '{dataset_name}' not found in results file. Skipping.")
            continue

        try:
            model_mae, model_mse, num_timesteps = extract_metrics_from_results(model_metrics, dataset_name)
        except KeyError as e:
            print(f"Warning: {e}. Skipping dataset '{dataset_name}'.")
            continue

        print(f"\nModel Metrics (from results file):")
        print(f"  Samples: {num_timesteps}")
        print(f"  Avg MAE: {model_mae:.4f}")
        print(f"  Avg MSE: {model_mse:.4f}")

        # Calculate baseline metrics
        baseline_metrics = calculate_baseline_metrics_from_dataset(
            dataset_name,
            args.data_split,
            args.disk_root_dir,
            num_samples=args.num_samples or num_timesteps
        )

        baseline_mae = baseline_metrics['avg_baseline_mae']
        baseline_mse = baseline_metrics['avg_baseline_mse']

        print(f"\nBaseline Metrics (from dataset):")
        print(f"  Avg Baseline MAE: {baseline_mae:.4f}")
        print(f"  Avg Baseline MSE: {baseline_mse:.4f}")

        # Calculate approximate relative metrics
        relative_metrics = calculate_approximate_relative_metrics(
            model_mae, model_mse, baseline_mae, baseline_mse
        )

        # Store results for this dataset
        all_results[dataset_name] = {
            'model_metrics': {
                'avg_dataset_amae': model_mae,
                'avg_dataset_amse': model_mse,
                'num_timesteps': num_timesteps
            },
            'baseline_metrics': baseline_metrics,
            'approximate_relative_metrics': relative_metrics
        }

        print(f"\nAPPROXIMATE RESULTS for {dataset_name}")
        print(f"Model MAE / Baseline MAE = {relative_metrics['approximate_baseline_relative_mae']:.4f}")
        print(f"Model MSE / Baseline MSE = {relative_metrics['approximate_baseline_relative_mse']:.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("ALL RESULTS SAVED")
    print(f"{'='*60}")
    print(f"Processed {len(all_results)} datasets")
    print(f"Full results saved to: {output_path}")


if __name__ == "__main__":
    main()