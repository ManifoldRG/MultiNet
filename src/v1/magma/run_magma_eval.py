# File: run_magma_eval.py
import argparse
import json
import logging
import sys
import os
from overcooked_inference import MagmaOvercookedInferenceClass

# Constants for worst-case Brier scores (duplicated minimally for dummy output)
MAX_BRIER_MAE_ERROR = 2.0
MAX_BRIER_MSE_ERROR = 2.0

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Magma model evaluation on datasets.")
    
    parser.add_argument('--data_file', type=str, required=True, help='Path to the single data file.')
    parser.add_argument('--output_dir', type=str, default='./results/v1/magma/', help='Directory to save the output results JSON file.')
    parser.add_argument('--dataset_name', type=str, default='overcooked', help='Name of the dataset being evaluated (e.g., overcooked).')
    parser.add_argument('--results_filename', type=str, default='magma_results.json', help='Name for the output results file.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inference.')
    
    args = parser.parse_args()
    
    # Map dataset_name to inference class
    inference_classes = {
        'overcooked': MagmaOvercookedInferenceClass,
        # Add mappings for other datasets here in the future
    }
    
    if args.dataset_name not in inference_classes:
        logger.error(f"Unsupported dataset: {args.dataset_name}. Supported: {list(inference_classes.keys())}")
        sys.exit(1)
    
    inference_class = inference_classes[args.dataset_name]
    instance = inference_class(args)
    
    try:
        instance.run_evaluation()
    except Exception as e:
        logger.critical(f"Critical error in main: {e}; generating dummy output")
        # Generate dummy based on class attributes
        num_actions = instance.num_actions
        batch_size = args.batch_size
        final_metrics = {
            args.dataset_name: {
                "exact_match": 0.0,
                "total_dataset_amse": batch_size * MAX_BRIER_MSE_ERROR,
                "total_dataset_amae": batch_size * MAX_BRIER_MAE_ERROR,
                "num_timesteps": batch_size,
                "avg_dataset_amse": MAX_BRIER_MSE_ERROR,
                "avg_dataset_amae": MAX_BRIER_MAE_ERROR,
                "normalized_amse": MAX_BRIER_MSE_ERROR,
                "normalized_amae": MAX_BRIER_MAE_ERROR,
                "normalized_quantile_filtered_amae": MAX_BRIER_MAE_ERROR,
                "max_relative_mae": MAX_BRIER_MAE_ERROR,
                "proportion_beyond_threshold_mae": 1.0,
                "recall": 0.0,
                "precision": 0.0,
                "precision_without_invalid": 0.0,
                "f1": 0.0,
                "f1_without_invalid": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "class_precisions": [0.0] * num_actions,
                "class_recalls": [0.0] * num_actions,
                "class_f1s": [0.0] * num_actions,
                "total_invalids": batch_size,
                "percentage_invalids": 100.0,
                "n_invalid_outputs": batch_size,
                "preds": [28] * batch_size,
                "gt_actions": [0] * batch_size,
                "action_probabilities": [[1.0 / num_actions] * num_actions] * batch_size
            }
        }
        results_file = os.path.join(args.output_dir, args.results_filename)
        try:
            with open(results_file, 'w') as f:
                json.dump(final_metrics, f, indent=4)
            logger.info(f"Success! Dummy results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save dummy results: {e}; printing metrics")
            print("\n--- Detailed Metrics ---\n" + json.dumps(final_metrics, indent=2))

if __name__ == "__main__":
    main()