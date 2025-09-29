import argparse
import logging
import sys
from src.v1.modules.Magma.scripts.overcooked_inference import MagmaOvercookedInferenceClass

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
        logger.critical(f"Critical error in main: {e}; unable to run evaluation.")

if __name__ == "__main__":
    main()