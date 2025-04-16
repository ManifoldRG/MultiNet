import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
from src.eval_utils import (get_exact_match_rate,
                            calculate_tp_fp_fn_counts,
                            get_micro_precision_from_counts, 
                            get_micro_recall_from_counts, 
                            get_micro_f1)
from definitions.procgen import ProcGenDefinitions
import json
import numpy as np
from dataclasses import dataclass, field, fields

@dataclass
class DatasetResults:
    all_preds: list[list[float]] = field(default_factory=list)
    all_gt: list[list[float]] = field(default_factory=list)
    
    total_batches: int = 0
    total_timesteps: int = 0
    eval_time: float = 0
    total_invalid_predictions: int = 0
    invalid_predictions_percentage: float = 0
    total_emr: float = 0
    total_micro_precision: float = 0
    total_micro_recall: float = 0
    total_micro_f1: float = 0
    total_clipped_emr: float = 0
    total_clipped_micro_precision: float = 0
    total_clipped_micro_recall: float = 0
    total_clipped_micro_f1: float = 0
    total_micro_precision_without_invalids: float = 0
    total_micro_f1_without_invalids: float = 0

    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }

json_base_dir = input('Enter the path to the base directory of the JSON files: ')
procgen_datasets = ProcGenDefinitions.DESCRIPTIONS.keys()

for dataset in procgen_datasets:
    print(f"\nProcessing dataset: {dataset}")
    json_file_path = os.path.join(json_base_dir, f'pi0_base_procgen_results_{dataset}.json')

    results = {}
    with open(json_file_path, 'r') as f:
        results = json.load(f)

    all_preds = results[dataset]['all_preds']
    all_gts = results[dataset]['all_gt']

    # Convert predictions and ground truths to numpy arrays
    predictions = np.array(all_preds)
    ground_truths = np.array(all_gts)

    # Initialize DatasetResults object
    dataset_results = DatasetResults()
    dataset_results.all_preds = all_preds
    dataset_results.all_gt = all_gts
    dataset_results.total_timesteps = len(predictions)

    # Get valid action space for the dataset
    action_space = sorted(ProcGenDefinitions.get_valid_action_space(dataset, 'default'))

    # Calculate unclipped metrics
    emr = get_exact_match_rate(predictions, ground_truths)
    total_tp, total_fp, total_fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(
        predictions, ground_truths, action_space
    )

    micro_precision = get_micro_precision_from_counts(total_tp, total_fp)
    micro_recall = get_micro_recall_from_counts(total_tp, total_fn)
    micro_f1 = get_micro_f1(micro_precision, micro_recall)

    # Calculate metrics without invalid predictions
    micro_precision_without_invalids = get_micro_precision_from_counts(total_tp, valid_fp)
    micro_f1_without_invalids = get_micro_f1(micro_precision_without_invalids, micro_recall)

    # Calculate clipped metrics
    clipped_predictions = np.clip(predictions, action_space[0], action_space[-1])
    clipped_emr = get_exact_match_rate(clipped_predictions, ground_truths)
    clipped_total_tp, clipped_total_fp, clipped_total_fn, _, _ = calculate_tp_fp_fn_counts(
        clipped_predictions, ground_truths, action_space
    )
    clipped_micro_precision = get_micro_precision_from_counts(clipped_total_tp, clipped_total_fp)
    clipped_micro_recall = get_micro_recall_from_counts(clipped_total_tp, clipped_total_fn)
    clipped_micro_f1 = get_micro_f1(clipped_micro_precision, clipped_micro_recall)

    # Store all metrics in dataset_results
    dataset_results.total_invalid_predictions = int(invalid_fp)
    dataset_results.invalid_predictions_percentage = (invalid_fp / len(predictions)) * 100
    dataset_results.total_emr = emr 
    dataset_results.total_micro_precision = micro_precision 
    dataset_results.total_micro_recall = micro_recall 
    dataset_results.total_micro_f1 = micro_f1 

    dataset_results.total_clipped_emr = clipped_emr 
    dataset_results.total_clipped_micro_precision = clipped_micro_precision 
    dataset_results.total_clipped_micro_recall = clipped_micro_recall 
    dataset_results.total_clipped_micro_f1 = clipped_micro_f1 

    dataset_results.total_micro_precision_without_invalids = micro_precision_without_invalids 
    dataset_results.total_micro_f1_without_invalids = micro_f1_without_invalids 

    # Save results
    output_file = os.path.join(json_base_dir, f'corrected_pi0_base_procgen_results_{dataset}.json')
    with open(output_file, 'w') as f:
        json.dump({dataset: dataset_results.to_dict()}, f, indent=4)
    
    print(f"Results saved to: {output_file}")
    print(f"Dataset: {dataset}")
    print(f"EMR: {emr:.4f}")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Invalid Predictions: {invalid_fp} ({dataset_results.invalid_predictions_percentage:.2f}%)")

    
