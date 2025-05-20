import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive Agg
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from collections import Counter
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import seaborn as sns
import pandas as pd
from pprint import pprint
from scipy.interpolate import make_interp_spline

current_file = Path(__file__).resolve()
project_root = next(
    (p for p in current_file.parents if p.name == "MultiNet"),
    current_file.parent
)
sys.path.append(str(project_root))
from definitions.procgen import ProcGenDefinitions
from src.eval_utils import get_precision_per_class, get_recall_per_class, get_f1_per_class, get_macro_precision, get_macro_recall, get_macro_f1, calculate_tp_fp_fn_counts, get_micro_precision_from_counts, get_micro_recall_from_counts, get_micro_f1

COLORS = ["#ea5545",
          "#ede15b",
          "#87bc45",
          "#27aeef",
          "#b33dc6"]

def load_results(results_dir, models=['gpt4o', 'openvla', 'pi0_base', 'pi0_fast', 'gpt4_1']):
    """Load results from json files in the specified directory"""
    results = {}

    for model_name in models:
        if model_name == 'pi0_base':
            results[model_name] = {}
            results_path = os.path.join(results_dir, model_name, 'procgen_results', 'corrected_results')
            for dataset in os.listdir(results_path):
                #print(dataset)
                # Split filename and look for matching dataset name
                filename_parts = dataset.split('_')
                #print(filename_parts)
                for part in filename_parts:
                    if part.split('.')[0] in ProcGenDefinitions.DESCRIPTIONS:
                        #print(part)
                        dataset_name = part.split('.')[0]
                        with open(os.path.join(results_path, dataset), 'r') as f:
                            results[model_name][dataset_name] = json.load(f)
                            #print(results[model_name][dataset_name])
                        break
    
        elif model_name == 'gpt4o':
            results[model_name] = {}
            results_path = os.path.join(results_dir, model_name, 'procgen_results', 'corrected_results')
            for dataset in os.listdir(results_path):
                filename_parts = dataset.split('_')
                for part in filename_parts:
                    if part in ProcGenDefinitions.DESCRIPTIONS:
                        dataset_name = part
                        with open(os.path.join(results_path, dataset), 'r') as f:
                            results[model_name][dataset_name] = json.load(f)
                            #print(results[model_name][dataset_name])
                        break
        elif model_name == 'gpt4_1':
            results[model_name] = {}
            results_path = os.path.join(results_dir, 'gpt4o', 'procgen_results', 'gpt4.1_new_prompt')
            for dataset in os.listdir(results_path):
                filename_parts = dataset.split('_')
                for part in filename_parts:
                    if part in ProcGenDefinitions.DESCRIPTIONS:
                        dataset_name = part
                        with open(os.path.join(results_path, dataset), 'r') as f:
                            results[model_name][dataset_name] = json.load(f)
                            #print(results[model_name][dataset_name])
                        break
        elif model_name == 'pi0_fast':
            if os.path.isdir(os.path.join(results_dir, model_name, 'procgen_results')):
                results[model_name] = {}
                for dataset in os.listdir(os.path.join(results_dir, model_name, 'procgen_results')):
                    filename_parts = dataset.split('_')[-1].split('.')
                    for part in filename_parts:
                        if part in ProcGenDefinitions.DESCRIPTIONS:
                            dataset_name = part
                            with open(os.path.join(results_dir, model_name, 'procgen_results', dataset), 'r') as f:
                                results[model_name][dataset_name] = json.load(f)
                            break
        else:
            if os.path.isdir(os.path.join(results_dir, model_name)):
                results[model_name] = {}
                for dataset in os.listdir(os.path.join(results_dir, model_name, 'procgen_results')):
                    filename_parts = dataset.split('_')
                    for part in filename_parts:
                        if part in ProcGenDefinitions.DESCRIPTIONS:
                            dataset_name = part
                            with open(os.path.join(results_dir, model_name, 'procgen_results', dataset), 'r') as f:
                                results[model_name][dataset_name] = json.load(f)
                            break
                
    return results

def plot_individual_models_macro_f1(results_dir, models):
    """Create individual plots for each model across subdatasets
       CHANGE KEYS/METRIC NAMES BASED ON THE RESULTS FILES
    """
    results = load_results(results_dir, models)
    #print(results.keys())
    #print(results['gpt4o'].keys())

    # Get list of all subdatasets -- common ones
    # subdatasets = list(results['pi0_fast'].keys())
    subdatasets = list(results[models[0]].keys())
    # Set width of bars and positions of the bars
    width = 0.35
    x = np.arange(len(subdatasets))
    
    # Create lists of normalized_brier_mae values for each model
    gpt4o_scores = [results['gpt4o'][dataset][dataset]['macro_f1'] for dataset in subdatasets]
    gpt4_1_scores = [results['gpt4_1'][dataset][dataset]['macro_f1'] for dataset in subdatasets]
    openvla_scores = []
    for dataset in subdatasets:
        if dataset in results['openvla'][dataset]:
            openvla_scores.append(results['openvla'][dataset][dataset]['macro_f1'])
        else:
            openvla_scores.append(results['openvla'][dataset]['macro_f1'])
    #print(results['pi0_base'].keys())
    pi0_base_scores = [results['pi0_base'][dataset][dataset]['macro_f1'] for dataset in subdatasets]

    pi0_fast_scores = []
    for dataset in subdatasets:
        if dataset in results['pi0_fast'].keys():
            pi0_fast_scores.append(results['pi0_fast'][dataset][dataset]['macro_f1'])
        else:
            pi0_fast_scores.append(0)

    # Create the figure and axis
    plt.figure(figsize=(12, 6))
    
    # Create bars
    # Increase spacing between groups by multiplying x by a factor
    x = x * 2  # Double the spacing between groups
    
    plt.bar(x - width/2, gpt4o_scores, width, label='GPT-4o', color=COLORS[0])
    plt.bar(x + width/2, openvla_scores, width, label='OpenVLA', color=COLORS[1])
    plt.bar(x + 3*width/2, pi0_base_scores, width, label='PI0 Base', color=COLORS[2])
    plt.bar(x + 5*width/2, pi0_fast_scores, width, label='PI0 Fast', color=COLORS[3])
    plt.bar(x + 7*width/2, gpt4_1_scores, width, label='GPT-4_1', color=COLORS[4])
    # Customize the plot
    plt.ylabel('Macro F1 score')
    plt.title('Model Performance Comparison Across Subdatasets')
    plt.xticks(x, subdatasets, rotation=45)
    plt.legend()
    plt.ylim(0, 0.15)  # Set y-axis limit to 0.15
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison_macro_f1.png'))
    plt.close()

def get_preds_and_gt_from_dict(model, data, dataset):
    result = {}
    if model in ['gpt4o', 'gpt4_1']:
        predictions = np.array(data[model][dataset][dataset]['preds']).flatten()
        ground_truth = np.array(data[model][dataset][dataset]['gt_actions']).flatten()
    elif model == 'pi0_fast':
        predictions = np.array(data[model][dataset][dataset]['all_preds']).flatten()
        ground_truth = np.array(data[model][dataset][dataset]['all_gt']).flatten()
    else:
        if dataset in data[model][dataset]:
            predictions = np.array(data[model][dataset][dataset]['all_preds']).flatten()
            ground_truth = np.array(data[model][dataset][dataset]['all_gt']).flatten()
        else:
            predictions = np.array(data[model][dataset]['all_preds']).flatten()
            ground_truth = np.array(data[model][dataset]['all_gt']).flatten()

    result["preds"] = predictions
    result["gt_actions"] = ground_truth
    return result

def sample_from_preds_and_gt(data, sample_size):
    """Sample a fixed number of predictions and ground truth data with replacement for bootstrap analysis.
    
    Args:
        data (dict): Dictionary containing 'preds' and 'gt_actions' arrays
        sample_size (int): Number of samples to take
        
    Returns:
        dict: Dictionary containing sampled 'preds' and 'gt_actions' arrays
    """
    # Get the total number of samples
    n_samples = len(data['preds'])
    
    # Generate random indices with replacement
    indices = np.random.choice(n_samples, size=sample_size, replace=True)
    
    # Sample the data
    sampled_data = {
        'preds': np.array(data['preds'])[indices],
        'gt_actions': np.array(data['gt_actions'])[indices]
    }
    
    return sampled_data

def get_macro_micro_metrics_std_per_dataset_from_bootstrap_sample(model, data, dataset, sample_size=20_000, n_iterations=200, metric_type='macro', with_invalids: bool = True):
    """Calculate standard deviation of macro metrics using bootstrap sampling.
    
    Args:
        model (str): Model name
        data (dict): Results data
        dataset (str): Dataset name
        sample_size (int): Number of samples to take in each bootstrap iteration
        n_iterations (int): Number of bootstrap iterations
        
    Returns:
        tuple: (precision_std, recall_std, f1_std)
    """
    try:
        if model in ['gpt4o', 'gpt4_1']:
            _ = data[model][dataset][dataset]['preds']
        else:
            if dataset in data[model][dataset]:
                _ = data[model][dataset][dataset]['all_preds']
            else:
                _ = data[model][dataset]['all_preds']
    except (KeyError, TypeError):
        # If the data doesn't exist, return zeros for all stds
        print(f"{model} doesn't have data for {dataset}, returning all 0s")
        return 0.0, 0.0, 0.0

    # Get the full dataset once, outside the loop
    all_actions = get_preds_and_gt_from_dict(model, data, dataset)
    valid_actions = ProcGenDefinitions.get_valid_action_space(dataset, 'default')
    
    # Initialize lists to store metrics
    sampled_data_precision = []
    sampled_data_recall = []
    sampled_data_f1 = []

    # # Store per-class metrics for analysis
    # class_precisions = {action: [] for action in valid_actions}
    # class_recalls = {action: [] for action in valid_actions}
    # class_f1s = {action: [] for action in valid_actions}

    # Perform bootstrap sampling
    for i in range(n_iterations):
        # Sample with replacement
        sampled_data = sample_from_preds_and_gt(all_actions, sample_size)
        
        # Calculate metrics on the sampled data
        if metric_type == 'macro':
            precisions = get_precision_per_class(sampled_data['preds'], sampled_data['gt_actions'], valid_actions)
            recalls = get_recall_per_class(sampled_data['preds'], sampled_data['gt_actions'], valid_actions)
            f1s = get_f1_per_class(precisions, recalls)
            # Store macro metrics
            sampled_data_precision.append(get_macro_precision(precisions))
            sampled_data_recall.append(get_macro_recall(recalls))
            sampled_data_f1.append(get_macro_f1(f1s))
        elif metric_type == 'micro':
            total_tp, total_fp, total_fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(sampled_data['preds'], sampled_data['gt_actions'], valid_actions)
            if with_invalids:
                precisions = get_micro_precision_from_counts(total_tp, total_fp)
                recalls = get_micro_recall_from_counts(total_tp, total_fn)
                f1s = get_micro_f1(precisions, recalls)
            else:
                precisions = get_micro_precision_from_counts(total_tp, valid_fp)
                recalls = get_micro_recall_from_counts(total_tp, total_fn)
                f1s = get_micro_f1(precisions, recalls)
            sampled_data_precision.append(precisions)
            sampled_data_recall.append(recalls)
            sampled_data_f1.append(f1s)
        else:
            raise ValueError(f"Invalid metric type: {metric_type}")

    #     # Store per-class metrics
    #     for action in valid_actions:
    #         class_precisions[action].append(precisions[action])
    #         class_recalls[action].append(recalls[action])
    #         class_f1s[action].append(f1s[action])

    
    # # Calculate per-class statistics
    # class_stats = {
    #     'precision': {action: {
    #         'mean': np.mean(class_precisions[action]),
    #         'std': np.std(class_precisions[action]),
    #     } for action in valid_actions},
    #     'recall': {action: {
    #         'mean': np.mean(class_recalls[action]),
    #         'std': np.std(class_recalls[action])
    #     } for action in valid_actions},
    #     'f1': {action: {
    #         'mean': np.mean(class_f1s[action]),
    #         'std': np.std(class_f1s[action])
    #     } for action in valid_actions}
    # }
    
    # Calculate standard deviations
    precision_std = np.std(sampled_data_precision)
    recall_std = np.std(sampled_data_recall)
    f1_std = np.std(sampled_data_f1)
    
    # if model == 'openvla':  # DEBUG
    #     print(f"{model} - {dataset}:")
    #     pprint(class_stats)

    return precision_std, recall_std, f1_std

def plot_cross_model_macro_micro_metric(results_dir, models, metric='recall', metric_type='macro', with_invalids: bool = True):
    """Create grouped bar plots comparing model performance across datasets, arranged in 4 vertical subplots"""
    results = load_results(results_dir)

    # Get list of all subdatasets -- common ones
    subdatasets = sorted(list(results[models[0]].keys()))
    
    # Calculate metrics for each model
    model_scores = {}
    model_scores_std = {}
    
    if metric_type == 'macro':
        if metric == 'precision':
            metric_key = 'macro_precision'
            metric_std_index = 0
        elif metric == 'recall':
            metric_key = 'macro_recall'
            metric_std_index = 1
        elif metric == 'f1':
            metric_key = 'macro_f1'
            metric_std_index = 2
        else:
            raise ValueError(f"Invalid metric: {metric}")
    elif metric_type == 'micro':
        if metric == 'precision':
            metric_key = 'micro_precision'
            metric_std_index = 0
        elif metric == 'recall':
            metric_key = 'micro_recall'
            metric_std_index = 1
        elif metric == 'f1':
            metric_key = 'micro_f1'
            metric_std_index = 2
        else:
            raise ValueError(f"Invalid metric: {metric}")

    # Collect scores for each model
    for model in models:
        scores = []
        scores_std = []
        for dataset in subdatasets:
            try:
                std = get_macro_micro_metrics_std_per_dataset_from_bootstrap_sample(
                    model, results, dataset, metric_type=metric_type, with_invalids=with_invalids)[metric_std_index]
                scores_std.append(std)
            except Exception as e:
                print(f"Error in {model}: {e}")
                scores_std.append(0.0)
            
            try:
                if model in ['gpt4o', 'gpt4_1']:
                    if metric_type == 'macro':
                        score = results[model][dataset][dataset][metric_key]
                    elif metric_type == 'micro':
                        if with_invalids:
                            score = results[model][dataset][dataset][metric_key.split('_')[-1]]
                        else:
                            if metric_key == 'micro_recall':
                                score = results[model][dataset][dataset][f'{metric_key.split("_")[-1]}']
                            else:
                                score = results[model][dataset][dataset][f'{metric_key.split("_")[-1]}_without_invalid']
                elif model == 'openvla':
                    if metric_type == 'macro':
                        if dataset in results[model][dataset]:
                            score = results[model][dataset][dataset][metric_key]
                        else:
                            score = results[model][dataset][metric_key]
                    elif metric_type == 'micro':
                        try:
                            score = results[model][dataset][dataset][f'total_{metric_key}']
                        except (KeyError, TypeError):
                            score = results[model][dataset][f'total_{metric_key}']
                elif model == 'pi0_base':
                    if metric_type == 'macro':
                        score = results[model][dataset][dataset][metric_key]
                    elif metric_type == 'micro':
                        if with_invalids:
                            score = results[model][dataset][dataset][f'total_{metric_key}']
                        else:
                            if metric_key == 'micro_recall':
                                score = results[model][dataset][dataset][f'total_{metric_key}']
                            else:
                                score = results[model][dataset][dataset][f'total_{metric_key}_without_invalids']
                elif model == 'pi0_fast':
                    if metric_type == 'macro':
                        if dataset in results[model].keys():
                            score = results[model][dataset][dataset][metric_key]
                        else:
                            score = results[model][dataset][metric_key]
                    elif metric_type == 'micro':
                        if with_invalids:
                            score = results[model][dataset][dataset][f'{metric_key}']
                        else:
                            if metric_key == 'micro_recall':
                                score = results[model][dataset][dataset][f'{metric_key}']
                            else:
                                score = results[model][dataset][dataset][f'{metric_key}_without_invalids']
                scores.append(score)
            except (KeyError, TypeError):
                scores.append(0)
        
        model_scores[model] = scores
        model_scores_std[model] = scores_std

    # Save metrics to JSON
    metrics_file = os.path.join(results_dir, "metrics.json")
    metrics = {model: {} for model in models}
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
    
    for model in models:
        metrics[model][metric_key] = model_scores[model]
        metrics[model][f'{metric_key}_std'] = model_scores_std[model]
    
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Split datasets into groups of 4
    num_datasets = len(subdatasets)
    datasets_per_subplot = 4
    num_subplots = (num_datasets + datasets_per_subplot - 1) // datasets_per_subplot

    # Create figure with subplots and adjust spacing
    fig = plt.figure(figsize=(15, 6*num_subplots))
    # Add space at the top for the title and legend
    gs = plt.GridSpec(num_subplots + 1, 1, height_ratios=[0.5] + [1]*num_subplots, hspace=0.4)
    
    # Create a special axes for the title and legend
    title_ax = fig.add_subplot(gs[0])
    title_ax.axis('off')  # Hide the axes
    
    # Create subplot axes
    axs = [fig.add_subplot(gs[i+1]) for i in range(num_subplots)]

    # Plot settings
    bar_width = 0.15
    opacity = 0.8

    # Create plots for each group of datasets
    for subplot_idx in range(num_subplots):
        ax = axs[subplot_idx]
        
        # Get the datasets for this subplot
        start_idx = subplot_idx * datasets_per_subplot
        end_idx = min(start_idx + datasets_per_subplot, num_datasets)
        current_datasets = subdatasets[start_idx:end_idx]
        
        # Calculate x positions
        x = np.arange(len(current_datasets))
        
        # Calculate maximum y value including error bars for this subplot
        max_y = 0
        for model in models:
            current_scores = model_scores[model][start_idx:end_idx]
            current_stds = model_scores_std[model][start_idx:end_idx]
            max_with_error = max([score + std for score, std in zip(current_scores, current_stds) if score > 0])
            max_y = max(max_y, max_with_error)
        
        # Add padding to y-axis limit (30% extra space)
        y_limit = max_y * 1.3
        
        # Plot bars for each model
        for i, model in enumerate(models):
            # Get scores for current datasets
            current_scores = model_scores[model][start_idx:end_idx]
            current_stds = model_scores_std[model][start_idx:end_idx]
            
            # Plot bars with error bars
            bars = ax.bar(x + i*bar_width, current_scores, bar_width,
                         alpha=opacity, color=COLORS[i], label=model,
                         yerr=current_stds, capsize=5, error_kw={'elinewidth': 2})
            
            # Add value labels on top of bars
            for idx, (value, std) in enumerate(zip(current_scores, current_stds)):
                if value > 0:  # Only show non-zero values
                    # Calculate label position
                    label_height = value + std
                    if label_height > y_limit * 0.85:  # If label would be too close to top
                        label_height = value - std  # Place label below error bar
                        va = 'top'
                    else:
                        va = 'bottom'
                    
                    ax.text(x[idx] + i*bar_width, label_height,
                           f'{value:.2f}', ha='center', va=va,
                           rotation=45, fontsize=16)
        
        # Customize subplot
        ax.set_ylabel(f'{metric_type.capitalize()} {metric.capitalize()}', fontsize=20)
        ax.set_xticks(x + bar_width * (len(models)-1)/2)
        ax.set_xticklabels(current_datasets, rotation=45, ha='right', fontsize=20)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Set y-axis limits
        ax.set_ylim(0, y_limit)

    # Add overall title
    title = f'Model {metric_type.capitalize()} {metric.capitalize()} Comparison'
    if not with_invalids:
        title += ' (Without Invalids)'
    title_ax.text(0.5, 0.7, title, fontsize=24, ha='center', va='bottom')

    # Create a single legend for all subplots
    handles, labels = axs[0].get_legend_handles_labels()
    title_ax.legend(handles, labels, 
                   loc='center',
                   bbox_to_anchor=(0.5, 0.2),
                   ncol=len(models),
                   fontsize=20)

    # Save the plot with high DPI
    if with_invalids:
        plt.savefig(os.path.join(results_dir, f'model_comparison_{metric_type}_{metric_key}_with_invalids.png'),
                    bbox_inches='tight', dpi=300)
    else:
        plt.savefig(os.path.join(results_dir, f'model_comparison_{metric_type}_{metric_key}_without_invalids.png'),
                    bbox_inches='tight', dpi=300)
    plt.close()

def plot_classwise_metrics(results_dir, model_name):
    """Create plots comparing different metrics for a single model across subdatasets
       CHANGE KEYS/METRIC NAMES BASED ON THE RESULTS FILES
    """
    results = load_results(results_dir)
    
    # Get list of all subdatasets for the model
    subdatasets = list(results[model_name].keys())
    print(subdatasets)
    
    # Set width of bars and positions
    width = 0.25
    
    # Get different metrics
    for dataset in subdatasets:
        print(dataset)
        # Get number of classes for this dataset
        if dataset in results[model_name][dataset]:
            # num_classes = len(results[model_name][dataset][dataset]['class_wise_metrics'])
            num_classes = len(ProcGenDefinitions.get_valid_action_space(dataset, 'default'))
            metrics = results[model_name][dataset][dataset]['class_wise_metrics']
        else:
            # num_classes = len(results[model_name][dataset]['class_wise_metrics'])
            num_classes = len(ProcGenDefinitions.get_valid_action_space(dataset, 'default'))
            metrics = results[model_name][dataset]['class_wise_metrics']

        # Create figure for this dataset
        plt.figure(figsize=(12, 6))
        
        # Plot precision for each class
        x = np.arange(num_classes)
        precisions = []
        recalls = []
        f1s = []
        for class_id in range(num_classes):
            if str(class_id) not in metrics:
                print(f"Class {class_id} not found in metrics")
                precisions.append(0)
                recalls.append(0)
                f1s.append(0)
            else:
                precisions.append(metrics[str(class_id)]['precision'])
                recalls.append(metrics[str(class_id)]['recall'])
                f1s.append(metrics[str(class_id)]['f1'])
        #plt.bar(x, precisions)
        plt.bar(x, recalls)
        # plt.bar(x, f1s)
        # Customize plot
        plt.ylabel('Recall')
        plt.xlabel('Class ID')
        plt.title(f'Class-wise Recall for {model_name} on {dataset}')
        plt.xticks(x, [str(i) for i in range(num_classes)])
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{model_name}_{dataset}_classwise_recall.png'))
        plt.close()

def plot_model_metrics(results_dir, model_name):
    """Create plots comparing different metrics for a single model across subdatasets
       CHANGE KEYS/METRIC NAMES BASED ON THE RESULTS FILES
    """
    results = load_results(results_dir)
    
    # Get list of all subdatasets for the model
    subdatasets = sorted(list(results[model_name].keys()))
    print(subdatasets)
    # Set width of bars and positions
    width = 0.25
    x = np.arange(len(subdatasets))
    
    # Get different metrics
    normalized_amae = []
    amae = []
    normalized_quantile_filtered_amae = []
    f1_scores = []
    
    if model_name == 'pi0_fast':
        avg_normalized_brier_mae_key = 'avg_normalized_brier_mae' 
        avg_dataset_brier_mae_key = 'avg_brier_mae'
        avg_quantile_filtered_normalized_brier_mae_key = 'avg_quantile_filtered_normalized_brier_mae'
    elif model_name in ['gpt4o', 'gpt4_1']:
        avg_normalized_brier_mae_key = 'normalized_amae'
        avg_dataset_brier_mae_key = 'avg_dataset_amae'
        avg_quantile_filtered_normalized_brier_mae_key = 'normalized_quantile_filtered_amae'
    elif model_name == 'openvla':
        avg_normalized_brier_mae_key = 'average_normalized_brier_mae'
        avg_dataset_brier_mae_key = 'avg_dataset_brier_mae'
        avg_quantile_filtered_normalized_brier_mae_key = 'average_quantile_filtered_normalized_brier_mae'
    else:
        avg_normalized_brier_mae_key = 'average_normalized_brier_mae'
        avg_dataset_brier_mae_key = 'average_brier_mae'
        avg_quantile_filtered_normalized_brier_mae_key = 'average_quantile_filtered_normalized_brier_mae'

    for dataset in subdatasets:
        if dataset in results[model_name][dataset]:
            normalized_amae.append(results[model_name][dataset][dataset][avg_normalized_brier_mae_key])
            amae.append(results[model_name][dataset][dataset][avg_dataset_brier_mae_key]) 
            normalized_quantile_filtered_amae.append(results[model_name][dataset][dataset][avg_quantile_filtered_normalized_brier_mae_key])
        else:
            normalized_amae.append(results[model_name][dataset][avg_normalized_brier_mae_key])
            amae.append(results[model_name][dataset][avg_dataset_brier_mae_key]) 
            normalized_quantile_filtered_amae.append(results[model_name][dataset][avg_quantile_filtered_normalized_brier_mae_key])
    
    metrics_file = os.path.join(results_dir, "metrics.json")
    
    # Initialize metrics dictionary structure
    metrics = {
        'gpt4o': {},
        'gpt4_1': {},
        'openvla': {},
        'pi0_fast': {}
    }
    
    # Try to load existing metrics if file exists
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
    
    # Update metrics for this model
    metrics[model_name]['normalized_amae'] = normalized_amae
    metrics[model_name]['amae'] = amae
    metrics[model_name]['normalized_quantile_filtered_amae'] = normalized_quantile_filtered_amae
    
    # Write back to file
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Create the figure and axis
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    # Create bars for each metric
    bars1 = ax.bar(x - width, normalized_amae, width, label='Normalized Brier MAE')
    bars2 = ax.bar(x, amae, width, label='Brier MAE')
    bars3 = ax.bar(x + width, normalized_quantile_filtered_amae, width, label='Normalized Quantile Filtered AMAE')

    # Find min and max values for better y-axis limits
    all_values = normalized_amae + amae + normalized_quantile_filtered_amae
    min_val = min(v for v in all_values if v > 0)  # Ignore zeros
    max_val = max(all_values)
    
    # Set y-axis limits to focus on the range where values exist
    # Start from 95% of minimum value to show a bit of context below
    y_min = min_val * 0.95
    y_max = max_val * 1.05  # Add 5% padding above max value
    
    # Create broken axis effect
    # Draw diagonal lines on axes to indicate the break
    d = .015  # Size of diagonal lines
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # Bottom left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Bottom right diagonal
    
    # Customize the plot
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel('Score')
    ax.set_title(f'Metrics Comparison for {model_name} Across Subdatasets')
    ax.set_xticks(x)
    ax.set_xticklabels(subdatasets, rotation=45, ha='right')
    ax.legend()
    
    # Add grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Format y-axis ticks to show more decimal places
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name}_metrics_comparison_normalized_and_brier_mae_and_quantile_filtered_amae.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def calculate_classwise_metrics(results_dir, model_name):
    """Calculate classwise metrics for a single model across subdatasets and save to JSON
    
    Args:
        results_dir (str): Directory containing results
        model_name (str): Name of the model to analyze
    """
    results = load_results(results_dir)
    subdatasets = list(results[model_name].keys())
    
    for dataset in subdatasets:
        print(f"Processing {dataset}...")
        if results[model_name][dataset][dataset].get('class_wise_metrics', None) is not None:
            print(f"Class-wise metrics already calculated for {dataset}")
            continue
        # Get the data path where the original JSON is stored
        if model_name == 'pi0_base':
            # json_path = os.path.join(results_dir, model_name, 'procgen_results', 'corrected_results')
            raise Exception("Not supported for pi0 base")
        elif model_name == 'gpt4o':
            json_path = os.path.join(results_dir, model_name, 'procgen_results', 'corrected_results')
            predictions = np.array(results[model_name][dataset][dataset]['preds'])
            ground_truth = np.array(results[model_name][dataset][dataset]['gt_actions'])
            dataset_path = os.path.join(json_path, f'{dataset}_results.json')
        elif model_name == 'gpt4_1':
            json_path = os.path.join(results_dir, model_name, 'procgen_results', 'gpt4.1_new_prompt')
            predictions = np.array(results[model_name][dataset][dataset]['preds'])
            ground_truth = np.array(results[model_name][dataset][dataset]['gt_actions'])
            dataset_path = os.path.join(json_path, f'4.1_{dataset}_results.json')
        elif model_name == 'pi0_fast':
            json_path = os.path.join(results_dir, model_name, 'procgen_results')
            predictions = np.array(results[model_name][dataset][dataset]['all_preds'])
            ground_truth = np.array(results[model_name][dataset][dataset]['all_gt'])
            dataset_path = os.path.join(json_path, f'{model_name}_procgen_results_{dataset}.json')
        else:
            json_path = os.path.join(results_dir, model_name, 'procgen_results')
            
        # Get predictions and ground truth
        try:
            valid_actions = ProcGenDefinitions.get_valid_action_space(dataset, 'default')
            
            # Calculate class-wise metrics
            class_precisions = get_precision_per_class(predictions, ground_truth, valid_actions)
            class_recalls = get_recall_per_class(predictions, ground_truth, valid_actions)
            class_f1s = get_f1_per_class(class_precisions, class_recalls)
            
            # Calculate macro metrics
            macro_precision = get_macro_precision(class_precisions)
            macro_recall = get_macro_recall(class_recalls)
            macro_f1 = get_macro_f1(class_f1s)
            
            # Create metrics dictionary
            class_wise_metrics = {}
            for action in valid_actions:
                class_wise_metrics[str(action)] = {
                    'precision': class_precisions[action],
                    'recall': class_recalls[action],
                    'f1': class_f1s[action]
                }
                
            # Add to the JSON data
            results[model_name][dataset][dataset]['class_wise_metrics'] = class_wise_metrics
            results[model_name][dataset][dataset]['macro_precision'] = macro_precision
            results[model_name][dataset][dataset]['macro_recall'] = macro_recall
            results[model_name][dataset][dataset]['macro_f1'] = macro_f1
            
            # Save back to JSON
            with open(dataset_path, 'w') as f:
                json.dump({dataset: results[model_name][dataset][dataset]}, f, indent=4)
                
            print(f"Saved class-wise metrics for {dataset}")
            
        except KeyError as e:
            print(f"Missing required data for {dataset}: {e}")
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            
    print("Completed class-wise metrics calculation")


def plot_cross_model_classwise_comparison(results_dir: str, models: list[str], with_invalids: bool = True):
    """Create comparative plots of class-wise metrics across all models."""
    results = load_results(results_dir)
    
    # Get common datasets across all models
    common_datasets = set(results[models[0]].keys())
    for model in models[1:]:
        common_datasets.intersection_update(results[model].keys())
    
    # Plot settings
    bar_width = 0.15  # Width of each bar
    
    # Initialize dictionaries to store metrics per action class across all datasets
    action_class_metrics = {}  # Will store metrics for each action class across all datasets
    
    # First pass: collect all possible action classes and their metrics
    for dataset in common_datasets:
        valid_actions = ProcGenDefinitions.get_valid_action_space(dataset, 'default')
        for action in valid_actions:
            if action not in action_class_metrics:
                action_class_metrics[action] = {model: {'precision': [], 'recall': [], 'f1': []} for model in models}
        
        # Collect metrics for each model
        for model in models:
            try:
                if dataset in results[model][dataset]:
                    metrics = results[model][dataset][dataset]['class_wise_metrics']
                else:
                    metrics = results[model][dataset]['class_wise_metrics']
                
                for action in valid_actions:
                    if str(action) in metrics:
                        action_class_metrics[action][model]['precision'].append(metrics[str(action)]['precision'])
                        action_class_metrics[action][model]['recall'].append(metrics[str(action)]['recall'])
                        action_class_metrics[action][model]['f1'].append(metrics[str(action)]['f1'])
                    else:
                        action_class_metrics[action][model]['precision'].append(0)
                        action_class_metrics[action][model]['recall'].append(0)
                        action_class_metrics[action][model]['f1'].append(0)
                        
            except KeyError as e:
                print(f"Missing data for {model} on {dataset}: {e}")
                continue
    
    # Create plots for each metric type
    for metric_name in ['Precision', 'Recall', 'F1 Score']:
        metric_key = metric_name.lower().replace(' ', '')
        if metric_key == 'f1score':
            metric_key = 'f1'
            
        plt.figure(figsize=(15, 8))
        x = np.arange(len(action_class_metrics))
        
        # Plot bars for each model
        for i, model in enumerate(models):
            averages = []
            stds = []
            
            for action in sorted(action_class_metrics.keys()):
                values = action_class_metrics[action][model][metric_key]
                averages.append(np.mean(values))
                stds.append(np.std(values))
            
            plt.bar(x + i*bar_width, averages, bar_width,
                   label=model, color=COLORS[i], alpha=0.7,
                   yerr=stds, capsize=5)
        
        plt.ylabel(f'Average {metric_name}')
        plt.xlabel('Action Class ID')
        plt.title(f'Average {metric_name} per Action Class Across All Datasets')
        plt.xticks(x + bar_width * (len(models)-1)/2, sorted(action_class_metrics.keys()))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.15)  # Add padding for labels
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'action_class_comparison_{metric_key}.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

def plot_action_difficulty_heatmap(results_dir: str, models: list[str], metric: str = 'f1'):
    """Create heatmap showing action difficulty across models and datasets.
    
    Args:
        results_dir (str): Directory containing results
        models (list[str]): List of model names to compare
        metric (str): Which metric to plot ('f1', 'precision', or 'recall')
    """
    results = load_results(results_dir)
    common_datasets = set(results[models[0]].keys())
    for model in models[1:]:
        common_datasets.intersection_update(results[model].keys())
    
    # Create data structure for heatmap
    for dataset in common_datasets:
        valid_actions = ProcGenDefinitions.get_valid_action_space(dataset, 'default')
        plt.figure(figsize=(12, 8))
        
        # Create matrix for heatmap
        heatmap_data = np.zeros((len(models), len(valid_actions)))
        
        # Fill matrix with metric values
        for i, model in enumerate(models):
            try:
                if dataset in results[model][dataset]:
                    metrics = results[model][dataset][dataset]['class_wise_metrics']
                else:
                    metrics = results[model][dataset]['class_wise_metrics']
                
                for j, action in enumerate(valid_actions):
                    if str(action) in metrics:
                        heatmap_data[i, j] = metrics[str(action)][metric]
            except KeyError as e:
                print(f"Missing data for {model} on {dataset}: {e}")
                continue
        
        # Create heatmap
        plt.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(label=f'{metric.capitalize()} Score')
        
        # Customize plot
        plt.title(f'Action {metric.capitalize()} Scores Across Models for {dataset}')
        plt.xlabel('Action Class ID')
        plt.ylabel('Model')
        plt.yticks(range(len(models)), models)
        plt.xticks(range(len(valid_actions)), [str(i) for i in valid_actions])
        
        # Add value annotations
        for i in range(len(models)):
            for j in range(len(valid_actions)):
                plt.text(j, i, f'{heatmap_data[i,j]:.2f}', 
                        ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{dataset}_{metric}_heatmap.png'))
        plt.close()

def plot_category_performance(results_dir: str, models: list[str]):
    """Generate plots comparing model performance across different dataset categories.
    
    Args:
        results_dir (str): Directory containing results
        models (list[str]): List of model names to compare
    """
    results = load_results(results_dir)
    
    # Define categorizations
    action_space_categories = {
        'Basic Movement': ['maze', 'climber', 'bigfish', 'leaper', 'miner'],
        'Movement + Special': ['bossfight', 'caveflyer', 'dodgeball', 'fruitbot', 'plunder'],
        'Complex Actions': ['ninja', 'starpilot']
    }
    
    game_mechanic_categories = {
        # 'Pure Navigation': ['maze', 'leaper', 'climber'],
        # 'Combat': ['bossfight', 'starpilot', 'plunder'],
        # 'Collection': ['chaser', 'fruitbot', 'heist', 'miner'],
        # 'Platforming': ['ninja', 'climber', 'coinrun', 'jumper']

        'Platformers': ['climber', 'coinrun', 'jumper', 'ninja'],
        'Maze/Navigational': ['caveflyer', 'chaser', 'maze', 'heist'],
        'Shooter/Combat': ['bossfight', 'starpilot', 'plunder'],
        'Collecting/Avoidance': ['bigfish', 'fruitbot', 'miner', 'leaper', 'dodgeball']
    }

    primary_objective_categories = {
        'Item Collection': ['fruitbot', 'miner', 'heist', 'maze'],  # Primary goal is collecting specific items
        'Target Reaching': ['climber', 'coinrun', 'jumper', 'ninja', 'leaper'],  # Reaching a specific point/item
        'Combat/Elimination': ['bossfight', 'plunder', 'starpilot', 'dodgeball'],  # Defeating enemies is primary
        'Survival/Growth': ['bigfish', 'chaser', 'caveflyer']  # Staying alive/growing stronger is key
    }
    
    reward_structure_categories = {
        'Single Terminal Goal': ['maze', 'ninja', 'jumper', 'heist', 'coinrun'],  # One main objective at end
        'Progressive Rewards': ['bigfish', 'chaser', 'climber', 'fruitbot', 'miner'],  # Multiple collectibles throughout
        'Combat-based Rewards': ['bossfight', 'starpilot', 'plunder', 'dodgeball'],  # Rewards from defeating enemies
        'Checkpoint-based': ['caveflyer', 'leaper']  # Progress through distinct stages
    }

    scene_layout_categories = {
        'Open/Scrolling Worlds': ['fruitbot', 'starpilot', 'jumper', 'caveflyer'],
        'Fixed/Constrained Arenas': ['bossfight', 'plunder', 'dodgeball', 'maze', 'heist']
    }

    task_type_categories = {
        'Object-centric (focus on items/characters)': ['bigfish', 'chaser', 'climber', 'coinrun', 'miner', 'ninja', 'fruitbot'],
        'Scene-centric (focus on layout/navigation)': ['caveflyer', 'maze', 'heist', 'jumper', 'starpilot', 'leaper', 'plunder', 'bossfight', 'dodgeball']
    }

    environment_interaction_categories = {
        'Static Navigation': ['maze', 'heist', 'coinrun'],  # Fixed environment to navigate
        'Dynamic Obstacles': ['caveflyer', 'leaper', 'climber', 'ninja'],  # Moving hazards/platforms
        'Resource Management': ['fruitbot', 'miner', 'chaser'],  # Collecting/managing resources
        'Enemy Interaction': ['bossfight', 'starpilot', 'plunder', 'bigfish', 'dodgeball']  # Primary interaction with enemies
    }

    state_action_categories = {
        'Direct Mapping': ['maze', 'climber', 'leaper'],
        'Strategic Planning': ['bossfight', 'starpilot', 'heist'],
        'Dynamic Response': ['ninja', 'caveflyer', 'dodgeball']
    }

    # movable_background_categories = {
    #     'Non-movable background':
    #         ['bigfish', 'miner', 'maze', 'bossfight', 'chaser', 'starpilot', 'heist', 'dodgeball', 'leaper'],
    #     'Movable background':
    #         ['plunder', 'ninja', 'jumper', 'caveflyer', 'climber', 'coinrun', 'fruitbot']
    # }
    
    # Define metrics to analyze
    metrics = {
        'macro_f1': 'Macro F1 Score',
        'macro_recall': 'Macro Recall',
        'macro_precision': 'Macro Precision',
    }
    
    # Plot settings
    bar_width = 0.15
    
    # Create plots for each categorization scheme
    categorizations = {
        'Action Space Complexity': action_space_categories,
        'Game Mechanics': game_mechanic_categories,
        'Primary Objective': primary_objective_categories,
        'Reward Structure': reward_structure_categories,
        'Scene Layout': scene_layout_categories,
        'Task Type': task_type_categories,
        'Environment Interaction': environment_interaction_categories,
        'State-Action Complexity': state_action_categories,
        # 'Movable Background': movable_background_categories
    }
    
    for cat_name, categories in categorizations.items():
        for metric_key, metric_name in metrics.items():
            plt.figure(figsize=(15, 8))
            
            # Calculate positions for bars
            x = np.arange(len(categories))
            
            # Plot bars for each model
            for i, model in enumerate(models):
                category_means = []
                category_stds = []
                
                for category, datasets in categories.items():
                    # Get metric values for all datasets in this category
                    category_values = []
                    for dataset in datasets:
                        try:
                            # Handle different result structures
                            if dataset in results[model] and dataset in results[model][dataset]:
                                value = results[model][dataset][dataset].get(metric_key, 0)
                            else:
                                value = results[model][dataset].get(metric_key, 0)
                            category_values.append(value)
                        except (KeyError, TypeError):
                            continue
                    
                    # Calculate mean and std if we have values
                    if category_values:
                        category_means.append(np.mean(category_values))
                        category_stds.append(np.std(category_values))
                    else:
                        category_means.append(0)
                        category_stds.append(0)
                
                # Plot bars with error bars
                plt.bar(x + i*bar_width, category_means, bar_width,
                       label=model, color=COLORS[i], alpha=0.7,
                       yerr=category_stds, capsize=5)
            
            # Customize plot
            plt.xlabel('Category')
            plt.ylabel(metric_name)
            plt.title(f'{cat_name}: {metric_name} by Category')
            plt.xticks(x + bar_width * (len(models)-1)/2, categories.keys(), rotation=45)
            plt.legend()
            
            # Add value labels on top of bars
            for i, model in enumerate(models):
                for j, value in enumerate(category_means):
                    plt.text(j + i*bar_width, value, f'{value:.2f}',
                           ha='center', va='bottom', rotation=45, fontsize=8)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(results_dir, 
                       f'category_performance_{cat_name.lower().replace(" ", "_")}_{metric_key}.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()

    # Create radar plots for overall category performance
    for metric_key, metric_name in metrics.items():
        plt.figure(figsize=(10, 10))
        
        # Set up the radar plot
        categories_list = list(categorizations.keys())
        angles = np.linspace(0, 2*np.pi, len(categories_list), endpoint=False)
        
        # Close the plot by appending the first value
        angles = np.concatenate((angles, [angles[0]]))
        
        # Plot for each model
        ax = plt.subplot(111, polar=True)
        for i, model in enumerate(models):
            values = []
            for cat_name, categories in categorizations.items():
                # Calculate average performance across all categories
                all_values = []
                for datasets in categories.values():
                    for dataset in datasets:
                        try:
                            if dataset in results[model] and dataset in results[model][dataset]:
                                value = results[model][dataset][dataset].get(metric_key, 0)
                            else:
                                value = results[model][dataset].get(metric_key, 0)
                            all_values.append(value)
                        except (KeyError, TypeError):
                            continue
                values.append(np.mean(all_values) if all_values else 0)
            
            # Close the plot by appending the first value
            values = np.concatenate((values, [values[0]]))
            
            # Plot the values
            ax.plot(angles, values, 'o-', label=model, color=COLORS[i], alpha=0.7)
            ax.fill(angles, values, color=COLORS[i], alpha=0.1)
        
        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories_list)
        
        plt.title(f'Overall Category Performance: {metric_name}')
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.savefig(os.path.join(results_dir,
                   f'category_performance_radar_{metric_key}.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()


def calculate_confusion_matrices_and_mcc(results_dir: str, models: list[str]):
    """Calculate confusion matrices and Matthews Correlation Coefficient for each model and dataset.
    
    Args:
        results_dir (str): Directory containing results
        models (list[str]): List of model names to compare
    """
    results = load_results(results_dir)
    common_datasets = set(results[models[0]].keys())
    for model in models[1:]:
        common_datasets.intersection_update(results[model].keys())
    
    # Dictionary to store results
    confusion_matrices = {model: {} for model in models}
    mcc_scores = {model: {} for model in models}
    union_confusion_matrices = {model: None for model in models}
    confusion_matrix_percentages = {model: {} for model in models} # New dictionary for percentages
    
    # Dictionary to store action class counts
    action_counts = {model: {} for model in models}
    
    # Process each model and dataset
    for model in models:
        all_preds = []  # Store all predictions for union confusion matrix
        all_gt = []     # Store all ground truth for union confusion matrix
        confusion_matrix_percentages[model] = {} # Initialize for current model
        
        for dataset in common_datasets:
            try:
                # Get predictions and ground truth based on model type
                if model in ['gpt4o', 'gpt4_1']:
                    predictions = np.array(results[model][dataset][dataset]['preds']).flatten()
                    ground_truth = np.array(results[model][dataset][dataset]['gt_actions']).flatten()
                elif model == 'pi0_fast':
                    predictions = np.array(results[model][dataset][dataset]['all_preds']).flatten()
                    ground_truth = np.array(results[model][dataset][dataset]['all_gt']).flatten()
                else:
                    if dataset in results[model][dataset]:
                        predictions = np.array(results[model][dataset][dataset]['all_preds']).flatten()
                        ground_truth = np.array(results[model][dataset][dataset]['all_gt']).flatten()
                    else:
                        predictions = np.array(results[model][dataset]['all_preds']).flatten()
                        ground_truth = np.array(results[model][dataset]['all_gt']).flatten()
                
                # Get valid actions for this dataset
                valid_actions = sorted(ProcGenDefinitions.get_valid_action_space(dataset, "default"))
                num_classes = len(valid_actions)
                
                # Count occurrences of each action class
                pred_counts = {}
                gt_counts = {}
                for action_idx, action in enumerate(valid_actions): # Use enumerate to get index for labels
                    pred_counts[action] = int(np.sum(predictions == action))
                    gt_counts[action] = int(np.sum(ground_truth == action))
                
                # Store counts in the action_counts dictionary
                action_counts[model][dataset] = {
                    "prediction": pred_counts,
                    "ground_truth": gt_counts
                }

                # Calculate confusion matrix
                cm = confusion_matrix(ground_truth, predictions, labels=valid_actions) # Use valid_actions as labels
                confusion_matrices[model][dataset] = cm
                
                # Calculate MCC
                mcc = matthews_corrcoef(ground_truth, predictions)
                mcc_scores[model][dataset] = mcc
                
                # Accumulate predictions and ground truth for union confusion matrix
                all_preds.extend(predictions)
                all_gt.extend(ground_truth)

                # Normalize confusion matrix row-wise for percentages
                row_sums = cm.sum(axis=1)[:, np.newaxis]
                # Handle zero-sum rows to avoid division by zero
                row_sums[row_sums == 0] = 1 
                cm_normalized_for_percentages = cm.astype('float') / row_sums
                
                # Store percentages in the new dictionary
                confusion_matrix_percentages[model][dataset] = {}
                for i, true_action in enumerate(valid_actions):
                    confusion_matrix_percentages[model][dataset][str(true_action)] = {}
                    for j, pred_action in enumerate(valid_actions):
                        confusion_matrix_percentages[model][dataset][str(true_action)][str(pred_action)] = cm_normalized_for_percentages[i, j]

                # Plot individual confusion matrix
                plt.figure(figsize=(10, 8))
                # Row normalize the confusion matrix for plotting (can reuse cm_normalized_for_percentages)
                sns.heatmap(cm_normalized_for_percentages, annot=True, fmt='.2f', cmap='YlOrRd',
                           xticklabels=[str(a) for a in valid_actions], # Use stringified valid_actions for labels
                           yticklabels=[str(a) for a in valid_actions])
                plt.title(f'Confusion Matrix for {model} on {dataset}\nMCC: {mcc:.3f}')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'confusion_matrix_{model}_{dataset}.png'))
                plt.close()
                
            except Exception as e:
                print(f"Error processing {model} on {dataset}: {e}")
                continue
        
        # Calculate union confusion matrix for this model
        try:
            # Get maximum number of classes across all datasets to define labels for union matrix
            # This ensures consistent sizing for the union matrix across all models
            all_possible_actions_across_datasets = set()
            for ds_name in common_datasets:
                 all_possible_actions_across_datasets.update(ProcGenDefinitions.get_valid_action_space(ds_name, 'default'))
            union_labels = sorted(list(all_possible_actions_across_datasets))

            union_cm = confusion_matrix(all_gt, all_preds, labels=union_labels)
            union_confusion_matrices[model] = union_cm
            
            # Calculate union MCC
            union_mcc = matthews_corrcoef(all_gt, all_preds)
            
            # Plot union confusion matrix
            plt.figure(figsize=(12, 10))
            # Row normalize the union confusion matrix
            row_sums = union_cm.sum(axis=1)[:, np.newaxis]
            # Handle zero-sum rows to avoid division by zero
            row_sums[row_sums == 0] = 1 
            union_cm_normalized = union_cm.astype('float') / row_sums
            sns.heatmap(union_cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                       xticklabels=[str(a) for a in union_labels], # Use stringified union_labels
                       yticklabels=[str(a) for a in union_labels])
            plt.title(f'Union Confusion Matrix for {model}\nMCC: {union_mcc:.3f}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'union_confusion_matrix_{model}.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error calculating union confusion matrix for {model}: {e}")
    
    # Save MCC scores to a text file
    with open(os.path.join(results_dir, 'mcc_scores.txt'), 'w') as f:
        f.write('Model | Dataset | MCC Score\n')
        f.write('-' * 50 + '\n')
        for model in models:
            for dataset in common_datasets:
                if dataset in mcc_scores[model]:
                    f.write(f'{model} | {dataset} | {mcc_scores[model][dataset]:.3f}\n')
    
    # Save action class counts to a JSON file
    with open(os.path.join(results_dir, 'action_class_counts.json'), 'w') as f:
        json.dump(action_counts, f, indent=2)

    # Save confusion matrix percentages to a JSON file
    with open(os.path.join(results_dir, 'confusion_matrix_percentages.json'), 'w') as f:
        json.dump(confusion_matrix_percentages, f, indent=2)

def plot_dataset_specific_metrics(results_dir):
    """Create plots showing precision, recall, and F1 scores for each dataset separately,
    with three subplots per dataset and action classes as columns.
    """
    results = load_results(results_dir)
    
    # Get all available models and datasets
    models = list(results.keys())
    datasets = list(results['gpt4o'].keys())  # Using gpt4o as reference for available datasets
    
    # For each dataset, create three subplots
    for dataset in datasets:
        # Get valid actions for this dataset
        valid_actions = sorted(ProcGenDefinitions.get_valid_action_space(dataset, 'default'))
        
        # Collect metrics for each model and action class
        action_metrics = {action: {
            'precision': [],
            'recall': [],
            'f1': [],
            'model_names': []
        } for action in valid_actions}
        
        # Collect metrics for each model
        for model in models:
            try:
                if dataset in results[model][dataset]:
                    metrics = results[model][dataset][dataset]['class_wise_metrics']
                else:
                    metrics = results[model][dataset]['class_wise_metrics']
                
                # For each action class, collect its metrics
                for action in valid_actions:
                    if str(action) in metrics:
                        action_metrics[action]['precision'].append(metrics[str(action)]['precision'])
                        action_metrics[action]['recall'].append(metrics[str(action)]['recall'])
                        action_metrics[action]['f1'].append(metrics[str(action)]['f1'])
                        action_metrics[action]['model_names'].append(model)
                    else:
                        action_metrics[action]['precision'].append(0)
                        action_metrics[action]['recall'].append(0)
                        action_metrics[action]['f1'].append(0)
                        action_metrics[action]['model_names'].append(model)
                        
            except KeyError as e:
                print(f"Missing data for {model} on {dataset}: {e}")
                continue
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(max(12, len(valid_actions) * 2.5), 18))
        
        # Add the main title
        fig.suptitle(f'Model Performance Metrics for {dataset}', fontsize=12, y=0.95)
        
        # Calculate bar positions
        num_models = len(models)
        bar_width = 0.15
        group_width = bar_width * num_models + 0.3  # Add padding between groups
        
        # Plot settings for each metric
        metric_settings = [
            ('precision', 'Precision', ax1, '#1f77b4'),
            ('recall', 'Recall', ax2, '#2ca02c'),
            ('f1', 'F1 Score', ax3, '#ff7f0e')
        ]
        
        # Create plots for each metric
        for metric_name, metric_label, ax, color in metric_settings:
            for model_idx, model in enumerate(models):
                values = []
                x_positions = []
                
                for action_idx, action in enumerate(valid_actions):
                    if model_idx < len(action_metrics[action]['model_names']):
                        values.append(action_metrics[action][metric_name][model_idx])
                        x_pos = action_idx * group_width + model_idx * bar_width
                        x_positions.append(x_pos)
                
                # Plot bars for this model
                bars = ax.bar(x_positions, values, bar_width,
                            label=model, color=COLORS[model_idx], alpha=0.7)
                
                # Add value labels on top of bars
                for i, value in enumerate(values):
                    ax.text(x_positions[i], value, f'{value:.2f}',
                           ha='center', va='bottom', rotation=0, fontsize=8)
            
            # Customize subplot
            ax.set_ylabel(metric_label, fontsize=12)
            ax.set_title(f'{metric_label}', fontsize=12, pad=10)
            
            # Set x-ticks at the center of each group
            group_centers = [i * group_width + (bar_width * (num_models - 1)) / 2 for i in range(len(valid_actions))]
            ax.set_xticks(group_centers)
            ax.set_xticklabels([f'Action {action}' for action in valid_actions], fontsize=10)
            
            # Add legend
            ax.legend()
            
            # Set y-axis limit with padding for labels
            ax.set_ylim(0, 1.15)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Increase tick label size
            ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the top margin to make room for the suptitle
        plt.savefig(os.path.join(results_dir, f'dataset_metrics_by_action_{dataset}.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()

def print_preds_gt_unique_value_counts(results_dir, datasets=None, models=None):
    results = load_results(results_dir)

    if datasets == None:
        datasets = results['gpt4o'].keys()

    if models == None:
        models = list(results.keys())

    # Analyze action value distributions for each dataset
    for dataset in datasets:
        print(f"\n=== Action Value Distribution for {dataset} ===")
        
        # Create a DataFrame for this dataset
        df = pd.DataFrame()
        
        # Get ground truth and predictions for each model
        for model in models:
            try:
                if model in ['pi0_base', 'pi0_fast']:
                    predictions = np.array(results[model][dataset][dataset]['all_preds']).flatten()
                    ground_truth = np.array(results[model][dataset][dataset]['all_gt']).flatten()
                elif model in ['gpt4o', 'gpt4_1']:
                    predictions = np.array(results[model][dataset][dataset]['preds']).flatten()
                    ground_truth = np.array(results[model][dataset][dataset]['gt_actions']).flatten()
                elif model == 'openvla':
                    try:
                        predictions = np.array(results[model][dataset]['all_preds']).flatten()
                        ground_truth = np.array(results[model][dataset]['all_gt']).flatten()
                    except KeyError:
                        predictions = np.array(results[model][dataset][dataset]['all_preds']).flatten()
                        ground_truth = np.array(results[model][dataset][dataset]['all_gt']).flatten()
                # Add to DataFrame
                df[f'{model}_gt'] = ground_truth
                df[f'{model}_preds'] = predictions
                
            except KeyError as e:
                print(f"Missing data for {model} on {dataset}: {e}")
                continue
        
        # Print unique values and their counts for each column
        for col in df.columns:
            value_counts = df[col].value_counts().sort_index()
            print(value_counts)
            print(f"Total unique values: {len(value_counts)}")
            print(f"Most common value: {value_counts.idxmax()} (count: {value_counts.max()})")
            print(f"Least common value: {value_counts.idxmin()} (count: {value_counts.min()})")

    """Create a plot comparing normalized Brier MAEs across different models.
    
    Args:
        results_dir (str): Directory containing results
        models (list[str]): List of model names to compare
    """
    results = load_results(results_dir)
    
    # Get common datasets across all models
    common_datasets = set(results[models[0]].keys())
    for model in models[1:]:
        common_datasets.intersection_update(results[model].keys())
    common_datasets = sorted(list(common_datasets))
    
    # Set width of bars and positions
    width = 0.15
    x = np.arange(len(common_datasets))
    
    # Create the figure and axis
    plt.figure(figsize=(15, 8))
    
    # Plot bars for each model
    for i, model in enumerate(models):
        # Get the correct key for normalized Brier MAE based on model
        if model == 'pi0_fast':
            metric_key = 'avg_normalized_brier_mae'
        elif model in ['gpt4o', 'gpt4_1']:
            metric_key = 'normalized_amae'
        else:
            metric_key = 'average_normalized_brier_mae'
        
        # Collect metrics for this model
        model_metrics = []
        for dataset in common_datasets:
            try:
                if dataset in results[model][dataset]:
                    model_metrics.append(results[model][dataset][dataset][metric_key])
                else:
                    model_metrics.append(results[model][dataset][metric_key])
            except KeyError:
                model_metrics.append(0)  # Use 0 if metric is missing
        
        # Plot bars for this model
        plt.bar(x + i*width, model_metrics, width, label=model, color=COLORS[i])
        
        # Add value labels on top of bars
        for j, value in enumerate(model_metrics):
            plt.text(x[j] + i*width, value, f'{value:.3f}',
                    ha='center', va='bottom', rotation=45, fontsize=8)
    
    # Customize the plot
    plt.ylabel('Normalized Brier MAE')
    plt.title('Normalized Brier MAE Comparison Across Models')
    plt.xticks(x + width * (len(models)-1)/2, common_datasets, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'normalized_brier_mae_comparison.png'))
    plt.close()

def plot_cross_model_brier_mae(results_dir):
    """Create a plot comparing Brier MAE across GPT-4o, GPT-4.1, OpenVLA, and PI0 Fast.
    Split into 4 vertically stacked subplots with 4 datasets each.
    
    Args:
        results_dir (str): Directory containing results
    """
    # Load results for all models
    models = ['gpt4o', 'gpt4_1', 'openvla', 'pi0_fast']
    results = load_results(results_dir, models)
    
    # Get list of all subdatasets -- common ones
    subdatasets = sorted(list(results[models[0]].keys()))
    
    # Set width of bars and positions
    width = 0.2  # Slightly wider bars since we have 4 models
    
    # Dictionary to map models to their Brier MAE keys
    brier_mae_keys = {
        'gpt4o': 'avg_dataset_amae',
        'gpt4_1': 'avg_dataset_amae',
        'openvla': 'avg_dataset_brier_mae',
        'pi0_fast': 'avg_brier_mae'
    }
    
    # Split datasets into groups of 4
    datasets_per_subplot = 4
    num_subplots = (len(subdatasets) + datasets_per_subplot - 1) // datasets_per_subplot

    # Create figure with subplots and adjust spacing
    fig = plt.figure(figsize=(15, 6*num_subplots))
    # Add space at the top for the title and legend
    gs = plt.GridSpec(num_subplots + 1, 1, height_ratios=[0.5] + [1]*num_subplots, hspace=0.4)
    
    # Create a special axes for the title and legend
    title_ax = fig.add_subplot(gs[0])
    title_ax.axis('off')  # Hide the axes
    
    # Create subplot axes
    axs = [fig.add_subplot(gs[i+1]) for i in range(num_subplots)]
    
    # Plot for each subplot (group of datasets)
    for subplot_idx in range(num_subplots):
        ax = axs[subplot_idx]
        
        # Get the datasets for this subplot
        start_idx = subplot_idx * datasets_per_subplot
        end_idx = min(start_idx + datasets_per_subplot, len(subdatasets))
        current_datasets = subdatasets[start_idx:end_idx]
        
        # Calculate x positions
        x = np.arange(len(current_datasets))
        
        # Plot bars for each model and track min/max values
        all_scores = []
        for i, model in enumerate(models):
            model_scores = []
            
            for dataset in current_datasets:
                try:
                    # Handle different data structures for different models
                    if model in ['gpt4o', 'gpt4_1']:
                        score = results[model][dataset][dataset][brier_mae_keys[model]]
                    elif model == 'pi0_fast':
                        if dataset in results[model].keys():
                            score = results[model][dataset][dataset][brier_mae_keys[model]]
                        else:
                            score = 0  # or some other default value
                    else:  # openvla
                        if dataset in results[model][dataset]:
                            score = results[model][dataset][dataset][brier_mae_keys[model]]
                        else:
                            score = results[model][dataset][brier_mae_keys[model]]
                    model_scores.append(score)
                    all_scores.append(score)
                except (KeyError, TypeError):
                    model_scores.append(0)
                    print(f"Missing data for {model} on {dataset}")
            
            # Plot bars with error bars
            bars = ax.bar(x + i*width, model_scores, width, 
                         label=model, color=COLORS[i], alpha=0.8,
                         edgecolor='black', linewidth=1)
            
            # Add value labels on top of bars
            for idx, value in enumerate(model_scores):
                if value > 0:  # Only show non-zero values
                    ax.text(x[idx] + i*width, value, f'{value:.2f}',
                           ha='center', va='bottom', rotation=45,
                           fontsize=20)
        
        # Calculate y-axis limits for this subplot
        non_zero_scores = [s for s in all_scores if s > 0]
        if non_zero_scores:
            min_val = min(non_zero_scores)
            max_val = max(non_zero_scores)
            # Start y-axis from 20% below the minimum non-zero value
            y_min = max(0, min_val - (max_val - min_val) * 0.2)
            # Add 30% padding above maximum value
            y_max = max_val + (max_val - min_val) * 0.3
            
            # Set y-axis limits
            ax.set_ylim(y_min, y_max)
            
            # Add broken axis indicator if not starting from 0
            if y_min > 0:
                d = .015  # Size of diagonal lines
                kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
                ax.plot((-d, +d), (-d, +d), **kwargs)
                ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        
        # Customize subplot
        ax.set_ylabel('Brier MAE', fontsize=20)
        ax.set_xticks(x + width * (len(models)-1)/2)
        ax.set_xticklabels(current_datasets, rotation=45, ha='right', fontsize=20)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # Add light vertical lines to separate datasets
        for i in x:
            ax.axvline(i - width/2, color='gray', linestyle='-', alpha=0.1)
    
    # Add overall title
    title_ax.text(0.5, 0.7, 'Model Brier MAE Comparison Across Subdatasets',
                 fontsize=24, ha='center', va='bottom')

    # Create a single legend for all subplots
    handles, labels = axs[0].get_legend_handles_labels()
    title_ax.legend(handles, labels, 
                   loc='center',
                   bbox_to_anchor=(0.5, 0.2),
                   ncol=len(models),
                   fontsize=20)
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, 'model_comparison_brier_mae.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_action_distributions(results_dir: str, models: list[str]):
    """Create plots comparing prediction and ground truth action distributions for each model,
    aggregated across all datasets.
    
    Args:
        results_dir (str): Directory containing results
        models (list[str]): List of model names to compare
    """
    from scipy.interpolate import make_interp_spline
    import numpy as np

    # Define colors - using more distinct colors
    GT_COLOR = '#0072B2'  # Bright green for ground truth
    PRED_COLOR = '#E69F00'  # Bright red for predictions
    OVERLAP_COLOR = '#999999'  # Gray for overlap
    GT_CURVE_COLOR = '#005C85'  # Darker green for ground truth curve
    PRED_CURVE_COLOR = '#B35C00'  # Darker red for prediction curve

    results = load_results(results_dir)
    common_datasets = set(results[models[0]].keys())
    for model in models[1:]:
        common_datasets.intersection_update(results[model].keys())
    
    # Process each model
    for model in models:
        # Initialize aggregated counts
        all_pred_counts = {}
        all_gt_counts = {}
        all_possible_actions = set()
        
        # Collect data across all datasets
        for dataset in common_datasets:
            try:
                # Get predictions and ground truth based on model type
                if model in ['gpt4o', 'gpt4_1']:
                    predictions = np.array(results[model][dataset][dataset]['preds']).flatten()
                    ground_truth = np.array(results[model][dataset][dataset]['gt_actions']).flatten()
                elif model == 'pi0_fast':
                    predictions = np.array(results[model][dataset][dataset]['all_preds']).flatten()
                    ground_truth = np.array(results[model][dataset][dataset]['all_gt']).flatten()
                else:
                    if dataset in results[model][dataset]:
                        predictions = np.array(results[model][dataset][dataset]['all_preds']).flatten()
                        ground_truth = np.array(results[model][dataset][dataset]['all_gt']).flatten()
                    else:
                        predictions = np.array(results[model][dataset]['all_preds']).flatten()
                        ground_truth = np.array(results[model][dataset]['all_gt']).flatten()
                
                # Get valid actions and update the set of all possible actions
                valid_actions = ProcGenDefinitions.get_valid_action_space(dataset, "default")
                all_possible_actions.update(valid_actions)
                
                # Update counts
                for action in valid_actions:
                    if action not in all_pred_counts:
                        all_pred_counts[action] = 0
                        all_gt_counts[action] = 0
                    
                    all_pred_counts[action] += int(np.sum(predictions == action))
                    all_gt_counts[action] += int(np.sum(ground_truth == action))
                
            except Exception as e:
                print(f"Error processing {model} on {dataset}: {e}")
                continue
        
        try:
            # Convert to sorted lists for plotting
            all_actions = sorted(list(all_possible_actions))
            
            # Convert to percentages
            total_preds = sum(all_pred_counts.values())
            total_gt = sum(all_gt_counts.values())
            pred_percentages = np.array([all_pred_counts.get(a, 0) / total_preds * 100 for a in all_actions])
            gt_percentages = np.array([all_gt_counts.get(a, 0) / total_gt * 100 for a in all_actions])
            
            # Create figure
            plt.figure(figsize=(15, 8))
            
            # Create stacked bars to show overlap
            x = np.arange(len(all_actions))
            
            # Plot the smaller of the two values as overlap
            overlap = np.minimum(pred_percentages, gt_percentages)
            gt_only = gt_percentages - overlap
            pred_only = pred_percentages - overlap
            
            # Plot stacked bars
            plt.bar(x, overlap, label='Overlap', color=OVERLAP_COLOR, alpha=0.7)
            plt.bar(x, gt_only, bottom=overlap, label='Ground Truth Only', color=GT_COLOR, alpha=0.7)
            plt.bar(x, pred_only, bottom=overlap, label='Predictions Only', color=PRED_COLOR, alpha=0.7)
            
            # Create smooth curves
            # Generate points for smooth curve
            x_smooth = np.linspace(x.min(), x.max(), 300)
            
            # Add padding points to make the curve more natural at the edges
            x_pad = np.concatenate(([x.min() - 1], x, [x.max() + 1]))
            gt_pad = np.concatenate(([0], gt_percentages, [0]))
            pred_pad = np.concatenate(([0], pred_percentages, [0]))
            
            # Create the spline functions
            gt_spline = make_interp_spline(x_pad, gt_pad, k=3)
            pred_spline = make_interp_spline(x_pad, pred_pad, k=3)
            
            # Generate smooth curves
            gt_smooth = gt_spline(x_smooth)
            pred_smooth = pred_spline(x_smooth)
            
            # Ensure no negative values in the smoothed curves
            gt_smooth = np.maximum(gt_smooth, 0)
            pred_smooth = np.maximum(pred_smooth, 0)
            
            # Plot the smooth curves with increased line width and solid lines
            plt.plot(x_smooth, gt_smooth, '-', color=GT_CURVE_COLOR, label='Ground Truth Trend', 
                    linewidth=3, zorder=5)  # Increased line width and zorder
            plt.plot(x_smooth, pred_smooth, '-', color=PRED_CURVE_COLOR, label='Prediction Trend',
                    linewidth=3, zorder=5)  # Increased line width and zorder
            
            # Customize the plot
            plt.xlabel('Action Class')
            plt.ylabel('Percentage of Total Actions (%)')
            plt.title(f'Aggregated Action Distribution Comparison for {model}\nAcross All Datasets')
            plt.xticks(x, [str(a) for a in all_actions])
            
            # Add value labels
            for i in range(len(all_actions)):
                # Only show labels for non-zero values
                if gt_percentages[i] > 0:
                    plt.text(i, gt_percentages[i], f'{gt_percentages[i]:.1f}%',
                           ha='center', va='bottom', rotation=45)
                if pred_percentages[i] > 0 and abs(pred_percentages[i] - gt_percentages[i]) > 0.1:
                    # Show prediction percentage only if it's significantly different from ground truth
                    plt.text(i, pred_percentages[i], f'{pred_percentages[i]:.1f}%',
                           ha='center', va='top', rotation=45)
            
            # Adjust y-axis limit to show all values with padding
            max_value = max(max(pred_percentages), max(gt_percentages))
            plt.ylim(0, max_value * 1.3)  # Add 30% padding
            
            # Add grid for better readability
            plt.grid(True, axis='y', alpha=0.3)
            
            # Place legend inside the plot at the top right corner
            plt.legend(loc='upper right', fontsize=14)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'unified_action_distribution_{model}.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error creating unified plot for {model}: {e}")
            continue

def plot_macro_recall_violin(results_dir: str, models: list[str]):
    """Create violin plots showing the distribution of macro recall scores across datasets for each model.
    
    Args:
        results_dir (str): Directory containing results
        models (list[str]): List of model names to compare
    """
    results = load_results(results_dir)
    
    # Get common datasets across all models
    common_datasets = set(results[models[0]].keys())
    for model in models[1:]:
        common_datasets.intersection_update(results[model].keys())
    common_datasets = sorted(list(common_datasets))
    
    # Collect macro recall scores for each model
    model_scores = {model: [] for model in models}
    
    for model in models:
        for dataset in common_datasets:
            try:
                if model in ['gpt4o', 'gpt4_1']:
                    score = results[model][dataset][dataset]['macro_recall']
                elif model == 'openvla':
                    if dataset in results[model][dataset]:
                        score = results[model][dataset][dataset]['macro_recall']
                    else:
                        score = results[model][dataset]['macro_recall']
                elif model == 'pi0_base':
                    score = results[model][dataset][dataset]['macro_recall']
                elif model == 'pi0_fast':
                    if dataset in results[model].keys():
                        score = results[model][dataset][dataset]['macro_recall']
                    else:
                        score = 0
                model_scores[model].append(score)
            except (KeyError, TypeError):
                model_scores[model].append(0)
                print(f"Missing data for {model} on {dataset}")
    
    # Create figure with extra space on right for stats
    plt.figure(figsize=(15, 8))  # Made figure wider to accommodate stats
    
    # Create violin plots
    violin_parts = plt.violinplot([model_scores[model] for model in models],
                                showmeans=True, showmedians=True)
    
    # Customize violin colors
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(COLORS[i])
        pc.set_alpha(0.7)
    
    # Customize mean and median lines
    violin_parts['cmeans'].set_color('black')
    violin_parts['cmedians'].set_color('red')
    
    # Add individual points in black
    for i, model in enumerate(models):
        # Add jittered points
        x = np.random.normal(i + 1, 0.04, size=len(model_scores[model]))
        plt.scatter(x, model_scores[model], alpha=0.4, color='black', s=30)
    
    # Calculate and display statistics
    for i, model in enumerate(models):
        scores = model_scores[model]
        mean = np.mean(scores)
        median = np.median(scores)
        std = np.std(scores)
        
        # Position text differently for OpenVLA
        x_pos = i + 1 + 0.3  # Slightly to the right of the violin
        if model == 'openvla':
            y_pos = min(scores) - 0.01  # Below the violin for OpenVLA
            v_align = 'top'  # Align text to top when below violin
        else:
            y_pos = max(scores) + 0.01  # Above the violin for others
            v_align = 'bottom'  # Align text to bottom when above violin
        
        # Add text annotations with clean formatting and white background
        plt.text(x_pos, y_pos,
                f'μ={mean:.2f}, m={median:.2f}, σ={std:.2f}',
                horizontalalignment='right',
                verticalalignment='top',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    # Customize plot
    plt.title('Distribution of Macro Recall Scores Across Datasets', fontsize=14, pad=20)
    plt.ylabel('Macro Recall Score', fontsize=12)
    plt.xticks(range(1, len(models) + 1), models, rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Adjust x-axis limits to make room for stats
    plt.xlim(0.5, len(models) + 1.5)
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(results_dir, 'macro_recall_violin_plot.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_macro_recall_heatmap(results_dir: str, models: list[str]):
    """Create a heatmap showing macro recall scores for all models across all subdatasets.
    
    Args:
        results_dir (str): Directory containing results
        models (list[str]): List of model names to compare
    """
    results = load_results(results_dir)
    
    # Get common datasets across all models
    common_datasets = set(results[models[0]].keys())
    for model in models[1:]:
        common_datasets.intersection_update(results[model].keys())
    common_datasets = sorted(list(common_datasets))
    
    # Create matrix for heatmap
    heatmap_data = np.zeros((len(common_datasets), len(models)))  # Transposed dimensions
    
    # Fill matrix with macro recall values
    for i, dataset in enumerate(common_datasets):  # Switched loop order
        for j, model in enumerate(models):
            try:
                if model in ['gpt4o', 'gpt4_1']:
                    score = results[model][dataset][dataset]['macro_recall']
                elif model == 'openvla':
                    if dataset in results[model][dataset]:
                        score = results[model][dataset][dataset]['macro_recall']
                    else:
                        score = results[model][dataset]['macro_recall']
                elif model == 'pi0_base':
                    score = results[model][dataset][dataset]['macro_recall']
                elif model == 'pi0_fast':
                    if dataset in results[model].keys():
                        score = results[model][dataset][dataset]['macro_recall']
                    else:
                        score = 0
                heatmap_data[i, j] = score
            except (KeyError, TypeError):
                heatmap_data[i, j] = 0
                print(f"Missing data for {model} on {dataset}")
    
    # Create figure
    plt.figure(figsize=(12, 15))  # Adjusted figure size for new orientation
    
    # Create heatmap with extra space at bottom for stats
    ax = plt.gca()
    sns.heatmap(heatmap_data, 
                annot=True,  # Show values in cells
                fmt='.2f',   # Format as 2 decimal places
                cmap='YlOrRd',  # Yellow to Orange to Red colormap
                xticklabels=models,
                yticklabels=common_datasets,
                cbar_kws={'label': 'Macro Recall'},
                ax=ax,
                annot_kws={'size': 16})  # Set heatmap value font size
    
    # Calculate column averages (per model)
    col_means = np.mean(heatmap_data, axis=0)
    
    # Add extra space at the bottom for the stats
    bottom = ax.get_position().y0
    height = ax.get_position().height
    ax.set_position([ax.get_position().x0, bottom + 0.05, ax.get_position().width, height - 0.05])
    
    # # Add mean stats below the heatmap
    # for j, mean in enumerate(col_means):
    #     plt.text(j + 0.5, len(common_datasets) + 0.5, f'μ={mean:.2f}',
    #             horizontalalignment='center',
    #             verticalalignment='center',
    #             fontsize=8,
    #             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    # Customize plot
    plt.title('Macro Recall Scores Across Models and Datasets', fontsize=20, pad=20)
    plt.xlabel('Models', fontsize=16)
    plt.ylabel('Datasets', fontsize=16)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(rotation=45, fontsize=16)
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(results_dir, 'macro_recall_heatmap.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_pi0_action_distribution_comparison(results_dir: str):
    """Create plots comparing prediction action distributions between pi0_fast and pi0_base.
    
    Args:
        results_dir (str): Directory containing results
    """
    # Define colors for better visualization
    PI0_FAST_COLOR = '#27aeef'  # Blue
    PI0_BASE_COLOR = '#87bc45'  # Green
    OVERLAP_COLOR = '#999999'   # Gray
    PI0_FAST_CURVE_COLOR = '#1b8bc7'  # Darker blue
    PI0_BASE_CURVE_COLOR = '#669934'  # Darker green

    results = load_results(results_dir)
    
    # Get common datasets between pi0_fast and pi0_base
    pi0_fast_datasets = set(results['pi0_fast'].keys())
    pi0_base_datasets = set(results['pi0_base'].keys())
    common_datasets = sorted(list(pi0_fast_datasets.intersection(pi0_base_datasets)))
    
    # Initialize aggregated counts
    pi0_fast_pred_counts = {}
    pi0_base_pred_counts = {}
    all_possible_actions = set()
    
    # Collect data across all datasets
    for dataset in common_datasets:
        try:
            # Get predictions for pi0_fast
            if dataset in results['pi0_fast'].keys():
                pi0_fast_preds = np.array(results['pi0_fast'][dataset][dataset]['all_preds']).flatten()
            else:
                continue
            
            # Get predictions for pi0_base
            pi0_base_preds = np.array(results['pi0_base'][dataset][dataset]['all_preds']).flatten()
            
            # Get valid actions and update the set of all possible actions
            valid_actions = ProcGenDefinitions.get_valid_action_space(dataset, "default")
            all_possible_actions.update(valid_actions)
            
            # Update counts
            for action in valid_actions:
                if action not in pi0_fast_pred_counts:
                    pi0_fast_pred_counts[action] = 0
                    pi0_base_pred_counts[action] = 0
                
                pi0_fast_pred_counts[action] += int(np.sum(pi0_fast_preds == action))
                pi0_base_pred_counts[action] += int(np.sum(pi0_base_preds == action))
                
        except Exception as e:
            print(f"Error processing dataset {dataset}: {e}")
            continue
    
    try:
        # Convert to sorted lists for plotting
        all_actions = sorted(list(all_possible_actions))
        
        # Convert to percentages
        total_pi0_fast = sum(pi0_fast_pred_counts.values())
        total_pi0_base = sum(pi0_base_pred_counts.values())
        pi0_fast_percentages = np.array([pi0_fast_pred_counts.get(a, 0) / total_pi0_fast * 100 for a in all_actions])
        pi0_base_percentages = np.array([pi0_base_pred_counts.get(a, 0) / total_pi0_base * 100 for a in all_actions])
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Create stacked bars to show overlap
        x = np.arange(len(all_actions))
        
        # Plot the smaller of the two values as overlap
        overlap = np.minimum(pi0_fast_percentages, pi0_base_percentages)
        pi0_fast_only = pi0_fast_percentages - overlap
        pi0_base_only = pi0_base_percentages - overlap
        
        # Plot stacked bars
        plt.bar(x, overlap, label='Overlap', color=OVERLAP_COLOR, alpha=0.7)
        plt.bar(x, pi0_fast_only, bottom=overlap, label='PI0 Fast Only', color=PI0_FAST_COLOR, alpha=0.7)
        plt.bar(x, pi0_base_only, bottom=overlap, label='PI0 Base Only', color=PI0_BASE_COLOR, alpha=0.7)
        
        # Create smooth curves
        # Generate points for smooth curve
        x_smooth = np.linspace(x.min(), x.max(), 300)
        
        # Add padding points to make the curve more natural at the edges
        x_pad = np.concatenate(([x.min() - 1], x, [x.max() + 1]))
        pi0_fast_pad = np.concatenate(([0], pi0_fast_percentages, [0]))
        pi0_base_pad = np.concatenate(([0], pi0_base_percentages, [0]))
        
        # Create the spline functions
        pi0_fast_spline = make_interp_spline(x_pad, pi0_fast_pad, k=3)
        pi0_base_spline = make_interp_spline(x_pad, pi0_base_pad, k=3)
        
        # Generate smooth curves
        pi0_fast_smooth = pi0_fast_spline(x_smooth)
        pi0_base_smooth = pi0_base_spline(x_smooth)
        
        # Ensure no negative values in the smoothed curves
        pi0_fast_smooth = np.maximum(pi0_fast_smooth, 0)
        pi0_base_smooth = np.maximum(pi0_base_smooth, 0)
        
        # Plot the smooth curves with increased line width
        plt.plot(x_smooth, pi0_fast_smooth, '-', color=PI0_FAST_CURVE_COLOR, label='PI0 Fast Trend', 
                linewidth=3, zorder=5)
        plt.plot(x_smooth, pi0_base_smooth, '-', color=PI0_BASE_CURVE_COLOR, label='PI0 Base Trend',
                linewidth=3, zorder=5)
        
        # Customize the plot
        plt.xlabel('Action Class', fontsize=16)
        plt.ylabel('Percentage of Total Actions (%)', fontsize=16)
        plt.title('PI0 Fast vs PI0 Base Action Distribution Comparison\nAcross All Datasets', fontsize=20)
        plt.xticks(x, [str(a) for a in all_actions], fontsize=16)
        plt.yticks(fontsize=16)
        
        # Add value labels
        for i in range(len(all_actions)):
            # Only show labels for non-zero values
            if pi0_fast_percentages[i] > 0:
                plt.text(i, pi0_fast_percentages[i], f'{pi0_fast_percentages[i]:.1f}%',
                       ha='center', va='bottom', rotation=45, fontsize=12)
            if pi0_base_percentages[i] > 0 and abs(pi0_base_percentages[i] - pi0_fast_percentages[i]) > 0.1:
                plt.text(i, pi0_base_percentages[i], f'{pi0_base_percentages[i]:.1f}%',
                       ha='center', va='top', rotation=45, fontsize=12)
        
        # Adjust y-axis limit to show all values with padding
        max_value = max(max(pi0_fast_percentages), max(pi0_base_percentages))
        plt.ylim(0, max_value * 1.3)  # Add 30% padding
        
        # Add grid for better readability
        plt.grid(True, axis='y', alpha=0.3)
        
        # Place legend inside the plot at the top right corner
        plt.legend(loc='upper right', fontsize=14)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'pi0_action_distribution_comparison.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error creating comparison plot: {e}")

if __name__ == "__main__":
    results_dir = "src/v0_2results"
    models = ['gpt4o', 'openvla', 'pi0_base', 'pi0_fast', 'gpt4_1']
    
    plot_pi0_action_distribution_comparison(results_dir)

    # # Generate violin plot for macro recall distribution
    # plot_macro_recall_violin(results_dir, models)
    
    # # Generate heatmap for macro recall scores
    # plot_macro_recall_heatmap(results_dir, models)
    
    # Generate action distribution plots
    # plot_action_distributions(results_dir, models)
    
    # plot_dataset_specific_metrics(results_dir)

    # Generate Brier MAE comparison plot
    # plot_cross_model_brier_mae(results_dir)

    # plot_model_metrics(results_dir, 'pi0_fast') #Change model as needed
    # plot_model_metrics(results_dir, 'gpt4o') #Change model as needed
    # plot_model_metrics(results_dir, 'gpt4_1') #Change model as needed
    # plot_model_metrics(results_dir, 'openvla') #Change model as needed
      # plot_dataset_specific_metrics(results_dir)

    # # Generate plots
    # calculate_classwise_metrics(results_dir, 'pi0_fast')
    # calculate_classwise_metrics(results_dir, 'gpt4o')
    # calculate_classwise_metrics(results_dir, 'gpt4_1')

    # plot_classwise_metrics(results_dir, 'gpt4o')
    # plot_classwise_metrics(results_dir, 'gpt4_1')
    # plot_classwise_metrics(results_dir, 'openvla')
    # plot_classwise_metrics(results_dir, 'pi0_base')
    # plot_classwise_metrics(results_dir, 'pi0_fast')

    # plot_cross_model_macro_micro_metric(results_dir, models, metric='recall', metric_type='micro', with_invalids=True)
    # plot_cross_model_macro_micro_metric(results_dir, models, metric='precision', metric_type='micro', with_invalids=True)
    # plot_cross_model_macro_micro_metric(results_dir, models, metric='f1', metric_type='micro', with_invalids=True)

    # plot_cross_model_macro_micro_metric(results_dir, models, metric='recall', metric_type='micro', with_invalids=False)
    # plot_cross_model_macro_micro_metric(results_dir, models, metric='precision', metric_type='micro', with_invalids=False)
    # plot_cross_model_macro_micro_metric(results_dir, models, metric='f1', metric_type='micro', with_invalids=False)

    # plot_cross_model_macro_micro_metric(results_dir, models, metric='recall', metric_type='macro', with_invalids=True)
    # plot_cross_model_macro_micro_metric(results_dir, models, metric='precision', metric_type='macro', with_invalids=True)
    # plot_cross_model_macro_micro_metric(results_dir, models, metric='f1', metric_type='macro', with_invalids=True)

    # Generate comparative plots
    # plot_cross_model_classwise_comparison(results_dir, models)
    
    # # # Calculate confusion matrices and MCC
    # calculate_confusion_matrices_and_mcc(results_dir, models)

    # # # # Generate heatmaps for each metric
    # # # for metric in ['f1', 'precision', 'recall']:
    # # #     plot_action_difficulty_heatmap(results_dir, models, metric)

    # # # Generate category performance plots
    # plot_category_performance(results_dir, models)

    # print_preds_gt_unique_value_counts(results_dir, models=models)
