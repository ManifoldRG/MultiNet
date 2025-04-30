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

current_file = Path(__file__).resolve()
project_root = next(
    (p for p in current_file.parents if p.name == "MultiNet"),
    current_file.parent
)
sys.path.append(str(project_root))
from definitions.procgen import ProcGenDefinitions
from src.eval_utils import get_precision_per_class, get_recall_per_class, get_f1_per_class, get_macro_precision, get_macro_recall, get_macro_f1

COLORS = ['#1f77b4',  # Blue
          '#d62728',  # Red
          '#2ca02c',  # Green
          '#9467bd',  # Purple
          '#ff7f0e']  # Orange

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

def plot_individual_models(results_dir, models):
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

def plot_individual_models_macro_recall(results_dir, models):
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
    gpt4o_scores = [results['gpt4o'][dataset][dataset]['macro_recall'] for dataset in subdatasets]
    gpt4_1_scores = [results['gpt4_1'][dataset][dataset]['macro_recall'] for dataset in subdatasets]
    openvla_scores = []
    for dataset in subdatasets:
        if dataset in results['openvla'][dataset]:
            openvla_scores.append(results['openvla'][dataset][dataset]['macro_recall'])
        else:
            openvla_scores.append(results['openvla'][dataset]['macro_recall'])
    #print(results['pi0_base'].keys())
    pi0_base_scores = [results['pi0_base'][dataset][dataset]['macro_recall'] for dataset in subdatasets]

    pi0_fast_scores = []
    for dataset in subdatasets:
        if dataset in results['pi0_fast'].keys():
            pi0_fast_scores.append(results['pi0_fast'][dataset][dataset]['macro_recall'])
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
    plt.ylabel('Macro Recall')
    plt.title('Model Performance Comparison Across Subdatasets')
    plt.xticks(x, subdatasets, rotation=45)
    # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode="expand", ncol=5)
    plt.legend()
    plt.ylim(0, 0.15)  # Set y-axis limit to 0.15
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison_macro_recall.png'))
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
        #plt.bar(x, recalls)
        plt.bar(x, f1s)
        # Customize plot
        plt.ylabel('F1 Score')
        plt.xlabel('Class ID')
        plt.title(f'Class-wise F1 Score for {model_name} on {dataset}')
        plt.xticks(x, [str(i) for i in range(num_classes)])
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{model_name}_{dataset}_classwise_f1_score.png'))
        plt.close()
    

def plot_model_metrics(results_dir, model_name):
    """Create plots comparing different metrics for a single model across subdatasets
       CHANGE KEYS/METRIC NAMES BASED ON THE RESULTS FILES
    """
    results = load_results(results_dir)
    
    # Get list of all subdatasets for the model
    subdatasets = list(results[model_name].keys())
    
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
    else:
        avg_normalized_brier_mae_key = 'average_normalized_brier_mae'
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
    # Create the figure and axis
    plt.figure(figsize=(12, 6))
    
    # Create bars for each metric
    plt.bar(x - width, normalized_amae, width, label='Normalized Brier MAE')
    plt.bar(x, amae, width, label='Brier MAE')
    plt.bar(x + width, normalized_quantile_filtered_amae, width, label='Normalized Quantile Filtered AMAE')
    # Customize the plot
    plt.ylabel('Score')
    plt.title(f'Metrics Comparison for {model_name} Across Subdatasets')
    plt.xticks(x, subdatasets, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name}_metrics_comparison_normalized_and_brier_mae_and_quantile_filtered_amae.png'))
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
            json_path = os.path.join(results_dir, model_name, 'procgen_results', 'corrected_results')
        elif model_name == 'gpt4o':
            json_path = os.path.join(results_dir, model_name, 'procgen_results', 'corrected_results')
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


def plot_cross_model_class_comparison(results_dir: str, models: list[str]):
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
            
            # Add value labels
            for j, v in enumerate(averages):
                plt.text(x[j] + i*bar_width, v + stds[j], f'{v:.2f}',
                        ha='center', va='bottom', rotation=45, fontsize=8)
        
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
    
    # Also create a table with the exact values
    table_data = []
    headers = ['Action Class'] + [f'{model} (mean ± std)' for model in models]
    
    for action in sorted(action_class_metrics.keys()):
        row = [str(action)]
        for model in models:
            f1_values = action_class_metrics[action][model]['f1']
            mean = np.mean(f1_values)
            std = np.std(f1_values)
            row.append(f'{mean:.3f} ± {std:.3f}')
        table_data.append(row)
    
    # Save table as text file
    with open(os.path.join(results_dir, 'action_class_performance_table.txt'), 'w') as f:
        # Write headers
        f.write(' | '.join(headers) + '\n')
        f.write('-' * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)) + '\n')
        
        # Write data rows
        for row in table_data:
            f.write(' | '.join(row) + '\n')

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
        'Pure Navigation': ['maze', 'leaper', 'climber'],
        'Combat': ['bossfight', 'starpilot', 'plunder'],
        'Collection': ['chaser', 'fruitbot', 'heist', 'miner'],
        'Platforming': ['ninja', 'climber', 'coinrun']
    }
    
    state_action_categories = {
        'Direct Mapping': ['maze', 'climber', 'leaper'],
        'Strategic Planning': ['bossfight', 'starpilot', 'heist'],
        'Dynamic Response': ['ninja', 'caveflyer', 'dodgeball']
    }
    
    # Define metrics to analyze
    metrics = {
        'macro_f1': 'Macro F1 Score',
        'normalized_amae': 'Normalized AMAE'  # Adjust key based on your results structure
    }
    
    # Plot settings
    bar_width = 0.15
    
    # Create plots for each categorization scheme
    categorizations = {
        'Action Space Complexity': action_space_categories,
        'Game Mechanics': game_mechanic_categories,
        'State-Action Complexity': state_action_categories
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
    
    # Process each model and dataset
    for model in models:
        all_preds = []  # Store all predictions for union confusion matrix
        all_gt = []     # Store all ground truth for union confusion matrix
        
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
                
                # Calculate confusion matrix
                cm = confusion_matrix(ground_truth, predictions, labels=range(num_classes))
                confusion_matrices[model][dataset] = cm
                
                # Calculate MCC
                mcc = matthews_corrcoef(ground_truth, predictions)
                mcc_scores[model][dataset] = mcc
                
                # Accumulate predictions and ground truth for union confusion matrix
                all_preds.extend(predictions)
                all_gt.extend(ground_truth)
                
                # Plot individual confusion matrix
                plt.figure(figsize=(10, 8))
                # Row normalize the confusion matrix
                row_sums = cm.sum(axis=1)[:, np.newaxis]
                # Handle zero-sum rows to avoid division by zero
                row_sums[row_sums == 0] = 1  # Replace zeros with ones to avoid division by zero
                cm_normalized = cm.astype('float') / row_sums
                sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                           xticklabels=range(num_classes),
                           yticklabels=range(num_classes))
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
            # Get maximum number of classes across all datasets
            max_classes = max(len(ProcGenDefinitions.get_valid_action_space(dataset, 'default')) 
                            for dataset in common_datasets)
            
            union_cm = confusion_matrix(all_gt, all_preds, labels=range(max_classes))
            union_confusion_matrices[model] = union_cm
            
            # Calculate union MCC
            union_mcc = matthews_corrcoef(all_gt, all_preds)
            
            # Plot union confusion matrix
            plt.figure(figsize=(12, 10))
            # Row normalize the union confusion matrix
            row_sums = union_cm.sum(axis=1)[:, np.newaxis]
            # Handle zero-sum rows to avoid division by zero
            row_sums[row_sums == 0] = 1  # Replace zeros with ones to avoid division by zero
            union_cm_normalized = union_cm.astype('float') / row_sums
            sns.heatmap(union_cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                       xticklabels=range(max_classes),
                       yticklabels=range(max_classes))
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


def print_preds_gt_unique_value_counts(results_dir):
    results = load_results(results_dir)
    models = list(results.keys())

    # Analyze action value distributions for each dataset
    for dataset in results['gpt4o'].keys():
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
            print(f"\n{col}:")
            print(value_counts)
            print(f"Total unique values: {len(value_counts)}")
            print(f"Most common value: {value_counts.idxmax()} (count: {value_counts.max()})")
            print(f"Least common value: {value_counts.idxmin()} (count: {value_counts.min()})")


if __name__ == "__main__":
    results_dir = "src/v0_2results"
    models = ['gpt4o', 'openvla', 'pi0_base', 'pi0_fast', 'gpt4_1']
    
    # plot_dataset_specific_metrics(results_dir)

    # # Generate plots
    # calculate_classwise_metrics(results_dir, 'pi0_fast')
    # plot_model_metrics(results_dir, 'pi0_fast') #Change model as needed
    # plot_classwise_metrics(results_dir, 'pi0_fast')

    # calculate_classwise_metrics(results_dir, 'gpt4o')
    # plot_model_metrics(results_dir, 'gpt4o') #Change model as needed
    # plot_classwise_metrics(results_dir, 'gpt4o')

    # calculate_classwise_metrics(results_dir, 'gpt4_1')
    # plot_model_metrics(results_dir, 'gpt4_1') #Change model as needed
    # plot_classwise_metrics(results_dir, 'gpt4_1')

    # plot_individual_models(results_dir, models)
    # plot_individual_models_macro_recall(results_dir, models)

    # # Generate comparative plots
    # plot_cross_model_class_comparison(results_dir, models)
    
    # # # Calculate confusion matrices and MCC
    # calculate_confusion_matrices_and_mcc(results_dir, models)

    # # # # Generate heatmaps for each metric
    # # # for metric in ['f1', 'precision', 'recall']:
    # # #     plot_action_difficulty_heatmap(results_dir, models, metric)

    # # # Generate category performance plots
    # plot_category_performance(results_dir, models)

    print_preds_gt_unique_value_counts(results_dir)
