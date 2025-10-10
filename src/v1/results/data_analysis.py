#%%
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from numbers import Number

#%% helper functions
def extract_key_from_json(data, key):
    """Recursively extract all values for a specified key from nested JSON-like data."""
    if isinstance(data, list):
        values = []
        for item in data:
            values.extend(extract_key_from_json(item, key))
        return values
    if isinstance(data, dict):
        values = []
        for k, v in data.items():
            if k == key:
                values.append(v)
            else:
                values.extend(extract_key_from_json(v, key))
        return values
    return []

def extract_per_subtask_metric(data, metric_key):
    """Return a mapping from subtask name to the specified metric."""
    result = {}
    if isinstance(data, list):
        for item in data:
            result.update(extract_per_subtask_metric(item, metric_key))
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict) and metric_key in v:
                result[k] = v[metric_key]
            else:
                result.update(extract_per_subtask_metric(v, metric_key))
    return result

# simple helpers to avoid repeating np.mean checks
def safe_mean(values):
    if values is None:
        return np.nan
    if isinstance(values, Number):
        values = [values]
    else:
        values = list(values)
    filtered = [v for v in values if pd.notna(v)]
    if not filtered:
        return np.nan
    # add a tiny value if mean is zero for visualization purposes
    mean = np.mean(filtered)
    if mean == 0:
        return 0.01
    return float(np.mean(filtered))


def aggregate_by_mapping(metric_map, mapping):
    aggregated = {}
    for dataset, value in metric_map.items():
        task = mapping.get(dataset, dataset)
        aggregated.setdefault(task, []).append(value)
    return {task: safe_mean(values) for task, values in aggregated.items()}

def barplot(dataframe, title, ylabel, xlabel, save_path, y='Exact Match Rate', ylim=(0, 1)):
    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Task', y=y, hue='Model', data=dataframe)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# mapping from dataset keys to consolidated OpenX subtasks
openx_subtasks_mapping = {
    'openx_bimanual': 'bimanual',
    'openx_quadrupedal': 'quadrupedal',
    'openx_mobile_manipulation': 'mobile_manipulation',
    'openx_single_arm': 'single_arm',
    'openx_wheeled_robot': 'wheeled_robot',
    'berkeley_gnm_sac_son': 'wheeled_robot',
    'utokyo_saytap_converted_externally_to_rlds': 'quadrupedal',
    'bridge': 'single_arm',
    'utokyo_arm_bimanual_converted_externally_to_rlds': 'bimanual',
    'utokyo_xarm_bimanual_converted_externally_to_rlds': 'bimanual',
    'fractal20220817_data': 'mobile_manipulation',
}

#%% load Pi-0 results
with open('./pi0/pi0_base_openx_results_final.json') as f:
    pi0_base_openx = json.load(f)
with open('./pi0/pi0_base_overcooked_results.json') as f:
    pi0_base_overcooked = json.load(f)
with open('./pi0/pi0_hf_bfcl_inference_results.json') as f:
    pi0_hf_bfcl_inference = json.load(f)
with open('./pi0/pi0_hf_piqa_inference_results.json') as f:
    pi0_hf_piqa = json.load(f)
with open('./pi0/pi0_hf_robovqa_inference_results.json') as f:
    pi0_hf_robovqa = json.load(f)
with open('./pi0/pi0_hf_sqa3d_inference_results.json') as f:
    pi0_hf_sqa3d = json.load(f)
pi0_odinw_results = []
for file in os.listdir('./pi0/odinw'):
    if file.endswith('.json'):
        with open(os.path.join('./pi0/odinw', file)) as f:
            pi0_odinw_results.append(json.load(f))

#%% load GPT-5 results
gpt5_openx = []
for file in os.listdir('./genesis/gpt_5/low_reasoning/openx'):
    if file.endswith('.json'):
        with open(os.path.join('./genesis/gpt_5/low_reasoning/openx', file)) as f:
            gpt5_openx.append(json.load(f))

gpt5_odinw = []
for file in os.listdir('./genesis/gpt_5/low_reasoning/odinw'):
    if file.endswith('.json'):
        with open(os.path.join('./genesis/gpt_5/low_reasoning/odinw', file)) as f:
            gpt5_odinw.append(json.load(f))

gpt5_overcooked = []
for file in os.listdir('./genesis/gpt_5/low_reasoning/overcooked_ai'):
    if file.endswith('.json'):
        with open(os.path.join('./genesis/gpt_5/low_reasoning/overcooked_ai', file)) as f:
            gpt5_overcooked.append(json.load(f))

gpt5_piqa = []
for file in os.listdir('./genesis/gpt_5/low_reasoning/piqa'):
    if file.endswith('.json'):
        with open(os.path.join('./genesis/gpt_5/low_reasoning/piqa', file)) as f:
            gpt5_piqa.append(json.load(f))

gpt5_robovqa = []
for file in os.listdir('./genesis/gpt_5/low_reasoning/robot_vqa'):
    if file.endswith('.json'):
        with open(os.path.join('./genesis/gpt_5/low_reasoning/robot_vqa', file)) as f:
            gpt5_robovqa.append(json.load(f))

gpt5_sqa3d = []
for file in os.listdir('./genesis/gpt_5/low_reasoning/sqa3d'):
    if file.endswith('.json'):
        with open(os.path.join('./genesis/gpt_5/low_reasoning/sqa3d', file)) as f:
            gpt5_sqa3d.append(json.load(f))

#%% load Magma results
with open('./magma/magma_openx_results_final.json') as f:
    magma_openx = json.load(f)
with open('./magma/piqa_results.json') as f:
    magma_piqa = json.load(f)
with open('./magma/robovqa_results.json') as f:
    magma_robovqa = json.load(f)
with open('./magma/sqa3d_results.json') as f:
    magma_sqa3d = json.load(f)
with open('./magma/magma_overcooked_results.json') as f:
    magma_overcooked = json.load(f)
with open('./magma/bfcl_results.json') as f:
    magma_bfcl = json.load(f)
magma_odinw_results = []
for file in os.listdir('./magma/odinw/corrected_results'):
    if file.endswith('.json'):
        with open(os.path.join('./magma/odinw/corrected_results', file)) as f:
            magma_odinw_results.append(json.load(f))

#%% exact match rate comparison
gpt5_piqa_emr = extract_key_from_json(gpt5_piqa, 'exact_match_rate')
pi0_piqa_emr = extract_key_from_json(pi0_hf_piqa, 'exact_match_rate')
magma_piqa_emr = extract_key_from_json(magma_piqa, 'exact_match_rate')

pi0_bfcl_emr = extract_key_from_json(pi0_hf_bfcl_inference, 'exact_match_accuracy')
magma_bfcl_emr = extract_key_from_json(magma_bfcl, 'exact_match_accuracy')
gpt5_bfcl_emr = [0.285]  # referenced from literature

gpt5_sqa3d_emr = extract_key_from_json(gpt5_sqa3d, 'exact_match_rate')
pi0_sqa3d_emr = extract_key_from_json(pi0_hf_sqa3d, 'exact_match_rate')
magma_sqa3d_emr = extract_key_from_json(magma_sqa3d, 'exact_match_rate_with_invalids')

gpt5_robovqa_emr = extract_key_from_json(gpt5_robovqa, 'exact_match_rate')
pi0_robovqa_emr = extract_key_from_json(pi0_hf_robovqa, 'exact_match_accuracy')
magma_robovqa_emr = extract_key_from_json(magma_robovqa, 'exact_match_rate_with_invalids')

gpt5_overcooked_emr_values = extract_key_from_json(gpt5_overcooked, 'exact_match')
gpt5_overcooked_emr = safe_mean(gpt5_overcooked_emr_values)
pi0_overcooked_emr = extract_key_from_json(pi0_base_overcooked, 'exact_match_rate')
magma_overcooked_emr = extract_key_from_json(magma_overcooked, 'exact_match_rate')

gpt5_odinw_emrs = extract_key_from_json(gpt5_odinw, 'exact_match_rate')
pi0_odinw_emrs = extract_key_from_json(pi0_odinw_results, 'exact_match_rate')
magma_odinw_emrs = extract_key_from_json(magma_odinw_results, 'exact_match_rate_with_invalids')

data = {
    'Task': ['PIQA', 'BFCL', 'SQA3D', 'RoboVQA', 'ODINW', 'Overcooked'],
    'GPT-5': [safe_mean(gpt5_piqa_emr),
              safe_mean(gpt5_bfcl_emr),
              safe_mean(gpt5_sqa3d_emr),
              safe_mean(gpt5_robovqa_emr),
              safe_mean(gpt5_odinw_emrs),
              gpt5_overcooked_emr],
    'Pi-0': [safe_mean(pi0_piqa_emr),
             safe_mean(pi0_bfcl_emr),
             safe_mean(pi0_sqa3d_emr),
             safe_mean(pi0_robovqa_emr),
             safe_mean(pi0_odinw_emrs),
             safe_mean(pi0_overcooked_emr)],
    'Magma': [safe_mean(magma_piqa_emr),
              safe_mean(magma_bfcl_emr),
              safe_mean(magma_sqa3d_emr),
              safe_mean(magma_robovqa_emr),
              safe_mean(magma_odinw_emrs),
              safe_mean(magma_overcooked_emr)]
}
df = pd.DataFrame(data)

df = df.replace(0, 0.01)
df_melted = df.melt(id_vars=['Task'], value_vars=['GPT-5', 'Pi-0', 'Magma'], var_name='Model', value_name='Exact Match Rate')
barplot(df_melted, 'Exact Match Rate Comparison between GPT-5, Pi-0 and Magma', 'Exact Match Rate', 'Task', './emr_comparison.pdf')

#%% recall comparison
recall_df = pd.DataFrame({
    'Task': ['ODINW', 'Overcooked'],
    'GPT-5': [safe_mean(extract_key_from_json(gpt5_odinw, 'recall')), safe_mean(extract_key_from_json(gpt5_overcooked, 'recall'))],
    'Pi-0': [safe_mean(extract_key_from_json(pi0_odinw_results, 'recall')), safe_mean(extract_key_from_json(pi0_base_overcooked, 'micro_recall'))],
    'Magma': [safe_mean(extract_key_from_json(magma_odinw_results, 'recall')), safe_mean(extract_key_from_json(magma_overcooked, 'micro_recall'))]
})
recall_melted = recall_df.melt(id_vars=['Task'], value_vars=['GPT-5', 'Pi-0', 'Magma'], var_name='Model', value_name='Recall')
barplot(recall_melted, 'Recall Comparison across Models', 'Recall', 'Task', './recall_comparison.pdf', y='Recall')

#%% precision comparison
precision_df = pd.DataFrame({
    'Task': ['ODINW', 'Overcooked'],
    'GPT-5': [safe_mean(extract_key_from_json(gpt5_odinw, 'precision')), safe_mean(extract_key_from_json(gpt5_overcooked, 'precision'))],
    'Pi-0': [safe_mean(extract_key_from_json(pi0_odinw_results, 'precision')), safe_mean(extract_key_from_json(pi0_base_overcooked, 'micro_precision'))],
    'Magma': [safe_mean(extract_key_from_json(magma_odinw_results, 'precision')), safe_mean(extract_key_from_json(magma_overcooked, 'micro_precision'))]
})
precision_melted = precision_df.melt(id_vars=['Task'], value_vars=['GPT-5', 'Pi-0', 'Magma'], var_name='Model', value_name='Precision')
barplot(precision_melted, 'Precision Comparison across Models', 'Precision', 'Task', './precision_comparison.pdf', y='Precision')

#%% f1 comparison
f1_df = pd.DataFrame({
    'Task': ['ODINW', 'Overcooked'],
    'GPT-5': [safe_mean(extract_key_from_json(gpt5_odinw, 'f1')), safe_mean(extract_key_from_json(gpt5_overcooked, 'f1'))],
    'Pi-0': [safe_mean(extract_key_from_json(pi0_odinw_results, 'f1')), safe_mean(extract_key_from_json(pi0_base_overcooked, 'micro_f1'))],
    'Magma': [safe_mean(extract_key_from_json(magma_odinw_results, 'f1')), safe_mean(extract_key_from_json(magma_overcooked, 'micro_f1'))]
})
f1_melted = f1_df.melt(id_vars=['Task'], value_vars=['GPT-5', 'Pi-0', 'Magma'], var_name='Model', value_name='F1')
barplot(f1_melted, 'F1 Comparison across Models', 'F1 Score', 'Task', './f1_comparison.pdf', y='F1')

#%% normalized AMSE comparison
pi0_openx_namse_raw = extract_per_subtask_metric(pi0_base_openx, 'normalized_amse')
magma_openx_namse_raw = extract_per_subtask_metric(magma_openx, 'normalized_amse')
gpt5_openx_namse_raw = {}
for result in gpt5_openx:
    gpt5_openx_namse_raw.update(extract_per_subtask_metric(result, 'normalized_amse'))

pi0_openx_namse_mapped = aggregate_by_mapping(pi0_openx_namse_raw, openx_subtasks_mapping)
magma_openx_namse_mapped = aggregate_by_mapping(magma_openx_namse_raw, openx_subtasks_mapping)
gpt5_openx_namse_mapped = aggregate_by_mapping(gpt5_openx_namse_raw, openx_subtasks_mapping)

base_tasks = list(dict.fromkeys(openx_subtasks_mapping.values()))
extra_tasks = sorted((set(gpt5_openx_namse_mapped.keys()) |
                      set(pi0_openx_namse_mapped.keys()) |
                      set(magma_openx_namse_mapped.keys())) - set(base_tasks))
task_names = base_tasks + extra_tasks
openx_df = pd.DataFrame({
    'Task': task_names,
    'GPT-5': [gpt5_openx_namse_mapped.get(task, np.nan) for task in task_names],
    'Pi-0': [pi0_openx_namse_mapped.get(task, np.nan) for task in task_names],
    'Magma': [magma_openx_namse_mapped.get(task, np.nan) for task in task_names]
})
openx_df = openx_df.replace(0, 0.01)
openx_df_melted = openx_df.melt(id_vars=['Task'], value_vars=['GPT-5', 'Pi-0', 'Magma'], var_name='Model', value_name='Normalized AMSE')
barplot(openx_df_melted, 'Normalized AMSE Comparison between GPT-5, Pi-0 and Magma on OpenX', 'Normalized AMSE', 'Task', './openx_namse_comparison.pdf', y='Normalized AMSE')

#%% normalized AMA comparison
gpt5_openx_amae_raw = {}
for result in gpt5_openx:
    gpt5_openx_amae_raw.update(extract_per_subtask_metric(result, 'normalized_amae'))
pi0_openx_amae_raw = extract_per_subtask_metric(pi0_base_openx, 'normalized_amae')
magma_openx_amae_raw = extract_per_subtask_metric(magma_openx, 'normalized_amae')

gpt5_openx_amae_mapped = aggregate_by_mapping(gpt5_openx_amae_raw, openx_subtasks_mapping)
pi0_openx_amae_mapped = aggregate_by_mapping(pi0_openx_amae_raw, openx_subtasks_mapping)
magma_openx_amae_mapped = aggregate_by_mapping(magma_openx_amae_raw, openx_subtasks_mapping)

amae_tasks = task_names.copy()
extra_amae_tasks = sorted((set(gpt5_openx_amae_mapped.keys()) |
                           set(pi0_openx_amae_mapped.keys()) |
                           set(magma_openx_amae_mapped.keys())) - set(amae_tasks))
amae_tasks.extend(extra_amae_tasks)

gpt5_overcooked_amae = safe_mean(extract_key_from_json(gpt5_overcooked, 'normalized_amae'))
magma_overcooked_amae = safe_mean(extract_key_from_json(magma_overcooked, 'normalized_amae'))

amae_task_names = amae_tasks + (['Overcooked'] if 'Overcooked' not in amae_tasks else [])
amae_df = pd.DataFrame({
    'Task': amae_task_names,
    'GPT-5': [gpt5_openx_amae_mapped.get(task, np.nan) for task in amae_tasks] + [gpt5_overcooked_amae],
    'Pi-0': [pi0_openx_amae_mapped.get(task, np.nan) for task in amae_tasks] + [np.nan],
    'Magma': [magma_openx_amae_mapped.get(task, np.nan) for task in amae_tasks] + [magma_overcooked_amae]
})

amae_df_melted = amae_df.melt(id_vars=['Task'], value_vars=['GPT-5', 'Pi-0', 'Magma'], var_name='Model', value_name='Normalized AMA')
barplot(amae_df_melted, 'Normalized AMAE Comparison across Models', 'Normalized AMA', 'Task', './amae_comparison.pdf', y='Normalized AMA', ylim=None)

#%% similarity score comparison with error bars
similarity_sources = {
    ('RoboVQA', 'GPT-5'): gpt5_robovqa,
    ('RoboVQA', 'Pi-0'): pi0_hf_robovqa,
    ('RoboVQA', 'Magma'): magma_robovqa,
    ('SQA3D', 'GPT-5'): gpt5_sqa3d,
    ('SQA3D', 'Pi-0'): pi0_hf_sqa3d,
    ('SQA3D', 'Magma'): magma_sqa3d,
    ('BFCL', 'GPT-5'): None,
    ('BFCL', 'Pi-0'): pi0_hf_bfcl_inference,
    ('BFCL', 'Magma'): magma_bfcl,
}

similarity_rows = []
model_order = ['GPT-5', 'Pi-0', 'Magma']
task_order = ['RoboVQA', 'SQA3D', 'BFCL']

for task in task_order:
    for model in model_order:
        source = similarity_sources.get((task, model))
        if source is None:
            similarity_rows.append({'Task': task, 'Model': model, 'Similarity Score': np.nan, 'Std': np.nan})
            continue
        scores = extract_key_from_json(source, 'avg_similarity_score')
        stds = extract_key_from_json(source, 'similarity_std')
        similarity_rows.append({
            'Task': task,
            'Model': model,
            'Similarity Score': safe_mean(scores),
            'Std': safe_mean(stds)
        })

similarity_df = pd.DataFrame(similarity_rows)
plot_data = similarity_df.dropna(subset=['Similarity Score']).copy()
plot_data['Model'] = pd.Categorical(plot_data['Model'], categories=model_order, ordered=True)
plot_data = plot_data.sort_values(['Task', 'Model'])

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=plot_data, x='Task', y='Similarity Score', hue='Model', order=task_order, hue_order=model_order)

for patch, (_, row) in zip(ax.patches, plot_data.iterrows()):
    std = row['Std']
    if not np.isnan(std):
        ax.errorbar(patch.get_x() + patch.get_width() / 2, patch.get_height(), yerr=std, ecolor='black', capsize=4, linewidth=1)

ax.set_title('Average Similarity Score Comparison across Models')
ax.set_ylabel('Average Similarity Score')
ax.set_xlabel('Task')
plt.tight_layout()
plt.savefig('./similarity_score_comparison.pdf')
plt.show()

#%% confusion matrix analysis
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def calculate_confusion_matrices():
    """Calculate and visualize multiclass confusion matrices for all three models on OdinW and PIQA datasets."""
    
    # Function to plot confusion matrix
    def plot_confusion_matrix(cm, title, labels=None, save_path=None):
        plt.figure(figsize=(10, 8))
        if labels is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    # Function to process OdinW datasets for all models
    def process_odinw_all_models():
        """Process OdinW datasets for all three models."""
        odinw_cms = {'Pi-0': {}, 'GPT-5': {}, 'Magma': {}}
        
        # Get dataset names from Pi-0 files
        odinw_dataset_names = []
        for filename in os.listdir('./pi0/odinw'):
            if filename.endswith('.json') and 'odinw' in filename:
                # Extract dataset name from filename
                # Expected format: pi0_hf_odinw_DatasetName_inference_results.json
                parts = filename.replace('.json', '').split('_')
                if len(parts) >= 4:
                    dataset_name = parts[3]  # DatasetName part
                    odinw_dataset_names.append(dataset_name)
        
        # Process each dataset for all models
        for dataset_name in odinw_dataset_names:
            print(f"Processing OdinW dataset: {dataset_name}")
            
            # Pi-0 processing
            pi0_filepath = f'./pi0/odinw/pi0_hf_odinw_{dataset_name}_inference_results.json'
            if os.path.exists(pi0_filepath):
                with open(pi0_filepath, 'r') as f:
                    pi0_data = json.load(f)
                
                if 'all_preds' in pi0_data and 'all_gt' in pi0_data:
                    preds = np.array(pi0_data['all_preds'])
                    gts = np.array(pi0_data['all_gt'])
                    
                    # Filter out invalid predictions (-1)
                    valid_mask = (preds != -1) & (gts != -1)
                    if np.sum(valid_mask) > 0:
                        valid_preds = preds[valid_mask]
                        valid_gts = gts[valid_mask]
                        
                        if len(valid_preds) > 0:
                            unique_labels = sorted(list(set(valid_gts) | set(valid_preds)))
                            cm = confusion_matrix(valid_gts, valid_preds, labels=unique_labels)
                            odinw_cms['Pi-0'][dataset_name] = {
                                'confusion_matrix': cm,
                                'labels': unique_labels,
                                'num_samples': len(valid_preds)
                            }
            
            # GPT-5 processing
            genesis_filepath = f'./genesis/gpt_5/low_reasoning/odinw/{dataset_name}_results.json'
            if os.path.exists(genesis_filepath):
                with open(genesis_filepath, 'r') as f:
                    genesis_data = json.load(f)
                
                # Genesis data is nested under dataset name
                if dataset_name in genesis_data:
                    data = genesis_data[dataset_name]
                    if 'preds' in data and 'gt_actions' in data:
                        preds = np.array(data['preds'])
                        gts = np.array(data['gt_actions'])
                        
                        # Filter out invalid predictions (assuming -1 or negative values)
                        valid_mask = (preds >= 0) & (gts >= 0)
                        if np.sum(valid_mask) > 0:
                            valid_preds = preds[valid_mask]
                            valid_gts = gts[valid_mask]
                            
                            if len(valid_preds) > 0:
                                unique_labels = sorted(list(set(valid_gts) | set(valid_preds)))
                                cm = confusion_matrix(valid_gts, valid_preds, labels=unique_labels)
                                odinw_cms['GPT-5'][dataset_name] = {
                                    'confusion_matrix': cm,
                                    'labels': unique_labels,
                                    'num_samples': len(valid_preds)
                                }
            
            # Magma processing
            magma_filepath = f'./magma/odinw/corrected_results/{dataset_name}.json'
            if os.path.exists(magma_filepath):
                with open(magma_filepath, 'r') as f:
                    magma_data = json.load(f)
                
                if 'preds' in magma_data and 'gt_labels' in magma_data:
                    preds = np.array(magma_data['preds'])
                    gts = np.array(magma_data['gt_labels'])
                    
                    # Filter out invalid predictions
                    valid_mask = (preds >= 0) & (gts >= 0)
                    if np.sum(valid_mask) > 0:
                        valid_preds = preds[valid_mask]
                        valid_gts = gts[valid_mask]
                        
                        if len(valid_preds) > 0:
                            unique_labels = sorted(list(set(valid_gts) | set(valid_preds)))
                            cm = confusion_matrix(valid_gts, valid_preds, labels=unique_labels)
                            odinw_cms['Magma'][dataset_name] = {
                                'confusion_matrix': cm,
                                'labels': unique_labels,
                                'num_samples': len(valid_preds)
                            }
        
        return odinw_cms
    
    # Function to process PIQA dataset for all models
    def process_piqa_all_models():
        """Process PIQA dataset for all three models."""
        piqa_cms = {}
        
        # Pi-0 PIQA
        if 'all_preds' in pi0_hf_piqa and 'all_gt' in pi0_hf_piqa:
            preds = np.array(pi0_hf_piqa['all_preds'])
            gts = np.array(pi0_hf_piqa['all_gt'])
            
            # Filter out invalid predictions (-1)
            valid_mask = (preds != -1) & (gts != -1)
            if np.sum(valid_mask) > 0:
                valid_preds = preds[valid_mask]
                valid_gts = gts[valid_mask]
                
                labels = [0, 1]  # PIQA is binary classification
                cm = confusion_matrix(valid_gts, valid_preds, labels=labels)
                piqa_cms['Pi-0'] = {
                    'confusion_matrix': cm,
                    'labels': labels,
                    'num_samples': len(valid_preds)
                }
        
        # Genesis PIQA
        if len(gpt5_piqa) > 0 and 'piqa' in gpt5_piqa[0]:
            piqa_data = gpt5_piqa[0]['piqa']
            if 'preds' in piqa_data and 'gt_labels' in piqa_data:
                preds = np.array(piqa_data['preds'])
                gts = np.array(piqa_data['gt_labels'])
                
                # Filter out invalid predictions (assuming -1 or None)
                valid_mask = (preds >= 0) & (gts >= 0)
                if np.sum(valid_mask) > 0:
                    valid_preds = preds[valid_mask]
                    valid_gts = gts[valid_mask]
                    
                    labels = [0, 1]  # PIQA is binary classification
                    cm = confusion_matrix(valid_gts, valid_preds, labels=labels)
                    piqa_cms['GPT-5'] = {
                        'confusion_matrix': cm,
                        'labels': labels,
                        'num_samples': len(valid_preds)
                    }
        
        # Magma PIQA
        if 'preds' in magma_piqa and 'gt_labels' in magma_piqa:
            preds = np.array(magma_piqa['preds'])
            gts = np.array(magma_piqa['gt_labels'])
            
            # Filter out invalid predictions
            valid_mask = (preds >= 0) & (gts >= 0)
            if np.sum(valid_mask) > 0:
                valid_preds = preds[valid_mask]
                valid_gts = gts[valid_mask]
                
                labels = [0, 1]  # PIQA is binary classification
                cm = confusion_matrix(valid_gts, valid_preds, labels=labels)
                piqa_cms['Magma'] = {
                    'confusion_matrix': cm,
                    'labels': labels,
                    'num_samples': len(valid_preds)
                }
        
        return piqa_cms
    
    # Process OdinW datasets for all models
    print("Processing OdinW datasets for all models...")
    odinw_cms = process_odinw_all_models()
    
    # Plot confusion matrices for OdinW datasets
    for model_name in ['Pi-0', 'GPT-5', 'Magma']:
        for dataset_name, cm_data in odinw_cms[model_name].items():
            # Only plot if we have reasonable amount of data and not too many classes
            if cm_data['num_samples'] >= 10 and len(cm_data['labels']) <= 20:
                plot_confusion_matrix(
                    cm_data['confusion_matrix'], 
                    f'{model_name} Confusion Matrix - OdinW {dataset_name}',
                    labels=cm_data['labels'],
                    save_path=f'./confusion_matrix_{model_name.lower().replace("-", "")}_odinw_{dataset_name}.pdf'
                )
    
    # Process PIQA dataset for all models
    print("Processing PIQA dataset for all models...")
    piqa_cms = process_piqa_all_models()
    
    # Plot PIQA confusion matrices
    for model_name, cm_data in piqa_cms.items():
        plot_confusion_matrix(
            cm_data['confusion_matrix'],
            f'{model_name} Confusion Matrix - PIQA',
            labels=['Option A', 'Option B'],
            save_path=f'./confusion_matrix_{model_name.lower()}_piqa.pdf'
        )
    
    # Print summary statistics
    print("\n=== Confusion Matrix Analysis Summary ===")
    
    print(f"\nOdinW Datasets:")
    for model_name in ['Pi-0', 'GPT-5', 'Magma']:
        print(f"  {model_name}:")
        if odinw_cms[model_name]:
            for dataset_name, cm_data in odinw_cms[model_name].items():
                print(f"    {dataset_name}: {cm_data['num_samples']} valid samples, {len(cm_data['labels'])} classes")
        else:
            print(f"    No valid data found")
    
    print(f"\nPIQA Dataset:")
    for model_name, cm_data in piqa_cms.items():
        print(f"  {model_name}: {cm_data['num_samples']} valid samples")
        
        # Calculate accuracy from confusion matrix
        cm = cm_data['confusion_matrix']
        accuracy = np.trace(cm) / np.sum(cm)
        print(f"    Accuracy: {accuracy:.4f}")
    
    return {
        'odinw_all_models': odinw_cms,
        'piqa_all_models': piqa_cms
    }

# Calculate confusion matrices
confusion_matrices = calculate_confusion_matrices()

print("\n=== Confusion Matrix Function Added ===")
print("You can now call calculate_confusion_matrices() to generate confusion matrices for:")
print("- OdinW datasets: GPT-5 and Magma models (Pi-0 has no valid predictions)")
print("- PIQA dataset: All three models (GPT-5, Pi-0, Magma)")
print("Generated confusion matrix PDFs will be saved in the current directory.")


# %%
