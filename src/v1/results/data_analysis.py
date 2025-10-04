#%%
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

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
    if not values:
        return np.nan
    return float(np.mean(values))

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
gpt5_bfcl_emr = [0.285]  # referenced from literature

gpt5_sqa3d_emr = extract_key_from_json(gpt5_sqa3d, 'exact_match_rate')
pi0_sqa3d_emr = extract_key_from_json(pi0_hf_sqa3d, 'exact_match_rate')
magma_sqa3d_emr = extract_key_from_json(magma_sqa3d, 'exact_match_rate_with_invalids')

gpt5_robovqa_emr = extract_key_from_json(gpt5_robovqa, 'exact_match_rate')
pi0_robovqa_emr = extract_key_from_json(pi0_hf_robovqa, 'exact_match_accuracy')
magma_robovqa_emr = extract_key_from_json(magma_robovqa, 'exact_match_rate_with_invalids')

gpt5_overcooked_emr = extract_key_from_json(gpt5_overcooked, 'exact_match')
gpt5_overcooked_emr = gpt5_overcooked_emr[0] if gpt5_overcooked_emr else np.nan
pi0_overcooked_emr = extract_key_from_json(pi0_base_overcooked, 'exact_match_rate')

gpt5_odinw_emrs = extract_key_from_json(gpt5_odinw, 'exact_match_rate')
pi0_odinw_emrs = extract_key_from_json(pi0_odinw_results, 'exact_match_rate')
magma_odinw_emrs = extract_key_from_json(magma_odinw_results, 'exact_match_rate_with_invalids')

data = {
    'Task': ['PIQA', 'BFCL', 'SQA3D', 'RoboVQA', 'ODINW', 'Overcooked'],
    'GPT-5': [np.mean(gpt5_piqa_emr),
                              None,
                              np.mean(gpt5_sqa3d_emr),
                              np.mean(gpt5_robovqa_emr),
                              np.mean(gpt5_odinw_emrs),
                              np.mean([gpt5_overcooked_emr])],
    'Pi-0': [np.mean(pi0_piqa_emr),
                             np.mean(pi0_bfcl_emr),
                             np.mean(pi0_sqa3d_emr),
                             np.mean(pi0_robovqa_emr),
                             np.mean(pi0_odinw_emrs),
                             np.mean(pi0_overcooked_emr)],
    'Magma': [np.mean(magma_piqa_emr),
                              None,
                               np.mean(magma_sqa3d_emr),
                               np.mean(magma_robovqa_emr),
                               np.mean(magma_odinw_emrs),
                              None]
}
df = pd.DataFrame(data)

df = df.replace(0, 0.01)
df_melted = df.melt(id_vars=['Task'], value_vars=['GPT-5', 'Pi-0', 'Magma'], var_name='Model', value_name='Exact Match Rate')
barplot(df_melted, 'Exact Match Rate Comparison between GPT-5, Pi-0 and Magma', 'Exact Match Rate', 'Task', './emr_comparison.pdf')

#%% recall comparison
recall_df = pd.DataFrame({
    'Task': ['ODINW', 'Overcooked'],
    'GPT-5': [np.mean(extract_key_from_json(gpt5_odinw, 'recall')), np.mean(extract_key_from_json(gpt5_overcooked, 'recall'))],
    'Pi-0': [np.mean(extract_key_from_json(pi0_odinw_results, 'recall')), 0],
    'Magma': [np.mean(extract_key_from_json(magma_odinw_results, 'recall')), 0]
})
recall_melted = recall_df.melt(id_vars=['Task'], value_vars=['GPT-5', 'Pi-0', 'Magma'], var_name='Model', value_name='Recall')
barplot(recall_melted, 'Recall Comparison across Models', 'Recall', 'Task', './recall_comparison.pdf', y='Recall')

#%% precision comparison
precision_df = pd.DataFrame({
    'Task': ['ODINW', 'Overcooked'],
    'GPT-5': [np.mean(extract_key_from_json(gpt5_odinw, 'precision')), np.mean(extract_key_from_json(gpt5_overcooked, 'precision'))],
    'Pi-0': [np.mean(extract_key_from_json(pi0_odinw_results, 'precision')), 0],
    'Magma': [np.mean(extract_key_from_json(magma_odinw_results, 'precision')), 0]
})
precision_melted = precision_df.melt(id_vars=['Task'], value_vars=['GPT-5', 'Pi-0', 'Magma'], var_name='Model', value_name='Precision')
barplot(precision_melted, 'Precision Comparison across Models', 'Precision', 'Task', './precision_comparison.pdf', y='Precision')

#%% f1 comparison
f1_df = pd.DataFrame({
    'Task': ['ODINW', 'Overcooked'],
    'GPT-5': [np.mean(extract_key_from_json(gpt5_odinw, 'f1')), np.mean(extract_key_from_json(gpt5_overcooked, 'f1'))],
    'Pi-0': [np.mean(extract_key_from_json(pi0_odinw_results, 'f1')), 0],
    'Magma': [np.mean(extract_key_from_json(magma_odinw_results, 'f1')), 0]
})
f1_melted = f1_df.melt(id_vars=['Task'], value_vars=['GPT-5', 'Pi-0', 'Magma'], var_name='Model', value_name='F1')
barplot(f1_melted, 'F1 Comparison across Models', 'F1 Score', 'Task', './f1_comparison.pdf', y='F1')

#%% normalized AMSE comparison
pi0_openx_namse = extract_per_subtask_metric(pi0_base_openx, 'normalized_amse')
magma_openx_namse = extract_per_subtask_metric(magma_openx, 'normalized_amse')
gpt5_openx_namse = {}
for result in gpt5_openx:
    gpt5_openx_namse.update(extract_per_subtask_metric(result, 'normalized_amse'))

pi0_openx_namse_mapped = {openx_subtasks_mapping.get(k, k): v for k, v in pi0_openx_namse.items()}
magma_openx_namse_mapped = {openx_subtasks_mapping.get(k, k): v for k, v in magma_openx_namse.items()}
gpt5_openx_namse_mapped = {openx_subtasks_mapping.get(k, k): v for k, v in gpt5_openx_namse.items()}

task_names = sorted(set(openx_subtasks_mapping.values()))
openx_df = pd.DataFrame({
    'Task': task_names,
    'GPT-5': [gpt5_openx_namse_mapped.get(task, 0) for task in task_names],
    'Pi-0': [pi0_openx_namse_mapped.get(task, 0) for task in task_names],
    'Magma': [magma_openx_namse_mapped.get(task, 0) for task in task_names]
})
openx_df = openx_df.replace(0, 0.01)
openx_df_melted = openx_df.melt(id_vars=['Task'], value_vars=['GPT-5', 'Pi-0', 'Magma'], var_name='Model', value_name='Normalized AMSE')
barplot(openx_df_melted, 'Normalized AMSE Comparison between GPT-5, Pi-0 and Magma on OpenX', 'Normalized AMSE', 'Task', './openx_namse_comparison.pdf', y='Normalized AMSE')

#%% normalized AMA comparison
gpt5_openx_amae = {}
for result in gpt5_openx:
    gpt5_openx_amae.update(extract_per_subtask_metric(result, 'normalized_amae'))
pi0_openx_amae = extract_per_subtask_metric(pi0_base_openx, 'normalized_amae')
magma_openx_amae = extract_per_subtask_metric(magma_openx, 'normalized_amae')

gpt5_openx_amae_mapped = {openx_subtasks_mapping.get(k, k): v for k, v in gpt5_openx_amae.items()}
pi0_openx_amae_mapped = {openx_subtasks_mapping.get(k, k): v for k, v in pi0_openx_amae.items()}
magma_openx_amae_mapped = {openx_subtasks_mapping.get(k, k): v for k, v in magma_openx_amae.items()}

amae_task_names = task_names + ['Overcooked']
amae_df = pd.DataFrame({
    'Task': amae_task_names,
    'GPT-5': [gpt5_openx_amae_mapped.get(task, 0) for task in task_names] + [extract_key_from_json(gpt5_overcooked, 'normalized_amae')[0]],
    'Pi-0': [pi0_openx_amae_mapped.get(task, 0) for task in task_names] + [np.nan],
    'Magma': [magma_openx_amae_mapped.get(task, 0) for task in task_names] + [np.nan]
})

amae_df_melted = amae_df.melt(id_vars=['Task'], value_vars=['GPT-5', 'Pi-0', 'Magma'], var_name='Model', value_name='Normalized AMA')
barplot(amae_df_melted, 'Normalized AMA Comparison across Models', 'Normalized AMA', 'Task', './amae_comparison.pdf', y='Normalized AMA', ylim=None)

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
    ('BFCL', 'Magma'): None,
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

