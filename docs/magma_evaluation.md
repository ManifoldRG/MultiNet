# Magma Model Evaluation Guide

This guide provides comprehensive instructions for evaluating the Magma model on various datasets in the MultiNet benchmark.

## Environment Setup

### Prerequisites

First, ensure you have completed the base MultiNet environment setup as described in the main README.

### Magma-Specific Setup

After setting up the base MultiNet environment, you need to install Magma-specific dependencies. The setup differs depending on which datasets you want to evaluate:

#### For Action Output Datasets (OpenX, Overcooked)

These datasets require the full Magma installation with all dependencies from `pyproject.toml`:

```bash
cd MultiNet/src/v1/modules/Magma
pip install -e .
```

You may have to change the transformers version to a custom version:
- Custom transformers from `git+https://github.com/jwyang/transformers.git@dev/jwyang-v4.48.2`

> **Note:** This installs all base dependencies including torch, transformers, tensorflow, and other required packages.

#### For Text Output Datasets (ODINW, PIQA, SQA3D, RoboVQA, BFCL)

For text generation and VQA datasets, you can use a lighter setup:

```bash
cd MultiNet/src/v1/modules/Magma
pip install -r text_gen_requirements.txt
```

This installs:
- `open_clip_torch`
- Custom transformers from `git+https://github.com/jwyang/transformers.git@dev/jwyang-v4.48.2`
- `bitsandbytes`

> **Note:** Make sure to run this after setting up the base MultiNet environment.

## Dataset Evaluations

> **Note:** All the scripts we provide for multimodal inference are single sample inference as of now.

### 1. ODINW (Object Detection in the Wild)

ODINW is an object detection benchmark consisting of multiple subdatasets.

**Script:** `odinw_single_inference.py`

**Command:**

```bash
cd MultiNet/src/v1/modules/Magma/scripts
python odinw_single_inference.py \
  --data_path < path to the odinw test data folder > \
  --dataset < name of the odinw subdataset (optional, evaluates all if not specified) > \
  --output_dir < directory to save results >
```

**Parameters:**
- `--model_name_or_path`: Model identifier (default: "microsoft/Magma-8B")
- `--data_path`: **Required** - Path to the ODINW test data folder
- `--dataset`: Name of the ODINW subdataset (default: None, evaluates all datasets)
- `--dtype`: Model data type - choices: ['fp16', 'bf16', 'fp32'] (default: "bf16")
- `--batch_size`: Batch size for inference (default: 1)
- `--max_seq_len`: Maximum sequence length for tokenization (default: 1024)
- `--max_new_tokens`: Max new tokens for generation (default: 75)
- `--temperature`: Generation temperature (default: 0.0)
- `--do_sample`: Enable sampling for generation (default: False)
- `--output_dir`: Directory to save the results file (default: "./results/odinw")

**Example:**

```bash
python odinw_single_inference.py \
  --data_path /path/to/odinw/test/ 
  --dataset AerialMaritimeDrone
```

---

### 2. PIQA (Physical Interaction QA)

PIQA is a question answering dataset focused on physical commonsense reasoning.

**Script:** `piqa_inference.py`

**Command:**

```bash
cd MultiNet/src/v1/modules/Magma/scripts
python piqa_inference.py \
  --data_path < path to the PIQA data file (e.g., test.jsonl) > \
  --output_dir < directory to save results > \
  --results_filename < name of the output results file > \
  --batch_size < batch size >
```

**Parameters:**
- `--model_name_or_path`: Model identifier (default: "microsoft/Magma-8B")
- `--data_path`: **Required** - Path to the PIQA data file (e.g., test.jsonl)
- `--dtype`: Model data type - choices: ['fp16', 'bf16', 'fp32'] (default: "fp16")
- `--batch_size`: Batch size for inference (default: 2)
- `--max_seq_len`: Maximum sequence length for tokenization (default: 1024)
- `--max_new_tokens`: Max new tokens for generation (default: 5)
- `--temperature`: Generation temperature (default: 0.0)
- `--do_sample`: Enable sampling for generation (default: False)
- `--output_dir`: Directory to save the results file (default: "./results")
- `--results_filename`: Name of the output results file (default: "piqa_results.json")

**Example:**

```bash
python piqa_inference.py \
  --data_path /path/to/test.jsonl \
  --batch_size 8
```

---

### 3. SQA3D (Situated Question Answering in 3D Scenes)

SQA3D is a question answering dataset for 3D scene understanding.

**Script:** `sqa3d_single_inference.py`

**Command:**

```bash
cd MultiNet/src/v1/modules/Magma/scripts
python sqa3d_single_inference.py \
  --data_path < path to the sqa3d test data folder > \
  --output_dir < directory to save results > \
  --results_filename < name of the output results file >
```

**Parameters:**
- `--model_name_or_path`: Model identifier (default: "microsoft/Magma-8B")
- `--data_path`: **Required** - Path to the SQA3D test data folder
- `--dtype`: Model data type - choices: ['fp16', 'bf16', 'fp32'] (default: "bf16")
- `--batch_size`: Batch size for inference (default: 1)
- `--max_seq_len`: Maximum sequence length for tokenization (default: 1024)
- `--max_new_tokens`: Max new tokens for generation (default: 75)
- `--temperature`: Generation temperature (default: 0.0)
- `--do_sample`: Enable sampling for generation (default: False)
- `--output_dir`: Directory to save the results file (default: "./results")
- `--results_filename`: Name of the output results file (default: "sqa3d_results.json")

**Example:**

```bash
python sqa3d_single_inference.py \
  --data_path /path/to/sqa3d/test/
```

---

### 4. RoboVQA (Robot Visual Question Answering)

RoboVQA is a visual question answering dataset for robotics scenarios.

**Script:** `robovqa_single_inference.py`

**Command:**

```bash
cd MultiNet/src/v1/modules/Magma/scripts
python robovqa_single_inference.py \
  --data_path /path/to/openx_multi_embodiment/test/
  --output_dir < directory to save results > \
  --results_filename < name of the output results file >
```

**Parameters:**
- `--model_name_or_path`: Model identifier (default: "microsoft/Magma-8B")
- `--data_path`: **Required** - Path to the RoboVQA test data folder
- `--dtype`: Model data type - choices: ['fp16', 'bf16', 'fp32'] (default: "bf16")
- `--batch_size`: Batch size for inference (default: 1)
- `--max_seq_len`: Maximum sequence length for tokenization (default: 1024)
- `--max_new_tokens`: Max new tokens for generation (default: 75)
- `--temperature`: Generation temperature (default: 0.0)
- `--do_sample`: Enable sampling for generation (default: False)
- `--output_dir`: Directory to save the results file (default: "./results")
- `--results_filename`: Name of the output results file (default: "robovqa_results.json")

**Example:**

```bash
python robovqa_single_inference.py \
  --data_path /path/to/openx_multi_embodiment/test/
```

---

### 5. Overcooked

Overcooked is a multi-agent cooperative game environment for evaluating action prediction.

**Script:** `overcooked_single_inference.py`

**Command:**

```bash
cd MultiNet/src/v1/modules/Magma/scripts
python overcooked_single_inference.py \
  --data_file < path to Overcooked pickle data file > \
  --output_dir < directory to save results > \
  --results_filename < name of the output results file > 
```

**Parameters:**
- `--data_file`: **Required** - Path to Overcooked pickle data file
- `--output_dir`: Directory to save the output results JSON file (default: "./results")
- `--results_filename`: Name for the output results file (default: "magma_overcooked_results.json")
- `--batch_size`: Batch size for inference (default: 4)
- `--max_samples`: Maximum number of samples to process, useful for testing (default: None)

**Example:**

```bash
python overcooked_single_inference.py \
  --data_file /path/to/overcooked_ai/test/2020_hh_trials_test.pickle
```

---

### 6. BFCL (Berkeley Function Calling Leaderboard)

BFCL evaluates function calling capabilities of language models.

**Script:** `bfcl_inference.py`

**Command:**

```bash
cd MultiNet/src/v1/modules/Magma/scripts
python bfcl_inference.py \
  --dataset_dir < directory containing the BFCL test dataset > \
  --output_dir < directory to save results > \
  --results_filename < name of the output results file > \
  --batch_size < batch size >
```

**Parameters:**
- `--model_name_or_path`: Model identifier (default: "microsoft/Magma-8B")
- `--dataset_dir`: **Required** - Directory containing the BFCL test dataset
- `--dtype`: Model data type - choices: ['fp16', 'bf16', 'fp32'] (default: "bf16")
- `--batch_size`: Batch size for inference (default: 2)
- `--max_seq_len`: Maximum sequence length for tokenization (default: 1024)
- `--max_new_tokens`: Max new tokens for generation (default: 150)
- `--temperature`: Generation temperature (default: 0.0)
- `--do_sample`: Enable sampling for generation (default: False)
- `--output_dir`: Directory to save the results file (default: "./results")
- `--results_filename`: Name of the output results file (default: "bfcl_magma_results.json")
- `--max_samples`: Maximum number of samples to process (default: None, processes all samples)

**Example:**

```bash
python bfcl_inference.py \
  --dataset_dir /path/to/bfcl/test \
  --batch_size 8
```

---

### 7. OpenX (Open X-Embodiment)

OpenX is a large-scale robotics dataset with multiple subdatasets across different robot embodiments.

#### 7.1 General OpenX Datasets

**Script:** `magma_openx_inference.py`

**Command:**

```bash
cd MultiNet/src/v1/modules/Magma/scripts
python magma_openx_inference.py \
  --dataset_dir < root directory of the OpenX dataset shards > \
  --output_dir < directory to save results > \
  --dataset_name < name of the dataset being evaluated > \
  --results_filename < name for the output results file > \
  --batch_size < batch size >
```

**Parameters:**
- `--dataset_dir`: **Required** - Root directory of the OpenX dataset shards containing the test directory
- `--output_dir`: Directory to save the output results JSON file (default: "./results")
- `--dataset_name`: Name of the dataset being evaluated (default: "openx")
  - Acceptable values:
    - `openx_mobile_manipulation`
    - `openx_single_arm`
    - `openx_bimanual`
    - `openx_wheeled_robot`
- `--results_filename`: Name for the output results file (default: "magma_openx_results.json")
- `--batch_size`: Batch size for inference (default: 2)
- `--num_shards`: Number of data shards to process. Processes all if not specified (default: None)

**Example:**

```bash
python magma_openx_inference.py \
  --dataset_dir /path/to/openx_single_arm/ \
  --dataset_name openx_single_arm \
  --batch_size 8
```

#### 7.2 OpenX Quadrupedal Dataset

**Script:** `magma_openx_quadrupedal_inference.py`

**Command:**

```bash
cd MultiNet/src/v1/modules/Magma/scripts
python magma_openx_quadrupedal_inference.py \
  --dataset_dir < root directory of the OpenX quadrupedal dataset shards > \
  --output_dir < directory to save results > \
  --dataset_name < name of the dataset being evaluated > \
  --results_filename < name for the output results file > \
  --batch_size < batch size >
```

**Parameters:**
- `--dataset_dir`: **Required** - Root directory of the OpenX quadrupedal dataset shards, containing the test directory
- `--output_dir`: Directory to save the output results JSON file (default: "./results")
- `--dataset_name`: Name of the dataset being evaluated (default: "openx_quadrupedal")
- `--results_filename`: Name for the output results file (default: "magma_openx_quadrupedal_results.json")
- `--batch_size`: Batch size for inference (default: 4)
- `--num_shards`: Number of data shards to process. Processes all if not specified (default: None)
- `--max_samples`: Maximum number of samples to process, useful for testing (default: None)

**Example:**

```bash
python magma_openx_quadrupedal_inference.py \
  --dataset_dir /path/to/openx_quadrupedal/ 
```

---

## Hardware Requirements

We recommend using GPUs with at least 40GB VRAM for optimal performance. The evaluations were conducted on:
- A100 40GB

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**: Reduce the `--batch_size` parameter if evaluating a dataset that we provide batch inference script for
2. **Transformers Version Issues**: Ensure you're using the correct transformers version from the custom fork: `git+https://github.com/jwyang/transformers.git@dev/jwyang-v4.48.2`

