# Pi0 Base Model Evaluation Guide

This guide provides comprehensive instructions for evaluating the Pi0 Base model on various datasets in the MultiNet benchmark.

## Environment Setup

### Prerequisites

First, ensure you have completed the base MultiNet environment setup as described in the main README.

### Pi0 Base-Specific Setup

We set up our conda environment and ran evaluations for Pi0 Base on GCP Instances with A100 40 GB VRAM GPUs. If you are using our code out-of-the-box, we recommend using the same infrastructure.

For setup, create a new conda environment and download the packages present in `src/eval/profiling/openpi/pyproject.toml`. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) before running the following commands:

```bash
cd MultiNet/src/eval/profiling/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Additional Setup for Vision-Language Datasets

For evaluating vision-language datasets (ODINW, PIQA, SQA3D, RoboVQA, BFCL), you need to clone the openpi submodule:

```bash
cd MultiNet/src/v1/modules
git submodule update --init openpi
```

## Dataset Evaluations

### 1. ODINW (Object Detection in the Wild)

ODINW is an object detection benchmark consisting of multiple subdatasets.

**Script:** `odinw_hf_inference.py`

**Command:**

```bash
cd MultiNet/src/v1/modules/openpi/scripts
python odinw_hf_inference.py \
  --dataset_dir < directory containing the ODinW dataset > \
  --output_dir < directory to store inference results > \
  --batch_size < batch size >
```

**Parameters:**
- `--dataset_dir`: **Required** - Directory containing the ODinW dataset
- `--output_dir`: Directory to store inference results (default: "./odinw_hf_inference_results")
- `--batch_size`: Batch size for inference (default: 8)
- `--model_id`: HuggingFace model identifier (default: "google/paligemma-3b-pt-224")
- `--device`: Device to run inference on (cuda, cpu, etc.). Auto-detect if not specified
- `--max_samples`: Maximum number of samples to process (default: all samples)

**Example:**

```bash
python odinw_hf_inference.py \
  --dataset_dir /path/to/root_data_dir \
  --batch_size 8
```

---

### 2. PIQA (Physical Interaction QA)

PIQA is a question answering dataset focused on physical commonsense reasoning.

**Script:** `piqa_hf_inference.py`

**Command:**

```bash
cd MultiNet/src/v1/modules/openpi/scripts
python piqa_hf_inference.py \
  --dataset_dir < directory containing the PIQA test jsonl > \
  --output_dir < directory to store inference results > \
  --batch_size < batch size >
```

**Parameters:**
- `--dataset_dir`: **Required** - Directory containing the PIQA test jsonl
- `--output_dir`: Directory to store inference results (default: "./piqa_hf_inference_results")
- `--batch_size`: Batch size for inference (default: 8)
- `--mask_image_tokens`: Whether to mask dummy image tokens in the input (default: False)
- `--model_id`: HuggingFace model identifier (default: "google/paligemma-3b-pt-224")
- `--device`: Device to run inference on (cuda, cpu, etc.). Auto-detect if not specified
- `--max_samples`: Maximum number of samples to process (default: all samples)

**Example:**

```bash
python piqa_hf_inference.py \
  --dataset_dir /path/to/piqa/test/ \
  --batch_size 8
```

---

### 3. SQA3D (Situated Question Answering in 3D Scenes)

SQA3D is a question answering dataset for 3D scene understanding.

**Script:** `sqa3d_hf_inference.py`

**Command:**

```bash
cd MultiNet/src/v1/modules/openpi/scripts
python sqa3d_hf_inference.py \
  --dataset_dir < directory containing the SQA3D test dataset > \
  --output_dir < directory to store inference results > \
  --batch_size < batch size >
```

**Parameters:**
- `--dataset_dir`: **Required** - Directory containing the SQA3D test dataset
- `--output_dir`: Directory to store inference results (default: "./sqa3d_hf_inference_results")
- `--batch_size`: Batch size for inference (default: 8)
- `--model_id`: HuggingFace model identifier (default: "google/paligemma-3b-pt-224")
- `--device`: Device to run inference on (cuda, cpu, etc.). Auto-detect if not specified
- `--max_samples`: Maximum number of samples to process (default: all samples)

**Example:**

```bash
python sqa3d_hf_inference.py \
  --dataset_dir /path/to/sqa3d/test/ \
  --batch_size 8
```

---

### 4. RoboVQA (Robot Visual Question Answering)

RoboVQA is a visual question answering dataset for robotics scenarios.

**Script:** `robovqa_hf_inference.py`

**Command:**

```bash
cd MultiNet/src/v1/modules/openpi/scripts
python robovqa_hf_inference.py \
  --dataset_dir < directory containing the RoboVQA dataset > \
  --output_dir < directory to store inference results > \
  --batch_size < batch size >
```

**Parameters:**
- `--dataset_dir`: **Required** - Directory containing the RoboVQA dataset
- `--output_dir`: Directory to store inference results (default: "./robovqa_hf_inference_results")
- `--dataset_name`: Name of the dataset (default: "openx_multi_embodiment")
- `--batch_size`: Batch size for inference (default: 4)
- `--model_id`: HuggingFace model identifier (default: "google/paligemma-3b-pt-224")
- `--device`: Device to run inference on (cuda, cpu, etc.). Auto-detect if not specified
- `--max_samples`: Maximum number of samples to process (default: all samples)

**Example:**

```bash
python robovqa_hf_inference.py \
  --dataset_dir /path/to/openx_multi_embodiment/ \
  --batch_size 8
```

---

### 5. BFCL (Berkeley Function Calling Leaderboard)

BFCL evaluates function calling capabilities of language models.

**Script:** `bfcl_hf_inference.py`

**Command:**

```bash
cd MultiNet/src/v1/modules/openpi/scripts
python bfcl_hf_inference.py \
  --dataset_dir < directory containing the BFCL test dataset > \
  --output_dir < directory to store inference results > \
  --batch_size < batch size >
```

**Parameters:**
- `--dataset_dir`: **Required** - Directory containing the BFCL test dataset
- `--output_dir`: Directory to store inference results (default: "./bfcl_hf_inference_results")
- `--batch_size`: Batch size for inference (default: 4)
- `--mask_image_tokens`: Whether to mask dummy image tokens in the input (default: False)
- `--model_id`: HuggingFace model identifier (default: "google/paligemma-3b-pt-224")
- `--device`: Device to run inference on (cuda, cpu, etc.). Auto-detect if not specified
- `--max_samples`: Maximum number of samples to process (default: all samples)

**Example:**

```bash
python bfcl_hf_inference.py \
  --dataset_dir /path/to/bfcl/test/ \
  --batch_size 8
```

---

### 6. Overcooked

Overcooked is a multi-agent cooperative game environment for evaluating action prediction.

**Script:** `overcooked_inference.py`

**Command:**

```bash
cd MultiNet/src/eval/profiling/openpi/scripts
python overcooked_inference.py \
  --output_dir < directory to store results > \
  --data_file < path to Overcooked pickle data file > \
  --batch_size < batch size >
```

**Parameters:**
- `--output_dir`: **Required** - Directory to store results
- `--data_file`: **Required** - Path to Overcooked pickle data file
- `--batch_size`: Batch size for inference (default: 5)
- `--max_samples`: Maximum number of samples to process, useful for testing (default: None)

**Example:**

```bash
python overcooked_inference.py \
  --output_dir ./results \
  --data_file /path/to/overcooked_ai/test/2020_hh_trials_test.pickle \
  --batch_size 8
```

---

### 7. OpenX (Open X-Embodiment)

OpenX is a large-scale robotics dataset with multiple subdatasets across different robot embodiments.

**Script:** `openx_inference.py`

**Command:**

```bash
cd MultiNet/src/eval/profiling/openpi/scripts
python openx_inference.py \
  --output_dir < directory to store results and dataset statistics > \
  --dataset_dir < root directory containing the openx dataset > \
  --batch_size < batch size >
```

**Parameters:**
- `--output_dir`: **Required** - Directory to store results and dataset statistics
- `--dataset_dir`: **Required** - Root directory containing the OpenX dataset (the dataset name is automatically extracted from the directory path)
  - Supported dataset names:
    - `openx_mobile_manipulation`
    - `openx_single_arm`
    - `openx_bimanual`
    - `openx_wheeled_robot`
    = `openx_quadrupedal`
- `--batch_size`: Batch size for inference (default: 5)
- `--num_shards`: Number of shards to process. If None, all shards are processed (default: None)

**Example:**

```bash
python openx_inference.py \
  --output_dir ./results \
  --dataset_dir /path/to/openx_single_arm/ \
  --batch_size 8
```

> **Note:** The script automatically extracts the dataset name from the directory path (e.g., `openx_single_arm` from `/path/to/openx_single_arm/`).

---

## Hardware Requirements

We recommend using GPUs with at least 40GB VRAM for optimal performance. The evaluations were conducted on A100 40GB GPUs.
