# GenESIS Framework - GPT Model Evaluation Guide

This guide provides comprehensive instructions for evaluating GPT models using the GenESIS framework on various datasets in the MultiNet benchmark.

## Overview

The GenESIS (General Evaluation System for Intelligence and Supervision) framework allows you to evaluate GPT models in a zero-shot setting using OpenAI's batch API. The evaluation process consists of two steps:
1. **Send batch jobs** - Submit evaluation requests to the OpenAI API
2. **Run evaluation** - Retrieve results and calculate metrics

## Environment Setup

### Prerequisites

First, ensure you have completed the base MultiNet environment setup as described in the main README.

### OpenAI API Key

You will need an OpenAI API key to use the GenESIS framework. The script will prompt you to enter it when running evaluations.

### Model Configuration

For MultiNet V1 GPT-5 evaluations, we use `gpt-5-2025-08-07`.

Other supported models can be found in `src/config.json`

## Dataset Evaluations

### 1. ODINW (Object Detection in the Wild)

ODINW is an object detection benchmark consisting of multiple subdatasets.

**Step 1: Send Batch Jobs**

```bash
cd MultiNet/scripts/eval_vlm
python send_batch_jobs_vlm_single_ds.py \
  --data_root_dir < Path to the root dataset directory containing the odinw directory > \
  --dataset_family odinw \
  --dataset_name < specific ODINW subdataset name > \
  --model < gpt model name > \
  --metadata_dir < path to save batch info > \
  --batch_size < batch size >
```

**Parameters:**
- `--data_root_dir`: **Required** - Path to the root dataset directory containing the odinw directory
- `--dataset_family`: **Required** - Set to `odinw`
- `--dataset_name`: **Required** - Specific ODINW subdataset name (see `definitions/odinw.py` for accepted dataset names)
  - Examples: `AerialMaritimeDrone`, `AmericanSignLanguageLetters`, `Aquarium`, `BCCD`, `ChessPieces`, `selfdrivingCar`, etc.
- `--model`: **Required** - GPT model name (see `src/config.json`)
- `--metadata_dir`: **Required** - Directory to save batch information
- `--batch_size`: Batch size for evaluation (default: 1)
- `--k_shots`: Number of few-shot examples (default: 0)

**Step 2: Run Evaluation**

```bash
python run_batch_eval_vlm.py \
  --batch_job_info_path < path to batch info json file > \
  --results_path < path to save results json >
```

---

### 2. PIQA (Physical Interaction QA)

PIQA is a question answering dataset focused on physical commonsense reasoning.

**Step 1: Send Batch Jobs**

```bash
cd MultiNet/scripts/eval_vlm
python send_batch_jobs_vlm_single_ds.py \
  --data_root_dir < Path to the the dataset directory containing the piqa directory > \
  --dataset_family piqa \
  --dataset_name piqa \
  --model < gpt model name > \
  --metadata_dir < path to save batch info > \
  --batch_size < batch size >
```

**Step 2: Run Evaluation**

```bash
python run_batch_eval_vlm.py \
  --batch_job_info_path < path to batch info json file > \
  --results_path < path to save results json >
```

---

### 3. SQA3D (Situated Question Answering in 3D Scenes)

SQA3D is a question answering dataset for 3D scene understanding.

**Step 1: Send Batch Jobs**

```bash
cd MultiNet/scripts/eval_vlm
python send_batch_jobs_vlm_single_ds.py \
  --data_root_dir < Path to the root dataset directory containing the sqa3d directory > \
  --dataset_family sqa3d \
  --dataset_name sqa3d \
  --model < gpt model name > \
  --metadata_dir < path to save batch info > \
  --batch_size < batch size >
```

**Step 2: Run Evaluation**

```bash
python run_batch_eval_vlm.py \
  --batch_job_info_path < path to batch info json file > \
  --results_path < path to save results json >
```

---

### 4. RoboVQA (Robot Visual Question Answering)

RoboVQA is a visual question answering dataset for robotics scenarios.

**Step 1: Send Batch Jobs**

```bash
cd MultiNet/scripts/eval_vlm
python send_batch_jobs_vlm_single_ds.py \
  --data_root_dir < Path to the root dataset directory containing the openx_multi_embodiment directory > \
  --dataset_family robot_vqa \
  --dataset_name robot_vqa \
  --model < gpt model name > \
  --metadata_dir < path to save batch info > \
  --batch_size < batch size >
```

**Step 2: Run Evaluation**

```bash
python run_batch_eval_vlm.py \
  --batch_job_info_path < path to batch info json file > \
  --results_path < path to save results json >
```

---

### 5. Overcooked

Overcooked is a multi-agent cooperative game environment for evaluating action prediction.

**Step 1: Send Batch Jobs**

```bash
cd MultiNet/scripts/eval_vlm
python send_batch_jobs_vlm_single_ds.py \
  --data_root_dir < Path to the root dataset directory containing the overcooked directory > \
  --dataset_family overcooked_ai \
  --dataset_name overcooked_ai \
  --model < gpt model name > \
  --metadata_dir < path to save batch info > \
  --batch_size < batch size >
```

**Step 2: Run Evaluation**

```bash
python run_batch_eval_vlm.py \
  --batch_job_info_path < path to batch info json file > \
  --results_path < path to save results json >
```

---

### 6. OpenX (Open X-Embodiment)

OpenX is a large-scale robotics dataset with multiple subdatasets across different robot embodiments.

**Step 1: Send Batch Jobs**

```bash
cd MultiNet/scripts/eval_vlm
python send_batch_jobs_vlm_single_ds.py \
  --data_root_dir < Path to the root dataset directory containing the openx directories > \
  --dataset_family openx \
  --dataset_name < specific OpenX subdataset name > \
  --model < gpt model name > \
  --metadata_dir < path to save batch info > \
  --batch_size < batch size >
```

**Parameters:**
- `--data_root_dir`: **Required** - Path to the root dataset directory containing the openx directories
- `--dataset_family`: **Required** - Set to `openx`
- `--dataset_name`: **Required** - Specific OpenX subdataset name
  - Acceptable values:
    - `openx_mobile_manipulation`
    - `openx_single_arm`
    - `openx_bimanual`
    - `openx_wheeled_robot`
    - `open_quadrupedal`
- `--model`: **Required** - GPT model name (see `src/config.json`)
- `--metadata_dir`: **Required** - Directory to save batch information
- `--batch_size`: Batch size for evaluation (default: 1)
- `--k_shots`: Number of few-shot examples (default: 0)

**Step 2: Run Evaluation**

```bash
python run_batch_eval_vlm.py \
  --batch_job_info_path < path to batch info json file > \
  --results_path < path to save results json >
```

---

## Notes

- **OpenAI API Key:** You will be prompted to enter your OpenAI API key when running both the batch submission and evaluation scripts
- **Batch Processing Time:** OpenAI batch jobs may take time to complete. You should wait for all batch jobs to finish before running the evaluation script
- **Batch Info Files:** The batch submission script generates a JSON file with a timestamp. Use this file path for the `--batch_job_info_path` argument in the evaluation script
- **Model Names:** Available model names are defined in `src/config.json`. The model we use for Multinet V1 is  `gpt-5-2025-08-07` 
- **Confirmation:** The script will ask for confirmation before sending batch jobs to ensure you want to proceed with the API calls