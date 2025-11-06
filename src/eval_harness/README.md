# MultiNet Submission Guide

This guide provides instructions for preparing and testing your model with the MultiNet evaluation harness, and submitting it for official benchmark evaluation.

## Table of Contents
- [Overview](#overview)
- [Creating Your Model Adapter](#creating-your-model-adapter)
- [Observation Format by Dataset](#observation-format-by-dataset)
- [Required Output Format](#required-output-format)
- [Configuration](#configuration)
- [Running Evaluations](#running-evaluations-local-testing)
- [Results and Troubleshooting](#results-and-troubleshooting)
- [Submitting for Official Evaluation](#submitting-for-official-evaluation-and-leaderboard)

## Overview

The MultiNet evaluation harness provides a standardized interface for evaluating vision-language models (VLMs), vision-language-action (VLA) models, and any other generalist models across diverse datasets. The harness:

1. Loads datasets and provides standardized observations to your model
2. Calls your model adapter's prediction methods
3. Validates outputs and computes metrics
4. Saves local results for verification; official benchmark results are produced by the MultiNet team

All evaluations run in Docker containers to ensure reproducibility.

## Creating Your Model Adapter

### Step 1: Inherit from ModelAdapter

Your model adapter must inherit from the base `ModelAdapter` class in `src/eval_harness/model_adapter.py`:

```python
from typing import List
from src.eval_harness.model_adapter import ModelAdapter

class MyModelAdapter(ModelAdapter):
    def __init__(self, model_name_or_path: str = "path/to/model", **kwargs):
        super().__init__()
        self.model_name = "my_model"  # Optional: for debugging/info
        self.model_type = "multiple_choice"  # Optional: for debugging/info
        self.model_name_or_path = model_name_or_path

    @property
    def supported_datasets(self) -> List[str]:
        """Return list of datasets this adapter supports."""
        return ["piqa", "odinw"]
```

**Important:** The evaluation script calls `adapter_class()` with no arguments, so your `__init__` method must have no required parameters. All parameters should be optional keyword arguments with default values.

### Step 2: Implement Required Methods

Your adapter must implement **at least one** of these methods:

#### `predict_action()` - For single predictions
```python
def predict_action(
    self,
    observation: Dict[str, Any],
    instruction: Optional[str] = None,
    dataset_name: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Predict action for a single observation.
    
    Args:
        observation: Dict with keys like 'image_observation', 'text_observation', etc.
        instruction: Task instruction or question
        dataset_name: Name of the dataset being evaluated
        history: Optional conversation history for multi-turn tasks (BFCL)
                 Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        
    Returns:
        Dict with:
            - "raw_output": str (raw model output text)
            - "extracted_outputs": int/str/np.ndarray/List (depends on task type)
    """
```

#### `batch_predict_actions()` - For batch predictions
```python
def batch_predict_actions(
    self,
    observations: List[Dict[str, Any]],
    instructions: Optional[List[str]] = None,
    dataset_name: Optional[str] = None,
    histories: Optional[List[List[Dict[str, str]]]] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Predict actions for a batch of observations.
    
    Returns:
        List of prediction dicts (same format as predict_action)
    """
```

**Note:** You can implement both methods for flexibility. Use `harness_dataset_config.txt` to specify which mode to use per dataset.

### Step 3: Initialize Your Model

Implement the `initialize()` method to load your model:

```python
def initialize(self, device: str = "cuda", seed: int = 42, **kwargs) -> None:
    """
    Load model weights and set up for inference.
    """
    self.set_seed(seed)
    # Load your model here
    self.model = YourModel.load(...)
    self._is_initialized = True
```

### Example Adapters

Complete working examples are provided in `src/eval_harness/adapters/magma/`:

- **`magma_mcq_adapter.py`** - Multiple choice (PIQA, ODinW)
- **`magma_vqa_adapter.py`** - Visual QA (SQA3D, RoboVQA)
- **`magma_overcooked_adapter.py`** - Discrete actions (Overcooked)
- **`magma_openx_adapter.py`** - Continuous actions (OpenX single-arm, bimanual, wheeled, mobile)
- **`magma_openx_quadrupedal_adapter.py`** - Continuous actions (OpenX quadrupedal)
- **`magma_bfcl_adapter.py`** - Multi-turn function calling (BFCL)

**Note:** One adapter can support multiple datasets with similar formats. For example:
- `magma_mcq_adapter.py` handles both PIQA and ODinW
- `magma_vqa_adapter.py` handles both SQA3D and RoboVQA (same input/output format)
- `magma_openx_adapter.py` handles 4 different OpenX morphologies

## Observation Format by Dataset

The evaluation harness (`scripts/eval_harness/evaluate.py`) provides standardized observations to your adapter. Here are the observation keys for each dataset:

### OpenX Datasets
**Datasets:** `openx_single_arm`, `openx_bimanual`, `openx_wheeled_robot`, `openx_quadrupedal`, `openx_mobile_manipulation`

```python
observation = {
    'image_observation': np.ndarray,  # RGB image (H, W, 3)
    'text_observation': str,          # Environment description
    'action_stats': Dict              # Action normalization statistics
}
instruction = "Task-specific instruction string"
```

### PIQA
**Dataset:** `piqa`

```python
observation = {
    'options': List[str]  # List of answer choices
}
instruction = "Goal: [goal text]\nChoose the better solution:\n0. [sol1]\n1. [sol2]"
```

### SQA3D
**Dataset:** `sqa3d`

```python
observation = {
    'image_observation': np.ndarray  # RGB scene image (H, W, 3)
}
instruction = "Question about the scene"
```

**Note:** Sample data for SQA3D is not provided in this repository, as the dataset is not publicly available. An adapter that works for RoboVQA should be most of the way there for SQA3D as well, as both datasets have similar input and output formats.

### RoboVQA
**Dataset:** `robot_vqa`

```python
observation = {
    'image_observation': np.ndarray  # RGB image (H, W, 3)
}
instruction = "Question about the robot or scene"
```

### ODinW
**Dataset:** `odinw`

```python
observation = {
    'image_observation': np.ndarray,  # RGB image (H, W, 3)
    'options': List[str]              # List of class names
}
instruction = "Which category best describes this image?\nOptions:\n0. [class1]\n1. [class2]\n..."
```

### Overcooked
**Dataset:** `overcooked_ai`

```python
observation = {
    'image_observation': np.ndarray,  # Game screenshot (H, W, 3)
    'text_observation': str,          # Action meanings
    'options': List[Tuple[int, int]]  # Available joint actions
}
instruction = "Layout: [layout]\nTime left: [time]s\nTime elapsed: [time]s"
```

### BFCL
**Dataset:** `bfcl`

```python
observation = {
    'text_observation': str  # Persistent context (function definitions, etc.)
}
instruction = "Current user message"
history = [
    {"role": "user", "content": "Previous user message"},
    {"role": "assistant", "content": "Previous assistant response"},
    # ... more turns
]
```

## Required Output Format

Your adapter's `predict_action()` and `batch_predict_actions()` methods **must** return predictions in this format:

```python
{
    "raw_output": str,           # Raw text output from your model
    "extracted_outputs": <type>  # Extracted answer (type varies by dataset)
}
```

### Extracted Output Types by Dataset

| Dataset(s) | Type | Description | Example |
|-----------|------|-------------|---------|
| PIQA | `int` | Choice index (0 to num_choices-1) | `0` |
| ODinW | `int` | Class index (0 to num_classes-1) | `2` |
| SQA3D, RoboVQA | `str` | Answer text | `"There are three chairs"` |
| Overcooked | `int` | Joint action index (0-35) | `14` |
| OpenX (all) | `np.ndarray` | Action vector (shape varies by morphology) | `np.array([0.1, -0.3, 0.5, ...])` |
| BFCL | `List[Dict]` | Function calls with parameters | `[{"name": "get_weather", "arguments": {"city": "Boston"}}]` |

### Example Return Values

**Multiple Choice (PIQA, ODinW):**
```python
return {
    "raw_output": "Based on the image, I would choose option 1 because...",
    "extracted_outputs": 1
}
```

**Text Generation (SQA3D, RoboVQA):**
```python
return {
    "raw_output": "The answer to the question is: three red objects",
    "extracted_outputs": "three red objects"
}
```

**Continuous Actions (OpenX):**
```python
return {
    "raw_output": "Action tokens: [0.15, -0.23, 0.44, 0.12, -0.08, 0.91, 1]",
    "extracted_outputs": np.array([0.15, -0.23, 0.44, 0.12, -0.08, 0.91, 1.0])
}
```

**Function Calling (BFCL):**
```python
return {
    "raw_output": "the functions I'd call are get_weather(city='Boston', units='celsius') and x(y=z)",
    "extracted_outputs": ['get_weather(city='Boston', units='celsius')', 'x(y=z)']
}
```

## Configuration

### Step 1: Configure `harness_dataset_config.txt`

Edit the configuration file at the project root:

```ini
# Global settings
models_dir=path/to/your/adapters
data_dir=src/eval_harness/sample_data

# Dataset-specific settings
# Format: dataset_name.key=value

piqa.adapter_module=my_model_adapter.py
piqa.batch_process=true
piqa.batch_size=4

odinw.adapter_module=my_model_adapter.py
odinw.batch_process=false
odinw.batch_size=1

openx_single_arm.adapter_module=my_openx_adapter.py
openx_single_arm.batch_process=false
openx_single_arm.batch_size=1
```

**Configuration Options:**
- `models_dir`: Directory containing your adapter Python files
- `data_dir`: Directory containing evaluation data
- `{dataset}.adapter_module`: Python file name of your adapter
- `{dataset}.batch_process`: Whether to use batch processing (`true`/`false`)
- `{dataset}.batch_size`: Batch size for processing

**Tip:** One adapter can handle multiple datasets. See the Magma example config: `src/eval_harness/adapters/magma/harness_dataset_config.txt`

### Step 2: Configure `Dockerfile`

Edit the `Dockerfile` at the project root to set up your model's environment. You can modify any part of the Dockerfile to ensure your adapter runs correctly:

```dockerfile
# Lines 20-25: Designated area for adapter requirements
#-------------------------------------------------------------------
# Install specific requirements for adapter
# Replace with your own requirements
COPY path/to/your_requirements.txt .
RUN pip install --no-cache-dir -r your_requirements.txt
#-------------------------------------------------------------------
```

**You can make changes anywhere in the Dockerfile**, including:
- Installing system packages (`apt-get install`, etc.)
- Setting environment variables (`ENV PATH=...`)
- Modifying system paths
- Installing additional dependencies
- Changing base images or Python versions

The goal is to create a container that successfully runs your adapter. Lines 20-25 are provided as a convenient starting point, but feel free to modify any section as needed.

**Example:** See `src/eval_harness/adapters/magma/Dockerfile` for a complete working example.

## Running Evaluations

### Build and Run

Once your adapter and configuration are ready, run on the provided sample datasets:

```bash
./build_and_run_eval_container.sh DATASET_NAME
```

**Examples:**
```bash
./build_and_run_eval_container.sh piqa
./build_and_run_eval_container.sh openx_single_arm
./build_and_run_eval_container.sh robot_vqa
```

**Note:** Sample data is provided for most datasets in `src/eval_harness/sample_data/`, except for SQA3D which is not publicly available without the authors' permission.

### What Happens During Evaluation

The script builds a Docker image with your dependencies, mounts your adapter and sample data, runs predictions, validates outputs, computes metrics, and saves results to `./eval_results/`

## Results and Troubleshooting

### Finding Your Results

Results are saved to `./eval_results/`:

```
eval_results/
├── {dataset_name}_{timestamp}_results.json
├── {dataset_name}_{timestamp}_metrics.json
└── {dataset_name}_{timestamp}_predictions.json
```

### Common Issues

**1. Adapter Not Found**
```
Error: Model adapter 'my_adapter.py' not found in 'path/to/adapters'
```
- Verify `models_dir` path in `harness_dataset_config.txt`
- Ensure adapter file exists at that location

**2. Wrong Output Format**
```
ValueError: Expected prediction to be a dict with 'raw_output' and 'extracted_outputs'
```
- Check your return format matches the required structure
- See [Required Output Format](#required-output-format)

**3. Wrong Output Type**
```
Expected extracted_outputs to be <int> for dataset piqa, got <str>
```
- Verify `extracted_outputs` type matches your dataset
- See [Extracted Output Types by Dataset](#extracted-output-types-by-dataset)

**4. Batch Processing Error**
```
TypeError: batch_predict_actions() missing required argument
```
- If `batch_process=true` in config, implement `batch_predict_actions()`
- Or set `batch_process=false` to use `predict_action()` instead

**5. Missing Dependencies**
```
ModuleNotFoundError: No module named 'your_package'
```
- Add missing packages to your requirements file
- Update Dockerfile to install them (modify the Dockerfile as needed)

### Validation Checklist

Before running evaluations, verify your adapter:

- [ ] Inherits from `ModelAdapter` with `supported_datasets` property
- [ ] Has `__init__` with **no required arguments** (only optional kwargs)
- [ ] Implements `predict_action()` and/or `batch_predict_actions()`
- [ ] Returns dict with `"raw_output"` and `"extracted_outputs"` in correct type
- [ ] Is configured in `harness_dataset_config.txt` with correct paths
- [ ] Has dependencies installed via `Dockerfile`

For reference, see example adapters in `src/eval_harness/adapters/magma/`

## Submitting for Official Evaluation and Leaderboard

After successfully testing your adapters on the sample data, submit your model for official evaluation. The MultiNet team will run your containerized adapter(s) on the full benchmark datasets and publish results to the leaderboard.

### Submission Process

1. **Test on sample data**: Verify your adapters work correctly as described in [Running Evaluations](#running-evaluations-local-testing)
   - Review logs and local results in `./eval_results/`
   - Fix any issues before submission

2. **Fork the repository**: Create a fork of the MultiNet repository to your GitHub account
   - Go to [https://github.com/ManifoldRG/MultiNet](https://github.com/ManifoldRG/MultiNet)
   - Click "Fork" in the top right

3. **Prepare your submission**: Organize your submission in your fork
   
   **Your adapter directory**:
   ```
   src/eval_harness/adapters/your_model_name/
   ├── your_adapter.py              # Your model adapter(s)
   ├── requirements.txt              # Your model's dependencies
   ├── README.md                     # Brief model description (name, type, supported datasets)
   └── results/                      # (optional) Sample run outputs from local tests
       ├── piqa_results.json
       └── ...
   ```
   
   **Top-level files** (required for containerized evaluation):
   ```
   MultiNet/
   ├── Dockerfile                    # Your edited Dockerfile with dependencies
   └── harness_dataset_config.txt    # Your configuration with adapter settings
   ```

4. **Open a Pull Request**:
   - Push your changes to your fork
   - Open a PR to the main MultiNet repository
   - Title: "Model Submission: [Your Model Name]"
   - Description: Model overview, supported datasets, and any relevant documentation

5. **Review process**: Our team will build your container and run it on the full benchmark datasets. Upon completion, we will add the official results to the leaderboard.

