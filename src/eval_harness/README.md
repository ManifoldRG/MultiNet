# MultiNet Evaluation Harness

This guide provides complete instructions for evaluating your model on the MultiNet benchmark using our standardized evaluation harness.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Creating Your Model Adapter](#creating-your-model-adapter)
- [Observation Format by Dataset](#observation-format-by-dataset)
- [Required Output Format](#required-output-format)
- [Configuration](#configuration)
- [Running Evaluations](#running-evaluations)
- [Results and Troubleshooting](#results-and-troubleshooting)

## Overview

The MultiNet evaluation harness provides a standardized interface for evaluating vision-language-action models across diverse datasets. The harness:

1. Loads datasets and provides standardized observations to your model
2. Calls your model adapter's prediction methods
3. Validates outputs and computes metrics
4. Saves results for leaderboard submission

All evaluations run in Docker containers to ensure reproducibility.

## Quick Start

**3-Step Process:**
1. Create model adapter(s) inheriting from `model_adapter.py`
2. Configure `harness_dataset_config.txt` and `Dockerfile`
3. Run `./build_and_run_eval_container.sh DATASET_NAME`

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

**Output Formats by Dataset:**

| Dataset Category | Datasets | Output Type |
|-----------------|----------|-------------|
| Multiple Choice | PIQA | `int` (choice index) |
| Text Generation | SQA3D, RoboVQA | `str` (answer text) |
| Classification | ODinW | `int` (class index) |
| Discrete Action | Overcooked | `int` (action index) |
| Continuous Action | OpenX (all variants) | `np.ndarray` (action vector) |
| Tool Use | BFCL | `List[Dict]` (function calls) |

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

**Note:** Notice that one adapter can support multiple datasets. For example:
- `magma_mcq_adapter.py` handles both PIQA and ODinW
- `magma_vqa_adapter.py` handles both SQA3D and RoboVQA
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

Edit the `Dockerfile` at the project root to install your model's dependencies:

```dockerfile
# Lines 20-25: Add your model's requirements
#-------------------------------------------------------------------
# Install specific requirements for adapter
# Replace with your own requirements
COPY path/to/your_requirements.txt .
RUN pip install --no-cache-dir -r your_requirements.txt
#-------------------------------------------------------------------
```

**Example:** See `src/eval_harness/adapters/magma/Dockerfile` for a complete working example.

## Running Evaluations

### Build and Run

Once your adapter and configuration are ready, run:

```bash
./build_and_run_eval_container.sh DATASET_NAME
```

**Examples:**
```bash
./build_and_run_eval_container.sh piqa
./build_and_run_eval_container.sh openx_single_arm
./build_and_run_eval_container.sh sqa3d
```

### What Happens During Evaluation

1. **Build Phase:**
   - Docker image is built with your dependencies
   - Your adapter code is copied into the container

2. **Evaluation Phase:**
   - Dataset is loaded from `/data` mount
   - Your adapter is initialized
   - Predictions are generated (batch or single mode)
   - Outputs are validated
   - Metrics are computed
   - Results are saved to `/results` mount

3. **Volume Mounts:**
   - `/models` → Your adapter directory
   - `/data` → Dataset directory
   - `/results` → Output directory (host: `./eval_results`)

### Available Datasets

- `piqa` - Physical commonsense reasoning
- `odinw` - Object detection/classification
- `sqa3d` - 3D scene question answering
- `robot_vqa` - Robot visual question answering
- `overcooked_ai` - Multi-agent coordination
- `bfcl` - Multi-turn function calling
- `openx_single_arm` - Single-arm manipulation
- `openx_bimanual` - Bimanual manipulation
- `openx_wheeled_robot` - Wheeled robot navigation
- `openx_quadrupedal` - Quadrupedal locomotion
- `openx_mobile_manipulation` - Mobile manipulation

## Results and Troubleshooting

### Finding Your Results

Evaluation results are saved to `./eval_results/` at the project root:

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
- Update Dockerfile to install them (lines 20-25)

### Validation Checklist

Before running evaluations, verify:

- [ ] Adapter inherits from base `ModelAdapter` class
- [ ] `supported_datasets` property implemented (returns `List[str]`)
- [ ] `super().__init__()` called with no arguments in your adapter's `__init__`
- [ ] `__init__` method has **no required arguments** (only optional kwargs with defaults allowed)
- [ ] At least one prediction method implemented (`predict_action` or `batch_predict_actions`)
- [ ] Output format includes both `"raw_output"` (str) and `"extracted_outputs"` (correct type)
- [ ] `harness_dataset_config.txt` configured with correct paths and settings
- [ ] `Dockerfile` updated to install your model's dependencies
- [ ] Adapter file exists in specified `models_dir`

### Getting Help

- Review example adapters in `src/eval_harness/adapters/magma/`
- Check the base adapter interface in `src/eval_harness/model_adapter.py`
- Examine the evaluation script: `scripts/eval_harness/evaluate.py`

## Submitting to the Leaderboard

After successfully testing your adapters on the sample data and running full evaluations, submit to the MultiNet leaderboard:

### Submission Process

1. **Test on sample data**: Verify your adapters work correctly using the sample data in `src/eval_harness/sample_data/`
   ```bash
   ./build_and_run_eval_container.sh piqa  # Test with sample data first
   ```

2. **Run full evaluations**: Run evaluations on all relevant datasets for your model type
   - Results will be saved to `./eval_results/`
   - Review metrics and verify correctness

3. **Fork the repository**: Create a fork of the MultiNet repository to your GitHub account
   - Go to [https://github.com/ManifoldRG/MultiNet](https://github.com/ManifoldRG/MultiNet)
   - Click "Fork" in the top right

4. **Prepare your submission**: Organize your submission in your fork
   
   **Top-level directory** (required for `build_and_run_eval_container.sh`):
   ```
   MultiNet/
   ├── Dockerfile                    # Your edited Dockerfile
   ├── harness_dataset_config.txt    # Your configuration
   └── build_and_run_eval_container.sh  # (unchanged)
   ```
   
   **Your adapter directory**:
   ```
   src/eval_harness/adapters/your_model_name/
   ├── your_adapter.py              # Your model adapter(s)
   ├── requirements.txt              # Your model's dependencies
   ├── README.md                     # Brief model description
   └── results/                      # Your evaluation results
       ├── piqa_results.json
       ├── sqa3d_results.json
       └── ...
   ```

5. **Include in your submission**:
   - **Top-level**: Modified `Dockerfile` and `harness_dataset_config.txt` (required for build script)
   - **Adapter directory**: All model adapter Python files
   - **Requirements**: Your model's dependencies (`requirements.txt`)
   - **Results**: Evaluation results from `eval_results/` directory
   - **Documentation**: A brief README describing:
     - Model name and type
     - Supported datasets
     - Any special setup or requirements
     - Link to model weights/checkpoint (if public)

6. **Open a Pull Request**:
   - Push your changes to your fork
   - Open a PR to the main MultiNet repository
   - Title: "Model Submission: [Your Model Name]"
   - Description should include:
     - Model overview
     - Datasets evaluated
     - Key results/metrics
     - Any relevant paper/documentation links

7. **Review process**: Our team will review your submission and add results to the leaderboard

