# MultiNet v1.1 - How to Run

## Prerequisites

```bash
cd src/v1_1

# Activate your environment (conda or venv)
conda activate multinet  # or source .venv/bin/activate

# Install core dependencies
pip install gymnasium minigrid numpy matplotlib pygame
```

---

## 1. Run the Test Suite

```bash
cd src/v1_1

# All tests (133 tests, excludes flaky perf tests)
python -m pytest tests/ -v --ignore=tests/test_performance.py

# Specific test files
python -m pytest tests/test_teleporters.py -v          # Teleporter mechanics
python -m pytest tests/test_exotic_tilings.py -v       # Archimedean tilings
python -m pytest tests/test_model_interface.py -v       # Model interface + NL + cross-domain
python -m pytest tests/test_tiling_generation.py -v     # Core tiling tests
```

---

## 2. Validate All Tasks (Beatable Path Check)

Proves every task JSON has a valid solution using BFS:

```bash
cd src/v1_1

python -c "
import sys, os
_sd = os.path.abspath('.')
if _sd in sys.path: sys.path.remove(_sd)
try: import gymnasium
except ImportError: pass
for k in [k for k in sys.modules if k == 'minigrid' or k.startswith('minigrid.')]: del sys.modules[k]
sys.path.insert(0, _sd)
from gridworld.task_validator import validate_all_tasks
validate_all_tasks()
"
```

Expected output: `16/16 tasks beatable`

---

## 3. Play Tasks Interactively (Pygame)

Play any task with keyboard controls:

```bash
cd src/v1_1

# Default (tier1 simple maze)
python play_task.py

# Specific task file
python play_task.py gridworld/tasks/tier3/gates_switches_002.json

# With trajectory recording
python play_task.py gridworld/tasks/tier5/teleporter_004.json --record
```

**Controls:**
| Key | Action |
|-----|--------|
| Up / W | Move forward |
| Left / A | Turn left |
| Right / D | Turn right |
| Space | Pick up item |
| X | Drop item |
| T / E | Toggle (doors, switches) |
| Backspace | Wait (no-op) |
| R | Reset current task |
| 1-5 | Switch to tier N |
| [ / ] | Previous / next task in tier |
| Q / Escape | Quit |

---

## 4. Visualize Tilings

Generate PNG images of all supported tilings:

```bash
cd src/v1_1

# All 5 tilings (square, hex, triangle, 3-4-6-4, 4-8-8)
python visualize_all_tilings.py

# Original 3 tilings only
python visualize_grids_proper.py
```

---

## 5. Run Model Evaluation

### Backend/Frontend Selection

```bash
cd src/v1_1

# Default: MiniGrid backend + discrete actions
python run_eval.py --model random --tier all

# MultiGrid backend with hexagonal tiling
python run_eval.py --model random --tier 1 --backend multigrid --tiling hex

# Natural language action mode (model outputs text commands)
python run_eval.py --model ollama --ollama-model qwen2.5vl:7b --tier 1 --action-mode nl
```

### Random Baseline

```bash
cd src/v1_1

# Evaluate random agent on all tiers
python run_eval.py --model random --tier all

# Single tier
python run_eval.py --model random --tier 1

# Range of tiers
python run_eval.py --model random --tier 1-3

# Save results to file
python run_eval.py --model random --tier all --output results/random_baseline.json
```

### Ollama VLM (e.g., Qwen2.5-VL-7B)

```bash
# First: install and start Ollama, pull a vision model
ollama pull qwen2.5vl:7b

# Run evaluation
python run_eval.py --model ollama --ollama-model qwen2.5vl:7b --tier 1

# Or use a different model
python run_eval.py --model ollama --ollama-model llava:7b --tier 1-3
```

### LM Studio VLM

```bash
# Start LM Studio with a vision model loaded

python run_eval.py --model lmstudio --lmstudio-model local-model --tier 1
```

### File-Based Protocol (Any External Model)

```bash
# The file-based protocol writes observations to a directory
# and waits for action responses. See model_interface.py FileBasedModelInterface.

python run_eval.py --model file_based --tier 1
```

---

## 6. VLM Vision Sanity Check

Verify that a VLM can see and identify objects in the gridworld before running action evaluation:

```bash
cd src/v1_1

# Run sanity check with Ollama VLM
python vlm_sanity_check.py --model ollama --ollama-model qwen2.5vl:7b

# Specific task
python vlm_sanity_check.py --model ollama --ollama-model qwen2.5vl:7b --task gridworld/tasks/tier3/key_switch_001.json

# All tiers (one representative task per tier)
python vlm_sanity_check.py --model ollama --ollama-model qwen2.5vl:7b --all-tiers --output results/sanity_check.json
```

Tests two categories:
- **Object Identification**: Can the VLM identify agents, goals, keys, doors, switches, hazards?
- **Spatial Reasoning**: Can the VLM describe grid dimensions, agent direction, relative positions?

---

## 7. Manual Web-Chat Smoke Tests

Use this when you want to drive ChatGPT, Claude, or Gemini through the normal web UI instead of the API.

```bash
cd src/v1_1

# One action per chat turn with short visual history
python chat_smoke_test.py \
  --task mazes/validation_10/V01_empty_room.json \
  --query-interval 1 \
  --history-images 2

# Multi-action turns plus optional LOOK
python chat_smoke_test.py \
  --task mazes/validation_10/V04_single_key.json \
  --query-interval 3 \
  --allow-look \
  --history-images 2 \
  --history-text-window 4
```

Each turn writes a packet directory under `/tmp/chat_smoke_<timestamp>/` containing:
- `current.png`
- optional `prior_*.png`
- `prompt.txt`
- `user_message.md`
- `state.json`

Attach the images in the packet to the chat UI, paste `user_message.md`, then paste the model's reply back into the terminal.

---

## 8. Partial Observability

Some tier 5 tasks use partial observability. Two modes are supported:

| Mode | Description | Example Task |
|------|------------|--------------|
| `full` | Agent sees entire grid (default) | All tier 1-4 tasks |
| `view_cone` | Agent sees only a cone in front (walls block vision) | `tier5/hidden_switch_001.json` |
| `fog_of_war` | Grid starts invisible, revealed as explored | `tier5/memory_003.json` |

Set in task JSON under `rules.observability`:
```json
{
  "rules": {
    "observability": "view_cone",
    "view_size": 5
  }
}
```

---

## 9. Task Structure

Tasks are organized by difficulty tier in `gridworld/tasks/`:

```
gridworld/tasks/
  tier1/   Pure navigation (maze solving)
    maze_simple_001.json
    maze_corridor_002.json
    maze_rooms_003.json
  tier2/   Key-door puzzles
    single_key_001.json
    multi_key_002.json
    colored_doors_003.json
  tier3/   Switches and gates
    key_switch_001.json
    gates_switches_002.json
    complex_deps_003.json
  tier4/   Pushable blocks and resource management
    push_block_001.json
    blocked_path_002.json
    consumable_003.json
  tier5/   Inference, multi-mechanism, teleporters
    hidden_switch_001.json
    infer_color_002.json
    memory_003.json
    teleporter_004.json
```

---

## 10. MultiGrid Tilings

Supported tiling types for the MultiGrid backend:

| Tiling | Directions | Description |
|--------|-----------|-------------|
| `square` | 4 (N,E,S,W) | Standard grid |
| `hex` | 6 (N,NE,SE,S,SW,NW) | Hexagonal grid |
| `triangle` | 3 (edge_0, edge_1, edge_2) | Triangular subdivision of hexagons |
| `3464` | up to 6 | Rhombitrihexagonal (mixed triangles, squares, hexagons) |
| `488` | up to 8 | Truncated square (octagons and squares) |

---

## 10. Architecture Summary

```
Task JSON  -->  TaskParser  -->  CustomMiniGridEnv
                                      |
                               MiniGridBackend (square grids)
                               MultiGridBackend (exotic tilings)
                                      |
                               GridRunner (episode execution)
                                      |
                               EvaluationHarness + ModelInterface
                                      |
                    Adapters: Pi0 | Magma | PaliGemma | Ollama | LMStudio
```

**Frontends (action interfaces):**
- Discrete action space (standard MiniGrid: 7 actions)
- Natural language commands (`nl_domain/nl_env.py`)
- Cross-domain canonical spec (`cross_domain/`)

---

## 11. Import Workaround

The local directory was renamed from `minigrid/` to `gridworld/` to avoid conflicts with the installed gymnasium `minigrid` package. No import workaround is needed -- just import directly:

```python
from gridworld.task_spec import TaskSpecification
from gridworld.backends import MiniGridBackend
```
