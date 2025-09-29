# Magma: Multimodal Agentic Models

Evaluating Magma on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).


#### LIBERO Setup
Clone and install LIBERO and other requirements:
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -r agents/libero/requirements.txt
cd LIBERO
pip install -e .
```

#### Quick Evaluation
The following code demonstrates how to run Magma on a single LIBERO task and evaluate its performance:
```
import numpy as np
from libero.libero import benchmark
from libero_env_utils import get_libero_env, get_libero_dummy_action, get_libero_obs, get_max_steps, save_rollout_video
from libero_magma_utils import get_magma_model, get_magma_prompt, get_magma_action

# Set up benchmark and task
benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_goal" # or libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()
task_id = 1
task = task_suite.get_task(task_id)

# Initialize environment
env, task_description = get_libero_env(task, resolution=256)
print(f"Task {task_id} description: {task_description}")

# Load MAGMA model
model_name = "microsoft/magma-8b-libero-goal"  # or your local path
processor, magma = get_magma_model(model_name)
prompt = get_magma_prompt(task_description, processor, magma.config)

# Run evaluation
num_steps_wait = 10
max_steps = get_max_steps(task_suite_name)

env.seed(0)
obs = env.reset()
init_states = task_suite.get_task_init_states(task_id) 
obs = env.set_init_state(init_states[0])

step = 0
replay_images = []
while step < max_steps + num_steps_wait:
    if step < num_steps_wait:
        obs, _, done, _ = env.step(get_libero_dummy_action())
        step += 1
        continue
    
    img = get_libero_obs(obs, resize_size=256)
    replay_images.append(img)
    action = get_magma_action(magma, processor, img, prompt, task_suite_name)
    obs, _, done, _ = env.step(action.tolist())
    step += 1

env.close()
save_rollout_video(replay_images, success=done, task_description=task_description)
```
**Notes:** The above script only tests one episode on a single task and visualizes MAGMA's trajectory with saved video. For comprehensive evaluation on each task suite, please use `eval_magma_libero.py`.
```
python eval_magma_libero.py \
  --model_name microsoft/Magma-8B-libero-object \
  --task_suite_name libero_object \

python eval_magma_libero.py \
  --model_name microsoft/Magma-8B-libero-spatial \
  --task_suite_name libero_spatial \

python eval_magma_libero.py \
  --model_name microsoft/Magma-8B-libero-goal \
  --task_suite_name libero_goal \
```

