import os

# Optional: paste a token here for quick runs, or set HF_TOKEN in your shell / `huggingface-cli login`.
_HF_TOKEN_FOR_THIS_SCRIPT = ""
if _HF_TOKEN_FOR_THIS_SCRIPT:
    os.environ["HF_TOKEN"] = _HF_TOKEN_FOR_THIS_SCRIPT

from nlu_benchmark.runner import EpisodeRunner
from nlu_benchmark.agents import HuggingFaceLLMAgent, HFLLMConfig

runner = EpisodeRunner.from_json("nlu_benchmark/sample mazes/V02_winding_corridor.json")

# Uses HFLLMConfig defaults (small Qwen on HF Router). Override model=... if needed.
agent = HuggingFaceLLMAgent(config=HFLLMConfig())

result = runner.run(agent)
print(result["success"])
