from nlu_benchmark.runner import EpisodeRunner
from nlu_benchmark.agents import LocalTransformersAgent, LocalLLMConfig

runner = EpisodeRunner.from_json("nlu_benchmark/sample mazes/V02_winding_corridor.json")

# Small local model (no HF inference credits required).
agent = LocalTransformersAgent(
    config=LocalLLMConfig(
        model="HuggingFaceTB/SmolLM2-360M-Instruct",
        max_new_tokens=16,
    )
)

result = runner.run(agent)
print(result["success"])
