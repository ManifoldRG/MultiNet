from nlu_benchmark.runner import EpisodeRunner
from nlu_benchmark.agents import RandomAgent

runner = EpisodeRunner.from_json("nlu_benchmark/sample mazes/V01_empty_room.json")

agent = RandomAgent()
result = runner.run(agent)

print("Success:", result["success"])
