import json
from pathlib import Path
from nlu_benchmark.env import GridWorldEnv


def load_maze(path) -> GridWorldEnv:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    maze = data["maze"]
    rows, cols = maze["dimensions"]
    walls = {tuple(w) for w in maze["walls"]}
    start = tuple(maze["start"])
    goal = tuple(maze["goal"])
    max_steps = data.get("max_steps", 100)
    mechanisms = data.get("mechanisms", {})
    return GridWorldEnv(
        rows=rows,
        cols=cols,
        walls=walls,
        start=start,
        goal=goal,
        max_steps=max_steps,
        mechanisms=mechanisms,
    )
