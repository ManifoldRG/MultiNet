#!/usr/bin/env python3
"""
Maze-shape perception probe for local Ollama vision models.

Renders a task image, sends it to an Ollama vision model, and asks the model
to describe the wall layout and overall maze shape without planning actions.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import urllib.request
from pathlib import Path

from PIL import Image

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification


def render_task_image(task_path: str):
    spec = TaskSpecification.from_json(task_path)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    image, _, _ = backend.reset(seed=spec.seed)
    return image


def encode_png(image) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ask_ollama(*, model: str, base_url: str, prompt: str, image, max_tokens: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encode_png(image)],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("response", "")


def build_prompt() -> str:
    return (
        "Look at this maze image. A wall is a solid black barrier tile that blocks movement "
        "and forms the corridor boundaries.\n"
        "Do not talk about solving the maze. Only describe the walls you can see and the "
        "overall shape they make.\n\n"
        "Answer these questions:\n"
        "1. What are the walls in this image?\n"
        "2. Describe the overall maze shape made by the walls.\n"
        "3. Are there long horizontal hallways, vertical corridors, turns, or loop-backs?\n"
        "4. Give a short summary of the maze layout from the full image.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe an Ollama vision model for maze-shape perception.")
    root = Path(__file__).resolve().parent
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument(
        "--task",
        default=str(root / "mazes" / "validation_10" / "V02_winding_corridor.json"),
        help="Task JSON path",
    )
    parser.add_argument("--output-image", default=None, help="Optional path to save the rendered image")
    parser.add_argument("--max-tokens", type=int, default=400, help="Max generated tokens")
    args = parser.parse_args()

    image = render_task_image(args.task)
    if args.output_image:
        Image.fromarray(image).save(args.output_image)

    response = ask_ollama(
        model=args.model,
        base_url=args.base_url,
        prompt=build_prompt(),
        image=image,
        max_tokens=args.max_tokens,
    )
    print(response)


if __name__ == "__main__":
    main()
