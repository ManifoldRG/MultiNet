#!/usr/bin/env python3
"""
Quick visual verification script for local Ollama vision models.
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
    return image, backend.get_mission_text()


def encode_png(image) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ask_ollama(*, model: str, base_url: str, prompt: str, image) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encode_png(image)],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 256},
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick Ollama vision check on a rendered task image.")
    root = Path(__file__).resolve().parent
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument(
        "--task",
        default=str(root / "mazes" / "validation_10" / "V01_empty_room.json"),
        help="Task JSON path",
    )
    parser.add_argument("--output-image", default=None, help="Optional path to save the rendered image")
    args = parser.parse_args()

    image, mission = render_task_image(args.task)
    if args.output_image:
        Image.fromarray(image).save(args.output_image)

    prompt = (
        "Describe this image.\n"
        "You must explicitly say:\n"
        "1. whether you can see the blue agent\n"
        "2. which direction the blue triangle is pointing\n"
        "3. whether you can see the green square goal\n"
        "4. where the green goal is relative to the blue agent\n"
        "5. whether there are walls immediately in front of the agent\n"
        "6. whether you can see any key, and if so what color it is\n"
        "7. whether you can see any door, and if so what color it is\n\n"
        f"Task context: {mission}"
    )

    response = ask_ollama(
        model=args.model,
        base_url=args.base_url,
        prompt=prompt,
        image=image,
    )
    print(response)


if __name__ == "__main__":
    main()
