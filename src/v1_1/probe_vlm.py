#!/usr/bin/env python3
"""
Small VLM probe CLI for MiniGrid v1.1.

Use this to smoke-test local vision models before running full evaluation.
It supports:
  - orientation probes: ask the model what direction the agent faces
  - action probes: ask the normal action adapter for a single next action
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from model_interface import ModelInput
from gridworld.actions import ACTION_NAMES
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification


DIR_NAMES = {0: "right", 1: "down", 2: "left", 3: "up"}


@dataclass
class ProbeContext:
    task_path: str
    task_id: str
    mission: str
    current_image: np.ndarray
    prior_images: list[np.ndarray]
    current_direction: int
    current_direction_name: str
    current_position: tuple[int, int]
    action_sequence: list[int]
    action_names: list[str]
    text_memory: str | None


def parse_action_sequence(raw: str | None) -> list[int]:
    """Parse a comma-separated action sequence."""
    if not raw:
        return []

    actions = []
    for piece in raw.split(","):
        token = piece.strip()
        if not token:
            continue
        action = int(token)
        if action not in ACTION_NAMES:
            raise ValueError(f"Invalid action id {action}; expected 0-6.")
        actions.append(action)
    return actions


def _build_text_memory(states: list[tuple[int, tuple[int, int], int]], actions: list[int]) -> str | None:
    if not actions:
        return None

    lines = []
    for index, action in enumerate(actions):
        direction, position, next_direction = states[index][0], states[index][1], states[index + 1][0]
        lines.append(
            f"step {index + 1}: action={ACTION_NAMES[action]}, "
            f"started_facing={DIR_NAMES[direction]}, "
            f"ended_facing={DIR_NAMES[next_direction]}, "
            f"position={position}"
        )
    return "\n".join(lines)


def collect_probe_context(
    task_path: str,
    actions: list[int],
    history_images: int = 0,
    include_text_history: bool = False,
) -> ProbeContext:
    """Reset a task, apply an action prefix, and collect current/prior frames."""
    spec = TaskSpecification.from_json(task_path)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)

    try:
        obs, state, _ = backend.reset(seed=spec.seed)
        mission = backend.get_mission_text()

        frames = [obs.copy()]
        states = [(state.agent_direction, state.agent_position, state.agent_direction)]

        for action in actions:
            obs, _, terminated, truncated, state, _ = backend.step(action)
            frames.append(obs.copy())
            states.append((state.agent_direction, state.agent_position, state.agent_direction))
            if terminated or truncated:
                break

        current_image = frames[-1]
        prior_images = [frame.copy() for frame in frames[:-1][-history_images:]]
        text_memory = _build_text_memory(states, actions[: len(states) - 1]) if include_text_history else None

        return ProbeContext(
            task_path=task_path,
            task_id=spec.task_id,
            mission=mission,
            current_image=current_image,
            prior_images=prior_images,
            current_direction=state.agent_direction,
            current_direction_name=DIR_NAMES[state.agent_direction],
            current_position=state.agent_position,
            action_sequence=actions[: len(states) - 1],
            action_names=[ACTION_NAMES[action] for action in actions[: len(states) - 1]],
            text_memory=text_memory,
        )
    finally:
        backend.close()


def _encode_png(image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ask_lmstudio(
    *,
    model: str,
    base_url: str,
    prompt: str,
    current_image: np.ndarray,
    prior_images: list[np.ndarray],
) -> str:
    content = [{"type": "text", "text": prompt}]
    for index, prior in enumerate(prior_images, start=1):
        content.append({"type": "text", "text": f"Previous image {index} (earlier timestep)."})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_encode_png(prior)}"},
        })
    content.append({"type": "text", "text": "Current image."})
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{_encode_png(current_image)}"},
    })

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


def ask_ollama(
    *,
    model: str,
    base_url: str,
    prompt: str,
    current_image: np.ndarray,
    prior_images: list[np.ndarray],
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [_encode_png(image) for image in [*prior_images, current_image]],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 256},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("response", "")


def run_orientation_probe(
    *,
    provider: str,
    model_name: str,
    base_url: str,
    context: ProbeContext,
) -> dict:
    prompt = (
        "You are inspecting MiniGrid images.\n"
        "The blue triangle is the agent.\n"
        "If previous images are present, they are earlier timesteps only.\n"
        "Answer only for the current image.\n"
        "Question: Which direction is the blue triangle pointing in the current image? "
        "Respond with exactly one word: up, down, left, or right."
    )
    if context.text_memory:
        prompt += f"\n\nRecent text memory:\n{context.text_memory}"

    if provider == "lmstudio":
        answer = ask_lmstudio(
            model=model_name,
            base_url=base_url,
            prompt=prompt,
            current_image=context.current_image,
            prior_images=context.prior_images,
        )
    else:
        answer = ask_ollama(
            model=model_name,
            base_url=base_url,
            prompt=prompt,
            current_image=context.current_image,
            prior_images=context.prior_images,
        )

    return {
        "probe_type": "orientation",
        "task_id": context.task_id,
        "task_path": context.task_path,
        "action_sequence": context.action_sequence,
        "action_names": context.action_names,
        "expected_direction": context.current_direction_name,
        "actual_direction_id": context.current_direction,
        "model_answer": answer.strip(),
        "used_prior_images": len(context.prior_images),
        "used_text_memory": bool(context.text_memory),
    }


def load_action_model(provider: str, model_name: str, base_url: str):
    if provider == "lmstudio":
        from adapters.lmstudio_vlm_adapter import LMStudioVLMAdapter

        model = LMStudioVLMAdapter(model=model_name, base_url=base_url)
        model.setup()
        return model

    from adapters.ollama_vlm_adapter import OllamaVLMAdapter

    return OllamaVLMAdapter(model=model_name, base_url=base_url)


def run_action_probe(
    *,
    provider: str,
    model_name: str,
    base_url: str,
    context: ProbeContext,
) -> dict:
    model = load_action_model(provider, model_name, base_url)
    try:
        output = model.predict(
            ModelInput(
                image=context.current_image,
                text_prompt=context.mission,
                action_space=ACTION_NAMES,
                step_number=len(context.action_sequence) + 1,
                max_steps=100,
                additional_context=context.text_memory,
                prior_images=context.prior_images,
            )
        )
    finally:
        model.teardown()

    return {
        "probe_type": "action",
        "task_id": context.task_id,
        "task_path": context.task_path,
        "mission": context.mission,
        "action_sequence": context.action_sequence,
        "action_names": context.action_names,
        "current_direction": context.current_direction_name,
        "current_position": list(context.current_position),
        "predicted_action": output.action,
        "predicted_action_name": ACTION_NAMES.get(output.action, str(output.action)),
        "reasoning": output.reasoning,
        "raw_output": output.raw_output,
        "used_prior_images": len(context.prior_images),
        "used_text_memory": bool(context.text_memory),
    }


def save_probe_images(context: ProbeContext, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(context.prior_images, start=1):
        Image.fromarray(image).save(out / f"prior_{index}.png")
    Image.fromarray(context.current_image).save(out / "current.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe local vision models on MiniGrid v1.1.")
    parser.add_argument("--probe", choices=["orientation", "action"], required=True)
    parser.add_argument("--model", choices=["lmstudio", "ollama"], required=True)
    parser.add_argument("--task", default=None, help="Task JSON path. Defaults to validation_10/V01.")
    parser.add_argument("--actions", default="", help="Comma-separated action prefix, e.g. '1,2,2'.")
    parser.add_argument("--history-images", type=int, default=0, help="How many prior frames to include.")
    parser.add_argument("--history-text", action="store_true", help="Include text summaries of prior steps.")
    parser.add_argument("--save-images-dir", default=None, help="Optional directory to save the probe frames.")
    parser.add_argument("--output", default=None, help="Optional JSON file for probe results.")
    parser.add_argument("--lmstudio-model", default="qwen/qwen3-vl-8b")
    parser.add_argument("--lmstudio-url", default="http://localhost:1234")
    parser.add_argument("--ollama-model", default="qwen2.5vl:7b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    task_path = args.task or str(root / "mazes" / "validation_10" / "V01_empty_room.json")
    actions = parse_action_sequence(args.actions)
    context = collect_probe_context(
        task_path=task_path,
        actions=actions,
        history_images=args.history_images,
        include_text_history=args.history_text,
    )

    if args.save_images_dir:
        save_probe_images(context, args.save_images_dir)

    provider = args.model
    model_name = args.lmstudio_model if provider == "lmstudio" else args.ollama_model
    base_url = args.lmstudio_url if provider == "lmstudio" else args.ollama_url

    try:
        if args.probe == "orientation":
            result = run_orientation_probe(
                provider=provider,
                model_name=model_name,
                base_url=base_url,
                context=context,
            )
        else:
            result = run_action_probe(
                provider=provider,
                model_name=model_name,
                base_url=base_url,
                context=context,
            )
    except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, ConnectionError) as exc:
        result = {
            "probe_type": args.probe,
            "task_id": context.task_id,
            "task_path": context.task_path,
            "error": str(exc),
            "used_prior_images": len(context.prior_images),
            "used_text_memory": bool(context.text_memory),
        }

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
