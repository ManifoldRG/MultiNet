"""
VLM Vision Sanity Check

Tests whether a VLM can see and understand MiniGrid rendered images.
Two test categories:
  1. Object Identification: Can the VLM identify objects in the scene?
  2. Spatial Reasoning: Can the VLM describe spatial relationships?

This is NOT an action prediction test. It validates that the VLM's visual
encoder correctly perceives the gridworld before we ask it to act.

Usage:
    python vlm_sanity_check.py --model ollama --ollama-model qwen2.5vl:7b
    python vlm_sanity_check.py --model lmstudio --lmstudio-model local-model
"""

from __future__ import annotations

import base64
import io
import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class VisionQuestion:
    """A single vision question about a rendered scene."""
    question: str
    expected_keywords: list[str]  # Keywords the answer should contain
    category: str  # "object_id" or "spatial"
    difficulty: int = 1  # 1-3


@dataclass
class VisionTestResult:
    """Result of a single vision test."""
    question: str
    category: str
    expected_keywords: list[str]
    model_answer: str
    matched_keywords: list[str]
    passed: bool
    error: str | None = None


@dataclass
class SanityCheckReport:
    """Full report from a sanity check run."""
    model_name: str
    task_id: str
    total_questions: int
    passed: int
    failed: int
    object_id_score: float  # 0-1
    spatial_score: float  # 0-1
    results: list[VisionTestResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "task_id": self.task_id,
            "total_questions": self.total_questions,
            "passed": self.passed,
            "failed": self.failed,
            "object_id_score": round(self.object_id_score, 3),
            "spatial_score": round(self.spatial_score, 3),
            "results": [
                {
                    "question": r.question,
                    "category": r.category,
                    "expected_keywords": r.expected_keywords,
                    "model_answer": r.model_answer,
                    "matched_keywords": r.matched_keywords,
                    "passed": r.passed,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


def generate_questions_for_task(task_spec, grid_state) -> list[VisionQuestion]:
    """Generate vision questions based on a task specification and its current state.

    Args:
        task_spec: TaskSpecification for the current task.
        grid_state: GridState from the backend after reset.

    Returns:
        List of VisionQuestion objects.
    """
    questions = []

    # --- Object Identification ---

    # Agent identification
    questions.append(VisionQuestion(
        question="Is there an agent (blue triangle) visible in this image? Describe its appearance.",
        expected_keywords=["agent", "triangle", "blue"],
        category="object_id",
        difficulty=1,
    ))

    # Goal identification
    questions.append(VisionQuestion(
        question="Is there a goal marker (green square) in this image? Where is it located?",
        expected_keywords=["goal", "green"],
        category="object_id",
        difficulty=1,
    ))

    # Wall identification
    if task_spec.maze.walls:
        questions.append(VisionQuestion(
            question="Are there walls (grey barriers) in this gridworld? Describe what you see.",
            expected_keywords=["wall", "grey", "gray", "barrier"],
            category="object_id",
            difficulty=1,
        ))

    # Key identification
    if task_spec.mechanisms.keys:
        key_colors = [k.color for k in task_spec.mechanisms.keys]
        questions.append(VisionQuestion(
            question="Are there any keys visible in the image? What color are they?",
            expected_keywords=["key"] + key_colors,
            category="object_id",
            difficulty=1,
        ))

    # Door identification
    if task_spec.mechanisms.doors:
        door_colors = [d.requires_key for d in task_spec.mechanisms.doors]
        questions.append(VisionQuestion(
            question="Are there any doors visible in the image? What color are they?",
            expected_keywords=["door"] + door_colors,
            category="object_id",
            difficulty=1,
        ))

    # Switch identification
    if task_spec.mechanisms.switches:
        questions.append(VisionQuestion(
            question="Is there a switch or button (yellow ball) in this image?",
            expected_keywords=["switch", "button", "yellow", "ball"],
            category="object_id",
            difficulty=2,
        ))

    # Hazard identification
    if task_spec.mechanisms.hazards:
        questions.append(VisionQuestion(
            question="Are there any hazards (red/orange lava tiles) visible in this image?",
            expected_keywords=["hazard", "lava", "red", "orange", "danger"],
            category="object_id",
            difficulty=2,
        ))

    # --- Spatial Reasoning ---

    # Grid dimensions
    w, h = task_spec.maze.dimensions
    questions.append(VisionQuestion(
        question=f"This is a {w}x{h} gridworld. How many columns and rows do you see?",
        expected_keywords=[str(w), str(h), "grid"],
        category="spatial",
        difficulty=2,
    ))

    # Agent direction
    dir_names = {0: "right", 1: "down", 2: "left", 3: "up"}
    agent_dir = grid_state.agent_direction
    questions.append(VisionQuestion(
        question="Which direction is the agent (blue triangle) facing? (up, down, left, or right)",
        expected_keywords=[dir_names.get(agent_dir, "right")],
        category="spatial",
        difficulty=2,
    ))

    # Goal relative to agent
    ax, ay = grid_state.agent_position
    gx, gy = task_spec.maze.goal.x, task_spec.maze.goal.y
    rel_parts = []
    if gy < ay:
        rel_parts.append("above")
    elif gy > ay:
        rel_parts.append("below")
    if gx > ax:
        rel_parts.append("right")
    elif gx < ax:
        rel_parts.append("left")
    if not rel_parts:
        rel_parts = ["same"]

    questions.append(VisionQuestion(
        question="Where is the goal (green square) relative to the agent (blue triangle)? Is it above, below, left, or right?",
        expected_keywords=rel_parts,
        category="spatial",
        difficulty=2,
    ))

    # Object count
    total_objects = (
        len(task_spec.mechanisms.keys)
        + len(task_spec.mechanisms.doors)
        + len(task_spec.mechanisms.switches)
        + len(task_spec.mechanisms.gates)
        + len(task_spec.mechanisms.blocks)
        + len(task_spec.mechanisms.hazards)
    )
    if total_objects > 0:
        questions.append(VisionQuestion(
            question="How many interactive objects (keys, doors, switches, blocks, hazards) do you see? Give an approximate count.",
            expected_keywords=[str(total_objects)],
            category="spatial",
            difficulty=3,
        ))

    return questions


def check_answer(answer: str, expected_keywords: list[str]) -> tuple[bool, list[str]]:
    """Check if an answer contains expected keywords.

    Uses case-insensitive matching. An answer passes if it matches
    at least one keyword from the list.

    Returns:
        (passed, list of matched keywords)
    """
    answer_lower = answer.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    return len(matched) > 0, matched


def ask_vlm_ollama(
    image: np.ndarray,
    question: str,
    model: str = "qwen2.5vl:7b",
    base_url: str = "http://localhost:11434",
) -> str:
    """Ask a vision question to an Ollama VLM.

    Args:
        image: RGB image array (H, W, 3)
        question: Text question about the image
        model: Ollama model name
        base_url: Ollama server URL

    Returns:
        Model's text response
    """
    if Image is None:
        raise ImportError("PIL (Pillow) required: pip install Pillow")

    img = Image.fromarray(image)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        "You are looking at a rendered gridworld environment from MiniGrid. "
        "The image shows a top-down view of a grid with various objects.\n\n"
        "Common objects:\n"
        "- Agent: blue triangle pointing in its facing direction\n"
        "- Goal: green square\n"
        "- Walls: grey squares\n"
        "- Keys: small colored key shapes\n"
        "- Doors: colored rectangles that block passages\n"
        "- Switches: yellow balls\n"
        "- Hazards: red/orange tiles (lava)\n\n"
        f"Question: {question}\n\n"
        "Answer concisely."
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
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


def ask_vlm_lmstudio(
    image: np.ndarray,
    question: str,
    model: str = "local-model",
    base_url: str = "http://localhost:1234",
) -> str:
    """Ask a vision question to an LM Studio VLM via OpenAI-compatible API."""
    if Image is None:
        raise ImportError("PIL (Pillow) required: pip install Pillow")

    img = Image.fromarray(image)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    system_msg = (
        "You are looking at a rendered gridworld environment from MiniGrid. "
        "Common objects: agent (blue triangle), goal (green square), "
        "walls (grey), keys (colored key shapes), doors (colored rectangles), "
        "switches (yellow balls), hazards (red/orange lava)."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ],
            },
        ],
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


def run_sanity_check(
    task_path: str,
    ask_fn,
    model_name: str = "unknown",
    verbose: bool = True,
) -> SanityCheckReport:
    """Run a full sanity check on a task.

    Args:
        task_path: Path to task JSON file
        ask_fn: Function(image, question) -> str that queries the VLM
        model_name: Name for reporting
        verbose: Print results as they come

    Returns:
        SanityCheckReport with all results
    """
    import sys
    import os

    _sd = os.path.abspath(os.path.dirname(__file__))
    if _sd not in sys.path:
        sys.path.insert(0, _sd)

    from gridworld.task_spec import TaskSpecification
    from gridworld.backends.minigrid_backend import MiniGridBackend

    spec = TaskSpecification.from_json(task_path)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    obs, state, _ = backend.reset(seed=spec.seed)

    questions = generate_questions_for_task(spec, state)
    results = []

    if verbose:
        print(f"\n=== VLM Sanity Check: {spec.task_id} ===")
        print(f"Model: {model_name}")
        print(f"Questions: {len(questions)}")
        print()

    for q in questions:
        try:
            answer = ask_fn(obs, q.question)
            passed, matched = check_answer(answer, q.expected_keywords)
            result = VisionTestResult(
                question=q.question,
                category=q.category,
                expected_keywords=q.expected_keywords,
                model_answer=answer.strip(),
                matched_keywords=matched,
                passed=passed,
            )
        except Exception as e:
            result = VisionTestResult(
                question=q.question,
                category=q.category,
                expected_keywords=q.expected_keywords,
                model_answer="",
                matched_keywords=[],
                passed=False,
                error=str(e),
            )

        results.append(result)

        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] [{q.category}] {q.question}")
            if result.error:
                print(f"  ERROR: {result.error}")
            else:
                print(f"  Answer: {result.model_answer[:120]}...")
                print(f"  Matched: {result.matched_keywords} / Expected: {q.expected_keywords}")
            print()

    # Compute scores
    obj_results = [r for r in results if r.category == "object_id"]
    spatial_results = [r for r in results if r.category == "spatial"]

    obj_score = sum(r.passed for r in obj_results) / max(len(obj_results), 1)
    spatial_score = sum(r.passed for r in spatial_results) / max(len(spatial_results), 1)

    report = SanityCheckReport(
        model_name=model_name,
        task_id=spec.task_id,
        total_questions=len(results),
        passed=sum(r.passed for r in results),
        failed=sum(not r.passed for r in results),
        object_id_score=obj_score,
        spatial_score=spatial_score,
        results=results,
    )

    if verbose:
        print(f"=== Results ===")
        print(f"Total: {report.passed}/{report.total_questions}")
        print(f"Object ID: {report.object_id_score:.0%}")
        print(f"Spatial:   {report.spatial_score:.0%}")

    return report


def run_sanity_check_all_tiers(
    ask_fn,
    model_name: str = "unknown",
    tasks_dir: str = "gridworld/tasks",
    verbose: bool = True,
) -> list[SanityCheckReport]:
    """Run sanity check across representative tasks from each tier.

    Picks one task per tier for efficiency.
    """
    from pathlib import Path
    tasks_path = Path(tasks_dir)
    reports = []

    # Pick one representative task per tier
    representative_tasks = {
        1: "maze_rooms_003.json",      # Walls only
        2: "colored_doors_003.json",    # Keys + doors
        3: "key_switch_001.json",       # Keys + doors + switches + gates
        4: "push_block_001.json",       # Blocks
        5: "memory_003.json",           # Multi-mechanism
    }

    for tier, task_file in sorted(representative_tasks.items()):
        task_path = tasks_path / f"tier{tier}" / task_file
        if not task_path.exists():
            if verbose:
                print(f"[SKIP] Tier {tier}: {task_file} not found")
            continue

        report = run_sanity_check(
            str(task_path), ask_fn, model_name, verbose
        )
        reports.append(report)

    if verbose and reports:
        print(f"\n=== Overall Summary ({model_name}) ===")
        avg_obj = sum(r.object_id_score for r in reports) / len(reports)
        avg_spatial = sum(r.spatial_score for r in reports) / len(reports)
        avg_total = sum(r.passed for r in reports) / sum(r.total_questions for r in reports)
        print(f"Average Object ID:  {avg_obj:.0%}")
        print(f"Average Spatial:    {avg_spatial:.0%}")
        print(f"Average Total:      {avg_total:.0%}")

    return reports


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLM Vision Sanity Check")
    parser.add_argument("--model", choices=["ollama", "lmstudio"], default="ollama")
    parser.add_argument("--ollama-model", default="qwen2.5vl:7b")
    parser.add_argument("--lmstudio-model", default="local-model")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--task", default=None, help="Specific task JSON path")
    parser.add_argument("--all-tiers", action="store_true", help="Run across all tiers")
    parser.add_argument("--output", default=None, help="Save results JSON")
    args = parser.parse_args()

    # Build ask function
    if args.model == "ollama":
        base_url = args.base_url or "http://localhost:11434"
        vlm_model = args.ollama_model
        model_name = f"ollama_{vlm_model}"

        def ask_fn(image, question):
            return ask_vlm_ollama(image, question, model=vlm_model, base_url=base_url)

    elif args.model == "lmstudio":
        base_url = args.base_url or "http://localhost:1234"
        vlm_model = args.lmstudio_model
        model_name = f"lmstudio_{vlm_model}"

        def ask_fn(image, question):
            return ask_vlm_lmstudio(image, question, model=vlm_model, base_url=base_url)

    if args.all_tiers:
        reports = run_sanity_check_all_tiers(ask_fn, model_name)
        if args.output:
            with open(args.output, "w") as f:
                json.dump([r.to_dict() for r in reports], f, indent=2)
            print(f"\nResults saved to {args.output}")
    elif args.task:
        report = run_sanity_check(args.task, ask_fn, model_name)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        # Default: run on a tier 2 task (has keys + doors, good visual variety)
        default_task = "gridworld/tasks/tier2/colored_doors_003.json"
        report = run_sanity_check(default_task, ask_fn, model_name)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"\nResults saved to {args.output}")
