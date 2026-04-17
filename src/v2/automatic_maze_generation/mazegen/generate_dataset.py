from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    MazeGenSpec,
    MazeInstance,
    MazeLayout,
    DistractorMode,
)
from .orchestrator import  sample_spec, build_valid_maze_with_retries




def maze_layout_to_payload(layout: MazeLayout, spec: MazeGenSpec, report: dict) -> dict[str, Any]:
    return {
        "task_id": f"{spec.backbone.value}_{spec.logic_chain.value}_{spec.seed}",
        "version": "0.1",
        "seed": spec.seed,
        "difficulty_tier": spec.difficulty_tier,
        "maze": {
            "dimensions": [layout.width, layout.height],
            "walls": [list(w) for w in sorted(layout.walls)],
            "start": list(layout.start),
            "goal": list(layout.goal),
        },
        "mechanisms": {
            "keys": [],
            "doors": [],
            "switches": [],
            "gates": [],
            "blocks": [],
            "teleporters": [],
            "hazards": [],
        },
        "metadata": layout.metadata,
        "validation": {
            "is_valid": report["is_valid"],
            "reasons": report["reasons"],
            "optimal_cost": report["solver_result"]["optimal_cost"],
            "optimal_path": [list(p) for p in report["solver_result"].get("path", [])],

        },
    }


def maze_instance_to_payload(maze: MazeInstance, spec: MazeGenSpec, report: dict) -> dict[str, Any]:
    payload = maze.to_json_like()
    payload["task_id"] = f"{spec.backbone.value}_{spec.logic_chain.value}_{spec.seed}"
    payload["version"] = "0.1"
    payload["seed"] = spec.seed
    payload["difficulty_tier"] = spec.difficulty_tier
    payload["validation"] = {
        "is_valid": report["is_valid"],
        "reasons": report["reasons"],
        "optimal_cost": report["solver_result"]["optimal_cost"],
        "interactions": report["solver_result"].get("interactions", []),
        "optimal_path": [list(p) for p in report["solver_result"].get("path", [])],
    }
    return payload


def to_payload(obj: MazeLayout | MazeInstance, spec: MazeGenSpec, report: dict) -> dict[str, Any]:
    if isinstance(obj, MazeInstance):
        return maze_instance_to_payload(obj, spec, report)
    return maze_layout_to_payload(obj, spec, report)


def main() -> None:
    out_dir = Path("generated_mazes")
    out_dir.mkdir(parents=True, exist_ok=True)

    target_n = 200
    accepted = 0
    seed = 1000
    attempts = 0
    max_attempts = 500

    manifest: list[dict[str, Any]] = []

    while accepted < target_n and attempts < max_attempts:
        attempts += 1
        spec = sample_spec(seed)
        seed += 1

        try:
            obj, report, final_spec = build_valid_maze_with_retries(spec, max_retries=10)
        except Exception as exc:
            print(f"[ERROR] seed={spec.seed} backbone={spec.backbone.value} logic={spec.logic_chain.value}: {exc}")
            continue


        payload = to_payload(obj, final_spec, report)
        distractor_tag = (
            "none"
            if final_spec.distractor_mode == DistractorMode.NONE
            else f"{final_spec.distractor_mode.value}{final_spec.max_distractors}"
        )

        fname = (
            f"maze_{accepted:03d}_"
            f"{final_spec.backbone.value}_"
            f"{final_spec.logic_chain.value}_"
            f"{distractor_tag}_"
            f"{final_spec.seed}.json"
        )
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        manifest.append(
            {
                "file": fname,
                "seed": final_spec.seed,
                "backbone": final_spec.backbone.value,
                "logic_chain": final_spec.logic_chain.value,
                "difficulty_tier": final_spec.difficulty_tier,
                "distractor_mode": final_spec.distractor_mode.value,
                "max_distractors": final_spec.max_distractors,
                "optimal_cost": report["solver_result"]["optimal_cost"],
            }
        )

        accepted += 1
        print(f"[OK] {accepted:02d}/{target_n} saved {fname}")

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nFinished. Accepted {accepted} mazes in {attempts} attempts.")
    print(f"Output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()