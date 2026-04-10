#!/usr/bin/env python3
"""
MultiNet v1.1 Evaluation CLI

Evaluate models on either the legacy tier directories or the authored
benchmark sets, starting with `validation_10`.

Usage:
    python run_eval.py --model random --benchmark validation_10
    python run_eval.py --model random --benchmark validation_10 --backend multigrid --tiling square
    python run_eval.py --model random --benchmark tiers --tier 1
    python run_eval.py --model ollama --ollama-model qwen2.5vl:7b --benchmark validation_10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def parse_tiers(tier_str: str) -> list[int]:
    """Parse tier specification: 'all', '1', '1-3', '2,4,5'."""
    if tier_str.lower() == "all":
        return [1, 2, 3, 4, 5]
    if "-" in tier_str:
        start, end = tier_str.split("-")
        return list(range(int(start), int(end) + 1))
    if "," in tier_str:
        return [int(t.strip()) for t in tier_str.split(",")]
    return [int(tier_str)]


def load_model(args) -> "ModelInterface":
    """Load model based on CLI arguments."""
    from model_interface import ModelInterface, RandomModelInterface, FileBasedModelInterface

    model_name = args.model.lower()

    if model_name == "random":
        return RandomModelInterface(seed=args.seed)

    elif model_name == "file_based":
        if not args.work_dir:
            raise ValueError("--work-dir required for file_based model")
        model = FileBasedModelInterface(work_dir=args.work_dir, timeout=args.timeout)
        model.setup()
        return model

    elif model_name == "ollama":
        from adapters.ollama_vlm_adapter import OllamaVLMAdapter
        model = OllamaVLMAdapter(
            model=args.ollama_model or "qwen2.5vl:7b",
            base_url=args.ollama_url or "http://localhost:11434",
            timeout=args.ollama_timeout,
            request_retries=args.ollama_retries,
            retry_sleep=args.ollama_retry_sleep,
        )
        return model

    elif model_name == "lmstudio":
        from adapters.lmstudio_vlm_adapter import LMStudioVLMAdapter
        model = LMStudioVLMAdapter(
            model=args.lmstudio_model or "google/gemma-3-4b-it",
            base_url=args.lmstudio_url or "http://localhost:1234",
        )
        model.setup()
        return model

    elif model_name == "pi0":
        # Pi0 adapter lives in the OpenPI profiling scripts directory (src/eval/profiling/openpi/scripts/)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval" / "profiling" / "openpi" / "scripts"))
        from minigrid_inference import Pi0MiniGridAdapter
        model = Pi0MiniGridAdapter()
        model.setup(device=args.device)
        return model

    elif model_name == "magma":
        # Magma adapter lives in the v1 Magma module scripts (src/v1/modules/Magma/scripts/)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1" / "modules" / "Magma" / "scripts"))
        from magma_minigrid_inference import MagmaMiniGridAdapter
        model = MagmaMiniGridAdapter()
        model.setup(device=args.device)
        return model

    elif model_name == "paligemma":
        from adapters.paligemma_adapter import PaliGemmaMiniGridAdapter
        model = PaliGemmaMiniGridAdapter()
        model.setup(device=args.device)
        return model

    else:
        raise ValueError(f"Unknown model: {model_name}. Options: random, file_based, ollama, lmstudio, pi0, magma, paligemma")


def main():
    parser = argparse.ArgumentParser(description="MultiNet v1.1 Evaluation CLI")
    parser.add_argument("--model", required=True,
                        help="Model to evaluate: random, file_based, ollama, lmstudio, pi0, magma, paligemma")
    parser.add_argument("--benchmark", default="validation_10",
                        choices=["validation_10", "tiers", "directory"],
                        help="Benchmark mode: validation_10, legacy tiers, or every JSON in --task-dir")
    parser.add_argument("--tier", default="all",
                        help="Tier(s) to evaluate: 'all', '1', '1-3', '2,4,5'")
    parser.add_argument("--backend", default="minigrid",
                        choices=["minigrid", "multigrid"],
                        help="Grid backend: minigrid (square) or multigrid (exotic tilings)")
    parser.add_argument("--tiling", default="square",
                        help="Tiling type for multigrid backend (default: square)")
    parser.add_argument("--action-mode", default="discrete",
                        choices=["discrete", "nl"],
                        help="Action mode: discrete (int actions) or nl (natural language)")
    parser.add_argument("--device", default="cpu",
                        help="Device for model inference (default: cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--task-dir", default=None,
                        help="Task directory (default: gridworld/tasks relative to this file)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path for results")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print step-by-step info")
    parser.add_argument("--history-images", type=int, default=2,
                        help="Number of prior frames to include in model input (default: 2)")
    parser.add_argument("--history-text", action=argparse.BooleanOptionalAction, default=True,
                        help="Include rolling text summaries of prior steps (default: enabled)")
    parser.add_argument("--history-text-window", type=int, default=3,
                        help="Number of prior text summary lines to include when text history is enabled (default: 3)")

    # Model-specific args
    parser.add_argument("--ollama-model", default=None,
                        help="Ollama model name (default: qwen2.5vl:7b)")
    parser.add_argument("--ollama-url", default=None,
                        help="Ollama API base URL")
    parser.add_argument("--ollama-timeout", type=int, default=600,
                        help="Per-request timeout in seconds for Ollama responses (default: 600)")
    parser.add_argument("--ollama-retries", type=int, default=1,
                        help="How many times to retry a timed-out/failed Ollama request (default: 1)")
    parser.add_argument("--ollama-retry-sleep", type=float, default=5.0,
                        help="Seconds to wait between Ollama retries (default: 5.0)")
    parser.add_argument("--lmstudio-model", default=None,
                        help="LM Studio model id (default: google/gemma-3-4b-it)")
    parser.add_argument("--lmstudio-url", default=None,
                        help="LM Studio OpenAI-compatible base URL")
    parser.add_argument("--work-dir", default=None,
                        help="Working directory for file_based model")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Timeout for file_based model (seconds)")

    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if args.task_dir is None:
        if args.benchmark == "validation_10":
            task_dir = str(root / "mazes" / "validation_10")
        else:
            task_dir = str(root / "gridworld" / "tasks")
    else:
        task_dir = args.task_dir

    tiers = parse_tiers(args.tier)

    print(f"Model: {args.model}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Backend: {args.backend}" + (f" ({args.tiling})" if args.backend == "multigrid" else ""))
    print(f"Action mode: {args.action_mode}")
    print(f"Task dir: {task_dir}")
    print(f"Device: {args.device}")
    print(f"History images: {args.history_images}")
    print(f"History text: {args.history_text}")
    if args.history_text:
        print(f"History text window: {args.history_text_window}")
    if args.benchmark == "tiers":
        print(f"Tiers: {tiers}")
    print()

    # Load model
    model = load_model(args)
    print(f"Loaded model: {model.model_name}")

    # Create backend
    from gridworld.backends import get_backend
    if args.backend == "multigrid":
        backend = get_backend("multigrid", tiling=args.tiling, render_mode="rgb_array")
    else:
        backend = get_backend("minigrid", render_mode="rgb_array")

    # Run evaluation
    from evaluation_harness import EvaluationHarness
    harness = EvaluationHarness(
        model,
        backend=backend,
        history_images=args.history_images,
        history_text=args.history_text,
        history_text_window=args.history_text_window,
    )

    try:
        if args.benchmark == "tiers":
            result = harness.evaluate_all(
                task_dir=task_dir,
                tiers=tiers,
                verbose=args.verbose,
            )

            print("\n" + "=" * 60)
            print(f"RESULTS: {result.model_name}")
            print("=" * 60)

            for tier, metrics in sorted(result.tier_metrics.items()):
                print(f"\nTier {tier}:")
                print(f"  Tasks: {metrics.num_tasks}")
                print(f"  Success: {metrics.num_success}/{metrics.num_tasks} ({metrics.success_rate:.1%})")
                print(f"  Avg Steps: {metrics.avg_steps:.1f}")
                print(f"  Avg Reward: {metrics.avg_reward:.3f}")

                for episode in metrics.results:
                    status = "PASS" if episode.success else "FAIL"
                    print(f"    [{status}] {episode.task_id}: steps={episode.steps_taken}, reward={episode.total_reward:.3f}")

            print(f"\nOverall:")
            print(f"  Success Rate: {result.overall_success_rate:.1%}")
            print(f"  Avg Steps: {result.overall_avg_steps:.1f}")
            print(f"  Avg Reward: {result.overall_avg_reward:.3f}")
        else:
            benchmark_name = args.benchmark if args.benchmark != "directory" else Path(task_dir).name
            result = harness.evaluate_task_dir(
                task_dir=task_dir,
                benchmark_name=benchmark_name,
                verbose=args.verbose,
            )

            print("\n" + "=" * 60)
            print(f"BENCHMARK: {result.benchmark_name}")
            print(f"MODEL: {result.model_name}")
            print("=" * 60)

            for task_result in result.task_results:
                status = "PASS" if task_result.success else "FAIL"
                ratio = f"{task_result.optimality_ratio:.2f}" if task_result.optimality_ratio is not None else "n/a"
                print(
                    f"[{status}] {task_result.task_id}: "
                    f"steps={task_result.steps_taken}, optimal={task_result.optimal_steps}, "
                    f"ratio={ratio}, points={task_result.points_earned:.2f}/{task_result.available_points:.2f}"
                )

            print("\nSummary:")
            print(f"  Tasks: {result.num_tasks}")
            print(f"  Success: {result.num_success}/{result.num_tasks} ({result.success_rate:.1%})")
            print(f"  Points: {result.total_points_earned:.2f}/{result.total_available_points:.2f} ({result.point_rate:.1%})")
            print(f"  Avg Optimality Ratio: {result.avg_optimality_ratio:.2f}")

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(task_dir).parent / "results" / f"{model.model_name}_{args.benchmark}_results.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

        result.save(str(output_path))
        print(f"\nResults saved to {output_path}")

    finally:
        harness.close()


if __name__ == "__main__":
    main()
