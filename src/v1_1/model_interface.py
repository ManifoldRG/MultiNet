"""
Standard Model Interface for MultiNet v1.1

Defines the abstract interface all model adapters must implement,
plus built-in baselines (random, file-based).
"""

from __future__ import annotations

import json
import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelInput:
    """Input to a model for action prediction."""
    image: np.ndarray              # (H, W, 3) uint8 RGB observation
    text_prompt: str               # Mission/task description
    action_space: dict[int, str]   # {action_id: action_name}
    step_number: int
    max_steps: int
    additional_context: str | None = None
    prior_images: list[np.ndarray] | None = None


@dataclass
class ModelOutput:
    """Output from a model prediction."""
    action: int                    # Predicted action ID
    confidence: float | None = None
    reasoning: str | None = None
    raw_output: str | None = None


class ModelInterface(ABC):
    """
    Abstract base class for all model adapters.

    Implementations must provide:
    - model_name property
    - predict() method

    Optional overrides:
    - predict_batch() for batched inference
    - setup() / teardown() for resource management
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Unique identifier for this model."""
        ...

    @property
    def supports_batched(self) -> bool:
        """Whether this model supports batched prediction."""
        return False

    @abstractmethod
    def predict(self, input: ModelInput) -> ModelOutput:
        """
        Predict the next action given an observation.

        Args:
            input: ModelInput with image, text prompt, and action space

        Returns:
            ModelOutput with predicted action
        """
        ...

    def predict_batch(self, inputs: list[ModelInput]) -> list[ModelOutput]:
        """
        Predict actions for a batch of observations.

        Default implementation loops over inputs. Override for efficiency.
        """
        return [self.predict(inp) for inp in inputs]

    def setup(self, device: str = "cpu") -> None:
        """
        Initialize model resources (load weights, etc.).

        Called once before evaluation begins. Override if needed.
        """
        pass

    def teardown(self) -> None:
        """
        Release model resources.

        Called after evaluation completes. Override if needed.
        """
        pass


class RandomModelInterface(ModelInterface):
    """Built-in random baseline that selects actions uniformly at random."""

    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)

    @property
    def model_name(self) -> str:
        return "random"

    def predict(self, input: ModelInput) -> ModelOutput:
        action_ids = list(input.action_space.keys())
        action = self._rng.choice(action_ids)
        return ModelOutput(
            action=int(action),
            confidence=1.0 / len(action_ids),
            reasoning="Random selection",
        )


class FileBasedModelInterface(ModelInterface):
    """
    File-based model protocol for external process integration.

    Writes observations to {work_dir}/input/step_N.json + step_N.png,
    waits for {work_dir}/output/step_N.json with {"action": int}.
    This enables external testers to use any language/framework.
    """

    def __init__(self, work_dir: str, timeout: float = 60.0, poll_interval: float = 0.1):
        self.work_dir = Path(work_dir)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.input_dir = self.work_dir / "input"
        self.output_dir = self.work_dir / "output"

    @property
    def model_name(self) -> str:
        return "file_based"

    def setup(self, device: str = "cpu") -> None:
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def predict(self, input: ModelInput) -> ModelOutput:
        step = input.step_number

        # Write input image as PNG
        from PIL import Image
        img = Image.fromarray(input.image)
        img.save(self.input_dir / f"step_{step}.png")

        # Write input metadata as JSON
        input_data = {
            "step_number": step,
            "max_steps": input.max_steps,
            "text_prompt": input.text_prompt,
            "action_space": {str(k): v for k, v in input.action_space.items()},
            "image_path": f"step_{step}.png",
        }
        if input.additional_context:
            input_data["additional_context"] = input.additional_context

        with open(self.input_dir / f"step_{step}.json", "w") as f:
            json.dump(input_data, f, indent=2)

        # Wait for output
        output_path = self.output_dir / f"step_{step}.json"
        start_time = time.time()
        while not output_path.exists():
            if time.time() - start_time > self.timeout:
                raise TimeoutError(
                    f"Timed out waiting for {output_path} after {self.timeout}s"
                )
            time.sleep(self.poll_interval)

        # Read output
        with open(output_path) as f:
            result = json.load(f)

        return ModelOutput(
            action=int(result["action"]),
            confidence=result.get("confidence"),
            reasoning=result.get("reasoning"),
            raw_output=json.dumps(result),
        )

    def teardown(self) -> None:
        pass
