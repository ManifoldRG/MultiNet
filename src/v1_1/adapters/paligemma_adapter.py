"""
PaliGemma Adapter for MultiNet v1.1

Uses Google's PaliGemma VLM for MiniGrid evaluation.
Lighter weight than Pi0/Magma, good for quick iteration.

Usage:
    adapter = PaliGemmaMiniGridAdapter()
    adapter.setup(device="cuda:0")
    output = adapter.predict(model_input)
"""

from __future__ import annotations

import re
import numpy as np
from PIL import Image

try:
    from ..model_interface import ModelInterface, ModelInput, ModelOutput
except ImportError:
    from model_interface import ModelInterface, ModelInput, ModelOutput


class PaliGemmaMiniGridAdapter(ModelInterface):
    """
    PaliGemma VLM adapter for MiniGrid evaluation.

    Uses google/paligemma2-3b-pt-896 or google/paligemma-3b-mix-448
    via the transformers library.
    """

    def __init__(self, model_id: str = "google/paligemma2-3b-pt-896", max_new_tokens: int = 32):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self.device = "cpu"

    @property
    def model_name(self) -> str:
        return f"paligemma_{self.model_id.split('/')[-1]}"

    def setup(self, device: str = "cpu") -> None:
        import torch
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        self.device = device
        dtype = torch.bfloat16 if "cuda" in device else torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()

    def predict(self, input: ModelInput) -> ModelOutput:
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError("Call setup() before predict()")

        # Convert observation to PIL image
        img = Image.fromarray(input.image).convert("RGB")

        # Build prompt
        action_lines = ", ".join(
            f"{aid}={aname}" for aid, aname in sorted(input.action_space.items())
        )
        prompt = (
            f"This is a gridworld navigation task. {input.text_prompt} "
            f"Actions: {action_lines}. "
            f"The blue triangle is the agent, green square is the goal. "
            f"Output the best action number (0-6):"
        )

        # Process and generate
        inputs = self.processor(
            text=prompt,
            images=img,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Decode only the generated tokens (skip input)
        input_len = inputs["input_ids"].shape[-1]
        raw_output = self.processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )

        # Parse action
        action, confidence, reasoning = self._parse_response(raw_output, input.action_space)

        return ModelOutput(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            raw_output=raw_output,
        )

    def _parse_response(
        self, text: str, action_space: dict[int, str]
    ) -> tuple[int, float | None, str | None]:
        """Parse action from model response."""
        valid_actions = set(action_space.keys())
        text = text.strip()

        match = re.search(r"\b([0-6])\b", text)
        if match:
            action = int(match.group(1))
            if action in valid_actions:
                return action, None, text

        text_lower = text.lower()
        for aid, aname in action_space.items():
            if aname.lower() in text_lower:
                return aid, None, text

        return 6, 0.0, f"Could not parse: {text[:100]}"

    def teardown(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        # Free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
