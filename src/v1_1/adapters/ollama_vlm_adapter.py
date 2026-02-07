"""
Ollama VLM Adapter for MultiNet v1.1

Connects to a local Ollama server to use open-source VLMs for MiniGrid evaluation.
Recommended model: qwen2.5vl:7b (best accuracy in the 7B VLM class).
Fallback options: llava:7b, llava:13b, minicpm-v.

Usage:
    adapter = OllamaVLMAdapter(model="qwen2.5vl:7b")
    output = adapter.predict(model_input)
"""

from __future__ import annotations

import base64
import io
import json
import re
import urllib.request
import urllib.error

import numpy as np
from PIL import Image

from ..model_interface import ModelInterface, ModelInput, ModelOutput


class OllamaVLMAdapter(ModelInterface):
    """
    Model adapter that connects to a local Ollama server for VLM inference.

    Sends image as base64 + text prompt, receives generated text, parses action.
    Works with any Ollama vision model (qwen2.5vl, llava, minicpm-v, etc.).
    """

    def __init__(
        self,
        model: str = "qwen2.5vl:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return f"ollama_{self.model}"

    def predict(self, input: ModelInput) -> ModelOutput:
        # Encode image as base64
        img = Image.fromarray(input.image)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Build prompt with action space description
        action_lines = "\n".join(
            f"  {aid}: {aname}" for aid, aname in sorted(input.action_space.items())
        )
        prompt = (
            f"You are controlling an agent in a gridworld environment.\n"
            f"Mission: {input.text_prompt}\n"
            f"Step: {input.step_number}/{input.max_steps}\n\n"
            f"Available actions:\n{action_lines}\n\n"
            f"Look at the image showing the current state of the environment. "
            f"The agent is the blue triangle. The goal is the green square.\n\n"
            f"Choose the best action to accomplish the mission. "
            f"Respond with ONLY the action number (0-6) on the first line, "
            f"then optionally explain your reasoning."
        )

        if input.additional_context:
            prompt += f"\n\nAdditional context: {input.additional_context}"

        # Call Ollama API
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            raw_output = result.get("response", "")
            action, confidence, reasoning = self._parse_response(raw_output, input.action_space)

            return ModelOutput(
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                raw_output=raw_output,
            )

        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError) as e:
            # Fallback to wait action if API is unreachable
            return ModelOutput(
                action=6,
                confidence=0.0,
                reasoning=f"API error: {e}",
                raw_output=str(e),
            )

    def _parse_response(
        self, text: str, action_space: dict[int, str]
    ) -> tuple[int, float | None, str | None]:
        """Parse action from model response text."""
        valid_actions = set(action_space.keys())
        text = text.strip()

        # Try to find a bare integer on the first line
        first_line = text.split("\n")[0].strip()
        match = re.search(r"\b([0-6])\b", first_line)
        if match:
            action = int(match.group(1))
            if action in valid_actions:
                reasoning = text[match.end():].strip() or None
                return action, None, reasoning

        # Try to find any integer in the full text
        matches = re.findall(r"\b([0-6])\b", text)
        if matches:
            action = int(matches[0])
            if action in valid_actions:
                return action, None, text

        # Try matching action names
        text_lower = text.lower()
        for aid, aname in action_space.items():
            if aname.lower() in text_lower:
                return aid, None, text

        # Fallback: wait
        return 6, 0.0, f"Could not parse action from: {text[:200]}"
