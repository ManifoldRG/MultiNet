"""
LMStudio VLM Adapter for MultiNet v1.1

Uses the OpenAI-compatible chat/completions endpoint provided by LMStudio.
Also works with any OpenAI-compatible vision API.

Usage:
    adapter = LMStudioVLMAdapter(model="qwen2.5-vl-7b")
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


class LMStudioVLMAdapter(ModelInterface):
    """
    Model adapter using the OpenAI-compatible API (LMStudio, vLLM, etc.).

    Sends image via data URL in chat completions format.
    """

    def __init__(
        self,
        model: str = "qwen2.5-vl-7b",
        base_url: str = "http://localhost:1234",
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return f"lmstudio_{self.model}"

    def predict(self, input: ModelInput) -> ModelOutput:
        # Encode image as base64 data URL
        img = Image.fromarray(input.image)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{img_b64}"

        # Build prompt
        action_lines = "\n".join(
            f"  {aid}: {aname}" for aid, aname in sorted(input.action_space.items())
        )
        text_prompt = (
            f"You are controlling an agent in a gridworld environment.\n"
            f"Mission: {input.text_prompt}\n"
            f"Step: {input.step_number}/{input.max_steps}\n\n"
            f"Available actions:\n{action_lines}\n\n"
            f"Look at the image showing the current state of the environment. "
            f"The agent is the blue triangle. The goal is the green square.\n\n"
            f"Choose the best action to accomplish the mission. "
            f"Respond with ONLY the action number (0-6) on the first line."
        )

        # OpenAI-compatible chat completions payload
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            raw_output = result["choices"][0]["message"]["content"]
            action, confidence, reasoning = self._parse_response(raw_output, input.action_space)

            return ModelOutput(
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                raw_output=raw_output,
            )

        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, KeyError) as e:
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

        first_line = text.split("\n")[0].strip()
        match = re.search(r"\b([0-6])\b", first_line)
        if match:
            action = int(match.group(1))
            if action in valid_actions:
                reasoning = text[match.end():].strip() or None
                return action, None, reasoning

        matches = re.findall(r"\b([0-6])\b", text)
        if matches:
            action = int(matches[0])
            if action in valid_actions:
                return action, None, text

        text_lower = text.lower()
        for aid, aname in action_space.items():
            if aname.lower() in text_lower:
                return aid, None, text

        return 6, 0.0, f"Could not parse action from: {text[:200]}"
