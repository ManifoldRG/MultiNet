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
import time
import urllib.request
import urllib.error

import numpy as np
from PIL import Image

try:
    from ..model_interface import ModelInterface, ModelInput, ModelOutput
except ImportError:
    from model_interface import ModelInterface, ModelInput, ModelOutput


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
        timeout: int = 600,
        request_retries: int = 1,
        retry_sleep: float = 5.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.request_retries = request_retries
        self.retry_sleep = retry_sleep

    @property
    def model_name(self) -> str:
        return f"ollama_{self.model}"

    def predict(self, input: ModelInput) -> ModelOutput:
        messages = self._build_messages(input)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_error: Exception | None = None
        total_attempts = self.request_retries + 1
        for attempt in range(1, total_attempts + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read().decode("utf-8"))

                raw_output = result.get("message", {}).get("content", "")
                action, confidence, reasoning = self._parse_response(raw_output, input.action_space)

                if attempt > 1:
                    retry_note = f"Ollama succeeded on retry {attempt}/{total_attempts}. "
                    reasoning = retry_note + reasoning if reasoning else retry_note.rstrip()

                return ModelOutput(
                    action=action,
                    confidence=confidence,
                    reasoning=reasoning,
                    raw_output=raw_output,
                )
            except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, ConnectionError) as exc:
                last_error = exc
                if attempt >= total_attempts:
                    break
                time.sleep(self.retry_sleep)

        return ModelOutput(
            action=6,
            confidence=0.0,
            reasoning=(
                f"API error after {total_attempts} attempt(s), timeout={self.timeout}s: {last_error}"
            ),
            raw_output=str(last_error),
        )

    def _build_prompt(self, input: ModelInput) -> str:
        return (
            "You are controlling the blue agent from images only.\n"
            "Objective: get to the green square goal.\n"
            f"Current step: {input.step_number}/{input.max_steps}\n\n"
            "You are graded on success and token efficiency.\n"
            "Both input and output tokens matter.\n\n"
            "Choose the next action from the images and the previous action result.\n"
            "Choose the action that best advances a complete route to the goal, not a greedy move toward where you guess the goal is.\n"
            "The correct next action may temporarily move away from the goal in order to follow an open corridor, pick up a key, open a door, or recover from a failed move.\n"
            "Do not assume the goal is visible. When the goal is off-screen, navigate by following open corridors and setting up a route.\n"
            "If the previous action failed, do not repeat the same failed move unless the image clearly changed.\n"
            "If the previous and current images are nearly the same, prefer an action that changes viewpoint or position instead of oscillating in place.\n\n"
            "For your response, provide exactly one line:\n"
            "Action: <action_id>\n\n"
            "Use only one of these action ids:\n"
            "0 turn_left\n"
            "1 turn_right\n"
            "2 move_forward\n"
            "3 pickup\n"
            "4 drop\n"
            "5 toggle\n"
            "6 done"
        )

    def _build_messages(self, input: ModelInput) -> list[dict]:
        messages: list[dict] = [
            {
                "role": "user",
                "content": self._build_prompt(input),
            }
        ]

        previous_image = (input.prior_images or [])[-1] if input.prior_images else None
        previous_action = self._extract_previous_action(input.additional_context)
        latest_result = self._extract_latest_result(input.additional_context)
        if previous_image is not None:
            previous_label = "unknown"
            if previous_action:
                previous_label = previous_action
            messages.append(
                {
                    "role": "user",
                    "content": f"This is the previous image after the action {previous_label} was taken.",
                    "images": [self._encode_image(previous_image)],
                }
            )
        if latest_result:
            messages.append(
                {
                    "role": "user",
                    "content": f"Previous action result: {latest_result}",
                }
            )

        current_content = "This is the current image."
        if input.additional_context:
            current_content += f"\n\nAdditional context:\n{input.additional_context}"
        messages.append(
            {
                "role": "user",
                "content": current_content,
                "images": [self._encode_image(input.image)],
            }
        )

        return messages

    def _encode_image(self, image: np.ndarray) -> str:
        img = Image.fromarray(image)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _extract_previous_action(self, additional_context: str | None) -> str | None:
        if not additional_context:
            return None
        lines = [line.strip() for line in additional_context.splitlines() if line.strip()]
        for line in reversed(lines):
            match = re.search(r"action=([a-z_]+)", line)
            if match:
                return match.group(1)
        return None

    def _extract_latest_result(self, additional_context: str | None) -> str | None:
        if not additional_context:
            return None
        lines = [line.strip() for line in additional_context.splitlines() if line.strip()]
        for line in reversed(lines):
            match = re.search(r"result=(.+?)(?:,\s*position=|$)", line)
            if match:
                return match.group(1).strip()
        return None

    def _parse_response(
        self, text: str, action_space: dict[int, str]
    ) -> tuple[int, float | None, str | None]:
        """Parse action from model response text."""
        valid_actions = set(action_space.keys())
        text = text.strip()

        action_line_match = re.search(r"(?im)^\s*action\s*:\s*([0-6])\s*$", text)
        if action_line_match:
            action = int(action_line_match.group(1))
            if action in valid_actions:
                return action, None, text

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
