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
from typing import Any

import numpy as np
from PIL import Image

try:
    from ..model_interface import ModelInterface, ModelInput, ModelOutput
except ImportError:
    from model_interface import ModelInterface, ModelInput, ModelOutput


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
        min_image_size: int = 1024,
        max_prior_images: int = 2,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_image_size = min_image_size
        self.max_prior_images = max_prior_images

    @property
    def model_name(self) -> str:
        return f"lmstudio_{self.model}"

    def setup(self, device: str = "cpu") -> None:
        """Verify the LM Studio API is reachable and the requested model exists."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/v1/models",
                headers={"Content-Type": "application/json"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError) as e:
            raise RuntimeError(
                f"Could not reach LM Studio at {self.base_url}. "
                f"Start the local server and load a vision-capable model. "
                f"Original error: {e}"
            ) from e

        models = payload.get("data", [])
        model_ids = {item.get("id", "") for item in models if isinstance(item, dict)}
        if model_ids and self.model not in model_ids:
            available = ", ".join(sorted(model_ids))
            raise RuntimeError(
                f"LM Studio is reachable at {self.base_url}, but model '{self.model}' is not loaded. "
                f"Available models: {available}"
            )

    def predict(self, input: ModelInput) -> ModelOutput:
        # Build a compact "standard" prompt: goal + action list + critical
        # movement semantics. The current failure mode on V1 is blindly
        # repeating forward, so the prompt explicitly asks the model to check
        # facing direction and whether forward is actually useful.
        action_lines = "\n".join(
            f"  {aid}: {aname}" for aid, aname in sorted(input.action_space.items())
        )
        text_prompt = (
            "You are controlling a top-down gridworld agent.\n"
            f"Mission: {input.text_prompt}\n"
            f"Step: {input.step_number}/{input.max_steps}\n\n"
            "Visual facts:\n"
            "- The blue triangle is the agent.\n"
            "- The triangle's pointing direction is the agent's current facing direction.\n"
            "- The green square is the goal.\n"
            "- Dark cells or wall tiles block movement.\n\n"
            "Images:\n"
            "- If previous images are shown, they are earlier timesteps for short-term memory only.\n"
            "- The CURRENT image is the last image in the sequence and is the one you should act on.\n\n"
            "Action list:\n"
            f"{action_lines}\n\n"
            "Decision rule:\n"
            "- First decide where the goal is relative to the agent.\n"
            "- Then check whether moving forward would actually move toward the goal or just hit a wall / keep the wrong heading.\n"
            "- If the agent is not facing the right direction, choose a turn action instead of moving forward.\n"
            "- Do not use action 6 unless the task is already complete.\n\n"
            "Respond with exactly one action number from 0 to 6 on the first line.\n"
            "Optionally give a very short reason on the second line."
        )
        prior_images = list(input.prior_images or [])[-self.max_prior_images:]
        attempt_sizes = list(range(len(prior_images), -1, -1))
        errors: list[str] = []

        for prior_count in attempt_sizes:
            try:
                raw_output = self._predict_once(
                    input=input,
                    text_prompt=text_prompt,
                    prior_images=prior_images[-prior_count:] if prior_count else [],
                )
                action, confidence, reasoning = self._parse_response(raw_output, input.action_space)
                if prior_count != len(prior_images):
                    fallback_note = (
                        f"LM Studio fallback: reduced prior images from "
                        f"{len(prior_images)} to {prior_count}."
                    )
                    reasoning = (
                        f"{fallback_note} {reasoning}".strip()
                        if reasoning
                        else fallback_note
                    )

                return ModelOutput(
                    action=action,
                    confidence=confidence,
                    reasoning=reasoning,
                    raw_output=raw_output,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, KeyError) as e:
                error_text = self._format_request_error(e, prior_count)
                errors.append(error_text)
                if prior_count == 0:
                    break

        return ModelOutput(
            action=6,
            confidence=0.0,
            reasoning=f"API error: {errors[-1]}",
            raw_output="\n".join(errors),
        )

    def _predict_once(
        self,
        input: ModelInput,
        text_prompt: str,
        prior_images: list[np.ndarray],
    ) -> str:
        if input.additional_context:
            text_prompt += f"\n\nText memory:\n{input.additional_context}"

        content: list[dict[str, Any]] = [{"type": "text", "text": text_prompt}]
        for idx, prior in enumerate(prior_images, start=1):
            content.append({
                "type": "text",
                "text": f"Previous image {idx} of {len(prior_images)} (older timestep).",
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": self._to_data_url(prior, min_size=max(512, self.min_image_size // 2))},
            })
        content.append({"type": "text", "text": "Current image (act using this image)."})
        content.append({
            "type": "image_url",
            "image_url": {"url": self._to_data_url(input.image, min_size=self.min_image_size)},
        })

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]

    def _prepare_image(self, image: np.ndarray, min_size: int | None = None) -> Image.Image:
        """Upscale small renders so orientation cues stay legible to VLMs."""
        img = Image.fromarray(image).convert("RGB")
        target_min_size = min_size or self.min_image_size
        if min(img.width, img.height) >= target_min_size:
            return img

        scale = max(1, int(np.ceil(target_min_size / min(img.width, img.height))))
        return img.resize(
            (img.width * scale, img.height * scale),
            Image.Resampling.NEAREST,
        )

    def _to_data_url(self, image: np.ndarray, min_size: int | None = None) -> str:
        img = self._prepare_image(image, min_size=min_size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_b64}"

    def _format_request_error(self, error: Exception, prior_count: int) -> str:
        details = f"request failed with {prior_count} prior image(s): {error}"
        if isinstance(error, urllib.error.HTTPError):
            try:
                body = error.read().decode("utf-8", errors="replace").strip()
            except Exception:
                body = ""
            if body:
                details = f"{details} | body={body}"
        return details

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
