from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from huggingface_hub import InferenceClient, get_token
from transformers import AutoModelForCausalLM, AutoTokenizer

# More stable defaults for local model downloads on Windows.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Keep empty in source. Prefer env var `HF_TOKEN` or `huggingface-cli login`.
_LOCAL_HF_TOKEN = ""
if _LOCAL_HF_TOKEN.strip() and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = _LOCAL_HF_TOKEN.strip()


class RandomAgent:
    def __call__(self, messages: list[dict]) -> str:
        actions = ["MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST"]
        return f"FINAL_ACTION: {random.choice(actions)}"


DEFAULT_ROUTER_MODEL = "meta-llama/Llama-3.1-8B-Instruct:cerebras"
DEFAULT_LOCAL_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


@dataclass
class HFLLMConfig:
    model: str = DEFAULT_ROUTER_MODEL
    temperature: float = 0.0
    max_tokens: int = 64
    timeout: Optional[float] = 30.0


@dataclass
class HuggingFaceLLMAgent:
    """Remote HF Router-backed chat-completions agent."""

    config: HFLLMConfig = field(default_factory=HFLLMConfig)
    client: Optional[InferenceClient] = None

    def __post_init__(self) -> None:
        if self.client is None:
            token = os.environ.get("HF_TOKEN") or get_token()
            if not token:
                raise ValueError(
                    "No Hugging Face token found. Set HF_TOKEN or run `huggingface-cli login`."
                )

            self.client = InferenceClient(
                api_key=token,
                timeout=self.config.timeout,
            )

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content.strip()


@dataclass
class LocalLLMConfig:
    # Open-source/open-weight local models (examples):
    # - Qwen/Qwen2.5-0.5B-Instruct
    # - google/gemma-2-2b-it
    model: str = DEFAULT_LOCAL_MODEL
    temperature: float = 0.0
    max_new_tokens: int = 64
    device_map: str = "auto"


@dataclass
class LocalTransformersAgent:
    """Local agent using Hugging Face Transformers (no inference credits)."""

    config: LocalLLMConfig = field(default_factory=LocalLLMConfig)
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None

    def __post_init__(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                device_map=self.config.device_map,
            )

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.temperature > 0,
        )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = generated[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    agent = LocalTransformersAgent(config=LocalLLMConfig())
    out = agent(
        [
            {"role": "system", "content": "Reply with one short sentence."},
            {"role": "user", "content": "What is 2+2?"},
        ]
    )
    print(out)
