"""
Magma Model Adapter for VQA Tasks (RoboVQA and SQA3D)

This adapter integrates the Magma model with VQA evaluation frameworks,
supporting both RoboVQA and SQA3D datasets with dataset-specific system prompts.
"""

import os
import sys
import re
from typing import Dict, Any, List, Optional

import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(ROOT_DIR)

from src.eval_harness.model_adapter import ModelAdapter
from definitions.sqa3d_prompt import SQA3DDefinitions
from definitions.robovqa_prompt import ROBOVQA_PROMPT


class MagmaVQAAdapter(ModelAdapter):
    """
    Adapter for Magma model on VQA tasks (RoboVQA and SQA3D).
    
    This adapter handles visual question answering with dataset-specific
    system prompts and shared text normalization logic.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "microsoft/Magma-8B",
        torch_dtype: str = "bf16",
        device_map: str = "auto",
        max_answer_length: int = 100
    ):
        """
        Initialize the Magma VQA adapter.
        
        Args:
            model_name_or_path: Path or identifier for the Magma model
            torch_dtype: Data type for model weights (bf16, fp16, or fp32)
            device_map: Device mapping strategy for model loading
            max_answer_length: Maximum number of tokens to generate
        """
        super().__init__()
        self.model_name = "magma"
        self.model_type = "text_generation"
        self.max_answer_length = max_answer_length
        
        self.model_name_or_path = model_name_or_path
        self.device_map = device_map
        
        # Map dtype string to torch dtype
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)
        
        self.model = None
        self.processor = None
        self.device = None

    @property
    def supported_datasets(self) -> List[str]:
        return ["robot_vqa", "sqa3d"]

    def initialize(
        self,
        device: str = "cuda",
        seed: int = 42,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs
    ) -> None:
        """
        Initialize the Magma model and processor.
        
        Args:
            device: Device to load model on (cuda or cpu)
            seed: Random seed for reproducibility
            temperature: Sampling temperature for generation
            do_sample: Whether to use sampling for generation
            **kwargs: Additional initialization parameters
        """
        print(f"Initializing Magma model: {self.model_name_or_path}")
        
        # Set random seeds
        self.set_seed(seed)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )
        
        # Configure tokenizer
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "left"
        
        # Set model to eval mode
        self.model.eval()
        
        # Store device info
        self.device = device if self.device_map == "auto" else self.device_map
        
        # Store generation parameters
        self.temperature = temperature
        self.do_sample = do_sample
        
        self._is_initialized = True
        print(f"Magma model initialized on {self.model.device}")
    
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict answer for a VQA task.
        
        Args:
            observation: Observation containing image_observation key
            instruction: Question text
            dataset_name: Name of the dataset ("robot_vqa" or "sqa3d")
            history: Optional conversation history (typically not used for VQA)
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with:
                - "raw_output": str (raw model output text)
                - "extracted_outputs": str (normalized answer text)
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        # Validate dataset
        if dataset_name and not self.is_dataset_supported(dataset_name):
            raise NotImplementedError(f"Dataset {dataset_name} not supported by Magma VQA adapter")
        
        if instruction is None:
            raise ValueError("Instruction is required for VQA tasks")
        
        # Preprocess image
        processed_image = self.preprocess_observation(observation, dataset_name)['image_observation']
        
        # Get dataset-specific system prompt
        system_prompt = self._get_system_prompt(dataset_name)
        
        # Build conversation
        inst_content = f"<image_start><image><image_end>\n{instruction}"
        prompt_content = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": inst_content}
        ]
        
        # Apply chat template
        prompt = self.processor.tokenizer.apply_chat_template(
            prompt_content, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process with processor
        inputs = self.processor(images=[processed_image], texts=prompt, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
        
        # Move to device and convert dtype
        inputs = inputs.to(self.model.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(self.torch_dtype)
        
        # Generation parameters
        generation_args = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_answer_length),
            "temperature": kwargs.get("temperature", self.temperature),
            "do_sample": kwargs.get("do_sample", self.do_sample),
            "use_cache": False,
            "past_key_values": None,
        }
        
        # Generate response
        with torch.inference_mode():
            generate_ids = self.model.generate(**inputs, **generation_args)
        
        # Decode response
        generate_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
        raw_output = self.processor.decode(generate_ids[0], skip_special_tokens=True).strip()
        
        # Normalize output
        normalized_output = self._normalize_text(raw_output)
        
        return {
            "raw_output": raw_output,
            "extracted_outputs": normalized_output
        }
    
    def batch_predict_actions(
        self,
        observations: List[Dict[str, Any]],
        instructions: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        histories: Optional[List[List[Dict[str, str]]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Predict answers for a batch of observations.
        
        For VQA tasks, this falls back to single-item processing since
        batch processing is not supported for Magma VQA adapter.
        
        Args:
            observations: List of observation dictionaries
            instructions: List of instruction strings
            dataset_name: Name of the dataset
            histories: Optional list of conversation histories
            **kwargs: Additional generation parameters
            
        Returns:
            List of prediction dictionaries, one per observation
        """
        raise NotImplementedError("Batch processing not supported for Magma VQA adapter")
    
    def _get_system_prompt(self, dataset_name: str) -> str:
        """
        Get dataset-specific system prompt.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            System prompt string
        """
        if dataset_name == "robot_vqa":
            return ROBOVQA_PROMPT
        elif dataset_name == "sqa3d":
            return SQA3DDefinitions.SYSTEM_PROMPT
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison by removing punctuation and extra spaces.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert numeric strings to word form
        numbers = "zero one two three four five six seven eight nine".split()
        int_numbers = [str(i) for i in range(10)]
        if text in int_numbers:
            text = numbers[int(text)]
        
        return text
    
    def _validate_text_output(self, output: Any) -> bool:
        """
        Validate that output is a valid text string.
        
        Args:
            output: Output to validate
            
        Returns:
            True if valid text string, False otherwise
        """
        return isinstance(output, str) and len(output.strip()) > 0
    
    def preprocess_observation(
        self,
        observation: Dict[str, Any],
        dataset_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Preprocess observation for VQA.
        
        Args:
            observation: Raw observation dictionary
            dataset_name: Name of the dataset
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed observation dictionary
        """
        processed_obs = observation.copy()
        
        # Preprocess image from standardized key
        if 'image_observation' in processed_obs:
            image = processed_obs['image_observation']
            
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            
            processed_obs['image_observation'] = image
                        
        return processed_obs
    
    def reset(self) -> None:
        """Reset model state (clears cache if applicable)."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
