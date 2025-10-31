"""
Magma Model Adapter for Multiple Choice Question (MCQ) Tasks

This adapter integrates the Magma model with MCQ evaluation frameworks,
supporting both visual classification (ODinW) and text-only reasoning (PIQA).
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
from definitions.odinw import ODinWDefinitions

# Import PIQA system prompt
PIQA_SYSTEM_PROMPT = """
    You are evaluating physical commonsense reasoning questions. You will be presented with a goal and possible solutions.
    Your task is to determine which solution is more appropriate for achieving the given goal.
    Output only the index of the correct solution, and nothing else.
    Do not provide any explanation, reasoning, or additional text.
"""


class MagmaMCQAdapter(ModelAdapter):
    """
    Adapter for Magma model on multiple choice question tasks.
    
    This adapter handles both visual classification (ODinW) and text-only reasoning (PIQA)
    with multiple choice questions and extracts integer class indices from model outputs.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "microsoft/Magma-8B",
        torch_dtype: str = "bf16",
        device_map: str = "auto"
    ):
        """
        Initialize the Magma MCQ adapter.
        
        Args:
            model_name_or_path: Path or identifier for the Magma model
            torch_dtype: Data type for model weights (bf16, fp16, or fp32)
            device_map: Device mapping strategy for model loading
        """
        super().__init__()
        self.model_name = "magma"
        self.model_type = "multiple_choice"
        
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
        return ["odinw", "piqa"]

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
        Predict class for multiple choice task (ODinW or PIQA).
        
        Args:
            observation: Observation containing image_observation (ODinW) or options (both)
            instruction: Question text with multiple choice options
            dataset_name: Name of the dataset ("odinw" or "piqa")
            history: Optional conversation history (typically not used for MCQ tasks)
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with:
                - "raw_output": str (raw model output text)
                - "extracted_outputs": int (class index 0 to num_classes-1, or -1 if invalid)
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        # Validate dataset
        if dataset_name and not self.is_dataset_supported(dataset_name):
            raise NotImplementedError(f"Dataset {dataset_name} not supported by Magma MCQ adapter")
        
        if instruction is None:
            raise ValueError("Instruction is required for MCQ tasks")
        
        # Check if image observation is present
        has_image = observation is not None and 'image_observation' in observation and observation['image_observation'] is not None
        
        if has_image:
            # Image-based case (ODinW): process image and include in conversation
            processed_image = self.preprocess_observation(observation, dataset_name)['image_observation']
            
            # Get dataset-specific system prompt for ODinW
            system_prompt_text = self._get_system_prompt(dataset_name)
            system_prompt = {"role": "system", "content": system_prompt_text}
            
            inst_content = f"<image_start><image><image_end>\n{instruction}"
            user_prompt = {"role": "user", "content": inst_content}
            prompt_content = [system_prompt, user_prompt]
            
            # Apply chat template
            prompt = self.processor.tokenizer.apply_chat_template(
                prompt_content, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process with image
            inputs = self.processor(images=[processed_image], texts=prompt, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
            inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
        else:
            # Text-only case (PIQA): use PIQA system prompt to match batch processing
            system_prompt_text = self._get_system_prompt("piqa")
            system_prompt = {"role": "system", "content": system_prompt_text}
            
            # Treat single example as batch of 1 with system prompt
            chats = [[system_prompt, {"role": "user", "content": instruction}]]
            
            # Apply chat template and tokenize directly
            input_ids = self.processor.tokenizer.apply_chat_template(
                chats,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_generation_prompt=True
            )
            
            # Move to device
            input_ids = input_ids.to(self.model.device)
        
        # Move to device and convert dtype
        if has_image:
            inputs = inputs.to(self.model.device)
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(self.torch_dtype)
        
        # Generation parameters
        generation_args = {
            "max_new_tokens": kwargs.get("max_new_tokens", 50),
            "temperature": kwargs.get("temperature", self.temperature),
            "do_sample": kwargs.get("do_sample", self.do_sample),
            "use_cache": True,
        }
        
        # Generate response
        with torch.inference_mode():
            if has_image:
                generate_ids = self.model.generate(**inputs, **generation_args)
            else:
                # Text-only case: use input_ids directly
                generate_ids = self.model.generate(input_ids=input_ids, **generation_args)
        
        # Decode response
        if has_image:
            generate_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
        else:
            # Text-only case: use input_ids length
            generate_ids = generate_ids[:, input_ids.shape[-1]:]
        raw_output = self.processor.decode(generate_ids[0], skip_special_tokens=True).strip()
        
        # Extract class index from output
        # Get number of classes from observation options
        options = observation.get('options', [])
        num_classes = len(options)
        if not options:
            num_classes = np.inf
            print(f"No MCQ options provided for observation, passing output along without validating range")
        class_idx = self._validate_and_extract_choice(raw_output, num_classes)
        
        return {
            "raw_output": raw_output,
            "extracted_outputs": class_idx
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
        Predict classes for a batch of observations.
        
        Args:
            observations: List of observation dictionaries, must contain 'options'
            instructions: List of instruction strings, must match number of observations
            dataset_name: Name of the dataset
            histories: Optional list of conversation histories
            **kwargs: Additional generation parameters
        """
        batch_size = len(observations)
        if batch_size == 0:
            return []
        
        # Check if any observation has image_observation (not supported in batch processing)
        for i, obs in enumerate(observations):
            if obs is not None and 'image_observation' in obs and obs['image_observation'] is not None:
                raise NotImplementedError(f"Batch processing not supported for image-based tasks (ODinW). Observation {i} contains image_observation.")
        
        # Validate instructions
        if instructions is None:
            raise ValueError("Instructions are required for PIQA batch prediction")
        
        if len(instructions) != batch_size:
            raise ValueError(f"Number of instructions ({len(instructions)}) must match observations ({batch_size})")
        
        # Get PIQA system prompt
        system_prompt_text = self._get_system_prompt("piqa")
        system_prompt = {"role": "system", "content": system_prompt_text}
        
        # Build conversations for all samples
        conversations = []
        for i in range(batch_size):
            conversation = [
                system_prompt,
                {"role": "user", "content": instructions[i]}
            ]
            conversations.append(conversation)
        
        # Apply chat template with padding for batch processing
        input_ids = self.processor.tokenizer.apply_chat_template(
            conversations,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_generation_prompt=True,
        ).to(self.model.device)
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 50),
            "temperature": kwargs.get("temperature", self.temperature),
            "do_sample": kwargs.get("do_sample", self.do_sample),
            "use_cache": True,
        }
        
        # Generate batch responses
        with torch.inference_mode():
            generate_ids = self.model.generate(input_ids=input_ids, **gen_kwargs)
            input_token_len = input_ids.shape[1]
            responses = self.processor.batch_decode(
                generate_ids[:, input_token_len:],
                skip_special_tokens=True,
            )
        
        # Extract class indices for each response
        predictions = []
        for i, raw_output in enumerate(responses):
            # Get number of classes from observation options
            options = observations[i].get('options', [])
            num_classes = len(options)
            if not options:
                num_classes = np.inf
                print(f"No MCQ options provided for observation, passing output along without validating range")
            
            class_idx = self._validate_and_extract_choice(raw_output.strip(), num_classes)
            
            predictions.append({
                "raw_output": raw_output.strip(),
                "extracted_outputs": class_idx
            })
        
        return predictions
    
    def _validate_and_extract_choice(self, output: str, num_classes: int) -> int:
        """
        Validate and extract choice index from model output.
        
        Args:
            output: Raw model output text
            num_classes: Number of valid class choices
            
        Returns:
            Integer class index (0 to num_classes-1) if valid, -1 if invalid
        """
        # Handle string outputs that might contain the choice
        if isinstance(output, str):
            # Specific to Magma - reject if starts with "Coordinate"
            if output.startswith("Coordinate"):
                return -1
                
            try:
                numbers = re.findall(r'\d+', output)
                if numbers:
                    int_num = int(numbers[0])
                    # Validate number is in valid range
                    if 0 <= int_num < num_classes:
                        return int_num
            except Exception:
                pass
        
        return -1
    
    def _get_system_prompt(self, dataset_name: str) -> str:
        """
        Get dataset-specific system prompt.
        
        Args:
            dataset_name: Name of the dataset ("odinw" or "piqa")
            
        Returns:
            System prompt string
        """
        if dataset_name == "odinw":
            return ODinWDefinitions.SYSTEM_PROMPT
        elif dataset_name == "piqa":
            return PIQA_SYSTEM_PROMPT
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def preprocess_observation(
        self,
        observation: Dict[str, Any],
        dataset_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Preprocess observation for MCQ classification.
        
        Args:
            observation: Raw observation dictionary
            dataset_name: Name of the dataset
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed observation dictionary
        """
        processed_obs = observation.copy()
        
        # Preprocess image from standardized key
        if 'image_observation' in processed_obs and processed_obs['image_observation'] is not None:
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
