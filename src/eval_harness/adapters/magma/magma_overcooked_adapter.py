"""
Magma Model Adapter for Overcooked

This adapter integrates the Magma model with the Overcooked evaluation framework.
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

OVERCOOKED_INSTRUCTION = [
    "We are running a simulation for two AI agents cooperatively playing Overcooked, a kitchen coordination game.",
    "Your role is to evaluate potential joint actions for both players based on the current game state.",
    "You should produce proper joint action outputs to maximize soup delivery through effective coordination.",
    "The current state consists of a game screenshot showing player positions, ingredients, cooking stations, objectives.",
    "Values for the elapsed time and the remaining time are also provided.",
    "Both players must work together - one player's actions affect the other's ability to complete tasks.",
    "Key game mechanics: Players can pick up ingredients, place them in pots (3 ingredients per soup), wait for cooking (20 time steps), then deliver completed soups to serving stations for points.",
    "Individual action meanings: {action_meaning}.",
    "Options available in the format 'Player 0 action, Player 1 action': {action_info}.",
    "Focus on maximizing soup delivery rate while maintaining smooth coordination between players within the time remaining.",
    "You MUST generate your output keeping the following format: {output_format}",
    "You should not include any other words or characters in your response."
]

OVERCOOKED_OUTPUT_FORMAT = """
    One of the 36 possible option indices (0 through 35) that represents the joint action to be taken by the two players.
"""

class MagmaOvercookedAdapter(ModelAdapter):
    """
    Adapter for Magma model on Overcooked.
    
    This adapter handles Overcooked gameplay and extracts joint action indices from model outputs.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "microsoft/Magma-8B",
        torch_dtype: str = "bf16",
        device_map: str = "auto"
    ):
        """
        Initialize the Magma Overcooked adapter.
        
        Args:
            model_name_or_path: Path or identifier for the Magma model
            torch_dtype: Data type for model weights (bf16, fp16, or fp32)
            device_map: Device mapping strategy for model loading
        """
        super().__init__()
        self.model_name = "magma"
        self.model_type = "discrete_action"
        
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
        return ["overcooked_ai"]

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
        Predict action for Overcooked.
        
        Args:
            observation: Observation containing image_observation or options
            instruction: Instruction text
            dataset_name: Name of the dataset ("overcooked_ai")
            history: Optional conversation history (typically not used for Overcooked)
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with:
                - "raw_output": str (raw model output text)
                - "extracted_outputs": int (action index 0 to num_actions-1, or -1 if invalid)
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        # Validate dataset
        if dataset_name and not self.is_dataset_supported(dataset_name):
            raise NotImplementedError(f"Dataset {dataset_name} not supported by Magma Overcooked adapter")
        
        if instruction is None:
            raise ValueError("Instruction is required for Overcooked")
        
        # Handle different observation types
        processed_image = None
        if observation is not None and 'image_observation' in observation:
            processed_image = self.preprocess_observation(observation, dataset_name)['image_observation']
        
        # Format system prompt
        system_prompt_text = "\n".join(OVERCOOKED_INSTRUCTION).format(
            action_meaning=observation['text_observation'], 
            action_info=observation['options'],
            output_format=OVERCOOKED_OUTPUT_FORMAT
        )

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
        
        
        inputs = self.processor(images=[processed_image], texts=prompt, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(self.torch_dtype)
        
        # Generation parameters
        generation_args = {
            "max_new_tokens": kwargs.get("max_new_tokens", 75),
            "temperature": kwargs.get("temperature", self.temperature),
            "do_sample": kwargs.get("do_sample", self.do_sample),
            "use_cache": True
        }
        
        # Generate response
        with torch.inference_mode():
            generate_ids = self.model.generate(**inputs, **generation_args)
        
        # Decode response
        generate_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
        raw_output = self.processor.decode(generate_ids[0], skip_special_tokens=True).strip()
        
        # Extract action index from output
        options = observation['options']
        num_actions = len(options.keys())
        action_idx = self._validate_and_extract_choice(raw_output, num_actions)
        
        return {
            "raw_output": raw_output,
            "extracted_outputs": action_idx
        }
    
    def batch_predict_actions(
        self,
        observations: List[Dict[str, Any]],
        instructions: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        histories: Optional[List[List[Dict[str, str]]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        
        raise NotImplementedError("Batch prediction is not implemented for Overcooked")
    
    def _validate_and_extract_choice(self, output: str, num_actions: int) -> int:
        """
        Validate and extract choice index from model output.
        
        Args:
            output: Raw model output text
            num_actions: Number of valid action choices
            
        Returns:
            Integer action index (0 to num_actions-1) if valid, -1 if invalid
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
                    if 0 <= int_num < num_actions:
                        return int_num
            except Exception:
                pass
        
        return -1
    
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
            image = Image.fromarray(image)
            processed_obs['image_observation'] = image
                        
        return processed_obs
    
    def reset(self) -> None:
        """Reset model state (clears cache if applicable)."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
