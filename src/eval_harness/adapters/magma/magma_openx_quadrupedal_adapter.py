"""
Magma Model Adapter for OpenX Quadrupedal Robot Tasks

This adapter integrates the Magma model with the OpenX quadrupedal locomotion task
evaluation framework, supporting continuous action prediction for robot control.
"""

import os
import sys
import ast
import json
from typing import Dict, Any, List, Optional, Union

import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(ROOT_DIR)

from src.eval_harness.model_adapter import ModelAdapter


class MagmaOpenXQuadrupedalAdapter(ModelAdapter):
    """
    Adapter for Magma model on OpenX quadrupedal robot tasks.
    
    This adapter handles continuous action prediction for quadrupedal locomotion,
    outputting 12-dimensional motor joint positions.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "microsoft/Magma-8B",
        torch_dtype: str = "bf16",
        device_map: str = "auto"
    ):
        """
        Initialize the Magma OpenX Quadrupedal adapter.
        
        Args:
            model_name_or_path: Path or identifier for the Magma model
            torch_dtype: Data type for model weights (bf16, fp16, or fp32)
            device_map: Device mapping strategy for model loading
        """
        super().__init__(
            model_name="Magma-8B-OpenX-Quadrupedal",
            model_type="continuous_action",
            supported_datasets=["openx_quadrupedal"]
        )
        
        self.model_name_or_path = model_name_or_path
        self.device_map = device_map
        self.action_dim = 12  # Quadrupedal robot has 12 motor joints
        
        # Map dtype string to torch dtype
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)
        
        self.model = None
        self.processor = None
        self.generation_args = None
        self.device = None
    
    def initialize(
        self,
        device: str = "cuda",
        seed: int = 42,
        **kwargs
    ) -> None:
        """
        Initialize the Magma model and processor.
        
        Args:
            device: Device to load model on (cuda or cpu)
            seed: Random seed for reproducibility
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
        
        # Set model to eval mode
        self.model.eval()
        
        # Store device info
        self.device = device if self.device_map == "auto" else self.device_map
        
        # Set generation parameters (deterministic for action prediction)
        self.generation_args = {
            "max_new_tokens": 256,
            "temperature": 0.0,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
        }
        
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
        Predict continuous action for quadrupedal robot control.
        
        Args:
            observation: Observation containing:
                - image_observation: View of the environment
                - text_observation: Environment description
                - options: Action space dict mapping indices to descriptions
                - action_stats: Statistics dict with min, max, mean, std, q01, q99
            instruction: what the agent should do
            dataset_name: Name of the dataset
            history: Optional conversation history
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with:
                - "raw_output": str (raw model output text)
                - "extracted_outputs": np.ndarray or None (parsed action vector)
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        # Validate dataset
        if dataset_name and not self.is_dataset_supported(dataset_name):
            raise NotImplementedError(f"Dataset {dataset_name} not supported by Magma OpenX adapter")
        
        if instruction is None:
            raise ValueError("Instruction (environment name) is required for OpenX tasks")
        
        # Extract observation components
        image = observation.get('image_observation')
        if image is None:
            raise ValueError("image_observation is required in observation dict")
        
        # Convert to PIL Image (same as inference script: Image.fromarray(img))
        image = Image.fromarray(image)
        
        # Extract action space and stats
        action_space = observation.get('options', {})
        action_stats = observation.get('action_stats', {})
        
        # Get environment description
        env_desc = observation.get('text_observation', '')
        if isinstance(env_desc, list):
            env_desc = ' '.join(env_desc)
        
        # Create prompt for the model
        prompt_text = self._format_action_prediction_prompt(
            task_name=instruction,
            env_description=env_desc,
            action_space=action_space,
            action_stats=action_stats
        )
        
        # Format using chat template with system message and image
        convs = [
            {
                "role": "system",
                "content": "You are an agent that can see, talk and act.",
            },
            {
                "role": "user",
                "content": f"<image>\n{prompt_text}",
            },
        ]
        
        prompt = self.processor.tokenizer.apply_chat_template(
            convs,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Handle image tokens
        if hasattr(self.model.config, 'mm_use_image_start_end') and self.model.config.mm_use_image_start_end:
            prompt = prompt.replace("<image>", "<image_start><image><image_end>")
        
        # Process inputs
        inputs = self.processor(images=image, texts=prompt, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
            inputs['pixel_values'] = inputs['pixel_values'].to(self.torch_dtype)
        
        # Generate output
        with torch.inference_mode():
            generate_ids = self.model.generate(**inputs, **self.generation_args)
        
        # Extract only the generated tokens (skip input prompt)
        generate_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
        raw_output = self.processor.decode(generate_ids[0], skip_special_tokens=True).strip()
        
        # Parse output to action array
        parsed_action = self._parse_action_output(raw_output)
        
        return {
            "raw_output": raw_output,
            "extracted_outputs": parsed_action
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
        Batch prediction is not implemented for this adapter
        """
        raise NotImplementedError("Batch prediction is not implemented for this adapter")
    
    def _format_action_prediction_prompt(
        self,
        task_name: str,
        env_description: str,
        action_space: dict,
        action_stats: dict
    ) -> str:
        """
        Format a prompt for continuous action prediction.
        
        Args:
            task_name: What the agent should do
            env_description: Description of the environment and task
            action_space: Dictionary mapping action indices to descriptions
            action_stats: Statistics dict with min, max, mean
            
        Returns:
            Formatted prompt string
        """
        # Build the main instruction
        prompt_parts = []
        
        prompt_parts.append(f'You are an AI agent to solve the following task: "{task_name}".')
        
        if env_description:
            prompt_parts.append(f"In this environment: {env_description}")
        
        prompt_parts.append(
            "You should produce a proper action output to achieve the final goal given the current state."
        )
        
        prompt_parts.append(
            "The current state is shown in the image."
        )
        
        # Build action description
        action_desc_parts = []
        action_desc_parts.append(
            "The actions available: A continuous action has the range of (minimum) ~ (maximum). "
            "Each dimension is described using statistics over the entire dataset, "
            "which includes the range between (minimum) ~ (maximum) and a mean of (mean)."
        )

        # Get action dimension
        if 'size' in action_stats:
            action_dim = action_stats['size'][0]
        else:
            action_dim = action_stats['min'].shape[0] if hasattr(action_stats['min'], 'shape') else len(action_stats['min'])
        
        # Describe each action dimension
        for i in range(action_dim):
            # Get verbal description if available
            verbal_desc = f"Action dimension {i}"
            if i in action_space:
                desc_tuple = action_space[i]
                if isinstance(desc_tuple, tuple) and len(desc_tuple) > 0:
                    verbal_desc = str(desc_tuple[0])
            
            # Add stats
            min_val = float(action_stats['min'][i])
            max_val = float(action_stats['max'][i])
            mean_val = float(action_stats['mean'][i])
            
            action_desc_parts.append(
                f"{i}. {verbal_desc} => Continuous. Range: {min_val:.4f} ~ {max_val:.4f}. Mean: {mean_val:.4f}."
            )
        
        prompt_parts.append('\n'.join(action_desc_parts))
        
        # Output format instruction
        prompt_parts.append(
            "You must generate your output keeping the following format: "
            "A list starting with '[' and ending with ']'. "
            "Each position corresponds to each action index and a value in that position represents "
            "the actual continuous value of that action."
        )
        
        prompt_parts.append(
            "You should not include any other words or characters in your response."
        )
        
        return ' '.join(prompt_parts)
    
    def _parse_action_output(self, output_text: str) -> Optional[np.ndarray]:
        """
        Parse model output text to action array.
        
        Tries multiple parsing strategies:
        1. ast.literal_eval for Python list format
        2. json.loads for JSON format
        
        Args:
            output_text: Raw output text from model
            
        Returns:
            Numpy array of parsed action values, or None if parsing fails
        """
        try:
            # Try to parse as list using ast.literal_eval
            parsed = ast.literal_eval(output_text.strip())
            if isinstance(parsed, list):
                return np.array([float(item) for item in parsed])
            else:
                # If not a list, wrap in list and convert
                return np.array([float(parsed)])
        except:
            try:
                # Try JSON parsing
                parsed = json.loads(output_text.strip())
                if isinstance(parsed, list):
                    return np.array([float(item) for item in parsed])
                else:
                    return np.array([float(parsed)])
            except:
                # Parsing failed
                return None
    
    def reset(self) -> None:
        """Reset model state (clears GPU cache if applicable)."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

