"""
Magma Model Adapter for OpenX Robotic Manipulation Tasks

This adapter integrates the Magma model with OpenX robotic manipulation evaluation,
supporting continuous action prediction for mobile manipulation, single-arm, bimanual,
and wheeled robot tasks.
"""

import sys

from typing import Dict, Any, List, Optional

import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path

# go up directories until we find the project root
PROJECT_ROOT_DIR = next(p for p in Path(__file__).parents if p.parts[-1] == 'MultiNet')
MAGMA_ROOT_DIR = PROJECT_ROOT_DIR / 'src' / 'v1' / 'modules' / 'Magma'
sys.path.append(str(MAGMA_ROOT_DIR))


from src.eval_harness.model_adapter import ModelAdapter
from src.v1.modules.Magma.data.openx.action_tokenizer import ActionTokenizer


class MagmaOpenXAdapter(ModelAdapter):
    """
    Adapter for Magma model on OpenX robotic manipulation tasks.

    This adapter handles continuous action prediction for the following OpenX morphologies:
    - openx_mobile_manipulation
    - openx_single_arm
    - openx_bimanual
    - openx_wheeled_robot

    All datasets use the same ActionTokenizer and unnormalization approach.
    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/Magma-8B",
        torch_dtype: str = "bf16",
        device_map: str = "auto"
    ):
        """
        Initialize the Magma OpenX adapter.

        Args:
            model_name_or_path: Path or identifier for the Magma model
            torch_dtype: Data type for model weights (bf16, fp16, or fp32)
            device_map: Device mapping strategy for model loading
        """
        super().__init__()
        self.model_name = "Magma-8B-OpenX"
        self.model_type = "continuous_action"

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
        self.action_tokenizer = None
        self.generation_args = None
        self.device = None

    @property
    def supported_datasets(self) -> List[str]:
        return [
            "openx_mobile_manipulation",
            "openx_single_arm",
            "openx_bimanual",
            "openx_wheeled_robot"
        ]

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

        # Initialize ActionTokenizer
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)

        # Set model to eval mode
        self.model.eval()

        # Store device info
        self.device = device if self.device_map == "auto" else self.device_map

        self.generation_args = {
            "max_new_tokens": 1000,
            "temperature": 0.7,
            "do_sample": True,
            "num_beams": 1,
            "use_cache": True,
        }

        self._is_initialized = True
        print(f"Magma model initialized on {self.model.device}")

    def _rel2abs_gripper_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        Converts relative gripper actions (+1 for closing, -1 for opening) to absolute actions (0 = closed; 1 = open).
        Simplified version for numpy arrays.
        """
        # Note =>> -1 for closing, 1 for opening, 0 for no change
        opening_mask = actions < -0.1
        closing_mask = actions > 0.1
        thresholded_actions = np.where(opening_mask, 1, np.where(closing_mask, -1, 0))

        # Convert to 0 = closed, 1 = open
        new_actions = thresholded_actions / 2 + 0.5

        return new_actions

    def _transform_action_stats(self, action_stats: dict, dataset_name: str) -> dict:
        """
        Transform action stats from original dimensions to model-expected dimensions.
        
        Args:
            action_stats: Original action stats dict with min, max, mean, std, q01, q99, etc.
            dataset_name: Name of the dataset
            
        Returns:
            Transformed action stats dict with dimensions matching model output
        """
        original_stats = {}
        for key in ['min', 'max', 'mean', 'std', 'q01', 'q99']:
            if key in action_stats:
                original_stats[key] = np.array(action_stats[key])
        
        original_dim = len(original_stats['q01']) if 'q01' in original_stats else 0
        
        if dataset_name == 'openx_wheeled_robot':
            # Transform: 2D -> 7D (formula: 3N + 1)
            # For N=2: [a, b] -> [a, b, 0, 0, 0, 0, 0]
            transformed_stats = {}
            for key, values in original_stats.items():
                if values is not None:
                    # Pad with zeros: 2 zeros, 2 zeros, one zero
                    padded = np.concatenate([
                        values,
                        np.zeros(len(values)),
                        np.zeros(len(values)),
                        np.zeros(1)
                    ])
                    transformed_stats[key] = padded.tolist()
            return transformed_stats
            
        elif dataset_name == 'openx_bimanual':
            # Transform: 14D -> 7D (take last 7 dimensions)
            transformed_stats = {}
            for key, values in original_stats.items():
                if values is not None:
                    transformed_stats[key] = values[-7:].tolist()
            return transformed_stats
            
        elif dataset_name in ['openx_mobile_manipulation', 'openx_single_arm']:
            # No transformation needed - already 7D after dict concatenation
            return action_stats
            
        else:
            # Unknown dataset - return as-is
            return action_stats

    def _inverse_transform_prediction(self, prediction: np.ndarray, original_stats: dict, dataset_name: str) -> np.ndarray:
        """
        Inverse transform prediction from model dimensions back to original dimensions.
        
        Args:
            prediction: Model prediction in transformed dimensions (7D)
            original_stats: Original action stats dict
            dataset_name: Name of the dataset
            
        Returns:
            Prediction in original dimensions
        """
        original_dim = len(original_stats['q01']) if 'q01' in original_stats else len(prediction)
        
        if dataset_name == 'openx_wheeled_robot':
            # Inverse transform: 7D -> 2D (take first N dimensions)
            return prediction[:original_dim]
            
        elif dataset_name == 'openx_bimanual':
            # Inverse transform: 7D -> 14D (pad with zeros at start)
            if original_dim > len(prediction):
                padding = np.zeros(original_dim - len(prediction))
                return np.concatenate([padding, prediction])
            else:
                return prediction
                
        elif dataset_name in ['openx_mobile_manipulation', 'openx_single_arm']:
            # No inverse transformation needed - already correct dimensions
            return prediction
            
        else:
            # Unknown dataset - return as-is
            return prediction

    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict continuous action for OpenX robotic manipulation.

        Args:
            observation: Observation containing:
                - image_observation: View of the environment
                - text_observation: Environment description
                - options: Action space dict mapping indices to descriptions
                - action_stats: Statistics dict with min, max, mean, std, q01, q99
            instruction: Task instruction
            dataset_name: Name of the dataset
            history: Optional conversation history
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with:
                - "raw_output": str (raw model output text)
                - "extracted_outputs": np.ndarray (unnormalized continuous action vector)
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # Validate dataset
        if dataset_name and not self.is_dataset_supported(dataset_name):
            raise NotImplementedError(f"Dataset {dataset_name} not supported by Magma OpenX adapter")

        # Extract observation components
        image = observation.get('image_observation')
        if image is None:
            raise ValueError("image_observation is required in observation dict")

        # Convert to PIL Image
        image = Image.fromarray(image)

        # Extract action stats (required for unnormalization)
        original_action_stats = observation.get('action_stats', {})
        if not original_action_stats:
            raise ValueError("action_stats is required for action unnormalization")
        
        # Transform stats to model-expected dimensions (if needed)
        if dataset_name:
            transformed_stats = self._transform_action_stats(original_action_stats, dataset_name)
        else:
            transformed_stats = original_action_stats
        
        if instruction is not None:
            user_inst = f"What action should the robot take to {instruction}?"
        else:
            user_inst = f"What action should the robot take to {observation['text_observation'].lower().rstrip('.')}?"

        convs = [
            {
                "role": "system",
                "content": "You are agent that can see, talk and act.",
            },
            {
                "role": "user",
                "content": f"<image>\n{user_inst}",
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
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
            inputs['pixel_values'] = inputs['pixel_values'].to(self.torch_dtype)

        # Generate output
        with torch.inference_mode():
            generate_ids = self.model.generate(**inputs, **self.generation_args)

        action_ids = generate_ids[0, -8:-1].cpu().tolist()
        action_ids = np.array(action_ids).astype(np.int64)

        normalized_action = self.action_tokenizer.decode_token_ids_to_actions(action_ids)

        # Model always outputs 7D, pad/truncate to match transformed stats dimension
        expected_transformed_dim = len(transformed_stats['q01']) if 'q01' in transformed_stats else 7
        actual_dim = len(normalized_action)
        
        if actual_dim != expected_transformed_dim:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Action dimension mismatch for {dataset_name}: "
                f"decoded {actual_dim} dims, expected {expected_transformed_dim} dims. "
                f"Adjusting to match expected dimension."
            )
            
            # Pad or truncate to match expected dimensions
            if actual_dim < expected_transformed_dim:
                # Pad with zeros
                padding = np.zeros(expected_transformed_dim - actual_dim)
                normalized_action = np.concatenate([normalized_action, padding])
            else:
                # Truncate
                normalized_action = normalized_action[:expected_transformed_dim]

        # Unnormalize using transformed stats
        unnormalized_action = self._unnormalize_action(normalized_action, transformed_stats)

        # Inverse transform prediction back to original dimensions
        if dataset_name:
            final_prediction = self._inverse_transform_prediction(unnormalized_action, original_action_stats, dataset_name)
        else:
            final_prediction = unnormalized_action

        return {
            "raw_output": str(action_ids.tolist()),
            "extracted_outputs": final_prediction
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
        Batch prediction is not implemented for OpenX datasets.
        """
        raise NotImplementedError("Batch prediction is not implemented for OpenX datasets")

    def _unnormalize_action(self, normalized_action: np.ndarray, action_stats: dict) -> np.ndarray:
        """
        Unnormalize action from [-1, 1] range to original action space.

        Args:
            normalized_action: Action in normalized space
            action_stats: Statistics dict with q01, q99

        Returns:
            Unnormalized action
        """
        # Same as src/v1/modules/Magma/agents/libero/libero_magma_utils.py and src/v1/modules/Magma/tools/simplerenv-magma/simpler_env/policies/magma/magma_model.py
        action_low, action_high = np.array(action_stats["q01"]), np.array(action_stats["q99"])
        return 0.5 * (normalized_action + 1) * (action_high - action_low) + action_low

    def reset(self) -> None:
        """Reset model state (clears GPU cache if applicable)."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
