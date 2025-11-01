"""
ModelAdapter Interface for MultiNet Evaluation Harness

This module defines the abstract base class that users must implement to integrate
their models with the MultiNet benchmarking toolkit. The interface standardizes
model interactions while allowing flexibility for different task types.

Users should inherit from ModelAdapter and implement 
the required abstract methods for their specific model.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import torch

class ModelAdapter(ABC):
    """
    Abstract base class for model adapters in the MultiNet evaluation harness.
    
    This interface standardizes how models interact with the evaluation pipeline,
    ensuring consistent behavior across different model implementations while
    providing the flexibility needed for various task types.
    
    Key Requirements:
    - Actions must be returned in the expected format for the target dataset

    Attributes:
        model_name (str): A human-readable name for the model
        model_type (str): Type of task (e.g., 'multiple_choice', 'text_generation', 'discrete_action')
        supported_datasets (List[str]): List of datasets this model supports
    """
    
    def __init__(self):
        """
        Initialize the ModelAdapter.
        
        Args:
            model_name: Human-readable name for the model
            model_type: Type of task this adapter handles
            supported_datasets: List of datasets this model supports
        """
        self._is_initialized = False
        
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize the model and load weights.
        
        This method should:
        - Load model weights/checkpoints
        - Set up the model for inference mode
        - Configure any necessary preprocessing
        - Set random seeds for deterministic inference
        
        Args:
            **kwargs: Additional initialization parameters (paths, device, etc.)
        """
        pass
    
    @property
    @abstractmethod
    def supported_datasets(self) -> List[str]:
        """
        Get the list of datasets this model supports.
        
        Returns:
            List of dataset names
        """
        pass

    @abstractmethod
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict action(s) for a given observation.
        
        This is the core inference method that takes observations and produces actions.
        Must be deterministic - use explicit seeds for any stochastic operations.
        
        Args:
            observation: Dictionary containing observation data. For example:
                - 'image': PIL Image or numpy array of shape (H, W, 3)
                - 'images': Dict of multiple images (e.g., {'rgb': img, 'depth': img})
                - 'state': Continuous state vector
                - 'text': Text observation/description
                - 'question': Question text for VQA/MCQ tasks
                - 'choices': List of choices for MCQ tasks
                - 'video_frames': List of video frames
            instruction: Task instruction string (for VLAs and instruction-following tasks)
            dataset_name: Name of the dataset (for dataset-specific processing)
            history: Optional conversation history for multi-turn tasks. Format:
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
                The model adapter should format this history appropriately for the model.
            **kwargs: Additional prediction parameters
            
        Returns:
            Task-specific output:
            - Multiple choice: Choice index (int)
            - Text generation: Generated text (str)
            - Counting: Count value (int)
            - Discrete action: Action index (int)
            - Continuous action: Action vector (np.ndarray)
            - Classification: Class index (int) or class name (str)
            - Grounding: Entity mappings (int, List[int], or Dict)
            
        Raises:
            ValueError: If observation format is unsupported
            NotImplementedError: If dataset is not supported
        """
        pass
    
    @abstractmethod
    def batch_predict_actions(
        self,
        observations: List[Dict[str, Any]],
        instructions: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        histories: Optional[List[List[Dict[str, str]]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Predict actions for a batch of observations.
        
        This method should be more efficient than calling predict_action in a loop.
        Default implementation can call predict_action repeatedly if no batch
        optimization is possible.
        
        Args:
            observations: List of observation dictionaries
            instructions: Optional list of task instruction strings
            dataset_name: Name of the dataset
            histories: Optional list of conversation histories for multi-turn tasks.
                Each history is a list of message dicts: [{"role": "user", "content": "..."}, ...]
                The model adapter should format each history appropriately for the model.
            **kwargs: Additional prediction parameters
            
        Returns:
            List of task-specific outputs, one per observation
        """
        pass
    
    def _preprocess_observation(
        self,
        observation: Dict[str, Any],
        dataset_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Preprocess observation data before inference.
        
        This method can be overridden to implement model-specific preprocessing
        such as image resizing, normalization, or format conversion.
        
        Args:
            observation: Raw observation dictionary
            dataset_name: Name of the dataset
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed observation dictionary
        """
        return observation
    
    def _postprocess_action(
        self,
        action: Union[np.ndarray, List[float], int, str],
        dataset_name: str,
        **kwargs
    ) -> Union[np.ndarray, List[float], int, str]:
        """
        Postprocess predicted actions.
        
        This method can be overridden to implement model-specific postprocessing
        such as action denormalization, clipping, or format conversion.
        
        Args:
            action: Raw predicted action
            dataset_name: Name of the dataset
            **kwargs: Additional postprocessing parameters
            
        Returns:
            Postprocessed action
        """
        return action
    
    def reset(self) -> None:
        """
        Reset model state (if stateful).
        
        This method should be called between episodes or evaluation runs
        to reset any internal state (e.g., hidden states, caches).
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model metadata including:
            - model_name: Human-readable model name
            - model_type: Type of task
            - supported_datasets: List of supported datasets
            - additional model-specific metadata
        """
        return {
            "model_name": getattr(self, 'model_name', 'Unknown'),
            "model_type": getattr(self, 'model_type', 'Unknown'),
            "supported_datasets": self.supported_datasets,
            "is_initialized": self._is_initialized
        }
    
    def is_dataset_supported(self, dataset_name: str) -> bool:
        """
        Check if this model supports the given dataset.
        
        Args:
            dataset_name: Name of the dataset to check
            
        Returns:
            True if dataset is supported, False otherwise
        """
        return dataset_name in self.supported_datasets
    
    def set_seed(self, seed: int) -> None:
        """
        Set random seed for deterministic inference.
        
        This method should set seeds for all random number generators used
        by the model (numpy, torch, tensorflow, etc.).
        
        Args:
            seed: Random seed value
        """
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
