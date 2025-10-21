"""
ModelAdapter Interface for MultiNet Evaluation Harness

This module defines the abstract base class that users must implement to integrate
their models with the MultiNet benchmarking toolkit. The interface standardizes
model interactions while allowing flexibility for different task types.

Users should inherit from the appropriate task-specific adapter and implement 
the required abstract methods for their specific model.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import torch

# TODO: the individual inherited adapters may not be fully implemented
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
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        supported_datasets: List[str]
    ):
        """
        Initialize the ModelAdapter.
        
        Args:
            model_name: Human-readable name for the model
            model_type: Type of task this adapter handles
            supported_datasets: List of datasets this model supports
        """
        self.model_name = model_name
        self.model_type = model_type
        self.supported_datasets = supported_datasets
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
    
    @abstractmethod
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Union[np.ndarray, List[float], int, str]:
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
    ) -> List[Union[np.ndarray, List[float], int, str]]:
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
    
    def preprocess_observation(
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
    
    def postprocess_action(
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
            "model_name": self.model_name,
            "model_type": self.model_type,
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


# Task-specific adapters (one per task type)
class MultipleChoiceAdapter(ModelAdapter):
    """
    Adapter for Multiple Choice Question (MCQ) tasks.
    
    Used for: PIQA
    Task type: Choose one option from multiple choices
    Output: Choice index (0 to num_choices-1)
    """
    
    def __init__(
        self,
        model_name: str,
        supported_datasets: List[str],
        num_choices: int = 4
    ):
        super().__init__(
            model_name=model_name,
            model_type="multiple_choice",
            supported_datasets=supported_datasets
        )
        self.num_choices = num_choices
    
    @abstractmethod
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        return_probabilities: bool = False,
        **kwargs
    ) -> Union[int, Dict[str, Any]]:
        """
        Predict choice for MCQ task.
        
        Args:
            observation: Observation containing 'question', 'choices', and optionally 'image'
            instruction: Optional task instruction
            dataset_name: Name of the dataset
            history: Optional conversation history (typically not used for MCQ tasks)
            return_probabilities: Whether to return choice probabilities
            **kwargs: Additional prediction parameters
            
        Returns:
            Choice index (int) or dict with choice and probabilities
        """
        pass


class TextGenerationAdapter(ModelAdapter):
    """
    Adapter for text generation tasks.
    
    Used for: SQA3D
    Task type: Generate text response to visual question or reasoning prompt
    Output: Text string
    """
    
    def __init__(
        self,
        model_name: str,
        supported_datasets: List[str],
        max_answer_length: int = 100
    ):
        super().__init__(
            model_name=model_name,
            model_type="text_generation",
            supported_datasets=supported_datasets
        )
        self.max_answer_length = max_answer_length
    
    @abstractmethod
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Generate text response.
        
        Args:
            observation: Observation containing 'image' and 'question' or prompt
            instruction: Optional task instruction
            dataset_name: Name of the dataset
            history: Optional conversation history (typically not used for single-turn VQA)
            **kwargs: Additional prediction parameters
            
        Returns:
            Generated text response
        """
        pass


class GroundingAdapter(ModelAdapter):
    """
    Adapter for entity grounding tasks.
    
    Used for: ODinW
    Task type: Match entities to bounding boxes or image regions
    Output: Entity-to-bbox mapping (indices or selections)
    """
    
    def __init__(
        self,
        model_name: str,
        supported_datasets: List[str]
    ):
        super().__init__(
            model_name=model_name,
            model_type="grounding",
            supported_datasets=supported_datasets
        )
    
    @abstractmethod
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Union[int, List[int], Dict[str, int]]:
        """
        Predict entity-to-bbox mappings.
        
        Args:
            observation: Observation containing:
                - 'image': Main image
                - 'bbox_images': List of cropped bounding box images
                - 'caption': Text with entities to ground
                - 'entities': List of entity mentions
            instruction: Optional task instruction
            dataset_name: Name of the dataset
            history: Optional conversation history (typically not used for grounding tasks)
            **kwargs: Additional prediction parameters
            
        Returns:
            Entity-to-bbox mapping (int index, list of indices, or dict mapping)
        """
        pass


class DiscreteActionAdapter(ModelAdapter):
    """
    Adapter for discrete action tasks.
    
    Used for: OvercookedAI
    Task type: Select one discrete action from fixed action space
    Output: Action index (0 to action_space_size-1)
    """
    
    def __init__(
        self,
        model_name: str,
        supported_datasets: List[str],
        action_space_size: int
    ):
        super().__init__(
            model_name=model_name,
            model_type="discrete_action",
            supported_datasets=supported_datasets
        )
        self.action_space_size = action_space_size
    
    @abstractmethod
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        return_probabilities: bool = False,
        **kwargs
    ) -> Union[int, Dict[str, Any]]:
        """
        Predict discrete action.
        
        Args:
            observation: Observation containing 'image' and optional 'state'
            instruction: Optional task instruction
            dataset_name: Name of the dataset
            history: Optional conversation history (typically not used for action prediction)
            return_probabilities: Whether to return action probabilities
            **kwargs: Additional prediction parameters
            
        Returns:
            Action index (int) or dict with action and probabilities
        """
        pass


class ContinuousActionAdapter(ModelAdapter):
    """
    Adapter for continuous action tasks.
    
    Used for: OpenX
    Task type: Predict continuous action vector
    Output: Continuous action array (e.g., robot joint positions/velocities)
    """
    
    def __init__(
        self,
        model_name: str,
        supported_datasets: List[str],
        action_dim: int,
        action_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        super().__init__(
            model_name=model_name,
            model_type="continuous_action",
            supported_datasets=supported_datasets
        )
        self.action_dim = action_dim
        self.action_bounds = action_bounds
    
    @abstractmethod
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Predict continuous action.
        
        Args:
            observation: Observation containing images, state, etc.
            instruction: Optional task instruction
            dataset_name: Name of the dataset
            history: Optional conversation history (typically not used for action prediction)
            **kwargs: Additional prediction parameters
            
        Returns:
            Continuous action vector
        """
        pass

class ToolUseAdapter(ModelAdapter):
    """
    Adapter for function calling and tool use tasks.
    
    Used for: BFCL (Berkeley Function Calling Leaderboard)
    Task type: Evaluate LLM's ability to invoke external functions, APIs, or tools
    Output: Function calls, tool invocations, or abstention decisions
    
    BFCL evaluates:
    - Serial and parallel function calls across programming languages
    - Memory and dynamic decision-making in multi-step scenarios
    - Long-horizon reasoning and stateful agentic behavior
    - Ability to abstain when appropriate
    
    Multi-turn History:
    - For multi-turn benchmarks, the evaluation harness manages conversation history
    - The adapter receives the full conversation history via the 'history' parameter
    - The adapter should format the history appropriately for the model (e.g., apply chat template)
    - The adapter is stateless: it receives full context and returns a single response
    """
    
    def __init__(
        self,
        model_name: str,
        supported_datasets: List[str]
    ):
        super().__init__(
            model_name=model_name,
            model_type="tool_use",
            supported_datasets=supported_datasets
        )

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
        Predict function calls or tool invocations.

        Args:
            observation: Observation containing available functions and state
            instruction: User query requiring tool use
            dataset_name: Name of the dataset
            history: Optional conversation history for multi-turn scenarios
            **kwargs: Additional prediction parameters
            
        Returns:
            Dictionary with "raw_output" and "extracted_calls" keys
        """
        pass
    
