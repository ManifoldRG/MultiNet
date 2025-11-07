"""
Magma Model Adapter for BFCL Function Calling Tasks

This adapter integrates the Magma model with the BFCL (Berkeley Function Calling 
Leaderboard) evaluation framework, supporting multi-turn conversation history and 
function call extraction.
"""

import os
import sys
import re
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(ROOT_DIR)

from src.eval_harness.model_adapter import ModelAdapter


# BFCL system prompt for function calling tasks
BFCL_SYSTEM_PROMPT = """You are an AI assistant that can call functions to complete tasks. You will be presented with conversation histories where each turn may require function calls.

For each turn, analyze the conversation history, which may include previous assistant responses in addition to user prompts, and respond with the correct function to call.
Format each function call as: function_name(param1=value1, param2=value2, ...)
Use only the exact function names available in the provided set of functions and append appropriate parameters.
Output only the function calls, no explanations or additional text."""


class MagmaBFCLAdapter(ModelAdapter):
    """
    Adapter for Magma model on BFCL function calling tasks.
    
    This adapter handles multi-turn conversations and extracts function calls
    from model outputs for evaluation on the BFCL benchmark.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "microsoft/Magma-8B",
        torch_dtype: str = "bf16",
        device_map: str = "auto",
        max_answer_length: int = 150
    ):
        """
        Initialize the Magma BFCL adapter.
        
        Args:
            model_name_or_path: Path or identifier for the Magma model
            torch_dtype: Data type for model weights (bf16, fp16, or fp32)
            device_map: Device mapping strategy for model loading
            max_answer_length: Maximum number of tokens to generate
        """
        super().__init__()
        self.model_name = "magma"
        self.model_type = "tool_use"
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
        return ["bfcl"]

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
        Predict function calls for a BFCL task with multi-turn history.
        
        Args:
            observation: Observation containing BFCL task information
            instruction: Current user instruction/query
            dataset_name: Name of the dataset (should be "bfcl")
            history: Optional conversation history as list of message dicts
                Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with:
                - "raw_output": str (raw model output text)
                - "extracted_outputs": List[str] (extracted function calls)
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        # Validate dataset
        if dataset_name and not self.is_dataset_supported(dataset_name):
            raise NotImplementedError(f"Dataset {dataset_name} not supported by Magma adapter")
        
        if instruction is None:
            raise ValueError("Instruction is required for BFCL tasks")
        
        # Build conversation with history
        conversation = self._build_chat_history(
            history=history,
            instruction=instruction,
            observation=observation
        )
        
        # Generate response
        raw_output = self._generate_response(conversation, **kwargs)
        
        # Extract function calls
        extracted_calls = self._extract_function_calls(raw_output)
        
        return {
            "raw_output": raw_output,
            "extracted_outputs": extracted_calls
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
        Predict function calls for a batch of observations with parallel inference.
        
        Args:
            observations: List of observation dictionaries
            instructions: List of instruction strings
            dataset_name: Name of the dataset
            histories: Optional list of conversation histories
            **kwargs: Additional generation parameters
            
        Returns:
            List of prediction dictionaries, one per observation
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        batch_size = len(observations)
        
        if batch_size == 0:
            return []
        
        # Default to empty histories if not provided
        if histories is None:
            histories = [None] * batch_size
        
        # Validate instructions
        if instructions is None:
            raise ValueError("Instructions are required for BFCL batch prediction")
        
        if len(instructions) != batch_size:
            raise ValueError(f"Number of instructions ({len(instructions)}) must match observations ({batch_size})")
        
        # Build conversations for all samples
        conversations = []
        for i in range(batch_size):
            conversation = self._build_chat_history(
                history=histories[i],
                instruction=instructions[i],
                observation=observations[i]
            )
            conversations.append(conversation)
        
        # Generate batch responses
        raw_outputs = self._generate_batch_responses(conversations, **kwargs)
        
        # Extract function calls for each response
        predictions = []
        for raw_output in raw_outputs:
            extracted_calls = self._extract_function_calls(raw_output)
            predictions.append({
                "raw_output": raw_output,
                "extracted_outputs": extracted_calls
            })
        
        return predictions
    
    def _build_chat_history(
        self,
        history: Optional[List[Dict[str, str]]],
        instruction: str,
        observation: Optional[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Build conversation list for chat template.
        
        Args:
            history: Previous conversation messages (without current turn)
            instruction: Current user instruction
            observation: Dict with 'text_observation' key containing persistent context
            
        Returns:
            List of message dictionaries for chat template
        """
        conversation = []
        
        # Add system prompt
        conversation.append({
            "role": "system",
            "content": BFCL_SYSTEM_PROMPT
        })
        
        # Add text_observation as FIRST user message (persistent context)
        if observation and 'text_observation' in observation:
            conversation.append({
                "role": "user", 
                "content": observation['text_observation']
            })
        
        # Add conversation history if present
        if history:
            conversation.extend(history)
        
        # Add current user instruction
        conversation.append({
            "role": "user",
            "content": instruction
        })
        
        return conversation
    
    def _generate_response(
        self,
        conversation: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate response for a single conversation.
        
        Args:
            conversation: List of message dictionaries
            max_new_tokens: Maximum tokens to generate (overrides default)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        # Apply chat template
        input_ids = self.processor.tokenizer.apply_chat_template(
            conversation,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_answer_length,
            "temperature": kwargs.get("temperature", self.temperature),
            "do_sample": kwargs.get("do_sample", self.do_sample),
            "use_cache": True,
        }
        
        # Generate
        with torch.inference_mode():
            generate_ids = self.model.generate(input_ids=input_ids, **gen_kwargs)
            input_token_len = input_ids.shape[1]
            response = self.processor.batch_decode(
                generate_ids[:, input_token_len:],
                skip_special_tokens=True,
            )[0]
        
        return response.strip()
    
    def _generate_batch_responses(
        self,
        conversations: List[List[Dict[str, str]]],
        max_new_tokens: Optional[int] = None,
        max_seq_len: int = 1024,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of conversations in parallel.
        
        Args:
            conversations: List of conversation message lists
            max_new_tokens: Maximum tokens to generate
            max_seq_len: Maximum sequence length for tokenization
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated text responses
        """
        # Apply chat template with padding
        input_ids = self.processor.tokenizer.apply_chat_template(
            conversations,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            add_generation_prompt=True,
        ).to(self.model.device)
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_answer_length,
            "temperature": kwargs.get("temperature", self.temperature),
            "do_sample": kwargs.get("do_sample", self.do_sample),
            "use_cache": True,
        }
        
        # Generate batch
        with torch.inference_mode():
            generate_ids = self.model.generate(input_ids=input_ids, **gen_kwargs)
            input_token_len = input_ids.shape[1]
            responses = self.processor.batch_decode(
                generate_ids[:, input_token_len:],
                skip_special_tokens=True,
            )
        
        return [response.strip() for response in responses]
    
    def _extract_function_calls(self, text: str) -> List[str]:
        """
        Extract function calls from model output using regex.
        
        Args:
            text: Raw model output text
            
        Returns:
            List of extracted function call strings
        """
        # Pattern to match function calls: function_name(args...)
        pattern = r'\b\w+\s*\([^)]*\)'
        calls = re.findall(pattern, text)
        
        # Normalize function calls
        normalized_calls = self._normalize_function_calls(calls)
        
        return normalized_calls
    
    def _normalize_function_calls(self, calls: List[str]) -> List[str]:
        """
        Normalize function call format.
        
        Args:
            calls: List of raw function call strings
            
        Returns:
            List of normalized function call strings
        """
        normalized = []
        for call in calls:
            # Remove extra whitespace
            normalized_call = call.strip()
            normalized_call = re.sub(r'\s+', ' ', normalized_call)
            normalized.append(normalized_call)
        
        return normalized
    
    def reset(self) -> None:
        """Reset model state (clears cache if applicable)."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

