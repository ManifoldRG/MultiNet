"""
Example BFCL Model Adapter with Multi-Turn History Support

This example demonstrates how to implement a model adapter for the BFCL dataset
that properly handles multi-turn conversation history.

Key Design:
- The observation parameter contains 'text_observation' with persistent context
- The history parameter contains previous turns (WITHOUT the current turn's messages)
- The instruction parameter contains the current user query
- The adapter builds: system_prompt + text_observation + history + current_instruction

From the dataloader, 'prompt' and 'turns' are used:
- 'prompt': Persistent observation/context (available functions) -> text_observation
- 'turns': Turn-specific user queries -> instruction per turn
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import re
from typing import Dict, Any, List, Optional
from src.eval_harness.model_adapter import ToolUseAdapter


class ExampleBFCLAdapter(ToolUseAdapter):
    """Example adapter for BFCL function calling tasks with multi-turn support."""
    
    def __init__(self):
        super().__init__(
            model_name="example-bfcl-model",
            supported_datasets=["bfcl"]
        )
        self.model = None
        self.tokenizer = None
    
    def initialize(self, device: str = "cuda", seed: int = 42, **kwargs) -> None:
        """Initialize the model and any necessary components."""
        self.set_seed(seed)
        
        # self.model = YourModel.from_pretrained("model-name")
        # self.tokenizer = YourTokenizer.from_pretrained("model-name")
        # self.model.to(device)
        # self.model.eval()
        
        print(f"Initialized {self.model_name} on {device}")
        self._is_initialized = True
    
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict function calls for BFCL task with multi-turn history.
        
        Args:
            observation: Dict with 'text_observation' key containing persistent context
            instruction: Current user query/instruction
            dataset_name: Should be "bfcl"
            history: Previous conversation turns (WITHOUT current turn)
                Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            **kwargs: Additional parameters
            
        Returns:
            Dict with:
                - "raw_output": str (full model response)
                - "extracted_outputs": List[str] (extracted function calls)
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        model_input = self._format_input_with_history(
            instruction=instruction,
            history=history,
            observation=observation
        )
        
        raw_output = self._generate_response(model_input)
        extracted_calls = self._extract_function_calls(raw_output)
        
        return {
            "raw_output": raw_output,
            "extracted_outputs": extracted_calls
        }
    
    def _format_input_with_history(
        self,
        instruction: str,
        history: Optional[List[Dict[str, str]]],
        observation: Optional[Dict[str, Any]]
    ) -> str:
        """
        Format model input with conversation history.
        
        Builds the full conversation context:
        1. System prompt
        2. Text observation (persistent context)
        3. Previous conversation history (if any)
        4. Current user instruction
        
        Note: The history does NOT contain the current turn's user message.
        That is provided separately via the instruction parameter.
        """
        parts = []
        
        # System prompt
        parts.append("System: You are an AI assistant that can call functions to complete tasks.")
        
        # Add text_observation as first user message
        if observation and 'text_observation' in observation:
            parts.append(f"User: {observation['text_observation']}")
        
        # Previous conversation turns (WITHOUT current turn)
        if history:
            parts.append("Conversation history:")
            for msg in history:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                parts.append(f"{role.capitalize()}: {content}")
            parts.append("")
        
        # Current user instruction
        parts.append(f"User: {instruction}")
        parts.append("Assistant: ")
        
        return "\n".join(parts)
    
    def _generate_response(self, model_input: str) -> str:
        """Generate function call response from model."""
        # inputs = self.tokenizer(model_input, return_tensors="pt")
        # outputs = self.model.generate(**inputs, max_new_tokens=150)
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return "answer: example_function(arg='value')"
    
    def _extract_function_calls(self, text: str) -> List[str]:
        """Extract function calls from model output."""
        pattern = r'\b\w+\s*\([^)]*\)'
        calls = re.findall(pattern, text)
        return self._normalize_calls(calls)
    
    def _normalize_calls(self, calls: List[str]) -> List[str]:
        """Normalize function call format."""
        normalized = []
        for call in calls:
            normalized_call = call.strip()
            normalized_call = re.sub(r'\s+', ' ', normalized_call)
            normalized.append(normalized_call)
        return normalized
    
    def _format_batch_inputs(
        self,
        observations: List[Optional[Dict[str, Any]]],
        instructions: Optional[List[str]] = None,
        histories: Optional[List[List[Dict[str, str]]]] = None
    ) -> List[str]:
        """Format multiple inputs for batched processing."""
        batch_inputs = []
        
        for i in range(len(observations)):
            instruction = instructions[i] if instructions else None
            history = histories[i] if histories else None
            observation = observations[i]
            
            formatted_input = self._format_input_with_history(
                instruction=instruction,
                history=history,
                observation=observation
            )
            batch_inputs.append(formatted_input)
        
        return batch_inputs
    
    def _generate_batch_responses(self, batch_inputs: List[str]) -> List[str]:
        """Generate responses for multiple inputs in parallel."""
        batch_outputs = []
        for input_text in batch_inputs:
            # inputs = self.tokenizer(batch_inputs, return_tensors="pt", padding=True)
            # outputs = self.model.generate(**inputs, max_new_tokens=150)
            # batch_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            batch_outputs.append(self._generate_response(input_text))
        
        return batch_outputs
    
    def batch_predict_actions(
        self,
        observations: List[Dict[str, Any]],
        instructions: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        histories: Optional[List[List[Dict[str, str]]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Batch prediction with parallel inference."""
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        batch_inputs = self._format_batch_inputs(
            observations=observations,
            instructions=instructions,
            histories=histories
        )
        
        batch_outputs = self._generate_batch_responses(batch_inputs)
        
        predictions = []
        for i, raw_output in enumerate(batch_outputs):
            extracted_calls = self._extract_function_calls(raw_output)
            predictions.append({
                "raw_output": raw_output,
                "extracted_outputs": extracted_calls
            })
        
        return predictions


# Example usage
if __name__ == "__main__":
    # Create adapter
    adapter = ExampleBFCLAdapter()
    adapter.initialize(device="cpu", seed=42)
    
    print("=== Single-Turn Prediction Example ===")
    
    # For BFCL, observation contains text_observation with persistent context
    observation = {
        'text_observation': 'You have access to Weather and Calendar functions. Weather.get_weather(city) and Calendar.add_event(title, date).'
    }
    
    # Example history (previous turn) - does NOT include current turn
    history = [
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "get_weather(city='default')"}
    ]
    
    # Current turn's user instruction (separate from history)
    instruction = "Add a reminder for tomorrow if it's going to rain"
    
    # Print full conversation
    print("\nFull Conversation:")
    print("Turn 1:")
    print(f"  User: What's the weather like?")
    print(f"  Assistant: get_weather(city='default')")
    print("Turn 2:")
    print(f"  User: {instruction}")
    
    # Make single prediction
    prediction = adapter.predict_action(
        observation=observation,
        instruction=instruction,
        dataset_name='bfcl',
        history=history
    )
    
    print(f"  Assistant: {prediction['raw_output']}")
    print(f"\nPrediction details:")
    print(f"  Raw output: {prediction['raw_output']}")
    print(f"  Extracted calls: {prediction['extracted_outputs']}")
    
    print("\n=== Batched Multi-Turn Prediction Example ===")
    
    # For BFCL, observations contain text_observation with persistent context
    # Three conversations at different turns: [turn 0, turn 1, turn 0]
    batch_observations = [
        {'text_observation': 'You have access to Weather functions. Weather.get_weather(city).'},
        {'text_observation': 'You have access to Weather and Calendar functions. Weather.get_weather(city) and Calendar.add_event(title, date).'},
        {'text_observation': 'You have access to Calendar functions. Calendar.add_event(title, date).'}
    ]
    
    batch_instructions = [
        "What's the weather?",
        "Set a reminder for tomorrow",
        "Add a meeting to my calendar"
    ]
    
    batch_histories = [
        [],  # First turn - no previous history
        [
            # Second turn - has history from first turn (WITHOUT current turn)
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "get_weather(city='default')"}
        ],
        []   # First turn - no previous history
    ]
    
    # Demonstrate batched inference
    print("Running batched inference for 3 conversations...")
    batch_predictions = adapter.batch_predict_actions(
        observations=batch_observations,
        instructions=batch_instructions,
        dataset_name='bfcl',
        histories=batch_histories
    )
    
    print(f"\nBatched Multi-Turn Conversations:")
    for i, (inst, hist, pred) in enumerate(zip(batch_instructions, batch_histories, batch_predictions)):
        print(f"\nConversation {i+1}:")
        
        # Print conversation history
        if hist:
            print("  Previous turns:")
            for j, msg in enumerate(hist):
                role = msg['role'].capitalize()
                content = msg['content']
                print(f"    {role}: {content}")
        
        # Print current turn
        print(f"  Current turn:")
        print(f"    User: {inst}")
        print(f"    Assistant: {pred['raw_output']}")
        
        print(f"  Prediction details:")
        print(f"    Raw output: {pred['raw_output']}")
        print(f"    Extracted calls: {pred['extracted_outputs']}")

