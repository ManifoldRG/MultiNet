from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path
import json


class BFCLDataset(Dataset):
    def __init__(self, questions_file: str, answers_file: str):
        """
        Initialize BFCL dataset for tool-use/function calling tasks.
        Always returns full multi-turn conversations.
        
        Args:
            questions_file: Path to questions JSONL file
            answers_file: Path to answers JSONL file
        """
        self.questions_file = Path(questions_file)
        self.answers_file = Path(answers_file)
        
        # Load and process data
        self.questions_data = self._load_questions()
        self.answers_data = self._load_answers()
        self.samples = self._create_conversation_pairs()
        
    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load questions from JSONL file."""
        if not self.questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {self.questions_file}")
        
        questions_data = []
        with open(self.questions_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if line.strip():  # Skip empty lines
                    try:
                        data = json.loads(line.strip())
                        data['line_id'] = line_idx  # Add line identifier
                        questions_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_idx + 1}: {e}")
                        continue
        
        print(f"Loaded {len(questions_data)} BFCL questions")
        return questions_data
    
    def _load_answers(self) -> List[Dict[str, Any]]:
        """Load answers from JSONL file."""
        if not self.answers_file.exists():
            raise FileNotFoundError(f"Answers file not found: {self.answers_file}")
        
        answers_data = []
        with open(self.answers_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if line.strip():  # Skip empty lines
                    try:
                        data = json.loads(line.strip())
                        data['line_id'] = line_idx  # Add line identifier
                        answers_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse answer line {line_idx + 1}: {e}")
                        continue
        
        print(f"Loaded {len(answers_data)} BFCL answers")
        return answers_data
    
    def _create_conversation_pairs(self) -> List[Dict[str, Any]]:
        """Create conversation-answer pairs by matching questions with answers."""
        # Create lookup dictionary for answers by ID
        answers_lookup = {}
        for answer in self.answers_data:
            conversation_id = answer.get('id')
            if conversation_id is not None:
                answers_lookup[conversation_id] = answer
        
        # Pair questions with their corresponding answers
        conversation_pairs = []
        missing_answers = 0
        
        for question in self.questions_data:
            conversation_id = question.get('id')
            if conversation_id is None:
                continue
                
            # Find corresponding answer
            answer = answers_lookup.get(conversation_id)
            if answer is None:
                missing_answers += 1
                print(f"Warning: No answer found for conversation {conversation_id}")
                continue
            
            # Create combined sample
            conversation_pair = {
                'conversation_id': conversation_id,
                'question_data': question,
                'answer_data': answer,
                'turns': question.get('question', []),
                'ground_truth': answer.get('ground_truth', []),
                'initial_config': question.get('initial_config', {}),
                'involved_classes': question.get('involved_classes', []),
                'path': question.get('path', [])
            }
            conversation_pairs.append(conversation_pair)
        
        if missing_answers > 0:
            print(f"Warning: {missing_answers} conversations have no corresponding answers")
        
        print(f"Created {len(conversation_pairs)} conversation pairs")
        return conversation_pairs
    
    
    def _format_full_conversation_prompt(self, conversation_data: Dict[str, Any]) -> str:
        """Format the complete multi-turn conversation as a tool-use prompt."""
        turns = conversation_data['turns']
        ground_truth = conversation_data['ground_truth']
        involved_classes = conversation_data['involved_classes']
        initial_config = conversation_data['initial_config']
        
        prompt_parts = []
        
        # Add initial context/environment setup
        if initial_config:
            prompt_parts.append("Initial Environment Configuration:")
            prompt_parts.append(f"{json.dumps(initial_config, indent=2)}")
            prompt_parts.append("")
        
        # Add available tools information
        if involved_classes:
            prompt_parts.append(f"Available Tool Classes: {', '.join(involved_classes)}")
            prompt_parts.append("")
        
        # Add multi-turn conversation
        prompt_parts.append("Multi-turn Conversation:")
        for turn_idx, turn_messages in enumerate(turns):
            prompt_parts.append(f"\n=== Turn {turn_idx + 1} ===")
            
            # Add user messages for this turn
            for msg in turn_messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                prompt_parts.append(f"{role.capitalize()}: {content}")
        
        # Add instruction
        prompt_parts.append("\n" + "="*50)
        prompt_parts.append("Task: For each turn in the conversation above, provide the sequence of function calls needed to complete the user's request.")
        prompt_parts.append("Format: Each function call should be in the format: function_name(param1=value1, param2=value2,...)")
        prompt_parts.append("Output the function calls for each turn in order.")
        
        return "\n".join(prompt_parts)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        total_turns = sum(len(conv['turns']) for conv in self.samples)
        avg_turns_per_conv = total_turns / len(self.samples) if self.samples else 0
        
        # Count involved classes
        all_classes = []
        for conv in self.samples:
            all_classes.extend(conv['involved_classes'])
        unique_classes = list(set(all_classes))
        
        return {
            'num_conversations': len(self.samples),
            'total_turns': total_turns,
            'avg_turns_per_conversation': avg_turns_per_conv,
            'unique_tool_classes': unique_classes,
            'num_tool_classes': len(unique_classes),
            'questions_file': str(self.questions_file),
            'answers_file': str(self.answers_file),
            'mode': 'full_conversations'
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _process_conversation_sample(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a full conversation sample."""
        turns = conversation_data['turns']
        ground_truth = conversation_data['ground_truth']
        
        # Format the complete conversation prompt
        formatted_prompt = self._format_full_conversation_prompt(conversation_data)
        
        # Create structured ground truth with turn-by-turn function calls
        structured_ground_truth = []
        for turn_idx, (turn_messages, expected_functions) in enumerate(zip(turns, ground_truth)):
            turn_ground_truth = {
                'turn_index': turn_idx,
                'user_messages': turn_messages,
                'expected_functions': expected_functions
            }
            structured_ground_truth.append(turn_ground_truth)
        
        # Create processed sample
        processed_sample = {
            'conversation_id': conversation_data['conversation_id'],
            'prompt': formatted_prompt,
            'turns': turns,
            'ground_truth_functions': ground_truth,
            'structured_ground_truth': structured_ground_truth,
            'initial_config': conversation_data['initial_config'],
            'involved_classes': conversation_data['involved_classes'],
            'path': conversation_data['path'],
            'num_turns': len(turns),
            'task_type': 'multi_turn_function_calling'
        }
        
        return processed_sample
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single conversation sample."""
        if idx < 0:
            idx = len(self) + idx
        
        # Always return full conversation
        conversation_data = self.samples[idx]
        processed_sample = self._process_conversation_sample(conversation_data)
        processed_sample['sample_id'] = idx
        return processed_sample


def custom_collate(batch):
    """Custom collate function for batching BFCL data."""
    result = defaultdict(list)
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    return result


def get_bfcl_dataloader(questions_file: str, answers_file: str, batch_size: int,
                       num_workers: int = 0, shuffle: bool = False) -> tuple:
    """
    Create BFCL dataloader for tool-use/function calling tasks.
    Always returns full multi-turn conversations.
    
    Args:
        questions_file: Path to questions JSONL file
        answers_file: Path to answers JSONL file
        batch_size: Batch size for the dataloader
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
    
    Returns:
        tuple: (dataset, dataloader) similar to other implementations
    """
    dataset = BFCLDataset(questions_file, answers_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate
    )
    return dataset, dataloader


def get_bfcl_test_dataloader(test_dir: str, batch_size: int,
                            num_workers: int = 0, shuffle: bool = False) -> tuple:
    """
    Create BFCL test dataloader using the processed test split.
    Always returns full multi-turn conversations.
    
    Args:
        test_dir: Path to directory containing test files
                 (e.g., "processed_datasets/bfcl_v3/test")
        batch_size: Batch size for the dataloader
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
    
    Returns:
        tuple: (dataset, dataloader)
    """
    test_path = Path(test_dir)
    
    # Look for standard test files
    questions_file = test_path / "questions.jsonl"
    answers_file = test_path / "answers.jsonl"
    
    if not questions_file.exists():
        raise FileNotFoundError(f"Test questions file not found: {questions_file}")
    if not answers_file.exists():
        raise FileNotFoundError(f"Test answers file not found: {answers_file}")
    
    return get_bfcl_dataloader(
        str(questions_file), 
        str(answers_file), 
        batch_size, 
        num_workers, 
        shuffle
    )


def get_bfcl_info(questions_file: str, answers_file: str) -> Dict[str, Any]:
    """
    Get BFCL dataset information without creating full dataloader.
    Always analyzes full conversations.
    
    Args:
        questions_file: Path to questions JSONL file
        answers_file: Path to answers JSONL file
    
    Returns:
        Dict containing dataset metadata
    """
    dataset = BFCLDataset(questions_file, answers_file)
    return dataset.get_dataset_info()
