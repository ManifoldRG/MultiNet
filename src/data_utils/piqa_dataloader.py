from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any
from collections import defaultdict
import json
from pathlib import Path


class PIQADataset(Dataset):
    def __init__(self, data_file: str):
        """
        Initialize PIQA dataset.
        
        Args:
            data_file: Path to the JSONL file containing PIQA data
        """
        self.data_file = data_file
        
        # Load all data
        self.samples = []
        self._load_all_data()
    
    def _load_all_data(self):
        """Load data from JSONL file."""
        print(f"Loading PIQA data from {self.data_file}")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data = json.loads(line.strip())
                    self.samples.append(data)
        
        print(f"Loaded {len(self.samples)} PIQA samples")
    

    
    def _process_sample(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single PIQA sample into the expected format."""
        
        # Create the question text combining goal and solutions
        goal = sample_data.get('goal', '')
        sol1 = sample_data.get('sol1', '')
        sol2 = sample_data.get('sol2', '')
        
        # Format the question as a multiple choice question
        question_text = f"Goal: {goal}\nSolution 1: {sol1}\nSolution 2: {sol2}\nWhich solution is better for the given goal?"
        
        # The label indicates which solution is correct (0 for sol1, 1 for sol2)
        label = sample_data.get('label', 0)
        correct_answer = sol1 if label == 0 else sol2
        
        # Create processed sample
        processed_sample = {
            'text_observation': question_text,
            'goal': goal,
            'sol1': sol1,
            'sol2': sol2,
            'correct_solution': correct_answer,
            'sample_id': sample_data.get('id', ''),
            'line_id': sample_data.get('line_id', 0)
        }
        
        return processed_sample
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Return single sample
        if idx < 0:
            idx = len(self.samples) + idx
        
        sample_data = self.samples[idx]
        return self._process_sample(sample_data)
    



def custom_collate(batch):
    """Custom collate function for batching PIQA data."""
    result = defaultdict(list)
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    return result


def get_piqa_dataloader(data_file: str, batch_size: int, 
                       num_workers: int = 0) -> tuple:
    """
    Create PIQA dataloader.
    
    Args:
        data_file: Path to JSONL file containing PIQA data
        batch_size: Batch size for the dataloader
        num_workers: Number of worker processes for data loading
    
    Returns:
        tuple: (dataset, dataloader) similar to other implementations
    """
    dataset = PIQADataset(data_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate
    )
    return dataset, dataloader


def get_piqa_test_dataloader(test_dir: str, batch_size: int, 
                           num_workers: int = 0) -> tuple:
    """
    Create PIQA test dataloader using the processed test split.
    
    Args:
        test_dir: Path to directory containing test.jsonl
        batch_size: Batch size for the dataloader
        num_workers: Number of worker processes for data loading
    
    Returns:
        tuple: (dataset, dataloader)
    """
    test_file = Path(test_dir) / "test.jsonl"
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    return get_piqa_dataloader(str(test_file), batch_size, num_workers)
