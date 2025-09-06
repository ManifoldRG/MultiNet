from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path
import json
import numpy as np


class SQA3DDataset(Dataset):
    def __init__(self, questions_file: str, annotations_file: str, images_dir: str):
        """
        Initialize SQA3D dataset for VQA task.
        
        Args:
            questions_file: Path to questions JSON file (e.g., v1_balanced_questions_test_scannetv2.json)
            annotations_file: Path to annotations JSON file (e.g., v1_balanced_sqa_annotations_test_scannetv2.json)
            images_dir: Path to directory containing scene images (PNG files from PLY conversion)
        """
        self.questions_file = Path(questions_file)
        self.annotations_file = Path(annotations_file)
        self.images_dir = Path(images_dir)
        
        # Load and process data
        self.questions_data = self._load_questions()
        self.annotations_data = self._load_annotations()
        self.samples = self._create_qa_pairs()
        
        # Create scene ID to image path mapping
        self.scene_to_image = self._create_scene_image_mapping()
        
    def _load_questions(self) -> Dict[str, Any]:
        """Load questions from JSON file."""
        if not self.questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {self.questions_file}")
        
        with open(self.questions_file, 'r') as f:
            questions_data = json.load(f)
        
        if 'questions' not in questions_data:
            raise ValueError(f"Questions file must contain 'questions' key: {self.questions_file}")
        
        return questions_data
    
    def _load_annotations(self) -> Dict[str, Any]:
        """Load annotations from JSON file."""
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        
        with open(self.annotations_file, 'r') as f:
            annotations_data = json.load(f)
        
        if 'annotations' not in annotations_data:
            raise ValueError(f"Annotations file must contain 'annotations' key: {self.annotations_file}")
        
        return annotations_data
    
    def _create_qa_pairs(self) -> List[Dict[str, Any]]:
        """Create question-answer pairs by matching questions with annotations."""
        questions = self.questions_data['questions']
        annotations = self.annotations_data['annotations']
        
        # Create lookup dictionary for annotations by question_id
        annotations_lookup = {}
        for annotation in annotations:
            question_id = annotation.get('question_id')
            if question_id is not None:
                annotations_lookup[question_id] = annotation
        
        # Pair questions with their corresponding annotations
        qa_pairs = []
        missing_annotations = 0
        
        for question in questions:
            question_id = question.get('question_id')
            if question_id is None:
                continue
                
            # Find corresponding annotation
            annotation = annotations_lookup.get(question_id)
            if annotation is None:
                missing_annotations += 1
                continue
            
            # Create combined sample
            qa_pair = {
                'question_id': question_id,
                'scene_id': question.get('scene_id', ''),
                'question': question.get('question', ''),
                'answer': annotation['answers'][0].get('answer', ''),
                'question_type': question.get('question_type', ''),
                'answer_type': annotation['answers'][0].get('answer_type', ''),
                'situation': question.get('situation', ''),
                'alternative': question.get('alternative', ''),
                'question_data': question,  # Keep original question data
                'annotation_data': annotation  # Keep original annotation data
            }
            qa_pairs.append(qa_pair)
        
        if missing_annotations > 0:
            print(f"Warning: {missing_annotations} questions have no corresponding annotations")
        
        print(f"Created {len(qa_pairs)} question-answer pairs")
        return qa_pairs
    
    def _create_scene_image_mapping(self) -> Dict[str, Path]:
        """Create mapping from scene IDs to image file paths."""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        scene_to_image = {}
        
        # Look for PNG files in the images directory and subdirectories
        for image_path in self.images_dir.rglob("*.png"):
            # Extract scene ID from filename
            # Expected format: scene0015_00_vh_clean_top_down.png or similar
            filename = image_path.stem
            
            # Try different patterns to extract scene ID
            scene_id = None
            
            # Pattern 1: scene0015_00_vh_clean_top_down -> scene0015_00
            if filename.startswith('scene') and '_vh_clean' in filename:
                parts = filename.split('_vh_clean')[0]
                scene_id = parts
            # Pattern 2: scene0015_00 -> scene0015_00
            elif filename.startswith('scene') and len(filename.split('_')) >= 2:
                parts = filename.split('_')
                if len(parts) >= 2:
                    scene_id = f"{parts[0]}_{parts[1]}"
            # Pattern 3: Direct scene ID
            elif filename.startswith('scene'):
                scene_id = filename
            
            if scene_id:
                scene_to_image[scene_id] = image_path
        
        print(f"Found {len(scene_to_image)} scene images")
        return scene_to_image
    
    def _load_scene_image(self, scene_id: str) -> Optional[np.ndarray]:
        """Load scene image as numpy array."""
        image_path = self.scene_to_image.get(scene_id)
        
        if image_path is None:
            print(f"Warning: No image found for scene {scene_id}")
            return None
        
        if not image_path.exists():
            print(f"Warning: Image file does not exist: {image_path}")
            return None
        
        try:
            # Load image using PIL
            image = Image.open(image_path)
            
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image, dtype=np.uint8)
            return image_array
            
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {e}")
            return None
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        question_types = defaultdict(int)
        answer_types = defaultdict(int)
        
        for sample in self.samples:
            if sample['question_type']:
                question_types[sample['question_type']] += 1
            if sample['answer_type']:
                answer_types[sample['answer_type']] += 1
        
        return {
            'num_samples': len(self.samples),
            'num_scenes': len(self.scene_to_image),
            'question_types': dict(question_types),
            'answer_types': dict(answer_types),
            'questions_file': str(self.questions_file),
            'annotations_file': str(self.annotations_file),
            'images_dir': str(self.images_dir)
        }
    
    def get_available_scenes(self) -> List[str]:
        """Get list of available scene IDs."""
        return list(self.scene_to_image.keys())
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _process_sample(self, sample_data: Dict[str, Any], scene_image: Optional[np.ndarray]) -> Dict[str, Any]:
        """Process a single SQA3D sample into the expected format, similar to PIQA."""
        
        # Get the original question and context
        original_question = sample_data.get('question', '')
        situation = sample_data.get('situation', '')
        alternative = sample_data.get('alternative', '')
        scene_id = sample_data.get('scene_id', '')
        
        # Create formatted question text similar to PIQA
        question_parts = []
        
        # Add scene context
        question_parts.append(f"Scene: {scene_id}")
        
        # Add situation if available
        if situation:
            question_parts.append(f"Situation: {situation}")
        
        # Add the main question
        question_parts.append(f"Question: {original_question}")
        
        # Add alternative context if available
        if alternative:
            question_parts.append(f"Alternative: {alternative}")
        
        # Add instruction for VQA task
        question_parts.append("Please answer the question/alternative question based on the provided scene image and context situation.")
        
        # Combine all parts into formatted question
        formatted_question = "\n".join(question_parts)
        
        # Create processed sample
        processed_sample = {
            'question': formatted_question,
            'original_question': original_question,
            'answer': sample_data.get('answer', ''),
            'scene_image': scene_image,
            'scene_id': scene_id,
            'question_id': sample_data.get('question_id'),
            'question_type': sample_data.get('question_type', ''),
            'answer_type': sample_data.get('answer_type', ''),
            'situation': situation,
            'alternative': alternative,
            'has_image': scene_image is not None
        }
        
        return processed_sample
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single VQA sample."""
        if idx < 0:
            idx = len(self.samples) + idx
        
        sample_data = self.samples[idx]
        
        # Load scene image
        scene_image = self._load_scene_image(sample_data['scene_id'])
        
        # Process sample similar to PIQA
        processed_sample = self._process_sample(sample_data, scene_image)
        
        # Add sample index
        processed_sample['sample_id'] = idx
        
        return processed_sample


def custom_collate(batch):
    """Custom collate function for batching SQA3D VQA data."""
    result = defaultdict(list)
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    return result


def get_sqa3d_dataloader(questions_file: str, annotations_file: str, images_dir: str,
                        batch_size: int, num_workers: int = 0, shuffle: bool = False) -> tuple:
    """
    Create SQA3D VQA dataloader.
    
    Args:
        questions_file: Path to questions JSON file
        annotations_file: Path to annotations JSON file  
        images_dir: Path to directory containing scene images
        batch_size: Batch size for the dataloader
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
    
    Returns:
        tuple: (dataset, dataloader) similar to other implementations
    """
    dataset = SQA3DDataset(questions_file, annotations_file, images_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate
    )
    return dataset, dataloader


def get_sqa3d_test_dataloader(test_dir: str, batch_size: int, 
                             num_workers: int = 0, shuffle: bool = False) -> tuple:
    """
    Create SQA3D test dataloader using the processed test split.
    
    Args:
        test_dir: Path to directory containing test files
                 (e.g., "processed_datasets/sqa3d/test")
        batch_size: Batch size for the dataloader
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
    
    Returns:
        tuple: (dataset, dataloader)
    """
    test_path = Path(test_dir)
    
    # Look for standard test files
    questions_file = test_path / "v1_balanced_questions_test_scannetv2.json"
    annotations_file = test_path / "v1_balanced_sqa_annotations_test_scannetv2.json"
    
    if not questions_file.exists():
        raise FileNotFoundError(f"Test questions file not found: {questions_file}")
    if not annotations_file.exists():
        raise FileNotFoundError(f"Test annotations file not found: {annotations_file}")
    
    # Images should be in scene subdirectories
    images_dir = test_path
    
    return get_sqa3d_dataloader(
        str(questions_file), 
        str(annotations_file), 
        str(images_dir),
        batch_size, 
        num_workers, 
        shuffle
    )


def get_sqa3d_info(questions_file: str, annotations_file: str, images_dir: str) -> Dict[str, Any]:
    """
    Get SQA3D dataset information without creating full dataloader.
    
    Args:
        questions_file: Path to questions JSON file
        annotations_file: Path to annotations JSON file
        images_dir: Path to directory containing scene images
    
    Returns:
        Dict containing dataset metadata
    """
    dataset = SQA3DDataset(questions_file, annotations_file, images_dir)
    return dataset.get_dataset_info()
