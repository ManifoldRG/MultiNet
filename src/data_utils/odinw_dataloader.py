from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path
import json
import numpy as np


class ODinWDataset(Dataset):
    def __init__(self, dataset_dir: str, transform=None):
        """
        Initialize ODinW dataset for classification.
        
        Args:
            dataset_dir: Path to specific ODinW sub-dataset directory 
                        (e.g., "processed_datasets/odinw/test/BCCD")
            transform: Optional image transforms to apply
        """
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        
        # Load dataset metadata
        self.dataset_name = self.dataset_dir.name
        self.categories = self._load_categories()
        self.samples = self._load_samples()
        
        # Create mappings for easy access
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.categories['categories']}
        self.category_name_to_id = {cat['name']: cat['id'] for cat in self.categories['categories']}
        
    def _load_categories(self) -> Dict[str, Any]:
        """Load category information from .."""
        categories_file = self.dataset_dir / "categories.json"
        if not categories_file.exists():
            raise FileNotFoundError(f"Categories file not found: {categories_file}")
        
        with open(categories_file, 'r') as f:
            categories_data = json.load(f)
            
        # Filter out categories where supercategory == "none"
        filtered = [c for c in categories_data["categories"] if c["supercategory"] != "none"]

        # Re-assign IDs sequentially starting from 0
        for new_id, cat in enumerate(filtered):
            cat["id"] = new_id

        # Update metadata
        categories_data["categories"] = filtered
        categories_data["num_categories"] = len(filtered)
        categories_data["category_names"] = [c["name"] for c in filtered]
        
        return categories_data
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load sample data from object_caption_pairs.json."""

        test_pairs_file = self.dataset_dir / "test_object_caption_pairs.json"
        pairs_file = self.dataset_dir / "object_caption_pairs.json"
        samples_file = None
        
        if test_pairs_file.exists():
            samples_file = test_pairs_file
        elif pairs_file.exists():
            samples_file = pairs_file
        else:
            raise FileNotFoundError(f"No object caption pairs file found in {self.dataset_dir}")
        
        
        with open(samples_file, 'r') as f:
            samples_data = json.load(f)
        
        # Validate that all required fields are present
        required_fields = ['bbox_image', 'category_id', 'category_name']
        for i, sample in enumerate(samples_data):
            for field in required_fields:
                if field not in sample:
                    raise ValueError(f"Missing required field '{field}' in sample {i}")
        
        return samples_data
    
    def _load_image(self, image_filename: str) -> Image.Image:
        """Load image from bbox_images directory."""
        image_path = self.dataset_dir / "bbox_images" / image_filename
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            image = Image.open(image_path)
            # Convert to RGB if not already (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
    
    def get_num_classes(self) -> int:
        """Get number of classification classes."""
        return self.categories['num_categories']
    
    def get_category_names(self) -> List[str]:
        """Get list of category names."""
        return self.categories['category_names']
    
    def get_dataset_name(self) -> str:
        """Get the name of this ODinW sub-dataset."""
        return self.dataset_name
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of samples across classes."""
        distribution = defaultdict(int)
        for sample in self.samples:
            category_name = sample['category_name']
            distribution[category_name] += 1
        return dict(distribution)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        return {
            'dataset_name': self.dataset_name,
            'num_classes': self.get_num_classes(),
            'num_samples': len(self.samples),
            'category_names': self.get_category_names(),
            'class_distribution': self.get_class_distribution(),
            'categories_metadata': self.categories
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _create_classification_question(self, sample_data: Dict[str, Any]) -> str:
        """Create a classification question prompt similar to PIQA format."""
        category_names = self.get_category_names()
        correct_category = sample_data['category_name']
        
        # Create options list with the correct answer and other categories
        options = []
        correct_idx = None
        
        options = category_names
        correct_idx = category_names.index(correct_category)
        
        # Create the question text
        question_text = f"What object is shown in this image from the {self.dataset_name} dataset?\n"
        
        for i, option in enumerate(options):
            question_text += f"Option {i}: {option}\n"
        
        question_text += f"Output the number (0-{len(options)-1}) of the correct option only."
        
        return question_text, correct_idx, options

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        if idx < 0:
            idx = len(self.samples) + idx
        
        sample_data = self.samples[idx]
        
        # Load image
        image = self._load_image(sample_data['bbox_image'])
        
        # Apply transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        
        # Create classification question
        question_text, correct_option_idx, options = self._create_classification_question(sample_data)
        
        # Create sample dictionary
        sample = {
            'image': image,
            'question': question_text,
            'correct_option_idx': correct_option_idx,
            'options': options,
            'category_id': sample_data['category_id'],
            'category_name': sample_data['category_name'],
            'dataset_name': self.dataset_name,
            'bbox_info': {
                'bbox': sample_data.get('bbox', []),
                'bbox_id': sample_data.get('bbox_id', idx),
                'annotation_id': sample_data.get('annotation_id', idx)
            },
            'sample_id': idx,
            'image_filename': sample_data['bbox_image']
        }
        
        return sample


def custom_collate(batch):
    """Custom collate function for batching ODinW data."""
    result = defaultdict(list)
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    return result


def get_odinw_dataloader(dataset_dir: str, 
                        batch_size: int,
                        num_workers: int = 0,
                        transform=None,
                        shuffle: bool = False) -> tuple:
    """
    Create ODinW classification dataloader for a specific sub-dataset.
    
    Args:
        dataset_dir: Path to specific ODinW sub-dataset directory 
                    (e.g., "processed_datasets/odinw/test/BCCD")
        batch_size: Batch size for the dataloader
        num_workers: Number of worker processes for data loading
        transform: Optional image transforms to apply
        shuffle: Whether to shuffle the data
    
    Returns:
        tuple: (dataset, dataloader) similar to other implementations
    """
    dataset = ODinWDataset(dataset_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate
    )
    return dataset, dataloader


def get_odinw_classification_info(dataset_dir: str) -> Dict[str, Any]:
    """
    Get classification metadata for a specific ODinW sub-dataset without creating full dataset.
    
    Args:
        dataset_dir: Path to specific ODinW sub-dataset directory
    
    Returns:
        Dict containing dataset metadata
    """
    dataset = ODinWDataset(dataset_dir)
    return dataset.get_dataset_info()


def list_available_odinw_datasets(odinw_root_dir: str) -> List[str]:
    """
    List all available ODinW sub-datasets in the root directory.
    
    Args:
        odinw_root_dir: Path to ODinW root directory (e.g., "processed_datasets/odinw/test")
    
    Returns:
        List of available dataset names
    """
    root_path = Path(odinw_root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"ODinW root directory not found: {root_path}")
    
    available_datasets = []
    for dataset_dir in root_path.iterdir():
        if dataset_dir.is_dir():
            # Check if it has the required files
            categories_file = dataset_dir / "categories.json"
            bbox_images_dir = dataset_dir / "bbox_images"
            pairs_file = dataset_dir / "object_caption_pairs.json"
            test_pairs_file = dataset_dir / "test_object_caption_pairs.json"
            
            if (categories_file.exists() and 
                bbox_images_dir.exists() and bbox_images_dir.is_dir() and
                (pairs_file.exists() or test_pairs_file.exists())):
                available_datasets.append(dataset_dir.name)
    
    return sorted(available_datasets)


def get_multi_odinw_info(odinw_root_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available ODinW datasets.
    
    Args:
        odinw_root_dir: Path to ODinW root directory
    
    Returns:
        Dict mapping dataset names to their metadata
    """
    available_datasets = list_available_odinw_datasets(odinw_root_dir)
    multi_info = {}
    
    for dataset_name in available_datasets:
        try:
            dataset_path = Path(odinw_root_dir) / dataset_name
            info = get_odinw_classification_info(str(dataset_path))
            multi_info[dataset_name] = info
        except Exception as e:
            print(f"Warning: Could not load info for {dataset_name}: {e}")
            continue
    
    return multi_info
