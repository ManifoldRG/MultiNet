"""
Centralized Dataset Processor v1.0

Processes and standardizes datasets downloaded via ./centralized_downloader.py.
Creates evaluation splits where necessary with reproducible identifiers.
"""

import abc
import json
import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import io
import csv
import pickle
import pygame
import pandas as pd
import subprocess

# Add the submodule to the path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'third_party' / 'overcooked_ai' / 'src'))

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, Recipe
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from ..utils.ply_to_2d import ply_to_top_down_png


# ============================================================================
# Core Types
# ============================================================================

@dataclass
class ProcessResult:
    """Result of dataset processing."""
    name: str
    success: bool
    output_path: Path
    test_split_created: bool = False
    test_identifiers_saved: bool = False
    files_processed: int = 0
    error: Optional[str] = None


@dataclass
class ProcessingStatus:
    """Status information returned by individual processors."""
    success: bool
    test_split_created: bool = False
    test_identifiers_saved: bool = False
    files_processed: int = 0
    error_message: Optional[str] = None


# ============================================================================
# Base Processor
# ============================================================================

class BaseProcessor(abc.ABC):
    """Base class for all dataset processors."""
    
    def __init__(self, name: str, input_dir: Path, output_dir: Path):
        self.name = name
        self.input_dir = input_dir / name
        self.output_dir = output_dir / name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{name}_processor")
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
    
    @abc.abstractmethod
    def process(self) -> ProcessResult:
        """Process the dataset. Return ProcessResult with details of what was done."""
        pass
    
    def verify_input(self) -> bool:
        """Verify input directory exists and has data."""
        if not self.input_dir.exists():
            self.logger.error(f"Input directory does not exist: {self.input_dir}")
            return False
        
        if not any(self.input_dir.iterdir()):
            self.logger.error(f"Input directory is empty: {self.input_dir}")
            return False
        
        return True
    
    def run(self) -> ProcessResult:
        """Main processing workflow."""
        try:
            self.logger.info(f"Processing {self.name}...")
            
            if not self.verify_input():
                return ProcessResult(self.name, False, self.output_dir, error="Input verification failed")
            
            result = self.process()
            
            if not result.success:
                error_msg = result.error or "Processing failed"
                return ProcessResult(self.name, False, self.output_dir, error=error_msg)
            
            self.logger.info(f"✓ {self.name} processed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"✗ {self.name} failed: {e}")
            return ProcessResult(self.name, False, self.output_dir, error=str(e))
    
    def save_test_identifiers(self, identifiers: List[Union[str, int]], filename: str = "test_identifiers.json") -> Path:
        """Save test split identifiers for reproducibility."""
        identifiers_file = self.output_dir / filename
        with open(identifiers_file, 'w') as f:
            json.dump({
                "test_identifiers": identifiers,
                "total_count": len(identifiers),
                "dataset": self.name
            }, f, indent=2)
        return identifiers_file


# ============================================================================
# Dataset Processors
# ============================================================================

class PIQAProcessor(BaseProcessor):
    """PIQA dataset processor with 20% test split creation."""
    
    def process(self) -> ProcessResult:
        """Process PIQA dataset and create test split."""
        try:
            # Locate PIQA files
            piqa_dir = self.input_dir / "physicaliqa-train-dev"
            if not piqa_dir.exists():
                self.logger.error("PIQA physicaliqa-train-dev directory not found")
                return ProcessResult(self.name, False, self.output_dir, error="PIQA physicaliqa-train-dev directory not found")
            
            train_file = piqa_dir / "train.jsonl"
            train_labels = piqa_dir / "train-labels.lst"
            
            # Load training data
            train_data = []
            with open(train_file, 'r') as f:
                for line_idx, line in enumerate(f):
                    data = json.loads(line.strip())
                    data['line_id'] = line_idx  # Add line identifier
                    train_data.append(data)
            
            # Load training labels
            with open(train_labels, 'r') as f:
                labels = [int(line.strip()) for line in f]
            
            # Add labels to data
            for i, data_point in enumerate(train_data):
                if i < len(labels):
                    data_point['label'] = labels[i]
            
            # Create 20% test split
            test_size = int(0.2 * len(train_data))
            test_indices = random.sample(range(len(train_data)), test_size)
            test_ids = [train_data[i]['id'] for i in test_indices]
            
            # Split data using IDs for reproducibility
            test_ids_set = set(test_ids)
            test_data = [data_point for data_point in train_data if data_point['id'] in test_ids_set]
            
            # Create output directories
            test_output = self.output_dir / "test"
            test_output.mkdir(exist_ok=True)
            
            
            # Save test data
            with open(test_output / "test.jsonl", 'w') as f:
                for data_point in test_data:
                    f.write(json.dumps(data_point) + '\n')
            
            # Save test identifiers
            self.save_test_identifiers(test_ids, "piqa_test_identifiers.json")
            
            self.logger.info(f"Created test split: {len(test_data)} samples from {len(train_data)} total")
            return ProcessResult(
                name=self.name,
                success=True,
                output_path=self.output_dir,
                test_split_created=True,
                test_identifiers_saved=True
            )
            
        except Exception as e:
            self.logger.error(f"Error processing PIQA: {e}")
            return ProcessResult(self.name, False, self.output_dir, error=str(e))


class ODinWProcessor(BaseProcessor):
    """ODinW dataset processor for bbox extraction and object-caption pairs."""
    
    def process(self) -> ProcessResult:
        """Process ODinW datasets and create bbox images with object-caption pairs."""
        try:
            datasets_processed = 0
            
            for dataset_dir in self.input_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                
                self.logger.info(f"Processing ODinW dataset: {dataset_dir.name}")
                
                # Recursively search for test folders
                test_path = self._find_test_folder(dataset_dir)
                if test_path is None:
                    self.logger.warning(f"No test folder found for {dataset_dir.name}")
                    continue
                
                self.logger.info(f"Found test folder at: {test_path.relative_to(dataset_dir)}")
                
                # Find annotations file
                annotations_file = test_path / "_annotations.coco.json"
                if not annotations_file.exists():
                    self.logger.warning(f"No COCO annotations found for {dataset_dir.name}")
                    continue
                
                # Load annotations
                with open(annotations_file, 'r') as f:
                    coco_data = json.load(f)
                
                # Create output directory for this dataset
                dataset_output = self.output_dir / "test" / dataset_dir.name
                dataset_output.mkdir(parents=True, exist_ok=True)
                
                # Create subdirectories for organized output
                bbox_images_output = dataset_output / "bbox_images"
                bbox_images_output.mkdir(exist_ok=True)
                
                # Process categories for classification
                categories = coco_data.get("categories", [])
                category_map = {cat["id"]: cat for cat in categories}
                
                # Create image lookup
                images = coco_data.get("images", [])
                image_map = {img["id"]: img for img in images}
                
                # Process annotations and extract bounding boxes
                annotations = coco_data.get("annotations", [])
                object_caption_pairs = []
                bbox_count = 0
                
                for annotation in annotations:
                    try:
                        # Get image info
                        image_id = annotation["image_id"]
                        if image_id not in image_map:
                            continue
                        
                        image_info = image_map[image_id]
                        image_file = test_path / image_info["file_name"]
                        
                        if not image_file.exists():
                            self.logger.warning(f"Image file does not exist: {image_file}")
                            continue
                        
                        # Load image
                        with Image.open(image_file) as img:
                            # Get bbox coordinates [x, y, width, height]
                            bbox = annotation["bbox"]
                            x, y, w, h = bbox
                            
                            # Ensure bbox is within image bounds
                            x = max(0, int(x))
                            y = max(0, int(y))
                            w = min(img.width - x, int(w))
                            h = min(img.height - y, int(h))
                            
                            if w <= 0 or h <= 0:
                                continue
                            
                            # Crop bbox region
                            bbox_region = img.crop((x, y, x + w, y + h))
                            
                            # Generate bbox image filename
                            bbox_filename = f"bbox_{bbox_count}_{image_info['file_name']}"
                            bbox_path = bbox_images_output / bbox_filename
                            
                            # Save cropped image
                            bbox_region.save(bbox_path)
                            
                            # Get category info
                            category_id = annotation["category_id"]
                            category_info = category_map.get(category_id, {})
                            category_name = category_info.get("name", "unknown")
                            
                            # Create object-caption pair
                            object_data = {
                                "bbox_image": bbox_filename,
                                "original_image": image_info["file_name"],
                                "category_id": category_id,
                                "category_name": category_name,
                                "bbox": bbox,
                                "bbox_id": bbox_count,
                                "annotation_id": annotation["id"],
                                "area": annotation.get("area", w * h),
                                "dataset": dataset_dir.name
                            }
                            
                            
                            object_caption_pairs.append(object_data)
                            bbox_count += 1
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to process annotation {annotation.get('id', 'unknown')}: {e}")
                        continue
                
                # Save object-caption pairs
                with open(dataset_output / "object_caption_pairs.json", 'w') as f:
                    json.dump(object_caption_pairs, f, indent=2)
                
                # Save categories for classification
                categories_data = {
                    "categories": categories,
                    "num_categories": len(categories),
                    "category_names": [cat["name"] for cat in categories],
                    "dataset": dataset_dir.name
                }
                
                with open(dataset_output / "categories.json", 'w') as f:
                    json.dump(categories_data, f, indent=2)
                
                # Save summary statistics
                summary = {
                    "dataset": dataset_dir.name,
                    "total_images": len(images),
                    "total_annotations": len(annotations),
                    "total_bbox_extracted": bbox_count,
                    "total_categories": len(categories),
                    "category_distribution": {}
                }
                
                # Calculate category distribution
                for pair in object_caption_pairs:
                    cat_name = pair["category_name"]
                    summary["category_distribution"][cat_name] = summary["category_distribution"].get(cat_name, 0) + 1
                
                with open(dataset_output / "summary.json", 'w') as f:
                    json.dump(summary, f, indent=2)
                
                datasets_processed += 1
                self.logger.info(f"Processed {dataset_dir.name}: {bbox_count} bbox images, {len(categories)} categories")
            
            if datasets_processed == 0:
                self.logger.error("No ODinW datasets were processed")
                return ProcessResult(self.name, False, self.output_dir, error="No ODinW datasets were processed")
            
            self.logger.info(f"Successfully processed {datasets_processed} ODinW datasets")
            return ProcessResult(self.name, True, self.output_dir, test_split_created=True, test_identifiers_saved=True)
            
        except Exception as e:
            self.logger.error(f"Error processing ODinW: {e}")
            return ProcessResult(self.name, False, self.output_dir, error=str(e))

    def _find_test_folder(self, dataset_dir: Path) -> Optional[Path]:
        """Recursively search for a test folder within the dataset directory."""
        def search_recursive(current_dir: Path, max_depth: int = 5) -> Optional[Path]:
            if max_depth <= 0:
                return None
            
            # Check if current directory is a test folder
            if current_dir.name == "test" and current_dir.is_dir():
                # Verify it contains images or annotations
                if any(current_dir.glob("*.jpg")) or any(current_dir.glob("*.png")) or (current_dir / "_annotations.coco.json").exists():
                    return current_dir
            
            # Search subdirectories
            for subdir in current_dir.iterdir():
                if subdir.is_dir():
                    result = search_recursive(subdir, max_depth - 1)
                    if result is not None:
                        return result
            
            return None
        
        # Find all potential test folders first
        all_test_folders = []
        
        def collect_test_folders(current_dir: Path, max_depth: int = 5):
            if max_depth <= 0:
                return
            
            if current_dir.name == "test" and current_dir.is_dir():
                # Verify it contains images or annotations
                if any(current_dir.glob("*.jpg")) or any(current_dir.glob("*.png")) or (current_dir / "_annotations.coco.json").exists():
                    all_test_folders.append(current_dir)
            
            for subdir in current_dir.iterdir():
                if subdir.is_dir():
                    collect_test_folders(subdir, max_depth - 1)
        
        # Collect all test folders
        collect_test_folders(dataset_dir)
        
        if not all_test_folders:
            return None
        
        if len(all_test_folders) == 1:
            return all_test_folders[0]
        
        # Apply prioritization rules when multiple test folders exist
        prioritized_folders = []
        
        # Rule 1: Prefer 'large' over 'tiled'
        large_folders = [f for f in all_test_folders if 'large' in str(f)]
        tiled_folders = [f for f in all_test_folders if 'tiled' in str(f)]
        
        if large_folders and tiled_folders:
            # Both exist, prefer large
            prioritized_folders = large_folders
            self.logger.info(f"Preferring large folders: {large_folders} for {dataset_dir.name}")
        elif large_folders:
            prioritized_folders = large_folders
            self.logger.info(f"Preferring large folders: {large_folders} for {dataset_dir.name}")
        elif tiled_folders:
            prioritized_folders = tiled_folders
            self.logger.info(f"Preferring tiled folders: {tiled_folders} for {dataset_dir.name}")
        else:
            prioritized_folders = all_test_folders
            self.logger.info(f"No large or tiled folders found, preferring all test folders: {all_test_folders} for {dataset_dir.name}")
        # Rule 2: Prefer folders with 'raw' in name, then non-augmented over augmented versions
        raw_folders = [f for f in prioritized_folders if 'raw' in str(f).lower()]
        
        if raw_folders:
            self.logger.info(f"Preferring raw folders: {raw_folders} for {dataset_dir.name}")
            return raw_folders[0]  # Return first raw folder found
        
        # If no raw folders, prefer non-augmented over augmented
        augmented_keywords = ['aug', 'augmented', 'augment']
        augmented_folders = [f for f in prioritized_folders if any(keyword in str(f).lower() for keyword in augmented_keywords)]
        non_augmented_folders = [f for f in prioritized_folders if not any(keyword in str(f).lower() for keyword in augmented_keywords)]
        
        if non_augmented_folders:
            self.logger.info(f"Preferring non-augmented folders: {non_augmented_folders} for {dataset_dir.name}")
            return non_augmented_folders[0]  # Prefer non-augmented
        else:
            self.logger.info(f"No non-augmented folders found, preferring prioritized folders: {prioritized_folders} for {dataset_dir.name}")
            return prioritized_folders[0]  # Return first from remaining folders


class SQA3DProcessor(BaseProcessor):
    """SQA3D dataset processor for test split extraction."""
    
    def process(self) -> ProcessResult:
        """Process SQA3D dataset and extract test split."""
        try:
            # Locate SQA3D files
            sqa_task_dir = self.input_dir / "sqa_task"
            if not sqa_task_dir.exists():
                self.logger.error("SQA3D sqa_task directory not found")
                return ProcessResult(self.name, False, self.output_dir, error="SQA3D sqa_task directory not found")
            
            balanced_dir = sqa_task_dir / "balanced"
            if not balanced_dir.exists():
                self.logger.error("SQA3D balanced directory not found")
                return ProcessResult(self.name, False, self.output_dir, error="SQA3D balanced directory not found")
            
            # Look for test data
            # Look for specific test files
            test_questions_file = balanced_dir / "v1_balanced_questions_test_scannetv2.json"
            test_annotations_file = balanced_dir / "v1_balanced_sqa_annotations_test_scannetv2.json"
            
            if test_questions_file.exists() and test_annotations_file.exists():
                # Copy test files
                test_output = self.output_dir / "test"
                test_output.mkdir(exist_ok=True)
                
                shutil.copy2(test_questions_file, test_output / "v1_balanced_questions_test_scannetv2.json")
                shutil.copy2(test_annotations_file, test_output / "v1_balanced_sqa_annotations_test_scannetv2.json")
                
                self.logger.info(f"Copied test files to {test_output}")
            else:
                self.logger.warning("Test files (v1_balanced_questions_test_scannetv2.json or v1_balanced_sqa_annotations_test_scannetv2.json) not found in SQA3D")
        
            # Process PLY files
            self.logger.info("Processing PLY files...")
            ply_result = self.process_ply()
            
            if not ply_result.success:
                self.logger.error(f"PLY processing failed: {ply_result.error}")
                return ProcessResult(self.name, False, self.output_dir, error=f"PLY processing failed: {ply_result.error}")
            
            files_processed += ply_result.files_processed
            self.logger.info(f"Successfully processed {ply_result.files_processed} PLY files")
            
            self.logger.info("Successfully processed SQA3D dataset")
            return ProcessResult(
                self.name, 
                True, 
                self.output_dir, 
                test_split_created=False, 
                test_identifiers_saved=False,
                files_processed=files_processed
            )
            
        except Exception as e:
            self.logger.error(f"Error processing SQA3D: {e}")
            return ProcessResult(self.name, False, self.output_dir, error=str(e))

        
    
    def process_ply(self) -> ProcessResult:
        """Process PLY files from scans directory and convert them to top-down PNG images."""
        try:
            # Check if scans directory exists
            scans_dir = self.input_dir / "scans"
            if not scans_dir.exists():
                self.logger.error(f"Scans directory not found: {scans_dir}")
                return ProcessResult(self.name, False, self.output_dir, error="Scans directory not found")
            
            # Create test output directory
            test_output = self.output_dir / "test"
            test_output.mkdir(parents=True, exist_ok=True)
            
            processed_files = 0
            scans = list(scans_dir.iterdir())
            
            if not scans:
                self.logger.warning("No scan directories found")
                return ProcessResult(self.name, True, self.output_dir, files_processed=0)
            
            for scan in scans:
                if not scan.is_dir():
                    continue
                    
                self.logger.info(f"Processing scan: {scan.name}")
                
                # Create output directory for this scan
                scan_output = test_output / scan.name
                scan_output.mkdir(parents=True, exist_ok=True)
                
                # Find PLY files in this scan
                ply_files = list(scan.glob("*.ply"))
                clean_ply_files = [f for f in ply_files if f.name.endswith("_clean.ply")]
                
                if not clean_ply_files:
                    self.logger.warning(f"No _clean.ply files found in {scan.name}")
                    continue
                
                # Process each _clean.ply file
                for ply_file in clean_ply_files:
                    try:
                        self.logger.info(f"Processing PLY file: {ply_file.name}")
                        # Convert PLY to top-down PNG
                        ply_to_top_down_png(str(ply_file), str(scan_output))
                        processed_files += 1
                        self.logger.info(f"Successfully processed {ply_file.name}")
                    except Exception as e:
                        self.logger.error(f"Failed to process {ply_file.name}: {e}")
                        continue
            
            self.logger.info(f"Successfully processed {processed_files} PLY files")
            return ProcessResult(
                name=self.name,
                success=True,
                output_path=self.output_dir,
                files_processed=processed_files
            )
            
        except Exception as e:
            self.logger.error(f"Error processing PLY files: {e}")
            return ProcessResult(self.name, False, self.output_dir, error=str(e))


class OvercookedAIProcessor(BaseProcessor):
    """Overcooked AI dataset processor for rendering states as visual observations."""

    def _state_from_string(self, state_str):
        """
        Parses a state string and returns a dictionary suitable for OvercookedState.from_dict.
        """
        state_str = state_str.strip()
        state_grid_rows = [row.strip() for row in state_str.split('\n')]
        
        players_data = []
        objects_data = []

        for y, row_chars in enumerate(state_grid_rows):
            for x, char in enumerate(row_chars):
                if char.isdigit():
                    # Assuming player 1 is '1', player 2 is '2', etc.
                    # Default orientation for now, as it's not in the compact string
                    players_data.append({
                        "position": [x, y],
                        "orientation": [0, -1], # Default to North
                        "held_object": None # Assuming no held objects in this compact string
                    })
                elif char == 'O':
                    objects_data.append({"name": "onion", "position": [x, y]})
                elif char == 'T':
                    objects_data.append({"name": "tomato", "position": [x, y]})
                elif char == 'D':
                    objects_data.append({"name": "dish", "position": [x, y]})
                # Add other object types if they appear in the state string

        # Construct the dictionary in the format expected by OvercookedState.from_dict
        state_dict = {
            "players": players_data,
            "objects": objects_data,
            "bonus_orders": [], # Not present in compact string, default empty
            "all_orders": [],   # Not present in compact string, default empty
            "timestep": 0       # Not present in compact string, default 0
        }
        return state_dict

    def _state_to_base64(self, state_obj, layout):
        """
        Converts a game state object (dict or string) to a base64 encoded image.
        """
        layout_grid = eval(layout)
        
        # Create a cleaned_layout_grid by removing player numbers
        cleaned_layout_grid = []
        for row in layout_grid:
            cleaned_row = []
            for char in row:
                if char.isdigit():
                    cleaned_row.append(' ') # Replace player numbers with empty space
                else:
                    cleaned_row.append(char)
            cleaned_layout_grid.append(cleaned_row)

        if isinstance(state_obj, str):
            try:
                # Pickled states are often stored as json strings
                state_dict = json.loads(state_obj)
            except json.JSONDecodeError:
                # CSV states might be stored as custom strings
                state_dict = self._state_from_string(state_obj)
        else:
            # Already a dict
            state_dict = state_obj
        
        state = OvercookedState.from_dict(state_dict)
        
        visualizer = StateVisualizer(grid=cleaned_layout_grid) # Use cleaned_layout_grid for visualizer
        img_surface = visualizer.render_state(state, grid=cleaned_layout_grid)
        
        img_byte_arr = io.BytesIO()
        pygame.image.save(img_surface, img_byte_arr, "screenshot.png")
        img_byte_arr.seek(0)
        img_bytes = img_byte_arr.read()
        return base64.b64encode(img_bytes).decode('utf-8')

    def _load_data_with_pandas_fallback(self, input_file):
        """
        Loads data from a pickle file, falling back to a temporary environment
        with an older pandas version if there's a version mismatch.
        """
        try:
            # Try loading with the current pandas version
            with open(input_file, 'rb') as infile:
                data = pickle.load(infile)
            if isinstance(data, pd.DataFrame):
                return data.to_dict('records')
            return data
        except (ValueError, ImportError) as e:
            self.logger.warning(f"Failed to load pickle with current pandas version: {e}")
            self.logger.info("Attempting to load with older pandas version in a temporary environment.")

            temp_dir = Path("./temp_pandas_env")
            try:
                # Create a temporary directory for the venv and intermediate file
                temp_dir.mkdir(exist_ok=True)
                venv_dir = temp_dir / ".venv"
                intermediate_csv = temp_dir / "intermediate.csv"

                # Create venv
                subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

                # Install older pandas
                pip_executable = str(venv_dir / "bin" / "pip")
                subprocess.run([pip_executable, "install", "pandas==1.5.3", "numpy<1.24"], check=True)

                # Run helper script
                helper_script = Path(__file__).parent.parent / "utils" / "unpickle_helper.py"
                python_executable = str(venv_dir / "bin" / "python")
                subprocess.run([python_executable, str(helper_script), str(input_file), str(intermediate_csv)], check=True)

                # Load data from intermediate CSV
                with open(intermediate_csv, 'r') as infile:
                    reader = csv.DictReader(infile)
                    return list(reader)
            finally:
                # Clean up the temporary environment
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

    def process(self) -> ProcessResult:
        """Process Overcooked AI dataset and convert states to images."""
        try:
            # Find the input file
            input_file = next(self.input_dir.glob("*.pickle"), None)
            is_pickle = True
            if not input_file:
                input_file = next(self.input_dir.glob("*.csv"), None)
                is_pickle = False

            if not input_file:
                return ProcessResult(self.name, False, self.output_dir, error="No pickle or CSV file found in input directory")

            Recipe.configure({})

            # Load data
            if is_pickle:
                # The Overcooked AI dataset was pickled with an old version of pandas (
                # 1.x),
                # which is incompatible with the current version (2.x). To handle this,
                # we use a fallback mechanism that creates a temporary virtual environment
                # with the old pandas version to unpickle the data and convert it to a
                # version-agnostic format (CSV).
                data = self._load_data_with_pandas_fallback(input_file)
            else: # It's a CSV file
                with open(input_file, 'r') as infile:
                    reader = csv.DictReader(infile)
                    data = list(reader)

            # Process data
            processed_data = []
            for i, row in enumerate(data):
                new_row = row.copy()
                new_row['state'] = self._state_to_base64(new_row['state'], new_row['layout'])
                processed_data.append(new_row)

            # Save data
            output_prefix = self.output_dir / "test" / input_file.stem
            output_prefix.parent.mkdir(parents=True, exist_ok=True)
            
            csv_output_path = output_prefix.with_suffix('.csv')
            pickle_output_path = output_prefix.with_suffix('.pickle')

            if processed_data:
                fieldnames = processed_data[0].keys()
                with open(csv_output_path, 'w', newline='') as outfile:
                    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(processed_data)
                self.logger.info(f"Data saved to {csv_output_path}")

            with open(pickle_output_path, 'wb') as outfile:
                pickle.dump(processed_data, outfile)
            self.logger.info(f"Data saved to {pickle_output_path}")

            return ProcessResult(
                name=self.name,
                success=True,
                output_path=self.output_dir,
                files_processed=len(processed_data)
            )

        except Exception as e:
            self.logger.error(f"Error processing Overcooked AI: {e}")
            return ProcessResult(self.name, False, self.output_dir, error=str(e))


class OpenXProcessor(BaseProcessor):
    """OpenX dataset processor with translation and test split creation."""
    
    def process(self) -> ProcessResult:
        """Process OpenX datasets with translation and test split creation."""
        try:
            # Import torchrlds function with robust path handling
            import sys
            from pathlib import Path
            
            # Add the parent src directory to sys.path if needed
            src_dir = Path(__file__).parent.parent
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            
            try:
                from control_translation.centralized_translation import torchrlds
            except ImportError:
                # Fallback: try direct import from file
                import importlib.util
                translation_file = src_dir / "control_translation" / "centralized_translation.py"
                spec = importlib.util.spec_from_file_location("centralized_translation", translation_file)
                translation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(translation_module)
                torchrlds = translation_module.torchrlds
            
            datasets_processed = 0
            test_splits_created = 0
            identifiers_saved = 0
            
            # Check if input directory exists and log its structure
            if not self.input_dir.exists():
                self.logger.error(f"OpenX input directory does not exist: {self.input_dir}")
                return ProcessResult(self.name, False, self.output_dir, error=f"OpenX input directory does not exist: {self.input_dir}")
            
            self.logger.info(f"Processing OpenX morphology from: {self.input_dir}")
            
            datasets_processed = 0
            test_splits_created = 0
            identifiers_saved = 0
            
            # Since self.input_dir is already specific to one morphology (e.g., openx_single_arm),
            # we should look for dataset directories directly, not more openx_ directories
            for dataset_dir in self.input_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                
                self.logger.info(f"Processing OpenX dataset: {dataset_dir.name}")
                
                # Check if this dataset already has test/val split or only train split
                has_explicit_test_val = dataset_dir.name.endswith('_test') or dataset_dir.name.endswith('_val')
                is_train_only = dataset_dir.name.endswith('_train')
                
                if has_explicit_test_val:
                    self.logger.info(f"Dataset {dataset_dir.name} already has explicit test/val split - copying as-is")
                    # Just copy the existing test/val data - no new test split created
                    dataset_output = self.output_dir
                    dataset_output.mkdir(parents=True, exist_ok=True)
                    
                    # Copy all shard files to output
                    shard_files = list(dataset_dir.glob("shard_*"))
                    for shard_file in shard_files:
                        try:
                            # Translate using torchrlds function
                            translated = torchrlds(str(shard_file), "openx", limit_schema=False)
                            
                            if translated is not None:
                                output_file = dataset_output / "test" / f"translated_{shard_file.name}"
                                tf.data.Dataset.save(translated, str(output_file))
                                
                        except Exception as e:
                            self.logger.warning(f"Failed to translate shard {shard_file}: {e}")
                    
                    datasets_processed += 1
                    # No test split created for datasets that already have test/val
                    continue
                
                elif is_train_only:
                    self.logger.info(f"Dataset {dataset_dir.name} has only train split - creating 10% test split")
                else:
                    self.logger.warning(f"Dataset {dataset_dir.name} has unexpected naming pattern - skipping")
                    continue
                
                # Find dataset shards (only for train-only datasets)
                shard_files = list(dataset_dir.glob("shard_*"))
                if not shard_files:
                    self.logger.warning(f"No shard files found in {dataset_dir.name}")
                    continue
                
                # Create 10% test split for datasets without explicit test/val
                test_size = max(1, int(0.1 * len(shard_files)))
                test_shards = random.sample(shard_files, test_size)
                test_shard_numbers = [int(f.name.split('_')[-1]) for f in test_shards]
                
                # Create output directories
                dataset_output = self.output_dir
                test_output = dataset_output / "test"
                test_output.mkdir(parents=True, exist_ok=True)
                
                # Process and translate only test split shards
                for shard_file in test_shards:
                    try:
                        # Translate using torchrlds function
                        translated = torchrlds(str(shard_file), "openx", limit_schema=False)
                        
                        if translated is not None:
                            output_file = test_output / f"translated_{shard_file.name}"
                            
                            # Save translated test data
                            tf.data.Dataset.save(translated, str(output_file))
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to translate test shard {shard_file}: {e}")
                
                # Save test split identifiers
                identifiers_file = dataset_output / "test_shard_identifiers.json"
                with open(identifiers_file, 'w') as f:
                    json.dump({
                        "test_shard_numbers": test_shard_numbers,
                        "total_shards": len(shard_files),
                        "test_ratio": 0.1,
                        "dataset": dataset_dir.name,
                        "morphology": self.name
                    }, f, indent=2)
                
                self.logger.info(f"Processed {dataset_dir.name}: {len(shard_files)} shards, {test_size} test shards")
                datasets_processed += 1
                test_splits_created += 1  # This dataset had a test split created
                identifiers_saved += 1
            
            if datasets_processed == 0:
                self.logger.error("No OpenX datasets were processed")
                return ProcessResult(self.name, False, self.output_dir, error="No OpenX datasets were processed")
            
            self.logger.info(f"Successfully processed {datasets_processed} OpenX datasets")
            return ProcessResult(
                self.name, 
                True, 
                self.output_dir,
                test_split_created=(test_splits_created > 0),
                test_identifiers_saved=(identifiers_saved > 0)
            )
            
        except Exception as e:
            self.logger.error(f"Error processing OpenX: {e}")
            return ProcessResult(self.name, False, self.output_dir, error=str(e))


class BFCLProcessor(BaseProcessor):
    """BFCL dataset processor with 20% test split creation."""
    
    def process(self) -> ProcessResult:
        """Process BFCL dataset and create test split."""
        try:
            # Load questions file
            questions_file = self.input_dir / "BFCL_v3_multi_turn_base.json"
            answers_file = self.input_dir / "possible_answer_BFCL_v3_multi_turn_base.json"
            
            if not questions_file.exists():
                self.logger.error("BFCL questions file not found")
                return ProcessResult(self.name, False, self.output_dir, error="BFCL questions file not found")
            
            # Load data
            questions_data = []
            with open(questions_file, 'r') as f:
                for line_idx, line in enumerate(f):
                    data = json.loads(line.strip())
                    data['line_id'] = line_idx  # Add line identifier
                    questions_data.append(data)
            
            answers_data = []
            if answers_file.exists():
                with open(answers_file, 'r') as f:
                    for line_idx, line in enumerate(f):
                        data = json.loads(line.strip())
                        data['line_id'] = line_idx
                        answers_data.append(data)
            
            # Create 20% test split based on IDs, not indices
            test_size = int(0.2 * len(questions_data))
            
            # Get all available IDs from questions
            all_question_ids = [q['id'] for q in questions_data]
            test_ids = random.sample(all_question_ids, test_size)
            test_ids_set = set(test_ids)
            
            # Split questions data using IDs
            test_questions = [q for q in questions_data if q['id'] in test_ids_set]
            
            # Split answers using the same test IDs
            test_answers = []
            if answers_data:
                # Create a lookup dict for answers by ID for efficient matching
                answers_by_id = {a['id']: a for a in answers_data if 'id' in a}
                test_answers = [answers_by_id[test_id] for test_id in test_ids if test_id in answers_by_id]
                
                self.logger.info(f"Matched {len(test_answers)} answers out of {len(test_ids)} test questions")
            
            # Create output directories
            test_output = self.output_dir / "test"
            test_output.mkdir(exist_ok=True)
            
            # Save test data
            with open(test_output / "questions.jsonl", 'w') as f:
                for data_point in test_questions:
                    f.write(json.dumps(data_point) + '\n')
            
            if test_answers:
                with open(test_output / "answers.jsonl", 'w') as f:
                    for data_point in test_answers:
                        f.write(json.dumps(data_point) + '\n')
            
            # Save test identifiers
            self.save_test_identifiers(test_ids, "bfcl_test_identifiers.json")
            
            self.logger.info(f"Created test split: {len(test_questions)} samples from {len(questions_data)} total")
            return ProcessResult(self.name, True, self.output_dir, test_split_created=True, test_identifiers_saved=True)
            
        except Exception as e:
            self.logger.error(f"Error processing BFCL: {e}")
            return ProcessResult(self.name, False, self.output_dir, error=str(e))


# ============================================================================
# Main Processor Manager
# ============================================================================

class DatasetProcessor:
    """Main manager for processing datasets."""
    
    def __init__(self, input_dir: Path = Path("./dataset_cache"), output_dir: Path = Path("./processed_datasets")):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
        # Registry of available processors
        self.processors = {
            "piqa": lambda: PIQAProcessor("piqa", self.input_dir, self.output_dir),
            "odinw": lambda: ODinWProcessor("odinw", self.input_dir, self.output_dir),
            "sqa3d": lambda: SQA3DProcessor("sqa3d", self.input_dir, self.output_dir),
            "overcooked_ai": lambda: OvercookedAIProcessor("overcooked_ai", self.input_dir, self.output_dir),
            "bfcl_v3": lambda: BFCLProcessor("bfcl_v3", self.input_dir, self.output_dir),
        }
        
        # Add OpenX morphology processors
        morphologies = ["single_arm", "mobile_manipulation", "quadrupedal", "bimanual", 
                       "human", "wheeled_robot", "multi_embodiment"]
        for morphology in morphologies:
            self.processors[f"openx_{morphology}"] = lambda m=morphology: OpenXProcessor(
                f"openx_{m}", self.input_dir, self.output_dir
            )
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def list_datasets(self) -> List[str]:
        """List available dataset processors."""
        return list(self.processors.keys())
    
    def process(self, dataset_name: str) -> ProcessResult:
        """Process a single dataset."""
        if dataset_name not in self.processors:
            return ProcessResult(dataset_name, False, self.output_dir,
                               error=f"Unknown dataset: {dataset_name}")
        
        processor = self.processors[dataset_name]()
        return processor.run()
    
    def process_multiple(self, dataset_names: List[str]) -> List[ProcessResult]:
        """Process multiple datasets."""
        results = []
        for name in dataset_names:
            result = self.process(name)
            results.append(result)
        return results
    
    def process_all(self) -> List[ProcessResult]:
        """Process all available datasets."""
        return self.process_multiple(self.list_datasets())
    
    def report(self, results: List[ProcessResult]) -> str:
        """Generate processing report."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        test_splits = [r for r in successful if r.test_split_created]
        
        report = f"""
Processing Report
================
Total: {len(results)} datasets
Successful: {len(successful)}
Failed: {len(failed)}
Test splits created: {len(test_splits)}

"""
        
        if successful:
            report += "✓ Successful:\n"
            for r in successful:
                status = " (test split created)" if r.test_split_created else ""
                report += f"  {r.name}{status}\n"
        
        if failed:
            report += "\n✗ Failed:\n"
            for r in failed:
                report += f"  {r.name}: {r.error}\n"
        
        return report


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset Processor")
    parser.add_argument("--input-dir", type=Path, default=Path("./dataset_cache"),
                       help="Input directory (from downloader)")
    parser.add_argument("--output-dir", type=Path, default=Path("./processed_datasets"),
                       help="Output directory")
    parser.add_argument("--list", action="store_true", help="List datasets")
    parser.add_argument("--process", nargs="+", help="Process datasets")
    parser.add_argument("--process-all", action="store_true", help="Process all")
    
    args = parser.parse_args()
    
    processor = DatasetProcessor(args.input_dir, args.output_dir)
    
    if args.list:
        print("Available datasets:")
        for dataset in processor.list_datasets():
            print(f"  {dataset}")
    
    elif args.process:
        results = processor.process_multiple(args.process)
        print(processor.report(results))
    
    elif args.process_all:
        results = processor.process_all()
        print(processor.report(results))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


