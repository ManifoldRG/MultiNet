"""
Centralized Dataset Downloader v1.0

Simple, modular system for downloading datasets to local storage.

"""

import abc
import logging
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm
import os
import gc as _gc
import psutil as _psutil
import torch as _torch
import tensorflow as _tf
import tensorflow_datasets as _tfds
import json
import argparse


# ============================================================================
# Core Types
# ============================================================================

class DatasetType(Enum):
    """Dataset categories."""
    LANGUAGE = "language"
    VISION = "vision" 
    ROBOTICS = "robotics"
    GAMEPLAY = "gameplay"
    MULTIMODAL = "multimodal"
    TOOL_USE = "tool_use"


@dataclass
class DownloadResult:
    """Result of a dataset download."""
    name: str
    success: bool
    path: Path
    size_mb: Optional[float] = None
    error: Optional[str] = None


# ============================================================================
# Base Downloader
# ============================================================================

class BaseDownloader(abc.ABC):
    """Base class for all dataset downloaders."""
    
    def __init__(self, name: str, dataset_type: DatasetType, cache_dir: Path):
        self.name = name
        self.dataset_type = dataset_type
        self.cache_dir = cache_dir / name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{name}_downloader")
    
    @abc.abstractmethod
    def download(self) -> bool:
        """Download the dataset. Return True if successful."""
        pass
    
    @abc.abstractmethod
    def verify(self) -> bool:
        """Verify download is complete. Return True if valid."""
        pass
    
    def run(self) -> DownloadResult:
        """Main download workflow."""
        try:
            self.logger.info(f"Downloading {self.name}...")
            
            if not self.download():
                return DownloadResult(self.name, False, self.cache_dir, error="Download failed")
            
            if not self.verify():
                return DownloadResult(self.name, False, self.cache_dir, error="Verification failed")
            
            size_mb = self._calculate_size()
            self.logger.info(f"✓ {self.name} downloaded successfully ({size_mb:.1f} MB)")
            
            return DownloadResult(self.name, True, self.cache_dir, size_mb)
            
        except Exception as e:
            self.logger.error(f"✗ {self.name} failed: {e}")
            return DownloadResult(self.name, False, self.cache_dir, error=str(e))
    
    def _calculate_size(self) -> float:
        """Calculate total size of downloaded files in MB."""
        total_bytes = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
        return total_bytes / (1024 * 1024)


# ============================================================================
# Dataset Downloaders
# ============================================================================

class PIQADownloader(BaseDownloader):
    """PIQA dataset downloader."""
    
    def __init__(self, cache_dir: Path):
        super().__init__("piqa", DatasetType.LANGUAGE, cache_dir)
        self.download_url = "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip"
        self.zip_file = self.cache_dir / "physicaliqa-train-dev.zip"
    
    def download(self) -> bool:
        """Download PIQA files."""
        try:
            self.logger.info("Downloading PIQA dataset...")
            
            # Download the zip file
            response = requests.get(self.download_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.zip_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading PIQA") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            self.logger.info("Download completed. Extracting files...")
            
            # Extract the zip file
            with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.cache_dir)
            
            # Remove the zip file to save space
            self.zip_file.unlink()
            
            self.logger.info("PIQA dataset extracted successfully")
            return True
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to download PIQA dataset: {e}")
            return False
        except zipfile.BadZipFile as e:
            self.logger.error(f"Failed to extract PIQA dataset: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error downloading PIQA: {e}")
            return False
    
    def verify(self) -> bool:
        """Verify PIQA download."""
        try:
            # Check for expected files in the extracted directory
            expected_files = [
                "physicaliqa-train-dev/train.jsonl",
                "physicaliqa-train-dev/dev.jsonl",
                "physicaliqa-train-dev/dev-labels.lst",
                "physicaliqa-train-dev/train-labels.lst",
            ]
            
            for file_path in expected_files:
                full_path = self.cache_dir / file_path
                if not full_path.exists():
                    self.logger.error(f"Missing expected file: {file_path}")
                    return False
                
                # Check if file is not empty
                if full_path.stat().st_size == 0:
                    self.logger.error(f"File is empty: {file_path}")
                    return False
            
            self.logger.info("PIQA dataset verification passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying PIQA dataset: {e}")
            return False


class ODinWDownloader(BaseDownloader):
    """ODinW dataset downloader."""
    
    def __init__(self, cache_dir: Path, dataset_names: Optional[List[str]] = None):
        super().__init__("odinw", DatasetType.VISION, cache_dir)
        self.root_url = "https://huggingface.co/GLIPModel/GLIP/resolve/main/odinw_35"
        self.all_datasets = [
            "AerialMaritimeDrone", "AmericanSignLanguageLetters", "Aquarium", "BCCD", 
            "ChessPieces", "CottontailRabbits", "DroneControl", "EgoHands", "HardHatWorkers", 
            "MaskWearing", "MountainDewCommercial", "NorthAmericaMushrooms", "OxfordPets", 
            "PKLot", "Packages", "PascalVOC", "Raccoon", "ShellfishOpenImages", "ThermalCheetah", 
            "UnoCards", "VehiclesOpenImages", "WildfireSmoke", "boggleBoards", "brackishUnderwater", 
            "dice", "openPoetryVision", "pistols", "plantdoc", "pothole", "selfdrivingCar", 
            "thermalDogsAndPeople", "vector", "websiteScreenshots"
        ]
        self.dataset_names = dataset_names if dataset_names else self.all_datasets
    
    def download(self) -> bool:
        """Download ODinW dataset."""
        try:
            self.logger.info(f"Downloading ODinW datasets: {', '.join(self.dataset_names)}")
            
            for dataset in self.dataset_names:
                if dataset not in self.all_datasets:
                    self.logger.warning(f"Dataset not found: {dataset}")
                    continue
                
                self.logger.info(f"Downloading dataset: {dataset}")
                
                # Download URL
                download_url = f"{self.root_url}/{dataset}.zip"
                zip_file_path = self.cache_dir / f"{dataset}.zip"
                
                # Download the zip file
                response = requests.get(download_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(zip_file_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {dataset}") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                # Extract the zip file
                self.logger.info(f"Extracting {dataset}...")
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(self.cache_dir)
                
                # Remove the zip file to save space
                zip_file_path.unlink()
                
                self.logger.info(f"Successfully downloaded and extracted {dataset}")
            
            return True
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to download ODinW dataset: {e}")
            return False
        except zipfile.BadZipFile as e:
            self.logger.error(f"Failed to extract ODinW dataset: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error downloading ODinW: {e}")
            return False
    
    def verify(self) -> bool:
        """Verify ODinW download."""
        try:
            # Check that all requested datasets were downloaded
            for dataset in self.dataset_names:
                if dataset not in self.all_datasets:
                    continue
                
                # Check if dataset directory exists
                dataset_path = self.cache_dir / dataset
                if not dataset_path.exists():
                    self.logger.error(f"Missing dataset directory: {dataset}")
                    return False
                
                # Check if directory is not empty
                if not any(dataset_path.iterdir()):
                    self.logger.error(f"Dataset directory is empty: {dataset}")
                    return False
                
                # Check for common expected files (images, annotations)
                has_files = False
                for file_path in dataset_path.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.json', '.txt', '.xml']:
                        has_files = True
                        break
                
                if not has_files:
                    self.logger.error(f"No valid files found in dataset: {dataset}")
                    return False
            
            self.logger.info("ODinW dataset verification passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying ODinW dataset: {e}")
            return False


class SQA3DDownloader(BaseDownloader):
    """SQA3D dataset downloader. Only the questions and annotations are downloaded. To get access to the ScanNet images, please refer to the instructions here - https://github.com/SilongYong/SQA3D/blob/master/ScanQA/README.md"""
    
    def __init__(self, cache_dir: Path):
        super().__init__("sqa3d", DatasetType.MULTIMODAL, cache_dir)
        self.download_url = "https://zenodo.org/record/7792397/files/sqa_task.zip?download=1"
        self.zip_file = self.cache_dir / "sqa_task.zip"
    
    def download(self) -> bool:
        """Download SQA3D dataset."""
        try:
            self.logger.info("Downloading SQA3D dataset...")
            
            # Download the zip file
            response = requests.get(self.download_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.zip_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading SQA3D") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            self.logger.info("Download completed. Extracting files...")
            
            # Extract the zip file
            with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.cache_dir)
            
            # Remove the zip file to save space
            self.zip_file.unlink()
            
            self.logger.info("SQA3D dataset extracted successfully")
            return True
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to download SQA3D dataset: {e}")
            return False
        except zipfile.BadZipFile as e:
            self.logger.error(f"Failed to extract SQA3D dataset: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error downloading SQA3D: {e}")
            return False
    
    def verify(self) -> bool:
        """Verify SQA3D download."""
        try:
            # Check if main directory exists
            sqa_task_dir = self.cache_dir / "sqa_task"
            if not sqa_task_dir.exists():
                self.logger.error("Missing sqa_task directory")
                return False
            
            # Check if balanced directory exists
            balanced_dir = sqa_task_dir / "balanced"
            if not balanced_dir.exists():
                self.logger.error("Missing sqa_task/balanced directory")
                return False
            
            # Check if balanced directory is not empty
            if not any(balanced_dir.iterdir()):
                self.logger.error("sqa_task/balanced directory is empty")
                return False
            
            # Count files in the balanced directory
            files_in_balanced = list(balanced_dir.rglob('*'))
            data_files = [f for f in files_in_balanced if f.is_file()]
            
            if not data_files:
                self.logger.error("No data files found in sqa_task/balanced directory")
                return False
            
            # Verify at least one file is not empty
            has_valid_files = False
            for data_file in data_files:
                if data_file.stat().st_size > 0:
                    has_valid_files = True
                    break
            
            if not has_valid_files:
                self.logger.error("All files in sqa_task/balanced are empty")
                return False
            
            self.logger.info(f"SQA3D dataset verification passed - found {len(data_files)} files in sqa_task/balanced")
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying SQA3D dataset: {e}")
            return False


class OvercookedAIDownloader(BaseDownloader):
    """Overcooked AI dataset downloader."""
    
    def __init__(self, cache_dir: Path):
        super().__init__("overcooked_ai", DatasetType.GAMEPLAY, cache_dir)
        self.download_url = "https://raw.githubusercontent.com/HumanCompatibleAI/overcooked_ai/master/src/human_aware_rl/static/human_data/cleaned/2020_hh_trials_test.pickle"
        self.pickle_file = self.cache_dir / "2020_hh_trials_test.pickle"
    
    def download(self) -> bool:
        """Download Overcooked AI cleaned data."""
        try:
            self.logger.info("Downloading Overcooked AI test data...")
            
            # Download the pickle file
            response = requests.get(self.download_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.pickle_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading Overcooked AI") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            self.logger.info("Overcooked AI test data downloaded successfully")
            return True
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to download Overcooked AI dataset: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error downloading Overcooked AI: {e}")
            return False
    
    def verify(self) -> bool:
        """Verify Overcooked AI download."""
        try:
            # Check if the pickle file exists
            if not self.pickle_file.exists():
                self.logger.error("Missing Overcooked AI pickle file: 2020_hh_trials_test.pickle")
                return False
            
            # Check if file is not empty
            if self.pickle_file.stat().st_size == 0:
                self.logger.error("Overcooked AI pickle file is empty")
                return False
            
            # Try to read the pickle file header to verify it's a valid pickle file
            try:
                import pickle
                with open(self.pickle_file, 'rb') as f:
                    # Just peek at the file to verify it's a valid pickle
                    pickle.load(f)
                self.logger.info("Overcooked AI dataset verification passed")
                return True
            except (pickle.PickleError, EOFError) as e:
                self.logger.error(f"Invalid pickle file format: {e}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error verifying Overcooked AI dataset: {e}")
            return False


class OpenXDownloader(BaseDownloader):

    
    MORPHOLOGY_TO_DATASET = {
        "single_arm": "droid",
        "mobile_manipulation": "aloha_mobile", 
        "quadrupedal": "utokyo_saytap_converted_externally_to_rlds",
        "bimanual": "utokyo_xarm_bimanual_converted_externally_to_rlds",
        "human": "io_ai_tech",
        "wheeled_robot": "berkeley_gnm_sac_son",
        "multi_embodiment": "robot_vqa",
    }

    def __init__(self, morphology: str, cache_dir: Path):
        super().__init__(f"openx_{morphology}", DatasetType.ROBOTICS, cache_dir)
        if morphology not in self.MORPHOLOGY_TO_DATASET:
            raise ValueError(f"Unknown OpenX morphology: {morphology}")
        self.morphology = morphology
        self.dataset_id = self.MORPHOLOGY_TO_DATASET[morphology]

    def _shard_and_save(self, ds, dataset_name: str, start_from_shard: int, shard_size: int) -> Optional[int]:
        #function to shard and save dataset so as to not run out of memory and download the dataset in chunks (episode by episode)
        for i, shard in enumerate(ds.batch(shard_size), start=start_from_shard):
            if os.path.exists(os.path.join(str(self.cache_dir), dataset_name, 'shard_'+str(i))) == True:
                self.logger.info(f'Shard {i} of {dataset_name} already downloaded')
                continue
                
            # Check RAM usage
            ram_usage = _psutil.virtual_memory().percent
            # If RAM usage is more than 80% free up memory and restart the sharding+saving procedure from the same shard
            if ram_usage > 80:
                self.logger.warning(f"RAM usage is {ram_usage}%. Restarting from shard {i}...")
                # Clean up resources after pausing the sharding+saving procedure
                del shard
                del ds
                _gc.collect()
                return i
        
            shard = _tf.data.Dataset.from_tensor_slices(shard)
            flattened_dataset = shard.flat_map(lambda x: x['steps'])
            dataset_dict = {i: item for i, item in enumerate(flattened_dataset.as_numpy_iterator())}
            os.makedirs(os.path.join(str(self.cache_dir), dataset_name), exist_ok=True)
            _torch.save(dataset_dict, f"{os.path.join(str(self.cache_dir), dataset_name)}/shard_{i}")

            # Print current RAM usage
            self.logger.info(f"Processed shard {i}. Current RAM usage: {ram_usage}%")
        
        return None

    def download(self) -> bool:
        try:
            # Try version combinations
            versions = ['0.0.0', '0.0.1', '0.1.0', '0.1.1', '1.0.0', '1.0.1', '1.1.0', '1.1.1']
            for version in versions:
                try:
                    temp_file_path = f'gs://gresearch/robotics/{self.dataset_id}/{version}'
                    builder = _tfds.builder_from_directory(builder_dir=temp_file_path)
                    break
                except:
                    continue
            else:
                raise ValueError(f'No version found for {self.dataset_id}')
            
            self.logger.info(f'Downloading {self.dataset_id}...')
            
            # Try splits in order: test, val, train
            try:
                b = builder.as_dataset(split='test')
                split_name = 'test'
                self.logger.info('Downloading test split')
            except:
                try:
                    b = builder.as_dataset(split='val')
                    split_name = 'val'
                    self.logger.info('Downloading val split')
                except:
                    b = builder.as_dataset(split='train')
                    split_name = 'train'
                    self.logger.info('Downloading train split')
            
            dataset_folder = f"{self.dataset_id}_{split_name}"
            shard_func_catch = 0
            while True:
                if shard_func_catch is not None:
                    shard_func_catch = self._shard_and_save(b, dataset_folder, shard_func_catch, 1)
                else:
                    break
            
            return True
        except Exception as e:
            self.logger.error(f'Error while downloading {self.dataset_id}: {e}')
            return False

    def verify(self) -> bool:
        try:
            for split in ("test", "val", "train"):
                dataset_dir = self.cache_dir / f"{self.dataset_id}_{split}"
                if dataset_dir.exists() and any(dataset_dir.glob('shard_*')):
                    return True
            self.logger.error("No shards found for any split")
            return False
        except Exception as e:
            self.logger.error(f"Error verifying OpenX dataset: {e}")
            return False


class BFCLDownloader(BaseDownloader):
    """BFCL v3 dataset downloader."""
    
    def __init__(self, cache_dir: Path):
        super().__init__("bfcl_v3", DatasetType.TOOL_USE, cache_dir)
        # GitHub raw URLs for the BFCL v3 files
        self.questions_url = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v3_multi_turn_base.json"
        self.answers_url = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v3_multi_turn_base.json"
        self.questions_file = self.cache_dir / "BFCL_v3_multi_turn_base.json"
        self.answers_file = self.cache_dir / "possible_answer_BFCL_v3_multi_turn_base.json"
    
    def download(self) -> bool:
        """Download BFCL v3 dataset."""
        try:
            self.logger.info("Downloading BFCL v3 dataset...")
            
            # Download questions file
            self.logger.info("Downloading BFCL v3 questions...")
            response = requests.get(self.questions_url, stream=True)
            response.raise_for_status()
            
            with open(self.questions_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            self.logger.info("Questions file downloaded successfully")
            
            # Download answers file
            self.logger.info("Downloading BFCL v3 answers...")
            response = requests.get(self.answers_url, stream=True)
            response.raise_for_status()
            
            with open(self.answers_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            self.logger.info("Answers file downloaded successfully")
            self.logger.info("BFCL v3 dataset downloaded successfully")
            return True
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to download BFCL v3 dataset: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error downloading BFCL v3: {e}")
            return False
    
    def verify(self) -> bool:
        """Verify BFCL download."""
        try:
            # Check if both files exist
            if not self.questions_file.exists():
                self.logger.error("Missing BFCL v3 questions file")
                return False
            
            if not self.answers_file.exists():
                self.logger.error("Missing BFCL v3 answers file")
                return False
            
            # Check if files are not empty
            if self.questions_file.stat().st_size == 0:
                self.logger.error("BFCL v3 questions file is empty")
                return False
            
            if self.answers_file.stat().st_size == 0:
                self.logger.error("BFCL v3 answers file is empty")
                return False
            
            # Verify JSON Lines format by trying to parse each line
            try:
                # Verify questions file (JSON Lines format)
                with open(self.questions_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:  # Skip empty lines
                            json.loads(line)
                
                # Verify answers file (JSON Lines format)
                with open(self.answers_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:  # Skip empty lines
                            json.loads(line)
                            
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON Lines format in BFCL v3 files at line {line_num}: {e}")
                return False
            
            self.logger.info("BFCL v3 dataset verification passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying BFCL v3 dataset: {e}")
            return False


# ============================================================================
# Main Downloader Manager
# ============================================================================

class DatasetDownloader:
    """Main manager for downloading datasets."""
    
    def __init__(self, cache_dir: Path = Path("./dataset_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
        # Registry of available downloaders
        self.downloaders = {
            "piqa": lambda: PIQADownloader(self.cache_dir),
            "odinw": lambda: ODinWDownloader(self.cache_dir),
            "sqa3d": lambda: SQA3DDownloader(self.cache_dir),
            "overcooked_ai": lambda: OvercookedAIDownloader(self.cache_dir),
            "bfcl_v3": lambda: BFCLDownloader(self.cache_dir),
            # OpenX morphologies
            "openx_single_arm": lambda: OpenXDownloader("single_arm", self.cache_dir),
            "openx_mobile_manipulation": lambda: OpenXDownloader("mobile_manipulation", self.cache_dir),
            "openx_quadrupedal": lambda: OpenXDownloader("quadrupedal", self.cache_dir),
            "openx_bimanual": lambda: OpenXDownloader("bimanual", self.cache_dir),
            "openx_human": lambda: OpenXDownloader("human", self.cache_dir),
            "openx_wheeled_robot": lambda: OpenXDownloader("wheeled_robot", self.cache_dir),
            "openx_multi_embodiment": lambda: OpenXDownloader("multi_embodiment", self.cache_dir),
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def list_datasets(self) -> List[str]:
        """List available datasets."""
        return list(self.downloaders.keys())
    
    def download(self, dataset_name: str) -> DownloadResult:
        """Download a single dataset."""
        if dataset_name not in self.downloaders:
            return DownloadResult(dataset_name, False, self.cache_dir, 
                                error=f"Unknown dataset: {dataset_name}")
        
        downloader = self.downloaders[dataset_name]()
        return downloader.run()
    
    def download_multiple(self, dataset_names: List[str]) -> List[DownloadResult]:
        """Download multiple datasets."""
        results = []
        for name in dataset_names:
            result = self.download(name)
            results.append(result)
        return results
    
    def download_all(self) -> List[DownloadResult]:
        """Download all available datasets."""
        return self.download_multiple(self.list_datasets())
    
    def status(self, dataset_name: str) -> Dict[str, str]:
        """Check download status of a dataset."""
        dataset_path = self.cache_dir / dataset_name
        
        if not dataset_path.exists():
            return {"status": "not_downloaded", "path": str(dataset_path)}
        
        files = list(dataset_path.rglob('*'))
        if not files:
            return {"status": "empty", "path": str(dataset_path)}
        
        file_count = len([f for f in files if f.is_file()])
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        size_mb = total_size / (1024 * 1024)
        
        return {
            "status": "downloaded",
            "path": str(dataset_path),
            "files": str(file_count),
            "size_mb": f"{size_mb:.1f}"
        }
    
    def report(self, results: List[DownloadResult]) -> str:
        """Generate download report."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        total_size = sum(r.size_mb for r in successful if r.size_mb)
        
        report = f"""
Download Report
===============
Total: {len(results)} datasets
Successful: {len(successful)}
Failed: {len(failed)}
Total Size: {total_size:.1f} MB

"""
        
        if successful:
            report += "✓ Successful:\n"
            for r in successful:
                size_info = f" ({r.size_mb:.1f} MB)" if r.size_mb else ""
                report += f"  {r.name}{size_info}\n"
        
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
    
    parser = argparse.ArgumentParser(description="Dataset Downloader")
    parser.add_argument("--cache-dir", type=Path, default=Path("./dataset_cache"),
                       help="Cache directory")
    parser.add_argument("--list", action="store_true", help="List datasets")
    parser.add_argument("--download", nargs="+", help="Download datasets")
    parser.add_argument("--download-all", action="store_true", help="Download all")
    parser.add_argument("--status", help="Check dataset status")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.cache_dir)
    
    if args.list:
        print("Available datasets:")
        for dataset in downloader.list_datasets():
            print(f"  {dataset}")
    
    elif args.status:
        status = downloader.status(args.status)
        print(f"Dataset: {args.status}")
        for key, value in status.items():
            print(f"  {key}: {value}")
    
    elif args.download:
        results = downloader.download_multiple(args.download)
        print(downloader.report(results))
    
    elif args.download_all:
        results = downloader.download_all()
        print(downloader.report(results))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


# ============================================================================
# Usage Examples
# ============================================================================

"""
USAGE:

CLI:
    python centralized_downloader.py --list
    python centralized_downloader.py --download piqa sqa3d
    python centralized_downloader.py --status piqa
    python centralized_downloader.py --download-all

Code:
    downloader = DatasetDownloader()
    
    # Download single dataset
    result = downloader.download("piqa")
    
    # Download multiple
    results = downloader.download_multiple(["piqa", "sqa3d"])
    
    # Check status
    status = downloader.status("piqa")
    
    # Generate report
    print(downloader.report(results))

EXTENDING:

    class NewDatasetDownloader(BaseDownloader):
        def __init__(self, cache_dir: Path):
            super().__init__("new_dataset", DatasetType.VISION, cache_dir)
        
        def download(self) -> bool:
            # Your download logic here
            return True
        
        def verify(self) -> bool:
            # Your verification logic here
            return True
    
    # Add to registry
    downloader.downloaders["new_dataset"] = lambda: NewDatasetDownloader(cache_dir)
"""
