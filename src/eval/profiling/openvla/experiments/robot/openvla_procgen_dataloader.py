from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# Constants for dataset-specific settings
PROCGEN_TASK_LABELS = {
    'bigfish': "eat other fishes",
    'bossfight': "unknown task",
    'caveflyer': "unknown task",
    'chaser': "unknown task",
    'climber': "unknown task",
    'coinrun': "unknown task",
    'dodgeball': "unknown task",
    'fruitbot': "unknown task",
    'heist': "unknown task",
    'jumper': "unknown task",
    'leaper': "unknown task",
    'maze': "unknown task",
    'miner': "unknown task",
    'ninja': "unknown task",
    'plunder': "unknown task",
    'starpilot': "unknown task"
}

class ProcgenDataset(Dataset):
    def __init__(self, tfds_shards: list[str]):
        """
        Initialize the Procgen dataset.
        
        Args:
            tfds_shards: list of paths to the translated Procgen dataset shards
        """
        self.tfds_shards = tfds_shards
        self.current_elem_idx = 0
        self.current_shard_idx = 0

    def _process_shards(self):
        """
        Process the shards of the dataset, yielding episodes.
        
        Each episode consists of a sequence of timesteps with observations, actions, and rewards.
        """
        current_episode = []

        for shard_idx, shard in enumerate(self.tfds_shards):
            # Skip shards that have already been processed
            if shard_idx < self.current_shard_idx:
                continue
                
            dataset = tf.data.Dataset.load(shard)

            # Process the input data for each element in the shard
            for elem_idx, elem in enumerate(dataset):
                # Skip elements that have already been processed
                if shard_idx == self.current_shard_idx and elem_idx < self.current_elem_idx:
                    continue

                # Process observations (images)
                if 'observations' in elem:
                    # Procgen observations are RGB images
                    observation = elem['observations'].numpy()
                    # Ensure the observation is in the right format (H, W, C) and uint8
                    if observation.dtype != np.uint8:
                        observation = (observation * 255).astype(np.uint8)
                else:
                    observation = None

                # Process actions
                if 'actions' in elem:
                    action = elem['actions'].numpy()
                else:
                    action = None

                # Process rewards
                if 'rewards' in elem:
                    reward = elem['rewards'].numpy()
                else:
                    reward = 0.0

                # Determine if this is the last step in an episode
                is_last = False
                if 'dones' in elem:
                    is_last = bool(elem['dones'].numpy())
                
                # Create a dictionary for this timestep
                step_data = {
                    'observation': observation,
                    'action': action,
                    'reward': reward,
                    'is_last': is_last
                }
                
                current_episode.append(step_data)
                
                # If this is the last step in an episode, yield the episode
                if is_last:
                    if elem_idx + 1 == len(dataset):
                        self.current_elem_idx = 0
                        self.current_shard_idx = shard_idx + 1
                    else:
                        self.current_elem_idx = elem_idx + 1
                        self.current_shard_idx = shard_idx
                    yield current_episode
                    current_episode = []

        # Yield any remaining episode
        if current_episode:
            self.current_shard_idx = 0
            self.current_elem_idx = 0
            yield current_episode
            current_episode = []

    def __len__(self) -> int:
        """
        Return the number of episodes in the dataset.
        """
        return sum(1 for _ in self._process_shards())

    def __getitem__(self, idx: int) -> dict[str, any]:
        """
        Get an episode by index.
        
        Args:
            idx: Index of the episode to retrieve
            
        Returns:
            A dictionary containing the processed episode data
        """
        if idx == 0:    
            self.current_shard_idx = 0
            self.current_elem_idx = 0
        for i, episode in enumerate(self._process_shards()):
            return self._process_episode(episode)
        raise IndexError("Episode index out of range")

    def _process_episode(self, episode: list[dict[str, any]]) -> dict[str, any]:
        """
        Process an episode into a format suitable for training or evaluation.
        
        Args:
            episode: list of timesteps in the episode
            
        Returns:
            A dictionary containing the processed episode data with standardized format:
            - observation: raw observations from the environment
            - continuous_observation: processed image observations ready for the model
            - action: actions taken in the environment
            - reward: rewards received
            - is_last: whether the timestep is the last in an episode
            - text_observation: text descriptions of the task
        """
        observations = []
        continuous_observations = []  # Processed images for the model
        actions = []
        rewards = []
        is_last = []
        text_observations = []  # Added text observations

        for timestep in episode:
            # Store original observation
            observations.append(timestep['observation'])
            
            # Process image data to correct format
            image_data = timestep['observation']
            if image_data is not None:
                if image_data.shape == (3, 64, 64):
                    # Convert from (channels, height, width) to (height, width, channels)
                    processed_image = np.transpose(image_data, (1, 2, 0))
                    logger.debug(f"Image shape transposed: {processed_image.shape}")
                    
                    # Ensure uint8 format
                    if processed_image.dtype != np.uint8:
                        processed_image = (processed_image * 255).astype(np.uint8)
                        logger.debug("Image dtype not uint8, converted to uint8")
                else:
                    processed_image = image_data
                
                continuous_observations.append(processed_image)
            else:
                continuous_observations.append(None)
            
            # Store other data
            actions.append(timestep['action'])
            rewards.append(timestep['reward'])
            is_last.append(timestep['is_last'])
            
            # Add text observation based on environment
            try:
                text_obs = PROCGEN_TASK_LABELS.get(dataset_name, "unknown task")
            except Exception as e:
                raise Exception(f"Failed to get text observation for dataset {dataset_name}: {e}")
            text_observations.append(text_obs)

        return {
            'observation': observations,
            'continuous_observation': continuous_observations,
            'action': actions,
            'reward': rewards,
            'is_last': is_last,
            'text_observation': text_observations
        }

def procgen_custom_collate(batch):
    """
    Custom collate function for the Procgen dataset.
    
    Args:
        batch: A batch of episodes
        
    Returns:
        A dictionary containing the collated batch data
    """
    # Initialize dictionaries to store the collected data
    collected_data = {
        'observation': [],
        'continuous_observation': [],
        'action': [],
        'reward': [],
        'is_last': [],
        'text_observation': []
    }

    # Collect data from the batch
    for item in batch:
        for key in collected_data:
            if item[key] is not None:
                collected_data[key].append(item[key])

    result = {}
    for key, value in collected_data.items():
        if value:  # Check if the list is not empty and is not None
            result[key] = value  # Keep as list for non-numeric data
        else:
            result[key] = None  # If no valid data, set to None

    return result

def get_procgen_dataloader(tfds_shards: list[str], batch_size: int, num_workers: int = 0) -> DataLoader:
    """
    Create a DataLoader for the Procgen dataset.
    
    Args:
        tfds_shards: list of paths to the translated Procgen dataset shards
        batch_size: Batch size for the DataLoader
        num_workers: Number of worker processes for the DataLoader
        
    Returns:
        A PyTorch DataLoader for the Procgen dataset
    """
    dataset = ProcgenDataset(tfds_shards)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=procgen_custom_collate
    )