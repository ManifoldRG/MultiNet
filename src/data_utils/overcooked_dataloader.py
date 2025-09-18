from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np
import base64
import io
import pickle
import csv
import json
import ast


class OvercookedDataset(Dataset):
    def __init__(self, data_file: str, by_episode: bool = False):
        """
        Initialize Overcooked dataset.
        
        Args:
            data_file: Path to the pickle file
            by_episode: If True, return full episodes; if False, return individual timesteps
        """
        self.data_file = data_file
        self.by_episode = by_episode
        self._action_stats = None
        self.cur_file_data = None
        self.cur_file_idx = None
        
        # Load all data and organize by episodes
        self.episodes = []
        self.timestep_to_episode = []  # Maps timestep index to (episode_idx, timestep_in_episode)
        
        # Create discrete action mapping
        self._create_discrete_action_mapping()
        
        self._load_all_data()
        
    def _create_discrete_action_mapping(self):
        """Create mapping between joint actions and discrete action indices."""
        # Define individual actions for each player
        # ACTUAL ACTIONS: NORTH, SOUTH, EAST, WEST, STAY, INTERACT
        single_actions = [
            (0, -1),   # NORTH
            (0, 1),    # SOUTH  
            (1, 0),    # EAST
            (-1, 0),   # WEST
            (0, 0),    # STAY
            (1, 1)     # INTERACT (this is a word in the actual dataset, we convert it to (1,1))
        ]
        
        # Create all possible joint actions (player0_action, player1_action)
        self.joint_to_discrete = {}
        self.discrete_to_joint = {}
        
        action_idx = 0
        for p0_action in single_actions:
            for p1_action in single_actions:
                # Create joint action as tuple for hashing
                joint_action = (tuple(p0_action), tuple(p1_action))
                
                # Map joint action to discrete index
                self.joint_to_discrete[joint_action] = action_idx
                self.discrete_to_joint[action_idx] = joint_action
                
                action_idx += 1
        
        # Total number of discrete actions: 6 * 6 = 36
        self.num_discrete_actions = len(self.joint_to_discrete)
        
        # Create special mapping for 'interact' string to (1,1)
        self.action_string_to_tuple = {
            'interact': (1, 1)
        }
    
    def _convert_joint_action_to_discrete(self, joint_action):
        """Convert joint action to discrete action index."""
        if not isinstance(joint_action, list) or len(joint_action) != 2:
            # Fallback: both players STAY
            return self.joint_to_discrete[((0, 0), (0, 0))]
        
        player0_action = joint_action[0]
        player1_action = joint_action[1]
        
        # Convert string actions to tuples
        if player0_action == 'interact':
            player0_action = self.action_string_to_tuple['interact']
        if player1_action == 'interact':
            player1_action = self.action_string_to_tuple['interact']
        
        # Ensure actions are tuples
        # Convert to tuple with warning if needed
        if not isinstance(player0_action, (list, tuple)):
            print(f"Warning: player0_action is not a list/tuple. Before: {player0_action}, After: (0, 0)")
            player0_action = (0, 0)
        else:
            player0_action = tuple(player0_action)
            
        if not isinstance(player1_action, (list, tuple)):
            print(f"Warning: player1_action is not a list/tuple. Before: {player1_action}, After: (0, 0)")
            player1_action = (0, 0)
        else:
            player1_action = tuple(player1_action)
        
        # Create joint action tuple
        joint_action_tuple = (player0_action, player1_action)
        
        # Return discrete action index, or default if not found
        return self.joint_to_discrete.get(joint_action_tuple, self.joint_to_discrete[((0, 0), (0, 0))])
    
    def convert_discrete_to_joint_action(self, discrete_action):
        """Convert discrete action index back to joint action format."""
        if discrete_action in self.discrete_to_joint:
            joint_action_tuple = self.discrete_to_joint[discrete_action]
            # Convert back to list format: [player0_action, player1_action]
            return [list(joint_action_tuple[0]), list(joint_action_tuple[1])]
        else:
            # Default fallback
            return [[0, 0], [0, 0]]
    
    def get_action_mapping_info(self):
        """Get information about the discrete action mapping."""
        return {
            'num_discrete_actions': self.num_discrete_actions,
            'joint_to_discrete': self.joint_to_discrete,
            'discrete_to_joint': self.discrete_to_joint,
            'action_descriptions': self._get_action_descriptions()
        }
    
    def _get_action_descriptions(self):
        """Get human-readable descriptions of each discrete action."""
        action_names = {
            (0, -1): "NORTH",
            (0, 1): "SOUTH", 
            (1, 0): "EAST",
            (-1, 0): "WEST",
            (0, 0): "STAY",
            (1, 1): "INTERACT"
        }
        
        descriptions = {}
        for discrete_idx, joint_action in self.discrete_to_joint.items():
            p0_action, p1_action = joint_action
            p0_name = action_names.get(p0_action, f"Custom{p0_action}")
            p1_name = action_names.get(p1_action, f"Custom{p1_action}")
            descriptions[discrete_idx] = f"Player0:{p0_name}, Player1:{p1_name}"
        
        return descriptions
    
    def _load_all_data(self):
        """Load data file and organize into episodes."""
        all_timesteps = []
        
        print(f"Loading data from {self.data_file}")
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)
            all_timesteps.extend(data)

        # Group timesteps by episode_id to form episodes - one episode refers to one overcooked game on a single layout as mentioned here - https://github.com/HumanCompatibleAI/overcooked_ai/tree/master/src/human_aware_rl/static/human_data
        episodes_dict = defaultdict(list)
        for timestep in all_timesteps:
            episode_id = timestep['trial_id']
            episodes_dict[episode_id].append(timestep)
        
        for episode_id in episodes_dict.keys():
            episode_timesteps = episodes_dict[episode_id]
            # Mark the last timestep in each episode
            if episode_timesteps:
                episode_timesteps[-1]['is_last_in_episode'] = True
            # Mark all other timesteps as not last
            for timestep in episode_timesteps[:-1]: timestep['is_last_in_episode'] = False
            self.episodes.append(episode_timesteps)
        
        # Create mapping from global timestep index to episode info
        if not self.by_episode:
            timestep_idx = 0
            for episode_idx, episode in enumerate(self.episodes):
                for timestep_in_episode, _ in enumerate(episode):
                    self.timestep_to_episode.append((episode_idx, timestep_in_episode))
                    timestep_idx += 1

    def _process_timestep(self, timestep_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single timestep into the expected format."""
        # Decode base64 image
        image_observation = self._decode_base64_image(timestep_data['state'])
        
        # Parse joint action - it's stored as a string representation of a list
        joint_action_str = timestep_data['joint_action']
        text_observation = timestep_data['layout_name'] #Use the layout name as the text observation
        try:
            joint_action = ast.literal_eval(joint_action_str)
        except (ValueError, SyntaxError) as e:
            # Raise an error instead of using fallback
            raise ValueError(f"Failed to parse joint_action '{joint_action_str}': {e}")
        
        # Convert joint action to discrete action index
        # Handle different formats:
        #NORTH = (0, -1)
        #SOUTH = (0, 1)
        #EAST = (1, 0)
        #WEST = (-1, 0)
        #STAY = (0, 0)
        #INTERACT = (1,1) ---> this is a word in the actual dataset, we convert it to (1,1)
        discrete_action = self._convert_joint_action_to_discrete(joint_action)
        
        # Create the processed timestep
        processed_timestep = {
            'text_observation': text_observation,
            'image_observation': image_observation,
            'action': discrete_action,  # Now a single discrete action index
            'reward': float(timestep_data['reward']),
            'is_last': timestep_data['is_last_in_episode'],
            # Additional fields that might be useful
            'score': float(timestep_data['score']),
            'time_left': float(timestep_data['time_left']),
            'time_elapsed': float(timestep_data['time_elapsed']),
            'episode_id': timestep_data['trial_id']
        }
        
        # Update action stats if needed
        if self._action_stats is None:
            self._action_stats = {
                'size': 1,  # Single discrete action
                'num_actions': self.num_discrete_actions,  # Total number of possible actions
                'min': 0,
                'max': self.num_discrete_actions - 1,
                'action_space_type': 'discrete'
            }
        
        return processed_timestep

    def _decode_base64_image(self, base64_str: str) -> np.ndarray:
        """Decode base64 string to numpy image array."""
        try:
            # Decode base64 string
            image_bytes = base64.b64decode(base64_str)
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image, dtype=np.uint8)
            
            return image_array
            
        except Exception as e:
            # Return a placeholder image if decoding fails
            raise ValueError(f"Failed to decode base64 image: {e}")


    @property
    def action_stats(self):
        """Get action statistics"""
        if self._action_stats is None and len(self.episodes) > 0:
            # Process first timestep to initialize stats
            first_timestep = self.episodes[0][0]
            self._process_timestep(first_timestep)
        return self._action_stats

    def __len__(self) -> int:
        if self.by_episode:
            return len(self.episodes)
        else:
            return len(self.timestep_to_episode)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.by_episode:
            # Return full episode
            if idx < 0:
                idx = len(self.episodes) + idx
            
            episode_data = self.episodes[idx]
            return self._process_episode(episode_data)
        else:
            # Return single timestep
            if idx < 0:
                idx = len(self.timestep_to_episode) + idx
            
            episode_idx, timestep_in_episode = self.timestep_to_episode[idx]
            timestep_data = self.episodes[episode_idx][timestep_in_episode]
            return self._process_timestep(timestep_data)

    def _process_episode(self, episode: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a full episode into the expected format (similar to ProcGen)."""
        text_observation = []
        image_observation = []
        etc_observations = {}
        concatenated_action_float = []
        reward = []
        is_last = []
        episode_id = []

        for timestep_data in episode:
            processed_timestep = self._process_timestep(timestep_data)
            
            text_observation.append(processed_timestep['text_observation'])
            image_observation.append(processed_timestep['image_observation'])
            concatenated_action_float.append(processed_timestep['action'])
            reward.append(processed_timestep['reward'])
            is_last.append(processed_timestep['is_last'])
            episode_id.append(processed_timestep['episode_id'])
            
            # Collect other observations
            for key, value in processed_timestep.items():
                if key not in ['text_observation', 'image_observation', 'action', 'reward', 'is_last', 'episode_id']:
                    if key not in etc_observations:
                        etc_observations[key] = []
                    etc_observations[key].append(value)

        result = {
            'text_observation': text_observation,
            'image_observation': image_observation,
            'action': concatenated_action_float,
            'reward': reward,
            'is_last': is_last,
            'episode_id': episode_id
        }

        return result


def custom_collate(batch):
    """Custom collate function for batching Overcooked data."""
    result = defaultdict(list)
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    return result


def get_overcooked_dataloader(data_file: str, batch_size: int, 
                            num_workers: int = 0, by_episode: bool = False) -> tuple:
    """
    Create Overcooked dataloader.
    
    Args:
        data_file: Path to pickle file containing Overcooked data
        batch_size: Batch size for the dataloader
        num_workers: Number of worker processes for data loading
        by_episode: If True, return full episodes; if False, return individual timesteps
    
    Returns:
        tuple: (dataset, dataloader) similar to ProcGen implementation
    """
    dataset = OvercookedDataset(data_file, by_episode=by_episode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate
    )
    return dataset, dataloader