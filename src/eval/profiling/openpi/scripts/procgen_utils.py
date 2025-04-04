import numpy as np
from definitions.procgen import ProcGenDefinitions


class ActionUtils:
    @staticmethod
    def set_procgen_unused_special_action_to_stand_still(actions: np.ndarray, dataset_name: str) -> np.ndarray:
        """
        Set unused action in procgen to stand still action index 4

        Args:
            actions (np.ndarray): Array of actions for an episode.
            dataset_name (str): The name of the dataset to use for default values.

        Returns:
            np.ndarray: The modified action array with invalid actions replaced with stand still (4).
        """
        valid_action_space = ProcGenDefinitions.get_valid_action_space(dataset_name)
        
        # Create a boolean mask for invalid actions
        invalid_mask = ~np.isin(actions, valid_action_space)
        
        # Create a copy and use the mask to replace invalid actions
        modified_actions = actions.copy()
        modified_actions[invalid_mask] = 4
        
        return modified_actions
