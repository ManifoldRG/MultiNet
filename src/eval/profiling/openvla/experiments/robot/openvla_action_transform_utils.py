import numpy as np
from typing import Callable
from .robot_utils import normalize_gripper_action, invert_gripper_action
from definitions.procgen import ProcGenDefinitions
from definitions.openx import OpenXDefinitions
import logging

logger = logging.getLogger(__name__)


def binarize_gripper_action_for_0_1_range(action: float) -> int:
    if action < 0:
        raise ValueError("Gripper action is less than 0. use normalize_gripper_action(action, binarize=True) from robot_utils.py instead.")

    if action > 0.5:
        return 1
    else:
        return 0

def discretize_range_0_to_pos_1_gripper_action_to_3_values(gripper_value: float) -> float:
    """
    Convert gripper action value to discrete format with closed (-1), open (1), and no movement (0) values
    
    Uses the thresholds based on binarize_gripper_actions() 
    in src/eval/profiling/openvla/prismatic/vla/datasets/rlds/utils/data_utils.py

    0.05 and 0.95 for range [0, 1]

    Args:
        gripper_value (float): The input gripper value.

    Returns:
        float: Standardized gripper action value.
            -1.0 for closed gripper (gripper_value < 0.05)
            1.0 for open gripper (gripper_value > 0.95)
            0.0 for no movement (0.05 <= gripper_value <= 0.95)
    """
    if gripper_value < 0.05:
        return -1.0  # Gripper closed
    elif gripper_value > 0.95:
        return 1.0  # Gripper open
    else:
        return 0.0  # Gripper doesn't move

def convert_action(action: np.ndarray, dataset_name: str):
    """
    Convert the predicted action from OpenVLA to the OpenX standard.

    see definitions/openx.py for more details.
    """
    def jaco_play_conversion(action: np.ndarray) -> np.ndarray:
        standard_action = np.zeros(4)  # Initialize with 4 elements
        
        gripper_value = action[-1]
        standard_action[0] = -1 * discretize_range_0_to_pos_1_gripper_action_to_3_values(gripper_value)
        
        standard_action[1:4] = action[:3]  # Copy the first 3 elements
        return standard_action

    def berkeley_cable_routing_conversion(action: np.ndarray) -> np.ndarray:
        return np.array([action[3], action[4], action[5], action[0], action[1], action[2]])

    def nyu_door_opening_conversion(action: np.ndarray) -> np.ndarray:
        action = normalize_gripper_action(action, binarize=False)
        return np.array([-1 * action[6], action[3], action[4], action[5], action[0], action[1], action[2]])
    
    def viola_conversion(action: np.ndarray) -> np.ndarray:
        action = normalize_gripper_action(action, binarize=True)  # normalize to [-1, 1]
        return np.array([-1 * action[6], action[3], action[4], action[5], action[0], action[1], action[2]])

    def berkeley_autolab_ur5_conversion(action: np.ndarray) -> np.ndarray:
        action = normalize_gripper_action(action, binarize=False)
        action[6] = -1 * discretize_range_0_to_pos_1_gripper_action_to_3_values(action[6])

        return np.array([action[6], action[3], action[4], action[5], action[0], action[1], action[2]])

    def toto_conversion(action: np.ndarray) -> np.ndarray:
        return np.array([action[3], action[4], action[5], action[0], action[1], action[2]])

    def columbia_cairlab_pusht_real_conversion(action: np.ndarray) -> np.ndarray:
        return np.array([1 - binarize_gripper_action_for_0_1_range(action[6]), action[3], action[4], action[5], action[0], action[1], action[2]])

    def nyu_rot_conversion(action: np.ndarray) -> np.ndarray:
        action[6] = binarize_gripper_action_for_0_1_range(action[6])
        return action

    def stanford_hydra_conversion(action: np.ndarray) -> np.ndarray:
        action[6] = 1 - binarize_gripper_action_for_0_1_range(action[6])
        return action

    def ucsd_kitchen_conversion(action: np.ndarray) -> np.ndarray:
        return np.array([action[0], action[1], action[2], action[3], action[4], action[5], binarize_gripper_action_for_0_1_range(action[6])])

    # NOTE: ucsd_pick_and_place is using velocity and torque for gripper not position
    def ucsd_pick_and_place_conversion(action: np.ndarray) -> np.ndarray:
        # the last gripper dimension action gets scaled based on the dataset statistics during inference in predict_action() in modeling_prismatic.py
        return np.array([action[0], action[1], action[2], action[6]])

    def usc_cloth_sim_conversion(action: np.ndarray) -> np.ndarray:
        return np.array([action[0], action[1], action[2], 1 - action[6]])

    def utokyo_pr2_conversion(action: np.ndarray) -> np.ndarray:

        return np.array([
            action[0], action[1], action[2],  # positional delta
            action[3], action[4], action[5],  # RPY angles
            binarize_gripper_action_for_0_1_range(action[6])
        ])
    
    def utokyo_xarm_pick_and_place_conversion(action: np.ndarray) -> np.ndarray:
        action[6] = binarize_gripper_action_for_0_1_range(action[6])
        return np.array([action[0], action[1], action[2], action[5], action[4], action[3], action[6]])

    def stanford_mask_vit_conversion(action: np.ndarray) -> np.ndarray:
        action = normalize_gripper_action(action, binarize=True)  # normalize to [-1, 1]
        return np.array([action[0], action[1], action[2], action[5], -1 * action[6]])

    #  NOTE: eth uses velocity and angular velocity
    def eth_agent_affordances_conversion(action: np.ndarray) -> np.ndarray:
        return action[:6]

    def imperialcollege_sawyer_wrist_cam_conversion(action: np.ndarray) -> np.ndarray:
        return np.array([action[0], action[1], action[2], action[3], action[4], action[5], binarize_gripper_action_for_0_1_range(action[6])])
    
    def conq_hose_manipulation_conversion(action: np.ndarray) -> np.ndarray:
        action[6] = binarize_gripper_action_for_0_1_range(action[6])
        return action

    def plex_robosuite_conversion(action: np.ndarray) -> np.ndarray:
        action = normalize_gripper_action(action, binarize=True)  # normalize to [-1, 1]
        return invert_gripper_action(action)

    def utokyo_xarm_bimanual_conversion(action: np.ndarray) -> np.ndarray:
        return action

    def openvla_to_bigfish_conversion(action: np.ndarray) -> np.ndarray:
        """
        Convert OpenVLA standard to Bigfish action space:
            0: Move left-down (-1, -1)
            1: Move down (0, -1)
            2: Move right-down (1, -1)
            3: Move left (-1, 0)
            4: Stand still (0, 0)
            5: Move right (1, 0)
            6: Move left-up (-1, 1)
            7: Move up (0, 1)
            8: Move right-up (1, 1)
            actions >= 9 represent special actions while the agent stands still
        
        In OpenVLA format, typically:
        - x dimension (index 0) corresponds to horizontal movement (LEFT/RIGHT)
        - z dimension (index 2) corresponds to vertical movement (UP/DOWN)

        """
        # Extract x and y dimensions (horizontal and vertical movement)
        # For Bigfish, we'll use x for LEFT/RIGHT and z for UP/DOWN
        x_movement = action[0]  # Horizontal movement (LEFT/RIGHT)
        z_movement = action[2]  # Vertical movement (UP/DOWN)
        
        # threshold for movement detection
        threshold = 0.3
        
        # Determine horizontal direction
        if x_movement < -threshold:
            horizontal = -1  # LEFT
        elif x_movement > threshold:
            horizontal = 1   # RIGHT
        else:
            horizontal = 0
            
        # Determine vertical direction
        if z_movement < -threshold:
            vertical = -1  # DOWN
        elif z_movement > threshold:
            vertical = 1   # UP
        else:
            vertical = 0

        # Calculate action index based on direction vectors
        # Map (horizontal, vertical) to action indices:
        # (-1, -1): 0, (0, -1): 1, (1, -1): 2
        # (-1, 0): 3, (0, 0): 4, (1, 0): 5
        # (-1, 1): 6, (0, 1): 7, (1, 1): 8
        action_index = (horizontal + 1) + (vertical + 1) * 3
        
        return [action_index]


    conversion_functions: dict[str, Callable[[np.ndarray, bool], np.ndarray]] = {
        'jaco_play': jaco_play_conversion,
        'berkeley_cable_routing': berkeley_cable_routing_conversion,
        'nyu_door_opening_surprising_effectiveness': nyu_door_opening_conversion,
        'viola': viola_conversion,
        'berkeley_autolab_ur5': berkeley_autolab_ur5_conversion,
        'toto': toto_conversion,
        'columbia_cairlab_pusht_real': columbia_cairlab_pusht_real_conversion,
        "nyu_rot_dataset_converted_externally_to_rlds": nyu_rot_conversion,
        'stanford_hydra_dataset_converted_externally_to_rlds': stanford_hydra_conversion,
        'ucsd_kitchen_dataset_converted_externally_to_rlds': ucsd_kitchen_conversion,
        'ucsd_pick_and_place_dataset_converted_externally_to_rlds': ucsd_pick_and_place_conversion,
        "usc_cloth_sim_converted_externally_to_rlds": usc_cloth_sim_conversion,
        "utokyo_pr2_opening_fridge_converted_externally_to_rlds": utokyo_pr2_conversion,
        'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds': utokyo_pr2_conversion,
        'utokyo_xarm_pick_and_place_converted_externally_to_rlds': utokyo_xarm_pick_and_place_conversion,
        'stanford_mask_vit_converted_externally_to_rlds': stanford_mask_vit_conversion,
        'eth_agent_affordances': eth_agent_affordances_conversion,
        'imperialcollege_sawyer_wrist_cam': imperialcollege_sawyer_wrist_cam_conversion,
        'conq_hose_manipulation': conq_hose_manipulation_conversion,
        'plex_robosuite': plex_robosuite_conversion,
        'utokyo_xarm_bimanual_converted_externally_to_rlds': utokyo_xarm_bimanual_conversion,
        'bigfish': openvla_to_bigfish_conversion
    }
    

    try:
        convert_func = conversion_functions.get(dataset_name)
    except KeyError:
        raise ValueError(f"Post inference manual conversion undefined for dataset: {dataset_name}")
    
    try:
        return convert_func(action)
    except ValueError as e:
        logger.error(f"Error during conversion for dataset {dataset_name}: {e}")
        raise


def drop_is_terminal_dim(action: np.ndarray, dataset_name: str) -> np.ndarray:
    if dataset_name == "berkeley_cable_routing":
        return drop_dimension(action, 3)
    elif dataset_name == "nyu_door_opening_surprising_effectiveness":
        return drop_dimension(action, 4)
    elif dataset_name == "viola":
        return drop_dimension(action, 4)
    elif dataset_name == "berkeley_autolab_ur5":
        return drop_dimension(action, 4)
    elif dataset_name == "toto":
        return drop_dimension(action, 3)
    elif dataset_name == "columbia_cairlab_pusht_real":
        return drop_dimension(action, 4)
    elif dataset_name == "ucsd_kitchen_dataset_converted_externally_to_rlds":
        return drop_dimension(action, 7)
    elif dataset_name == "utokyo_pr2_opening_fridge_converted_externally_to_rlds" \
        or dataset_name == "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds":
        return drop_dimension(action, 7)
    elif dataset_name == "imperialcollege_sawyer_wrist_cam":
        return drop_dimension(action, 7)
    elif dataset_name in ProcGenDefinitions.DESCRIPTIONS.keys():  # no is_terminal dimension in procgen
        return action
    elif dataset_name in OpenXDefinitions.DESCRIPTIONS.keys():
        return action
    else:
        raise ValueError(f"Unknown dataset {dataset_name} for drop_is_terminal_dim")
    

def drop_dimension(action: np.ndarray, index: int) -> np.ndarray:
    """
    Drop a specific dimension from a NumPy array.

    Args:
        action (np.ndarray): The input NumPy array.
        index (int): The index of the dimension to drop.

    Returns:
        np.ndarray: A new NumPy array with the specified dimension removed.

    Raises:
        IndexError: If the index is out of bounds for the input array.
    """
    if index < 0 or index >= action.shape[0]:
        raise IndexError(f"Index {index} is out of bounds for array of shape {action.shape}")
    
    return np.delete(action, index)


def clip_out_of_range_action_to_default(action: np.ndarray, dataset_name: str) -> np.ndarray:
    """
    Clip an action array to a default value if it is outside the specified range.

    Args:
        action (np.ndarray): The action array to clip.
        dataset_name (str): The name of the dataset to use for default values.

    Returns:
        np.ndarray: The clipped action array.
    """
    if dataset_name == "bigfish":
        if action[0] >= 9:
            return [4]  # default bigfish special action index to stand still based on procgen codebase
        return action
    else:
        return action
