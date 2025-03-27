import numpy as np
from typing import Callable
from .robot_utils import normalize_gripper_action, invert_gripper_action
from definitions.procgen import ProcGenDefinitions
from definitions.openx import OpenXDefinitions
import logging

logger = logging.getLogger(__name__)


class SimpleMapping:
    """Decode action using simple mapping strategy with only reordering and gripper processing"""
    ACTION_ORDER_TYPES: dict[str, list[int]] = {
            'XYZ_RPY_GRIPPER': [0, 1, 2, 3, 4, 5, 6],
            'GRIPPER_XYZ': [6, 0, 1, 2],
            'RPY_XYZ': [3, 4, 5, 0, 1, 2],
            'GRIPPER_RPY_XYZ': [6, 3, 4, 5, 0, 1, 2],
            'XYZ_GRIPPER': [0, 1, 2, 6],
            'XYZ_YPR_GRIPPER': [0, 1, 2, 5, 4, 3, 6],
            'XYZ_Y_GRIPPER': [0, 1, 2, 5, 6],
            'XYZ_RPY': [0, 1, 2, 3, 4, 5],
        }

    REORDER_PATTERNS: dict[str, list[int]] = {
        'jaco_play': ACTION_ORDER_TYPES['GRIPPER_XYZ'],
        'berkeley_cable_routing': ACTION_ORDER_TYPES['RPY_XYZ'],
        'nyu_door_opening_surprising_effectiveness': ACTION_ORDER_TYPES['GRIPPER_RPY_XYZ'],
        'viola': ACTION_ORDER_TYPES['GRIPPER_RPY_XYZ'],
        'berkeley_autolab_ur5': ACTION_ORDER_TYPES['GRIPPER_RPY_XYZ'],
        'toto': ACTION_ORDER_TYPES['RPY_XYZ'],
        'columbia_cairlab_pusht_real': ACTION_ORDER_TYPES['GRIPPER_RPY_XYZ'],
        'nyu_rot_dataset_converted_externally_to_rlds': ACTION_ORDER_TYPES['XYZ_RPY_GRIPPER'],
        'stanford_hydra_dataset_converted_externally_to_rlds': ACTION_ORDER_TYPES['XYZ_RPY_GRIPPER'],
        'ucsd_kitchen_dataset_converted_externally_to_rlds': ACTION_ORDER_TYPES['XYZ_RPY_GRIPPER'],
        'ucsd_pick_and_place_dataset_converted_externally_to_rlds': ACTION_ORDER_TYPES['XYZ_GRIPPER'],  # NOTE: ucsd_pick_and_place is using velocity and torque for gripper not position
        'usc_cloth_sim_converted_externally_to_rlds': ACTION_ORDER_TYPES['XYZ_GRIPPER'],
        'utokyo_pr2_opening_fridge_converted_externally_to_rlds': ACTION_ORDER_TYPES['XYZ_RPY_GRIPPER'],
        'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds': ACTION_ORDER_TYPES['XYZ_RPY_GRIPPER'],
        'utokyo_xarm_pick_and_place_converted_externally_to_rlds': ACTION_ORDER_TYPES['XYZ_YPR_GRIPPER'],
        'stanford_mask_vit_converted_externally_to_rlds': ACTION_ORDER_TYPES['XYZ_Y_GRIPPER'],
        'eth_agent_affordances': ACTION_ORDER_TYPES['XYZ_RPY'],  # NOTE: eth uses velocity and angular velocity
        'imperialcollege_sawyer_wrist_cam': ACTION_ORDER_TYPES['XYZ_RPY_GRIPPER'],
        'conq_hose_manipulation': ACTION_ORDER_TYPES['XYZ_RPY_GRIPPER'],
        'plex_robosuite': ACTION_ORDER_TYPES['XYZ_RPY_GRIPPER'],
        'utokyo_xarm_bimanual_converted_externally_to_rlds': ACTION_ORDER_TYPES['XYZ_RPY_GRIPPER'],  # TODO: simple mapping undefined for 14D bimanual action space
    }

    GRIPPER_PATTERNS: dict[str, str] = {
        'jaco_play': 'discrete_neg',
        'nyu_door_opening': 'normalize_neg',
        'viola': 'normalize_neg',
        'berkeley_autolab_ur5': 'discrete_neg',
        'columbia_cairlab_pusht': 'binary_complement',
        'nyu_rot': 'binary',
        'stanford_hydra': 'binary_complement',
        'ucsd_kitchen': 'binary',
        'usc_cloth_sim': 'complement',
        'utokyo_pr2': 'binary',
        'utokyo_xarm_pick_and_place': 'binary',
        'stanford_mask_vit': 'normalize_binary_neg',
        'imperialcollege_sawyer': 'binary',
        'conq_hose_manipulation': 'binary',
        'plex_robosuite': 'normalize_binary_neg',
    }

    @staticmethod
    def binarize_gripper_action_to_0_1(gripper_value: float) -> int:
        """
        Binarizes a gripper action value to 0 or 1.
        
        Args:
            gripper_value (float): The input gripper action value.
        
        Returns:
            int: Binarized gripper action value (0 or 1).
        """
        if gripper_value < 0 or gripper_value > 1:
            raise ValueError("Gripper action is outside the expected range of [0, 1]")

        if gripper_value < 0:
            raise ValueError("Gripper action is less than 0. use normalize_gripper_action(action, binarize=True) from robot_utils.py instead.")

        if gripper_value > 0.5:
            return 1
        else:
            return 0

    @staticmethod
    def binarize_gripper_action_to_neg1_1(gripper_value: float) -> float:
        """Binarizes a gripper action value to -1 or 1."""
        if gripper_value < -1 or gripper_value > 1:
            raise ValueError("Gripper action is outside the expected range of [-1, 1]")

        return np.sign(gripper_value)

    @staticmethod
    def discretize_gripper_action_to_neg1_0_1(gripper_value: float) -> float:
        """
        Convert gripper action value to discrete format with closed (-1), open (1), and no movement (0) values
        
        Uses the thresholds based on binarize_gripper_actions() 
        in src/eval/profiling/openvla/prismatic/vla/datasets/rlds/utils/data_utils.py

        0.05 and 0.95 for range [0, 1]

        Args:
            gripper_value (float): The input gripper value.

        Returns:
            float: Standardized gripper action value.
                -1.0 for closed gripper (gripper_value < thresholds[0])
                1.0 for open gripper (gripper_value > thresholds[1])
                0.0 for no movement (thresholds[0] <= gripper_value <= thresholds[1])
        """
        if gripper_value < 0 or gripper_value > 1:
            raise ValueError("Gripper action is outside the expected range of [-1, 1]")

        if gripper_value < 0.05:
            return -1.0  # Gripper closed
        elif gripper_value > 0.95:
            return 1.0  # Gripper open
        else:
            return 0.0  # Gripper doesn't move

    @staticmethod
    def normalize_gripper_action_from_0_1_to_neg1_1(gripper_value: float) -> float:
        """
        Modified from robot_utils.normalize_gripper_action()

        Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].

        Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
        Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
        the dataset wrapper.

        Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

        Args:
            gripper_value: The input gripper value in range [0,1]

        Returns:
            float: The normalized gripper value in range [-1,1].
        """
        if gripper_value < 0 or gripper_value > 1:
            raise ValueError("Gripper action is outside the expected range of [0, 1]")

        # Just normalize the last action to [-1,+1].
        orig_low, orig_high = 0.0, 1.0
        gripper_value = 2 * (gripper_value - orig_low) / (orig_high - orig_low) - 1

        return gripper_value

    @staticmethod
    def decode_action(action: np.ndarray, dataset_name: str) -> np.ndarray:
        """
        Convert OpenVLA (7D) action format to dataset-specific action space using simple mapping.

        This function handles action space conversion through two mechanisms:
        1. Reordering: Rearranging action dimensions (e.g., [x,y,z] -> [z,y,x])
        2. Gripper Processing: Converting gripper values between different formats

        Args:
            action: Input action array in OpenVLA format [x,y,z,rx,ry,rz,gripper]
            dataset_name: Name of the target dataset
        """
        def transform_gripper(value: float, transform_type: str) -> float:
            """
            Apply transformation to gripper value based on specified type.
            
            Args:
                value: Input gripper value
                transform_type: Type of transformation to apply
                
            Returns:
                float: Transformed gripper value
                
            Raises:
                ValueError: If transform_type is unknown
            """
            transforms = {
                'binary': lambda x: SimpleMapping.binarize_gripper_action_to_0_1(x),
                'binary_complement': lambda x: 1 - SimpleMapping.binarize_gripper_action_to_0_1(x),
                'discrete': lambda x: SimpleMapping.discretize_gripper_action_to_neg1_0_1(x),
                'discrete_neg': lambda x: -1 * SimpleMapping.discretize_gripper_action_to_neg1_0_1(x),
                'normalize': lambda x: SimpleMapping.normalize_gripper_action_from_0_1_to_neg1_1(x),
                'normalize_neg': lambda x: -1 * SimpleMapping.normalize_gripper_action_from_0_1_to_neg1_1(x),
                'normalize_binary_neg': lambda x: -1 * SimpleMapping.binarize_gripper_action_to_neg1_1(SimpleMapping.normalize_gripper_action_from_0_1_to_neg1_1(x)),
                'complement': lambda x: 1 - x,
            }
            if transform_type not in transforms:
                raise ValueError(f"Unknown gripper transform type: {transform_type}")
            return transforms[transform_type](value)

        try:
            # Apply gripper transformation first to OpenVLA standard format
            if dataset_name in SimpleMapping.GRIPPER_PATTERNS:
                transform = SimpleMapping.GRIPPER_PATTERNS[dataset_name]
                gripper_val = action[6]  # gripper value is at index 6 in OpenVLA format
                action[6] = transform_gripper(gripper_val, transform)

            # Apply dimension reordering
            if dataset_name in SimpleMapping.REORDER_PATTERNS:
                indices = SimpleMapping.REORDER_PATTERNS[dataset_name]
                action = np.array([action[i] for i in indices])

            return action

        except Exception as e:
            logger.error(f"Error converting action for dataset {dataset_name}: {str(e)}")
            raise


class ManualRuleMapping:
    """Decode action using manual rule mapping strategy using assumed rules for converting between environments"""
    SUPPORTED_DATASET_NAMES = ['bigfish']

    @staticmethod
    def decode_action(action: np.ndarray, dataset_name: str) -> np.ndarray:
        """Decode action using manual rule mapping"""
        if dataset_name not in ManualRuleMapping.SUPPORTED_DATASET_NAMES:
            raise ValueError(f"dataset {dataset_name} undefined for manual rule mapping action decoding strategy")

        return ManualRuleMapping.openvla_to_bigfish_conversion(action)
    
    @staticmethod
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

    # For small scale experiments with Procgen using manual rule mapping action decoding strategy
    @staticmethod
    def filter_bigfish_expert_special_actions(action: np.ndarray, dataset_name: str) -> np.ndarray:
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


# === utils for expert actions ===
class ExpertActionUtils:
    
    @staticmethod
    def drop_is_terminal_dim(action: np.ndarray, dataset_name: str) -> np.ndarray:
        if dataset_name == "berkeley_cable_routing":
            return ExpertActionUtils.drop_dimension(action, 3)
        elif dataset_name == "nyu_door_opening_surprising_effectiveness":
            return ExpertActionUtils.drop_dimension(action, 4)
        elif dataset_name == "viola":
            return ExpertActionUtils.drop_dimension(action, 4)
        elif dataset_name == "berkeley_autolab_ur5":
            return ExpertActionUtils.drop_dimension(action, 4)
        elif dataset_name == "toto":
            return ExpertActionUtils.drop_dimension(action, 3)
        elif dataset_name == "columbia_cairlab_pusht_real":
            return ExpertActionUtils.drop_dimension(action, 4)
        elif dataset_name == "ucsd_kitchen_dataset_converted_externally_to_rlds":
            return ExpertActionUtils.drop_dimension(action, 7)
        elif dataset_name == "utokyo_pr2_opening_fridge_converted_externally_to_rlds" \
            or dataset_name == "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds":
            return ExpertActionUtils.drop_dimension(action, 7)
        elif dataset_name == "imperialcollege_sawyer_wrist_cam":
            return ExpertActionUtils.drop_dimension(action, 7)
        elif dataset_name in ProcGenDefinitions.DESCRIPTIONS.keys():  # no is_terminal dimension in procgen
            return action
        elif dataset_name in OpenXDefinitions.DESCRIPTIONS.keys():
            return action
        else:
            raise ValueError(f"Unknown dataset {dataset_name} for drop_is_terminal_dim")


    @staticmethod
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
