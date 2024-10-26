import numpy as np
from typing import Callable
from .robot_utils import normalize_gripper_action, invert_gripper_action

def binarize_gripper_action(action: float) -> int:
    if action > 0.5:
        return 1
    elif action < 0:
        return -1
    else:
        return 0


def convert_action(action: np.ndarray, dataset_name: str):
    """
    Convert the predicted action from OR to the OpenVLA standard.

    see definitions/openx.py for more details.
    """
    def jaco_play_conversion(action: np.ndarray) -> np.ndarray:        
        standard_action = np.zeros(4)  # Initialize with 4 elements
        
        gripper_value = action[-1]
        if gripper_value < 0.33:
            standard_action[0] = 2.0  # Gripper closed
        elif gripper_value > 0.67:
            standard_action[0] = 0.0  # Gripper open
        else:
            standard_action[0] = 1.0  # Gripper doesn't move
        
        standard_action[1:4] = action[:3]  # Copy the first 3 elements
        return standard_action

    def berkeley_cable_routing_conversion(action: np.ndarray) -> np.ndarray:
        return np.array([action[3], action[4], action[5], action[3], action[4], action[5]])

    def nyu_door_opening_conversion(action: np.ndarray) -> np.ndarray:
        action = normalize_gripper_action(action, binarize=True)
        return np.array([-1 * action[6], action[3], action[4], action[5], action[0], action[1], action[2]])
    
    def viola_conversion(action: np.ndarray) -> np.ndarray:
        action = normalize_gripper_action(action, binarize=False)  # normalize to [-1, 1]
        return np.array([binarize_gripper_action(-1 * action[6]), action[3], action[4], action[5], action[0], action[1], action[2]])

    def berkeley_autolab_ur5_conversion(action: np.ndarray) -> np.ndarray:
        action = normalize_gripper_action(action, binarize=True)
        return np.array([-1 * action[6], action[3], action[4], action[5], action[0], action[1], action[2]])

    def toto_conversion(action: np.ndarray) -> np.ndarray:
        return np.array([action[3], action[4], action[5], action[0], action[1], action[2]])

    def columbia_cairlab_pusht_real_conversion(action: np.ndarray) -> np.ndarray:
        return np.array([1 - binarize_gripper_action(action[6]), action[3], action[4], action[5], action[0], action[1], action[2]])

    def nyu_rot_conversion(action: np.ndarray) -> np.ndarray:
        action[6] = 1 - binarize_gripper_action(action[6])
        return action

    def stanford_hydra_conversion(action: np.ndarray) -> np.ndarray:
        action[6] = 1 - binarize_gripper_action(action[6])
        return action

    def ucsd_kitchen_conversion(action: np.ndarray) -> np.ndarray:
        return np.array([action[0], action[1], action[2], action[3], action[4], action[5], 1 - binarize_gripper_action(action[6])])

    # FIXME: ucsd_pick_and_place is using velocity and torque for gripper not
    def ucsd_pick_and_place_conversion(action: np.ndarray) -> np.ndarray:
        action = normalize_gripper_action(action, binarize=False)
        return np.array([action[0], action[1], action[2], action[6]])

    def usc_cloth_sim_conversion(action: np.ndarray) -> np.ndarray:
        return np.array([action[0], action[1], action[2], -1 * action[6]])

    def utokyo_pr2_conversion(action: np.ndarray) -> np.ndarray:

        return np.array([
            action[0], action[1], action[2],  # positional delta
            action[3], action[4], action[5],  # RPY angles
            1 - binarize_gripper_action(action[6])
        ])

    def stanford_mask_vit_conversion(action: np.ndarray) -> np.ndarray:
        action = normalize_gripper_action(action, binarize=False)  # normalize to [-1, 1]
        return np.array([action[0], action[1], action[2], action[5], binarize_gripper_action(-1 * action[6])])


    #  FIXME: eth uses velocity and angular velocity
    def eth_agent_affordances_conversion(action: np.ndarray) -> np.ndarray:
        return action[:6]

    def imperialcollege_sawyer_wrist_cam_conversion(action: np.ndarray) -> np.ndarray:
        return np.array([action[0], action[1], action[2], action[3], action[4], action[5], 1 - binarize_gripper_action(action[6])])
    
    def conq_hose_manipulation_conversion(action: np.ndarray) -> np.ndarray:
        action[6] = binarize_gripper_action(action[6])
        return action

    def plex_robosuite_conversion(action: np.ndarray) -> np.ndarray:
        action = normalize_gripper_action(action, binarize=True)  # normalize to [-1, 1]
        return invert_gripper_action(action)

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
        'stanford_mask_vit_converted_externally_to_rlds': stanford_mask_vit_conversion,
        'eth_agent_affordances': eth_agent_affordances_conversion,
        'imperialcollege_sawyer_wrist_cam': imperialcollege_sawyer_wrist_cam_conversion,
        'conq_hose_manipulation': conq_hose_manipulation_conversion,
        'plex_robosuite': plex_robosuite_conversion,
    }
    
    
    try:
        convert_func = conversion_functions.get(dataset_name)
    except KeyError:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return convert_func(action)


def drop_is_terminal_dim(action: np.ndarray, dataset_name: str) -> np.ndarray:
    if dataset_name == "berkeley_cable_routing":
        return drop_dimension(action, 3)
    if dataset_name == "nyu_door_opening_surprising_effectiveness":
        return drop_dimension(action, 4)
    if dataset_name == "viola":
        return drop_dimension(action, 4)
    if dataset_name == "berkeley_autolab_ur5":
        return drop_dimension(action, 4)
    if dataset_name == "toto":
        return drop_dimension(action, 3)
    if dataset_name == "columbia_cairlab_pusht_real":
        return drop_dimension(action, 4)
    if dataset_name == "ucsd_kitchen_dataset_converted_externally_to_rlds":
        return drop_dimension(action, 7)
    if dataset_name == "utokyo_pr2_opening_fridge_converted_externally_to_rlds" \
        or dataset_name == "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds":
        return drop_dimension(action, 7)
    if dataset_name == "imperialcollege_sawyer_wrist_cam":
        return drop_dimension(action, 7)
    return action


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