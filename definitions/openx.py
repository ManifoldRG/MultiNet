DESCRIPTIONS = {
    "berkeley_autolab_ur5": {
        "take the tiger out of the red bowl and put it in the grey bowl": [
            "The stuffed animal (tiger) always starts in the red bowl.",
            "The positions of the two bowls are randomized on the table, while the gripper is initialized to a fixed pose.",
            "Technically, the pick-and-place task only requires translation actions of the gripper."
        ],
        "sweep the green cloth to the left side of the table": [
            "The cloth is randomly initialized at a place on the right side of the table, and the gripper needs to push it to the left side horizontally.",
            "The gripper's starting pose is randomly initialized by adding noises from a fixed position.",
            "Technically, the sweeping task only requires translation actions of the gripper."
        ],
        "pick up the blue cup and put it into the brown cup": [
            "The positions of the two cups are randomized on the table, and the gripper's starting pose is random.",
            "Technically, the stacking task only requires translation actions of the gripper."
        ],
        "put the ranch bottle into the pot": [
            "The position of the pot is fixed, while the position of the ranch bottle is randomized.",
            "The gripper's starting pose is fixed. This task involves both translation and rotation actions."
        ]
    }
}

ACTION_SPACES = {
    "berkeley_autolab_ur5": {
        "take the tiger out of the red bowl and put it in the grey bowl": {
            0: ("The delta change in X axis with respect to the robot base frame", -0.02, 0.02),
            1: ("The delta change in Y axis with respect to the robot base frame", -0.02, 0.02),
            2: ("The delta change in Z axis with respect to the robot base frame", -0.02, 0.02),
            3: ("The delta change in roll with respect to the robot base frame", -0.06666666666, 0.06666666666),
            4: ("The delta change in pitch with respect to the robot base frame", -0.06666666666, 0.06666666666),
            5: ("The delta change in yaw with respect to the robot base frame", -0.06666666666, 0.06666666666),
            6: ("The delta change in gripper closing action", {1: "Gripper closing from an open state", -1: "Gripper opening from a closed state", 0: "No change"}),
            7: ("Termination", {1: "Yes", 0: "No"})
        },
        "sweep the green cloth to the left side of the table": {
            0: ("The delta change in X axis with respect to the robot base frame", -0.02, 0.02),
            1: ("The delta change in Y axis with respect to the robot base frame", -0.02, 0.02),
            2: ("The delta change in Z axis with respect to the robot base frame", -0.02, 0.02),
            3: ("The delta change in roll with respect to the robot base frame", -0.06666666666, 0.06666666666),
            4: ("The delta change in pitch with respect to the robot base frame", -0.06666666666, 0.06666666666),
            5: ("The delta change in yaw with respect to the robot base frame", -0.06666666666, 0.06666666666),
            6: ("The delta change in gripper closing action", {1: "Gripper closing from an open state", -1: "Gripper opening from a closed state", 0: "No change"}),
            7: ("Termination", {1: "Yes", 0: "No"})
        },
        "pick up the blue cup and put it into the brown cup": {
            0: ("The delta change in X axis with respect to the robot base frame", -0.02, 0.02),
            1: ("The delta change in Y axis with respect to the robot base frame", -0.02, 0.02),
            2: ("The delta change in Z axis with respect to the robot base frame", -0.02, 0.02),
            3: ("The delta change in roll with respect to the robot base frame", -0.06666666666, 0.06666666666),
            4: ("The delta change in pitch with respect to the robot base frame", -0.06666666666, 0.06666666666),
            5: ("The delta change in yaw with respect to the robot base frame", -0.06666666666, 0.06666666666),
            6: ("The delta change in gripper closing action", {1: "Gripper closing from an open state", -1: "Gripper opening from a closed state", 0: "No change"}),
            7: ("Termination", {1: "Yes", 0: "No"})
        },
        "put the ranch bottle into the pot": {
            0: ("The delta change in X axis with respect to the robot base frame", -0.02, 0.02),
            1: ("The delta change in Y axis with respect to the robot base frame", -0.02, 0.02),
            2: ("The delta change in Z axis with respect to the robot base frame", -0.02, 0.02),
            3: ("The delta change in roll with respect to the robot base frame", -0.06666666666, 0.06666666666),
            4: ("The delta change in pitch with respect to the robot base frame", -0.06666666666, 0.06666666666),
            5: ("The delta change in yaw with respect to the robot base frame", -0.06666666666, 0.06666666666),
            6: ("The delta change in gripper closing action", {1: "Gripper closing from an open state", -1: "Gripper opening from a closed state", 0: "No change"}),
            7: ("Termination", {1: "Yes", 0: "No"})
        }
    }
}

ACTION_EXCLUSIVENESS = {
    "berkeley_autolab_ur5": {
        "take the tiger out of the red bowl and put it in the grey bowl": False,
        "sweep the green cloth to the left side of the table": False,
        "pick up the blue cup and put it into the brown cup": False,
        "put the ranch bottle into the pot": False
    }
}
