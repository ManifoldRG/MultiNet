import numpy as np
class OpenXDefinitions:
    DESCRIPTIONS = {
        "fractal20220817_data": {},
        "kuka": {
            "pick anything": [
                "Choose a grasp point, and then execute the desired grasp strategy.",
                "Update the grasp strategy continuously based on the most recent observations."
            ]
        },
        "bridge": {},
        "taco_play": {},
        "jaco_play": {},
        "berkeley_cable_routing": {
            "route cable": [
                "Route the cable through a number of tight-fitting clips mounted on the table."
            ] 
        },
        "roboturk": {
            "create tower": [
                "By stacking the cups and bowls, create the tallest tower."
            ],
            "layout laundry": [
                "Layout an article of clothing on the table such that it lies flat without folds."
            ],
            "object search": [
                "Find all instances of a certain target object category.",
                "This can be plush animal, plastic water bottle, or paper napkin.",
                "Fit them into the corresponding bin."
            ]
        },
        "nyu_door_opening_surprising_effectiveness": {
            "open door": [
                "Open a cabinet door."
            ]
        },
        "viola": {},
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
        },
        "toto": {
            "pour": [
                "Pour the nuts into the pink plastic cup."
            ]
        },
        "language_table": {},
        "columbia_cairlab_pusht_real": {
            "the task requires pushing a t-shaped block (gray) to a fixed target (green) with a circular end-effector (blue). both observation and control frequencies are 10hz": [
                "Push T-shaped block into a fixed goal pose.",
                "Move to a fixed exit zone."
            ]
        },
        "nyu_rot_dataset_converted_externally_to_rlds": {},
        "stanford_hydra_dataset_converted_externally_to_rlds": {},
        "nyu_franka_play_dataset_converted_externally_to_rlds": {},
        "maniskill_dataset_converted_externally_to_rlds": {},
        "furniture_bench_dataset_converted_externally_to_rlds": {},
        "cmu_franka_exploration_dataset_converted_externally_to_rlds": {},
        "ucsd_kitchen_dataset_converted_externally_to_rlds": {},
        "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {},
        "austin_sirius_dataset_converted_externally_to_rlds": {},
        "bc_z": {},
        "usc_cloth_sim_converted_externally_to_rlds": {},
        "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {}, 
        "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {},
        "utokyo_saytap_converted_externally_to_rlds": {},
        "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {},
        "utokyo_xarm_bimanual_converted_externally_to_rlds": {},
        "berkeley_mvp_converted_externally_to_rlds": {},
        "berkeley_rpt_converted_externally_to_rlds": {},
        "kaist_nonprehensile_converted_externally_to_rlds": {},
        "stanford_mask_vit_converted_externally_to_rlds": {
            "push something": [
                "Push the bowl or move the object into an unseen dustpan."
            ]
        },
        "tokyo_u_lsmo_converted_externally_to_rlds": {},
        "dlr_sara_pour_converted_externally_to_rlds": {},
        "dlr_sara_grid_clamp_converted_externally_to_rlds": {
            "place grid clamp": [
                "Place the grid clamp in the grids on the table, similar to placing a peg in the hole."
            ]
        },
        "dlr_edan_shared_control_converted_externally_to_rlds": {},
        "asu_table_top_converted_externally_to_rlds": {},
        "stanford_robocook_converted_externally_to_rlds": {},
        "eth_agent_affordances": {},
        "imperialcollege_sawyer_wrist_cam": {
            "grasp can": [
                "Grasp a can placed horizontally on a table and lift it."
            ],
            "hang cup": [
                "Starting with a cup in the end-effector, place it on a tree mug holder."
            ],
            "insert cap in bottle": [
                "Starting with a bottle cap in the end-effector, insert it in an empty bottle on the table."
            ],
            "insert toast": [
                "Starting with a toy bread slice in the end-effector, insert it in a toy toaster on the table."
            ],
            "open bottle": [
                "Remove the cap from a bottle on a table by grasping and lifting the cap."
            ],
            "open lid": [
                "Remove the lid from a pot on the table by grasping it and lifting it."
            ],
            "pick up bottle": [
                "Pick up a bottle placed horizontally on the table."
            ],
            "pick up kettle": [
                "Pick up a toy kettle from the handle."
            ],
            "pick up mug": [
                "Pick up a mug from the table (no need to grasp it from the handle)."
            ],
            "pick up pan": [
                "Pick up a toy pan from the table, grasping it from the handle."
            ],
            "pour in mug": [
                "Starting with a cup in the end-effector, pour into a mug on the table - success is detected by dropping a marble from the cup to the mug, mimicking a liquid."
            ],
            "put cup in dishwasher": [
                "Starting with a cup in the end-effector, place it in an empty area of a toy dishwasher rack on the table."
            ],
            "stack bowls": [
                "Starting with a bowl in the end-effector, stack it on top of another bowl on the table."
            ],
            "swipe": [
                "Starting with a dust brush in the end-effector, swipe a marble into a dustpan on the table."
            ]
        },
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {},
        "utaustin_mutex": {},
        "cmu_play_fusion": {},
        "cmu_stretch": {},
        "berkeley_gnm_recon": {
            "navigate to the goal": [
                "Ignore distractors, and explore a non-stationary environment, successfully discovering and navigating to the visually-specified goal."
            ]
        },
        "berkeley_gnm_cory_hall": {
            "navigate to the goal": [
                "Autonomously navigate complex and unstructured environments such as roads, buildings, or forests."
            ]
        },
        "berkeley_gnm_sac_son": {
            "navigate to the goal": [
                "Navigate pedestrian-rich indoor and outdoor environments such as offices, school buildings."
            ]
        },
        "conq_hose_manipulation": {},
        "dobbe": {},
        "io_ai_tech": {},
        "robo_set": {},
        "plex_robosuite": {}
    }

    ACTION_SPACES = {
        "fractal20220817_data": {
            "default": {

                0: ("X axis displacement for the robot's base movement"),
                1: ("Y axis displacement for the robot's base movement"),
                2: ("Vertical rotation displacement for the robot's base movement"),
                3: ("Gripper of the robot closed or open", {1.0: "Gripper closed", 0.0: "Gripper open"}),
                4: ("Roll displacement for the robot's arm movement"),
                5: ("Pitch displacement for the robot's arm movement"),
                6: ("Yaw displacement for the robot's arm movement"),
                7: ("X axis displacement for the robot's arm movement"),
                8: ("Y axis displacement for the robot's arm movement"),
                9: ("Z axis displacement for the robot's arm movement")

            }
        },
        "kuka": {
            "default": {
            0: ("X axis displacement of the robot base in meters", -1, 1),
            1: ("Y axis displacement of the robot base in meters", -1, 1),
            2: ("The displacement in vertical rotation of the robot base in radians", 0, 0.6366),
            3: ("Gripper closed or open", {1.0: "Gripper closed", 0.0: "Gripper open"}),
            4: ("The delta change in roll of the robot gripper pose in radians", 0, 0.6366),
            5: ("The delta change in pitch of the robot gripper pose in radians", 0, 0.6366),
            6: ("The delta change in yaw of the robot gripper pose in radians", 0, 0.6366),
            7: ("X axis displacement of the robot gripper pose in meters", -1, 1),
            8: ("Y axis displacement of the robot gripper pose in meters", -1, 1),
            9: ("Z axis displacement of the robot gripper pose in meters", -1, 1)
            }
        },
        "bridge": {
            "default": {
                0: ("The delta change in the roll for the robot"),
                1: ("The delta change in the pitch for the robot"),
                2: ("The delta change in the yaw for the robot"),
                3: ("Termination", {1.0: "The robot has reached the target location", 0.0: "The robot has not reached the target location"}),
                4: ("X axis displacement for the robot"),
                5: ("Y axis displacement for the robot"),
                6: ("Z axis displacement for the robot")
            }
        },
        "taco_play": {
            "default": {
                0: None
            }
        },
        "jaco_play": {
            "default": {
                0: ("Gripper closed or open or doesn't move", {1.0: "Gripper closed", -1.0: "Gripper open", 0.0: "Gripper doesn't move"}),
                1: ("X axis displacement for the robot"),
                2: ("Y axis displacement for the robot"),
                3: ("Z axis displacement for the robot")
            }
        },
        "berkeley_cable_routing": {
            "default": {
                0: ("The delta X axis rotation delta with respect to the robot's end effector frame"),
                1: ("The delta Y axis rotation delta with respect to the robot's end effector frame"),
                2: ("The delta Z axis rotation delta with respect to the robot's end effector frame"),
                3: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
                4: ("The X axis displacement with respect to the robot's end effector frame"),
                5: ("The Y axis displacement with respect to the robot's end effector frame"),
                6: ("The Z axis displacement with respect to the robot's end effector frame")
            }
        },
        "roboturk": {
            "default": {
                0: None
            }
        },
        "nyu_door_opening_surprising_effectiveness": {
            "default": {
                0: ("Closedness of the gripper", -1.0, 1.0),
                1: ("The X axis rotation delta of the robot in Euler coordinates"),
                2: ("The Y axis rotation delta of the robot in Euler coordinates"),
                3: ("The Z axis rotation delta of the robot in Euler coordinates"),
                4: ("Termination", {1.0: "The robot has reached the target location", 0.0: "The robot has not reached the target location"}),
                5: ("The X axis displacement in meters of the robot"),
                6: ("The Y axis displacement in meters of the robot"),
                7: ("The Z axis displacement in meters of the robot")
            }
        },
        "viola": {
            "default": {
                0 : ("Gripper closed or open", {-1.0: "Gripper open", 1.0: "Gripper closed"}),
                1: ("The X axis rotation delta of the robot"),
                2: ("The Y axis rotation delta of the robot"),
                3: ("The Z axis rotation delta of the robot"),
                4: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
                5: ("The X axis displacement of the robot"),
                6: ("The Y axis displacement of the robot"),
                7: ("The Z axis displacement of the robot")
            }
        },
        "berkeley_autolab_ur5": {
            "default": {
                0: ("The delta change in gripper closing action", {1: "Gripper closing from an open state", -1: "Gripper opening from a closed state", 0: "No change"}),
                1: ("The delta change in roll with respect to the robot base frame", -0.06666666666, 0.06666666666),
                2: ("The delta change in pitch with respect to the robot base frame", -0.06666666666, 0.06666666666),
                3: ("The delta change in yaw with respect to the robot base frame", -0.06666666666, 0.06666666666),
                4: ("Termination", {1: "Yes", 0: "No"}),
                5: ("The delta change in X axis with respect to the robot base frame", -0.02, 0.02),
                6: ("The delta change in Y axis with respect to the robot base frame", -0.02, 0.02),
                7: ("The delta change in Z axis with respect to the robot base frame", -0.02, 0.02)
            }
        },
        "toto": {
            "pour": {
                0: ("X axis rotation delta of the robot's end effector"),
                1: ("Y axis rotation delta of the robot's end effector"),
                2: ("Z axis rotation delta of the robot's end effector"),
                3: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
                4: ("The X axis displacement of the robot's end effector"),
                5: ("The Y axis displacement of the robot's end effector"),
                6: ("The Z axis displacement of the robot's end effector")
            }
        },
        "language_table": {
            "default": {
                0: ("X axis co-ordinate in the Cartesian setpoint of the end effector"),
                1: ("Y axis co-ordinate in the Cartesian setpoint of the end effector")
            }
        },
        "columbia_cairlab_pusht_real": {
            "default": {
                0: ("Absolute gripper closedness state", {0.0: "open", 1.0: "closed"}),
                1: ("X axis rotation delta of the robot's end effector"),
                2: ("Y axis rotation delta of the robot's end effector"),
                3: ("Z axis rotation delta of the robot's end effector"),
                4: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
                5: ("The X axis displacement of the robot's end effector"),
                6: ("The Y axis displacement of the robot's end effector"),
                7: ("The Z axis displacement of the robot's end effector")
            }
        },
        "nyu_rot_dataset_converted_externally_to_rlds": {
            "default": {
                0: ("X axis displacement of end effector of the robot"),
                1: ("Y axis displacement of end effector of the robot"),
                2: ("Z axis displacement of end effector of the robot"),
                3: ("X axis rotation delta of end effector of the robot"),
                4: ("Y axis rotation delta of end effector of the robot"),
                5: ("Z axis rotation delta of end effector of the robot"),
                6: ("Absolute gripper closedness state", {0.0: "closed", 1.0: "open"})
            }
        },
        "stanford_hydra_dataset_converted_externally_to_rlds": {
            "default": {
                0: ("X axis displacement of end effector of the robot", -1,1),
                1: ("Y axis displacement of end effector of the robot", -1,1),
                2: ("Z axis displacement of end effector of the robot", -1,1),
                3: ("X axis rotation delta of end effector of the robot in euler angles", -2*np.pi,2*np.pi),
                4: ("Y axis rotation delta of end effector of the robot in euler angles", -2*np.pi,2*np.pi),
                5: ("Z axis rotation delta of end effector of the robot in euler angles", -2*np.pi,2*np.pi),
                6: ("Absolute gripper closedness state", {1: "closed", 0: "open"})
            }
        },
        "nyu_franka_play_dataset_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "maniskill_dataset_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "furniture_bench_dataset_converted_externally_to_rlds": {
            "default": {
                0: ("X axis displacement of end effector of the robot in meters", -1,1),
                1: ("Y axis displacement of end effector of the robot in meters", -1,1),
                2: ("Z axis displacement of end effector of the robot in meters", -1,1),
                3: ("Scalar component of the quarternion which encodes the amount of rotation around the xyz axes", -1,1),
                4: ("Vector component of the qarternion which represents the X axis component of the rotation", -1,1),
                5: ("Vector component of the qarternion which represents the Y axis component of the rotation", -1,1),
                6: ("Vector component of the qarternion which represents the Z axis component of the rotation", -1,1),
                7: ("Gripper closedness action", {1: "closed", -1: "open"})
            }
        },
        "cmu_franka_exploration_dataset_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "ucsd_kitchen_dataset_converted_externally_to_rlds": {
            "default": {
                0: ("X axis displacement of end effector of the robot"),
                1: ("Y axis displacement of end effector of the robot"),
                2: ("Z axis displacement of end effector of the robot"),
                3: ("Roll of the end effector of the robot"),
                4: ("Pitch of the end effector of the robot"),
                5: ("Yaw of the end effector of the robot"),
                6: ("Gripper open or closed position of the robot", {0.0: "closed", 1.0: "open"}),
                7: ("Termination status of the episode",{1: "Yes", 0: "No"})
            }
        },
        "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
            "default": {
                0: ("X axis velocity of robot gripper"),
                1: ("Y axis velocity of robot gripper"),
                2: ("Z axis velocity of robot gripper"),
                3: ("Gripper open/close torque")
            }
        },
        "austin_sirius_dataset_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "bc_z": {
            "default": {
                0: None
            }
        },
        "usc_cloth_sim_converted_externally_to_rlds": {
            "default": {
                0: ("Movement of the picker along the X axis"),
                1: ("Movement of the picker along the Y axis"),
                2: ("Movement of the picker along the Z axis"),
                3: ("Activation state of the picker. A value greater than or equal to 0.5 represents activation and picking of the cloth, while a value less than 0.5 represents deactivation.")
            }
        },
        "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {
            "default": {
                0: ("X axis displacement of end effector of the robot"),
                1: ("Y axis displacement of end effector of the robot"),
                2: ("Z axis displacement of end effector of the robot"),
                3: ("Roll of the end effector of the robot"),
                4: ("Pitch of the end effector of the robot"),
                5: ("Yaw of the end effector of the robot"),
                6: ("Gripper open or closed", {0.0: "closed", 1.0: "open"}),
                7: ("Termination status of the episode",{1: "Yes", 0: "No"})
            }
        },
        "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {
            "default": {
                0: ("X axis displacement of end effector of the robot"),
                1: ("Y axis displacement of end effector of the robot"),
                2: ("Z axis displacement of end effector of the robot"),
                3: ("Roll of the end effector of the robot"),
                4: ("Pitch of the end effector of the robot"),
                5: ("Yaw of the end effector of the robot"),
                6: ("Gripper open or closed", {0.0: "closed", 1.0: "open"}),
                7: ("Termination status of the episode",{1: "Yes", 0: "No"})
            }
        },
        "utokyo_saytap_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
            "default": {
                0: ("X axis displacement of the end-effector of the robot"),
                1: ("Y axis displacement of the end-effector of the robot"),
                2: ("Z axis displacement of the end-effector of the robot"),
                3: ("Yaw of the robot"),
                4: ("Pitch of the robot"),
                5: ("Roll of the robot"),
                6: ("Gripper open or closed position of the robot", {0.0: "closed", 1.0: "open"})
            }
        },
        "utokyo_xarm_bimanual_converted_externally_to_rlds": {
            "default":{
                0: ("X axis displacement of the end-effector of the left arm of the robot"),
                1: ("Y axis displacement of the end-effector of the left arm of the robot"),
                2: ("Z axis displacement of the end-effector of the left arm of the robot"),
                3: ("Yaw of the left arm of the robot"),
                4: ("Pitch of the left arm of the robot"),
                5: ("Roll of the left arm of the robot"),
                6: ("Gripper open or closed position of the left arm of the robot", {0.0: "closed", 1.0: "open"}),
                7: ("X axis displacement of the end-effector of the right arm of the robot"),
                8: ("Y axis displacement of the end-effector of the right arm of the robot"),
                9: ("Z axis displacement of the end-effector of the right arm of the robot"),
                10: ("Yaw of the right arm of the robot"),
                11: ("Pitch of the right arm of the robot"),
                12: ("Roll of the right arm of the robot"),
                13: ("Gripper open or closed position of the right arm of the robot", {0.0: "closed", 1.0: "open"}),
            }
        },
        "berkeley_mvp_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "berkeley_rpt_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "kaist_nonprehensile_converted_externally_to_rlds": {
            "default": {
                0: ("X axis displacement of the end-effector of the robot in meters"),
                1: ("Y axis displacement of the end-effector of the robot in meters"),
                2: ("Z axis displacement of the end-effector of the robot in meters"),
                3: ("Roll of the end-effector of the robot in radians"),
                4: ("Pitch of the end-effector of the robot in radians"),
                5: ("Yaw of the end-effector of the robot in radians"),
                6: ("Proportional gain coefficient for joint 1 of the robot"),
                7: ("Proportional gain coefficient for joint 2 of the robot"),
                8: ("Proportional gain coefficient for joint 3 of the robot"),
                9: ("Proportional gain coefficient for joint 4 of the robot"),
                10: ("Proportional gain coefficient for joint 5 of the robot"),
                11: ("Proportional gain coefficient for joint 6 of the robot"),
                12: ("Proportional gain coefficient for joint 7 of the robot"),
                13: ("Joint damping ratio coefficient for joint 1 of the robot"),
                14: ("Joint damping ratio coefficient for joint 2 of the robot"),
                15: ("Joint damping ratio coefficient for joint 3 of the robot"),
                16: ("Joint damping ratio coefficient for joint 4 of the robot"),
                17: ("Joint damping ratio coefficient for joint 5 of the robot"),
                18: ("Joint damping ratio coefficient for joint 6 of the robot"),
                19: ("Joint damping ratio coefficient for joint 7 of the robot")
            }
        },
        "stanford_mask_vit_converted_externally_to_rlds": {
            "default": {
                0: ("X axis of the gripper position in meters"),
                1: ("Y axis of the gripper position in meters"),
                2: ("Z axis of the gripper position in meters"),
                3: ("Change in Yaw angle of the end effector of the robot"),
                4: ("Gripper closedness state", {1.0: "closed", -1.0: "open"})
            }
        },
        "tokyo_u_lsmo_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "dlr_sara_pour_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "dlr_sara_grid_clamp_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "dlr_edan_shared_control_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "asu_table_top_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "stanford_robocook_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "eth_agent_affordances": {
            "default":{
                0: ("End effector velocity of the robot in the world X direction", -1, 1),
                1: ("End effector velocity of the robot in the world Y direction", -1, 1),
                2: ("End effector velocity of the robot in the world Z direction", -1, 1),
                3: ("End effector angular velocity of the robot's rotation around the world X axis", -2*np.pi, 2*np.pi),
                4: ("End effector angular velocity of the robot's rotation around the world Y axis", -2*np.pi, 2*np.pi),
                5: ("End effector angular velocity of the robot's rotation around the world Z axis", -2*np.pi, 2*np.pi),
            }
        },
        "imperialcollege_sawyer_wrist_cam": {
            "default": {
                0: ("The delta change in the X axis of the robot's end-effector frame"),
                1: ("The delta change in the Y axis of the robot's end-effector frame"),
                2: ("The delta change in the Z axis of the robot's end-effector frame"),
                3: ("The rotation delta change in the Z axis of the robot's end-effector frame in euler angles"),
                4: ("The rotation delta change in the Y axis of the robot's end-effector frame in euler angles"),
                5: ("The rotation delta change in the X axis of the robot's end-effector frame in euler angles"),
                6: ("The open or close state of the robot's gripper", {1.0: "open", 0.0: "closed"}),
                7: ("Termination status of the episode",{1.0: "Yes", 0.0: "No"})
            }
        },
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "utaustin_mutex": {

            "default": {
                0: None
            }
        },

        "cmu_play_fusion": {
            "default": {
                0: None
            }
        },
        "cmu_stretch": {
            "default": {
                0: None
            }
        },

        "berkeley_gnm_recon": {
            "default": {
                0: None
            }
        },

        "berkeley_gnm_cory_hall": {
            "default": {
                0: None
            }
        },

        "berkeley_gnm_sac_son": {
            "default": {
                0: None
            }
        },

        "conq_hose_manipulation": {
            "default": {
                0: ("X axis displacement of end effector of the robot"),
                1: ("Y axis displacement of end effector of the robot"),
                2: ("Z axis displacement of end effector of the robot"),
                3: ("Roll of the end effector of the robot"),
                4: ("Pitch of the end effector of the robot"),
                5: ("Yaw of the end effector of the robot"),
                6: ("Gripper open or closed", {0.0: "closed", 1.0: "open"})
            }
        },
        "dobbe": {
            "default": {
                0: ("Absolute X axis position of the robot's end-effector"),
                1: ("Absolute Y axis position of the robot's end-effector"),
                2: ("Absolute Z axis position of the robot's end-effector"),
                3: ("Roll of the end effector of the robot"),
                4: ("Pitch of the end effector of the robot"),
                5: ("Yaw of the end effector of the robot"),
                6: ("Gripper opening value", 0,1)
            }
        },
        "io_ai_tech": {
            "default": {
                0: None
            }
        },
        "robo_set": {
            "default": {
                0: None
            }
        },
        "plex_robosuite": {
            "default": {
                0: ("X axis displacement of the end-effector of the robot"),
                1: ("Y axis displacement of the end-effector of the robot"),
                2: ("Z axis displacement of the end-effector of the robot"),
                3: ("X axis delta rotation of the end-effector of the robot"),
                4: ("Y axis delta rotation of the end-effector of the robot"),
                5: ("Z axis delta rotation of the end-effector of the robot"),
                6: ("Gripper opening value", {1.0: "Gripper closed", -1.0: "Gripper open"})

            }
        }
    }

    ACTION_EXCLUSIVENESS = {
        "fractal20220817_data": {
            "default": False
        },
        "kuka": {
            "default": False
        },
        "bridge": {
            "default": False
        },
        "taco_play": {
            "default": False
        },
        "jaco_play": {
            "default": False
        },
        "berkeley_cable_routing": {
            "default": False
        },
        "roboturk": {
            "default": False
        },
        "nyu_door_opening_surprising_effectiveness": {
            "default": False
        },
        "viola": {
            "default": False
        },
        "berkeley_autolab_ur5": {
            "default": False
        },
        "toto": {
            "default": False
        },
        "language_table": {
            "default": False
        },
        "columbia_cairlab_pusht_real": {
            "default": False
        },
        "nyu_rot_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "stanford_hydra_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "nyu_franka_play_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "maniskill_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "furniture_bench_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "cmu_franka_exploration_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "ucsd_kitchen_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "austin_sirius_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "bc_z": {
            "default": False
        },
        "usc_cloth_sim_converted_externally_to_rlds": {
            "default": False
        },
        "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {
            "default": False
        },
        "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {
            "default": False
        },
        "utokyo_saytap_converted_externally_to_rlds": {
            "default": False
        },
        "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
            "default": False
        },
        "utokyo_xarm_bimanual_converted_externally_to_rlds": {
            "default": False
        },
        "berkeley_mvp_converted_externally_to_rlds": {
            "default": False
        },
        "berkeley_rpt_converted_externally_to_rlds": {
            "default": False
        },
        "kaist_nonprehensile_converted_externally_to_rlds": {
            "default": False
        },
        "stanford_mask_vit_converted_externally_to_rlds": {
            "default": False
        },
        "tokyo_u_lsmo_converted_externally_to_rlds": {
            "default": False
        },
        "dlr_sara_pour_converted_externally_to_rlds": {
            "default": False
        },
        "dlr_sara_grid_clamp_converted_externally_to_rlds": {
            "default": False
        },
        "dlr_edan_shared_control_converted_externally_to_rlds": {
            "default": False
        },
        "asu_table_top_converted_externally_to_rlds": {
            "default": False
        },
        "stanford_robocook_converted_externally_to_rlds": {
            "default": False
        },
        "eth_agent_affordances": {
            "default": False
        },
        "imperialcollege_sawyer_wrist_cam": {
            "default": False
        },
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
            "default": False
        },
        "utaustin_mutex": {
            "default": False
        },
        "cmu_play_fusion": {
            "default": False
        },
        "cmu_stretch": {
            "default": False
        },
        "berkeley_gnm_recon": {
            "default": False
        },
        "berkeley_gnm_cory_hall": {
            "default": False
        },
        "berkeley_gnm_sac_son": {
            "default": False
        },
        "conq_hose_manipulation": {
            "default": False
        },
        "dobbe": {
            "default": False
        },
        "io_ai_tech": {
            "default": False
        },
        "robo_set": {
            "default": False
        },
        "plex_robosuite": {
            "default": False
        }
    }

    ADDITIONAL_INSTRUCTIONS = {
        "berkeley_autolab_ur5": {
            "default": [
                "The robot state represents [joint0, joint1, joint2, joint3, joint4, joint5, x, y, z, qx, qy, qz, qw, gripper_is_closed, action_blocked].",
                "action_blocked is binary: 1 if the gripper opening/closing action is being executed and no other actions can be performed; 0 otherwise."
            ]
        }
    }
