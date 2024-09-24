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
    },
    "rt_1_robot_action": {
        "pick and place items": [  # 130 tasks
            "Lift the object off the surface."
        ],
        "move object near another object": [  # 337 tasks
            "Move the first object near the second."
        ],
        "place objects upright": [  # 7 tasks
            "Place an elongated object upright."
        ],
        "open a drawer": [  # 3 tasks
            "Open any of the cabinet drawers."
        ],
        "close a drawer": [  # 3 tasks
            "Close any of the cabinet drawers."
        ],
        "place object into receptacle": [  # 84 tasks
            "Place an object into a receptacle."
        ],
        "pick object into receptacle and place on the counter": [  # 162 tasks
            "Pick an object up from a location and then place it on the counter."
        ]
    },
    "qt_opt": {
        "grasp and pick an object": [
            "Choose a grasp point, and then execute the desired grasp strategy.",
            "Update the grasp strategy continuously based on the most recent observations."
        ]
    },
    "berkeley_bridge": {
        "pick and place": [
            "Put corn in pot.",
            "Put carrot on plate."
        ],
        "push": [
            "Push an object."
        ],
        "reorient objects": [
            "Flip pot upright."
        ],
        "sweep": [
            "Sweep beans into a pile with a bar."
        ],
        "open a door or drawer": [
            "Open drawer.",
            "Open door."
        ],
        "close a door or drawer": [
            "Close drawer.",
            "Close door."
        ],
        "stack blocks": [
            "Stack green block on yellow block."
        ],
        "fold cloths": [
            "Fold thin blue cloth over object.",
            "Fold thick gray cloth over object."
        ],
        "wipe a surface": [
            "Wipe the table with the cloth."
        ],
        "twist knobs": [
            "Twist the knob."
        ],
        "flip a switch": [
            "Flip LED switch."
        ], 
        "turn faucets": [
            "Turn on a faucet."
        ],
        "zip a zipper": [
            "Zip a zipper."
        ]
    },
    "freiburg_franka_play": {
        "interact with toy blocks": [
            "Rotate block left.", 
            "Rotate block right.", 
            "Push block left.", 
            "Push block right."
        ],
        "pick an object": [
            "Lift the block on top of the drawer.",
            "Lift the block inside the drawer.", 
            "Lift the block from the slider.",
            "Lift the block from the container.", 
            "Lift the block from the table."
        ],
        "place an object": [
            "Place the block on top of the drawer.",
            "Place the block inside the drawer.",
            "Place the block in the slider.", 
            "Place the block in the container."
        ],
        "stack objects": [
            "Stack the blocks."
        ],
        "unstack objects": [
            "Unstack the blocks."
        ],
        "open drawer": [
            "Open drawer."
        ],
        "close drawer": [
            "Close drawer."
        ],
        "open sliding door": [
            "Move slider left.",
            "Move slider right."
        ],
        "turn LED lights by pushing buttons": [
            "Turn red light on.",
            "Turn red light off.", 
            "Turn green light on.", 
            "Turn green light off.", 
            "Turn blue light on.",
            "Turn blue light off."
        ]
    },
    "usc_jaco_play": {
        "pick up an object": [
            "Pick up the orange fruit."
        ],
        "put an object down": [
            "Put the black bowl in the sink."
        ]
    },
    "berkeley_cable_routing": {
        "pick up an object": [
            "Pick up the cable."
        ],
        "route a cable": [
            "Route the cable through a number of tight-fitting clips mounted on the table."
        ] 
    },
    "roboturk": {
        "flatten laundry": [
            "Layout laundry."
        ],
        "build a tower from bowls": [
            "Create a tower from bowls."
        ],
        "search objects": [
            "Search for objects."
        ]
    },
    "nyu_vinn": {
        "push objects": [
            "Push objects."
        ],
        "stack objects": [
            "Stack objects."
        ],
        "open door": [
            "Open a cabinet door."
        ]
    },
    "austin_viola": {
        "sort": [
            "Arrange a dining table."
        ],
        "BUDS kitchen": [
            "Make coffee."
        ],
        "stack": [
            "Stack objects."
        ]
    },
    "toto_benchmark": {
        "scoop": [
            "Take a scoop of mixed nuts from the gray ceramic bowl."
        ],
        "pour": [
            "Pour the nuts into the pink plastic cup."
        ]
    },
    "language_table": {
        "push objects": [ # push move nudge put touch 
            "Push the yellow hexagon to the top right corner.",
            "Push the red circle to the bottom right corner.",
            "Push the green star to the bottom left corner."
        ],
        "move objects": [
            "Move the yellow heart to the yellow hexagon.",
            "Move the red star to the red circle."
        ],
        "nudge objects": [
            "Nudge the green star down and left a bit.",
            "Nudge the green circle closer to the green star."
        ],
        "put objects": [
            "Put the red circle on the right side of the blue cube."
        ],
        "touch objects": [
            "Touch the right side of the yellow hexagon."
        ],
        "slide objects": [
            "Slide the heart right."
        ]
    },
    "columbia_pusht_dataset": {
        "push t-shaped blocks": [
            "Push T-shaped block into a fixed goal pose.",
            "Move to a fixed exit zone."
        ]
    },
    "stanford_kuka_multimodal": {  # mentioned here it's been dropped https://docs.google.com/spreadsheets/d/1pwYeYQLkcBWVHwgt0OtL0KlL7VlfZ7q6Y3xfobbOZ7U/edit?usp=sharing
        "insert pegs into holes": [
            # "Move the arm in a free space to make contact with the box.",
            # "Align the peg with the hole."
            "Insert differently-shaped pegs into differently-shaped holes.",
            "Holes have low tolerances (~2mm)."
        ]
    },
    "nyu_rot": {
        # robots task
        "close door": [
            "Close the door."
        ],
        "hang hanger": [
            "Hang the hanger."
        ],
        "erase board": [
            "Erase the board."
        ],
        "reach": [
            "Reach for an object."
        ],
        "hang mug": [
            "Hang the mug."
        ],
        "hang bag": [
            "Hang the bag."
        ],
        "turn knob": [
            "Turn the knob."
        ],
        "stack cups": [
            "Stack the cups."
        ],
        "press switch": [
            "Press the switch."
        ],
        "peg (easy)": [
            "Peg an object into a box with high tolerance."
        ],
        "peg (medium)": [
            "Peg an object into a box with medium tolerance."
        ],
        "peg (hard)": [
            "Peg an object into a box with small tolerance."
        ],
        "open box": [
            "Open the box."
        ],
        "pour": [
            "Pour the drink."
        ]
    },
    "stanford_hydra": {
        "assemble square": [
            "Assemble the square."
        ],
        "hang tool": [
            "Hang the tool."
        ],
        "insert peg": [
            "Insert the peg."
        ],
        "make coffee": [
            "Pick up pod.",
            "Insert pod.",
            "Close lid.",
            "Pick up mug.",
            "Place mug.",
            "Press button."
        ],
        "make toast": [
            "Open toaster.",
            "Pick spatula.",
            "Pick bread.",
            "Place bread.",
            "Place spatula.",
            "Close toaster.",
            "Turn on."
        ],
        "sort dishes": [
            "Pick spoon.",
            "Place spoon.",
            "Grasp plate.",
            "Insert plate.",
            "Pick mug.",
            "Hang mug."
        ]
    },
    "austin_buds": { # dropped during curation, and can't also find it
        "pick object": [
            "Pick the object."
        ],
        "place object in a pot": [
            "Place the object in the pot."
        ],
        "pick up a tool and push objects together using the tool": [
            "Push the object using the picked-up tool."
        ]
    },
    "nyu_franka_play": {
        "open microwave": [
            "Open the microwave door."
        ],
        "open oven door": [
            "Open the oven door."
        ],
        "put the pot in the sink": [
            "Put the pot in the sink."
        ],
        "operate the stove knobs": [
            "Operate the stove knobs."
        ],
    },
    "maniskill": {
        "pick an object": [
            "Pick an isolated object.",
            "Pick an object from the clutter."
        ],
        "move object to another position": [
            "Move the object to its goal position."
        ],
        "stack cubes": [
            "Stack a red cube onto a green cube."
        ],
        "insert peg into box": [
            "Insert a peg into the box."
        ],
        "assemble kits": [
            "Assemble the kits."
        ],
        "plug a charger": [
            "Plug the charger into the outlet on the wall."
        ],
        "turn on a faucet": [
            "Turn on a faucet."
        ]
    },
    "furniture_bench": {
        "grasp": [
            "Grasp a lamp.",
            "Grasp a square table.",
            "Grasp a drawer.",
            "Grasp a cabinet.",
            "Grasp a round table.",
            "Grasp a desk.", 
            "Grasp a stool.",
            "Grasp a chair."
        ],
        "place": [
            "Place a lamp.",
            "Place a square table.",
            "Place a drawer.",
            "Place a cabinet.",
            "Place a round table.",
            "Place a desk.", 
            "Place a stool.",
            "Place a chair."
        ],
        "insert": [
            "Insert a lamp.",
            "Insert a square table.",
            "Insert a drawer.",
            "Insert a cabinet.",
            "Insert a round table.",
            "Insert a desk.", 
            "Insert a stool.",
            "Insert a chair."
        ],
        "screw": [
            "Screw a lamp.",
            "Screw a square table.",
            "Screw a drawer.",
            "Screw a cabinet.",
            "Screw a round table.",
            "Screw a desk.", 
            "Screw a stool.",
            "Screw a chair."
        ]
    },
    "cmu_franka_exploration": {
        "pick veggies": [
            "Pick veggies."
        ],
        "lift knife": [
            "Lift a knife."
        ],
        "open cabinet": [
            "Open a cabinet."
        ],
        "pull drawer": [
            "Pull a drawer."
        ],
        "open dishwasher": [
            "Open a dishwasher."
        ],
        "garbage can": [
            "Garbage can."
        ]
    },
    "ucsd_kitchen": {
    },
    "ucsd_pick_place": {
        "reach": [
            "Reach for an object."
        ],
        "pick": [
            "Pick an object."
        ]
    },
    "austin_sailor": {
        "open": [
            "Open a cabinet."
        ],
        "close": [
            "Close a cabinet."
        ],
        "move": [
            "Move a kettle."
        ],
        "turn on": [
            "Turn on a stove.",
            "Turn on lights."
        ],
        "turn off": [
            "Turn off a stove.",
            "Turn off lights."
        ],
        "pick": [
            "Pick an object."
        ],
        "place": [
            "Place an object."
        ],
        "push": [
            "Push a block."
        ],
        "set up the table": [
            "Set up the table."
        ],
        "clean up the table": [
            "Clean up the table."
        ]
    },
    "austin_sirius": {
        "assemble nut": [
            "Assemble nut."
        ],
        "hang tool": [
            "Hang a tool."
        ],
        "insert gear": [
            "Insert a gear."
        ],
        "pack coffee pod": [
            "Pack coffee pod."
        ]
    },
    "bc_z": {
        "place an object": [
            "Place an object.",
            "Place bottle in ceramic bowl.",
            "Place white sponge in purple bowl.",
            "Place grapes in red bowl.",
            "Place banana in ceramic cup."
        ],
        "push an object": [
            "Push an object.",
            "Push a purple bowl across the table.",
        ],
        "wipe an object": [
            "Wipe an object.",
            "Wipe tray with sponge.",
            "Wipe table surface with banana.",
            "Wipe a surface with brush."
        ],
        "stack an object": [
            "Stack an object.",
            "Stack bowls into tray.",
        ],
        "knock an object over": [
            "Knock an object over.",
            "Knock the paper cup over."
        ],
        "drag an object": [
            "Drag an object.",
            "Drag grapes across the table.",
        ],
        "open": [
            "Open a door."
        ],
        "empty bin": [
            "Empty a bin."
        ],
        "pick up an object": [
            "Pick up an object.",
            "Pick up grapes.",
        ]
    },
    "usc_cloth_sim": {
        "straighten a rope": [
            "Straighten a rope."
        ],
        "fold cloth": [
            "Fold cloth."
        ],
        "fold cloth diagonally pinned": [
            "Fold cloth diagonally pinned."
        ],
        "fold cloth diagonally unpinned": [
            "Fold cloth diagonally unpinned."
        ]
    },
    "tokyo_pr2_fridge_opening": {  # dropped during curation
        "open fridge": [
            "Open the fridge."
        ]
    },
    "tokyo_pr2_tabletop_manipulation": {
        "pick up an object": [
            "Pick up an object.",
            "Pick up grape.",
            "Pick up bread."
        ],
        "place an object": [
            "Place an object.",
            "Place grape.",
            "Place bread."
        ],
        "fold cloths": [
            "Fold cloths."
        ]
    },
    "saytap": {
        "trot": [
            "Trot forward slowly.",
            "Trot forward fast."
        ],
        "lift": [
            "Lift front right leg.",
            "Lift front left leg.",
            "Lift rear right leg.",
            "Lift rear left leg.",
        ],
        "pace": [
            "Pace forward fast.",
            "Pace forward slowly.",
            "Pace backward fast.",
            "Pace backward slowly.",
        ],
        "back off": [
            "Back off! Don't hurt that squirrel."
        ],
        "act": [
            "Act as if the ground is very hot.",
            "Act as if you have a limping front right leg.",
            "Act as if you have a limping front left leg.",
            "Act as if you have a limping rear right leg.",
            "Act as if you have a limping rear left leg."
        ],
        "go": [
            "Go catch that squirrel on the tree."
        ]
    },
    "utokyo_xarm_pickplace": {
        "pick up an object": [
            "Pick up an object.",
            "Pick up the white plate."
        ],
        "place an object": [
            "Place an object.",
            "Place the white plate on the red plate."
        ]
    },
    "utokyo_xarm_bimanual": {
        "reach for an object": [
            "Reach for an object.",
            "Reach for the towel on the table."
        ],
        "unfold an object": [
            "Unfold an object.",
            "Unfold the wrinkled towel."
        ]
    },
    "robonet": {  # not part of our used data
        "placeholder": []
    },
    "berkeley_mvp_data": {
        "reach for an object": [
            "Reach for an object."
        ],
        "push an object": [
            "Push an object.",
            "Move an object."
        ],
        "pick up an object": [
            "Pick up an object."
        ]
    },
    "berkeley_rpt_data": {
        "pick up an object": [
            "Pick up an object."
        ],
        "stack an object": [
            "Stack an object."
        ],
        "destack an object": [
            "Destack an object."
        ]
    },
    "kaist_nonprehensile_objects": {
        "push an object": [
            "Push an object.",
            "Objects are subjected to external forces to induce translational movement."
        ],
        "drag an object": [
            "Drag an object.",
            "Objects are dragged across the surface without lifting."
        ],
        "rotate an object": [
            "Rotate an object.",
            "Objects are rotated around the vertical axis."
        ],
        "topple an object": [
            "Topple an object.",
            "Objects are caused to fall or rise from an initial stable position to another orientation."
        ]
    },
    "qut_dynamic_grasping": {  # not used in our dataset
        "placeholder": []
    },
    "stanford_maskvit_data": {
        "pick up an object": [
            "Pick up an object."
        ],
        "push an object": [
            "Push an object."
        ]
    },
    "lsmo_dataset": {
        "avoid an obstacle": [
            "Avoid the obstacle on the table."
        ],
        "reach for an object": [
            "Reach for an object."
        ]
    },
    "dlr_sara_pour_dataset": {
        "move towards an object": [
            "Move towards an object."
        ],
        "pour into a cup": [
            "Pour ping-pong balls from a cup held in the end-effector into the cup placed on the table."
        ]
    },
    "dlr_sara_grid_clamp_dataset": {
        "place an object": [
            "Place an object.",
            "Place the grid clamp in the grids on the table, similar to placing a peg in the hole."
        ]
    },
    "dlr_wheelchair_shared_control": {
        "grasp an object": [
            "Grasp an object on the tabletop.",
            "Grasp an object on the shelf."
        ]
    },
    "asu_tabletop_manipulation": {
        "pick an object": [
            "Pick an object."
        ],
        "push an object": [
            "Push an object.",
            "Push an object across the table."
        ],
        "rotate an object": [
            "Rotate an object."
        ],
        "avoid an obstacle": [
            "Avoid an obstacle."
        ],
        "place an object": [
            "Place an object in relation to other objects in the environment."
        ]
    },
    "stanford_robocook": {
        "pinch the dough": [
            "Pinch the dough with an asymmetric gripper.",
            "Pinch the dough with a two-plane symmetric gripper.",
            "Pinch the dough with a two-rod symmetric gripper."
        ],
        "press the dough": [
            "Press the dough with a circle press.",
            "Press the dough with a square press.",
            "Press the dough with a circle punch.",
            "Press the dough with a square punch.",
        ],
        "roll the dough": [
            "Roll the dough with a small roller.",
            "Roll the dough with a large roller."
        ]
    },
    "eth_agent_affordances": {
        "open door": [
            "Open the door starting from different initial positions and door angles."
        ],
        "close door": [
            "Close the door starting from different initial positions and door angles."
        ],
        "open drawer": [
            "Open the drawer starting from different initial positions and drawer angles."
        ],
        "close drawer": [
            "Close the drawer starting from different initial positions and drawer angles."
        ]
    },
    "imperial_wrist_cam": {
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
        "pick up apple": [
            "Pick up an apple from the table."
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
        "pick up shoe": [
            "Pick up a shoe from the table."
        ],
        "pour in mug": [
            "Starting with a cup in the end-effector, pour into a mug on the table - success is detected by dropping a marble from the cup to the mug, mimicking a liquid."
        ],
        "put apple in pot": [
            "Starting with an apple in the end-effector, drop it in a pot on the table."
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
    "cmu_franka_pick_insert_data": {
        "insert onto square peg": [
            "For this task we restrict the orientations of the square ring (blue object) and the peg on which to insert.",
            "This allows the robot to perform the task without changing gripper orientations.",
            "Further, we use a region of 40cm × 30cm in front of the robot to spawn both the base and ring.",
            "Finally, the default task configuration provides 20 different peg colors, of which we use the first 10 colors for training and remaining 10 colors for robustness experiments."
        ],
        "pick and lift small": [
            "For this task, we again use a region of 40cm × 30cm in front of the robot to spawn all objects.",
            "We also restrict the orientation of each object such that it can be grasped directly without requiring gripper orientation changes."
        ],
        "sort shapes": [
            "The default configuration for the shape-sorting task considers 4 different shaped objects (see Figure 3 Bottom-Left) – square, cylinder, triangle, star, moon.",
            "In the default RLBench configuration most objects directly stick to the robot finger and are simply dropped into the hole for task completion.",
            "However, with closed loop control we find that non-symmetric objects (star, triangle, and moon) can have significant post-grasp displacement such that it is impossible to insert these objects without changing gripper orientation.",
            "Hence, we exclude these two objects from evaluation and only use symmetric square and cylinder objects."
        ],
        "take usb out": [
            "This task requires the robot to unplug a USB inserted into the computer.", 
            "However, the default configuration for this task requires 6-dof control.",
            "To avoid this, we create smaller computer and USB assets and mount them vertically on the table such that the USB can be unplugged without changing hand orientation."
        ]
    },
    "austin_mutex": {
        # this data has +150 different tasks, with not enough info about them in the paper
        "put an object": [
            "Put an object.",
            "Put the bowl on the table."
        ],
        "open an object's door": [
            "Open an object's door.",
            "Open the air fryer door."
        ],
        "take out an object": [
            "Take out a tray from the oven."
        ],
        "place an object": [
            "Place bread on the tray."
        ],
        "hold an object": [
            "Hold an object."
        ],
        "move an object": [
            "Move an object."
        ],
        "grip an object": [
            "Grip an object."
        ]
    },
    "cmu_play_fusion": {
        "pick up an object": [
            "Pick up an object.",
            "Pick up a block."
        ],
        "open an object": [
            "Open an object.",
            "Open a door.",
            "Open a drawer."
        ],
        "place an object": [
            "Place an object.",
            "Place a block.",
            "Place a block in the slider."
        ],
        "close an object": [
            "Close an object.",
            "Close a door.",
            "Close a drawer."
        ],
        "turn on an object": [
            "Turn on an object.",
            "Turn on the LED.",
            "Turn on the lights."
        ],
        "push an object": [
            "Push an object.",
            "Push the block."
        ]
    },
    "cmu_stretch": {
        # they didn't provide enough info about the tasks
        "open an object": [
            "Open an object.",
            "Open a door.",
            "Open a dishwasher.",
            "Open a cabinet."
        ],
        "slide an object": [
            "Slide an object.",
            "Slide a door."
        ],
        "pull an object": [
            "Pull an object.",
            "Pull out a drawer."
        ],
        "lift an object": [
            "Lift an object.",
            "Lift a lid.",
            "Lift a knife."
        ],
        "garbage an object": [
            "Garbage an object.",
            "Garbage a can."
        ]
    },
    "recon": {
        "explore environment": [
            "Ignore distractors, and explore a non-stationary environment, successfully discovering and navigating to the visually-specified goal."
        ]
    },
    "coryhall": {
        "navigate hallways": [
            "Autonomously navigate complex and unstructured environments such as roads, buildings, or forests."
        ]
    },
    "sacson": {
        "navigate environments": [
            "Navigate pedestrian-rich indoor and outdoor environments such as offices, school buildings."
        ]
    },
    "conqhose": {
        "grab an object": [
            "Grab an object.",
            "Grab the end of the vacuum hose around in an office environment."
        ],
        "lift an object": [
            "Lift an object.",
            "Lift the end of the vacuum hose around in an office environment."
        ],
        "drag an object": [
            "Drag an object.",
            "Drag the end of the vacuum hose around in an office environment."
        ]
    },
    "dobbe": {
        "pick up an object": [
            "Pick up an object.",
            "Pick up paper towel roll.",
            "Pick up paper bag.",
            "Pick up hat.",
            "Pick up trash bag.",
            "Pick up hand towel.",
            "Pick up kitchen towel.",
            "Pick up tissue roll."
        ],
        "open an object": [
            "Open an object.",
            "Open a door.",
            "Open cabinet door.",
            "Open shower curtain.",
            "Open dishwasher door.",
            "Open air fryer door.",
            "Open freezer door.",
            "Open vertical window blinds."
        ],
        "close an object": [
            "Close an object.",
            "Close a door.",
            "Close cabinet door.",
            "Close shower curtain.",
            "Close dishwasher door.",
            "Close air fryer door."
        ],
        "place an object": [
            "Place an object.",
            "Place keychain.",
            "Place spice.",
            "Place massager."
        ],
        "pull an object": [
            "Pull an object.",
            "Pull out dining chair.",
            "Pull book from shelf.",
            "Pull chair.",
            "Pull desk chair.",
            "Pull out dining chair.",
            "Pull side table.",
            "Pull out dining stool."
        ],
        "flush toilet": [
            "Flush toilet."
        ],
        "straighten cushion": [
            "Straighten cushion."
        ],
        "pour an object": [
            "Pour an object.",
            "Pour chocolate almond."
        ],
        "unplug an object": [
            "Unplug an object.",
            "Unplug charger."
        ],
        "rotate an object": [
            "Rotate an object.",
            "Rotate speaker knob."
        ],
        "adjust an object": [
            "Adjust an object.",
            "Adjust oven knob."
        ],
        "push an object": [
            "Push an object.",
            "Push toaster button."
        ],
        "put an object": [
            "Put an object.",
            "Put rag in laundry."
        ]
    },
    "io_ai_office_picknplace": {
        "pick up an object": [
            "Pick up an object.",
            "Pick up the glue from the plate.",
            "Pick up the stapler."
        ],
        "place an object": [
            "Place an object.",
            "Place the glue on the plate.",
            "Place the stapler on the desk."
        ]
    },
    "roboset": {
        "wipe an object": [
            "Wipe an object."
        ],
        "pick up an object": [
            "Pick up an object."
        ],
        "place an object": [
            "Place an object."
        ],
        "cap an object": [
            "Cap an object."
        ],
        "uncap an object": [
            "Uncap an object."
        ]
    },
    "spoc": {
        "fetch an object": [
            "Find and pick up an object."
        ],
        "navigate to an object": [
            "Locate an object."
        ],
        "search for an object": [
            "Search for an object.",
            "Search for a bed."
        ],
        "pick up an object": [
            "Pick up a specified object in agent line of sight.",
        ],
        "find an object": [
            "Find an object.",
            "Find a hosueplant."
        ],
        "place an object": [
            "Place an object."
        ]
    },
    "plex_robotsuite": {
        "open an object": [
            "Open an object."
        ],
        "close an object": [
            "Close an object."
        ],
        "pick up an object": [
            "Pick up an object."
        ],
        "place an object": [
            "Place an object."
        ],
        "put an object": [
            "Put an object."
        ],
        "insert an object": [
            "Insert an object."
        ],
        "lift an object": [
            "Lift an object."
        ],
        "assemble an object": [
            "Assemble an object."
        ],
        "stack an object": [
            "Stack an object."
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
    },
    "rt_1_robot_action": {
        "pick and place items": {
            0: None
        },
        "move object near another object": {
            0: None
        },
        "place objects up-right": {
            0: None
        },
        "open a drawer": {
            0: None
        },
        "close a drawer": {
            0: None
        },
        "place object into receptacle": {
            0: None
        },
        "pick object into receptacle and place on the counter": {
            0: None
        }
    },
    "qt_opt": {
        "grasp and pick an object": {
            0: None
        }
    },
    "berkeley_bridge": {
        "pick and place": {
            0: None
        },
        "push": {
            0: None
        },
        "reorient objects": {
            0: None
        },
        "sweep": {
            0: None
        },
        "open a door or drawer": {
            0: None
        },
        "close a door or drawer": {
            0: None
        },
        "stack blocks": {
            0: None
        },
        "fold cloths": {
            0: None
        },
        "wipe a surface": {
            0: None
        },
        "twist knobs": {
            0: None
        },
        "flip a switch": {
            0: None
        },
        "turn faucets": {
            0: None
        },
        "zip a zipper": {
            0: None
        }
    },
    "freiburg_franka_play": {
        "interact with toy blocks": {
            0: None
        },
        "pick an object": {
            0: None
        },
        "place an object": {
            0: None
        },
        "stack objects": {
            0: None
        },
        "unstack objects": {
            0: None
        },
        "open drawer": {
            0: None
        },
        "close drawer": {
            0: None
        },
        "open sliding door": {
            0: None
        },
        "turn LED lights by pushing buttons": {
            0: None
        }
    },
    "usc_jaco_play": {
        "pick up an object": {
            0: None
        },
        "put an object down": {
            0: None
        }
    },
    "berkeley_cable_routing": {
        "pick up an object": {
            0: None
        },
        "route a cable": {
            0: None
        }
    },
    "roboturk": {
        "flatten laundry": {
            0: None
        },
        "build a tower from bowls": {
            0: None
        },
        "search objects": {
            0: None
        }
    },
    "nyu_vinn": {
        "push objects": {
            0: None
        },
        "stack objects": {
            0: None
        },
        "open door": {
            0: None
        }
    },
    "austin_viola": {
        "sort": {
            0 : None
        },
        "BUDS kitchen": {
            0: None
        },
        "stack": {
            0: None
        }
    },
    "toto_benchmark": {
        "scoop": {
            0: None
        },
        "pour": {
            0: None
        }
    },
    "language_table": {
        "push objects": {
            0: None
        },
        "move objects": {
            0: None
        },
        "nudge objects": {
            0: None
        },
        "put objects": {
            0: None
        },
        "touch objects": {
            0: None
        },
        "slide objects": {
            0: None
        }
    },
    "columbia_pusht_dataset": {
        "push t-shaped blocks": {
            0: None
        }
    },
    "stanford_kuka_multimodal": {
        "insert pegs into holes": {
            0: None
        }
    },
    "nyu_rot": {
        "close door": {
            0: None
        },
        "hang hanger": {
            0: None
        },
        "erase board": {
            0: None
        },
        "reach": {
            0: None
        },
        "hang mug": {
            0: None
        },
        "hang bag": {
            0: None
        },
        "turn knob": {
            0: None
        },
        "stack cups": {
            0: None
        },
        "press switch": {
            0: None
        },
        "peg (easy)": {
            0: None
        },
        "peg (medium)": {
            0: None
        },
        "peg (hard)": {
            0: None
        },
        "open box": {
            0: None
        },
        "pour": {
            0: None
        }
    },
    "stanford_hydra": {
        "assemble square": {
            0: None
        },
        "hang tool": {
            0: None
        },
        "insert peg": {
            0: None
        },
        "make coffee": {
            0: None
        },
        "make toast": {
            0: None
        },
        "sort dishes": {
            0: None
        }
    },
    "austin_buds": {
        "pick object": {
            0: None
        },
        "place object in a pot": {
            0: None
        },
        "pick up a tool and push objects together using the tool": {
            0: None
        }
    },
    "nyu_franka_play": {
        "open microwave": {
            0: None
        },
        "open oven door": {
            0: None
        },
        "put the pot in the sink": {
            0: None
        },
        "operate the stove knobs": {
            0: None
        },
    },
    "maniskill": {
        "pick an object": {
            0: None
        },
        "move object to another position": {
            0: None
        },
        "stack cubes": {
            0: None
        },
        "insert peg into box": {
            0: None
        },
        "assemble kits": {
            0: None
        },
        "plug a charger": {
            0: None
        },
        "turn on a faucet": {
            0: None
        }
    },
    "furniture_bench": {
        "grasp": {
            0: None
        },
        "place": {
            0: None
        },
        "insert": {
            0: None
        },
        "screw": {
            0: None
        }
    },
    "cmu_franka_exploration": {
        "pick veggies": {
            0: None
        },
        "lift knife": {
            0: None
        },
        "open cabinet": {
            0: None
        },
        "pull drawer": {
            0: None
        },
        "open dishwasher": {
            0: None
        },
        "garbage can": {
            0: None
        }
    },
    "ucsd_kitchen": {
    },
    "ucsd_pick_place": {
        "reach": {
            0: None
        },
        "pick": {
            0: None
        },
    },
    "austin_sailor": {
        "open": {
            0: None
        },
        "close": {
            0: None
        },
        "move": {
            0: None
        },
        "turn on": {
            0: None
        },
        "turn off": {
            0: None
        },
        "pick": {
            0: None
        },
        "place": {
            0: None
        },
        "push": {
            0: None
        },
        "set up the table": {
            0: None
        },
        "clean up the table": {
            0: None
        },
    },
    "austin_sirius": {
        "assemble nut": {
            0: None
        },
        "hang tool": {
            0: None
        },
        "insert gear": {
            0: None
        },
        "pack coffee pod": {
            0: None
        },
    },
    "bc_z": {
        "place an object": {
            0: None
        },
        "push an object": {
            0: None
        },
        "wipe an object": {
            0: None
        },
        "stack an object": {
            0: None
        },
        "knock an object over": {
            0: None
        },
        "drag an object": {
            0: None
        },
        "open": {
            0: None
        },
        "empty bin": {
            0: None
        },
        "pick up an object": {
            0: None
        },
    },
    "usc_cloth_sim": {
        "straighten a rope": {
            0: None
        },
        "fold cloth": {
            0: None
        },
        "fold cloth diagonally pinned": {
            0: None
        },
        "fold cloth diagonally unpinned": {
            0: None
        },
    },
    "tokyo_pr2_fridge_opening": {  # dropped during curation
        "open fridge": {
            0: None
        }
    },
    "tokyo_pr2_tabletop_manipulation": {
        "pick up an object": {
            0: None
        },
        "place an object": {
            0: None
        },
        "fold cloths": {
            0: None
        },
    },
    "saytap": {
        "trot": {
            0: None
        },
        "lift": {
            0: None
        },
        "pace": {
            0: None
        },
        "back off": {
            0: None
        },
        "act": {
            0: None
        },
        "go": {
            0: None
        },
    },
    "utokyo_xarm_pickplace": {
        "pick up an object": {
            0: None
        },
        "place an object": {
            0: None
        },
    },
    "utokyo_xarm_bimanual": {
        "reach for an object": {
            0: None
        },
        "unfold an object": {
            0: None
        },
    },
    "robonet": {  # not part of our used data
        "placeholder": {
            0: None
        }
    },
    "berkeley_mvp_data": {
        "reach for an object": {
            0: None
        },
        "push an object": {
            0: None
        },
        "pick up an object": {
            0: None
        }
    },
    "berkeley_rpt_data": {
        "pick up a object": {
            0: None
        },
        "stack an object": {
            0: None
        },
        "destack an object": {
            0: None
        }
    },
    "kaist_nonprehensile_objects": {
        "push an object": {
            0: None
        },
        "drag an object": {
            0: None
        },
        "rotate an object": {
            0: None
        },
        "topple an object": {
            0: None
        }
    },
    "qut_dynamic_grasping": {  # not used in our dataset
        "placeholder": {
            0: None
        }
    },
    "stanford_maskvit_data": {
        "pick up an object": {
            0: ("The change in cartesian end-effector position X", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            1: ("The change in cartesian end-effector position Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            2: ("The change in cartesian end-effector position Z", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            3: ("The change in yaw", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            4: ("The gripper closing action", {0: "Close", 1: "Open"}),  # it was never mentioned what is closed and opened mapped to
        },
        "push an object": {
            0: ("The change in cartesian end-effector position X", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            1: ("The change in cartesian end-effector position Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            2: ("The change in cartesian end-effector position Z", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            3: ("The change in yaw", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            4: ("The gripper closing action", {0: "Close", 1: "Open"}),  # it was never mentioned what is closed and opened mapped to
        }
    },
    "lsmo_dataset": {
        "avoid an obstacle": {
            0: None
        },
        "reach for an object": {
            0: None
        }
    },
    "dlr_sara_pour_dataset": {
        "move towards an object": {
            0: None
        },
        "pour into a cup": {
            0: None
        }
    },
    "dlr_sara_grid_clamp_dataset": {
        "place an object": {
            0: None
        }
    },
    "dlr_wheelchair_shared_control": {
        "grasp an object": {
            0: None
        }
    },
    "asu_tabletop_manipulation": {
        "pick an object": {
            0: ("The change in cartesian end-effector position X", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            1: ("The change in cartesian end-effector position Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,
            2: ("The change in cartesian end-effector position Z", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,
            3: ("The change in end-effector orientation R", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,,
            4: ("The change in end-effector orientation P", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,,,
            5: ("The change in end-effector orientation Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,,,
            6: ("The gripper closing action", {0: "Close", 1: "Open"}),  # it was never mentioned what is closed and opened mapped to
        },
        "push an object": {
            0: ("The change in cartesian end-effector position X", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            1: ("The change in cartesian end-effector position Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,
            2: ("The change in cartesian end-effector position Z", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,
            3: ("The change in end-effector orientation R", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,,
            4: ("The change in end-effector orientation P", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,,,
            5: ("The change in end-effector orientation Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,,,
            6: ("The gripper closing action", {0: "Close", 1: "Open"}),  # it was never mentioned what is closed and opened mapped to
        },
        "rotate an object": {
            0: ("The change in cartesian end-effector position X", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            1: ("The change in cartesian end-effector position Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,
            2: ("The change in cartesian end-effector position Z", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,
            3: ("The change in end-effector orientation R", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,,
            4: ("The change in end-effector orientation P", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,,,
            5: ("The change in end-effector orientation Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,,,
            6: ("The gripper closing action", {0: "Close", 1: "Open"}),  # it was never mentioned what is closed and opened mapped to
        },
        "avoid an obstacle": {
            0: ("The change in cartesian end-effector position X", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            1: ("The change in cartesian end-effector position Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,
            2: ("The change in cartesian end-effector position Z", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,
            3: ("The change in end-effector orientation R", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,,
            4: ("The change in end-effector orientation P", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,,,
            5: ("The change in end-effector orientation Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,,,
            6: ("The gripper closing action", {0: "Close", 1: "Open"}),  # it was never mentioned what is closed and opened mapped to
        },
        "place an object": {
            0: ("The change in cartesian end-effector position X", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            1: ("The change in cartesian end-effector position Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,
            2: ("The change in cartesian end-effector position Z", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,
            3: ("The change in end-effector orientation R", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,,
            4: ("The change in end-effector orientation P", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,,,
            5: ("The change in end-effector orientation Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper,,,
            6: ("The gripper closing action", {0: "Close", 1: "Open"}),  # it was never mentioned what is closed and opened mapped to
        }
    },
    "stanford_robocook": {
        "pinch the dough": {
            0: None
        },
        "press the dough": {
            0: None
        },
        "roll the dough": {
            0: None
        }
    },
    "eth_agent_affordances": {
        "open door": {
            0: None
        },
        "close door": {
            0: None
        },
        "open drawer": {
            0: None
        },
        "close drawer": {
            0: None
        }
    },
    "imperial_wrist_cam": {
        "grasp can": {
            0: None
        },
        "hang cup": {
            0: None
        },
        "insert cap in bottle": {
            0: None
        },
        "insert toast": {
            0: None
        },
        "open bottle": {
            0: None
        },
        "open lid": {
            0: None
        },
        "pick up apple": {
            0: None
        },
        "pick up bottle": {
            0: None
        },
        "pick up kettle": {
            0: None
        },
        "pick up mug": {
            0: None
        },
        "pick up pan": {
            0: None
        },
        "pick up shoe": {
            0: None
        },
        "pour in mug": {
            0: None
        },
        "put apple in pot": {
            0: None
        },
        "put cup in dishwasher": {
            0: None
        },
        "stack bowls": {
            0: None
        },
        "swipe": {
            0: None
        }
    },
    "cmu_franka_pick_insert_data": {
        # they only mentioned it's 6D
        "insert onto square peg": {
            0: None
        },
        "pick and lift small": {
            0: None
        },
        "sort shapes": {
            0: None
        },
        "take usb out": {
            0: None
        },
    },
    "austin_mutex": {
        "put an object": {
            0: None
        },
        "open an object's door": {
            0: None
        },
        "take out an object": {
            0: None
        },
        "place an object": {
            0: None
        },
        "hold an object": {
            0: None
        },
        "move an object": {
            0: None
        },
        "grip an object": {
            0: None
        }
    },
    "cmu_play_fusion": {
        "pick up an object": {
            0: None
        },
        "open an object": {
            0: None
        },
        "place an object": {
            0: None
        },
        "close an object": {
            0: None
        },
        "turn on an object": {
            0: None
        },
        "push an object": {
            0: None
        }
    },
    "cmu_stretch": {
        # it's mentioned that it's a 6D action space
        "open an object": {
            0: None
        },
        "slide an object": {
            0: None
        },
        "pull an object": {
            0: None
        },
        "lift an object": {
            0: None
        },
        "garbage an object": {
            0: None
        }
    },
    "recon": {
        "explore environment": {
            0: None
        }
    },
    "coryhall": {
        "navigate hallways": {
            0: None
        }
    },
    "sacson": {
        "navigate environments": {
            0: None
        }
    },
    "conqhose": {
        "grab an object": {
            0: ("The delta pose of hand in current hand frame for X", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            1: ("The delta pose of hand in current hand frame for Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            2: ("The delta pose of hand in current hand frame for Z", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            3: ("The delta pose of hand in current hand frame for R", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            4: ("The delta pose of hand in current hand frame for P", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            5: ("The delta pose of hand in current hand frame for Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            6: ("The delta pose of hand in gripper closing action", None),  # no info were found about this
            7: ("Termination", {1: "True", 0: "False"})
        },
        "lift an object": {
            0: ("The delta pose of hand in current hand frame for X", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            1: ("The delta pose of hand in current hand frame for Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            2: ("The delta pose of hand in current hand frame for Z", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            3: ("The delta pose of hand in current hand frame for R", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            4: ("The delta pose of hand in current hand frame for P", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            5: ("The delta pose of hand in current hand frame for Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            6: ("The delta pose of hand in gripper closing action", None),  # no info were found about this
            7: ("Termination", {1: "True", 0: "False"})
        },
        "drag an object": {
            0: ("The delta pose of hand in current hand frame for X", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            1: ("The delta pose of hand in current hand frame for Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            2: ("The delta pose of hand in current hand frame for Z", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            3: ("The delta pose of hand in current hand frame for R", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            4: ("The delta pose of hand in current hand frame for P", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            5: ("The delta pose of hand in current hand frame for Y", None, None),  # should be a continuous value, but they never said anything about the ranges in the paper
            6: ("The delta pose of hand in gripper closing action", None),  # no info were found about this
            7: ("Termination", {1: "True", 0: "False"})
        },
    },
    "dobbe": {
        # it's mentioned it's 7 dims, but not enough info given about to write the actions spaces descriptions
        "pick up an object": {
            0: None
        },
        "open an object": {
            0: None
        },
        "close an object": {
            0: None
        },
        "place an object": {
            0: None
        },
        "pull an object": {
            0: None
        },
        "flush toilet": {
            0: None
        },
        "straighten cushion": {
            0: None
        },
        "pour an object": {
            0: None
        },
        "unplug an object": {
            0: None
        },
        "rotate an object": {
            0: None
        },
        "adjust an object": {
            0: None
        },
        "push an object": {
            0: None
        },
        "put an object": {
            0: None
        }
    },
    "io_ai_office_picknplace": {
        "pick up an object": {
            0: None
        },
        "place an object": {
            0: None
        }
    },
    "roboset": {
        "wipe an object": {
            0: None
        },
        "pick up an object": {
            0: None
        },
        "place an object": {
            0: None
        },
        "cap an object": {
            0: None
        },
        "uncap an object": {
            0: None
        },
    },
    "spoc": {  # TODO: they described the action space in their paper, get back to it
        "fetch an object": {
            0: None,
        },
        "navigate to an object": {
            0: None,
        },
        "search for an object": {
            0: None,
        },
        "pick up an object": {
            0: None,
        },
        "find an object": {
            0: None,
        },
        "place an object": {
            0: None,
        }
    },
    "plex_robotsuite": {
        # it's mentioned that the actions space is 7 dims, 6 dims for the gripper pose control, and 1 of opening and closing, that's it
        "open an object": {
            0: None,
        },
        "close an object": {
            0: None,
        },
        "pick up an object": {
            0: None,
        },
        "place an object": {
            0: None,
        },
        "put an object": {
            0: None,
        },
        "insert an object": {
            0: None,
        },
        "lift an object": {
            0: None,
        },
        "assemble an object": {
            0: None,
        },
        "stack an object": {
            0: None,
        },
    }
}

ACTION_EXCLUSIVENESS = {
    "berkeley_autolab_ur5": {
        "take the tiger out of the red bowl and put it in the grey bowl": False,
        "sweep the green cloth to the left side of the table": False,
        "pick up the blue cup and put it into the brown cup": False,
        "put the ranch bottle into the pot": False
    },
    "rt_1_robot_action": {
        "pick and place items": False,
        "move object near another object": False,
        "place objects upright": False,
        "open a drawer": False,
        "close a drawer": False,
        "place object into receptacle": False,
        "pick object into receptacle and place on the counter": False
    },
    "qt_opt": {
        "grasp and pick an object": False
    },
    "berkeley_bridge": {
        "pick and place": False,
        "push": False,
        "reorient objects": False,
        "sweep": False,
        "open a door or drawer": False,
        "close a door or drawer": False,
        "stack blocks": False,
        "fold cloths": False,
        "wipe a surface": False,
        "twist knobs": False,
        "flip a switch": False,
        "turn faucets": False,
        "zip a zipper": False
    },
    "freiburg_franka_play": {
        "interact with toy blocks": False,
        "pick an object": False,
        "place an object": False,
        "stack objects": False,
        "unstack objects": False,
        "open drawer": False,
        "close drawer": False,
        "open sliding door": False,
        "turn LED lights by pushing buttons": False
    },
    "usc_jaco_play": {
        "pick up an object": False,
        "put an object down": False
    },
    "berkeley_cable_routing": {
        "pick up an object": False,
        "route a cable": False
    },
    "roboturk": {
        "flatten laundry": False,
        "build a tower from bowls": False,
        "search objects": False
    },
    "nyu_vinn": {
        "push objects": False,
        "stack objects": False,
        "open door": False
    },
    "austin_viola": {
        "sort": False,
        "BUDS kitchen": False,
        "stack": False
    },
    "toto_benchmark": {
        "scoop": False,
        "pour": False
    },
    "language_table": {
        "push objects": False,
        "move objects": False,
        "nudge objects": False,
        "put objects": False,
        "touch objects": False,
        "slide objects": False
    },
    "columbia_pusht_dataset": {
        "push t-shaped blocks": False
    },
    "stanford_kuka_multimodal": {
        "insert pegs into holes": False
    },
    "nyu_rot": {
        "close door": False,
        "hang hanger": False,
        "erase board": False,
        "reach": False,
        "hang mug": False,
        "hang bag": False,
        "turn knob": False,
        "stack cups": False,
        "press switch": False,
        "peg (easy)": False,
        "peg (medium)": False,
        "peg (hard)": False,
        "open box": False,
        "pour": False
    },
    "stanford_hydra": {
        "assemble square": False,
        "hang tool": False,
        "insert peg": False,
        "make coffee": False,
        "make toast": False,
        "sort dishes": False
    },
    "austin_buds": {
        "pick object": False,
        "place object in a pot": False,
        "pick up a tool and push objects together using the tool": False
    },
    "nyu_franka_play": {
        "open microwave": False,
        "open oven door": False,
        "put the pot in the sink": False,
        "operate the stove knobs": False,
    },
    "maniskill": {
        "pick an object": False,
        "move object to another position": False,
        "stack cubes": False,
        "insert peg into box": False,
        "assemble kits": False,
        "plug a charger": False,
        "turn on a faucet": False
    },
    "furniture_bench": {
        "grasp": False,
        "place": False,
        "insert": False,
        "screw": False,
    },
    "cmu_franka_exploration": {
        "pick veggies": False,
        "lift knife": False,
        "open cabinet": False,
        "pull drawer": False,
        "open dishwasher": False,
        "garbage can": False,
    },
    "ucsd_kitchen": {
    },
    "ucsd_pick_place": {
        "reach": False,
        "pick": False,
    },
    "austin_sailor": {
        "open": False,
        "close": False,
        "move": False,
        "turn on": False,
        "turn off": False,
        "pick": False,
        "place": False,
        "push": False,
        "set up the table": False,
        "clean up the table": False,
    },
    "austin_sirius": {
        "assemble nut": False,
        "hang tool": False,
        "insert gear": False,
        "pack coffee pod": False
    },
    "bc_z": {
        "place an object": False,
        "push an object": False,
        "wipe an object": False,
        "stack an object": False,
        "knock an object over": False,
        "drag an object": False,
        "open": False,
        "empty bin": False,
        "pick up an object": False,
    },
    "usc_cloth_sim": {
        "straighten a rope": False,
        "fold cloth": False,
        "fold cloth diagonally pinned": False,
        "fold cloth diagonally unpinned": False
    },
    "tokyo_pr2_fridge_opening": {  # dropped during curation
        "open fridge": False
    },
    "tokyo_pr2_tabletop_manipulation": {
        "pick up an object": False,
        "place an object": False,
        "fold cloths": False
    },
    "saytap": {
        "trot": False,
        "lift": False,
        "pace": False,
        "back off": False,
        "act": False,
        "go": False
    },
    "utokyo_xarm_pickplace": {
        "pick up an object": False,
        "place an object": False
    },
    "utokyo_xarm_bimanual": {
        "reach for an object": False,
        "unfold an object": False
    },
    "robonet": {  # not part of our used data
        "placeholder": False
    },
    "berkeley_mvp_data": {
        "reach for an object": False,
        "push an object": False,
        "pick up an object": False
    },
    "berkeley_rpt_data": {
        "pick up an object": False,
        "stack an object": False,
        "destack an object": False
    },
    "kaist_nonprehensile_objects": {
        "push an object": False,
        "drag an object": False,
        "rotate an object": False,
        "topple an object": False
    },
    "qut_dynamic_grasping": {  # not used in our dataset
        "placeholder": False
    },
    "stanford_maskvit_data": {
        "pick up an object": False,
        "push an object": False
    },
    "lsmo_dataset": {
        "avoid an obstacle": False,
        "reach for an object": False
    },
    "dlr_sara_pour_dataset": {
        "move towards an object": False,
        "pour into a cup": False
    },
    "dlr_sara_grid_clamp_dataset": {
        "place an object": False
    },
    "dlr_wheelchair_shared_control": {
        "grasp an object": False
    },
    "asu_tabletop_manipulation": {
        "pick an object": False,
        "push an object": False,
        "rotate an object": False,
        "avoid an obstacle": False,
        "place an object": False
    },
    "stanford_robocook": {
        "pinch the dough": False,
        "press the dough": False,
        "roll the dough": False
    },
    "eth_agent_affordances": {
        "open door": False,
        "close door": False,
        "open drawer": False,
        "close drawer": False
    },
    "imperial_wrist_cam": {
        "grasp can": False,
        "hang cup": False,
        "insert cap in bottle": False,
        "insert toast": False,
        "open bottle": False,
        "open lid": False,
        "pick up apple": False,
        "pick up bottle": False,
        "pick up kettle": False,
        "pick up mug": False,
        "pick up pan": False,
        "pick up shoe": False,
        "pour in mug": False,
        "put apple in pot": False,
        "put cup in dishwasher": False,
        "stack bowls": False,
        "swipe": False
    },
    "cmu_franka_pick_insert_data": {
        "insert onto square peg": False,
        "pick and lift small": False,
        "sort shapes": False,
        "take usb out": False
    },
    "austin_mutex": {
        "put an object": False,
        "open an object's door": False,
        "take out an object": False,
        "place an object": False,
        "hold an object": False,
        "move an object": False,
        "grip an object": False
    },
    "cmu_play_fusion": {
        "pick up an object": False,
        "open an object": False,
        "place an object": False,
        "close an object": False,
        "turn on an object": False,
        "push an object": False,
    },
    "cmu_stretch": {
        "open an object": False,
        "slide an object": False,
        "pull an object": False,
        "lift an object": False,
        "garbage an object": False,
    },
    "recon": {
        "explore environment": False
    },
    "coryhall": {
        "navigate hallways": False
    },
    "sacson": {
        "navigate environments": False
    },
    "conqhose": {
        "grab an object": False,
        "lift an object": False,
        "drag an object": False
    },
    "dobbe": {
        "pick up an object": False,
        "open an object": False,
        "close an object": False,
        "place an object": False,
        "pull an object": False,
        "flush toilet": False,
        "straighten cushion": False,
        "pour an object": False,
        "unplug an object": False,
        "rotate an object": False,
        "adjust an object": False,
        "push an object": False,
        "put an object": False
    },
    "io_ai_office_picknplace": {
        "pick up an object": False,
        "place an object": False
    },
    "roboset": {
        "wipe an object": False,
        "pick up an object": False,
        "place an object": False,
        "cap an object": False,
        "uncap an object": False
    },
    "spoc": {
        "fetch an object": False,
        "navigate to an object": False,
        "search for an object": False,
        "pick up an object": False,
        "find an object": False,
        "place an object": False
    },
    "plex_robotsuite": {
        "open an object": False,
        "close an object": False,
        "pick up an object": False,
        "place an object": False,
        "put an object": False,
        "insert an object": False,
        "lift an object": False,
        "assemble an object": False,
        "stack an object": False
    }
}

ADDITIONAL_INSTRUCTIONS = {
    "berkeley_autolab_ur5": {
        "take the tiger out of the red bowl and put it in the grey bowl": [
            "The continuous observation represents [joint0, joint1, joint2, joint3, joint4, joint5, x, y, z, qx, qy, qz, qw, gripper_is_closed, action_blocked].",
            "action_blocked is binary: 1 if the gripper opening/closing action is being executed and no other actions can be performed; 0 otherwise."
        ],
        "sweep the green cloth to the left side of the table": [
            "The continuous observation represents [joint0, joint1, joint2, joint3, joint4, joint5, x, y, z, qx, qy, qz, qw, gripper_is_closed, action_blocked].",
            "action_blocked is binary: 1 if the gripper opening/closing action is being executed and no other actions can be performed; 0 otherwise."
        ],
        "pick up the blue cup and put it into the brown cup": [
            "The continuous observation represents [joint0, joint1, joint2, joint3, joint4, joint5, x, y, z, qx, qy, qz, qw, gripper_is_closed, action_blocked].",
            "action_blocked is binary: 1 if the gripper opening/closing action is being executed and no other actions can be performed; 0 otherwise."
        ],
        "put the ranch bottle into the pot": [
            "The continuous observation represents [joint0, joint1, joint2, joint3, joint4, joint5, x, y, z, qx, qy, qz, qw, gripper_is_closed, action_blocked].",
            "action_blocked is binary: 1 if the gripper opening/closing action is being executed and no other actions can be performed; 0 otherwise."
        ]
    }
}
