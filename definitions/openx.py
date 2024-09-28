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
    "fractal20220817_data": {
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
    "kuka": {
        "grasp and pick an object": [
            "Choose a grasp point, and then execute the desired grasp strategy.",
            "Update the grasp strategy continuously based on the most recent observations."
        ]
    },
    "bridge": {
        "put corn in pot": [
            "Put corn in a pot."
        ],
        "put carrot on plate": [
            "Put carrot on a plate."
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
        "open door": [
            "Open door."
        ],
        "Open drawer": [
            "Open drawer."
        ],
        "close door": [
            "Close door."
        ],
        "close drawer": [
            "Close drawer."
        ],
        "stack blocks": [
            "Stack green block on yellow block."
        ],
        "fold thin blue cloth over object": [
            "Fold thin blue cloth over object."
        ],
        "fold thick gray cloth over object": [
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
    "taco_play": {
        "rotate block left": [
            "Rotate block left."
        ],
        "rotate block right": [
            "Rotate block right.", 
        ],
        "push block left": [
            "Push block left."
        ],
        "push block right": [
            "Push block right."
        ],
        "lift the block on top of the drawer": [
            "Lift the block on top of the drawer."
        ],
        "lift the block inside the drawer": [
            "Lift the block inside the drawer."
        ],
        "lift the block from the slider": [
            "Lift the block from the slider."
        ],
        "lift the block from the container": [
            "Lift the block from the container."
        ],
        "lift the block from the table": [
            "Lift the block from the table."
        ],
        "place the block on top of the drawer": [
            "Place the block on top of the drawer."
        ],
        "place the block inside the drawer": [
            "Place the block inside the drawer."
        ],
        "place the block in the slider": [
            "Place the block in the slider."
        ],
        "place the block in the container": [
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
        "move slider left": [
            "Move slider left."
        ],
        "move slider right": [
            "Move slider right."
        ],
        "turn red light on": [
            "Turn red light on by pushing the button."
        ],
        "turn red light off": [
            "Turn red light off pushing the button."
        ],
        "turn green light on": [
            "Turn green light on pushing the button."
        ],
        "turn green light off": [
            "Turn green light off pushing the button."
        ],
        "turn blue light on": [
            "Turn blue light on pushing the button."
        ],
        "turn blue light off": [
            "Turn blue light off pushing the button."
        ]
    },
    "jaco_play": {
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
    "nyu_door_opening_surprising_effectiveness": {
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
    "viola": {
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
    "toto": {
        "scoop": [
            "Take a scoop of mixed nuts from the gray ceramic bowl."
        ],
        "pour": [
            "Pour the nuts into the pink plastic cup."
        ]
    },
    "language_table": {
        "push the yellow hexagon to the top right corner": [
            "Push the yellow hexagon to the top right corner."
        ],
        "push the red circle to the bottom right corner": [
            "Push the red circle to the bottom right corner."
        ],
        "push the green star to the bottom left corner": [
            "Push the green star to the bottom left corner."
        ],
        "move the yellow heart to the yellow hexagon": [
            "Move the yellow heart to the yellow hexagon."
        ],
        "move the red star to the red circle": [
            "Move the red star to the red circle."
        ],
        "nudge the green star down and left a bit": [
            "Nudge the green star down and left a bit."
        ],
        "nudge the green circle closer to the green star": [
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
    "columbia_cairlab_pusht_real": {
        "push t-shaped blocks": [
            "Push T-shaped block into a fixed goal pose.",
            "Move to a fixed exit zone."
        ]
    },
    "nyu_rot_dataset_converted_externally_to_rlds": {
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
    "stanford_hydra_dataset_converted_externally_to_rlds": {
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
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
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
    "maniskill_dataset_converted_externally_to_rlds": {
        "pick an object": [
            "Pick an isolated object.",
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
    "furniture_bench_dataset_converted_externally_to_rlds": {
        "grasp a lamp": [
            "Grasp a lamp."
        ],
        "grasp a square table": [
            "Grasp a square table."
        ],
        "grasp a drawer": [
            "Grasp a drawer."
        ],
        "grasp a cabinet": [
            "Grasp a cabinet."
        ],
        "grasp a round table": [
            "Grasp a round table."
        ],
        "grasp a desk": [
            "Grasp a desk."
        ],
        "grasp a stool": [
            "Grasp a stool."
        ],
        "grasp a chair": [
            "Grasp a chair."
        ],
        "place a lamp": [
            "Place a lamp."
        ],
        "place a square table": [
            "Place a square table."
        ],
        "place a drawer": [
            "Place a drawer."
        ],
        "place a cabinet": [
            "Place a cabinet."
        ],
        "place a round table": [
            "Place a round table."
        ],
        "place a desk": [
            "Place a desk."
        ],
        "place a stool": [
            "Place a stool."
        ],
        "place a chair": [
            "Place a chair."
        ],
        "insert a lamp": [
            "Insert a lamp."
        ],
        "insert a square table": [
            "Insert a square table."
        ],
        "insert a drawer": [
            "Insert a drawer."
        ],
        "insert a cabinet": [
            "Insert a cabinet."
        ],
        "insert a round table": [
            "Insert a round table."
        ],
        "insert a desk": [
            "Insert a desk."
        ],
        "insert a stool": [
            "Insert a stool."
        ],
        "insert a chair": [
            "Insert a chair."
        ],
        "screw a lamp": [
            "Screw a lamp."
        ],
        "screw a square table": [
            "Screw a square table."
        ],
        "screw a drawer": [
            "Screw a drawer."
        ],
        "screw a cabinet": [
            "Screw a cabinet."
        ],
        "screw a round table": [
            "Screw a round table."
        ],
        "screw a desk": [
            "Screw a desk."
        ],
        "screw a stool": [
            "Screw a stool."
        ],
        "screw a chair": [
            "Screw a chair."
        ]
    },
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": {
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
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {  # TODO: check this
    },
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
        "reach": [
            "Reach for an object."
        ],
        "pick": [
            "Pick an object."
        ]
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
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
        "place bottle in ceramic bowl": [
            "Place bottle in ceramic bowl"
        ],
        "place white sponge in purple bowl": [
            "Place white sponge in purple bowl."
        ],
        "place grapes in red bowl": [
            "Place grapes in red bowl."
        ],
        "place banana in ceramic cup.": [
            "Place banana in ceramic cup."
        ],
        "push an object": [
            "Push a purple bowl across the table.",
        ],
        "wipe tray with sponge": [
            "Wipe tray with sponge."
        ],
        "wipe table surface with banana.": [
            "Wipe table surface with banana.",
        ],
        "wipe a surface with brush": [
            "Wipe a surface with brush."
        ],
        "stack bowls into tray": [
            "Stack bowls into tray.",
        ],
        "knock the paper cup over": [
            "Knock the paper cup over."
        ],
        "drag grapes across the table": [
            "Drag grapes across the table."
        ],
        "open": [
            "Open a door."
        ],
        "empty bin": [
            "Empty a bin."
        ],
        "pick up grapes": [
            "Pick up grapes.",
        ]
    },
    "usc_cloth_sim_converted_externally_to_rlds": {
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
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {
        "open fridge": [
            "Open the fridge."
        ]
    },
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {
        "pick up an object": [
            "Pick up an object."
        ],
        "place an object": [
            "Place an object."
        ],
        "fold cloths": [
            "Fold cloths."
        ]
    },
    "utokyo_saytap_converted_externally_to_rlds": {
        "trot forward slowly": [
            "Trot forward slowly."
        ],
        "trot forward fast": [
            "Trot forward fast."
        ],
        "lift front right leg": [
            "Lift front right leg.",
        ],
        "lift front left leg": [
            "Lift front left leg."
        ],
        "lift rear right leg": [
            "Lift rear right leg."
        ],
        "lift rear left leg": [
            "Lift rear left leg."
        ],
        "pace forward fast": [
            "Pace forward fast."
        ],
        "pace forward slowly": [
            "Pace forward slowly."
        ],
        "pace backward fast": [
            "Pace backward fast."
        ],
        "pace backward slowly": [
            "Pace backward slowly."
        ],
        "back off": [
            "Back off! Don't hurt that squirrel."
        ],
        "act as if the ground is very hot": [
            "Act as if the ground is very hot."
        ],
        "act as if you have a limping front right leg": [
            "Act as if you have a limping front right leg."
        ],
        "act as if you have a limping front left leg": [
            "Act as if you have a limping front left leg."
        ],
        "act as if you have a limping rear right leg": [
            "Act as if you have a limping rear right leg."
        ],
        "act as if you have a limping rear left leg": [
            "Act as if you have a limping rear left leg."
        ],
        "go": [
            "Go catch that squirrel on the tree."
        ]
    },
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "pick up an object": [
            "Pick up an object."
        ],
        "place an object": [
            "Place an object."
        ]
    },
    "utokyo_xarm_bimanual_converted_externally_to_rlds": {
        "reach for an object": [
            "Reach for an object."
        ],
        "unfold an object": [
            "Unfold an object.",
            "Unfold the wrinkled towel."
        ]
    },
    "berkeley_mvp_converted_externally_to_rlds": {
        "reach for an object": [
            "Reach for an object."
        ],
        "push an object": [
            "Push or move an object."
        ],
        "pick up an object": [
            "Pick up an object."
        ]
    },
    "berkeley_rpt_converted_externally_to_rlds": {
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
    "kaist_nonprehensile_converted_externally_to_rlds": {
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
    "stanford_mask_vit_converted_externally_to_rlds": {
        "pick up an object": [
            "Pick up an object."
        ],
        "push an object": [
            "Push an object."
        ]
    },
    "tokyo_u_lsmo_converted_externally_to_rlds": {
        "avoid an obstacle": [
            "Avoid the obstacle on the table."
        ],
        "reach for an object": [
            "Reach for an object."
        ]
    },
    "dlr_sara_pour_converted_externally_to_rlds": {
        "move towards an object": [
            "Move towards an object."
        ],
        "pour into a cup": [
            "Pour ping-pong balls from a cup held in the end-effector into the cup placed on the table."
        ]
    },
    "dlr_sara_grid_clamp_converted_externally_to_rlds": {
        "place an object": [
            "Place the grid clamp in the grids on the table, similar to placing a peg in the hole."
        ]
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
        "grasp an object on the tabletop": [
            "Grasp an object on the tabletop."
        ],
        "grasp an object on the shelf": [
            "Grasp an object on the shelf."
        ]
    },
    "asu_table_top_converted_externally_to_rlds": {
        "pick an object": [
            "Pick an object."
        ],
        "push an object": [
            "Push an object.",
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
    "stanford_robocook_converted_externally_to_rlds": {
        "pinch the dough with an asymmetric gripper": [
            "Pinch the dough with an asymmetric gripper."
        ],
        "pinch the dough with a two-plane symmetric gripper": [
            "Pinch the dough with a two-plane symmetric gripper."
        ],
        "pinch the dough with a two-rod symmetric gripper": [
            "Pinch the dough with a two-rod symmetric gripper."
        ],
        "press the dough with a circle press": [
            "Press the dough with a circle press."
        ],
        "press the dough with a square press": [
            "Press the dough with a square press."
        ],
        "press the dough with a circle punch": [
            "Press the dough with a circle punch."
        ],
        "press the dough with a square punch": [
            "Press the dough with a square punch."
        ],
        "roll the dough with a small roller": [
            "Roll the dough with a small roller."
        ],
        "roll the dough with a large roller": [
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
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
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
    "utaustin_mutex": {
        "put an object": [
            "Put an object.",
        ],
        "open an object's door": [
            "Open an object's door.",
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
            "Pick up an object."
        ],
        "open a door": [
            "Open a door."
        ],
        "open a drawer": [
            "Open a drawer."
        ],
        "place an object": [
            "Place an object.",
        ],
        "close a door": [
            "Close a door.",
        ],
        "close a drawer": [
            "Close a drawer."
        ],
        "turn on lights": [
            "Turn on the LED lights."
        ],
        "push an object": [
            "Push an object."
        ]
    },
    "cmu_stretch": {
        "open a door": [
            "Open a door."
        ],
        "open a dishwasher": [
            "Open a dishwasher."
        ],
        "open a cabinet": [
            "Open a cabinet."
        ],
        "open a drawer": [
            "Open a drawer."
        ],
        "slide a door": [
            "Slide a door."
        ],
        "pull out a drawer": [
            "Pull out a drawer."
        ],
        "lift a lid": [
            "Lift a lid."
        ],
        "lift a knife": [
            "Lift a knife."
        ],
        "garbage a can": [
            "Garbage a can."
        ]
    },
    "berkeley_gnm_recon": {
        "explore environment": [
            "Ignore distractors, and explore a non-stationary environment, successfully discovering and navigating to the visually-specified goal."
        ]
    },
    "berkeley_gnm_cory_hall": {
        "navigate hallways": [
            "Autonomously navigate complex and unstructured environments such as roads, buildings, or forests."
        ]
    },
    "berkeley_gnm_sac_son": {
        "navigate environments": [
            "Navigate pedestrian-rich indoor and outdoor environments such as offices, school buildings."
        ]
    },
    "conq_hose_manipulation": {
        "grab the end of the vacuum hose around in an office environment": [
            "Grab the end of the vacuum hose around in an office environment."
        ],
        "lift the end of the vacuum hose around in an office environment": [
            "Lift the end of the vacuum hose around in an office environment."
        ],
        "drag the end of the vacuum hose around in an office environment": [
            "Drag the end of the vacuum hose around in an office environment."
        ]
    },
    "dobbe": {
        "pick up paper towel roll": [
            "Pick up paper towel roll."
        ],
        "pick up paper bag": [
            "Pick up paper bag."
        ],
        "pick up hat": [
            "Pick up hat."
        ],
        "pick up trash bag": [
            "Pick up trash bag."
        ],
        "pick up hand towel": [
            "Pick up hand towel."
        ],
        "pick up kitchen towel": [
            "Pick up kitchen towel."
        ],
        "pick up tissue roll": [
            "Pick up tissue roll."
        ],
        "open a door": [
            "Open a door."
        ],
        "open cabinet door": [
            "Open cabinet door."
        ],
        "open shower curtain": [
            "Open shower curtain."
        ],
        "open dishwasher door": [
            "Open dishwasher door."
        ],
        "open air fryer door": [
            "Open air fryer door."
        ],
        "open freezer door": [
            "Open freezer door."
        ],
        "open vertical window blinds": [
            "Open vertical window blinds."
        ],
        "close a door": [
            "Close a door."
        ],
        "close cabinet door": [
            "Close cabinet door."
        ],
        "close shower curtain": [
            "Close shower curtain."
        ],
        "close dishwasher door": [
            "Close dishwasher door."
        ],
        "close air fryer door": [
            "Close air fryer door."
        ],
        "place keychain": [
            "Place keychain."
        ],
        "place spice": [
            "Place spice."
        ],
        "place massager": [
            "Place massager."
        ],
        "pull out dining chair": [
            "Pull out dining chair."
        ],
        "pull book from shelf": [
            "Pull book from shelf."
        ],
        "pull chair": [
            "Pull chair."
        ],
        "pull desk chair": [
            "Pull desk chair."
        ],
        "pull side table": [
            "Pull side table."
        ],
        "pull out dining stool": [
            "Pull out dining stool."
        ],
        "flush toilet": [
            "Flush toilet."
        ],
        "straighten cushion": [
            "Straighten cushion."
        ],
        "pour chocolate almond": [
            "Pour chocolate almond."
        ],
        "unplug charger": [
            "Unplug charger."
        ],
        "rotate speaker knob": [
            "Rotate speaker knob."
        ],
        "adjust oven knob": [
            "Adjust oven knob."
        ],
        "push toaster button": [
            "Push toaster button."
        ],
        "put rag in laundry": [
            "Put rag in laundry."
        ]
    },
    "io_ai_tech": {
        "pick up the glue from the plate": [
            "Pick up the glue from the plate."
        ],
        "pick up the stapler": [
            "Pick up the stapler."
        ],
        "place the glue on the plate": [
            "Place the glue on the plate."
        ],
        "place the stapler on the desk": [
            "Place the stapler on the desk."
        ]
    },
    "robo_set": {
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
        ],
        "pick up an object": [
            "Pick up a specified object in agent line of sight.",
        ],
        "find an object": [
            "Find an object.",
        ],
        "place an object": [
            "Place an object."
        ]
    },
    "plex_robosuite": {
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
    "fractal20220817_data": {
        "pick and place items": {
            0: ("Gripper of the robot closed or open", {1.0: "Gripper closed", 0.0: "Gripper open"}),
            1: ("X axis displacement for the robot's arm movement"),
            2: ("Y axis displacement for the robot's arm movement"),
            3: ("Z axis displacement for the robot's arm movement"),
            4: ("Roll displacement for the robot's arm movement"),
            5: ("Pitch displacement for the robot's arm movement"),
            6: ("Yaw displacement for the robot's arm movement"),
            7: ("Yaw displacement for the robot's base movement"),
            8: ("X axis displacement for the robot's base movement"),
            9: ("Y axis displacement for the robot's base movement")
        },
        "move object near another object": {
            0: ("Gripper of the robot closed or open", {1.0: "Gripper closed", 0.0: "Gripper open"}),
            1: ("X axis displacement for the robot's arm movement"),
            2: ("Y axis displacement for the robot's arm movement"),
            3: ("Z axis displacement for the robot's arm movement"),
            4: ("Roll displacement for the robot's arm movement"),
            5: ("Pitch displacement for the robot's arm movement"),
            6: ("Yaw displacement for the robot's arm movement"),
            7: ("Yaw displacement for the robot's base movement"),
            8: ("X axis displacement for the robot's base movement"),
            9: ("Y axis displacement for the robot's base movement")
        },
        "place objects up-right": {
            0: ("Gripper of the robot closed or open", {1.0: "Gripper closed", 0.0: "Gripper open"}),
            1: ("X axis displacement for the robot's arm movement"),
            2: ("Y axis displacement for the robot's arm movement"),
            3: ("Z axis displacement for the robot's arm movement"),
            4: ("Roll displacement for the robot's arm movement"),
            5: ("Pitch displacement for the robot's arm movement"),
            6: ("Yaw displacement for the robot's arm movement"),
            7: ("Yaw displacement for the robot's base movement"),
            8: ("X axis displacement for the robot's base movement"),
            9: ("Y axis displacement for the robot's base movement")
        },
        "open a drawer": {
            0: ("Gripper of the robot closed or open", {1.0: "Gripper closed", 0.0: "Gripper open"}),
            1: ("X axis displacement for the robot's arm movement"),
            2: ("Y axis displacement for the robot's arm movement"),
            3: ("Z axis displacement for the robot's arm movement"),
            4: ("Roll displacement for the robot's arm movement"),
            5: ("Pitch displacement for the robot's arm movement"),
            6: ("Yaw displacement for the robot's arm movement"),
            7: ("Yaw displacement for the robot's base movement"),
            8: ("X axis displacement for the robot's base movement"),
            9: ("Y axis displacement for the robot's base movement")
        },
        "close a drawer": {
            0: ("Gripper of the robot closed or open", {1.0: "Gripper closed", 0.0: "Gripper open"}),
            1: ("X axis displacement for the robot's arm movement"),
            2: ("Y axis displacement for the robot's arm movement"),
            3: ("Z axis displacement for the robot's arm movement"),
            4: ("Roll displacement for the robot's arm movement"),
            5: ("Pitch displacement for the robot's arm movement"),
            6: ("Yaw displacement for the robot's arm movement"),
            7: ("Yaw displacement for the robot's base movement"),
            8: ("X axis displacement for the robot's base movement"),
            9: ("Y axis displacement for the robot's base movement")
        },
        "place object into receptacle": {
            0: ("Gripper of the robot closed or open", {1.0: "Gripper closed", 0.0: "Gripper open"}),
            1: ("X axis displacement for the robot's arm movement"),
            2: ("Y axis displacement for the robot's arm movement"),
            3: ("Z axis displacement for the robot's arm movement"),
            4: ("Roll displacement for the robot's arm movement"),
            5: ("Pitch displacement for the robot's arm movement"),
            6: ("Yaw displacement for the robot's arm movement"),
            7: ("Yaw displacement for the robot's base movement"),
            8: ("X axis displacement for the robot's base movement"),
            9: ("Y axis displacement for the robot's base movement")
        },
        "pick object into receptacle and place on the counter": {
            0: ("Gripper of the robot closed or open", {1.0: "Gripper closed", 0.0: "Gripper open"}),
            1: ("X axis displacement for the robot's arm movement"),
            2: ("Y axis displacement for the robot's arm movement"),
            3: ("Z axis displacement for the robot's arm movement"),
            4: ("Roll displacement for the robot's arm movement"),
            5: ("Pitch displacement for the robot's arm movement"),
            6: ("Yaw displacement for the robot's arm movement"),
            7: ("Yaw displacement for the robot's base movement"),
            8: ("X axis displacement for the robot's base movement"),
            9: ("Y axis displacement for the robot's base movement")
        }
    },
    "kuka": {
        "grasp and pick an object": {
           0: ("Gripper closed or open", {1.0: "Gripper closed", 0.0: "Gripper open"}),
           1: ("X axis displacement of the robot gripper pose in meters", -1, 1),
           2: ("Y axis displacement of the robot gripper pose in meters", -1, 1),
           3: ("Z axis displacement of the robot gripper pose in meters", -1, 1),
           4: ("The delta change in roll of the robot gripper pose in radians", 0, 0.6366),
           5: ("The delta change in pitch of the robot gripper pose in radians", 0, 0.6366),
           6: ("The delta change in yaw of the robot gripper pose in radians", 0, 0.6366),
           7: ("The displacement in vertical rotation of the robot base in radians", 0, 0.6366),
           8: ("X axis displacement of the robot base in meters", -1, 1),
           9: ("Y axis displacement of the robot base in meters", -1, 1)
        }
    },
    "bridge": {
        "put corn in pot": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("The delta change in the roll for the robot"),
            4: ("The delta change in the pitch for the robot"),
            5: ("The delta change in the yaw for the robot"),
            6: ("Termination", {1.0: "The robot has reached the target location", 0.0: "The robot has not reached the target location"}),
            7: ("Open or close the gripper", {1.0: "Gripper open", 0.0: "Gripper closed"})
        },
        "put carrot on plate": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("The delta change in the roll for the robot"),
            4: ("The delta change in the pitch for the robot"),
            5: ("The delta change in the yaw for the robot"),
            6: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            7: ("Open or close the gripper", {1: "Gripper open", 0: "Gripper closed"})
        },
        "push": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("The delta change in the roll for the robot"),
            4: ("The delta change in the pitch for the robot"),
            5: ("The delta change in the yaw for the robot"),
            6: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            7: ("Open or close the gripper", {1: "Gripper open", 0: "Gripper closed"})
        },
        "reorient objects": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("The delta change in the roll for the robot"),
            4: ("The delta change in the pitch for the robot"),
            5: ("The delta change in the yaw for the robot"),
            6: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            7: ("Open or close the gripper", {1: "Gripper open", 0: "Gripper closed"})
        },
        "sweep": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("The delta change in the roll for the robot"),
            4: ("The delta change in the pitch for the robot"),
            5: ("The delta change in the yaw for the robot"),
            6: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            7: ("Open or close the gripper", {1: "Gripper open", 0: "Gripper closed"})
        },
        "open door": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("The delta change in the roll for the robot"),
            4: ("The delta change in the pitch for the robot"),
            5: ("The delta change in the yaw for the robot"),
            6: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            7: ("Open or close the gripper", {1: "Gripper open", 0: "Gripper closed"})
        },
        "Open drawer": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("The delta change in the roll for the robot"),
            4: ("The delta change in the pitch for the robot"),
            5: ("The delta change in the yaw for the robot"),
            6: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            7: ("Open or close the gripper", {1: "Gripper open", 0: "Gripper closed"})
        },
        "close door": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("The delta change in the roll for the robot"),
            4: ("The delta change in the pitch for the robot"),
            5: ("The delta change in the yaw for the robot"),
            6: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            7: ("Open or close the gripper", {1: "Gripper open", 0: "Gripper closed"})
        },
        "close drawer": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("The delta change in the roll for the robot"),
            4: ("The delta change in the pitch for the robot"),
            5: ("The delta change in the yaw for the robot"),
            6: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            7: ("Open or close the gripper", {1: "Gripper open", 0: "Gripper closed"})
        },
        "stack blocks": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("The delta change in the roll for the robot"),
            4: ("The delta change in the pitch for the robot"),
            5: ("The delta change in the yaw for the robot"),
            6: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            7: ("Open or close the gripper", {1: "Gripper open", 0: "Gripper closed"})
        },
        "fold thin blue cloth over object": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("The delta change in the roll for the robot"),
            4: ("The delta change in the pitch for the robot"),
            5: ("The delta change in the yaw for the robot"),
            6: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            7: ("Open or close the gripper", {1: "Gripper open", 0: "Gripper closed"})
        },
        "fold thick gray cloth over object": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("The delta change in the roll for the robot"),
            4: ("The delta change in the pitch for the robot"),
            5: ("The delta change in the yaw for the robot"),
            6: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            7: ("Open or close the gripper", {1: "Gripper open", 0: "Gripper closed"})
        },
        "wipe a surface": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("The delta change in the roll for the robot"),
            4: ("The delta change in the pitch for the robot"),
            5: ("The delta change in the yaw for the robot"),
            6: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            7: ("Open or close the gripper", {1: "Gripper open", 0: "Gripper closed"})
        },
        "twist knobs": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("The delta change in the roll for the robot"),
            4: ("The delta change in the pitch for the robot"),
            5: ("The delta change in the yaw for the robot"),
            6: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            7: ("Open or close the gripper", {1: "Gripper open", 0: "Gripper closed"})
        },
        "flip a switch": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("The delta change in the roll for the robot"),
            4: ("The delta change in the pitch for the robot"),
            5: ("The delta change in the yaw for the robot"),
            6: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            7: ("Open or close the gripper", {1: "Gripper open", 0: "Gripper closed"})
        },
        "turn faucets": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("The delta change in the roll for the robot"),
            4: ("The delta change in the pitch for the robot"),
            5: ("The delta change in the yaw for the robot"),
            6: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            7: ("Open or close the gripper", {1: "Gripper open", 0: "Gripper closed"})
        },
        "zip a zipper": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("The delta change in the roll for the robot"),
            4: ("The delta change in the pitch for the robot"),
            5: ("The delta change in the yaw for the robot"),
            6: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            7: ("Open or close the gripper", {1: "Gripper open", 0: "Gripper closed"})
        }
    },
    "taco_play": {
        "rotate block left": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "rotate block right": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "push block left": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "push block right": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "lift the block on top of the drawer": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "lift the block inside the drawer": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "lift the block from the slider": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "lift the block from the container": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "lift the block from the table": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "place the block on top of the drawer": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "place the block inside the drawer": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "place the block in the slider": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "place the block in the container": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "stack objects": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "unstack objects": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "open drawer": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "close drawer": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "move slider left": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "move slider right": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "turn red light on": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "turn red light off": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "turn green light on": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "turn green light off": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "turn blue light on": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        },
        "turn blue light off": {
            0: ("X axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            1: ("Y axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            2: ("Z axis displacement for the end effector of the robot in 3D space",-1.0, 1.0),
            3: ("X axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            4: ("Y axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            5: ("Z axis displacement for the end effector of the robot in the robot's base frame", -1.0, 1.0),
            6: ("Open or close the gripper", {-1.0: "Gripper open", 1.0: "Gripper closed"})
        }
    },
    "jaco_play": {
        "pick up an object": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("Gripper closed or open or doesn't move", {2.0: "Gripper closed", 0.0: "Gripper open", 1.0: "Gripper doesn't move"})
        },
        "put an object down": {
            0: ("X axis displacement for the robot"),
            1: ("Y axis displacement for the robot"),
            2: ("Z axis displacement for the robot"),
            3: ("Gripper closed or open or doesn't move", {2.0: "Gripper closed", 0.0: "Gripper open", 1.0: "Gripper doesn't move"})
        }
    },
    "berkeley_cable_routing": {
        "pick up an object": {
            0: ("The delta X axis rotation delta with respect to the robot's end effector frame"),
            1: ("The delta Y axis rotation delta with respect to the robot's end effector frame"),
            2: ("The delta Z axis rotation delta with respect to the robot's end effector frame"),
            3: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            4: ("The X axis displacement with respect to the robot's end effector frame"),
            5: ("The Y axis displacement with respect to the robot's end effector frame"),
            6: ("The Z axis displacement with respect to the robot's end effector frame")
        },
        "route a cable": {
            0: ("The X axis rotation delta with respect to the robot's end effector frame"),
            1: ("The Y axis rotation delta with respect to the robot's end effector frame"),
            2: ("The Z axis rotation delta with respect to the robot's end effector frame"),
            3: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            4: ("The X axis displacement with respect to the robot's end effector frame"),
            5: ("The Y axis displacement with respect to the robot's end effector frame"),
            6: ("The Z axis displacement with respect to the robot's end effector frame")
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
    "nyu_door_opening_surprising_effectiveness": {
        "push objects": {
            0: ("The X axis displacement in meters of the robot"),
            1: ("The Y axis displacement in meters of the robot"),
            2: ("The Z axis displacement in meters of the robot"),
            3: ("Width of the gripper fingertips describing how open it is", 0.0, 3.0),
            4: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            5: ("The X axis rotation delta of the robot in Euler coordinates"),
            6: ("The Y axis rotation delta of the robot in Euler coordinates"),
            7: ("The Z axis rotation delta of the robot in Euler coordinates")
        },
        "stack objects": {
            0: ("The X axis displacement in meters of the robot"),
            1: ("The Y axis displacement in meters of the robot"),
            2: ("The Z axis displacement in meters of the robot"),
            3: ("Width of the gripper fingertips describing how open it is", 0.0, 3.0),
            4: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            5: ("The X axis rotation delta of the robot in Euler coordinates"),
            6: ("The Y axis rotation delta of the robot in Euler coordinates"),
            7: ("The Z axis rotation delta of the robot in Euler coordinates")
        },
        "open door": {
            0: ("The X axis displacement in meters of the robot"),
            1: ("The Y axis displacement in meters of the robot"),
            2: ("The Z axis displacement in meters of the robot"),
            3: ("Width of the gripper fingertips describing how open it is", 0.0, 3.0),
            4: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            5: ("The X axis rotation delta of the robot in Euler coordinates"),
            6: ("The Y axis rotation delta of the robot in Euler coordinates"),
            7: ("The Z axis rotation delta of the robot in Euler coordinates")
        }
    },
    "viola": {
        "sort": {
            0 : ("Gripper closed or open", {-1.0: "Gripper open", 1.0: "Gripper closed"}),
            1: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            2: ("The X axis displacement of the robot"),
            3: ("The Y axis displacement of the robot"),
            4: ("The Z axis displacement of the robot"),
            5: ("The X axis rotation delta of the robot"),
            6: ("The Y axis rotation delta of the robot"),
            7: ("The Z axis rotation delta of the robot")
        },
        "BUDS kitchen": {
            0 : ("Gripper closed or open", {-1.0: "Gripper open", 1.0: "Gripper closed"}),
            1: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            2: ("The X axis displacement of the robot"),
            3: ("The Y axis displacement of the robot"),
            4: ("The Z axis displacement of the robot"),
            5: ("The X axis rotation delta of the robot"),
            6: ("The Y axis rotation delta of the robot"),
            7: ("The Z axis rotation delta of the robot")
        },
        "stack": {
            0 : ("Gripper closed or open", {-1.0: "Gripper open", 1.0: "Gripper closed"}),
            1: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
            2: ("The X axis displacement of the robot"),
            3: ("The Y axis displacement of the robot"),
            4: ("The Z axis displacement of the robot"),
            5: ("The X axis rotation delta of the robot"),
            6: ("The Y axis rotation delta of the robot"),
            7: ("The Z axis rotation delta of the robot")
        }
    },
    "toto": {
        "scoop": {
            0: None
        },
        "pour": {
            0: None
        }
    },
    "language_table": {
        "push the yellow hexagon to the top right corner": {
            0: ("X axis co-ordinate in the Cartesian setpoint of the end effector"),
            1: ("Y axis co-ordinate in the Cartesian setpoint of the end effector")

        },
        "push the red circle to the bottom right corner": {
            0: ("X axis co-ordinate in the Cartesian setpoint of the end effector"),
            1: ("Y axis co-ordinate in the Cartesian setpoint of the end effector")
        },
        "push the green star to the bottom left corner": {
            0: ("X axis co-ordinate in the Cartesian setpoint of the end effector"),
            1: ("Y axis co-ordinate in the Cartesian setpoint of the end effector")
        },
        "move the yellow heart to the yellow hexagon": {
            0: ("X axis co-ordinate in the Cartesian setpoint of the end effector"),
            1: ("Y axis co-ordinate in the Cartesian setpoint of the end effector")
        },
        "move the red star to the red circle": {
            0: ("X axis co-ordinate in the Cartesian setpoint of the end effector"),
            1: ("Y axis co-ordinate in the Cartesian setpoint of the end effector")
        },
        "nudge the green star down and left a bit": {
            0: ("X axis co-ordinate in the Cartesian setpoint of the end effector"),
            1: ("Y axis co-ordinate in the Cartesian setpoint of the end effector")
        },
        "nudge the green circle closer to the green star": {
            0: ("X axis co-ordinate in the Cartesian setpoint of the end effector"),
            1: ("Y axis co-ordinate in the Cartesian setpoint of the end effector")
        },
        "put objects": {
            0: ("X axis co-ordinate in the Cartesian setpoint of the end effector"),
            1: ("Y axis co-ordinate in the Cartesian setpoint of the end effector")
        },
        "touch objects": {
            0: ("X axis co-ordinate in the Cartesian setpoint of the end effector"),
            1: ("Y axis co-ordinate in the Cartesian setpoint of the end effector")
        },
        "slide objects": {
            0: ("X axis co-ordinate in the Cartesian setpoint of the end effector"),
            1: ("Y axis co-ordinate in the Cartesian setpoint of the end effector")
        }
    },
    "columbia_cairlab_pusht_real": {
        "push t-shaped blocks": {
            0: None
        }
    },
    "nyu_rot_dataset_converted_externally_to_rlds": {
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
    "stanford_hydra_dataset_converted_externally_to_rlds": {
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
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
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
    "maniskill_dataset_converted_externally_to_rlds": {
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
    "furniture_bench_dataset_converted_externally_to_rlds": {
        "grasp a lamp": {
            0: None
        },
        "grasp a square table": {
            0: None
        },
        "grasp a drawer": {
            0: None
        },
        "grasp a cabinet": {
            0: None
        },
        "grasp a round table": {
            0: None
        },
        "grasp a desk": {
            0: None
        },
        "grasp a stool": {
            0: None
        },
        "grasp a chair": {
            0: None
        },
        "place a lamp": {
            0: None
        },
        "place a square table": {
            0: None
        },
        "place a drawer": {
            0: None
        },
        "place a cabinet": {
            0: None
        },
        "place a round table": {
            0: None
        },
        "place a desk": {
            0: None
        },
        "place a stool": {
            0: None
        },
        "place a chair": {
            0: None
        },
        "insert a lamp": {
            0: None
        },
        "insert a square table": {
            0: None
        },
        "insert a drawer": {
            0: None
        },
        "insert a cabinet": {
            0: None
        },
        "insert a round table": {
            0: None
        },
        "insert a desk": {
            0: None
        },
        "insert a stool": {
            0: None
        },
        "insert a chair": {
            0: None
        },
        "screw a lamp": {
            0: None
        },
        "screw a square table": {
            0: None
        },
        "screw a drawer": {
            0: None
        },
        "screw a cabinet": {
            0: None
        },
        "screw a round table": {
            0: None
        },
        "screw a desk": {
            0: None
        },
        "screw a stool": {
            0: None
        },
        "screw a chair": {
            0: None
        }
    },
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": {
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
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
    },
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
        "reach": {
            0: None
        },
        "pick": {
            0: None
        },
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
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
        }
    },
    "bc_z": {
        "place bottle in ceramic bowl": {
            0: None
        },
        "place white sponge in purple bowl": {
            0: None
        },
        "place grapes in red bowl": {
            0: None
        },
        "place banana in ceramic cup.": {
            0: None
        },
        "push an object": {
            0: None
        },
        "wipe tray with sponge": {
            0: None
        },
        "wipe table surface with banana.": {
            0: None
        },
        "wipe a surface with brush": {
            0: None
        },
        "stack bowls into tray": {
            0: None
        },
        "knock the paper cup over": {
            0: None
        },
        "drag grapes across the table": {
            0: None
        },
        "open": {
            0: None
        },
        "empty bin": {
            0: None
        },
        "pick up grapes": {
            0: None
        }
    },
    "usc_cloth_sim_converted_externally_to_rlds": {
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
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {
        "open fridge": {
            0: None
        }
    },
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {
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
    "utokyo_saytap_converted_externally_to_rlds": {
        "trot forward slowly": {
            0: None
        },
        "trot forward fast": {
            0: None
        },
        "lift front right leg": {
            0: None
        },
        "lift front left leg": {
            0: None
        },
        "lift rear right leg": {
            0: None
        },
        "lift rear left leg": {
            0: None
        },
        "pace forward fast": {
            0: None
        },
        "pace forward slowly": {
            0: None
        },
        "pace backward fast": {
            0: None
        },
        "pace backward slowly": {
            0: None
        },
        "back off": {
            0: None
        },
        "act as if the ground is very hot": {
            0: None
        },
        "act as if you have a limping front right leg": {
            0: None
        },
        "act as if you have a limping front left leg": {
            0: None
        },
        "act as if you have a limping rear right leg": {
            0: None
        },
        "act as if you have a limping rear left leg": {
            0: None
        },
        "go": {
            0: None
        }
    },
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "pick up an object": {
            0: None
        },
        "place an object": {
            0: None
        },
    },
    "utokyo_xarm_bimanual_converted_externally_to_rlds": {
        "reach for an object": {
            0: None
        },
        "unfold an object": {
            0: None
        },
    },
    "berkeley_mvp_converted_externally_to_rlds": {
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
    "berkeley_rpt_converted_externally_to_rlds": {
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
    "kaist_nonprehensile_converted_externally_to_rlds": {
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
    "stanford_mask_vit_converted_externally_to_rlds": {
        "pick up an object": {
            0: None
        },
        "push an object": {
            0: None
        }
    },
    "tokyo_u_lsmo_converted_externally_to_rlds": {
        "avoid an obstacle": {
            0: None
        },
        "reach for an object": {
            0: None
        }
    },
    "dlr_sara_pour_converted_externally_to_rlds": {
        "move towards an object": {
            0: None
        },
        "pour into a cup": {
            0: None
        }
    },
    "dlr_sara_grid_clamp_converted_externally_to_rlds": {
        "place an object": {
            0: None
        }
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
        "grasp an object on the tabletop": {
            0: None
        },
        "grasp an object on the shelf": {
            0: None
        }
    },
    "asu_table_top_converted_externally_to_rlds": {
        "pick an object": {
            0: None
        },
        "push an object": {
            0: None
        },
        "rotate an object": {
            0: None
        },
        "avoid an obstacle": {
            0: None
        },
        "place an object": {
            0: None
        }
    },
    "stanford_robocook_converted_externally_to_rlds": {
        "pinch the dough with an asymmetric gripper": {
            0: None
        },
        "pinch the dough with a two-plane symmetric gripper": {
            0: None
        },
        "pinch the dough with a two-rod symmetric gripper": {
            0: None
        },
        "press the dough with a circle press": {
            0: None
        },
        "press the dough with a square press": {
            0: None
        },
        "press the dough with a circle punch": {
            0: None
        },
        "press the dough with a square punch": {
            0: None
        },
        "roll the dough with a small roller": {
            0: None
        },
        "roll the dough with a large roller": {
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
    "imperialcollege_sawyer_wrist_cam": {
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
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
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
    "utaustin_mutex": {
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
        "open a door": {
            0: None
        },
        "open a drawer": {
            0: None
        },
        "place an object": {
            0: None
        },
        "close a door": {
            0: None
        },
        "close a drawer": {
            0: None
        },
        "turn on lights": {
            0: None
        },
        "push an object": {
            0: None
        }
    },
    "cmu_stretch": {
        "open a door": {
            0: None
        },
        "open a dishwasher": {
            0: None
        },
        "open a cabinet": {
            0: None
        },
        "open a drawer": {
            0: None
        },
        "slide a door": {
            0: None
        },
        "pull out a drawer": {
            0: None
        },
        "lift a lid": {
            0: None
        },
        "lift a knife": {
            0: None
        },
        "garbage a can": {
            0: None
        }
    },
    "berkeley_gnm_recon": {
        "explore environment": {
            0: None
        }
    },
    "berkeley_gnm_cory_hall": {
        "navigate hallways": {
            0: None
        }
    },
    "berkeley_gnm_sac_son": {
        "navigate environments": {
            0: None
        }
    },
    "conq_hose_manipulation": {
        "grab the end of the vacuum hose around in an office environment": {
            0: None
        },
        "lift the end of the vacuum hose around in an office environment": {
            0: None
        },
        "drag the end of the vacuum hose around in an office environment": {
            0: None
        }
    },
    "dobbe": {
        "pick up paper towel roll": {
            0: None
        },
        "pick up paper bag": {
            0: None
        },
        "pick up hat": {
            0: None
        },
        "pick up trash bag": {
            0: None
        },
        "pick up hand towel": {
            0: None
        },
        "pick up kitchen towel": {
            0: None
        },
        "pick up tissue roll": {
            0: None
        },
        "open a door": {
            0: None
        },
        "open cabinet door": {
            0: None
        },
        "open shower curtain": {
            0: None
        },
        "open dishwasher door": {
            0: None
        },
        "open air fryer door": {
            0: None
        },
        "open freezer door": {
            0: None
        },
        "open vertical window blinds": {
            0: None
        },
        "close a door": {
            0: None
        },
        "close cabinet door": {
            0: None
        },
        "close shower curtain": {
            0: None
        },
        "close dishwasher door": {
            0: None
        },
        "close air fryer door": {
            0: None
        },
        "place keychain": {
            0: None
        },
        "place spice": {
            0: None
        },
        "place massager": {
            0: None
        },
        "pull out dining chair": {
            0: None
        },
        "pull book from shelf": {
            0: None
        },
        "pull chair": {
            0: None
        },
        "pull desk chair": {
            0: None
        },
        "pull side table": {
            0: None
        },
        "pull out dining stool": {
            0: None
        },
        "flush toilet": {
            0: None
        },
        "straighten cushion": {
            0: None
        },
        "pour chocolate almond": {
            0: None
        },
        "unplug charger": {
            0: None
        },
        "rotate speaker knob": {
            0: None
        },
        "adjust oven knob": {
            0: None
        },
        "push toaster button": {
            0: None
        },
        "put rag in laundry": {
            0: None
        }
    },
    "io_ai_tech": {
        "pick up the glue from the plate": {
            0: None
        },
        "pick up the stapler": {
            0: None
        },
        "place the glue on the plate": {
            0: None
        },
        "place the stapler on the desk": {
            0: None
        }
    },
    "robo_set": {
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
    "spoc": {
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
    "plex_robosuite": {
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
    "fractal20220817_data": {
        "pick and place items": False,
        "move object near another object": False,
        "place objects upright": False,
        "open a drawer": False,
        "close a drawer": False,
        "place object into receptacle": False,
        "pick object into receptacle and place on the counter": False
    },
    "kuka": {
        "grasp and pick an object": False
    },
    "bridge": {
        "put corn in pot": False,
        "put carrot on plate": False,
        "push": False,
        "reorient objects": False,
        "sweep": False,
        "open door": False,
        "Open drawer": False,
        "close door": False,
        "close drawer": False,
        "stack blocks": False,
        "fold thin blue cloth over object": False,
        "fold thick gray cloth over object": False,
        "wipe a surface": False,
        "twist knobs": False,
        "flip a switch": False,
        "turn faucets": False,
        "zip a zipper": False
    },
    "taco_play": {
        "rotate block left": False,
        "rotate block right": False,
        "push block left": False,
        "push block right": False,
        "lift the block on top of the drawer": False,
        "lift the block inside the drawer": False,
        "lift the block from the slider": False,
        "lift the block from the container": False,
        "lift the block from the table": False,
        "place the block on top of the drawer": False,
        "place the block inside the drawer": False,
        "place the block in the slider": False,
        "place the block in the container": False,
        "stack objects": False,
        "unstack objects": False,
        "open drawer": False,
        "close drawer": False,
        "move slider left": False,
        "move slider right": False,
        "turn red light on": False,
        "turn red light off": False,
        "turn green light on": False,
        "turn green light off": False,
        "turn blue light on": False,
        "turn blue light off": False
    },
    "jaco_play": {
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
    "nyu_door_opening_surprising_effectiveness": {
        "push objects": False,
        "stack objects": False,
        "open door": False
    },
    "viola": {
        "sort": False,
        "BUDS kitchen": False,
        "stack": False
    },
    "toto": {
        "scoop": False,
        "pour": False
    },
    "language_table": {
        "push the yellow hexagon to the top right corner": False,
        "push the red circle to the bottom right corner": False,
        "push the green star to the bottom left corner": False,
        "move the yellow heart to the yellow hexagon": False,
        "move the red star to the red circle": False,
        "nudge the green star down and left a bit": False,
        "nudge the green circle closer to the green star": False,
        "put objects": False,
        "touch objects": False,
        "slide objects": False
    },
    "columbia_cairlab_pusht_real": {
        "push t-shaped blocks": False
    },
    "nyu_rot_dataset_converted_externally_to_rlds": {
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
    "stanford_hydra_dataset_converted_externally_to_rlds": {
        "assemble square": False,
        "hang tool": False,
        "insert peg": False,
        "make coffee": False,
        "make toast": False,
        "sort dishes": False
    },
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "open microwave": False,
        "open oven door": False,
        "put the pot in the sink": False,
        "operate the stove knobs": False,
    },
    "maniskill_dataset_converted_externally_to_rlds": {
        "pick an object": False,
        "move object to another position": False,
        "stack cubes": False,
        "insert peg into box": False,
        "assemble kits": False,
        "plug a charger": False,
        "turn on a faucet": False
    },
    "furniture_bench_dataset_converted_externally_to_rlds": {
        "grasp a lamp": False,
        "grasp a square table": False,
        "grasp a drawer": False,
        "grasp a cabinet": False,
        "grasp a round table": False,
        "grasp a desk": False,
        "grasp a stool": False,
        "grasp a chair": False,
        "place a lamp": False,
        "place a square table": False,
        "place a drawer": False,
        "place a cabinet": False,
        "place a round table": False,
        "place a desk": False,
        "place a stool": False,
        "place a chair": False,
        "insert a lamp": False,
        "insert a square table": False,
        "insert a drawer": False,
        "insert a cabinet": False,
        "insert a round table": False,
        "insert a desk": False,
        "insert a stool": False,
        "insert a chair": False,
        "screw a lamp": False,
        "screw a square table": False,
        "screw a drawer": False,
        "screw a cabinet": False,
        "screw a round table": False,
        "screw a desk": False,
        "screw a stool": False,
        "screw a chair": False
    },
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": {
        "pick veggies": False,
        "lift knife": False,
        "open cabinet": False,
        "pull drawer": False,
        "open dishwasher": False,
        "garbage can": False,
    },
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
    },
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
        "reach": False,
        "pick": False,
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
        "assemble nut": False,
        "hang tool": False,
        "insert gear": False,
        "pack coffee pod": False
    },
    "bc_z": {
        "place bottle in ceramic bowl": False,
        "place white sponge in purple bowl": False,
        "place grapes in red bowl": False,
        "place banana in ceramic cup.": False,
        "push an object": False,
        "wipe tray with sponge": False,
        "wipe table surface with banana.": False,
        "wipe a surface with brush": False,
        "stack bowls into tray": False,
        "knock the paper cup over": False,
        "drag grapes across the table": False,
        "open": False,
        "empty bin": False,
        "pick up grapes": False
    },
    "usc_cloth_sim_converted_externally_to_rlds": {
        "straighten a rope": False,
        "fold cloth": False,
        "fold cloth diagonally pinned": False,
        "fold cloth diagonally unpinned": False
    },
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {
        "open fridge": False
    },
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {
        "pick up an object": False,
        "place an object": False,
        "fold cloths": False
    },
    "utokyo_saytap_converted_externally_to_rlds": {
        "trot forward slowly": False,
        "trot forward fast": False,
        "lift front right leg": False,
        "lift front left leg": False,
        "lift rear right leg": False,
        "lift rear left leg": False,
        "pace forward fast": False,
        "pace forward slowly": False,
        "pace backward fast": False,
        "pace backward slowly": False,
        "back off": False,
        "act as if the ground is very hot": False,
        "act as if you have a limping front right leg": False,
        "act as if you have a limping front left leg": False,
        "act as if you have a limping rear right leg": False,
        "act as if you have a limping rear left leg": False,
        "go": False
    },
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "pick up an object": False,
        "place an object": False
    },
    "utokyo_xarm_bimanual_converted_externally_to_rlds": {
        "reach for an object": False,
        "unfold an object": False
    },
    "berkeley_mvp_converted_externally_to_rlds": {
        "reach for an object": False,
        "push an object": False,
        "pick up an object": False
    },
    "berkeley_rpt_converted_externally_to_rlds": {
        "pick up an object": False,
        "stack an object": False,
        "destack an object": False
    },
    "kaist_nonprehensile_converted_externally_to_rlds": {
        "push an object": False,
        "drag an object": False,
        "rotate an object": False,
        "topple an object": False
    },
    "stanford_mask_vit_converted_externally_to_rlds": {
        "pick up an object": False,
        "push an object": False
    },
    "tokyo_u_lsmo_converted_externally_to_rlds": {
        "avoid an obstacle": False,
        "reach for an object": False
    },
    "dlr_sara_pour_converted_externally_to_rlds": {
        "move towards an object": False,
        "pour into a cup": False
    },
    "dlr_sara_grid_clamp_converted_externally_to_rlds": {
        "place an object": False
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
        "grasp an object on the tabletop": False,
        "grasp an object on the shelf": False
    },
    "asu_table_top_converted_externally_to_rlds": {
        "pick an object": False,
        "push an object": False,
        "rotate an object": False,
        "avoid an obstacle": False,
        "place an object": False
    },
    "stanford_robocook_converted_externally_to_rlds": {
        "pinch the dough with an asymmetric gripper": False,
        "pinch the dough with a two-plane symmetric gripper": False,
        "pinch the dough with a two-rod symmetric gripper": False,
        "press the dough with a circle press": False,
        "press the dough with a square press": False,
        "press the dough with a circle punch": False,
        "press the dough with a square punch": False,
        "roll the dough with a small roller": False,
        "roll the dough with a large roller": False
    },
    "eth_agent_affordances": {
        "open door": False,
        "close door": False,
        "open drawer": False,
        "close drawer": False
    },
    "imperialcollege_sawyer_wrist_cam": {
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
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
        "insert onto square peg": False,
        "pick and lift small": False,
        "sort shapes": False,
        "take usb out": False
    },
    "utaustin_mutex": {
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
        "open a door": False,
        "open a drawer": False,
        "place an object": False,
        "close a door": False,
        "close a drawer": False,
        "turn on lights": False,
        "push an object": False
    },
    "cmu_stretch": {
        "open a door": False,
        "open a dishwasher": False,
        "open a cabinet": False,
        "open a drawer": False,
        "slide a door": False,
        "pull out a drawer": False,
        "lift a lid": False,
        "lift a knife": False,
        "garbage a can": False
    },
    "berkeley_gnm_recon": {
        "explore environment": False
    },
    "berkeley_gnm_cory_hall": {
        "navigate hallways": False
    },
    "berkeley_gnm_sac_son": {
        "navigate environments": False
    },
    "conq_hose_manipulation": {
        "grab the end of the vacuum hose around in an office environment": False,
        "lift the end of the vacuum hose around in an office environment": False,
        "drag the end of the vacuum hose around in an office environment": False
    },
    "dobbe": {
        "pick up paper towel roll": False,
        "pick up paper bag": False,
        "pick up hat": False,
        "pick up trash bag": False,
        "pick up hand towel": False,
        "pick up kitchen towel": False,
        "pick up tissue roll": False,
        "open a door": False,
        "open cabinet door": False,
        "open shower curtain": False,
        "open dishwasher door": False,
        "open air fryer door": False,
        "open freezer door": False,
        "open vertical window blinds": False,
        "close a door": False,
        "close cabinet door": False,
        "close shower curtain": False,
        "close dishwasher door": False,
        "close air fryer door": False,
        "place keychain": False,
        "place spice": False,
        "place massager": False,
        "pull out dining chair": False,
        "pull book from shelf": False,
        "pull chair": False,
        "pull desk chair": False,
        "pull side table": False,
        "pull out dining stool": False,
        "flush toilet": False,
        "straighten cushion": False,
        "pour chocolate almond": False,
        "unplug charger": False,
        "rotate speaker knob": False,
        "adjust oven knob": False,
        "push toaster button": False,
        "put rag in laundry": False
    },
    "io_ai_tech": {
        "pick up the glue from the plate": False,
        "pick up the stapler": False,
        "place the glue on the plate": False,
        "place the stapler on the desk": False
    },
    "robo_set": {
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
    "plex_robosuite": {
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
            "The robot state represents [joint0, joint1, joint2, joint3, joint4, joint5, x, y, z, qx, qy, qz, qw, gripper_is_closed, action_blocked].",
            "action_blocked is binary: 1 if the gripper opening/closing action is being executed and no other actions can be performed; 0 otherwise."
        ],
        "sweep the green cloth to the left side of the table": [
            "The robot state represents [joint0, joint1, joint2, joint3, joint4, joint5, x, y, z, qx, qy, qz, qw, gripper_is_closed, action_blocked].",
            "action_blocked is binary: 1 if the gripper opening/closing action is being executed and no other actions can be performed; 0 otherwise."
        ],
        "pick up the blue cup and put it into the brown cup": [
            "The robot state represents [joint0, joint1, joint2, joint3, joint4, joint5, x, y, z, qx, qy, qz, qw, gripper_is_closed, action_blocked].",
            "action_blocked is binary: 1 if the gripper opening/closing action is being executed and no other actions can be performed; 0 otherwise."
        ],
        "put the ranch bottle into the pot": [
            "The robot state represents [joint0, joint1, joint2, joint3, joint4, joint5, x, y, z, qx, qy, qz, qw, gripper_is_closed, action_blocked].",
            "action_blocked is binary: 1 if the gripper opening/closing action is being executed and no other actions can be performed; 0 otherwise."
        ]
    }
}