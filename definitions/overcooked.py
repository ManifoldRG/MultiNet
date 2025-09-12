class OverCookedDefinitions:
    DESCRIPTIONS = {
        "overcooked_ai": {
            "cramped_corridor": ["coordinate in cramped corridor layout"],
            "asymmetric_advantages_tomato": ["coordinate in asymmetric advatages layout"],
            "counter_circuit": ["coordinate around counter circuit layout"],
            "soup_coordination": ["coordinate for soup preparation tasks"],
            "marshmallow_experiment": ["complete marshmallow experiment scenario"],
            "inverse_marshmallow_experiment": ["complete inverse marshmallow experiment scenario"],
            "marshmallow_experiment_coordination": ["coordinate marshmallow experiment tasks"],
            "you_shall_not_pass": ["navigate you shall not pass scenario"]
        }
    }

    ACTION_SPACES = {
        "overcooked_ai": {
            "default": {
                0: (
                    "Both players move NORTH",
                ),
                1: (
                    "Player 0 moves NORTH, Player 1 moves SOUTH",
                ),
                2: (
                    "Player 0 moves NORTH, Player 1 moves EAST",
                ),
                3: (
                    "Player 0 moves NORTH, Player 1 moves WEST",
                ),
                4: (
                    "Player 0 moves NORTH, Player 1 stays in place",
                ),
                5: (
                    "Player 0 moves NORTH, Player 1 interacts with environment",
                ),
                6: (
                    "Player 0 moves SOUTH, Player 1 moves NORTH",
                ),
                7: (
                    "Both players move SOUTH",
                ),
                8: (
                    "Player 0 moves SOUTH, Player 1 moves EAST",
                ),
                9: (
                    "Player 0 moves SOUTH, Player 1 moves WEST",
                ),
                10: (
                    "Player 0 moves SOUTH, Player 1 stays in place",
                ),
                11: (
                    "Player 0 moves SOUTH, Player 1 interacts with environment",
                ),
                12: (
                    "Player 0 moves EAST, Player 1 moves NORTH",
                ),
                13: (
                    "Player 0 moves EAST, Player 1 moves SOUTH",
                ),
                14: (
                    "Both players move EAST",
                ),
                15: (
                    "Player 0 moves EAST, Player 1 moves WEST",
                ),
                16: (
                    "Player 0 moves EAST, Player 1 stays in place",
                ),
                17: (
                    "Player 0 moves EAST, Player 1 interacts with environment",
                ),
                18: (
                    "Player 0 moves WEST, Player 1 moves NORTH",
                ),
                19: (
                    "Player 0 moves WEST, Player 1 moves SOUTH",
                ),
                20: (
                    "Player 0 moves WEST, Player 1 moves EAST",
                ),
                21: (
                    "Both players move WEST",
                ),
                22: (
                    "Player 0 moves WEST, Player 1 stays in place",
                ),
                23: (
                    "Player 0 moves WEST, Player 1 interacts with environment",
                ),
                24: (
                    "Player 0 stays in place, Player 1 moves NORTH",
                ),
                25: (
                    "Player 0 stays in place, Player 1 moves SOUTH",
                ),
                26: (
                    "Player 0 stays in place, Player 1 moves EAST",
                ),
                27: (
                    "Player 0 stays in place, Player 1 moves WEST",
                ),
                28: (
                    "Both players stay in place",
                ),
                29: (
                    "Player 0 stays in place, Player 1 interacts with environment",
                ),
                30: (
                    "Player 0 interacts with environment, Player 1 moves NORTH",
                ),
                31: (
                    "Player 0 interacts with environment, Player 1 moves SOUTH",
                ),
                32: (
                    "Player 0 interacts with environment, Player 1 moves EAST",
                ),
                33: (
                    "Player 0 interacts with environment, Player 1 moves WEST",
                ),
                34: (
                    "Player 0 interacts with environment, Player 1 stays in place",
                ),
                35: (
                    "Both players interact with environment",
                ),
            }
        }
    }

    ACTION_EXCLUSIVENESS = {"overcooked_ai": {"default": False}}
    
    ADDITIONAL_INSTRUCTIONS = {}

    ACTION_MEANINGS = {
        "NORTH": {
            "coordinate": "(0, -1)",
            "description": """
            Move the player one grid cell upward on the kitchen layout. 
            This action changes the player's position vertically by decreasing their y-coordinate by 1. 
            Use this to navigate toward cooking stations, ingredient spawns, or serving areas located above the current position. 
            Cannot move if there's a wall, counter, or another player blocking the path.
            """,
        },
        "SOUTH": {
            "coordinate": "(0, 1)",
            "description": """
            Move the player one grid cell downward on the kitchen layout. 
            This action changes the player's position vertically by increasing their y-coordinate by 1. 
            Use this to navigate toward cooking stations, ingredient spawns, or serving areas located below the current position. 
            Cannot move if there's a wall, counter, or another player blocking the path."
            """,
        },
        "EAST": {
            "coordinate": "(1, 0)",
            "description": """
            Move the player one grid cell to the right on the kitchen layout. 
            This action changes the player's position horizontally by increasing their x-coordinate by 1. 
            Use this to navigate toward cooking stations, ingredient spawns, or serving areas located to the right of the current position. 
            Cannot move if there's a wall, counter, or another player blocking the path.
            """,
        },
        "WEST": {
            "coordinate": "(-1, 0)",
            "description": """
            Move the player one grid cell to the left on the kitchen layout. 
            This action changes the player's position horizontally by decreasing their x-coordinate by 1. 
            Use this to navigate toward cooking stations, ingredient spawns, or serving areas located to the left of the current position. 
            Cannot move if there's a wall, counter, or another player blocking the path.
            """,
        },
        "STAY": {
            "coordinate": "(0, 0)",
            "description": """
            Keep the player in their current position without moving. 
            This action maintains the player's current x and y coordinates. 
            Use this when waiting for a cooking timer to complete, when the desired adjacent tile is temporarily blocked by the other player, 
            when coordinating timing with your partner, or when the player is already optimally positioned for their next intended interaction.
            """,
        },
        "INTERACT": {
            "coordinate": "(1, 1)",
            "description": """
            Perform context-sensitive interactions with kitchen objects adjacent to or at the player's current position. 
            The specific interaction depends on what objects are nearby and what the player is currently holding and what
            task is assigned to the player/s.
            """,
        },
    }
