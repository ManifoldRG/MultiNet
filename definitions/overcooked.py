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
                0: "Player 0: NORTH, Player 1: NORTH",
                1: "Player 0: NORTH, Player 1: SOUTH",
                2: "Player 0: NORTH, Player 1: EAST",
                3: "Player 0: NORTH, Player 1: WEST",
                4: "Player 0: NORTH, Player 1: STAY",
                5: "Player 0: NORTH, Player 1: INTERACT",
                6: "Player 0: SOUTH, Player 1: NORTH",
                7: "Player 0: SOUTH, Player 1: SOUTH",
                8: "Player 0: SOUTH, Player 1: EAST",
                9: "Player 0: SOUTH, Player 1: WEST",
                10: "Player 0: SOUTH, Player 1: STAY",
                11: "Player 0: SOUTH, Player 1: INTERACT",
                12: "Player 0: EAST, Player 1: NORTH",
                13: "Player 0: EAST, Player 1: SOUTH",
                14: "Player 0: EAST, Player 1: EAST",
                15: "Player 0: EAST, Player 1: WEST",
                16: "Player 0: EAST, Player 1: STAY",
                17: "Player 0: EAST, Player 1: INTERACT",
                18: "Player 0: WEST, Player 1: NORTH",
                19: "Player 0: WEST, Player 1: SOUTH",
                20: "Player 0: WEST, Player 1: EAST",
                21: "Player 0: WEST, Player 1: WEST",
                22: "Player 0: WEST, Player 1: STAY",
                23: "Player 0: WEST, Player 1: INTERACT",
                24: "Player 0: STAY, Player 1: NORTH",
                25: "Player 0: STAY, Player 1: SOUTH",
                26: "Player 0: STAY, Player 1: EAST",
                27: "Player 0: STAY, Player 1: WEST",
                28: "Player 0: STAY, Player 1: STAY",
                29: "Player 0: STAY, Player 1: INTERACT",
                30: "Player 0: INTERACT, Player 1: NORTH",
                31: "Player 0: INTERACT, Player 1: SOUTH",
                32: "Player 0: INTERACT, Player 1: EAST",
                33: "Player 0: INTERACT, Player 1: WEST",
                34: "Player 0: INTERACT, Player 1: STAY",
                35: "Player 0: INTERACT, Player 1: INTERACT",
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
            """,
        },
        "SOUTH": {
            "coordinate": "(0, 1)",
            "description": """
            Move the player one grid cell downward on the kitchen layout.
            """,
        },
        "EAST": {
            "coordinate": "(1, 0)",
            "description": """
            Move the player one grid cell to the right on the kitchen layout.
            """,
        },
        "WEST": {
            "coordinate": "(-1, 0)",
            "description": """
            Move the player one grid cell to the left on the kitchen layout.
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
            The specific interaction depends on what objects are nearby, what the player is currently holding and what task 
            is assigned to the player(s).
            """,
        },
    }
