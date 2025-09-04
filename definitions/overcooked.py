class OverCookedDefinitions:
    DESCRIPTIONS = {
        "overcooked_ai": {
            "cramped_corridor": "coordinate in cramped corridor layout",
            "asymmetric_advantages_tomato": "coordinate in asymmetric advatages layout",
            "counter_circuit": "coordinate around counter circuit layout",
            "soup_coordination": "coordinate for soup preparation tasks",
            "marshmallow_experiment": "complete marshmallow experiment scenario",
            "inverse_marshmallow_experiment": "complete inverse marshmallow experiment scenario",
            "marshmallow_experiment_coordination": "coordinate marshmallow experiment tasks",
            "you_shall_not_pass": "navigate you shall not pass scenario"
        }
    }

    ACTION_SPACES = {
        "overcooked_ai": {
            "default": {
                0: (
                    "Both players move NORTH",
                    "Joint Action: {NORTH, NORTH} -> {(0, -1), (0, -1)}",
                ),
                1: (
                    "Player 0 moves NORTH, Player 1 moves SOUTH",
                    "Joint Action: {NORTH, SOUTH} -> {(0, -1), (0, 1)}",
                ),
                2: (
                    "Player 0 moves NORTH, Player 1 moves EAST",
                    "Joint Action: {NORTH, EAST} -> {(0, -1), (1, 0)}",
                ),
                3: (
                    "Player 0 moves NORTH, Player 1 moves WEST",
                    "Joint Action: {NORTH, WEST} -> {(0, -1), (-1, 0)}",
                ),
                4: (
                    "Player 0 moves NORTH, Player 1 stays in place",
                    "Joint Action: {NORTH, STAY} -> {(0, -1), (0, 0)}",
                ),
                5: (
                    "Player 0 moves NORTH, Player 1 interacts with environment",
                    "Joint Action: {NORTH, INTERACT} -> {(0, -1), (1, 1)}",
                ),
                6: (
                    "Player 0 moves SOUTH, Player 1 moves NORTH",
                    "Joint Action: {SOUTH, NORTH} -> {(0, 1), (0, -1)}",
                ),
                7: (
                    "Both players move SOUTH",
                    "Joint Action: {SOUTH, SOUTH} -> {(0, 1), (0, 1)}",
                ),
                8: (
                    "Player 0 moves SOUTH, Player 1 moves EAST",
                    "Joint Action: {SOUTH, EAST} -> {(0, 1), (1, 0)}",
                ),
                9: (
                    "Player 0 moves SOUTH, Player 1 moves WEST",
                    "Joint Action: {SOUTH, WEST} -> {(0, 1), (-1, 0)}",
                ),
                10: (
                    "Player 0 moves SOUTH, Player 1 stays in place",
                    "Joint Action: {SOUTH, STAY} -> {(0, 1), (0, 0)}",
                ),
                11: (
                    "Player 0 moves SOUTH, Player 1 interacts with environment",
                    "Joint Action: {SOUTH, INTERACT} -> {(0, 1), (1, 1)}",
                ),
                12: (
                    "Player 0 moves EAST, Player 1 moves NORTH",
                    "Joint Action: {EAST, NORTH} -> {(1, 0), (0, -1)}",
                ),
                13: (
                    "Player 0 moves EAST, Player 1 moves SOUTH",
                    "Joint Action: {EAST, SOUTH} -> {(1, 0), (0, 1)}",
                ),
                14: (
                    "Both players move EAST",
                    "Joint Action: {EAST, EAST} -> {(1, 0), (1, 0)}",
                ),
                15: (
                    "Player 0 moves EAST, Player 1 moves WEST",
                    "Joint Action: {EAST, WEST} -> {(1, 0), (-1, 0)}",
                ),
                16: (
                    "Player 0 moves EAST, Player 1 stays in place",
                    "Joint Action: {EAST, STAY} -> {(1, 0), (0, 0)}",
                ),
                17: (
                    "Player 0 moves EAST, Player 1 interacts with environment",
                    "Joint Action: {EAST, INTERACT} -> {(1, 0), (1, 1)}",
                ),
                18: (
                    "Player 0 moves WEST, Player 1 moves NORTH",
                    "Joint Action: {WEST, NORTH} -> {(-1, 0), (0, -1)}",
                ),
                19: (
                    "Player 0 moves WEST, Player 1 moves SOUTH",
                    "Joint Action: {WEST, SOUTH} -> {(-1, 0), (0, 1)}",
                ),
                20: (
                    "Player 0 moves WEST, Player 1 moves EAST",
                    "Joint Action: {WEST, EAST} -> {(-1, 0), (1, 0)}",
                ),
                21: (
                    "Both players move WEST",
                    "Joint Action: {WEST, WEST} -> {(-1, 0), (-1, 0)}",
                ),
                22: (
                    "Player 0 moves WEST, Player 1 stays in place",
                    "Joint Action: {WEST, STAY} -> {(-1, 0), (0, 0)}",
                ),
                23: (
                    "Player 0 moves WEST, Player 1 interacts with environment",
                    "Joint Action: {WEST, INTERACT} -> {(-1, 0), (1, 1)}",
                ),
                24: (
                    "Player 0 stays in place, Player 1 moves NORTH",
                    "Joint Action: {STAY, NORTH} -> {(0, 0), (0, -1)}",
                ),
                25: (
                    "Player 0 stays in place, Player 1 moves SOUTH",
                    "Joint Action: {STAY, SOUTH} -> {(0, 0), (0, 1)}",
                ),
                26: (
                    "Player 0 stays in place, Player 1 moves EAST",
                    "Joint Action: {STAY, EAST} -> {(0, 0), (1, 0)}",
                ),
                27: (
                    "Player 0 stays in place, Player 1 moves WEST",
                    "Joint Action: {STAY, WEST} -> {(0, 0), (-1, 0)}",
                ),
                28: (
                    "Both players stay in place",
                    "Joint Action: {STAY, STAY} -> {(0, 0), (0, 0)}",
                ),
                29: (
                    "Player 0 stays in place, Player 1 interacts with environment",
                    "Joint Action: {STAY, INTERACT} -> {(0, 0), (1, 1)}",
                ),
                30: (
                    "Player 0 interacts with environment, Player 1 moves NORTH",
                    "Joint Action: {INTERACT, NORTH} -> {(1, 1), (0, -1)}",
                ),
                31: (
                    "Player 0 interacts with environment, Player 1 moves SOUTH",
                    "Joint Action: {INTERACT, SOUTH} -> {(1, 1), (0, 1)}",
                ),
                32: (
                    "Player 0 interacts with environment, Player 1 moves EAST",
                    "Joint Action: {INTERACT, EAST} -> {(1, 1), (1, 0)}",
                ),
                33: (
                    "Player 0 interacts with environment, Player 1 moves WEST",
                    "Joint Action: {INTERACT, WEST} -> {(1, 1), (-1, 0)}",
                ),
                34: (
                    "Player 0 interacts with environment, Player 1 stays in place",
                    "Joint Action: {INTERACT, STAY} -> {(1, 1), (0, 0)}",
                ),
                35: (
                    "Both players interact with environment",
                    "Joint Action: {INTERACT, INTERACT} -> {(1, 1), (1, 1)}",
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
            This includes: picking up ingredients from dispensers or counters, placing held ingredients into cooking pots, 
            placing cooked soups onto serving counters, delivering completed dishes to serving windows for points, 
            dropping held items onto empty counter spaces, and picking up completed soups from cooking pots. 
            The specific interaction depends on what objects are nearby and what the player is currently holding.
            """,
        },
    }
