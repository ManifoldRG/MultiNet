class OverCookedDefinitions:
    DESCRIPTIONS = {
        "cramped_corridor": {
            "coordinate in tight spaces to deliver soups": [
                "Navigate narrow corridors with your partner.",
                "Coordinate movement to avoid blocking each other.", 
                "Collect onions and place them in pots.",
                "Wait for soup to cook, then deliver to serving area.",
                "Optimize task division in constrained space."
            ]
        },
        "asymmetric_advantages_tomato": {
            "leverage role specialization for tomato soup": [
                "Utilize asymmetric layout advantages for each player.",
                "Specialize in different cooking tasks based on position.",
                "Collect tomatoes and place them in pots.",
                "Coordinate soup cooking and delivery timing.",
                "Adapt strategy based on partner's role."
            ]
        },
        "counter_circuit": {
            "coordinate around circular kitchen layout": [
                "Navigate circular counter arrangement efficiently.",
                "Plan movement patterns to avoid collisions.",
                "Collect ingredients and manage cooking stations.",
                "Deliver completed soups to serving locations.",
                "Maintain smooth workflow around the circuit."
            ]
        },
        "soup_coordination": {
            "coordinate complex soup preparation": [
                "Coordinate multi-step soup cooking process.",
                "Divide ingredient collection and cooking tasks.",
                "Time soup preparation and delivery efficiently.",
                "Communicate implicitly through actions.",
                "Maximize soup delivery rate through coordination."
            ]
        },
        "marshmallow_experiment": {
            "balance immediate vs delayed rewards": [
                "Choose between immediate small rewards or delayed larger rewards.",
                "Coordinate timing of soup deliveries.",
                "Plan long-term cooking strategy with partner.",
                "Resist suboptimal short-term actions.",
                "Optimize cumulative reward over time."
            ]
        },
        "inverse_marshmallow_experiment": {
            "prioritize immediate rewards over delayed ones": [
                "Focus on quick soup deliveries over complex coordination.",
                "Take immediate cooking opportunities.",
                "Minimize waiting time between actions.",
                "Adapt to partner's immediate action preferences.",
                "Balance speed with coordination efficiency."
            ]
        },
        "marshmallow_experiment_coordination": {
            "coordinate delayed gratification strategies": [
                "Coordinate complex timing for maximum rewards.",
                "Plan multi-soup cooking sequences with partner.",
                "Balance individual patience with team coordination.",
                "Execute synchronized cooking and delivery.",
                "Optimize long-term team performance."
            ]
        },
        "you_shall_not_pass": {
            "navigate blocking scenarios": [
                "Overcome movement blocking situations.",
                "Coordinate to unblock stuck partner.",
                "Plan alternative routes when blocked.",
                "Communicate intent through positioning.",
                "Resolve deadlock situations cooperatively."
            ]
        }
    }
    
    ACTION_SPACES = {
        "overcooked_ai": {
            "default": {
                0: ("Both players move NORTH", "Joint Action: (NORTH, NORTH) -> ((0, -1), (0, -1))"),
                1: ("Player 0 moves NORTH, Player 1 moves SOUTH", "Joint Action: (NORTH, SOUTH) -> ((0, -1), (0, 1))"),
                2: ("Player 0 moves NORTH, Player 1 moves EAST", "Joint Action: (NORTH, EAST) -> ((0, -1), (1, 0))"),
                3: ("Player 0 moves NORTH, Player 1 moves WEST", "Joint Action: (NORTH, WEST) -> ((0, -1), (-1, 0))"),
                4: ("Player 0 moves NORTH, Player 1 stays in place", "Joint Action: (NORTH, STAY) -> ((0, -1), (0, 0))"),
                5: ("Player 0 moves NORTH, Player 1 interacts with environment", "Joint Action: (NORTH, INTERACT) -> ((0, -1), (1, 1))"),
                6: ("Player 0 moves SOUTH, Player 1 moves NORTH", "Joint Action: (SOUTH, NORTH) -> ((0, 1), (0, -1))"),
                7: ("Both players move SOUTH", "Joint Action: (SOUTH, SOUTH) -> ((0, 1), (0, 1))"),
                8: ("Player 0 moves SOUTH, Player 1 moves EAST", "Joint Action: (SOUTH, EAST) -> ((0, 1), (1, 0))"),
                9: ("Player 0 moves SOUTH, Player 1 moves WEST", "Joint Action: (SOUTH, WEST) -> ((0, 1), (-1, 0))"),
                10: ("Player 0 moves SOUTH, Player 1 stays in place", "Joint Action: (SOUTH, STAY) -> ((0, 1), (0, 0))"),
                11: ("Player 0 moves SOUTH, Player 1 interacts with environment", "Joint Action: (SOUTH, INTERACT) -> ((0, 1), (1, 1))"),
                12: ("Player 0 moves EAST, Player 1 moves NORTH", "Joint Action: (EAST, NORTH) -> ((1, 0), (0, -1))"),
                13: ("Player 0 moves EAST, Player 1 moves SOUTH", "Joint Action: (EAST, SOUTH) -> ((1, 0), (0, 1))"),
                14: ("Both players move EAST", "Joint Action: (EAST, EAST) -> ((1, 0), (1, 0))"),
                15: ("Player 0 moves EAST, Player 1 moves WEST", "Joint Action: (EAST, WEST) -> ((1, 0), (-1, 0))"),
                16: ("Player 0 moves EAST, Player 1 stays in place", "Joint Action: (EAST, STAY) -> ((1, 0), (0, 0))"),
                17: ("Player 0 moves EAST, Player 1 interacts with environment", "Joint Action: (EAST, INTERACT) -> ((1, 0), (1, 1))"),
                18: ("Player 0 moves WEST, Player 1 moves NORTH", "Joint Action: (WEST, NORTH) -> ((-1, 0), (0, -1))"),
                19: ("Player 0 moves WEST, Player 1 moves SOUTH", "Joint Action: (WEST, SOUTH) -> ((-1, 0), (0, 1))"),
                20: ("Player 0 moves WEST, Player 1 moves EAST", "Joint Action: (WEST, EAST) -> ((-1, 0), (1, 0))"),
                21: ("Both players move WEST", "Joint Action: (WEST, WEST) -> ((-1, 0), (-1, 0))"),
                22: ("Player 0 moves WEST, Player 1 stays in place", "Joint Action: (WEST, STAY) -> ((-1, 0), (0, 0))"),
                23: ("Player 0 moves WEST, Player 1 interacts with environment", "Joint Action: (WEST, INTERACT) -> ((-1, 0), (1, 1))"),
                24: ("Player 0 stays in place, Player 1 moves NORTH", "Joint Action: (STAY, NORTH) -> ((0, 0), (0, -1))"),
                25: ("Player 0 stays in place, Player 1 moves SOUTH", "Joint Action: (STAY, SOUTH) -> ((0, 0), (0, 1))"),
                26: ("Player 0 stays in place, Player 1 moves EAST", "Joint Action: (STAY, EAST) -> ((0, 0), (1, 0))"),
                27: ("Player 0 stays in place, Player 1 moves WEST", "Joint Action: (STAY, WEST) -> ((0, 0), (-1, 0))"),
                28: ("Both players stay in place", "Joint Action: (STAY, STAY) -> ((0, 0), (0, 0))"),
                29: ("Player 0 stays in place, Player 1 interacts with environment", "Joint Action: (STAY, INTERACT) -> ((0, 0), (1, 1))"),
                30: ("Player 0 interacts with environment, Player 1 moves NORTH", "Joint Action: (INTERACT, NORTH) -> ((1, 1), (0, -1))"),
                31: ("Player 0 interacts with environment, Player 1 moves SOUTH", "Joint Action: (INTERACT, SOUTH) -> ((1, 1), (0, 1))"),
                32: ("Player 0 interacts with environment, Player 1 moves EAST", "Joint Action: (INTERACT, EAST) -> ((1, 1), (1, 0))"),
                33: ("Player 0 interacts with environment, Player 1 moves WEST", "Joint Action: (INTERACT, WEST) -> ((1, 1), (-1, 0))"),
                34: ("Player 0 interacts with environment, Player 1 stays in place", "Joint Action: (INTERACT, STAY) -> ((1, 1), (0, 0))"),
                35: ("Both players interact with environment", "Joint Action: (INTERACT, INTERACT) -> ((1, 1), (1, 1))")
            }
        }
    }
    
    
    ACTION_EXCLUSIVENESS = {
        "overcooked_ai": {
            "default": False            
        }
    }