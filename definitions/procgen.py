class ProcGenDefinitions:
    DESCRIPTIONS = {
        "bigfish": {
            "eat the smaller fish and avoid the bigger fish": [
                "Become bigger by eating other fish that are smaller.",
                "Avoid coming in contact with other bigger fish."
            ]
        },
        "bossfight": {
            "destroy the boss and survive": [
                "Dodge incoming projectiles.",
                "Use meteors for cover.",
                "Damage the boss when its shields are down.",
                "Repeat damaging the boss until destroyed."
            ]
        },
        "caveflyer": {
            "navigate the caves to reach the exit": [
                "Reach the exit of the cave network.",
                "Destroy target objects with lasers.",
                "Avoid stationary and moving obstacles."
            ]
        },
        "chaser": {
            "collect orbs and avoid enemies": [
                "Collect all green orbs.",
                "Collect stars to make enemies vulnerable.",
                "Eat vulnerable enemies.",
                "Avoid enemies when not vulnerable."
            ]
        },
        "climber": {
            "climb to the top and collect stars": [
                "Climb platforms.",
                "Collect stars.",
                "Avoid lethal flying monsters."
            ]
        },
        "coinrun": {
            "reach and collect the coin": [
                "Reach the far right of the level.",
                "Dodge stationary saw obstacles.",
                "Avoid pacing enemies.",
                "Avoid chasms."
            ]
        },
        "dodgeball": {
            "hit enemies and reach platform": [
                "Avoid touching walls.",
                "Hit all enemies with balls.",
                "Reach the unlocked platform."
            ]
        },
        "fruitbot": {
            "collect fruit and avoid non-fruit": [
                "Navigate between gaps in walls.",
                "Collect fruit.",
                "Avoid collecting non-fruit objects.",
                "Use keys to unlock gates.",
                "Reach the end of the level."
            ]
        },
        "heist": {
            "steal the gem": [
                "Collect keys of different colors.",
                "Open colored locks.",
                "Reach and collect the hidden gem."
            ]
        },
        "jumper": {
            "find and collect the carrot": [
                "Navigate the open world.",
                "Use double jump to reach platforms.",
                "Avoid spike obstacles.",
                "Find and collect the carrot."
            ]
        },
        "leaper": {
            "cross the lanes": [
                "Avoid cars in the first lanes.",
                "Hop from log to log in the river lanes.",
                "Reach the finish line."
            ]
        },
        "maze": {
            "find the cheese": [
                "Navigate the maze.",
                "Find and collect the cheese."
            ]
        },
        "miner": {
            "collect diamonds and exit": [
                "Dig through dirt.",
                "Avoid falling boulders and diamonds.",
                "Collect all diamonds.",
                "Proceed through the exit."
            ]
        },
        "ninja": {
            "reach the mushroom": [
                "Jump across ledges.",
                "Avoid bomb obstacles.",
                "Use throwing stars to clear bombs.",
                "Collect the mushroom."
            ]
        },
        "plunder": {
            "destroy enemy ships": [
                "Fire cannonballs at enemy ships.",
                "Avoid hitting friendly ships.",
                "Target specific enemy ship colors.",
                "Manage the timer."
            ]
        },
        "starpilot": {
            "survive and defeat enemies": [
                "Dodge enemy projectiles.",
                "Defeat fast and slow enemies.",
                "Destroy stationary turrets.",
                "Navigate through clouds and meteors."
            ]
        }
    }


    movement_actions = {
        0: 'LEFT + DOWN',
        1: 'LEFT',
        2: 'LEFT + UP',
        3: 'DOWN',
        4: 'Do Nothing',
        5: 'UP',
        6: 'RIGHT + DOWN',
        7: 'RIGHT',
        8: 'RIGHT + UP'
    }

    special_actions = {
        9:  "D",
        10: "A",
        11: "W",
        12: "S",
        13: "Q",
        14: "E"
    }
    
    ACTION_SPACES = {
        "default": {
            "default": {
                0 : ("Agent movement action", movement_actions),
            }
        },
        "bossfight": {
            "default": {
                0 : ("Agent movement action", movement_actions),
                1 : ("Special action", {9 : "Fire a bullet in the direction the agent ship is facing"})
            }
        },
        "caveflyer": {
            "default": {
                0 : ("Agent movement action", movement_actions),
                1 : ("Special action", {9 : "Fire a bullet in the direction the agent is facing"})
            }
        },
        "dodgeball": {
            "default": {
                0 : ("Agent movement action", movement_actions),
                1 : ("Special action", {9 : "Fire a ball in the direction the agent is facing"})
            }
        },
        "fruitbot": {
            "default": {
                0 : ("Agent movement action", movement_actions),
                1 : ("Special action", {9 : "Fire a key in the direction the agent is facing"})
            }
        },
        "plunder": {
            "default": {
                0 : ("Agent movement action", movement_actions),
                1 : ("Special action", {9 : "Fire a cannonball in the direction the agent ship is facing"})
            }
        },
        "starpilot": {
            "default": {
                0 : ("Agent movement action", movement_actions),
                1 : ("Special action", {9 : "Fire a bullet in the direction the agent is facing", 10 : "Fire a bullet in the direction directly opposite of where the agent is facing"})
            }
        },
        "ninja": {
            "default": {
                0 : ("Agent movement action", movement_actions),
                1 : ("Special action", 
                     {9 : "Fire a throwing star in the direction the agent is facing in a horizontal line", 
                      10: "Fire a throwing star in the direction the agent is facing 45 degrees diagonally up from the horizontal line",
                      11: "Fire a throwing star in a vertical line upwards from the agent",
                      12: "Fire a throwing star in the direction the agent is facing 45 degrees diagonally down from the horizontal line"
                      }
                     )
            }
        },
        
    }

    ACTION_EXCLUSIVENESS = {
        "default": {
            "default": True
        }
    }

    ADDITIONAL_INSTRUCTIONS = {
    }

    ACTION_DECODE_STRATEGIES = {
        "default": "naive_dim_extension"
    }