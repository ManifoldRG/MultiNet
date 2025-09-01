OVERCOOKED_INSTRUCTION = [
    "We are running a simulation for two AI agents cooperatively playing Overcooked, a kitchen coordination game.",
    "Your role is to evaluate potential joint actions for both players based on the current game state.",
    'The layout scenario is "{env_name}".',
    "In this kitchen layout: {env_desc}",
    "You should produce proper joint action outputs to maximize soup delivery through effective coordination.",
    "The current state consists of a game screen image showing player positions, ingredients, cooking stations, and objectives.",
    "Both players must work together - one player's actions affect the other's ability to complete tasks.",
    "Key game mechanics: Players can pick up ingredients, place them in pots (3 ingredients per soup), wait for cooking (20 time steps), then deliver completed soups to serving stations for points.",
    "Action meanings: NORTH=(0,-1) moves up, SOUTH=(0,1) moves down, EAST=(1,0) moves right, WEST=(-1,0) moves left, STAY=(0,0) remains in place, INTERACT=(1,1) picks up/places items/serves soups.",
    "The actions available: {action_info}",
    "You MUST generate your output keeping the following format: {output_format}",
    "You should not include any other words or characters in your response.",
    "Consider coordination: avoid blocking each other, divide tasks efficiently, and time actions for maximum soup throughput.",
    "{additional_inst}",
]


def format_instruction_prompt(
    env_name: str,
    env_desc: str,
    action_space: str,
    only_one_action: False,
    additional_inst: str = None,
) -> str:
    instruction_format = " ".join(OVERCOOKED_INSTRUCTION)

    # Build action descriptions
    action_info = "Each action represents a joint action for both players simultaneously. Select the action index that best coordinates both players' movements and interactions.\n"
    action_info += action_space

    output_format = "{Action index}, A single integer from 0 to 35 representing the selected joint action index."

    # Format final prompt
    if additional_inst is None:
        additional_inst = "Focus on maximizing soup delivery rate while maintaining smooth coordination between players."

    system_prompt = instruction_format.format(
        env_name=env_name,
        env_desc=env_desc,
        action_info=action_info,
        output_format=output_format,
        additional_inst=additional_inst,
    )

    return system_prompt
