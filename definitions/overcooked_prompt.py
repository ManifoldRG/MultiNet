OVERCOOKED_INSTRUCTION = [
    "We are running a simulation for two AI agents cooperatively playing Overcooked AI in layout '{env_name}', a kitchen coordination game.",
    "Your role is to evaluate potential joint actions for both players based on the current game state.",
    "In this kitchen layout: {env_desc}",
    "You should produce proper joint action outputs to maximize soup delivery through effective coordination.",
    "The current state consists of a game screen image showing player positions, ingredients, cooking stations, and objectives.",
    "Both players must work together - one player's actions affect the other's ability to complete tasks.",
    "Key game mechanics: Players can pick up ingredients, place them in pots (3 ingredients per soup), wait for cooking (20 time steps), then deliver completed soups to serving stations for points.",
    "Individual Action meanings: {action_meaning}",
    "Each available action is integer from 0 to 35 containing both players individual actions, i.e., represents a joint action for both players simultaneously. Select the action index that best coordinates both players' movements and interactions.\n"
    "The available options are key-value pairs, 'Option index': (Option description)."
    "Actions Available: {action_info}",
    "You MUST generate probabilities for all the actions within range[0, 35], including 0 and 35.",
    "You MUST generate your output keeping the following format: {output_format}",
    "You should not include any other words or characters in your response.",
    "Consider coordination: avoid blocking each other, divide tasks efficiently, and time actions for maximum soup throughput.",
    "{additional_inst}",
]

def format_instruction_prompt(
    env_name: str,
    env_desc: str,
    action_meaning: str, 
    action_space: str,
    additional_inst: str = None,
) -> str:
    instruction_format = " ".join(OVERCOOKED_INSTRUCTION)


    output_format = """
    A list starting with '[' and ending with ']'. Each position corresponds to each one of the 36 action index. 
    Each position MUST be a hashmap starting with '{' and ending with '}'. The hashmap should contain 
    a key for each option index of that action, and the value for that key corresponds to the probability 
    that this option should be selected as the next step. There MUST be all possible actions in the hashmap
    with probabilities assigned to each one of the actions. ALL probabilities across all actions, as opposed 
    to per action or hashmap, MUST sum up to 1.0. In other words, All assigned probabilities across the entire 
    action index (across all 36 actions) must sum to exactly 1.0..
    """
    # Format final prompt
    if additional_inst is None:
        additional_inst = "Focus on maximizing soup delivery rate while maintaining smooth coordination between players."

    system_prompt = instruction_format.format(
        env_name=env_name,
        env_desc=env_desc,
        action_meaning=action_meaning,
        action_info=action_space,
        output_format=output_format,
        additional_inst=additional_inst,
    )

    return system_prompt
