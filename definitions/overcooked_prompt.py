OVERCOOKED_INSTRUCTION = [
    "We are running a simulation for two AI agents cooperatively playing Overcooked AI in layout '{env_name}', a kitchen coordination game.",
    "Your role is to evaluate potential joint actions for both players based on the current game state.",
    "In this kitchen layout: {env_desc}",
    "You should produce proper joint action outputs to maximize soup delivery through effective coordination.",
    "The current state consists of a game screen image showing player positions, ingredients, cooking stations, and objectives.",
    "Both players must work together - one player's actions affect the other's ability to complete tasks.",
    "Key game mechanics: Players can pick up ingredients, place them in pots (3 ingredients per soup), wait for cooking (20 time steps), then deliver completed soups to serving stations for points.",
    "Individual Action meanings: {action_meaning}",
    "Each available action is integer from 0 to 35 containing both players individual actions, i.e., represents a joint action for both players simultaneously.\n"
    "Actions Available: {action_info}",
    "You MUST generate probabilities for all the actions within range[0, 35], including 0 and 35.",
    "You MUST generate your output keeping the following format: {output_format}",
    "The sum of probabilities score of all 36 actions MUST be equal to 1.",
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
    Generate a probability distribution as a list of exactly 36 decimal values formatted as [p1, p2, p3, ..., p36], 
    where each position in the list corresponds to one of the 36 possible action indices (0 through 35). Every 
    probability value must be a decimal number between 0.0 and 1.0 inclusive, representing the likelihood of selecting 
    that particular action. The list must contain all 36 probability values with no missing entries, and critically, 
    the sum of all 36 probabilities must equal exactly 1.0 to form a valid probability distribution. Ensure that when 
    all values are added together, they total precisely 1.0 - not more, not less - as this represents a complete probabilistic 
    coverage of all possible actions.
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
