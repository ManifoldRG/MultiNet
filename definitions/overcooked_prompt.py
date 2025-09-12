OVERCOOKED_INSTRUCTION = [
    "We are running a simulation for two AI agents cooperatively playing Overcooked AI in layout '{env_name}', a kitchen coordination game.",
    "Your role is to evaluate potential joint actions for both players based on the current game state.",
    "You should produce proper joint action outputs to maximize soup delivery through effective coordination.",
    "The current state consists of a game screen image showing player positions, ingredients, cooking stations, and objectives.",
    "Both players must work together - one player's actions affect the other's ability to complete tasks.",
    "Key game mechanics: Players can pick up ingredients, place them in pots (3 ingredients per soup), wait for cooking (20 time steps), then deliver completed soups to serving stations for points.",
    "Individual Action meanings: {action_meaning}",
    "Actions Available: {action_info}",
    "You MUST generate probabilities for all the actions within range[0, 35], including 0 and 35.",
    "You MUST generate your output keeping the following format: {output_format}",
    "You should not include any other words or characters in your response.",
    "{additional_inst}",
]

def format_instruction_prompt(
    env_name: str,
    action_meaning: str, 
    action_space: str,
    additional_inst: str = None,
) -> str:
    instruction_format = " ".join(OVERCOOKED_INSTRUCTION)


    output_format = """
    A list starting with '[' and ending with ']'. Each position corresponds to each one of the 36 action index. 
    Generate a probability distribution as a list of exactly 36 decimal values formatted as [p0, p1, p2, p3, ..., p35], 
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
        action_meaning=action_meaning,
        action_info=action_space,
        output_format=output_format,
        additional_inst=additional_inst,
    )

    return system_prompt
