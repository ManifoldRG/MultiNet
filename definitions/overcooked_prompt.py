OVERCOOKED_INSTRUCTION = [
    "We are running a simulation for two AI agents cooperatively playing Overcooked in layout '{env_name}', a kitchen coordination game.",
    "Your role is to evaluate potential joint actions for both players based on the current game state.",
    "You should produce proper joint action outputs to maximize soup delivery through effective coordination.",
    "The current state consists of a game screenshot showing player positions, ingredients, cooking stations, objectives.",
    "Values for the elapsed time and the remaining time are also provided.",
    "Both players must work together - one player's actions affect the other's ability to complete tasks.",
    "Key game mechanics: Players can pick up ingredients, place them in pots (3 ingredients per soup), wait for cooking (20 time steps), then deliver completed soups to serving stations for points.",
    "Individual action meanings: {action_meaning}.",
    "Options available in the format 'Player 0 action, Player 1 action': {action_info}.",
    "Time remaining: {time_left}.",
    "Elapsed time: {time_elapsed}.",
    "Focus on maximizing soup delivery rate while maintaining smooth coordination between players within the time remaining.",
    "You MUST generate your output keeping the following format: {output_format}",
    "You should not include any other words or characters in your response.",
    "{additional_inst}",
]

def format_instruction_prompt(
    env_name: str,
    action_meaning: str, 
    action_space: str,
    time_left: float,
    time_elapsed: float,
    additional_inst: str = None,
) -> str:
    instruction_format = " ".join(OVERCOOKED_INSTRUCTION)


    output_format = """
        A list starting with '[' and ending with ']'. Each position corresponds to each one of the 36 option indices. 
        Generate a probability distribution as a list of exactly 36 decimal values formatted as [p0, p1, p2, p3, ..., p35], 
        where each position in the list corresponds to one of the 36 possible option indices (0 through 35). Every 
        probability value must be a decimal number between 0.0 and 1.0 inclusive, representing the probability of selecting 
        that particular option. The list must contain all 36 probability values with no missing entries. 
        Critically, the sum of all 36 probabilities MUST equal exactly 1.0 to form a valid probability distribution.
    """
    # Format final prompt
    if additional_inst is None:
        additional_inst = ""

    system_prompt = instruction_format.format(
        env_name=env_name,
        action_meaning=action_meaning,
        action_info=action_space,
        time_left=time_left,
        time_elapsed=time_elapsed,
        output_format=output_format,
        additional_inst=additional_inst,
    )

    return system_prompt
