INSTRUCTION = [
    "We are running a simulation for an AI agent playing a video game. Your role is to evaluate potential moves based on a snapshot.",
    "The description of this hypothetical scenario is \"{env_name}\".",
    "In this environment: {env_desc}",
    "You should produce a proper action output to achieve the final goal given the current progress so far given the current state information.",
    "The current state consists of an image, which is a snapshot of the game screen, and a text description of the objective.",
    "The actions available: {action_desc}",
    "You MUST generate your output keeping the following format: {output_format}",
    "You should not include any other words or characters in your response.",
    "{additional_inst}"
]


def format_instruction_prompt(env_name: str, env_desc: str, action_space: dict, only_one_action: bool, additional_inst: str=None) -> str:
    instruction_format = ' '.join(INSTRUCTION)

    actions = []
    discrete_only = True
    continuous_only = True
    for idx, tup in action_space.items():
        if len(tup) == 2:  # Discrete
            sent = f"{idx}. {tup[0]} => Discrete. Options: {tup[1]}."            
            continuous_only = False
        elif len(tup) == 3:  # Continuous
            sent = f"{idx}. {tup[0]} => Continous. Range: {tup[1]} ~ {tup[2]}."
            discrete_only = False
        elif len(tup) == 4:  # No verbal description or just verbal description with no stats
            sent = f"{idx}. {tup[0]} => Continuous. Range: {tup[1]} ~ {tup[2]}. Mean: {tup[3]}."
            discrete_only = False
        actions.append(sent)
        
    # Making the action description.    
    discrete_desc = "A discrete action has the available options as key-value pairs, {Option index}: {Option description}."
    cont_desc = "A continuous action has the range of {minimum} ~ {maximum}. A continuous action without a verbal description is described using the statistics of the action space over the entire dataset, which includes the range for each dimension between {minimum} ~ {maximum} and a mean of {mean}."
    
    if discrete_only:
        action_desc = [discrete_desc, ]
    elif continuous_only:
        action_desc = [cont_desc, ]
    else:
        action_desc = [f"{discrete_desc} {cont_desc}", ]
    action_desc.extend(actions)
    action_desc = '\n'.join(action_desc) + '\n'

    # Identifying the output format.
    if only_one_action:
        output_format = "A list starting with '[' and ending with ']'. "
        if discrete_only:
            output_format += "Each position corresponds to each action index. Each position MUST be a hashmap starting with '{' and ending with '}'. The hashmap should contain a key for each option index of that action, and the value for that key corresponds to the probability that this option should be selected as the next step. ALL probabilities across all actions, as opposed to per action or hashmap, MUST sum up to 1.0."
        elif continuous_only:
            output_format += "Each position corresponds to each action index. Each position MUST be a tuple with 2 values in this format: (the actual value of that action, the probability that this action should be taken). ALL probabilities across all actions MUST sum up to 1.0."
        else:
            output_format += "Each position corresponds to each action index. If the action is continuous, the corresponding position MUST be a tuple with 2 values in this format: (the actual value of that action, the probability that this action should be taken). If the action for an action index is discrete, the corresponding position MUST be a hashmap starting with '{' and ending with '}'. The hashmap should contain a key for each option index of that action, and the value for that key corresponds to the probability that this action-option combination should be selected as the next step. ALL probabilities across all actions, as opposed to per action or hashmap, MUST sum up to 1.0."
 
    else:
        output_format = "A list starting with '[' and ending with ']'. "
        if discrete_only:
            output_format += "Each position corresponds to each action index. Each position in that list should be a hashmap starting with '{' and ending with '}'. The hashmap contains a key for each option index of that action and the value for that key corresponds to the probability that this option should be selected. The probabilities of all option indices belonging to the action index must sum up to 1.0."
        elif continuous_only:
            output_format += "Each position corresponds to each action index and a value in that position represents the actual value (continuous) of that action."
        else:
            output_format += "Each position corresponds to each action index. If the action is continous, then a value in that position represents actual value (continuous) of that action. On the other hand, if the action is discrete, then a value in that position should be a hashmap starting with '{' and ending with '}', which should contain a key for each option index (discrete) of that action where the value for that key corresponds to the probability that this option should be selected. The probabilities of all option indices belonging to an action index must sum up to 1.0."
            
    if additional_inst is not None:
        system_prompt = instruction_format.format(env_name=env_name, env_desc=env_desc, action_desc=action_desc, output_format=output_format, additional_inst=additional_inst)
    else:
        system_prompt = instruction_format.format(env_name=env_name, env_desc=env_desc, action_desc=action_desc, output_format=output_format, additional_inst="")
    return system_prompt
