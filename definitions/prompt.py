INSTRUCTION = [
    "You are an AI agent to solve the task called \"{env_name}\".",
    "In this environment: {env_desc}",
    "You should produce a proper action output to achieve the final goal given the current progress so far given the current state information.",
    "The current state can be any forms, such as images, continuous/discrete vectors, or texts.",
    "The actions available: {action_desc}",
    "You must generate your output keeping the following format: {output_format}",
    "You should not include any other words or characters in your response.",
    "{additional_inst}"
]


def format_instruction_prompt(env_name: str, env_desc: str, action_space: dict, only_one_action: bool, additional_inst: str=None) -> str:
    instruction_format = ' '.join(INSTRUCTION)

    # Making the action description.
    action_desc = [
        "A discrete action has the available options as key-value pairs, {Option index}: {Option description}. A continuous action has the range of {minimum} ~ {maximum}. A continuous action without a verbal description is described using the statistics of the action space over the entire dataset, which includes the range for each dimension between {minimum} ~ {maximum} and a mean of {mean}.",
    ]
    for idx, tup in action_space.items():
        if len(tup) == 2:  # Discrete
            sent = f"{idx}. {tup[0]} => Discrete. Options: {tup[1]}."
        elif len(tup) == 3:  # Continuous
            sent = f"{idx}. {tup[0]} => Continous. Range: {tup[1]} ~ {tup[2]}."
        elif len(tup) == 4:  # No verbal description
            sent = f"{idx}. {tup[0]} => Continuous. Range: {tup[1]} ~ {tup[2]}. Mean: {tup[3]}."
        action_desc.append(sent)
    action_desc = '\n'.join(action_desc) + '\n'

    # Identifying the output format.
    if only_one_action:
        output_format = "{Action index} {Option index (discrete) or Actual value (continuous)} separated by a single whitespace. You should generate only two numbers without any other characters."
    else:
        output_format = "A list starting with '[' and ending with ']'. Each position corresponds to each action index and a value in that position represents the option index (discrete) or actual value (continuous) of that action."

    if additional_inst is not None:
        system_prompt = instruction_format.format(env_name=env_name, env_desc=env_desc, action_desc=action_desc, output_format=output_format, additional_inst=additional_inst)
    else:
        system_prompt = instruction_format.format(env_name=env_name, env_desc=env_desc, action_desc=action_desc, output_format=output_format, additional_inst="")
    return system_prompt
