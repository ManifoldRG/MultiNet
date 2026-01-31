"""
MiniGrid Prompt Template for VLM Evaluation

Formats instruction prompts for the gridworld evaluation domain.
"""

INSTRUCTION = [
    "You are controlling an agent in a gridworld puzzle.",
    "The environment is \"{env_name}\".",
    "Task: {env_desc}",
    "You see a top-down view of the grid. The agent is shown as a red triangle pointing in its facing direction.",
    "Walls are grey, floors are light colored, and the goal is marked in green.",
    "Objects: Keys are small colored shapes, doors are colored rectangles, switches are yellow circles.",
    "The available actions are: {action_desc}",
    "Output format: {output_format}",
    "Respond with ONLY the action output, no explanations.",
    "{additional_inst}"
]


def format_instruction_prompt(
    env_name: str,
    env_desc: str,
    action_space: dict,
    only_one_action: bool,
    additional_inst: str = None
) -> str:
    """
    Format the instruction prompt for VLM evaluation.

    Args:
        env_name: Name of the environment/task
        env_desc: Description of the task objectives
        action_space: Dictionary defining the action space
        only_one_action: Whether only one action should be selected
        additional_inst: Additional instructions to append

    Returns:
        Formatted instruction prompt string
    """
    instruction_format = ' '.join(INSTRUCTION)

    # Format action descriptions
    actions = []
    for idx, tup in action_space.items():
        if len(tup) == 2:  # Discrete action with options
            desc, options = tup
            if isinstance(options, dict):
                # Format options as ID: Description pairs
                opts_str = ", ".join([f"{k}: {v}" for k, v in options.items()])
                sent = f"Action options: {opts_str}"
            else:
                sent = f"{idx}. {desc} => Options: {options}"
        else:
            sent = f"{idx}. {tup}"
        actions.append(sent)

    action_desc = '\n'.join(actions)

    # Determine output format
    if only_one_action:
        output_format = (
            "A single integer representing the action ID (0-6). "
            "For example: 2 (to move forward)"
        )
    else:
        output_format = (
            "A list of action IDs. For example: [2] for a single forward move, "
            "or [0, 2] for turn left then move forward."
        )

    # Build final prompt
    if additional_inst is not None and additional_inst.strip():
        prompt = instruction_format.format(
            env_name=env_name,
            env_desc=env_desc,
            action_desc=action_desc,
            output_format=output_format,
            additional_inst=additional_inst
        )
    else:
        prompt = instruction_format.format(
            env_name=env_name,
            env_desc=env_desc,
            action_desc=action_desc,
            output_format=output_format,
            additional_inst=""
        )

    return prompt


def format_simple_prompt(
    task_description: str,
    tier: int = 2,
    include_action_space: bool = True
) -> str:
    """
    Format a simplified prompt for quick evaluation.

    Args:
        task_description: Brief task description
        tier: Difficulty tier (1-5)
        include_action_space: Whether to include action space info

    Returns:
        Formatted prompt string
    """
    prompt_parts = [
        "You are an agent in a gridworld puzzle.",
        f"Task: {task_description}",
        "The image shows your current view of the grid.",
        "The red triangle is you (pointing in your facing direction).",
        "Green square is the goal. Grey cells are walls.",
    ]

    if include_action_space:
        if tier == 1:
            prompt_parts.append(
                "Actions: 0=turn left, 1=turn right, 2=move forward, 6=wait"
            )
        else:
            prompt_parts.append(
                "Actions: 0=turn left, 1=turn right, 2=move forward, "
                "3=pickup, 4=drop, 5=toggle/interact, 6=wait"
            )

    prompt_parts.append("Output: A single integer (0-6) for your next action.")

    return " ".join(prompt_parts)


def format_observation_context(
    agent_pos: tuple[int, int],
    agent_dir: int,
    carrying: str = None,
    visible_objects: list[str] = None
) -> str:
    """
    Format contextual information about the current observation.

    Args:
        agent_pos: Agent's (x, y) position
        agent_dir: Agent's facing direction (0=right, 1=down, 2=left, 3=up)
        carrying: What the agent is carrying (if anything)
        visible_objects: List of visible object descriptions

    Returns:
        Context string to append to prompt
    """
    dir_names = {0: "right", 1: "down", 2: "left", 3: "up"}
    context_parts = [
        f"Agent position: ({agent_pos[0]}, {agent_pos[1]})",
        f"Facing: {dir_names.get(agent_dir, 'unknown')}"
    ]

    if carrying:
        context_parts.append(f"Carrying: {carrying}")

    if visible_objects:
        context_parts.append(f"Visible objects: {', '.join(visible_objects)}")

    return " | ".join(context_parts)
