from nlu_benchmark.env import GridState
from nlu_benchmark.renderer import NLRenderer
from nlu_benchmark.parser import ActionParser


class EpisodeRunner:
    def __init__(self, env, renderer, parser, max_steps=50, allow_regex_fallback=True):
        self.env = env
        self.renderer = renderer
        self.parser = parser
        self.max_steps = max_steps
        self.allow_regex_fallback = allow_regex_fallback

    @classmethod
    def from_json(cls, path, allow_regex_fallback=True):
        from nlu_benchmark.loader import load_maze
        env = load_maze(path)
        return cls(
            env=env,
            renderer=NLRenderer(),
            parser=ActionParser(),
            max_steps=env.initial.max_steps,
            allow_regex_fallback=allow_regex_fallback,
        )

    def build_system_prompt(self) -> str:
        return (
            "You are an agent navigating a grid world maze.\n"
            "Choose exactly one action from: MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST.\n"
            "Put your final chosen action on the last line in the format:\n"
            "FINAL_ACTION: <ACTION>\n"
            "Do not output more than one final action."
        )

    def build_user_prompt(self, state: GridState, last_feedback: str) -> str:
        full_map = self.renderer.render_full_map(state)
        return (
            f"Maze description:\n{full_map}\n\n"
            f"Current position: {state.agent_pos}\n"
            f"Goal position: {state.goal}\n"
            f"Last result: {last_feedback}\n"
            f"Step {state.step_count + 1} of {state.max_steps}.\n"
            "What is your next action?"
        )

    def run(self, agent):
        state = self.env.reset()
        last_feedback = "Episode start."
        messages = [{"role": "system", "content": self.build_system_prompt()}]
        transcript = []

        while state.step_count < self.max_steps:
            print(f"Step {state.step_count + 1} of {state.max_steps}")
            user_prompt = self.build_user_prompt(state, last_feedback)
            messages.append({"role": "user", "content": user_prompt})

            model_text = agent(messages)
            action, status = self.parser.parse(model_text, allow_regex_fallback=self.allow_regex_fallback)

            transcript.append({
                "position_before": state.agent_pos,
                "model_text": model_text,
                "parsed_action": action,
                "parse_status": status,
            })

            if action is None:
                last_feedback = (
                    "Parse error: I could not identify a valid action. "
                    "Reply with exactly one of MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST."
                )
                messages.append({"role": "assistant", "content": model_text})
                continue

            prev_pos = state.agent_pos
            state, event = self.env.step(action)

            if event.type in {"WALL", "OOB"}:
                last_feedback = f"Parsed action: {action}. {event.message} You remain at {prev_pos}."
            elif event.type == "DONE":
                last_feedback = f"Parsed action: {action}. {event.message}"
                transcript[-1]["position_after"] = state.agent_pos
                transcript[-1]["event_type"] = event.type
                transcript[-1]["feedback"] = last_feedback
                print(f"Success at step {state.step_count}: {transcript[-1]}")

                return {
                    "success": True,
                    "steps_used": state.step_count,
                    "final_state": state,
                    "transcript": transcript,
                }
            else:
                last_feedback = f"Parsed action: {action}. {event.message}"

            transcript[-1]["position_after"] = state.agent_pos
            transcript[-1]["event_type"] = event.type
            transcript[-1]["feedback"] = last_feedback
            messages.append({"role": "assistant", "content": model_text})

            print(f"Status at step {state.step_count}: {transcript[-1]}")
        return {
            "success": False,
            "steps_used": state.step_count,
            "final_state": state,
            "transcript": transcript,
        }