from nlu_benchmark.env import GridState


class NLRenderer:
    def render_full_map(self, state: GridState) -> str:
        wall_str = ", ".join(f"({r},{c})" for r, c in sorted(state.walls)) or "none"
        parts = [
            f"The world is a {state.rows} by {state.cols} grid.",
            "Coordinates are given as (row, column).",
            "The top-left corner is (1,1).",
            f"The start is at {state.start}.",
            f"The goal is at {state.goal}.",
            f"The following cells are walls: {wall_str}.",
        ]

        for key in state.keys:
            r, c = key["position"]
            parts.append(f"There is a {key['color']} key at ({r},{c}).")

        for door in state.doors:
            r, c = door["position"]
            parts.append(
                f"There is a locked {door['requires_key']} door at ({r},{c})."
                f" It requires the {door['requires_key']} key to open."
            )

        for switch in state.switches:
            r, c = switch["position"]
            controls = ", ".join(switch.get("controls", []))
            parts.append(
                f"There is a {switch.get('switch_type', 'toggle')} switch at ({r},{c})."
                f" It controls: {controls}."
            )

        for gate in state.gates:
            r, c = gate["position"]
            parts.append(
                f"There is a gate ({gate['id']}) at ({r},{c})."
                f" It is initially {gate.get('initial_state', 'closed')}."
            )

        return "\n".join(parts)
