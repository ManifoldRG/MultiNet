#!/usr/bin/env python3
"""
Manual chat-interface smoke test runner for MiniGrid v1.1.

This runner is for frontier web-chat testing where the model is controlled
through ChatGPT / Claude / Gemini manually rather than through an API.

It exports a prompt packet for each query turn:
  - current frame PNG
  - optional prior frame PNGs
  - prompt text to paste into the chat UI
  - machine-readable state JSON

The model reply can contain one or more actions. When `--allow-look` is set,
the reply may also include `LOOK` to request an updated frame before consuming
the full action budget.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

LOOK_TOKEN = "LOOK"

ACTION_NAMES = {
    0: "turn_left",
    1: "turn_right",
    2: "move_forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

ACTION_DESCRIPTIONS = {
    0: "Turn left (rotate 90 degrees counter-clockwise)",
    1: "Turn right (rotate 90 degrees clockwise)",
    2: "Move forward (one cell in facing direction)",
    3: "Pick up (grab object in front of agent)",
    4: "Drop (release held object)",
    5: "Toggle (interact with object in front)",
    6: "Done/Wait (no action, stay in place)",
}

ACTION_ALIASES = {
    "left": 0,
    "turn_left": 0,
    "turn left": 0,
    "right": 1,
    "turn_right": 1,
    "turn right": 1,
    "forward": 2,
    "move_forward": 2,
    "move forward": 2,
    "pickup": 3,
    "pick_up": 3,
    "pick up": 3,
    "drop": 4,
    "toggle": 5,
    "interact": 5,
    "wait": 6,
    "done": 6,
    "no_op": 6,
    "no-op": 6,
    "noop": 6,
}


@dataclass
class ParsedReply:
    actions: list[int]
    requested_look: bool


def parse_model_reply(raw: str, *, max_actions: int, allow_look: bool) -> ParsedReply:
    """Parse a pasted web-chat reply into actions and an optional LOOK request."""
    actions: list[int] = []
    requested_look = False

    for raw_line in re.split(r"[\n,]+", raw):
        line = raw_line.strip()
        if not line:
            continue

        # Strip common bullet/numbering prefixes.
        line = re.sub(r"^\s*(?:[-*]|\d+[.)]|action\s*\d*:?)\s*", "", line, flags=re.IGNORECASE)
        if not line:
            continue

        normalized = line.strip().strip("`").strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = normalized.replace("-", " ").replace("_", " ")
        normalized = normalized.strip()

        if allow_look and normalized == "look":
            requested_look = True
            break

        action_id = _parse_action_token(normalized)
        if action_id is None:
            continue

        actions.append(action_id)
        if max_actions > 0 and len(actions) >= max_actions:
            break

    if not actions and not requested_look:
        raise ValueError(
            "Could not parse any action from the reply. Use one action per line, "
            "for example `move_forward`, `turn_right`, `2`, or `LOOK`."
        )

    return ParsedReply(actions=actions, requested_look=requested_look)


def _count_reply_tokens_proxy(raw: str) -> int:
    """Cheap proxy for reply token usage based on action-like segments."""
    parts = [piece.strip() for piece in re.split(r"[\n,]+", raw) if piece.strip()]
    return len(parts)


def _parse_action_token(normalized: str) -> int | None:
    digit_match = re.match(r"^([0-6])(?:\b|[^0-9].*)?$", normalized)
    if digit_match:
        return int(digit_match.group(1))

    if normalized in ACTION_ALIASES:
        return ACTION_ALIASES[normalized]

    compact = normalized.replace(" ", "_")
    if compact in ACTION_NAMES.values():
        return next(action_id for action_id, action_name in ACTION_NAMES.items() if action_name == compact)

    for action_id, action_name in ACTION_NAMES.items():
        pretty = action_name.replace("_", " ")
        if normalized == pretty:
            return action_id

    return None


def build_prompt(
    *,
    step_number: int,
    max_steps: int,
    action_budget: int,
    allow_look: bool,
    text_history: str | None,
    prior_image_count: int,
) -> str:
    lines = [
        "You are controlling the blue agent from images only.",
        "Objective: get to the green square goal.",
        f"Current step: {step_number}/{max_steps}",
        "",
        "You are graded on success and token efficiency.",
        "Both input and output tokens matter.",
        "LOOK requests are not free.",
        "",
        "Available actions:",
    ]
    for action_id in sorted(ACTION_NAMES):
        lines.append(f"{action_id}: {ACTION_NAMES[action_id]} - {ACTION_DESCRIPTIONS[action_id]}")

    lines.append("")
    if action_budget > 0:
        lines.append(f"Reply with up to {action_budget} action(s), one per line.")
    else:
        lines.append("Reply with as many actions as you want, one per line.")
    lines.append("Use only action ids `0-6` or exact action names like `move_forward`.")

    if allow_look:
        lines.append(
            "If you want a refreshed image before continuing, write `LOOK` on its own line "
            "after the last action you want executed."
        )

    lines.extend([
        "Do not explain your reasoning.",
        "Do not restate the task.",
    ])

    if prior_image_count:
        lines.extend([
            "",
            f"There are {prior_image_count} earlier frame(s) attached for short-term visual history.",
            "The current image is the most recent frame.",
        ])

    if text_history:
        lines.extend([
            "",
            "Recent action history:",
            text_history,
        ])

    return "\n".join(lines).strip() + "\n"


class ChatSmokeSession:
    def __init__(
        self,
        *,
        task_path: str,
        session_dir: str,
        query_interval: int,
        allow_look: bool,
        history_images: int,
        history_text_window: int,
    ):
        self.task_path = str(Path(task_path).resolve())
        self.session_dir = Path(session_dir)
        self.query_interval = query_interval
        self.allow_look = allow_look
        self.history_images = history_images
        self.history_text_window = history_text_window

        from gridworld.backends.minigrid_backend import MiniGridBackend
        from gridworld.task_spec import TaskSpecification
        from gridworld.task_validator import compute_difficulty

        self.spec = TaskSpecification.from_json(self.task_path)
        self.backend = MiniGridBackend(render_mode="rgb_array")
        self.backend.configure(self.spec)
        self.difficulty = compute_difficulty(self.spec)

        self.packet_index = 0
        self.query_index = 0
        self.frame_history: list[np.ndarray] = []
        self.text_history: list[str] = []
        self.transcript_path = self.session_dir / "transcript.jsonl"

        self.obs: Optional[np.ndarray] = None
        self.state = None
        self.mission = ""
        self.done = False
        self.success = False
        self.packet_metrics: list[dict] = []

    def start(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.obs, self.state, _ = self.backend.reset(seed=self.spec.seed)
        current_frame = self.backend.render().copy()
        self.obs = current_frame
        self.frame_history = [current_frame]
        self.mission = self.backend.get_mission_text()
        self._write_session_metadata()

    def close(self) -> None:
        self._write_summary()
        self.backend.close()

    def _write_session_metadata(self) -> None:
        metadata = {
            "task_path": self.task_path,
            "task_id": self.spec.task_id,
            "seed": self.spec.seed,
            "query_interval": self.query_interval,
            "allow_look": self.allow_look,
            "history_images": self.history_images,
            "history_text_window": self.history_text_window,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        (self.session_dir / "session.json").write_text(_json_dumps(metadata, indent=2))

    def export_packet(self) -> Path:
        packet_dir = self.session_dir / f"packet_{self.packet_index:03d}"
        packet_dir.mkdir(parents=True, exist_ok=True)

        current_frame = self.backend.render().copy()
        if self.frame_history:
            self.frame_history[-1] = current_frame
        else:
            self.frame_history = [current_frame]

        if self.history_images > 0:
            prior_images = [frame.copy() for frame in self.frame_history[:-1][-self.history_images:]]
        else:
            prior_images = []
        for index, image in enumerate(prior_images, start=1):
            Image.fromarray(image).save(packet_dir / f"prior_{index}.png")
        Image.fromarray(current_frame).save(packet_dir / "current.png")

        text_history = None
        if self.text_history and self.history_text_window > 0:
            text_history = "\n".join(self.text_history[-self.history_text_window:])

        prompt = build_prompt(
            step_number=self.state.step_count,
            max_steps=self.state.max_steps,
            action_budget=self.query_interval,
            allow_look=self.allow_look,
            text_history=text_history,
            prior_image_count=len(prior_images),
        )

        packet = {
            "task_id": self.spec.task_id,
            "task_path": self.task_path,
            "step_count": self.state.step_count,
            "max_steps": self.state.max_steps,
            "query_index": self.query_index,
            "packet_index": self.packet_index,
            "query_interval": self.query_interval,
            "allow_look": self.allow_look,
            "position": list(self.state.agent_position),
            "direction": self.state.agent_direction,
            "current_image": "current.png",
            "prior_images": [f"prior_{i}.png" for i in range(1, len(prior_images) + 1)],
            "prompt_file": "prompt.txt",
        }
        prompt_char_count = len(prompt)
        recent_text_history_count = len(self.text_history[-self.history_text_window:]) if self.history_text_window > 0 else 0
        packet_metrics = {
            "packet_index": self.packet_index,
            "query_index": self.query_index,
            "step_count": int(self.state.step_count),
            "prompt_char_count": prompt_char_count,
            "prompt_word_count_est": len(prompt.split()),
            "attached_image_count": len(prior_images) + 1,
            "attached_prior_image_count": len(prior_images),
            "recent_text_history_count": recent_text_history_count,
        }
        self.packet_metrics.append(packet_metrics)

        (packet_dir / "prompt.txt").write_text(prompt)
        (packet_dir / "state.json").write_text(_json_dumps(packet, indent=2))
        (packet_dir / "debug_state.json").write_text(
            _json_dumps(
                {
                    "packet": packet,
                    "packet_metrics": packet_metrics,
                    "grid_state": self.state.to_dict(),
                    "recent_text_history": self.text_history[-self.history_text_window:]
                    if self.history_text_window > 0 else [],
                },
                indent=2,
            )
        )
        (packet_dir / "user_message.md").write_text(
            "# Paste This Into The Chat UI\n\n"
            "Attach `current.png` and any `prior_*.png` files from this packet, then paste:\n\n"
            "```text\n"
            f"{prompt}"
            "```\n"
        )
        _save_contact_sheet(
            packet_dir / "contact_sheet.png",
            prior_images=prior_images,
            current_image=current_frame,
        )
        (packet_dir / "debug_readme.md").write_text(
            "# Packet Debug\n\n"
            f"- `packet_index`: {self.packet_index}\n"
            f"- `query_index`: {self.query_index}\n"
            f"- `step_count`: {self.state.step_count}\n"
            f"- `history_images_attached`: {len(prior_images)}\n"
            "- `contact_sheet.png` shows earlier frames left-to-right and the current frame last.\n"
            "- `debug_state.json` includes the serialized `GridState` and recent text history.\n"
        )

        self.packet_index += 1
        return packet_dir

    def apply_reply(self, reply_text: str) -> ParsedReply:
        parsed = parse_model_reply(
            reply_text,
            max_actions=self.query_interval,
            allow_look=self.allow_look,
        )

        self._append_transcript({
            "type": "model_reply",
            "query_index": self.query_index,
            "step_count": self.state.step_count,
            "raw_reply": reply_text,
            "reply_char_count": len(reply_text),
            "reply_word_count_est": _count_reply_tokens_proxy(reply_text),
            "parsed_actions": parsed.actions,
            "parsed_action_count": len(parsed.actions),
            "parsed_action_names": [ACTION_NAMES[a] for a in parsed.actions],
            "requested_look": parsed.requested_look,
        })

        for action in parsed.actions:
            previous_position = tuple(self.state.agent_position)
            previous_direction = int(self.state.agent_direction)
            previous_carrying = self.state.agent_carrying
            previous_open_doors = set(self.state.open_doors)
            previous_open_gates = set(self.state.open_gates)
            previous_active_switches = set(self.state.active_switches)
            previous_goal_reached = bool(self.state.goal_reached)

            self.obs, reward, terminated, truncated, self.state, info = self.backend.step(action)
            current_frame = self.backend.render().copy()
            self.obs = current_frame
            self.frame_history.append(current_frame)
            state_changed = (
                tuple(self.state.agent_position) != previous_position
                or int(self.state.agent_direction) != previous_direction
                or self.state.agent_carrying != previous_carrying
                or set(self.state.open_doors) != previous_open_doors
                or set(self.state.open_gates) != previous_open_gates
                or set(self.state.active_switches) != previous_active_switches
                or bool(self.state.goal_reached) != previous_goal_reached
                or reward != 0
            )
            blocked_or_no_effect = not state_changed
            self.text_history.append(
                f"step {self.state.step_count}: action={ACTION_NAMES[action]}, "
                f"from={list(previous_position)} facing={previous_direction}, "
                f"to={list(self.state.agent_position)} facing={self.state.agent_direction}, "
                f"reward={reward:.3f}"
            )
            self._append_transcript({
                "type": "env_step",
                "query_index": self.query_index,
                "step_count": self.state.step_count,
                "action": action,
                "action_name": ACTION_NAMES[action],
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "position": list(self.state.agent_position),
                "direction": self.state.agent_direction,
                "carrying": self.state.agent_carrying,
                "state_changed": state_changed,
                "blocked_or_no_effect": blocked_or_no_effect,
                "info": info,
            })
            if terminated or truncated:
                self.done = True
                self.success = bool(terminated and reward > 0)
                break

        self.query_index += 1
        return parsed

    def _append_transcript(self, record: dict) -> None:
        with self.transcript_path.open("a") as handle:
            handle.write(_json_dumps(record) + "\n")

    def _write_summary(self) -> None:
        if self.state is None:
            return

        model_replies = 0
        total_reply_char_count = 0
        total_reply_word_count_est = 0
        total_actions_proposed = 0
        look_requests = 0
        blocked_or_no_effect_actions = 0
        state_changed_actions = 0
        if self.transcript_path.exists():
            for line in self.transcript_path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("type") == "model_reply":
                    model_replies += 1
                    total_reply_char_count += int(record.get("reply_char_count", 0))
                    total_reply_word_count_est += int(record.get("reply_word_count_est", 0))
                    total_actions_proposed += int(record.get("parsed_action_count", 0))
                    look_requests += int(bool(record.get("requested_look")))
                elif record.get("type") == "env_step":
                    blocked_or_no_effect_actions += int(bool(record.get("blocked_or_no_effect")))
                    state_changed_actions += int(bool(record.get("state_changed")))

        optimal_steps = int(self.difficulty.optimal_steps)
        optimality_ratio = None
        if optimal_steps > 0:
            optimality_ratio = self.state.step_count / optimal_steps

        summary = {
            "task_id": self.spec.task_id,
            "success": self.success,
            "final_step_count": int(self.state.step_count),
            "max_steps": int(self.state.max_steps),
            "solver": self.difficulty.to_dict(),
            "optimal_steps": optimal_steps,
            "optimality_ratio": optimality_ratio,
            "query_count": model_replies,
            "look_requests": look_requests,
            "total_actions_proposed": total_actions_proposed,
            "blocked_or_no_effect_actions": blocked_or_no_effect_actions,
            "state_changed_actions": state_changed_actions,
            "total_prompt_char_count": sum(item["prompt_char_count"] for item in self.packet_metrics),
            "total_prompt_word_count_est": sum(item["prompt_word_count_est"] for item in self.packet_metrics),
            "total_attached_images": sum(item["attached_image_count"] for item in self.packet_metrics),
            "total_reply_char_count": total_reply_char_count,
            "total_reply_word_count_est": total_reply_word_count_est,
            "packet_metrics": self.packet_metrics,
        }
        (self.session_dir / "summary.json").write_text(_json_dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual web-chat smoke test runner for MiniGrid.")
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--task",
        default=str(root / "mazes" / "validation_10" / "V01_empty_room.json"),
        help="Task JSON path.",
    )
    parser.add_argument(
        "--session-dir",
        default=None,
        help="Directory for exported packets and transcript. Defaults to /tmp/chat_smoke_<timestamp>.",
    )
    parser.add_argument(
        "--query-interval",
        type=int,
        default=0,
        help="Maximum number of env actions to execute from each pasted model reply. 0 means unlimited.",
    )
    parser.add_argument(
        "--allow-look",
        action="store_true",
        help="Allow `LOOK` as a chat-side control token to request a refreshed frame.",
    )
    parser.add_argument(
        "--history-images",
        type=int,
        default=2,
        help="How many prior frames to export with each packet.",
    )
    parser.add_argument(
        "--history-text-window",
        type=int,
        default=3,
        help="How many recent action summaries to include in the prompt.",
    )
    return parser.parse_args()


def _to_jsonable(value):
    """Recursively convert NumPy values and tuples into JSON-safe data."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _json_dumps(value, *, indent: int | None = None) -> str:
    return json.dumps(_to_jsonable(value), indent=indent)


def _save_contact_sheet(path: Path, *, prior_images: list[np.ndarray], current_image: np.ndarray) -> None:
    """Save a simple labeled contact sheet for packet debugging."""
    frames = [*prior_images, current_image]
    labels = [*[f"prior_{i}" for i in range(1, len(prior_images) + 1)], "current"]
    pil_frames = [Image.fromarray(frame).convert("RGB") for frame in frames]

    widths = [image.width for image in pil_frames]
    heights = [image.height for image in pil_frames]
    label_band = 24
    gap = 8
    sheet = Image.new(
        "RGB",
        (sum(widths) + gap * (len(pil_frames) - 1), max(heights) + label_band),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)

    x = 0
    for image, label in zip(pil_frames, labels):
        sheet.paste(image, (x, label_band))
        draw.text((x + 4, 4), label, fill=(20, 20, 20))
        x += image.width + gap

    sheet.save(path)


def main() -> None:
    args = parse_args()
    if args.query_interval < 0:
        raise ValueError("--query-interval must be >= 0")

    if args.session_dir:
        session_dir = args.session_dir
    else:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        session_dir = f"/tmp/chat_smoke_{stamp}"

    session = ChatSmokeSession(
        task_path=args.task,
        session_dir=session_dir,
        query_interval=args.query_interval,
        allow_look=args.allow_look,
        history_images=max(0, args.history_images),
        history_text_window=max(0, args.history_text_window),
    )
    session.start()

    print(f"Session directory: {session.session_dir}")
    print(f"Task: {session.spec.task_id}")
    print(
        "Commands while pasting replies: `/quit` to stop, `/packet` to re-export the current packet."
    )

    try:
        while not session.done:
            packet_dir = session.export_packet()
            print(f"\nPacket ready: {packet_dir}")
            print(
                "Attach the packet images to your chat UI, paste `user_message.md`, "
                "then paste the model reply here. Finish with an empty line."
            )
            reply = _read_multiline_reply().strip()

            if reply == "/quit":
                break
            if reply == "/packet":
                continue

            try:
                parsed = session.apply_reply(reply)
            except ValueError as exc:
                print(f"Parse error: {exc}")
                continue

            if session.done:
                break

            if parsed.requested_look:
                print("Model requested LOOK. A refreshed packet will be exported next.")
            else:
                print(
                    "Executed actions: "
                    + ", ".join(ACTION_NAMES[action] for action in parsed.actions)
                )

        status = "success" if session.success else "stopped"
        print(f"\nSession finished: {status}")
        print(f"Transcript: {session.transcript_path}")
    finally:
        session.close()


def _read_multiline_reply() -> str:
    lines: list[str] = []
    while True:
        try:
            line = input("> " if not lines else "")
        except EOFError:
            break
        if not line.strip():
            if lines:
                break
            continue
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
