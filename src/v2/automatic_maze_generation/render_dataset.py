# render_dataset.py
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


CELL = 40  # pixels-ish via figure scale


def _extract_payload_fields(payload: dict):
    maze = payload["maze"]
    mechs = payload.get("mechanisms", {})

    width, height = maze["dimensions"]
    walls = {tuple(w) for w in maze["walls"]}
    start = tuple(maze["start"])
    goal = tuple(maze["goal"])

    keys = mechs.get("keys", [])
    doors = mechs.get("doors", [])
    switches = mechs.get("switches", [])
    gates = mechs.get("gates", [])

    return width, height, walls, start, goal, keys, doors, switches, gates



def _color_to_facecolor(name: str) -> str:
    mapping = {
        "red": "#e74c3c",
        "blue": "#3498db",
        "green": "#2ecc71",
        "yellow": "#f1c40f",
        "purple": "#9b59b6",
        "orange": "#e67e22",
    }
    return mapping.get(name.lower(), "#95a5a6")


def _draw_centered_text(ax, x: int, y: int, height: int, text: str, fontsize: int = 10, color: str = "black"):
    ax.text(
        x + 0.5,
        height - 1 - y + 0.5,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight="bold",
    )


def _draw_key(ax, x: int, y: int, height: int, color_name: str):
    face = _color_to_facecolor(color_name)
    cy = height - 1 - y + 0.5

    # colored circle badge
    ax.add_patch(Circle((x + 0.5, cy), 0.28, facecolor=face, edgecolor="black", linewidth=1.0))
    # key icon / fallback letter
    ax.text(
        x + 0.5,
        cy,
        "⚷",   # if this glyph looks odd in your env, replace with "K"
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        fontweight="bold",
    )


def _draw_door(ax, x: int, y: int, height: int, color_name: str):
    face = _color_to_facecolor(color_name)
    by = height - 1 - y

    # colored inner door rectangle
    ax.add_patch(
        Rectangle(
            (x + 0.18, by + 0.12),
            0.64,
            0.76,
            facecolor=face,
            edgecolor="black",
            linewidth=1.0,
        )
    )
    # small doorknob
    ax.add_patch(Circle((x + 0.68, by + 0.5), 0.04, facecolor="white", edgecolor="white"))


def _draw_switch(ax, x: int, y: int, height: int, label: str):
    by = height - 1 - y

    ax.add_patch(
        Rectangle(
            (x + 0.15, by + 0.2),
            0.7,
            0.6,
            facecolor="#dfe6e9",
            edgecolor="black",
            linewidth=1.0,
        )
    )
    ax.text(
        x + 0.5,
        by + 0.5,
        label,
        ha="center",
        va="center",
        fontsize=9,
        color="black",
        fontweight="bold",
    )


def _draw_gate(ax, x: int, y: int, height: int, label: str):
    by = height - 1 - y

    # gate bars
    for dx in [0.22, 0.38, 0.54, 0.70]:
        ax.plot([x + dx, x + dx], [by + 0.15, by + 0.85], color="black", linewidth=1.4)
    ax.plot([x + 0.18, x + 0.74], [by + 0.18, by + 0.18], color="black", linewidth=1.4)
    ax.plot([x + 0.18, x + 0.74], [by + 0.82, by + 0.82], color="black", linewidth=1.4)

    ax.text(
        x + 0.5,
        by + 0.5,
        label,
        ha="center",
        va="center",
        fontsize=8,
        color="black",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.8),
    )




def _extract_optimal_path(payload: dict):
    validation = payload.get("validation", {})
    return [tuple(p) for p in validation.get("optimal_path", [])]






def _draw_optimal_path(ax, path, height: int):
    if not path:
        return

    xs = [x + 0.5 for x, y in path]
    ys = [height - 1 - y + 0.5 for x, y in path]

    ax.plot(
        xs,
        ys,
        linewidth=3.0,
        alpha=0.45,
        zorder=2,
    )

    # mark start of path a little more clearly
    ax.scatter(
        [xs[0]],
        [ys[0]],
        s=35,
        alpha=0.7,
        zorder=3,
    )



def render_maze_payload(payload: dict, output_path: Path) -> None:
    width, height, walls, start, goal, keys, doors, switches, gates = _extract_payload_fields(payload)
    optimal_path = _extract_optimal_path(payload)

    fig_w = max(6, width * 0.55)
    fig_h = max(4, height * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # base grid
    for x in range(width):
        for y in range(height):
            is_wall = (x, y) in walls
            facecolor = "black" if is_wall else "white"
            ax.add_patch(
                Rectangle(
                    (x, height - 1 - y),
                    1,
                    1,
                    facecolor=facecolor,
                    edgecolor="lightgray",
                    linewidth=0.8,
                    zorder=0,
                )
            )

    # path overlay first, so icons remain visible above it
    _draw_optimal_path(ax, optimal_path, height)

    # start / goal
    sx, sy = start
    gx, gy = goal
    ax.add_patch(Rectangle((sx, height - 1 - sy), 1, 1, facecolor="#c8f7c5", edgecolor="black", linewidth=1.2, zorder=4))
    ax.add_patch(Rectangle((gx, height - 1 - gy), 1, 1, facecolor="#f7d6c5", edgecolor="black", linewidth=1.2, zorder=4))
    _draw_centered_text(ax, sx, sy, height, "S", fontsize=11)
    _draw_centered_text(ax, gx, gy, height, "G", fontsize=11)

    # keys
    for key in keys:
        x, y = key["position"]
        color_name = key.get("color", "gray")
        _draw_key(ax, x, y, height, color_name)

    # doors
    for door in doors:
        x, y = door["position"]
        color_name = door.get("requires_key", "gray")
        _draw_door(ax, x, y, height, color_name)

    # switches
    for sw in switches:
        x, y = sw["position"]
        _draw_switch(ax, x, y, height, "S")

    # gates
    for gate in gates:
        x, y = gate["position"]
        _draw_gate(ax, x, y, height, "G")

    title = payload.get("task_id", output_path.stem)
    ax.set_title(title)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)




def main() -> None:
    input_dir = Path("generated_mazes")
    # input_dir = Path("nlu_benchmark/sample mazes")
    output_dir = input_dir / "pngs"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(p for p in input_dir.glob("*.json") if p.name != "manifest.json")
    if not json_files:
        print("No maze JSON files found in generated_mazes/")
        return

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            payload = json.load(f)

        out_path = output_dir / f"{jf.stem}.png"
        render_maze_payload(payload, out_path)
        print(f"[OK] rendered {out_path.name}")

    print(f"\nRendered {len(json_files)} PNGs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()