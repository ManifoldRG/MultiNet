"""Regression tests for v2 schema fields and backend fidelity."""

from pathlib import Path
import sys

_v1_1_dir = str(Path(__file__).resolve().parent.parent)
if _v1_1_dir not in sys.path:
    sys.path.insert(0, _v1_1_dir)

from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import TaskValidator, compute_difficulty
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.backends.multigrid_backend import MultiGridBackend
from gridworld.scoring import compute_12d_score


def test_v2_schema_round_trip():
    spec = TaskSpecification.from_dict({
        "task_id": "v2_roundtrip",
        "seed": 7,
        "difficulty_tier": 2,
        "maze": {
            "dimensions": [8, 8],
            "walls": [],
            "start": [1, 1],
            "goal": [6, 6],
        },
        "mechanisms": {
            "keys": [{"id": "kR", "position": [2, 1], "color": "red"}],
            "doors": [{"id": "DR", "position": [4, 1], "requires_key": "red"}],
        },
        "goal": {"type": "reach_position", "target": [6, 6]},
        "dependency_chain": {
            "depth": 1,
            "sequence": [
                {"step": 1, "type": "key-door", "element": "kR", "unlocks": "DR"}
            ],
            "notation": "kR -> DR -> G",
        },
        "distractors": [
            {
                "type": "wrong_color_key",
                "element_id": "kY",
                "description": "No matching door",
            }
        ],
        "metadata": {"chain_pattern": "C1", "wall_topology": "open"},
        "max_steps": 50,
    })

    restored = TaskSpecification.from_dict(spec.to_dict())
    assert restored.dependency_chain is not None
    assert restored.dependency_chain.depth == 1
    assert restored.dependency_chain.sequence[0].element == "kR"
    assert restored.distractors is not None
    assert restored.distractors[0].element_id == "kY"
    assert restored.metadata == {"chain_pattern": "C1", "wall_topology": "open"}


def test_validator_does_not_recollect_consumed_key():
    spec = TaskSpecification.from_dict({
        "task_id": "consumed_key_no_recollect",
        "seed": 1,
        "difficulty_tier": 3,
        "maze": {
            "dimensions": [8, 3],
            "walls": [],
            "start": [1, 1],
            "goal": [6, 1],
        },
        "mechanisms": {
            "keys": [{"id": "k1", "position": [2, 1], "color": "red"}],
            "doors": [
                {"id": "d1", "position": [3, 1], "requires_key": "red"},
                {"id": "d2", "position": [5, 1], "requires_key": "red"},
            ],
        },
        "rules": {"key_consumption": True},
        "goal": {"type": "reach_position", "target": [6, 1]},
        "max_steps": 40,
    })

    is_beatable, _, _ = TaskValidator(spec).validate()
    assert is_beatable is False


def test_minigrid_respects_initial_switch_state():
    spec = TaskSpecification.from_dict({
        "task_id": "switch_initial_on",
        "seed": 5,
        "difficulty_tier": 2,
        "maze": {
            "dimensions": [8, 8],
            "walls": [],
            "start": [1, 1],
            "goal": [6, 6],
        },
        "mechanisms": {
            "switches": [
                {
                    "id": "s1",
                    "position": [2, 2],
                    "controls": ["g1"],
                    "switch_type": "toggle",
                    "initial_state": "on",
                }
            ],
            "gates": [{"id": "g1", "position": [4, 4], "initial_state": "closed"}],
        },
        "goal": {"type": "reach_position", "target": [6, 6]},
        "max_steps": 40,
    })

    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    _, state, _ = backend.reset(seed=5)

    assert "s1" in state.active_switches
    assert "g1" in state.open_gates


def test_multigrid_backend_preserves_mechanism_types():
    spec = TaskSpecification.from_dict({
        "task_id": "multigrid_fidelity",
        "seed": 9,
        "difficulty_tier": 4,
        "maze": {
            "dimensions": [8, 8],
            "walls": [[3, 3]],
            "start": [1, 1],
            "goal": [6, 6],
        },
        "mechanisms": {
            "keys": [{"id": "k1", "position": [2, 1], "color": "red"}],
            "doors": [{"id": "d1", "position": [3, 1], "requires_key": "red"}],
            "switches": [
                {
                    "id": "s1",
                    "position": [2, 2],
                    "controls": ["g1"],
                    "switch_type": "toggle",
                    "initial_state": "off",
                }
            ],
            "gates": [{"id": "g1", "position": [4, 2], "initial_state": "closed"}],
            "blocks": [{"id": "b1", "position": [2, 3], "color": "grey"}],
            "hazards": [{"id": "h1", "position": [2, 4], "hazard_type": "lava"}],
            "teleporters": [
                {"id": "tp1", "position_a": [5, 1], "position_b": [6, 4], "bidirectional": True}
            ],
        },
        "goal": {"type": "reach_position", "target": [6, 6]},
        "max_steps": 60,
    })

    backend = MultiGridBackend(tiling="square", render_mode="rgb_array")
    backend.configure(spec)
    _, state, _ = backend.reset(seed=9)

    objects = backend.env.state.objects
    assert objects["k1"].obj_type == "key"
    assert objects["d1"].obj_type == "door"
    assert objects["s1"].obj_type == "switch"
    assert objects["g1"].obj_type == "gate"
    assert objects["h1"].obj_type == "hazard"
    assert objects["tp1_a"].obj_type == "teleporter"
    assert objects["tp1_b"].obj_type == "teleporter"
    assert "wall_3_3" in objects
    assert state.block_positions["b1"]


def test_mechanism_necessity_detects_bypassable_door():
    spec = TaskSpecification.from_dict({
        "task_id": "unnecessary_door",
        "seed": 11,
        "difficulty_tier": 2,
        "maze": {
            "dimensions": [8, 8],
            "walls": [[3, 2], [3, 4]],
            "start": [1, 3],
            "goal": [6, 3],
        },
        "mechanisms": {
            "keys": [{"id": "kR", "position": [2, 1], "color": "red"}],
            "doors": [{"id": "DR", "position": [3, 3], "requires_key": "red"}],
        },
        "goal": {"type": "reach_position", "target": [6, 3]},
        "max_steps": 40,
    })

    violations = TaskValidator(spec).validate_mechanism_necessity()
    assert any("kR" in violation for violation in violations)


def test_chain_ordering_passes_for_validation_v6():
    path = Path(__file__).resolve().parent.parent / "mazes" / "validation_10" / "V06_chain_ks.json"
    spec = TaskSpecification.from_json(str(path))
    assert TaskValidator(spec).validate_chain_ordering() is True


def test_distractor_safety_passes_for_validation_tasks():
    base = Path(__file__).resolve().parent.parent / "mazes" / "validation_10"
    for name in ("V09_distractor_simple.json", "V10_distractor_chain.json"):
        spec = TaskSpecification.from_json(str(base / name))
        assert TaskValidator(spec).validate_distractor_safety() == []


def test_fragility_unbreakable_for_empty_room():
    path = Path(__file__).resolve().parent.parent / "mazes" / "validation_10" / "V01_empty_room.json"
    spec = TaskSpecification.from_json(str(path))
    report = TaskValidator(spec).compute_fragility()
    assert report.min_steps_to_break == -1
    assert report.is_fragile is False


def test_fragility_detects_single_bad_block_push():
    spec = TaskSpecification.from_dict({
        "task_id": "fragile_block",
        "seed": 12,
        "difficulty_tier": 4,
        "maze": {
            "dimensions": [8, 6],
            "walls": [[4, 1], [5, 1], [6, 1], [2, 2], [4, 2], [5, 2], [6, 2], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4]],
            "start": [3, 1],
            "goal": [6, 3]
        },
        "mechanisms": {
            "blocks": [{"id": "b1", "position": [3, 2], "color": "grey"}]
        },
        "goal": {"type": "reach_position", "target": [6, 3]},
        "max_steps": 40
    })
    assert TaskValidator(spec).validate()[0] is True
    report = TaskValidator(spec).compute_fragility()
    assert report.min_steps_to_break == 1
    assert report.is_fragile is True
    assert any("push:" in step for step in report.breaking_sequences[0])


def test_12d_score_matches_validation_pair_expectations():
    base = Path(__file__).resolve().parent.parent / "mazes" / "validation_10"
    v6 = TaskSpecification.from_json(str(base / "V06_chain_ks.json"))
    v7 = TaskSpecification.from_json(str(base / "V07_chain_sk.json"))
    v4 = TaskSpecification.from_json(str(base / "V04_single_key.json"))
    v9 = TaskSpecification.from_json(str(base / "V09_distractor_simple.json"))

    score_v6 = compute_12d_score(v6)
    score_v7 = compute_12d_score(v7)
    score_v4 = compute_12d_score(v4)
    score_v9 = compute_12d_score(v9)

    assert len(score_v6.dimensions) == 12
    assert all(value >= 0 for value in score_v6.dimensions)
    assert score_v6.dimensions[8] == score_v7.dimensions[8]
    assert score_v6.dimensions[4] == score_v7.dimensions[4]
    assert score_v6.dimensions[5] == score_v7.dimensions[5]
    assert score_v4.dimensions[6] == 0
    assert score_v9.dimensions[6] == 2
    assert score_v9.dimensions[7] > score_v4.dimensions[7]


def test_validation_mazes_pass_plan_checks():
    base = Path(__file__).resolve().parent.parent / "mazes" / "validation_10"
    for name in [f"V0{i}_{suffix}.json" for i, suffix in []]:
        pass
    for name in [
        "V01_empty_room.json",
        "V02_winding_corridor.json",
        "V03_multi_path.json",
        "V04_single_key.json",
        "V05_single_switch.json",
        "V06_chain_ks.json",
        "V07_chain_sk.json",
        "V08_chain_kk.json",
        "V09_distractor_simple.json",
        "V10_distractor_chain.json",
    ]:
        spec = TaskSpecification.from_json(str(base / name))
        validator = TaskValidator(spec)
        assert validator.validate_mechanism_necessity() == []
        assert validator.validate_chain_ordering() is True
        assert validator.validate_distractor_safety() == []


def test_validation_mazes_match_authored_dimensions_and_shared_layouts():
    base = Path(__file__).resolve().parent.parent / "mazes" / "validation_10"

    v01 = TaskSpecification.from_json(str(base / "V01_empty_room.json"))
    v02 = TaskSpecification.from_json(str(base / "V02_winding_corridor.json"))
    v03 = TaskSpecification.from_json(str(base / "V03_multi_path.json"))
    v04 = TaskSpecification.from_json(str(base / "V04_single_key.json"))
    v05 = TaskSpecification.from_json(str(base / "V05_single_switch.json"))
    v06 = TaskSpecification.from_json(str(base / "V06_chain_ks.json"))
    v07 = TaskSpecification.from_json(str(base / "V07_chain_sk.json"))
    v08 = TaskSpecification.from_json(str(base / "V08_chain_kk.json"))
    v09 = TaskSpecification.from_json(str(base / "V09_distractor_simple.json"))
    v10 = TaskSpecification.from_json(str(base / "V10_distractor_chain.json"))

    assert v01.maze.dimensions == (8, 8)
    assert v02.maze.dimensions == (20, 8)
    assert v03.maze.dimensions == (12, 12)
    assert v04.maze.dimensions == (14, 12)
    assert v05.maze.dimensions == (14, 12)
    assert v06.maze.dimensions == (14, 12)
    assert v07.maze.dimensions == (14, 12)
    assert v08.maze.dimensions == (14, 12)
    assert v09.maze.dimensions == (16, 12)
    assert v10.maze.dimensions == (16, 12)

    assert v06.maze.walls == v07.maze.walls == v08.maze.walls
    assert v06.maze.start == v07.maze.start == v08.maze.start
    assert v06.maze.goal == v07.maze.goal == v08.maze.goal

    assert len(v09.distractors or []) == 2
    assert len(v10.distractors or []) == 1
    assert len(v09.maze.walls) < (v09.maze.dimensions[0] - 2) * (v09.maze.dimensions[1] - 2)
    assert len(v10.maze.walls) < (v10.maze.dimensions[0] - 2) * (v10.maze.dimensions[1] - 2)


def test_backtracking_detection_matches_plan_examples():
    base = Path(__file__).resolve().parent.parent / "mazes" / "validation_10"
    for name in ["V01_empty_room.json", "V02_winding_corridor.json", "V03_multi_path.json"]:
        report = compute_difficulty(TaskSpecification.from_json(str(base / name)))
        assert report.backtrack_count == 0
        assert report.optimal_path
    v6 = compute_difficulty(TaskSpecification.from_json(str(base / "V06_chain_ks.json")))
    assert v6.backtrack_count > 0


def test_distractor_safety_detects_bad_block_push():
    spec = TaskSpecification.from_dict({
        "task_id": "bad_block_distractor",
        "seed": 14,
        "difficulty_tier": 4,
        "maze": {
            "dimensions": [8, 6],
            "walls": [[4, 1], [5, 1], [6, 1], [2, 2], [4, 2], [5, 2], [6, 2], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4]],
            "start": [3, 1],
            "goal": [6, 3]
        },
        "mechanisms": {
            "blocks": [{"id": "bD", "position": [3, 2], "color": "grey"}]
        },
        "goal": {"type": "reach_position", "target": [6, 3]},
        "distractors": [
            {"type": "spatial_block", "element_id": "bD", "description": "Can be pushed into the only corridor."}
        ],
        "max_steps": 50
    })

    assert TaskValidator(spec).validate()[0] is True
    violations = TaskValidator(spec).validate_distractor_safety()
    assert any("bD" in violation for violation in violations)


def test_scoring_plan_ordering_properties():
    base = Path(__file__).resolve().parent.parent / "mazes" / "validation_10"
    v1 = compute_12d_score(TaskSpecification.from_json(str(base / "V01_empty_room.json")))
    v4 = compute_12d_score(TaskSpecification.from_json(str(base / "V04_single_key.json")))
    v6 = compute_12d_score(TaskSpecification.from_json(str(base / "V06_chain_ks.json")))
    v7 = compute_12d_score(TaskSpecification.from_json(str(base / "V07_chain_sk.json")))
    v8 = compute_12d_score(TaskSpecification.from_json(str(base / "V08_chain_kk.json")))
    v9 = compute_12d_score(TaskSpecification.from_json(str(base / "V09_distractor_simple.json")))
    v10 = compute_12d_score(TaskSpecification.from_json(str(base / "V10_distractor_chain.json")))

    assert v1.composite == sum(d * w for d, w in zip(v1.dimensions, v1.weights))
    assert v6.dimensions[4] == v7.dimensions[4]
    assert v6.dimensions[5] == v7.dimensions[5]
    assert v6.dimensions[5] != v8.dimensions[5]
    assert v4.dimensions[6] == 0 and v4.dimensions[7] == 0
    assert v9.dimensions[6] == 2 and v9.dimensions[7] > 0
    assert v1.composite < v4.composite < v10.composite
