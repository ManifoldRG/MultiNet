# Appendix E: Test Cases and Episode Walkthroughs

**Status:** Implementation-Ready

## E.1 Overview

This document provides:
1. **Unit test specifications** for core MultiGrid components
2. **Episode walkthroughs** demonstrating expected behavior
3. **Cross-tiling equivalence tests** ensuring consistent task semantics
4. **Edge case documentation** for implementation validation

## E.2 Unit Test Specifications

### E.2.1 Tiling Graph Generation Tests

```python
# test_tiling_generation.py

import pytest
from multigrid.tilings import SquareTiling, HexTiling, TriangleTiling

class TestTilingGeneration:
    """Tests for tiling graph generation."""

    @pytest.mark.parametrize("tiling_class,expected_dirs", [
        (SquareTiling, 4),
        (HexTiling, 6),
        (TriangleTiling, 3),
    ])
    def test_direction_count(self, tiling_class, expected_dirs):
        """Each tiling type has correct number of directions."""
        tiling = tiling_class()
        assert len(tiling.directions) == expected_dirs

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_cell_count(self, tiling_class):
        """Grid generates expected number of cells."""
        tiling = tiling_class()
        cells = tiling.generate_graph(width=10, height=8, seed=42)

        if tiling_class == SquareTiling:
            assert len(cells) == 80  # 10 * 8
        elif tiling_class == HexTiling:
            assert len(cells) == 80  # Rectangular hex grid
        elif tiling_class == TriangleTiling:
            assert len(cells) == 80  # Same as square for now

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_boundary_cells_have_fewer_neighbors(self, tiling_class):
        """Cells at grid boundary have fewer neighbors than interior."""
        tiling = tiling_class()
        cells = tiling.generate_graph(width=5, height=5, seed=0)

        # Corner cells should have minimum neighbors
        # Interior cells should have maximum neighbors
        neighbor_counts = [len(c.neighbors) for c in cells.values()]

        assert min(neighbor_counts) < max(neighbor_counts)

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_adjacency_symmetry(self, tiling_class):
        """If A neighbors B, then B neighbors A."""
        tiling = tiling_class()
        cells = tiling.generate_graph(width=5, height=5, seed=0)

        for cell_id, cell in cells.items():
            for direction, neighbor_id in cell.neighbors.items():
                neighbor = cells[neighbor_id]
                # Neighbor should have some direction pointing back
                assert cell_id in neighbor.neighbors.values(), \
                    f"Asymmetric: {cell_id} -> {neighbor_id} but not reverse"

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_seed_determinism(self, tiling_class):
        """Same seed produces identical graph."""
        tiling1 = tiling_class()
        tiling2 = tiling_class()

        cells1 = tiling1.generate_graph(10, 10, seed=12345)
        cells2 = tiling2.generate_graph(10, 10, seed=12345)

        assert set(cells1.keys()) == set(cells2.keys())
        for cell_id in cells1:
            assert cells1[cell_id].neighbors == cells2[cell_id].neighbors
```

### E.2.2 Coordinate Conversion Tests

```python
# test_coordinates.py

import pytest
import math
from multigrid.tilings import SquareTiling, HexTiling, TriangleTiling

class TestCoordinateConversion:
    """Tests for canonical <-> cell coordinate conversion."""

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_canonical_roundtrip_center(self, tiling_class):
        """Converting to cell and back gives approximately same position."""
        tiling = tiling_class()
        tiling.generate_graph(10, 10, seed=0)

        # Test center of grid
        x, y = 0.5, 0.5
        cell_id = tiling.canonical_to_cell(x, y)
        x2, y2 = tiling.cell_to_canonical(cell_id)

        # Should be within half a cell width
        assert abs(x - x2) < 0.15
        assert abs(y - y2) < 0.15

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_canonical_corners(self, tiling_class):
        """Corner positions map to boundary cells."""
        tiling = tiling_class()
        tiling.generate_graph(10, 10, seed=0)

        corners = [(0.01, 0.01), (0.99, 0.01), (0.01, 0.99), (0.99, 0.99)]

        for x, y in corners:
            cell_id = tiling.canonical_to_cell(x, y)
            assert cell_id in tiling.cells, f"Corner ({x},{y}) mapped to invalid cell"

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_cell_positions_unique(self, tiling_class):
        """Each cell has a unique canonical position."""
        tiling = tiling_class()
        tiling.generate_graph(10, 10, seed=0)

        positions = set()
        for cell_id in tiling.cells:
            pos = tiling.cell_to_canonical(cell_id)
            # Round to avoid floating point issues
            pos_rounded = (round(pos[0], 6), round(pos[1], 6))
            assert pos_rounded not in positions, f"Duplicate position for {cell_id}"
            positions.add(pos_rounded)
```

### E.2.3 Distance Computation Tests

```python
# test_distance.py

import pytest
from multigrid.tilings import SquareTiling, HexTiling, TriangleTiling

class TestDistance:
    """Tests for distance computation."""

    def test_square_manhattan_distance(self):
        """Square grid distance equals Manhattan distance."""
        tiling = SquareTiling()
        tiling.generate_graph(10, 10, seed=0)

        # Cells 3 apart horizontally
        d = tiling.distance("sq_5_2", "sq_5_5")
        assert d == 3

        # Cells 2 apart vertically
        d = tiling.distance("sq_3_5", "sq_5_5")
        assert d == 2

        # Diagonal: Manhattan = 4
        d = tiling.distance("sq_3_3", "sq_5_5")
        assert d == 4

    def test_hex_distance(self):
        """Hex grid distance uses hex metric."""
        tiling = HexTiling()
        tiling.generate_graph(10, 10, seed=0)

        # Adjacent cells are distance 1
        for cell_id, cell in tiling.cells.items():
            for neighbor_id in cell.neighbors.values():
                assert tiling.distance(cell_id, neighbor_id) == 1

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_distance_zero_to_self(self, tiling_class):
        """Distance from cell to itself is 0."""
        tiling = tiling_class()
        tiling.generate_graph(5, 5, seed=0)

        for cell_id in tiling.cells:
            assert tiling.distance(cell_id, cell_id) == 0

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_distance_symmetry(self, tiling_class):
        """Distance is symmetric."""
        tiling = tiling_class()
        cells = tiling.generate_graph(5, 5, seed=0)

        cell_ids = list(cells.keys())[:10]  # Sample 10 cells
        for i, id1 in enumerate(cell_ids):
            for id2 in cell_ids[i+1:]:
                assert tiling.distance(id1, id2) == tiling.distance(id2, id1)
```

### E.2.4 Action Execution Tests

```python
# test_actions.py

import pytest
from multigrid.env import MultiGridEnv, Action
from multigrid.tilings import SquareTiling, HexTiling

class TestActions:
    """Tests for action execution."""

    @pytest.fixture
    def simple_task(self):
        """Simple task spec for testing."""
        return {
            "task_id": "test_001",
            "seed": 42,
            "scene": {
                "bounds": {"width": 1.0, "height": 1.0},
                "objects": [
                    {
                        "id": "cube_red",
                        "type": "movable",
                        "color": "red",
                        "position": {"x": 0.5, "y": 0.5},
                        "size": 0.1
                    }
                ],
                "agent": {
                    "position": {"x": 0.2, "y": 0.2},
                    "facing": 0
                }
            },
            "goal": {
                "predicate": "object_in_zone",
                "object_id": "cube_red",
                "zone_id": "zone_blue"
            },
            "limits": {"max_steps": 100},
            "tiling": {"type": "square", "grid_size": {"width": 10, "height": 10}}
        }

    def test_forward_movement(self, simple_task):
        """Agent moves forward in facing direction."""
        env = MultiGridEnv(simple_task, tiling="square")
        obs, info = env.reset(seed=42)

        initial_cell = env.state.agent.cell_id
        initial_facing = env.state.agent.facing

        obs, reward, term, trunc, info = env.step(Action.FORWARD)

        # Agent should have moved
        assert env.state.agent.cell_id != initial_cell or info.get("invalid_action")

    def test_turn_changes_facing(self, simple_task):
        """Turn actions change facing without moving."""
        env = MultiGridEnv(simple_task, tiling="square")
        env.reset(seed=42)

        initial_cell = env.state.agent.cell_id
        initial_facing = env.state.agent.facing

        env.step(Action.TURN_RIGHT)

        assert env.state.agent.cell_id == initial_cell  # Didn't move
        assert env.state.agent.facing == (initial_facing + 1) % 4  # Facing changed

    def test_invalid_move_into_wall(self, simple_task):
        """Moving into boundary returns invalid_action."""
        # Modify task to put agent at corner facing wall
        simple_task["scene"]["agent"]["position"] = {"x": 0.05, "y": 0.05}
        simple_task["scene"]["agent"]["facing"] = 0  # Facing north (into wall)

        env = MultiGridEnv(simple_task, tiling="square")
        env.reset(seed=42)

        obs, reward, term, trunc, info = env.step(Action.FORWARD)

        assert info.get("invalid_action") == True

    def test_pickup_object(self, simple_task):
        """Agent can pick up adjacent objects."""
        # Position agent next to object
        simple_task["scene"]["agent"]["position"] = {"x": 0.4, "y": 0.5}
        simple_task["scene"]["agent"]["facing"] = 1  # Facing east (toward object)

        env = MultiGridEnv(simple_task, tiling="square")
        env.reset(seed=42)

        assert env.state.agent.holding is None

        # Move forward to object's cell
        env.step(Action.FORWARD)

        # Pick up
        env.step(Action.PICKUP)

        assert env.state.agent.holding is not None
        assert env.state.agent.holding.id == "cube_red"
```

## E.3 Episode Walkthroughs

### E.3.1 Walkthrough 1: Simple Navigation (Square Grid)

**Task**: Move agent from start to goal cell.

**Initial State**:
```
+---+---+---+---+---+
|   |   |   |   |   |
+---+---+---+---+---+
|   | A |   |   |   |   A = Agent (facing east)
+---+---+---+---+---+
|   |   |   |   |   |
+---+---+---+---+---+
|   |   |   | G |   |   G = Goal zone
+---+---+---+---+---+
|   |   |   |   |   |
+---+---+---+---+---+
```

**Optimal Solution** (5 actions):
1. `FORWARD` - Move east to (1, 2)
2. `FORWARD` - Move east to (1, 3)
3. `TURN_RIGHT` - Face south
4. `FORWARD` - Move south to (2, 3)
5. `FORWARD` - Move south to (3, 3) - **GOAL REACHED**

**Expected State Sequence**:

| Step | Agent Cell | Facing | Action | Result |
|------|-----------|--------|--------|--------|
| 0 | (1, 1) | east | - | Initial |
| 1 | (1, 2) | east | FORWARD | Moved |
| 2 | (1, 3) | east | FORWARD | Moved |
| 3 | (1, 3) | south | TURN_RIGHT | Turned |
| 4 | (2, 3) | south | FORWARD | Moved |
| 5 | (3, 3) | south | FORWARD | **Goal!** |

**Metrics**:
- Success: True
- Steps taken: 5
- Optimal steps: 5
- Efficiency: 1.0
- Invalid actions: 0

### E.3.2 Walkthrough 2: Object Manipulation (Hex Grid)

**Task**: Push red cube into blue zone.

**Initial State** (hex grid, pointy-top):
```
     _____         _____
    /     \       /     \
   /   A   \_____/       \
   \  -->  /     \       /
    \_____/  [R]  \_____/
    /     \       /     \
   /       \_____/  (B)  \     A = Agent (facing SE)
   \       /     \       /     [R] = Red cube
    \_____/       \_____/      (B) = Blue zone
```

**Solution** (4 actions):
1. `FORWARD` - Move to cube's cell
2. `PUSH` - Push cube southeast into goal zone
3. **GOAL REACHED** (cube in zone)

**Expected Behavior**:
- Push moves cube in agent's facing direction
- Agent stays in place after push
- Zone detection triggers when cube center in zone bounds

### E.3.3 Walkthrough 3: Complex Navigation (Triangle Grid)

**Task**: Navigate to goal avoiding obstacles.

**Initial State** (triangle grid):
```
Row 0:  ▲   ▽   ▲   ▽   ▲
Row 1:  ▽   ▲   ▽   ▲   ▽
Row 2:  ▲  [W]  ▲  [W]  ▲     [W] = Wall
Row 3:  ▽   ▲   ▽   ▲   ▽
Row 4:  ▲   ▽  [G]  ▽   ▲     [G] = Goal

Agent starts at (0, 2), facing "vertical"
```

**Key Challenge**:
- Triangle grid has only 3 neighbors per cell
- Must navigate around walls using limited movement options
- Agent orientation affects available directions

**Solution Strategy**:
1. Move down-left to avoid first wall
2. Continue diagonally
3. Navigate to goal cell

### E.3.4 Walkthrough 4: Cross-Tiling Equivalence

**Test**: Same task should be solvable on all tilings.

**Canonical Task Spec**:
```json
{
  "task_id": "equivalence_test_001",
  "scene": {
    "objects": [
      {"id": "target", "position": {"x": 0.2, "y": 0.2}},
      {"id": "goal", "type": "zone", "position": {"x": 0.8, "y": 0.8}}
    ],
    "agent": {"position": {"x": 0.1, "y": 0.1}}
  },
  "goal": {"predicate": "object_in_zone", "object_id": "target", "zone_id": "goal"}
}
```

**Expected Outcomes**:

| Tiling | Solvable | Min Steps (approx) | Notes |
|--------|----------|-------------------|-------|
| Square | Yes | 12-15 | Direct path |
| Hex | Yes | 10-13 | More direct diagonal |
| Triangle | Yes | 15-20 | Limited movement |

**Verification**:
- All tilings should succeed
- State correspondence at goal: object position within zone bounds
- Metrics comparable within tiling-appropriate ranges

## E.4 Edge Case Tests

### E.4.1 Boundary Conditions

```python
# test_edge_cases.py

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_agent_at_corner(self):
        """Agent at corner has limited movement options."""
        task = create_task_with_agent_at(position=(0.01, 0.01))
        env = MultiGridEnv(task, tiling="square")
        env.reset()

        # Only 2 directions should be valid (east and south)
        cell = env.state.agent.cell_id
        neighbors = env.tiling.cells[cell].neighbors
        assert len(neighbors) == 2

    def test_push_at_boundary(self):
        """Cannot push object off grid."""
        task = create_task_with_object_at_boundary()
        env = MultiGridEnv(task, tiling="square")
        env.reset()

        # Position agent to push object off grid
        # ... setup ...

        obs, reward, term, trunc, info = env.step(Action.PUSH)
        assert info["invalid_action"] == True

    def test_very_large_grid(self):
        """Large grids don't cause performance issues."""
        import time

        task = create_task(grid_size=100)
        env = MultiGridEnv(task, tiling="square")

        start = time.time()
        env.reset()
        reset_time = time.time() - start

        assert reset_time < 1.0  # Should reset in under 1 second

        start = time.time()
        for _ in range(100):
            env.step(Action.FORWARD)
        step_time = time.time() - start

        assert step_time < 0.5  # 100 steps in under 0.5 seconds

    def test_seed_zero(self):
        """Seed 0 is valid and produces deterministic results."""
        env1 = MultiGridEnv(task, tiling="square")
        env2 = MultiGridEnv(task, tiling="square")

        obs1, _ = env1.reset(seed=0)
        obs2, _ = env2.reset(seed=0)

        assert (obs1 == obs2).all()

    def test_max_steps_truncation(self):
        """Episode truncates at max_steps."""
        task = create_task(max_steps=5)
        env = MultiGridEnv(task, tiling="square")
        env.reset()

        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(Action.WAIT)

        assert truncated == True
        assert terminated == False  # Goal not reached
```

### E.4.2 Object Interaction Edge Cases

```python
class TestObjectInteractions:
    """Tests for object interaction edge cases."""

    def test_pickup_while_holding(self):
        """Cannot pick up when already holding object."""
        env = setup_env_with_agent_holding_object()

        obs, reward, term, trunc, info = env.step(Action.PICKUP)
        assert info["invalid_action"] == True

    def test_drop_with_nothing(self):
        """Cannot drop when not holding anything."""
        env = setup_env_with_empty_hands()

        obs, reward, term, trunc, info = env.step(Action.DROP)
        assert info["invalid_action"] == True

    def test_push_nothing(self):
        """Pushing empty cell is invalid."""
        env = setup_env_facing_empty_cell()

        obs, reward, term, trunc, info = env.step(Action.PUSH)
        assert info["invalid_action"] == True

    def test_push_chain(self):
        """Pushing object into another object fails."""
        env = setup_env_with_adjacent_objects()

        obs, reward, term, trunc, info = env.step(Action.PUSH)
        assert info["invalid_action"] == True  # Cannot push into occupied cell
```

### E.4.3 Zone Computation Edge Cases

```python
class TestZones:
    """Tests for zone-related edge cases."""

    def test_zone_at_boundary(self):
        """Zone at grid boundary is correctly computed."""
        tiling = SquareTiling()
        tiling.generate_graph(10, 10, seed=0)

        # Zone at corner with radius 2
        zone_cells = compute_zone_cells(
            center=(0, 0), radius=2, width=10, height=10
        )

        # Should only include cells within bounds
        for cell_id in zone_cells:
            assert cell_id in tiling.cells

    def test_zone_radius_zero(self):
        """Radius 0 zone contains only center cell."""
        zone_cells = compute_zone_cells(
            center=(5, 5), radius=0, width=10, height=10
        )

        assert len(zone_cells) == 1

    def test_consecutive_steps_in_zone(self):
        """Goal requires N consecutive steps in zone."""
        task = create_task(consecutive_steps=3)
        env = MultiGridEnv(task)
        env.reset()

        # Move object into zone
        # ... actions to get object in zone ...

        # Should not succeed immediately
        assert not env.state.check_goal()

        # Wait 2 more steps
        env.step(Action.WAIT)
        assert not env.state.check_goal()

        env.step(Action.WAIT)
        assert env.state.check_goal()  # Now 3 consecutive steps
```

## E.5 Cross-Tiling Test Matrix

The following tests ensure consistent behavior across all tilings:

| Test Case | Square | Hex | Triangle | Expected |
|-----------|--------|-----|----------|----------|
| Empty grid navigation | ✓ | ✓ | ✓ | Agent reaches goal |
| Single obstacle avoidance | ✓ | ✓ | ✓ | Agent navigates around |
| Object pickup/drop | ✓ | ✓ | ✓ | Object state changes |
| Object push | ✓ | ✓ | ✓ | Object moves in facing dir |
| Zone detection | ✓ | ✓ | ✓ | Goal triggered when in zone |
| Boundary collision | ✓ | ✓ | ✓ | Invalid action returned |
| Max steps truncation | ✓ | ✓ | ✓ | Episode truncates |
| Deterministic reset | ✓ | ✓ | ✓ | Same seed = same state |

## E.6 Performance Benchmarks

```python
# test_performance.py

import pytest
import time

class TestPerformance:
    """Performance benchmark tests."""

    @pytest.mark.parametrize("grid_size", [10, 25, 50])
    @pytest.mark.parametrize("tiling", ["square", "hex", "triangle"])
    def test_reset_time(self, grid_size, tiling):
        """Reset should complete within time budget."""
        task = create_task(grid_size=grid_size)
        env = MultiGridEnv(task, tiling=tiling)

        times = []
        for _ in range(10):
            start = time.time()
            env.reset()
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)

        # Soft guideline: < 100ms for small grids, < 500ms for large
        if grid_size <= 25:
            assert avg_time < 0.1
        else:
            assert avg_time < 0.5

    @pytest.mark.parametrize("tiling", ["square", "hex", "triangle"])
    def test_step_throughput(self, tiling):
        """Step should achieve target throughput."""
        task = create_task(grid_size=20)
        env = MultiGridEnv(task, tiling=tiling)
        env.reset()

        start = time.time()
        for _ in range(1000):
            env.step(Action.TURN_RIGHT)
        elapsed = time.time() - start

        steps_per_second = 1000 / elapsed
        assert steps_per_second > 1000  # At least 1000 steps/sec
```

## E.7 Regression Test Suite

```python
# test_regression.py

"""
Regression tests for known issues.
Add new tests here when bugs are discovered and fixed.
"""

class TestRegression:

    def test_hex_neighbor_at_odd_row(self):
        """
        Regression: Hex neighbor computation was incorrect for odd rows
        in odd-r offset coordinate system.
        Fixed in commit: [hash]
        """
        tiling = HexTiling()
        tiling.generate_graph(10, 10, seed=0)

        # Cell at odd row should have correct neighbors
        # ... specific test case ...

    def test_triangle_facing_after_move(self):
        """
        Regression: Agent facing wasn't updated correctly when moving
        between triangles of different orientations.
        Fixed in commit: [hash]
        """
        # ... specific test case ...
```
