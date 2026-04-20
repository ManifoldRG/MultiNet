# test_performance.py

import pytest
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.env import MultiGridEnv, Action


def create_task(grid_size=10, max_steps=100):
    """Helper to create a task spec for performance testing."""
    return {
        "task_id": "perf_test",
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
                "position": {"x": 0.1, "y": 0.1},
                "facing": 0
            }
        },
        "goal": {
            "predicate": "reach_position",
            "position": {"x": 0.9, "y": 0.9}
        },
        "limits": {"max_steps": max_steps},
        "tiling": {"type": "square", "grid_size": {"width": grid_size, "height": grid_size}}
    }


class TestPerformance:
    """Performance benchmark tests."""

    @pytest.mark.parametrize("grid_size", [10, 25, 50])
    @pytest.mark.parametrize("tiling", ["square", "hex", "triangle"])
    def test_reset_time(self, grid_size, tiling):
        """Reset should complete within time budget."""
        task = create_task(grid_size=grid_size)
        task["tiling"]["type"] = tiling

        env = MultiGridEnv(task, tiling=tiling)

        times = []
        for _ in range(10):
            start = time.time()
            env.reset()
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        # Soft guidelines from spec
        if grid_size <= 25:
            assert avg_time < 0.2, \
                f"{tiling} grid {grid_size}x{grid_size} reset took {avg_time:.3f}s (should be < 0.2s)"
        else:
            assert avg_time < 0.7, \
                f"{tiling} grid {grid_size}x{grid_size} reset took {avg_time:.3f}s (should be < 0.7s)"

        print(f"\n{tiling} {grid_size}x{grid_size}: avg={avg_time*1000:.1f}ms, max={max_time*1000:.1f}ms")

    @pytest.mark.parametrize("tiling", ["square", "hex", "triangle"])
    def test_step_throughput(self, tiling):
        """Step should achieve target throughput."""
        task = create_task(grid_size=20, max_steps=1100)
        task["tiling"]["type"] = tiling

        env = MultiGridEnv(task, tiling=tiling)
        env.reset()

        # Measure throughput over 1000 steps
        start = time.time()
        for _ in range(1000):
            env.step(Action.TURN_RIGHT)
        elapsed = time.time() - start

        steps_per_second = 1000 / elapsed

        # Soft guidelines - triangle grid has more cells and is expected to be slower
        if tiling == "triangle":
            assert steps_per_second > 60, \
                f"{tiling} achieved {steps_per_second:.0f} steps/sec (should be > 60)"
        else:
            assert steps_per_second > 600, \
                f"{tiling} achieved {steps_per_second:.0f} steps/sec (should be > 600)"

        print(f"\n{tiling} throughput: {steps_per_second:.0f} steps/sec")

    def test_large_grid_scalability(self):
        """Test that very large grids are still performant."""
        task = create_task(grid_size=100)
        env = MultiGridEnv(task, tiling="square")

        # Reset time
        start = time.time()
        env.reset()
        reset_time = time.time() - start

        assert reset_time < 2.0, \
            f"Large grid (100x100) reset took {reset_time:.2f}s (should be < 2.0s)"

        # Step throughput - with rendering this will be slower
        start = time.time()
        for _ in range(100):
            env.step(Action.FORWARD)
        step_time = time.time() - start

        # Relaxed constraint - with rendering overhead
        assert step_time < 4.25, \
            f"Large grid (100x100) 100 steps took {step_time:.2f}s (should be < 4.25s)"

        print(f"\n100x100 grid: reset={reset_time*1000:.0f}ms, 100 steps={step_time*1000:.0f}ms")

    @pytest.mark.parametrize("tiling", ["square", "hex", "triangle"])
    def test_memory_efficiency(self, tiling):
        """Test that environment instances don't consume excessive memory."""
        psutil = pytest.importorskip("psutil")
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create multiple environment instances
        envs = []
        for i in range(10):
            task = create_task(grid_size=20)
            task["tiling"]["type"] = tiling
            task["task_id"] = f"test_{i}"

            env = MultiGridEnv(task, tiling=tiling)
            env.reset()
            envs.append(env)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_per_env = (final_memory - initial_memory) / 10

        # Each environment should use less than 10MB
        assert memory_per_env < 10, \
            f"{tiling} env uses {memory_per_env:.1f}MB (should be < 10MB)"

        print(f"\n{tiling} memory per env: {memory_per_env:.1f}MB")

        # Clean up
        del envs

    def test_rapid_reset_performance(self):
        """Test rapid reset/step cycles."""
        task = create_task(grid_size=10, max_steps=5)
        env = MultiGridEnv(task, tiling="square")

        start = time.time()
        for _ in range(100):
            env.reset()
            for _ in range(5):
                env.step(Action.TURN_RIGHT)
        elapsed = time.time() - start

        episodes_per_second = 100 / elapsed

        assert episodes_per_second > 50, \
            f"Rapid reset achieved {episodes_per_second:.0f} episodes/sec (should be > 50)"

        print(f"\nRapid reset: {episodes_per_second:.0f} episodes/sec")


class TestScalability:
    """Tests for system scalability."""

    @pytest.mark.parametrize("num_objects", [1, 10, 50])
    def test_many_objects(self, num_objects):
        """Test performance with many objects in scene."""
        task = create_task(grid_size=20)

        # Add many objects
        objects = []
        for i in range(num_objects):
            x = 0.1 + (i % 5) * 0.15
            y = 0.1 + (i // 5) * 0.15
            objects.append({
                "id": f"cube_{i}",
                "type": "movable",
                "color": "red" if i % 2 == 0 else "blue",
                "position": {"x": x, "y": y},
                "size": 0.1
            })
        task["scene"]["objects"] = objects

        env = MultiGridEnv(task, tiling="square")

        # Measure reset time
        start = time.time()
        env.reset()
        reset_time = time.time() - start

        # Reset time should scale reasonably
        expected_time = 0.05 + (num_objects * 0.002)  # Base + per-object
        assert reset_time < expected_time, \
            f"Reset with {num_objects} objects took {reset_time:.3f}s"

        # Measure step time
        start = time.time()
        for _ in range(100):
            env.step(Action.TURN_RIGHT)
        step_time = time.time() - start

        # Step time should not be significantly affected by number of objects
        assert step_time < 0.15, \
            f"100 steps with {num_objects} objects took {step_time:.3f}s"

        print(f"\n{num_objects} objects: reset={reset_time*1000:.1f}ms, 100 steps={step_time*1000:.1f}ms")

    def test_concurrent_environments(self):
        """Test that multiple environments can coexist without interference."""
        tasks = []
        envs = []

        # Create 5 different environments with varying seeds and agent positions
        for i in range(5):
            task = create_task(grid_size=10)
            task["seed"] = 100 + i
            task["task_id"] = f"concurrent_{i}"
            # Vary agent start position to ensure different states
            x = 0.1 + (i * 0.15)
            y = 0.1 + (i * 0.15)
            task["scene"]["agent"]["position"] = {"x": x, "y": y}
            tasks.append(task)

            env = MultiGridEnv(task, tiling="square")
            env.reset(seed=100 + i)
            envs.append(env)

        # Step each environment independently
        for i, env in enumerate(envs):
            for _ in range(10):
                env.step(Action.FORWARD)

        # Verify environments maintain independent states
        # Check that at least some environments have different states
        different_states = 0
        for i in range(len(envs)):
            for j in range(i + 1, len(envs)):
                if envs[i].state.agent.cell_id != envs[j].state.agent.cell_id or \
                   envs[i].state.agent.facing != envs[j].state.agent.facing:
                    different_states += 1

        # At least half of the environment pairs should have different states
        total_pairs = len(envs) * (len(envs) - 1) // 2
        assert different_states >= total_pairs // 2, \
            f"Only {different_states}/{total_pairs} environment pairs have different states"
