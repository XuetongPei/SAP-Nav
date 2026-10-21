import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

spec = importlib.util.spec_from_file_location("eval_langmap", Path(__file__).parents[1] / "scripts/eval_langmap.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_proprioception_identity_and_left_turn():
    identity = [0, 0, 0, 1]
    obs = mod.proprioception([0, 0, -2], identity, [0, 0, 0], identity)
    np.testing.assert_allclose(obs["gps"], [2, 0])
    np.testing.assert_allclose(obs["compass"], [0])
    left = [0, math.sqrt(0.5), 0, math.sqrt(0.5)]
    obs = mod.proprioception([-2, 0, 0], left, [0, 0, 0], left)
    np.testing.assert_allclose(obs["gps"], [2, 0], atol=1e-6)
    np.testing.assert_allclose(obs["compass"], [0], atol=1e-6)
    obs = mod.proprioception([0, 0, 0], left, [0, 0, 0], identity)
    np.testing.assert_allclose(obs["compass"], [math.pi / 2], atol=1e-6)


def test_success_distance_is_strict_and_spl_uses_walked_distance():
    assert mod.scores(2, 0.249, 4) == (1.0, 0.5)
    assert mod.scores(2, 0.25, 4) == (0.0, 0.0)
    assert mod.scores(math.inf, math.inf, 4) == (0.0, 0.0)


def test_single_goal_budget_and_policy_input_boundary(tmp_path, monkeypatch):
    class World:
        steps = 0
        placed = 0
        def load_scene(self, scene):
            pass
        def place(self, pos, rot):
            self.placed += 1
        def pose(self):
            return np.array([self.steps * 0.25, 0, 0]), np.array([0, 0, 0, 1])
        def observe(self):
            return {"rgb": np.zeros((2, 2, 3), dtype=np.uint8), "depth": np.ones((2, 2, 1), dtype=np.float32)}
        def geodesic(self, pos, targets):
            return 2.0 if self.steps % 500 == 0 and self.steps == 0 else 0.1
        def move(self, action):
            self.steps += 1
            return False
    class Policy:
        starts = 0
        def set_episode_frame(self, world, start_position, start_rotation):
            pass
        def act(self, **request):
            assert set(request) == {"observations", "goal_attributes", "first", "seed"}
            assert set(request["goal_attributes"]) == {"object_category", "clip_goal", "instruction", "level"}
            assert set(request["observations"]) == {"rgb", "depth", "gps", "compass", "heading"}
            self.starts += int(request["first"])
            return {"action": 1, "reason": "test", "target_detected": False}
    from embodiedscore_envs.benchmarks.env.schema import ObjectGoal
    ep = SimpleNamespace(episode_id="test_single", scene=None, start_position=[0, 0, 0],
                         start_rotation=[0, 0, 0, 1], goal=ObjectGoal(category="chair", instances=()),
                         instruction="chair")
    for directory in ("episodes", "trajectories"):
        (tmp_path / directory).mkdir()
    monkeypatch.setattr(mod, "targets_of", lambda goal: np.zeros((1, 3)))
    monkeypatch.setattr(mod, "goal_attributes", lambda ep, level: {
        "object_category": "chair", "clip_goal": "Seems like there is a chair ahead.",
        "instruction": "Find a chair", "level": level})
    world, policy = World(), Policy()
    result = mod.evaluate(world, policy, ep, "object", tmp_path, 0, 42)
    assert world.placed == 1
    assert policy.starts == 1
    assert result["steps"] == 500
    assert result["success"] == 1.0  # time limit still uses terminal distance
    assert all(r["end_reason"] == "step_limit" for r in result["subgoals"])


def test_shortest_path_waypoint_transform_inverts_map_coordinates():
    from scripts.langmap_shortest_path import waypoint_to_world
    for yaw in (0, math.pi / 6, math.pi / 2, -math.pi / 3):
        rot = [0, math.sin(yaw / 2), 0, math.cos(yaw / 2)]
        origin = [4, 2, -7]
        target = [-2, 3, 6]
        gps = mod.proprioception(target, rot, origin, rot)["gps"]
        xy = [gps[0], -gps[1]]
        np.testing.assert_allclose(waypoint_to_world(xy, origin, rot, 3), target, atol=1e-6)


def test_shortest_path_arrival_and_unreachable_waypoint():
    from scripts.langmap_shortest_path import ShortestPathLocalPolicy
    class World:
        reachable = True
        def follower(self, radius, fresh):
            assert radius == 0.25 and fresh
            return SimpleNamespace(next_action=lambda goal: None)
        def pose(self):
            return np.array([0, 1, 0]), np.array([0, 0, 0, 1])
        def snap(self, target):
            np.testing.assert_allclose(target, [0, 1, -2])
            return np.asarray(target)
        def geodesic(self, current, targets):
            return 0.1 if self.reachable else math.inf
    world = World()
    policy = ShortestPathLocalPolicy(world, [0, 0, 0], [0, 0, 0, 1])
    assert policy.act([2, 0], stop=True)["action"] == 0
    assert policy.act([2, 0], stop=False)["action"] == 2
    world.reachable = False
    result = policy.act([2, 0], stop=False)
    assert result["action"] == 0 and result["local_status"] == "unreachable_waypoint"
