"""IsaacEnv — the environment body of the Isaac engine (layer L1), a plain
``gymnasium.Env``; ``habitat_env.HabitatEnv`` is its habitat twin.

What it does: hold an episode list and an IsaacBody, put the agent at an
episode's start, execute one action per ``step`` (a movement or STOP), and
report facts — observation + position / rotation / heading / pitch /
collided / distance_to_goal / stop_called / goal_index.

What it deliberately does not do: count steps or enforce a budget (gym's
``TimeLimit``), compute any metric (``VLNVerseMetrics``), clip or normalise
depth (``DepthClip``), keep a "next episode" cursor (episodes are chosen
through ``reset(options=...)`` or sampled with the env's RNG), know which
benchmark it is running, or put the goal or instruction into the observation
(they are episode data and travel in ``info``).

Facts are in the Isaac world frame the split files use: metres, z up,
``rotation`` as ``(w, x, y, z)``; ``distance_to_goal`` is the ground-plane
euclidean distance (there is no navmesh and no geodesic in this engine).
``goal_index`` is always 0 — VLNverse has one goal per episode; the key is
kept so every body reports the same facts.

``reward`` is always 0.0: these are evaluation environments.
"""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .schema import Act, Episode, targets_of, to_dict
from .sim import IsaacBody, IsaacSettings, IsaacWorld

_DISCRETE_DEFAULT = (Act.STOP, Act.FORWARD, Act.LEFT, Act.RIGHT)


class IsaacEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        episodes: Sequence[Episode],
        body: IsaacBody,
        actions: Sequence[Act] = _DISCRETE_DEFAULT,
        gpu_id: int = 0,
        settings: IsaacSettings | None = None,
        render_mode: str | None = None,
    ) -> None:
        if not episodes:
            raise ValueError("IsaacEnv: no episodes")
        self.episodes: tuple[Episode, ...] = tuple(episodes)
        self.body = body
        self.actions: tuple[Act, ...] = tuple(Act(a) for a in actions)
        if self.actions != tuple(Act(i) for i in range(len(self.actions))):
            raise ValueError(f"actions must be a prefix of the Act order, got {self.actions}")
        if Act.LOOK_UP in self.actions or Act.LOOK_DOWN in self.actions:
            raise ValueError("LOOK actions are not served by the Isaac worker (yaw-only poses)")
        if Act.SUBTASK_STOP in self.actions:
            raise ValueError("SUBTASK_STOP needs a GoalSequence goal; VLNverse episodes have one goal")
        self.render_mode = render_mode
        self._by_id = {e.episode_id: e for e in self.episodes}
        self._world = IsaacWorld(body, gpu_id=gpu_id, settings=settings)   # Isaac boots on the first reset
        self._episode: Episode | None = None
        self._stop_called = False
        self._done = False
        self._last_rgb: np.ndarray | None = None

        obs = {"rgb": spaces.Box(0, 255, shape=(body.rgb.height, body.rgb.width, 3), dtype=np.uint8)}
        if body.depth is not None:
            obs["depth"] = spaces.Box(0.0, np.inf, shape=(body.depth.height, body.depth.width, 1), dtype=np.float32)
        self.observation_space = spaces.Dict(obs)
        self.action_space = spaces.Discrete(len(self.actions))

    # ---- episode -----------------------------------------------------------------------
    @property
    def episode(self) -> Episode:
        if self._episode is None:
            raise RuntimeError("IsaacEnv: call reset() first")
        return self._episode

    @property
    def goal_index(self) -> int:
        return 0

    def current_goal(self):
        return self.episode.goal

    def _select(self, options: dict[str, Any] | None) -> Episode:
        options = options or {}
        if "episode" in options:
            return self.episodes[int(options["episode"])]
        if "episode_id" in options:
            return self._by_id[str(options["episode_id"])]
        return self.episodes[int(self.np_random.integers(len(self.episodes)))]

    # ---- gym API -----------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        ep = self._select(options)
        self._world.load_scene(ep.scene)
        self._episode = ep
        self._stop_called = False
        self._done = False
        self._world.place(ep.start_position, ep.start_rotation)
        obs = self._observe()
        info = self._facts(collided=False)
        info["episode"] = ep.as_dict()
        info["goal"] = to_dict(ep.goal)
        return obs, info

    def step(self, action: int):
        if self._done:
            raise RuntimeError("IsaacEnv: episode is over; reset() first")
        a = Act(int(action))
        if a not in self.actions:
            raise ValueError(f"action {a!r} not in this environment's table {self.actions}")
        collided = False
        if a == Act.STOP:
            self._stop_called = True
            self._done = True
        else:
            collided = self._world.move(int(a))
        obs = self._observe()
        return obs, 0.0, self._done, False, self._facts(collided=collided)

    def render(self):
        if self.render_mode == "rgb_array":
            return None if self._last_rgb is None else self._last_rgb.copy()
        return None

    def close(self) -> None:
        self._world.close()

    # ---- facts ---------------------------------------------------------------------------
    def _observe(self) -> dict[str, np.ndarray]:
        obs = self._world.observe()
        self._last_rgb = obs["rgb"]
        return obs

    def _facts(self, *, collided: bool) -> dict[str, Any]:
        pos, rot = self._world.pose()
        targets = targets_of(self.episode.goal, dtype=np.float64)
        return {
            "position": pos.tolist(),                 # x, y, z (z = camera height)
            "rotation": rot.tolist(),                 # w, x, y, z
            "heading": self._world.heading(),
            "pitch": self._world.pitch(),
            "collided": bool(collided),
            "distance_to_goal": None if targets is None else self._world.distance(pos, targets),
            "stop_called": self._stop_called,
            "goal_index": 0,
        }

    @property
    def world(self) -> IsaacWorld:
        """The simulator facade — for wrappers and tools that need panoramas,
        the freemap, or to move the body without stepping the environment."""
        return self._world


class IsaacPolarEnv(IsaacEnv):
    """VLNverse's macro-action protocol (the zero-shot line's ``step_hightolow``):
    the action is ``[angle_rad, distance_m, elevation_deg]`` — rotate, then
    advance. ``angle == 0 and distance == 0`` is STOP, as in that line.
    ``info`` additionally carries ``deviation_m``, how far the occupancy snap
    moved the intended point. The ``-upstream`` variant of the VLNverse lines
    serves this interface, as the EQA lines' ``-upstream`` serves their pose
    protocol."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("actions", None)
        super().__init__(*args, actions=(Act.STOP,), **kwargs)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)

    def step(self, action):
        if self._done:
            raise RuntimeError("IsaacPolarEnv: episode is over; reset() first")
        angle, distance, elevation = (float(v) for v in np.asarray(action, dtype=np.float64).reshape(3))
        if angle == 0.0 and distance == 0.0:
            self._stop_called = True
            self._done = True
            obs = self._observe()
            info = self._facts(collided=False)
            info["deviation_m"] = 0.0
            return obs, 0.0, True, False, info
        collided, deviation = self._world.move_polar(angle, distance, elevation)
        obs = self._observe()
        info = self._facts(collided=collided)
        info["deviation_m"] = deviation
        return obs, 0.0, False, False, info
