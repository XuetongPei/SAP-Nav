"""LiberoEnv — the environment body of the LIBERO engine (layer L1), a plain
``gymnasium.Env``; ``HabitatEnv`` / ``IsaacEnv`` are its navigation twins.

What it does: hold an episode list and a LiberoBody, put the scene into an
episode's init state, run the simulator, and report facts — observation +
end-effector pose / gripper opening / success / ticks / object poses.

What it deliberately does not do: enforce a budget (gym's ``TimeLimit``
counts steps; the tick cap below is the simulator's horizon guard), compute
any metric (``ManipMetrics``), keep a "next episode" cursor, or put the
instruction into the observation (episode data travel in ``info``).

Two protocols on the same facts:

* ``LiberoEnv`` — LIBERO's own: one ``step`` is one 20 Hz control tick, the
  action is the OSC_POSE input ``[dx, dy, dz, dax, day, daz, gripper]`` in
  [-1, 1] (unit = 5 cm / 0.5 rad of goal shift; gripper +1 close, -1 open).
* ``LiberoPoseEnv`` — the macro protocol for agents that plan in poses: the
  action is an ABSOLUTE end-effector target ``[x, y, z, ax, ay, az, gripper]``
  (world frame, metres, axis-angle, gripper as above); one ``step`` runs a
  closed loop that feeds OSC bounded deltas (at most 2 cm / 0.05 rad of goal
  shift per tick) until the pose is within tolerance (1.5 cm / 0.1 rad), ``max_ticks_per_move``
  is spent, or the position error has stalled (no 2 mm gain over 25 ticks — a
  target out of reach or blocked by the table; without this a failed move
  burned the full cap, and eleven of them the whole tick guard, 2026-09-10).
  ``info["converged"]`` says which, ``info["stalled"]`` whether it gave up. A target whose position is
  ``inf`` is a HOLD: ``GRIPPER_HOLD_TICKS`` zero-motion ticks with the gripper
  command — how a grasp or a release is executed (fingers need ~60 ticks to
  close on an object); ``hold_action(gripper)`` builds one.

``terminated`` is LIBERO's success (every BDDL goal predicate holds) when
``terminate_on_success`` is set — LIBERO's own semantics, the per-tick
protocol's default. The macro protocol defaults it off: the episode runs
until the agent's own stop or the budget, and ``info["success"]`` /
``ManipMetrics`` latch the first tick the goal held, which is what LIBERO
would have scored. ``truncated`` fires at the tick cap (settle ticks excluded, as OpenVLA's
``max_steps + num_steps_wait`` loop counts). ``reward`` is 0.0.

The tick cap is the declaration's ``ticks``, unless the episode carries its
own in ``info["max_ticks"]``: LIBERO's cap is per *suite* (OpenVLA's
``TASK_MAX_STEPS``), so a line whose episodes come from more than one base
suite — the LIBERO-Plus lines, one perturbation kind across all four — has
to carry it per episode. A line drawn from a single suite never sets the key
and every episode uses the declaration's.

Facts are in robosuite's world frame: metres, z up, ``eef_rotation`` as
``(w, x, y, z)``.
"""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .schema import Episode, to_dict
from .sim.libero import LiberoBody, LiberoWorld, load_init_states
from .sim.libero.world import GRIPPER_CLOSE, GRIPPER_OPEN, mat_to_wxyz

OSC_MAX_POS_M = 0.05        # robosuite osc_pose.json output_max: a unit input shifts the goal 5 cm
OSC_MAX_ROT_RAD = 0.5       # ... and 0.5 rad
STEP_POS_M = 0.02           # bounded goal advance per tick in the pose loop (2 cm: the OSC tracks ~half of it; 5 mm made a 25 cm move miss a 100-tick cap)
STEP_ROT_RAD = 0.05         # ... and ~3°
POS_TOL_M = 0.015           # the OSC's steady-state error at the far side of the table is 1.0-1.4 cm (measured 2026-09-10); 1 cm never converged there
ROT_TOL_RAD = 0.10
MAX_TICKS_PER_MOVE = 200
STALL_TICKS = 25            # give up a move whose position error has not improved by STALL_GAIN_M over this many ticks
STALL_GAIN_M = 0.002
GRIPPER_HOLD_TICKS = 60


class LiberoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        episodes: Sequence[Episode],
        body: LiberoBody,
        max_ticks: int | None = None,
        gpu_id: int = 0,
        render_mode: str | None = None,
        terminate_on_success: bool = True,
    ) -> None:
        if not episodes:
            raise ValueError("LiberoEnv: no episodes")
        self.episodes: tuple[Episode, ...] = tuple(episodes)
        self.body = body
        self.max_ticks = int(max_ticks) if max_ticks is not None else None
        self._episode_ticks: int | None = self.max_ticks    # this episode's cap (info["max_ticks"] when it carries one)
        self.terminate_on_success = bool(terminate_on_success)
        self.render_mode = render_mode
        self._by_id = {e.episode_id: e for e in self.episodes}
        self._world = LiberoWorld(body, gpu_id=gpu_id)
        self._init_states: dict[str, np.ndarray] = {}      # bddl file -> (N, state)
        self._episode: Episode | None = None
        self._done = False
        self._succeeded = False
        self._ticks = 0          # every control tick of the episode, the settle ticks included
        self._settle = 0         # the settle ticks of this episode: outside the tick cap, as OpenVLA counts
        self._last_rgb: np.ndarray | None = None

        h, w = body.rgb.height, body.rgb.width
        obs = {"rgb": spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8),
               "proprio": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)}
        if body.wrist is not None:
            obs["wrist"] = spaces.Box(0, 255, shape=(body.wrist.height, body.wrist.width, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(obs)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

    # ---- episode -----------------------------------------------------------------------
    @property
    def episode(self) -> Episode:
        if self._episode is None:
            raise RuntimeError("LiberoEnv: call reset() first")
        return self._episode

    @property
    def goal_index(self) -> int:
        return 0

    def current_goal(self):
        return self.episode.goal

    @property
    def ticks(self) -> int:
        return self._ticks

    def _select(self, options: dict[str, Any] | None) -> Episode:
        options = options or {}
        if "episode" in options:
            return self.episodes[int(options["episode"])]
        if "episode_id" in options:
            return self._by_id[str(options["episode_id"])]
        return self.episodes[int(self.np_random.integers(len(self.episodes)))]

    @staticmethod
    def episode_ticks(episode: Episode, default: int | None) -> int | None:
        """This episode's tick cap: ``info["max_ticks"]`` when the loader set
        one (a line whose episodes span several LIBERO suites), else the
        declaration's."""
        cap = episode.info.get("max_ticks")
        return default if cap is None else int(cap)

    def _horizon(self) -> int:
        cap = self._episode_ticks if self._episode_ticks is not None else 1000
        return cap + self.body.settle_ticks + 10

    # ---- gym API -----------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        ep = self._select(options)
        self._episode_ticks = self.episode_ticks(ep, self.max_ticks)
        self._world.load_scene(ep.scene, horizon=self._horizon())
        states = self._init_states.get(ep.scene.bddl_file)
        if states is None:
            states = self._init_states[ep.scene.bddl_file] = load_init_states(ep.scene)
        k = int(ep.info["init_state_index"])
        self._episode = ep
        self._done = False
        self._succeeded = False
        self._ticks = self._settle = self._world.reset(states[k])
        obs = self._observe()
        info = self._facts()
        info["episode"] = ep.as_dict()
        info["goal"] = to_dict(ep.goal)
        info["robot_base"] = self._world.robot_base()
        return obs, info

    def step(self, action):
        if self._done:
            raise RuntimeError("LiberoEnv: episode is over; reset() first")
        self._tick(np.asarray(action, dtype=np.float64).reshape(7))
        obs = self._observe()
        return obs, 0.0, self._done, self._truncated(), self._facts()

    def _tick(self, action: np.ndarray) -> bool:
        """One control tick; True when the episode ended on it (success or the tick cap)."""
        success = self._world.tick(action)
        self._ticks += 1
        if success:
            self._succeeded = True
            if self.terminate_on_success:
                self._done = True
        return self._done or self._truncated()

    def _truncated(self) -> bool:
        return self._episode_ticks is not None and self._ticks - self._settle >= self._episode_ticks

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

    def _facts(self, **extra: Any) -> dict[str, Any]:
        pos, rot = self._world.eef_pose()
        return {
            "eef_position": pos.tolist(),                 # x, y, z — world frame, metres
            "eef_rotation": mat_to_wxyz(rot).tolist(),    # w, x, y, z
            "gripper_open": self._world.gripper_open(),   # 0 closed .. 100 open
            "success": self._succeeded,                   # latched: the goal held at some tick of this episode
            "ticks": self._ticks,
            "objects": self._world.objects(),             # privileged: for metrics and debugging only
            "goal_index": 0,
            **extra,
        }

    @property
    def world(self) -> LiberoWorld:
        return self._world


class LiberoPoseEnv(LiberoEnv):
    """The macro protocol: ``step([x, y, z, ax, ay, az, gripper])`` drives the
    end-effector to an absolute world-frame pose in a closed loop over OSC
    ticks (see the module docstring); ``hold(gripper, ticks)`` settles the
    gripper. ``info`` adds ``converged``, ``move_ticks``, ``position_error_m``
    and ``rotation_error_rad``."""

    def __init__(self, *args: Any, pos_tol_m: float = POS_TOL_M, rot_tol_rad: float = ROT_TOL_RAD,
                 max_ticks_per_move: int = MAX_TICKS_PER_MOVE, hold_ticks: int = GRIPPER_HOLD_TICKS,
                 terminate_on_success: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, terminate_on_success=terminate_on_success, **kwargs)
        self.hold_ticks = int(hold_ticks)
        self.pos_tol_m, self.rot_tol_rad = float(pos_tol_m), float(rot_tol_rad)
        self.max_ticks_per_move = int(max_ticks_per_move)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64)

    def step(self, action):
        if self._done:
            raise RuntimeError("LiberoPoseEnv: episode is over; reset() first")
        from scipy.spatial.transform import Rotation

        a = np.asarray(action, dtype=np.float64).reshape(7)
        grip = GRIPPER_CLOSE if a[6] > 0 else GRIPPER_OPEN
        if not np.all(np.isfinite(a[:3])):
            return self._hold(grip)
        target_pos, target_rot = a[:3], Rotation.from_rotvec(a[3:6]).as_matrix()
        converged, ended, stalled, n = False, False, False, 0
        pos_err = rot_err = float("nan")
        best_err, best_at = float("inf"), 0
        while n < self.max_ticks_per_move:
            pos, rot = self._world.eef_pose()
            d_pos = target_pos - pos
            d_rot = Rotation.from_matrix(target_rot @ rot.T).as_rotvec()
            pos_err, rot_err = float(np.linalg.norm(d_pos)), float(np.linalg.norm(d_rot))
            if pos_err < self.pos_tol_m and rot_err < self.rot_tol_rad:
                converged = True
                break
            err = pos_err + 0.1 * rot_err          # 0.1 rad counts as 1 cm
            if err < best_err - STALL_GAIN_M:
                best_err, best_at = err, n
            elif n - best_at >= STALL_TICKS:
                stalled = True
                break
            if pos_err > STEP_POS_M:
                d_pos = d_pos * (STEP_POS_M / pos_err)
            if rot_err > STEP_ROT_RAD:
                d_rot = d_rot * (STEP_ROT_RAD / rot_err)
            cmd = np.concatenate([np.clip(d_pos / OSC_MAX_POS_M, -1, 1), np.clip(d_rot / OSC_MAX_ROT_RAD, -1, 1), [grip]])
            n += 1
            if self._tick(cmd):
                ended = True
                break
        if ended:
            pos, _ = self._world.eef_pose()
            pos_err = float(np.linalg.norm(target_pos - pos))
        obs = self._observe()
        info = self._facts(converged=converged, stalled=stalled, move_ticks=n, position_error_m=pos_err,
                           rotation_error_rad=rot_err)
        return obs, 0.0, self._done, self._truncated(), info

    @staticmethod
    def hold_action(gripper: float) -> np.ndarray:
        """The step action that holds the pose and commands the gripper (+ close / - open)."""
        return np.array([np.inf, np.inf, np.inf, 0.0, 0.0, 0.0, float(gripper)], dtype=np.float64)

    def _hold(self, grip: float):
        cmd = np.array([0, 0, 0, 0, 0, 0, grip], dtype=np.float64)
        n = 0
        for _ in range(self.hold_ticks):
            n += 1
            if self._tick(cmd):
                break
        obs = self._observe()
        return obs, 0.0, self._done, self._truncated(), self._facts(converged=True, stalled=False, move_ticks=n,
                                                                    position_error_m=0.0, rotation_error_rad=0.0)
