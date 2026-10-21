"""RobocasaEnv — the environment body of the RoboCasa engine (layer L1), a
plain ``gymnasium.Env``; ``LiberoEnv`` is its fixed-base twin and
``HabitatEnv`` / ``IsaacEnv`` the navigation ones.

What it does: hold an episode list and a RobocasaBody, generate the
episode's kitchen under its seed, run the simulator, and report facts —
observation + end-effector pose / gripper opening / mobile-base pose /
success / ticks / object poses / the episode's language.

What it deliberately does not do: enforce a budget (gym's ``TimeLimit``
counts steps; the tick cap below is the simulator's horizon guard), compute
any metric (``ManipMetrics``), keep a "next episode" cursor, or put the
instruction into the observation (episode data travel in ``info``).

Two protocols on the same facts:

* ``RobocasaEnv`` — RoboCasa's own: one ``step`` is one 20 Hz control tick,
  the action is RoboCasa's 12-D gym action in [-1, 1] (``sim/robocasa/world.py``
  transcribes the layout and its two sources). The arm delta is 5 cm / 0.5 rad
  per unit **in the arm's base frame**; ``a[6]`` and ``a[11]`` are flags
  thresholded at 0.5 (gripper close, base mode).
* ``RobocasaPoseEnv`` — the macro protocol for agents that plan in poses, on
  a robot that can also drive. One ``step`` is one of three moves, in this
  precedence:

      base move   ``a[7:10]`` (dx, dy, dyaw) not all zero — drive the mobile
                  base by that displacement in its own frame, arm holding,
                  base mode on
      pose move   ``a[0:3]`` finite — an ABSOLUTE end-effector target
                  ``[x, y, z, ax, ay, az]`` (world frame, metres, axis-angle)
                  driven in a closed loop of bounded OSC deltas
      gripper hold ``a[0:3]`` at ``inf`` and no base move — ``hold_ticks``
                  zero-motion ticks with the gripper command, exactly as
                  ``LiberoPoseEnv`` reads an ``inf`` target (fingers need
                  ~60 ticks to close on an object)

  ``a[6]`` is the gripper on every move (> 0 close, <= 0 open).
  ``hold_action(gripper)`` and ``base_action(dx, dy, dyaw, gripper)`` build
  the two special forms. ``info["converged"]`` says whether the move landed.

``terminated`` is RoboCasa's success check when ``terminate_on_success`` is
set — RoboCasa's own semantics (``robocasa/utils/env_utils.py``
``run_random_rollouts`` breaks the rollout on ``info["success"]``), the
per-tick protocol's default. The macro protocol defaults it off: the episode
runs until the agent's own stop or the budget, and ``info["success"]`` /
``ManipMetrics`` latch the first tick the goal held, which is what RoboCasa
would have scored. ``truncated`` fires at the tick cap (settle ticks
excluded). ``reward`` is 0.0.

Facts are in robosuite's world frame: metres, z up, ``eef_rotation`` as
``(w, x, y, z)``.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .schema import Episode, to_dict
from .sim.robocasa import RobocasaBody, RobocasaWorld
from .sim.robocasa.world import (ACTION_DIM, A_BASE, A_EEF_POS, A_EEF_ROT, A_GRIPPER, A_MODE, BASE_MODE_OFF,
                                 BASE_MODE_ON, GRIPPER_CLOSE, GRIPPER_OPEN, mat_to_wxyz, zero_action)

OSC_MAX_POS_M = 0.05        # robosuite default_pandaomron.json output_max: a unit input shifts the goal 5 cm
OSC_MAX_ROT_RAD = 0.5       # ... and 0.5 rad
STEP_POS_M = 0.02           # bounded goal advance per tick in the pose loop (the OSC tracks about half of it)
STEP_ROT_RAD = 0.05         # ... and ~3°
POS_TOL_M = 0.01
ROT_TOL_RAD = 0.10
MAX_TICKS_PER_MOVE = 200
GRIPPER_HOLD_TICKS = 60

BASE_POS_TOL_M = 0.05       # a mobile base parks to 5 cm ...
BASE_YAW_TOL_RAD = 0.05     # ... and ~3°
BASE_POS_GAIN = 8.0         # metres of error -> base JOINT_VELOCITY command (clipped to [-1, 1])
BASE_YAW_GAIN = 4.0         # radians of error -> base yaw command
BASE_MIN_CMD = 0.20         # the base creeps to a halt under a bare proportional law: a floor on the
                            # command while the error is outside tolerance, so the last centimetres close
MAX_TICKS_PER_BASE_MOVE = 300

POSE_ACTION_DIM = 10        # [x, y, z, ax, ay, az, gripper, dx, dy, dyaw]
P_BASE = slice(7, 10)


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _floor(vec: np.ndarray, minimum: float) -> np.ndarray:
    """Scale a 2-D command up so its magnitude is at least ``minimum`` (never
    past the [-1, 1] the controller takes), keeping its direction."""
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9 or norm >= minimum:
        return vec
    return np.clip(vec * (minimum / norm), -1.0, 1.0)


class RobocasaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        episodes: Sequence[Episode],
        body: RobocasaBody,
        max_ticks: int | None = None,
        tick_scale: float = 1.0,
        gpu_id: int = 0,
        render_mode: str | None = None,
        terminate_on_success: bool = True,
    ) -> None:
        if not episodes:
            raise ValueError("RobocasaEnv: no episodes")
        self.episodes: tuple[Episode, ...] = tuple(episodes)
        self.body = body
        self.max_ticks = int(max_ticks) if max_ticks is not None else None
        self.tick_scale = float(tick_scale)
        self.terminate_on_success = bool(terminate_on_success)
        self._cap: int | None = None      # this episode's tick cap (see _tick_cap)
        self.render_mode = render_mode
        self._by_id = {e.episode_id: e for e in self.episodes}
        self._world = RobocasaWorld(body, gpu_id=gpu_id)
        self._episode: Episode | None = None
        self._done = False
        self._succeeded = False
        self._ticks = 0          # every control tick of the episode, the settle ticks included
        self._settle = 0         # the settle ticks of this episode: outside the tick cap
        self._last_rgb: np.ndarray | None = None

        h, w = body.rgb.height, body.rgb.width
        obs = {"rgb": spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8),
               "proprio": spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)}
        for key, cam in (("wrist", body.wrist), ("aux", body.aux)):
            if cam is not None:
                obs[key] = spaces.Box(0, 255, shape=(cam.height, cam.width, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(obs)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)

    # ---- episode -----------------------------------------------------------------------
    @property
    def episode(self) -> Episode:
        if self._episode is None:
            raise RuntimeError("RobocasaEnv: call reset() first")
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

    def _tick_cap(self, ep: Episode) -> int | None:
        """The episode's control-tick cap. RoboCasa's horizon is per task
        (``robocasa/utils/dataset_registry.py``), carried in
        ``episode.info["horizon"]``; ``tick_scale`` is the multiple of it this
        variant allows (1 upstream, ``TICK_GUARD`` for the macro protocol).
        A declared ``max_ticks`` overrides it with one number for the line."""
        if self.max_ticks is not None:
            return self.max_ticks
        horizon = ep.info.get("horizon")
        return None if horizon is None else int(round(self.tick_scale * int(horizon)))

    def _horizon(self, cap: int | None) -> int:
        return (cap if cap is not None else 1000) + self.body.settle_ticks + 10

    # ---- gym API -----------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        ep = self._select(options)
        self._cap = self._tick_cap(ep)
        self._world.load_scene(ep.scene, horizon=self._horizon(self._cap))
        self._episode = ep
        self._done = False
        self._succeeded = False
        self._ticks = self._settle = self._world.reset(int(ep.scene.seed))
        obs = self._observe()
        info = self._facts()
        episode = ep.as_dict()
        if self._world.language:      # RoboCasa words the instruction for the sampled objects
            episode["instruction"] = self._world.language
        info["episode"] = episode
        info["goal"] = to_dict(ep.goal)
        info["robot_base"] = self._world.robot_base()
        info["fixtures"] = self._world.fixtures()
        return obs, info

    def step(self, action):
        if self._done:
            raise RuntimeError("RobocasaEnv: episode is over; reset() first")
        self._tick(np.asarray(action, dtype=np.float64).reshape(ACTION_DIM))
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
        return self._cap is not None and self._ticks - self._settle >= self._cap

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
        base_pos, base_yaw = self._world.base_pose()
        return {
            "eef_position": pos.tolist(),                    # x, y, z — world frame, metres
            "eef_rotation": mat_to_wxyz(rot).tolist(),       # w, x, y, z
            "gripper_open": self._world.gripper_open(),      # 0 closed .. 100 open
            "base_position": base_pos.tolist(),              # the mobile base, world frame
            "base_yaw_rad": base_yaw,
            "language": self._world.language,                # RoboCasa's per-episode instruction
            "success": self._succeeded,                      # latched: the check held at some tick of this episode
            "ticks": self._ticks,
            "tick_cap": self._cap,                           # this episode's control-tick cap
            "objects": self._world.objects(),                # privileged: for metrics and debugging only
            "goal_index": 0,
            **extra,
        }

    @property
    def world(self) -> RobocasaWorld:
        return self._world


class RobocasaPoseEnv(RobocasaEnv):
    """The macro protocol on a mobile manipulator: one ``step`` is one base
    move, one absolute end-effector move, or one gripper hold (see the module
    docstring). ``info`` adds ``converged``, ``move_ticks``, ``move_kind``,
    ``position_error_m`` and ``rotation_error_rad``."""

    def __init__(self, *args: Any, pos_tol_m: float = POS_TOL_M, rot_tol_rad: float = ROT_TOL_RAD,
                 max_ticks_per_move: int = MAX_TICKS_PER_MOVE, hold_ticks: int = GRIPPER_HOLD_TICKS,
                 base_pos_tol_m: float = BASE_POS_TOL_M, base_yaw_tol_rad: float = BASE_YAW_TOL_RAD,
                 max_ticks_per_base_move: int = MAX_TICKS_PER_BASE_MOVE,
                 terminate_on_success: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, terminate_on_success=terminate_on_success, **kwargs)
        self.hold_ticks = int(hold_ticks)
        self.pos_tol_m, self.rot_tol_rad = float(pos_tol_m), float(rot_tol_rad)
        self.max_ticks_per_move = int(max_ticks_per_move)
        self.base_pos_tol_m, self.base_yaw_tol_rad = float(base_pos_tol_m), float(base_yaw_tol_rad)
        self.max_ticks_per_base_move = int(max_ticks_per_base_move)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(POSE_ACTION_DIM,), dtype=np.float64)

    # ---- the three moves ----------------------------------------------------------------
    @staticmethod
    def hold_action(gripper: float) -> np.ndarray:
        """The step action that holds the pose and commands the gripper (+ close / - open)."""
        a = np.zeros(POSE_ACTION_DIM, dtype=np.float64)
        a[A_EEF_POS] = np.inf
        a[A_GRIPPER] = float(gripper)
        return a

    @staticmethod
    def base_action(dx: float, dy: float, dyaw: float, gripper: float = -1.0) -> np.ndarray:
        """The step action that drives the mobile base by (dx, dy) metres and
        ``dyaw`` radians **in the base's own frame**, arm holding."""
        a = RobocasaPoseEnv.hold_action(gripper)
        a[P_BASE] = [float(dx), float(dy), float(dyaw)]
        return a

    @staticmethod
    def pose_action(position, rotvec, gripper: float) -> np.ndarray:
        """The step action that drives the end-effector to an absolute world pose."""
        a = np.zeros(POSE_ACTION_DIM, dtype=np.float64)
        a[A_EEF_POS] = np.asarray(position, dtype=np.float64).reshape(3)
        a[A_EEF_ROT] = np.asarray(rotvec, dtype=np.float64).reshape(3)
        a[A_GRIPPER] = float(gripper)
        return a

    # ---- gym API -------------------------------------------------------------------------
    def step(self, action):
        if self._done:
            raise RuntimeError("RobocasaPoseEnv: episode is over; reset() first")
        a = np.asarray(action, dtype=np.float64).reshape(POSE_ACTION_DIM)
        grip = GRIPPER_CLOSE if a[A_GRIPPER] > 0 else GRIPPER_OPEN
        if np.any(a[P_BASE] != 0.0):
            return self._drive(a[P_BASE], grip)
        if not np.all(np.isfinite(a[A_EEF_POS])):
            return self._hold(grip)
        return self._reach(a[A_EEF_POS], a[A_EEF_ROT], grip)

    def _finish(self, **extra: Any):
        obs = self._observe()
        return obs, 0.0, self._done, self._truncated(), self._facts(**extra)

    def _reach(self, target_pos: np.ndarray, target_rotvec: np.ndarray, grip: float):
        """Closed loop of bounded OSC deltas onto an absolute world-frame pose.
        The OSC's input frame is the arm's base (``default_pandaomron.json``
        ``input_ref_frame: "base"``), so the world-frame error is rotated into
        it before it is scaled by the controller's output_max."""
        from scipy.spatial.transform import Rotation

        target_rot = Rotation.from_rotvec(target_rotvec).as_matrix()
        converged, ended, n = False, False, 0
        pos_err = rot_err = float("nan")
        while n < self.max_ticks_per_move:
            pos, rot = self._world.eef_pose()
            _, frame = self._world.arm_frame()
            d_pos = target_pos - pos
            d_rot_mat = target_rot @ rot.T
            d_rot = Rotation.from_matrix(d_rot_mat).as_rotvec()
            pos_err, rot_err = float(np.linalg.norm(d_pos)), float(np.linalg.norm(d_rot))
            if pos_err < self.pos_tol_m and rot_err < self.rot_tol_rad:
                converged = True
                break
            if pos_err > STEP_POS_M:
                d_pos = d_pos * (STEP_POS_M / pos_err)
            if rot_err > STEP_ROT_RAD:
                d_rot = d_rot * (STEP_ROT_RAD / rot_err)
                d_rot_mat = Rotation.from_rotvec(d_rot).as_matrix()
            d_pos_base = frame.T @ d_pos
            d_rot_base = Rotation.from_matrix(frame.T @ d_rot_mat @ frame).as_rotvec()
            cmd = zero_action(grip)
            cmd[A_EEF_POS] = np.clip(d_pos_base / OSC_MAX_POS_M, -1.0, 1.0)
            cmd[A_EEF_ROT] = np.clip(d_rot_base / OSC_MAX_ROT_RAD, -1.0, 1.0)
            cmd[A_MODE] = BASE_MODE_OFF
            n += 1
            if self._tick(cmd):
                ended = True
                break
        if ended:
            pos, _ = self._world.eef_pose()
            pos_err = float(np.linalg.norm(target_pos - pos))
        return self._finish(converged=converged, move_ticks=n, move_kind="pose",
                            position_error_m=pos_err, rotation_error_rad=rot_err)

    def _drive(self, delta: np.ndarray, grip: float):
        """Closed loop on the mobile base: (dx, dy, dyaw) in the base's own
        frame at the start of the move. The base part is JOINT_VELOCITY
        (``default_pandaomron.json``), so the command is a proportional
        velocity; base mode is on so the arm tracks the moving base
        (``composite_controller.HybridMobileBase.set_goal``)."""
        start_pos, start_yaw = self._world.base_pose()
        c, s = np.cos(start_yaw), np.sin(start_yaw)
        target_xy = start_pos[:2] + np.array([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1]])
        target_yaw = start_yaw + float(delta[2])
        converged, n = False, 0
        pos_err = rot_err = float("nan")
        while n < self.max_ticks_per_base_move:
            pos, yaw = self._world.base_pose()
            e_world = target_xy - pos[:2]
            e_yaw = _wrap(target_yaw - yaw)
            pos_err, rot_err = float(np.linalg.norm(e_world)), abs(e_yaw)
            if pos_err < self.base_pos_tol_m and rot_err < self.base_yaw_tol_rad:
                converged = True
                break
            cy, sy = np.cos(yaw), np.sin(yaw)
            e_body = np.array([cy * e_world[0] + sy * e_world[1], -sy * e_world[0] + cy * e_world[1]])
            drive = np.clip(e_body * BASE_POS_GAIN, -1.0, 1.0)
            if pos_err >= self.base_pos_tol_m:
                drive = _floor(drive, BASE_MIN_CMD)
            turn = float(np.clip(e_yaw * BASE_YAW_GAIN, -1.0, 1.0))
            if rot_err >= self.base_yaw_tol_rad:
                turn = math.copysign(max(abs(turn), BASE_MIN_CMD), turn)
            cmd = zero_action(grip)
            cmd[A_BASE] = [drive[0], drive[1], turn]
            cmd[A_MODE] = BASE_MODE_ON
            n += 1
            if self._tick(cmd):
                break
        return self._finish(converged=converged, move_ticks=n, move_kind="base",
                            position_error_m=pos_err, rotation_error_rad=rot_err)

    def _hold(self, grip: float):
        cmd = zero_action(grip)
        n = 0
        for _ in range(self.hold_ticks):
            n += 1
            if self._tick(cmd):
                break
        return self._finish(converged=True, move_ticks=n, move_kind="gripper",
                            position_error_m=0.0, rotation_error_rad=0.0)
