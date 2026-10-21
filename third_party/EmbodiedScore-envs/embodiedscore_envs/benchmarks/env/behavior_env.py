"""BehaviorEnv — the environment body of the BEHAVIOR engine (layer L1), a
plain ``gymnasium.Env``; ``RobocasaEnv`` is its single-arm mobile twin,
``RobotwinEnv`` its bimanual fixed-base one.

What it does: hold an episode list and a BehaviorBody, load an episode's task
instance into OmniGibson, run control ticks, and report facts — observation +
both arms' end-effector poses / gripper openings / the mobile base pose /
BEHAVIOR's success and per-predicate goal status / ticks / task-object poses.

What it deliberately does not do: enforce the step budget (gym's ``TimeLimit``
counts steps; the tick cap below is BEHAVIOR's own ``Timeout``), compute any
metric (``BehaviorMetrics``), keep a "next episode" cursor, or put the
instruction into the observation (episode data travel in ``info``).

Two protocols on the same facts:

* ``BehaviorEnv`` — the challenge's own. One ``step`` is one 30 Hz control
  tick and the action is the flat 23-D vector the evaluator feeds the robot,
  ``[base 3 | torso 4 | left arm 7 | left gripper 1 | right arm 7 | right
  gripper 1]`` (``learning/utils/eval_utils.py``
  ``ACTION_QPOS_INDICES["R1Pro"]`` L49-58; the arms and the torso are
  ABSOLUTE joint angles, the base a velocity in ``[-1, 1]³``, each gripper a
  finger target — ``og_teleop_cfg.py`` ``R1_CONTROLLER_CONFIG`` L181-232).
  A zero vector is NOT a no-op on this robot: use ``world.no_op_action()``.

* ``BehaviorPoseEnv`` — the macro protocol for agents that plan in poses, on
  a two-armed robot that also drives: ``RobotwinPoseEnv``'s bimanual layout
  with ``RobocasaPoseEnv``'s base move appended, 17 wide —

      ``[Lx, Ly, Lz, Lax, Lay, Laz, Lgrip,
         Rx, Ry, Rz, Rax, Ray, Raz, Rgrip,
         dx, dy, dyaw]``

  One ``step`` is one of three moves, in this precedence:

      base move    ``a[14:17]`` not all zero — drive the holonomic base by that
                   displacement in its own frame, both arms holding
      pose move    at least one arm's target position finite — an ABSOLUTE
                   end-effector target ``[x, y, z, ax, ay, az]`` (world frame,
                   metres, axis-angle) per arm, driven in a closed loop of
                   bounded goal advances; an arm at ``inf`` HOLDS and only its
                   gripper command applies
      gripper hold both arms at ``inf`` and no base move — ``hold_ticks``
                   zero-motion ticks with the gripper commands, exactly as
                   ``LiberoPoseEnv`` reads an ``inf`` target (fingers need
                   ticks to close on an object)

  The gripper field is the package's convention (``+1`` closed, ``-1`` open,
  linear between), mapped onto that gripper controller's own output limits.
  Arm targets reach the robot through the ``InverseKinematicsController`` in
  ``mode: "absolute_pose"`` the challenge documents as a permitted action-space
  substitution (``docs/challenge/evaluation.md`` § "Configure Robot Action
  Space"), and the world re-expresses each target in the frame that controller
  reads. ``info["converged"]`` says whether the move landed; a BEHAVIOR target
  is often simply out of reach from where the base happens to stand.

**What the macro protocol does not command: the torso.** The R1 Pro's four
trunk joints are part of the challenge's own action space (``[base 3 | torso 4
| ...]``) and they pitch the whole upper body, so they set where the arms can
reach at all. This protocol holds them at whatever posture the instance loaded
with, which fixes each arm's reachable set to one shell around a fixed torso.
A line that needs to reach low shelves or high cupboards needs the torso;
extending the action with an absolute torso target, or letting the closed loop
solve for one, is the outstanding work on this protocol. ``BehaviorEnv``, the
per-tick variant, commands the torso already — it is the challenge's own 23-D
action.

``terminated`` is BEHAVIOR's own success (every goal predicate of some
grounding satisfied) when ``terminate_on_success`` is set — the challenge's
semantics, since its ``PredicateGoal`` is a success termination condition
(``tasks/behavior_task.py`` L197-202) and the evaluator stops the rollout on
``terminated`` — the per-tick protocol's default. The macro protocol defaults
it off: the episode runs until the agent's own stop or the budget, and
``info["success"]`` / ``BehaviorMetrics`` latch the first tick the goal held,
which is what the challenge would have scored. ``truncated`` fires at the
episode's tick cap, which is the challenge's own per-task budget. ``reward``
is 0.0.

Facts are in OmniGibson's world frame: metres, z up, ``eef_rotation`` as
``(w, x, y, z)``.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .schema import Episode, to_dict
from .sim.behavior import BehaviorBody, BehaviorWorld
from .sim.behavior.body import ARMS
from .sim.behavior.world import quat_to_rotvec, rotvec_to_quat

ARM_ACTION = 7                 # per arm in the macro protocol: x y z + axis-angle + gripper
POSE_ACTION_DIM = 17           # two arms + (dx, dy, dyaw)
P_BASE = slice(14, 17)

POS_TOL_M = 0.02               # a macro arm move counts as reached inside this
ROT_TOL_RAD = 0.15             # ... and this
STEP_POS_M = 0.02              # bounded goal advance per tick, as the LIBERO and RoboCasa pose loops have.
STEP_ROT_RAD = 0.05            # OmniGibson's IK is a ONE-STEP Jacobian solve (ik_controller.compute_control ->
                               # compute_ik_qpos): handed a goal far from the current pose it returns a joint
                               # target far from the current configuration, and the arm integrates that away
                               # from the goal instead of toward it. Measured 2026-09-11 on turning_on_radio:
                               # a 5 cm goal commanded outright drove the end effector 1.35 m away in 13 ticks.
MAX_TICKS_PER_MOVE = 250       # 30 Hz ticks one arm move may spend (8.3 s of simulated time; measured
                               # 2026-09-10: a reach across this robot's workspace does not settle inside 4 s)
GRIPPER_HOLD_TICKS = 30        # ... a gripper hold (1 s: the fingers close and the assisted grasp latches)

BASE_POS_TOL_M = 0.05          # the holonomic base parks to 5 cm ...
BASE_YAW_TOL_RAD = 0.05        # ... and ~3°
BASE_POS_GAIN = 2.0            # metres of error -> base velocity command (clipped to [-1, 1])
BASE_YAW_GAIN = 2.0            # radians of error -> base yaw command
BASE_MIN_CMD = 0.15            # a floor on the command while the error is outside tolerance, so the last
                               # centimetres close under a bare proportional law
MAX_TICKS_PER_BASE_MOVE = 300  # 10 s of simulated time; at 0.75 m/s that is about 7 m


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _floor(vec: np.ndarray, minimum: float) -> np.ndarray:
    """Scale a 2-D command up so its magnitude is at least ``minimum`` (never
    past the [-1, 1] the controller takes), keeping its direction."""
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9 or norm >= minimum:
        return vec
    return np.clip(vec * (minimum / norm), -1.0, 1.0)


def _advance(here: np.ndarray, there: np.ndarray, step: float) -> np.ndarray:
    """A point at most ``step`` metres from ``here`` on the way to ``there``."""
    delta = np.asarray(there, dtype=np.float64) - np.asarray(here, dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    return np.asarray(there, dtype=np.float64) if distance <= step else here + delta * (step / distance)


def _advance_rotation(here_wxyz: np.ndarray, there_wxyz: np.ndarray, step: float) -> np.ndarray:
    """An orientation at most ``step`` radians from ``here_wxyz`` toward ``there_wxyz``."""
    from scipy.spatial.transform import Rotation

    def _r(q):
        w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
        return Rotation.from_quat([x, y, z, w])

    a, b = _r(here_wxyz), _r(there_wxyz)
    delta = (b * a.inv()).as_rotvec()
    angle = float(np.linalg.norm(delta))
    if angle <= step:
        return np.asarray(there_wxyz, dtype=np.float64).reshape(4)
    x, y, z, w = (Rotation.from_rotvec(delta * (step / angle)) * a).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def _rot_error(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> float:
    """Geodesic angle between two ``(w, x, y, z)`` orientations, radians."""
    from scipy.spatial.transform import Rotation

    def _r(q):
        w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
        return Rotation.from_quat([x, y, z, w])

    return float(np.linalg.norm((_r(a_wxyz) * _r(b_wxyz).inv()).as_rotvec()))


class BehaviorEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    #: the world's controller layout — the challenge's own absolute joint control
    LAYOUT = "joint"

    def __init__(
        self,
        episodes: Sequence[Episode],
        body: BehaviorBody,
        max_ticks: int | None = None,
        tick_scale: float = 1.0,
        gpu_id: int = 0,
        render_mode: str | None = None,
        terminate_on_success: bool = True,
        data_path: str | None = None,
    ) -> None:
        if not episodes:
            raise ValueError("BehaviorEnv: no episodes")
        self.episodes: tuple[Episode, ...] = tuple(episodes)
        self.body = body
        self.max_ticks = int(max_ticks) if max_ticks is not None else None
        self.tick_scale = float(tick_scale)
        self.terminate_on_success = bool(terminate_on_success)
        self._cap: int | None = None      # this episode's tick cap (see _tick_cap)
        self.render_mode = render_mode
        self._by_id = {e.episode_id: e for e in self.episodes}
        self._world = BehaviorWorld(body, gpu_id=gpu_id, layout=self.LAYOUT, data_path=data_path)
        self._episode: Episode | None = None
        self._done = False
        self._succeeded = False
        self._ticks = 0          # every control tick of the episode, the settle ticks included
        self._settle = 0         # the settle ticks of this episode: outside the tick cap
        self._last_rgb: np.ndarray | None = None

        h, w = body.rgb.height, body.rgb.width
        obs = {"rgb": spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8),
               "proprio": spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32)}
        if body.wrist is not None:
            wrist = spaces.Box(0, 255, shape=(body.wrist.height, body.wrist.width, 3), dtype=np.uint8)
            obs["wrist"], obs["wrist_right"] = wrist, wrist
        self.observation_space = spaces.Dict(obs)
        # The challenge's action is unbounded (the arms take absolute joint angles with no command limits);
        # the true per-part limits are the robot's, enforced by its controllers.
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(23,), dtype=np.float64)

    # ---- episode -----------------------------------------------------------------------
    @property
    def episode(self) -> Episode:
        if self._episode is None:
            raise RuntimeError("BehaviorEnv: call reset() first")
        return self._episode

    @property
    def goal_index(self) -> int:
        return 0

    def current_goal(self):
        return self.episode.goal

    @property
    def ticks(self) -> int:
        return self._ticks

    @property
    def world(self) -> BehaviorWorld:
        return self._world

    def _select(self, options: dict[str, Any] | None) -> Episode:
        options = options or {}
        if "episode" in options:
            return self.episodes[int(options["episode"])]
        if "episode_id" in options:
            return self._by_id[str(options["episode_id"])]
        return self.episodes[int(self.np_random.integers(len(self.episodes)))]

    def _tick_cap(self, ep: Episode) -> int | None:
        """The episode's control-tick cap. BEHAVIOR's is per task — twice the
        mean length of that task's 200 human demonstrations (``eval.py``
        L146-153), carried in ``episode.info["max_steps"]``. ``tick_scale`` is
        the multiple of it this variant allows; a declared ``max_ticks``
        overrides it with one number for the line."""
        if self.max_ticks is not None:
            return self.max_ticks
        budget = ep.info.get("max_steps")
        return None if budget is None else int(round(self.tick_scale * int(budget)))

    # ---- gym API -----------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        ep = self._select(options)
        self._cap = self._tick_cap(ep)
        self._world.load_scene(ep.scene, max_steps=self._cap if self._cap is not None else 5000)
        self._episode = ep
        self._done = False
        self._succeeded = False
        self._ticks = self._settle = self._world.reset(ep.scene)
        obs = self._observe()
        info = self._facts()
        episode = ep.as_dict()
        if self._world.language:      # BDDL's own natural-language goal for this activity
            episode["instruction"] = ep.instruction or self._world.language
        info["episode"] = episode
        info["goal"] = to_dict(ep.goal)
        info["robot_base"] = self._world.robot_base()
        return obs, info

    def step(self, action):
        if self._done:
            raise RuntimeError("BehaviorEnv: episode is over; reset() first")
        self._tick(np.asarray(action, dtype=np.float64).reshape(-1))
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
        base_pos, base_yaw = self._world.base_pose()
        arms: dict[str, Any] = {}
        for arm in ARMS:
            pos, quat = self._world.eef_pose(arm)
            arms[arm] = {"eef_position": pos.tolist(), "eef_rotation": quat.tolist(),
                         "gripper_open": self._world.gripper_open(arm),
                         "grasping": self._world.is_grasping(arm)}
        return {
            "arms": arms,                                     # per arm: pose (world frame, metres, w x y z) + gripper
            "base_position": base_pos.tolist(),               # the holonomic base, world frame
            "base_yaw_rad": base_yaw,
            "trunk_qpos": self._world.trunk_qpos().tolist(),  # the four torso joints, radians
            "language": self._world.language,                 # BDDL's natural-language goal
            "success": self._succeeded,                       # latched: the goal held at some tick of this episode
            "q_score": self._world.q_score(),                 # the challenge's ranking metric for this rollout
            "goal_status": self._world.goal_status(),         # per-predicate: which indices hold now
            "n_predicates": self._world.n_predicates(),
            "travel_m": self._world.travel(),                 # base / left / right accumulated displacement
            "ticks": self._ticks,
            "tick_cap": self._cap,                            # this episode's control-tick cap
            "objects": self._world.objects(),                 # privileged: for metrics and debugging only
            "goal_index": 0,
            **extra,
        }


class BehaviorPoseEnv(BehaviorEnv):
    """The macro protocol on a two-armed mobile manipulator: one ``step`` is one
    base move, one absolute end-effector move (either arm or both), or one
    gripper hold (see the module docstring). ``info`` adds ``converged``,
    ``move_ticks``, ``move_kind``, ``position_error_m`` and
    ``rotation_error_rad``."""

    LAYOUT = "ik"

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
    def hold_action(left_gripper: float = -1.0, right_gripper: float = -1.0) -> np.ndarray:
        """The step action that holds both arms and commands the grippers
        (``+1`` close, ``-1`` open)."""
        a = np.zeros(POSE_ACTION_DIM, dtype=np.float64)
        for i, grip in enumerate((left_gripper, right_gripper)):
            a[i * ARM_ACTION:i * ARM_ACTION + 3] = np.inf
            a[i * ARM_ACTION + 6] = float(grip)
        return a

    @staticmethod
    def arm_action(arm: str, position, rotvec, gripper: float, other_gripper: float = -1.0) -> np.ndarray:
        """The step action that drives ONE arm to an absolute world pose while
        the other holds its pose with ``other_gripper``."""
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        i = ARMS.index(arm)
        grips = (gripper, other_gripper) if i == 0 else (other_gripper, gripper)
        a = BehaviorPoseEnv.hold_action(*grips)
        a[i * ARM_ACTION:i * ARM_ACTION + 3] = np.asarray(position, dtype=np.float64).reshape(3)
        a[i * ARM_ACTION + 3:i * ARM_ACTION + 6] = np.asarray(rotvec, dtype=np.float64).reshape(3)
        return a

    @staticmethod
    def base_action(dx: float, dy: float, dyaw: float,
                    left_gripper: float = -1.0, right_gripper: float = -1.0) -> np.ndarray:
        """The step action that drives the holonomic base by (dx, dy) metres and
        ``dyaw`` radians **in the base's own frame**, both arms holding."""
        a = BehaviorPoseEnv.hold_action(left_gripper, right_gripper)
        a[P_BASE] = [float(dx), float(dy), float(dyaw)]
        return a

    # ---- gym API -------------------------------------------------------------------------
    def step(self, action):
        if self._done:
            raise RuntimeError("BehaviorPoseEnv: episode is over; reset() first")
        a = np.asarray(action, dtype=np.float64).reshape(POSE_ACTION_DIM)
        grips = {arm: float(a[i * ARM_ACTION + 6]) for i, arm in enumerate(ARMS)}
        if np.any(a[P_BASE] != 0.0):
            return self._drive(a[P_BASE], grips)
        targets = {}
        for i, arm in enumerate(ARMS):
            position = a[i * ARM_ACTION:i * ARM_ACTION + 3]
            if np.all(np.isfinite(position)):
                targets[arm] = (position, a[i * ARM_ACTION + 3:i * ARM_ACTION + 6])
        if not targets:
            return self._hold(grips)
        return self._reach(targets, grips)

    def _finish(self, **extra: Any):
        obs = self._observe()
        return obs, 0.0, self._done, self._truncated(), self._facts(**extra)

    def _command(self, grips: dict[str, float]) -> np.ndarray:
        """A tick command that holds everything (the trunk and both arms at
        their current pose, the base still) with the given gripper commands."""
        cmd = self._world.no_op_action()
        slices = self._world.action_slices
        for arm, grip in grips.items():
            closed, opened = self._world.gripper_limits(arm)
            cmd[slices[f"gripper_{arm}"]] = closed + (1.0 - float(np.clip(grip, -1.0, 1.0))) / 2.0 * (opened - closed)
        return cmd

    def _reach(self, targets: dict[str, tuple[np.ndarray, np.ndarray]], grips: dict[str, float]):
        """Closed loop on the ``absolute_pose`` IK controllers, until every
        driven arm is inside tolerance of its FINAL target.

Each tick commands a goal at most ``STEP_POS_M`` /
        ``STEP_ROT_RAD`` from where the end effector is now, not the final
        target itself. That bound is what makes the loop stable: OmniGibson's
        IK is a one-step Jacobian solve, so a goal far from the current pose
        returns a joint target far from the current configuration and the arm
        integrates away from the goal. The LIBERO and RoboCasa pose loops bound
        their goal advance the same way.

        There is deliberately NO stall abort here, unlike the LIBERO loop. One
        was tried on 2026-09-11 and reverted: this arm's reach has a large
        initial transient — the end effector swings well past the target before
        the IK settles — so a 25-tick no-improvement window fires during the
        swing and leaves the arm mid-flight. With it the contract move ended
        1.46 m out; without it the same move converges. A target the arm cannot
        reach therefore costs the full ``MAX_TICKS_PER_MOVE``.

        The bounded goal is then re-expressed in the frame the controller reads
        (``ik_controller.py`` ``_update_goal`` L264-268 — see
        ``world.to_controller_frame``, which never assumes which link that
        frame hangs off)."""
        want = {arm: (np.asarray(p, dtype=np.float64).reshape(3), rotvec_to_quat(r)) for arm, (p, r) in targets.items()}
        slices = self._world.action_slices
        converged, ended, n = False, False, 0
        pos_err = rot_err = float("nan")
        while n < self.max_ticks_per_move:
            here = {arm: self._world.eef_pose(arm) for arm in want}
            errors = {arm: (float(np.linalg.norm(want[arm][0] - here[arm][0])),
                            _rot_error(want[arm][1], here[arm][1])) for arm in want}
            pos_err = max(e[0] for e in errors.values())
            rot_err = max(e[1] for e in errors.values())
            if pos_err < self.pos_tol_m and rot_err < self.rot_tol_rad:
                converged = True
                break
            cmd = self._command(grips)
            for arm, (target_pos, target_quat) in want.items():
                # The goal is a bounded step from where the end effector is NOW, so the IK is
                # never handed an error larger than STEP_POS_M and the arm tracks it. Walking a
                # fixed path from the start pose instead was tried and is worse: the goal then
                # runs ahead of an arm that lags, and the +5 cm contract move ended 1.18 m out
                # (2026-09-11) where this form lands it. The LIBERO loop measures from the
                # current pose for the same reason.
                pos, quat = here[arm]
                step_pos = _advance(pos, target_pos, STEP_POS_M)
                step_quat = _advance_rotation(quat, target_quat, STEP_ROT_RAD)
                local_pos, local_quat = self._world.to_controller_frame(arm, step_pos, step_quat)
                cmd[slices[f"arm_{arm}"]] = np.concatenate([local_pos, quat_to_rotvec(local_quat)])
            n += 1
            if self._tick(cmd):
                ended = True
                break
        if ended:
            pos_err = max(float(np.linalg.norm(t[0] - self._world.eef_pose(arm)[0])) for arm, t in want.items())
        return self._finish(converged=converged, move_ticks=n, move_kind="pose",
                            position_error_m=pos_err, rotation_error_rad=rot_err)

    def _drive(self, delta: np.ndarray, grips: dict[str, float]):
        """Closed loop on the holonomic base: (dx, dy, dyaw) in the base's own
        frame at the start of the move. The base part is
        ``HolonomicBaseJointController`` in velocity mode with input limits
        ``[-1, 1]³`` and output ``±[0.75, 0.75, 1.0]`` m/s · m/s · rad/s
        (``og_teleop_cfg.py`` L217-224), so the command is a proportional
        velocity."""
        slices = self._world.action_slices
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
            cmd = self._command(grips)
            cmd[slices["base"]] = [drive[0], drive[1], turn]
            n += 1
            if self._tick(cmd):
                break
        return self._finish(converged=converged, move_ticks=n, move_kind="base",
                            position_error_m=pos_err, rotation_error_rad=rot_err)

    def _hold(self, grips: dict[str, float]):
        n = 0
        for _ in range(self.hold_ticks):
            n += 1
            if self._tick(self._command(grips)):
                break
        return self._finish(converged=True, move_ticks=n, move_kind="gripper",
                            position_error_m=0.0, rotation_error_rad=0.0)
