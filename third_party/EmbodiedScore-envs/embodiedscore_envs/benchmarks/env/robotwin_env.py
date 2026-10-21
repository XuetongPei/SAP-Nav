"""RobotwinEnv — the environment body of the RoboTwin engine (layer L1), a
plain ``gymnasium.Env``; ``LiberoEnv`` is its single-arm manipulation twin
and ``HabitatEnv`` / ``IsaacEnv`` its navigation ones.

What it does: hold an episode list and a RobotwinBody, build an episode's
scene under RoboTwin's own seed, run actions, and report facts —
observation + BOTH arms' end-effector poses / gripper openings / success /
ticks / actor poses.

What it deliberately does not do: enforce the step budget (gym's
``TimeLimit`` counts steps; the tick cap below is RoboTwin's own per-task
cap), compute any metric (``ManipMetrics``), keep a "next episode" cursor,
or put the instruction into the observation (episode data travel in
``info``).

Two protocols on the same facts:

* ``RobotwinEnv`` — RoboTwin's own. One ``step`` is one ``take_action``
  call (``envs/_base_task.py:1486``), which is what RoboTwin's per-task cap
  in ``env_cfg/task_config/_eval_step_limit.yml`` counts. The action is the
  flat vector RoboTwin's evaluator feeds it
  (``scripts/eval_policy_xpolicylab.py:716``), in one of its two forms:

      ``action_type="qpos"``  ``[left arm joints (arm_dof), left gripper,
                                right arm joints (arm_dof), right gripper]``
                              — 14 wide on aloha-agilex. The joint targets
                              are time-parameterised by TOPP at 1/250 s and
                              run to completion.
      ``action_type="ee"``    ``[left x y z qw qx qy qz, left gripper,
                                right x y z qw qx qy qz, right gripper]``
                              — 16 wide. Each arm's pose is planned to by
                              the embodiment's planner and run to completion.

  Both grippers are RoboTwin's normalised value, 0 closed .. 1 open
  (``envs/robot/robot.py:628``). The layout is
  ``envs/_base_task.py:1504-1528`` verbatim.

* ``RobotwinPoseEnv`` — the macro protocol for agents that plan in poses,
  the bimanual generalisation of ``LiberoPoseEnv``. One ``step`` is an
  ABSOLUTE end-effector target FOR EACH ARM:

      ``[Lx, Ly, Lz, Lax, Lay, Laz, Lgrip,  Rx, Ry, Rz, Rax, Ray, Raz, Rgrip]``

  world frame, metres, axis-angle, 14 wide. There is no arm selector: an arm
  whose target position is not finite (``inf``) HOLDS — it keeps the pose it
  is in and only its gripper command applies, which is how a one-armed move
  is written and how ``LiberoPoseEnv``'s ``inf`` hold generalises to two
  arms. ``hold_action`` / ``arm_action`` build the common shapes. The
  gripper field is LIBERO's sign convention widened to RoboTwin's range:
  ``+1`` closed, ``-1`` open, and anything between maps linearly
  (``(1 - g) / 2`` is RoboTwin's normalised value), so a half-open grasp is
  expressible.

  One macro step is a closed loop over RoboTwin ``take_action(..., "ee")``
  calls: RoboTwin plans and executes the whole motion inside one call, so
  the loop usually converges on the first, and retries only while an arm is
  still outside tolerance and ``max_ticks_per_move`` calls are unspent.
  ``info["converged"]`` says which.

``terminated`` is RoboTwin's success (the task's own ``check_success``
predicate) when ``terminate_on_success`` is set — RoboTwin's own semantics
(``scripts/eval_policy_xpolicylab.py:1034``: an episode ends on
``eval_success`` or the step cap), the per-action protocol's default. The
macro protocol defaults it off: the episode runs until the agent's own stop
or the budget, and ``info["success"]`` / ``ManipMetrics`` latch the first
action the goal held, which is what RoboTwin would have scored.
``truncated`` fires at the episode's tick cap (settle steps excluded — they
are physics steps, not actions). ``reward`` is 0.0.

Facts are in SAPIEN's world frame: metres, z up, rotations ``(w, x, y, z)``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .schema import Episode, to_dict
from .sim.robotwin import ARMS, RobotwinBody, RobotwinWorld

POS_TOL_M = 0.01              # a macro move counts as reached inside this (LIBERO's tolerance)
ROT_TOL_RAD = 0.10            # ... and this
MAX_TICKS_PER_MOVE = 3        # RoboTwin plans and runs the whole motion per call; retries are for planner failures
GRIPPER_HOLD_TICKS = 2        # take_action calls a hold spends before giving up on the gripper reaching its command
GRIPPER_TOL = 0.05            # normalised gripper units a hold accepts as reached
ARM_ACTION = 7                # per arm in the macro protocol: x y z + axis-angle + gripper


def _wxyz(rotvec: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def _rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return Rotation.from_quat([x, y, z, w]).as_rotvec()


def _rot_error(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> float:
    """Geodesic angle between two ``(w, x, y, z)`` orientations, radians."""
    from scipy.spatial.transform import Rotation

    def _r(q):
        w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
        return Rotation.from_quat([x, y, z, w])

    return float(np.linalg.norm((_r(a_wxyz) * _r(b_wxyz).inv()).as_rotvec()))


def gripper_to_robotwin(g: float) -> float:
    """The macro protocol's gripper field to RoboTwin's normalised value:
    ``+1`` closed -> 0.0, ``-1`` open -> 1.0, linear in between."""
    return float(np.clip((1.0 - float(g)) / 2.0, 0.0, 1.0))


class RobotwinEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        episodes: Sequence[Episode],
        body: RobotwinBody,
        max_ticks: int | None = None,
        budget: Callable[[Any, Episode], int] | None = None,
        action_type: str = "qpos",
        gpu_id: int = 0,
        root: str | None = None,
        render_mode: str | None = None,
        terminate_on_success: bool = True,
    ) -> None:
        if not episodes:
            raise ValueError("RobotwinEnv: no episodes")
        if action_type not in ("qpos", "ee"):
            raise ValueError(f"RobotwinEnv: action_type must be 'qpos' or 'ee', got {action_type!r}")
        self.episodes: tuple[Episode, ...] = tuple(episodes)
        self.body = body
        self.action_type = action_type
        self.max_ticks = int(max_ticks) if max_ticks is not None else None
        self.budget = budget
        self.terminate_on_success = bool(terminate_on_success)
        self.render_mode = render_mode
        self._by_id = {e.episode_id: e for e in self.episodes}
        self._world = RobotwinWorld(body, gpu_id=gpu_id, root=root)
        self._episode: Episode | None = None
        self._done = False
        self._succeeded = False
        self._ticks = 0            # RoboTwin actions spent this episode
        self._tick_cap = 0         # this episode's cap on them
        self._last_rgb: np.ndarray | None = None

        h, w = body.head.height, body.head.width
        obs: dict[str, spaces.Space] = {
            "rgb": spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, shape=(body.qpos_dim,), dtype=np.float32),
        }
        if body.wrist is not None:
            wrist = spaces.Box(0, 255, shape=(body.wrist.height, body.wrist.width, 3), dtype=np.uint8)
            obs["left_wrist"], obs["right_wrist"] = wrist, wrist
        if body.front is not None:
            obs["front"] = spaces.Box(0, 255, shape=(body.front.height, body.front.width, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(obs)
        width = body.qpos_dim if action_type == "qpos" else body.ee_dim
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(width,), dtype=np.float64)

    # ---- episode -----------------------------------------------------------------------
    @property
    def episode(self) -> Episode:
        if self._episode is None:
            raise RuntimeError("RobotwinEnv: call reset() first")
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
    def world(self) -> RobotwinWorld:
        return self._world

    def _select(self, options: dict[str, Any] | None) -> Episode:
        options = options or {}
        if "episode" in options:
            return self.episodes[int(options["episode"])]
        if "episode_id" in options:
            return self._by_id[str(options["episode_id"])]
        return self.episodes[int(self.np_random.integers(len(self.episodes)))]

    def _cap(self, ep: Episode) -> int:
        """This episode's cap on RoboTwin actions: the explicit override, else
        the declaration's per-episode budget, else the task's own cap from
        ``_eval_step_limit.yml`` (carried in ``episode.info``)."""
        if self.max_ticks is not None:
            return self.max_ticks
        if self.budget is not None:
            return int(self.budget(self._world, ep))
        return int(ep.info["step_limit"])

    # ---- gym API -----------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        ep = self._select(options)
        self._world.load_scene(ep.scene)
        self._tick_cap = self._cap(ep)
        seed, attempts = self._world.reset(
            seed=int(ep.info["seed"]), episode_index=ep.index, step_limit=self._tick_cap,
            need_plan=bool(options.get("need_plan", False) if options else False),
            seed_window=int(ep.info.get("seed_window", 1)))
        self._episode = ep
        self._done = False
        self._succeeded = False
        self._ticks = 0
        obs = self._observe()
        info = self._facts()
        info["episode"] = ep.as_dict()
        info["goal"] = to_dict(ep.goal)
        info["robot_base"] = self._world.robot_base()
        info["step_budget"] = self._tick_cap
        info["seed"] = seed                  # the seed inside the episode's window that produced a settled scene
        info["seed_attempts"] = attempts
        return obs, info

    def step(self, action):
        if self._done:
            raise RuntimeError("RobotwinEnv: episode is over; reset() first")
        self._tick(np.asarray(action, dtype=np.float64).reshape(-1), self.action_type)
        obs = self._observe()
        return obs, 0.0, self._done, self._truncated(), self._facts()

    def _tick(self, action: np.ndarray, action_type: str) -> bool:
        """One RoboTwin action; True when the episode ended on it (success or
        the cap). RoboTwin's ``take_action`` is a silent no-op once its own
        counter reaches the cap, so the cap is checked here first."""
        if self._truncated():
            return True
        success = self._world.take_action(action, action_type=action_type)
        self._ticks += 1
        if success:
            self._succeeded = True
            if self.terminate_on_success:
                self._done = True
        return self._done or self._truncated()

    def _truncated(self) -> bool:
        return self._tick_cap > 0 and self._ticks >= self._tick_cap

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

    def _arms(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            pos, quat = self._world.arm_pose(arm)
            out[arm] = {
                "eef_position": pos.tolist(),          # x, y, z — world frame, metres
                "eef_rotation": quat.tolist(),         # w, x, y, z
                "gripper_open": self._world.gripper_open(arm),   # 0 closed .. 100 open
            }
        return out

    def _facts(self, **extra: Any) -> dict[str, Any]:
        return {
            "arms": self._arms(),
            "success": self._succeeded,                # latched: the goal held at some action of this episode
            "ticks": self._ticks,
            "objects": self._world.objects(),          # privileged: for metrics and debugging only
            "goal_index": 0,
            **extra,
        }


class RobotwinPoseEnv(RobotwinEnv):
    """The macro protocol: ``step`` takes an absolute end-effector target per
    arm (see the module docstring) and drives each arm there in a closed loop
    of RoboTwin ``take_action(..., "ee")`` calls. ``info`` adds ``converged``,
    ``move_ticks``, ``position_error_m`` / ``rotation_error_rad`` (the worst
    over the arms that were commanded) and a per-arm ``moves`` block."""

    def __init__(self, *args: Any, pos_tol_m: float = POS_TOL_M, rot_tol_rad: float = ROT_TOL_RAD,
                 max_ticks_per_move: int = MAX_TICKS_PER_MOVE, hold_ticks: int = GRIPPER_HOLD_TICKS,
                 terminate_on_success: bool = False, **kwargs: Any) -> None:
        kwargs.pop("action_type", None)
        super().__init__(*args, action_type="ee", terminate_on_success=terminate_on_success, **kwargs)
        self.pos_tol_m, self.rot_tol_rad = float(pos_tol_m), float(rot_tol_rad)
        self.max_ticks_per_move = int(max_ticks_per_move)
        self.hold_ticks = int(hold_ticks)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(2 * ARM_ACTION,), dtype=np.float64)

    # ---- the action shapes ----------------------------------------------------------------
    @staticmethod
    def hold_action(left_gripper: float = 0.0, right_gripper: float = 0.0) -> np.ndarray:
        """Both arms hold their pose; only the gripper commands apply
        (``+1`` close, ``-1`` open)."""
        block = [np.inf, np.inf, np.inf, 0.0, 0.0, 0.0]
        return np.array(block + [float(left_gripper)] + block + [float(right_gripper)], dtype=np.float64)

    @staticmethod
    def arm_action(arm: str, position: Sequence[float], rotvec: Sequence[float], gripper: float,
                   other_gripper: float = 0.0) -> np.ndarray:
        """Move one arm to ``position`` / ``rotvec``; the other holds its pose
        with ``other_gripper`` as its command."""
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        move = list(np.asarray(position, dtype=np.float64).reshape(3)) + \
            list(np.asarray(rotvec, dtype=np.float64).reshape(3)) + [float(gripper)]
        hold = [np.inf, np.inf, np.inf, 0.0, 0.0, 0.0, float(other_gripper)]
        return np.array(move + hold if arm == "left" else hold + move, dtype=np.float64)

    # ---- the loop --------------------------------------------------------------------------
    def _targets(self, action: np.ndarray) -> dict[str, dict[str, Any]]:
        """The action split per arm; a non-finite position means HOLD."""
        out: dict[str, dict[str, Any]] = {}
        for i, arm in enumerate(ARMS):
            block = action[i * ARM_ACTION:(i + 1) * ARM_ACTION]
            moving = bool(np.all(np.isfinite(block[:3])))
            pos, quat = self._world.arm_pose(arm)
            out[arm] = {
                "moving": moving,
                "position": block[:3].copy() if moving else pos,
                "rotation": _wxyz(block[3:6]) if moving else quat,
                "gripper": gripper_to_robotwin(block[6]),
            }
        return out

    def _ee_vector(self, targets: dict[str, dict[str, Any]]) -> np.ndarray:
        """RoboTwin's ``action_type='ee'`` vector: pose + gripper, per arm."""
        parts: list[float] = []
        for arm in ARMS:
            t = targets[arm]
            parts.extend(np.asarray(t["position"], dtype=np.float64).reshape(3).tolist())
            parts.extend(np.asarray(t["rotation"], dtype=np.float64).reshape(4).tolist())
            parts.append(float(t["gripper"]))
        return np.asarray(parts, dtype=np.float64)

    def _errors(self, targets: dict[str, dict[str, Any]]) -> dict[str, tuple[float, float]]:
        out = {}
        for arm in ARMS:
            pos, quat = self._world.arm_pose(arm)
            t = targets[arm]
            out[arm] = (float(np.linalg.norm(np.asarray(t["position"]) - pos)), _rot_error(t["rotation"], quat))
        return out

    def step(self, action):
        if self._done:
            raise RuntimeError("RobotwinPoseEnv: episode is over; reset() first")
        a = np.asarray(action, dtype=np.float64).reshape(2 * ARM_ACTION)
        targets = self._targets(a)
        if not any(targets[arm]["moving"] for arm in ARMS):
            return self._hold(targets)

        converged, ended, n = False, False, 0
        errors = self._errors(targets)
        while n < self.max_ticks_per_move:
            errors = self._errors(targets)
            if all(errors[arm][0] < self.pos_tol_m and errors[arm][1] < self.rot_tol_rad
                   for arm in ARMS if targets[arm]["moving"]):
                converged = True
                break
            n += 1
            if self._tick(self._ee_vector(targets), "ee"):
                ended = True
                break
        if not ended and not converged:
            errors = self._errors(targets)
            converged = all(errors[arm][0] < self.pos_tol_m and errors[arm][1] < self.rot_tol_rad
                            for arm in ARMS if targets[arm]["moving"])
        return self._result(targets, errors, converged, n)

    def _hold(self, targets: dict[str, dict[str, Any]]):
        """Neither arm moves: re-issue the pose the arms are already in with
        the new gripper commands until the grippers reach them. RoboTwin ramps
        a gripper across the motion it is bundled with, so a hold is how a
        grasp or a release is executed."""
        n = 0
        while n < self.hold_ticks:
            if all(abs(self._world.gripper_value(arm) - targets[arm]["gripper"]) < GRIPPER_TOL for arm in ARMS):
                break
            n += 1
            if self._tick(self._ee_vector(targets), "ee"):
                break
        return self._result(targets, self._errors(targets), True, n)

    def _result(self, targets: dict[str, dict[str, Any]], errors: dict[str, tuple[float, float]],
                converged: bool, n: int):
        commanded = [arm for arm in ARMS if targets[arm]["moving"]] or list(ARMS)
        obs = self._observe()
        info = self._facts(
            converged=bool(converged),
            move_ticks=int(n),
            position_error_m=max(errors[arm][0] for arm in commanded),
            rotation_error_rad=max(errors[arm][1] for arm in commanded),
            moves={arm: {"commanded": targets[arm]["moving"], "position_error_m": errors[arm][0],
                         "rotation_error_rad": errors[arm][1]} for arm in ARMS},
        )
        return obs, 0.0, self._done, self._truncated(), info
