"""CalvinEnv — the environment body of the CALVIN engine (layer L1), a plain
``gymnasium.Env``; ``LiberoEnv`` / ``RobocasaEnv`` are its manipulation
twins. (The name collides with the upstream distribution ``calvin_env``,
which only ``sim/calvin/world.py`` imports; python 3 has no implicit
relative imports, so the two never shadow each other.)

What it does: hold an episode list and a CalvinBody, put the play table into
an episode's initial state, run the simulator, drive **the chain**, and
report facts — observation + end-effector pose / gripper opening / the
current instruction / chain progress / ticks.

What it deliberately does not do: enforce the episode budget (gym's
``TimeLimit`` counts steps; the caps below are per sub-task), compute any
metric (``ChainMetrics``), keep a "next episode" cursor, or put the
instruction into the observation (episode data travel in ``info``).

## The chain

A CALVIN episode is not one task but an ordered chain of five
language-conditioned sub-tasks, and the protocol is the evaluator's
(``calvin_models/calvin_agent/evaluation/evaluate_policy.py``
``evaluate_sequence``:124 / ``rollout``:147):

* the initial state is reset once, for the whole chain (:128-129);
* sub-task *i* is announced by its language annotation and rolled out for at
  most ``EP_LEN`` = 360 control ticks (:38, :160);
* **the task oracle closes it, not the agent**: every tick the evaluator
  asks ``Tasks.get_task_info_for_set(start_info, current_info, {subtask})``
  (:172) — a predicate over the change since the sub-task's own start
  snapshot — and returns as soon as it holds (:176);
* on success the chain advances and the next instruction is revealed, on
  failure ``evaluate_sequence`` **returns immediately** (:143): the chain
  ends at the first failure or at five.

So there is no SUBTASK_STOP here: the sub-goals of the ``GoalSequence`` are
closed automatically. ``info["subtask_closed"]`` is True on the step one
closed, and ``info["instruction"]`` / ``info["goal_index"]`` then already
name the next one — which is how the bareES bridge announces it to the agent
(the GOAT lines do the same through the tool-result channel, but there the
agent declares the close).

``terminated`` is the chain completed (all five); ``truncated`` is a
sub-task that ran out of its budget, which ends the chain short. ``reward``
is 0.0. Every step also refuses once the chain is over.

## Two protocols on the same facts

* ``CalvinEnv`` — CALVIN's own: one ``step`` is one 30 Hz control tick and
  the action is the 7-D relative command ``[dx, dy, dz, dax, day, daz,
  gripper]``, a unit being 2 cm / 0.05 rad of *target-pose* shift
  (``sim/calvin/world.py`` transcribes the scaling and its source); gripper
  ``+1`` opens and ``-1`` closes, CALVIN's sign. Budget: ``EP_LEN`` ticks per
  sub-task.
* ``CalvinPoseEnv`` — the macro protocol for agents that plan in poses: the
  action is an ABSOLUTE end-effector target ``[x, y, z, ax, ay, az,
  gripper]`` (world frame, metres, axis-angle) with the package's gripper
  sign (**> 0 closes**, as on the LIBERO / RoboCasa lines; the world inverts
  it for CALVIN). Unlike those engines this needs no bounded-delta loop:
  ``Robot.apply_action`` takes an absolute pose natively, so a move holds
  the target until the TCP is within tolerance, ``max_ticks_per_move`` is
  spent, or the position error has stalled. ``info["converged"]`` says which,
  ``info["stalled"]`` whether it gave up. A target whose position is ``inf``
  is a HOLD: ``hold_ticks`` ticks at the current pose with the gripper
  command — how a grasp or a release is executed; ``hold_action(gripper)``
  builds one. Budget: ``max_moves_per_subtask`` macro moves per sub-task,
  under a tick guard of ``max_ticks`` ticks per sub-task.

Facts are in pybullet's world frame: metres, z up, ``eef_rotation`` as
``(w, x, y, z)``.
"""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .schema import Episode, to_dict
from .sim.calvin import CalvinBody, CalvinWorld
from .sim.calvin.world import GRIPPER_CLOSE, GRIPPER_OPEN, mat_to_wxyz, rotvec_to_euler

POS_TOL_M = 0.005           # the steady-state error of CALVIN's bullet IK + position control is 2-4 mm over
                            # the table (measured 2026-09-10 at four targets); 1 cm stopped the loop 7 mm out
ROT_TOL_RAD = 0.05          # ... and its rotation residual under 0.01 rad
MAX_TICKS_PER_MOVE = 120    # 30 Hz: 4 s of holding one target — a reachable pose lands in ~30 ticks
STALL_TICKS = 20            # give up a move whose error has not improved by STALL_GAIN_M over this many ticks
STALL_GAIN_M = 0.002
GRIPPER_HOLD_TICKS = 20     # CALVIN's fingers close in ~10 ticks at gripper_force 200

ACTION_DIM = 7


class CalvinEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        episodes: Sequence[Episode],
        body: CalvinBody,
        max_ticks: int | None = None,
        max_moves_per_subtask: int | None = None,
        gpu_id: int = 0,
        render_mode: str | None = None,
    ) -> None:
        if not episodes:
            raise ValueError("CalvinEnv: no episodes")
        self.episodes: tuple[Episode, ...] = tuple(episodes)
        self.body = body
        self.max_ticks = int(max_ticks) if max_ticks is not None else None            # per sub-task
        self.max_moves_per_subtask = int(max_moves_per_subtask) if max_moves_per_subtask else None
        self.render_mode = render_mode
        self._by_id = {e.episode_id: e for e in self.episodes}
        self._world = CalvinWorld(body, gpu_id=gpu_id)
        self._episode: Episode | None = None
        self._sequence: tuple[str, ...] = ()
        self._instructions: tuple[str, ...] = ()
        self._idx = 0            # the sub-task in force
        self._chain = 0          # sub-tasks solved consecutively from the start — CALVIN's per-episode result
        self._start_info: dict[str, Any] | None = None     # the oracle's reference for the sub-task in force
        self._terminated = False
        self._truncated = False
        self._closed_now = False
        self._ticks = 0          # every control tick of the episode
        self._settle = 0
        self._sub_ticks = 0      # control ticks spent on the sub-task in force
        self._sub_moves = 0      # step() calls spent on the sub-task in force
        self._last_rgb: np.ndarray | None = None

        h, w = body.rgb.height, body.rgb.width
        obs = {"rgb": spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8),
               "proprio": spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32)}
        if body.wrist is not None:
            obs["wrist"] = spaces.Box(0, 255, shape=(body.wrist.height, body.wrist.width, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(obs)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)

    # ---- episode -----------------------------------------------------------------------
    @property
    def episode(self) -> Episode:
        if self._episode is None:
            raise RuntimeError("CalvinEnv: call reset() first")
        return self._episode

    @property
    def goal_index(self) -> int:
        """The sub-task in force — the package's sub-goal cursor (GOAT's too)."""
        return self._idx

    def current_goal(self):
        """The ``ManipGoal`` of the sub-task in force, or None once the chain is over."""
        goals = self.episode.goal.goals
        return goals[self._idx] if self._idx < len(goals) else None

    @property
    def ticks(self) -> int:
        return self._ticks

    @property
    def chain_length(self) -> int:
        return self._chain

    @property
    def over(self) -> bool:
        return self._terminated or self._truncated

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
        self._sequence = tuple(ep.info["sequence"])
        self._instructions = tuple(ep.info["instructions"])
        self._idx = self._chain = 0
        self._terminated = self._truncated = self._closed_now = False
        self._ticks = self._settle = self._world.reset(np.asarray(ep.info["robot_obs"], dtype=np.float64),
                                                       np.asarray(ep.info["scene_obs"], dtype=np.float64))
        self._sub_ticks = self._sub_moves = 0
        self._start_info = self._world.snapshot()
        obs = self._observe()
        info = self._facts()
        info["episode"] = ep.as_dict()
        info["goal"] = to_dict(ep.goal)
        info["robot_base"] = self._world.robot_base()
        return obs, info

    def step(self, action):
        self._guard()
        self._sub_moves += 1
        self._tick_relative(np.asarray(action, dtype=np.float64).reshape(ACTION_DIM))
        self._charge_move()
        obs = self._observe()
        return obs, 0.0, self._terminated, self._truncated, self._facts()

    def _guard(self) -> None:
        """Refuse a step past the end of the chain, and clear the
        "a sub-task closed" flag so it describes THIS step only."""
        if self.over:
            raise RuntimeError("CalvinEnv: the chain is over; reset() first")
        self._closed_now = False

    # ---- the chain ---------------------------------------------------------------------
    def _tick_relative(self, action: np.ndarray) -> bool:
        self._world.tick_relative(action)
        return self._after_tick()

    def _tick_absolute(self, position: np.ndarray, euler: np.ndarray, gripper: float) -> bool:
        self._world.tick_absolute(position, euler, gripper)
        return self._after_tick()

    def _after_tick(self) -> bool:
        """Count the tick, ask the oracle, advance or end the chain. True when
        the chain is over or a sub-task closed on this tick (either way the
        caller must stop feeding the current move)."""
        self._ticks += 1
        self._sub_ticks += 1
        if self._world.achieved(self._start_info, {self._sequence[self._idx]}):
            self._close_subtask()
            return True
        if self.max_ticks is not None and self._sub_ticks >= self.max_ticks:
            self._truncated = True          # the sub-task ran out of ticks: the chain ends here
            return True
        return False

    def _close_subtask(self) -> None:
        """The oracle says the sub-task in force is solved: score it, reveal
        the next instruction, restart the sub-task's budget and its oracle
        reference (``rollout`` snapshots ``start_info`` per sub-task)."""
        self._chain += 1
        self._idx += 1
        self._closed_now = True
        if self._idx >= len(self._sequence):
            self._terminated = True         # all five: the chain is complete
            return
        self._sub_ticks = self._sub_moves = 0
        self._start_info = self._world.snapshot()

    def _charge_move(self) -> None:
        """A macro-move budget, when the variant has one: a sub-task that
        spends ``max_moves_per_subtask`` moves without closing has failed,
        which ends the chain — the same rule as the tick budget."""
        if self._closed_now or self.over or self.max_moves_per_subtask is None:
            return
        if self._sub_moves >= self.max_moves_per_subtask:
            self._truncated = True

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
        done = self._idx >= len(self._sequence)
        facts = {
            "eef_position": pos.tolist(),                  # x, y, z — world frame, metres
            "eef_rotation": mat_to_wxyz(rot).tolist(),     # w, x, y, z
            "gripper_open": self._world.gripper_open(),    # 0 closed .. 100 open
            "instruction": "" if done else self._instructions[self._idx],   # the sub-task in force
            "subtask": "" if done else self._sequence[self._idx],
            "goal_index": self._idx,
            "n_goals": len(self._sequence),
            "subtask_closed": self._closed_now,            # one closed on this step: announce the next
            "chain_length": self._chain,                   # CALVIN's per-episode result (0..5)
            "success": self._chain >= 1,                   # at least one sub-task solved
            "chain_complete": self._terminated,
            "ticks": self._ticks,
            "subtask_ticks": self._sub_ticks,
            "subtask_tick_cap": self.max_ticks,
            "subtask_moves": self._sub_moves,
            "subtask_move_cap": self.max_moves_per_subtask,
            "objects": self._world.objects(),              # privileged: for metrics and debugging only
            **extra,
        }
        return facts

    @property
    def world(self) -> CalvinWorld:
        return self._world


class CalvinPoseEnv(CalvinEnv):
    """The macro protocol: ``step([x, y, z, ax, ay, az, gripper])`` holds an
    absolute end-effector target until the TCP arrives (see the module
    docstring); a position of ``inf`` settles the gripper instead.
    ``info`` adds ``converged``, ``stalled``, ``move_ticks``,
    ``position_error_m`` and ``rotation_error_rad``."""

    def __init__(self, *args: Any, pos_tol_m: float = POS_TOL_M, rot_tol_rad: float = ROT_TOL_RAD,
                 max_ticks_per_move: int = MAX_TICKS_PER_MOVE, hold_ticks: int = GRIPPER_HOLD_TICKS,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hold_ticks = int(hold_ticks)
        self.pos_tol_m, self.rot_tol_rad = float(pos_tol_m), float(rot_tol_rad)
        self.max_ticks_per_move = int(max_ticks_per_move)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(ACTION_DIM,), dtype=np.float64)

    @staticmethod
    def hold_action(gripper: float) -> np.ndarray:
        """The step action that holds the pose and commands the gripper
        (the package's sign: + closes, - opens)."""
        return np.array([np.inf, np.inf, np.inf, 0.0, 0.0, 0.0, float(gripper)], dtype=np.float64)

    @staticmethod
    def pose_action(position, rotvec, gripper: float) -> np.ndarray:
        a = np.zeros(ACTION_DIM, dtype=np.float64)
        a[:3] = np.asarray(position, dtype=np.float64).reshape(3)
        a[3:6] = np.asarray(rotvec, dtype=np.float64).reshape(3)
        a[6] = float(gripper)
        return a

    def step(self, action):
        self._guard()
        from scipy.spatial.transform import Rotation

        a = np.asarray(action, dtype=np.float64).reshape(ACTION_DIM)
        # the package's macro gripper sign (+ closes) into CALVIN's (+ opens)
        grip = GRIPPER_CLOSE if a[6] > 0 else GRIPPER_OPEN
        self._sub_moves += 1
        if not np.all(np.isfinite(a[:3])):
            return self._hold(grip)
        target_pos = a[:3]
        target_rot = Rotation.from_rotvec(a[3:6]).as_matrix()
        target_euler = rotvec_to_euler(a[3:6])
        converged, ended, stalled, n = False, False, False, 0
        pos_err = rot_err = float("nan")
        best_err, best_at = float("inf"), 0
        while n < self.max_ticks_per_move:
            pos, rot = self._world.eef_pose()
            pos_err = float(np.linalg.norm(target_pos - pos))
            rot_err = float(np.linalg.norm(Rotation.from_matrix(target_rot @ rot.T).as_rotvec()))
            if pos_err < self.pos_tol_m and rot_err < self.rot_tol_rad:
                converged = True
                break
            err = pos_err + 0.1 * rot_err          # 0.1 rad counts as 1 cm
            if err < best_err - STALL_GAIN_M:
                best_err, best_at = err, n
            elif n - best_at >= STALL_TICKS:
                stalled = True
                break
            n += 1
            if self._tick_absolute(target_pos, target_euler, grip):
                ended = True
                break
        if ended:
            pos, _ = self._world.eef_pose()
            pos_err = float(np.linalg.norm(target_pos - pos))
        self._charge_move()
        obs = self._observe()
        info = self._facts(converged=converged, stalled=stalled, move_ticks=n, position_error_m=pos_err,
                           rotation_error_rad=rot_err)
        return obs, 0.0, self._terminated, self._truncated, info

    def _hold(self, grip: float):
        """Settle the gripper at the current target pose — CALVIN has no
        zero-motion action, so the pose is re-commanded as the absolute
        target it already holds."""
        pos, rot = self._world.eef_pose()
        from scipy.spatial.transform import Rotation

        euler = Rotation.from_matrix(rot).as_euler("xyz")
        n = 0
        for _ in range(self.hold_ticks):
            n += 1
            if self._tick_absolute(pos, euler, grip):
                break
        self._charge_move()
        obs = self._observe()
        return obs, 0.0, self._terminated, self._truncated, self._facts(
            converged=True, stalled=False, move_ticks=n, position_error_m=0.0, rotation_error_rad=0.0)
