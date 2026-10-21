"""BehaviorWorld — the simulator facade of the BEHAVIOR engine (layer L0): the
only module that imports omnigibson / isaacsim, lazily, at construction.

It owns one ``og.Environment`` at a time, loads a task instance into it, runs
control ticks, and reads facts: the frames, each arm's end-effector pose and
gripper opening, the mobile base pose, BEHAVIOR's own success and its
per-predicate goal status, and the accumulated travel the challenge's
efficiency metrics are built from. It never counts steps or budgets (the env
body does) and never decides an action (the bodies do).

**Isaac boots once per process.** ``omnigibson/simulator.py`` L386 asserts
``og.sim is None`` when a simulator is created, and ``_launch_app`` L148-240
builds Isaac's ``SimulationApp`` once into the module global ``og.app``
(``_launch_simulator`` L342-344 is guarded by ``if not og.app:``). What *can*
be repeated in the same process is the stage: ``og.clear()``
(``omnigibson/__init__.py`` L70-130) tears the simulator down and relaunches
it against the same app — "Clear the stage and then call launch again to make
og.sim point to a new simulator instance". So this world never restarts Isaac
and reloads at the cheapest level the change allows:

    same scene model, same task   -> replay the instance state only
    new task or new scene model   -> ``og.clear()`` + a fresh ``og.Environment`` (a task's
                                     objects live in its own scene file, so ``update_task``
                                     cannot serve a task change — see ``load_scene``)

``og.shutdown()`` is never called: with ``og.app`` alive it closes the app,
and the SIGINT handler installed at ``omnigibson/__init__.py`` L166 routes
into it — ``close()`` here only clears the stage.

**Two controller layouts, one robot.** ``layout="joint"`` is the challenge's
own (``joylo/gello/robots/sim_robot/og_teleop_cfg.py`` ``R1_CONTROLLER_CONFIG``
L181-232): base velocity, trunk and both arms ``JointController`` in ABSOLUTE
position, both grippers ``MultiFingerGripperController`` in ``"smooth"``
mode — a 23-D action, ``[base 3 | torso 4 | left arm 7 | left gripper 1 |
right arm 7 | right gripper 1]`` (``learning/utils/eval_utils.py``
``ACTION_QPOS_INDICES["R1Pro"]`` L49-58, in the order
``r1.py`` ``_raw_controller_order`` L166-170 gives). ``layout="ik"`` replaces
the two arm entries with ``InverseKinematicsController`` in ``mode:
"absolute_pose"`` — one of the four substitutions the challenge's own
``docs/challenge/evaluation.md`` § "Configure Robot Action Space" documents
for participants, applied through the same hook the evaluator uses
(``eval.py`` L144-145, ``cfg["robots"][0]["controller_config"].update(...)``)
— so an arm command is an absolute end-effector pose ``[x, y, z, ax, ay, az]``
(``controllers/ik_controller.py`` ``_update_goal`` L268-288), expressed in the
frame the controller's own reference pose lives in. That frame is the
ARTICULATION ROOT, not ``robot.get_position_orientation()``: on a holonomic
base the two are different links and confusing them makes every reach miss by
however far the base has driven. ``to_controller_frame`` composes through the
controller's own reference so the frame never has to be named.

The action layout is never assumed: ``action_slices`` is read off the live
robot's ``controller_order`` and each controller's ``command_dim``.

Frames: OmniGibson world frame, metres, z up. Quaternions leave OmniGibson as
``(x, y, z, w)`` and are converted here to the package's ``(w, x, y, z)``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .body import ARMS, CAMERA_PRIMS, ROBOT_NAME, BehaviorBody
from .scene import BehaviorSceneRef

# eval.py L48-50 — the challenge's own episode bookkeeping.
NUM_EVAL_INSTANCES = 10
NUM_TRAIN_INSTANCES = 200

# eval.py L281-286 — physics steps with every task entity held still after an instance load.
SETTLE_PHYSICS_STEPS = 25

# eval.py L58-59 — the evaluator widens the grasp window for hard grasps.
GRASP_WINDOW_S = 0.75


def _import_omnigibson() -> Any:
    try:
        import omnigibson as og
    except ImportError as e:
        raise ImportError(
            "the BEHAVIOR lines need the `omnigibson` package (BEHAVIOR-1K v3.7.2 + Isaac Sim 4.5.0) in this "
            "interpreter — see INSTALL-behavior.md"
        ) from e
    return og


def _xyzw_to_wxyz(q: Any) -> np.ndarray:
    q = np.asarray(_to_numpy(q), dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _wxyz_to_xyzw(q: Any) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _to_numpy(value: Any) -> np.ndarray:
    """torch tensors, numpy arrays and sequences alike -> a numpy array."""
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach().cpu().numpy()
    return np.asarray(value)


def _tensor(value: Any) -> Any:
    """-> a float32 torch tensor. ``omnigibson.utils.transform_utils`` is
    ``torch.compile``d and rejects anything else, while OmniGibson's own
    compute backend hands numpy back for some reads (the control dict's, with
    ``gm.USE_GPU_DYNAMICS`` off)."""
    import torch as th

    if isinstance(value, th.Tensor):
        return value.to(dtype=th.float32)
    return th.as_tensor(np.asarray(value, dtype=np.float32))


def quat_to_rotvec(quat_wxyz: Any) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(_wxyz_to_xyzw(quat_wxyz)).as_rotvec().astype(np.float64)


def rotvec_to_quat(rotvec: Any) -> np.ndarray:
    """axis-angle -> ``(w, x, y, z)``."""
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


class BehaviorWorld:
    """One BEHAVIOR-1K environment. ``layout`` picks the controller set:
    ``"joint"`` the challenge's own, ``"ik"`` its documented absolute-pose
    substitution for the two arms."""

    def __init__(self, body: BehaviorBody, gpu_id: int = 0, layout: str = "joint",
                 data_path: str | os.PathLike | None = None) -> None:
        if layout not in ("joint", "ik"):
            raise ValueError(f"BehaviorWorld: layout must be 'joint' or 'ik', got {layout!r}")
        self.body = body
        self.gpu_id = int(gpu_id)
        self.layout = layout
        self._data_path = str(data_path) if data_path else None
        self._env: Any = None
        self._model_key: str | None = None
        self._task_key: tuple[str, str, int] | None = None
        self._obs: dict[str, Any] | None = None
        self._info: dict[str, Any] = {}
        self._success = False
        self._scene: BehaviorSceneRef | None = None
        self._initial_predicates: list[list[bool]] = []
        self._travel = {"base": 0.0, "left": 0.0, "right": 0.0}
        self._last_points: dict[str, np.ndarray] = {}

    # ---- boot ----------------------------------------------------------------------------
    def _prepare(self) -> Any:
        """Set every global BEHAVIOR's evaluator sets before the environment is
        built — ``gm`` macros lock on first read (``macros.py`` L20-31), so this
        must run before any ``og.Environment`` exists (``eval.py`` L53-59)."""
        if self._data_path:
            os.environ.setdefault("OMNIGIBSON_DATA_PATH", self._data_path)
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
        os.environ.setdefault("OMNIGIBSON_GPU_ID", str(self.gpu_id))
        og = _import_omnigibson()
        from omnigibson.macros import gm, macros

        # A second world in one process (the tests build several stacks) sets the same
        # values again; a macro that has been read locks, so the write goes through
        # ``unlocked()`` — the values are the evaluator's either way.
        with gm.unlocked():
            gm.ENABLE_FLATCACHE = True        # eval.py L53
            gm.USE_GPU_DYNAMICS = False       # eval.py L54
            gm.ENABLE_TRANSITION_RULES = True  # eval.py L55
            gm.HEADLESS = True
        with macros.unlocked():
            macros.robots.manipulation_robot.GRASP_WINDOW = GRASP_WINDOW_S   # eval.py L58-59
        return og

    def _config(self, scene: BehaviorSceneRef, max_steps: int) -> dict[str, Any]:
        """The challenge's environment config for this task, then the two edits
        the evaluator makes: the observation modalities and the timeout."""
        from gello.robots.sim_robot.og_teleop_utils import generate_robot_config, load_available_tasks
        from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES, generate_basic_environment_config

        available = load_available_tasks()
        if scene.task not in available:
            raise KeyError(f"{scene.task}: not in joylo/sampled_task/available_tasks.yaml")
        task_cfg = available[scene.task][scene.definition_id]
        if str(task_cfg["scene_model"]) != scene.scene_model:
            raise ValueError(f"{scene.task}: available_tasks.yaml says scene {task_cfg['scene_model']!r}, "
                             f"the declaration says {scene.scene_model!r}")
        cfg = generate_basic_environment_config(task_name=scene.task, task_cfg=task_cfg)
        cfg["env"]["action_frequency"] = int(self.body.action_freq)
        cfg["env"]["rendering_frequency"] = int(self.body.render_freq)
        cfg["env"]["physics_frequency"] = int(self.body.physics_freq)
        cfg["task"]["activity_definition_id"] = int(scene.definition_id)
        cfg["task"]["termination_config"]["max_steps"] = int(max_steps)     # eval.py L146-153
        cfg["task"]["include_obs"] = False                                  # eval.py L154
        if self.body.partial_scene_load:
            cfg["scene"]["load_room_types"] = self._relevant_rooms(scene, task_cfg)
        cfg["robots"] = [generate_robot_config(task_name=scene.task, task_cfg=task_cfg)]
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]             # eval.py L142
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES[self.body.robot].keys())   # eval.py L143
        if self.layout == "ik":
            cfg["robots"][0]["controller_config"].update(self._ik_controllers())                # eval.py L144-145
        return cfg

    @staticmethod
    def _relevant_rooms(scene: BehaviorSceneRef, task_cfg: dict[str, Any]) -> list[str]:
        """eval.py L133-136 — the rooms ``partial_scene_load`` keeps."""
        from gello.robots.sim_robot.og_teleop_utils import augment_rooms, get_task_relevant_room_types

        rooms = get_task_relevant_room_types(activity_name=scene.task)
        return list(augment_rooms(rooms, task_cfg["scene_model"], scene.task))

    @staticmethod
    def _ik_controllers() -> dict[str, Any]:
        """The arm substitution ``docs/challenge/evaluation.md`` § "Configure
        Robot Action Space" documents: an absolute end-effector pose per arm.
        ``absolute_pose`` requires both command limits to be None
        (``ik_controller.py`` L140-142). Everything else is OmniGibson's own IK
        config for this robot's arms verbatim (``manipulation_robot.py``
        ``_default_arm_ik_controller_configs`` L1313-1336) — which is the
        point: only ``mode`` differs from what OmniGibson would build itself.

        In particular it does NOT set ``pos_kp``. An earlier version carried the
        challenge's ``pos_kp: 150`` across from the ``JointController`` this
        replaces, which was an invention: OmniGibson leaves the gain unset for
        an IK controller and its default of 50 applies. Measuring 2026-09-11
        settled it — the arm diverged the same way at 150 and at 50, so the
        gain was never the cause (that is the unbounded goal, see
        ``behavior_env.STEP_POS_M``). The gain is gone because it was not ours
        to choose, not because it was the bug."""
        return {
            f"arm_{arm}": {
                "name": "InverseKinematicsController",
                "mode": "absolute_pose",
                "command_input_limits": None,
                "command_output_limits": None,
                "use_impedances": False,
                "smoothing_filter_size": 2,
            }
            for arm in ARMS
        }

    def load_scene(self, scene: BehaviorSceneRef, max_steps: int) -> None:
        """Make the environment hold ``scene``'s task, at the cheapest level the
        change allows (see the module docstring)."""
        og = self._prepare()
        if self._env is None:
            self._env = og.Environment(configs=self._config(scene, max_steps))
        elif scene.model_key != self._model_key or scene.task_key != self._task_key:
            # A task change rebuilds the stage too: the challenge loads a scene file PER task
            # instance (``<scene>_task_<task>_instances``), so the objects of a new task are
            # not in the scene that was loaded for the old one — ``env.update_task`` then
            # fails in behavior_task.py L433 ("BDDL object instance … should exist in cached
            # metadata from loaded scene"). Measured 2026-09-10 when the env checker reset to
            # a random episode.
            og.clear()
            self._env = og.Environment(configs=self._config(scene, max_steps))
        else:
            # Same task: only the budget can differ. behavior_task.py L197-202 names the condition "timeout".
            self._env.task_config["termination_config"]["max_steps"] = int(max_steps)
            self.task._termination_conditions["timeout"]._max_steps = int(max_steps)
        self._apply_camera_rig()
        self._model_key, self._task_key, self._scene = scene.model_key, scene.task_key, scene
        self._obs = None

    def _apply_camera_rig(self) -> None:
        """The body's resolutions onto the robot's three cameras, then reload the
        observation space — what the challenge's own wrappers do
        (``learning/wrappers/rgb_low_res_wrapper.py`` L16-29)."""
        robot = self._env.robots[0]
        for camera_id, prim in CAMERA_PRIMS.items():
            sensor_name = prim.split("::")[1]
            sensor = robot.sensors.get(sensor_name)
            if sensor is None:
                continue
            if camera_id == self.body.rgb_camera:
                sensor.horizontal_aperture = float(self.body.head_aperture_mm)
            if camera_id in self.body.cameras:
                width, height = self.body.resolution_of(camera_id)
                sensor.image_width, sensor.image_height = int(width), int(height)
        self._env.load_observation_space()

    def close(self) -> None:
        """Clear the stage. Isaac's app stays up: it cannot be relaunched in this
        process, and ``og.shutdown()`` would take the process with it."""
        if self._env is None:
            return
        try:
            og = _import_omnigibson()
            og.clear()
        except Exception:
            pass
        self._env, self._model_key, self._task_key, self._obs = None, None, None, None

    @property
    def env(self) -> Any:
        return self._env

    @property
    def task(self) -> Any:
        return self._env.task

    @property
    def robot(self) -> Any:
        return self._env.robots[0]

    # ---- the action vector ----------------------------------------------------------------
    @property
    def action_slices(self) -> dict[str, slice]:
        """Where each controller's command sits in the flat action, read off the
        live robot (``robot.controller_order`` and each controller's
        ``command_dim``) rather than assumed."""
        out, start = {}, 0
        for name in self.robot.controller_order:
            width = int(self.robot.controllers[name].command_dim)
            out[name] = slice(start, start + width)
            start += width
        return out

    @property
    def action_dim(self) -> int:
        return int(self.robot.action_dim)

    def no_op_action(self) -> np.ndarray:
        """The action that commands no motion. On this robot a zero vector is
        NOT a no-op: the trunk and both arms are absolute ``JointController``s,
        so holding still means commanding the current joint angles. Every
        controller's own ``compute_no_op_action`` says what that is
        (``controllers/controller_base.py`` L403)."""
        action = np.zeros(self.action_dim, dtype=np.float64)
        control_dict = self.robot.get_control_dict()
        for name, span in self.action_slices.items():
            action[span] = _to_numpy(self.robot.controllers[name].compute_no_op_action(control_dict)).reshape(-1)
        return action

    def gripper_limits(self, arm: str) -> tuple[float, float]:
        """(closed, open) command of that gripper — the controller's own output
        limits, so a "smooth" MultiFingerGripperController's finger target is
        never hard-coded."""
        controller = self.robot.controllers[f"gripper_{arm}"]
        low, high = controller.command_output_limits
        low, high = float(_to_numpy(low).reshape(-1)[0]), float(_to_numpy(high).reshape(-1)[0])
        return (low, high) if not getattr(controller, "_inverted", False) else (high, low)

    # ---- episode --------------------------------------------------------------------------
    def reset(self, scene: BehaviorSceneRef) -> int:
        """Load the instance's initial state — every task-relevant object's
        state and the robot's pose replayed from the shipped JSON, then 25
        physics steps with everything held still, exactly as the challenge's
        evaluator does (``eval.py`` ``load_task_instance`` L241-291). Returns
        the control ticks spent (the settle is physics, not control, so this is
        ``body.settle_ticks``)."""
        import omnigibson as og

        self._load_instance(scene)
        self._env.reset()
        self._success = False
        self._travel = {"base": 0.0, "left": 0.0, "right": 0.0}
        self._last_points = {}
        self._obs = self._get_obs()
        self._info = {}
        # TaskMetric.start_callback (metrics/task_metric.py L22-25): the predicates already true at t0.
        self._initial_predicates = [[bool(pred.evaluate()) for pred in option]
                                    for option in self.task.ground_goal_state_options]
        self._accumulate_travel()
        ticks = 0
        for _ in range(int(self.body.settle_ticks)):
            self.tick(self.no_op_action())
            ticks += 1
        del og
        return ticks

    def _load_instance(self, scene: BehaviorSceneRef) -> None:
        import omnigibson as og
        from omnigibson.macros import gm
        from omnigibson.utils.asset_utils import get_task_instance_path
        from omnigibson.utils.python_utils import recursively_convert_to_torch

        path = os.path.join(get_task_instance_path(scene.scene_model), "json",
                            f"{scene.scene_model}_task_{scene.task}_instances", f"{scene.tro_stem}-tro_state.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{scene.task} instance {scene.instance_id}: no {path} — the "
                f"2025-challenge-task-instances release is not under {gm.DATA_PATH} (INSTALL-behavior.md § Data)")
        with open(path) as fh:
            tro_states = recursively_convert_to_torch(json.load(fh))
        for key, state in tro_states.items():
            if key == "robot_poses":
                pose = state[self.robot.model_name][0]
                self.robot.set_position_orientation(pose["position"], pose["orientation"])
                self._env.scene.write_task_metadata(key=key, data=state)
            else:
                self.task.object_scope[key].load_state(state, serialized=False)
        # Put the arms, torso and grippers back where the challenge starts them.
        # The instance file carries the robot's BASE pose and nothing else, and the
        # state captured just below becomes the episode's initial state — so without
        # this an episode would inherit whatever configuration the previous one left
        # the arms in, and no two runs of a split would agree. A fresh process is
        # already at this posture: it is the robot config's own reset_joint_pos,
        # ROBOT_RESET_JOINT_POS["R1Pro"] with the trunk filled in (og_teleop_utils.py
        # generate_robot_config L1524-1532). robot.reset() keeps the base joints, so
        # the pose just set survives (holonomic_base_robot.py L250-259).
        self.robot.reset()
        for _ in range(SETTLE_PHYSICS_STEPS):
            og.sim.step_physics()
            for entity in self.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()
        self._env.scene.update_initial_file()
        self._env.scene.reset()

    def tick(self, action: np.ndarray) -> bool:
        """One control tick at ``body.action_freq``. Returns BEHAVIOR's own
        success (``info["done"]["success"]``, i.e. every goal predicate of some
        grounding satisfied)."""
        import torch as th

        a = np.nan_to_num(np.asarray(action, dtype=np.float64).reshape(self.action_dim))
        obs, _reward, _terminated, _truncated, info = self._env.step(th.as_tensor(a, dtype=th.float32))
        self._obs, self._info = obs, info
        self._success = bool((info.get("done") or {}).get("success", False))
        self._accumulate_travel()
        return self._success

    def _accumulate_travel(self) -> None:
        """The challenge's efficiency terms: per-step L2 displacement of the
        base and of each end effector, summed (``metrics/agent_metric.py``
        L26-45)."""
        points = {"base": self.base_position(), **{arm: self.eef_position(arm) for arm in ARMS}}
        for key, point in points.items():
            previous = self._last_points.get(key)
            if previous is not None:
                self._travel[key] += float(np.linalg.norm(point - previous))
            self._last_points[key] = point

    def _get_obs(self) -> dict[str, Any]:
        obs, _info = self._env.get_obs()
        return obs

    # ---- facts ----------------------------------------------------------------------------
    @property
    def success(self) -> bool:
        return self._success

    @property
    def language(self) -> str:
        """The activity's natural-language goal, as BDDL words it
        (``tasks/behavior_task.py`` L339,
        ``activity_natural_language_goal_conditions``)."""
        goals = getattr(self.task, "activity_natural_language_goal_conditions", None) or ()
        return "; ".join(str(g) for g in goals)

    def goal_status(self) -> dict[str, list[int]]:
        """Which goal predicates hold right now, by index —
        ``{"satisfied": [...], "unsatisfied": [...]}`` over grounding 0.
        ``tasks/behavior_task.py`` L509-515 writes it into the dict that
        ``task_base.py`` L364-366 then nests under ``info["done"]``."""
        status = (self._info.get("done") or {}).get("goal_status") or {}
        return {k: [int(i) for i in v] for k, v in status.items()}

    def q_score(self) -> float:
        """The challenge's ranking metric for this rollout, verbatim from
        ``omnigibson/metrics/task_metric.py`` ``end_callback`` L30-45: 1.0 on
        success, else the largest fraction, over the goal's groundings, of
        predicates that were false at t0 and are true now."""
        if self._success:
            return 1.0
        options = self.task.ground_goal_state_options
        if not options or not self._initial_predicates:
            return 0.0
        return float(max(
            sum(int(not was_true and bool(pred.evaluate())) for pred, was_true in zip(option, before)) / len(option)
            for option, before in zip(options, self._initial_predicates)
        ))

    def n_predicates(self) -> int:
        """How many goal predicates grounding 0 has — the denominator of the
        challenge's partial credit."""
        options = self.task.ground_goal_state_options
        return int(len(options[0])) if options else 0

    def travel(self) -> dict[str, float]:
        """Accumulated travel in metres: ``base``, ``left``, ``right``
        (``metrics/agent_metric.py`` ``gather_results``)."""
        return {k: float(v) for k, v in self._travel.items()}

    def observe(self) -> dict[str, np.ndarray]:
        assert self._obs is not None, "reset() first"
        out = {"rgb": self._frame(self.body.rgb_camera)}
        if self.body.wrist is not None:
            out["wrist"] = self._frame(self.body.wrist_camera)
            out["wrist_right"] = self._frame(self.body.aux_camera)
        out["proprio"] = self.proprio()
        return out

    def _frame(self, camera_id: str) -> np.ndarray:
        """The camera's RGB frame. OmniGibson nests observations
        robot -> sensor -> modality and serves RGBA; the port drops alpha."""
        sensor_name = CAMERA_PRIMS[camera_id].split("::")[1]
        frame = _to_numpy(self._obs[ROBOT_NAME][sensor_name]["rgb"])
        return np.ascontiguousarray(frame[..., :3].astype(np.uint8))

    def proprio(self) -> np.ndarray:
        """Per arm: eef position 3 + eef axis-angle 3 + gripper opening 1;
        then base position 3 + base yaw 1 — 18 numbers for this two-armed
        mobile manipulator."""
        parts: list[np.ndarray] = []
        for arm in ARMS:
            position, quat = self.eef_pose(arm)
            parts += [position, quat_to_rotvec(quat), np.array([self.gripper_open(arm) / 100.0])]
        base_position, base_yaw = self.base_pose()
        parts += [base_position, np.array([base_yaw])]
        return np.concatenate(parts).astype(np.float32)

    def eef_position(self, arm: str) -> np.ndarray:
        return _to_numpy(self.robot.get_eef_position(arm)).astype(np.float64).reshape(3)

    def eef_pose(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        """(position (3,), ``(w, x, y, z)``) of that arm's end effector, world frame."""
        return self.eef_position(arm), _xyzw_to_wxyz(self.robot.get_eef_orientation(arm))

    def controller_reference(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        """That arm's end-effector pose as the IK controller itself sees it —
        ``eef_<arm>_pos_relative`` / ``_quat_relative`` of the control dict,
        which is what ``_update_goal`` compares an ``absolute_pose`` command
        against. ``to_controller_frame`` is the inverse road: it must map the
        current world pose back onto exactly these numbers."""
        control = self.robot.get_control_dict()
        return (_to_numpy(control[f"eef_{arm}_pos_relative"]).astype(np.float64).reshape(3),
                _xyzw_to_wxyz(control[f"eef_{arm}_quat_relative"]))

    def eef_pose_relative(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        """That arm's end-effector pose relative to the robot's own base frame
        (``manipulation_robot.py`` ``get_relative_eef_pose`` L1040-1055). NOTE
        this is the ``base_footprint_link`` frame on a holonomic base, which is
        NOT the frame the IK controller reads — see ``controller_reference``."""
        position, quat = self.robot.get_relative_eef_pose(arm)
        return _to_numpy(position).astype(np.float64).reshape(3), _xyzw_to_wxyz(quat)

    def to_controller_frame(self, arm: str, position: Any, quat_wxyz: Any) -> tuple[np.ndarray, np.ndarray]:
        """A world-frame end-effector pose expressed in the frame an
        ``absolute_pose`` IK command is read in.

        That frame is NOT ``robot.get_position_orientation()``. On a holonomic
        base the two differ: the robot's own accessor is overridden to return
        the moving ``base_footprint_link`` (``holonomic_base_robot.py``
        L261-275), while the controller's ``eef_<arm>_pos_relative`` comes from
        ``ControllableObjectViewAPI.get_link_relative_position_orientation``
        (``controllable_object.py`` L668-675, ``usd_utils.py`` L1079-1102),
        whose origin is the ARTICULATION ROOT — ``base_footprint_x``, the
        stationary virtual link the six base joints hang off. Anchoring a
        target on the wrong one makes the arm run away by exactly the distance
        the base has driven.

        So the frame is never named here. The target is expressed relative to
        the CURRENT end-effector pose in the world, and that relative transform
        is then re-anchored on the end-effector pose the controller itself
        reports — whatever frame that is:

            T_frame_target = T_frame_eef  ∘  (T_world_eef)^-1 ∘ T_world_target
        """
        from omnigibson.utils import transform_utils as T

        control = self.robot.get_control_dict()
        # transform_utils is torch.compile'd and takes tensors only; the control dict's entries
        # come back through OmniGibson's compute backend, which is numpy here.
        ref_pos, ref_quat = _tensor(control[f"eef_{arm}_pos_relative"]), _tensor(control[f"eef_{arm}_quat_relative"])
        eef_pos, eef_quat = self.robot.eef_links[arm].get_position_orientation()
        target_pos = _tensor(np.asarray(position, dtype=np.float64).reshape(3))
        target_quat = _tensor(_wxyz_to_xyzw(quat_wxyz))
        d_pos, d_quat = T.relative_pose_transform(target_pos, target_quat, _tensor(eef_pos), _tensor(eef_quat))
        out_pos, out_quat = T.pose_transform(ref_pos, ref_quat, _tensor(d_pos), _tensor(d_quat))
        return (_to_numpy(out_pos).astype(np.float64).reshape(3), _xyzw_to_wxyz(out_quat))

    def gripper_open(self, arm: str) -> float:
        """The gripper opening, 0 (closed) .. 100 (fully open), from the finger
        joint positions against that gripper's own joint limits."""
        idx = _to_numpy(self.robot.gripper_control_idx[arm]).astype(int).reshape(-1)
        qpos = _to_numpy(self.robot.get_joint_positions())[idx]
        lower = _to_numpy(self.robot.joint_lower_limits)[idx]
        upper = _to_numpy(self.robot.joint_upper_limits)[idx]
        span = np.where(np.abs(upper - lower) < 1e-9, 1.0, upper - lower)
        return float(np.clip(np.mean((qpos - lower) / span), 0.0, 1.0) * 100.0)

    def is_grasping(self, arm: str) -> bool:
        return bool(_to_numpy(self.robot.is_grasping(arm=arm)).reshape(-1)[0])

    def base_position(self) -> np.ndarray:
        return _to_numpy(self.robot.get_position_orientation()[0]).astype(np.float64).reshape(3)

    def base_pose(self) -> tuple[np.ndarray, float]:
        """(position (3,), yaw) of the mobile base in the world frame."""
        from scipy.spatial.transform import Rotation

        position, quat = self.robot.get_position_orientation()
        yaw = float(Rotation.from_quat(_to_numpy(quat).reshape(4)).as_euler("xyz")[2])
        return _to_numpy(position).astype(np.float64).reshape(3), yaw

    def robot_base(self) -> list[float]:
        return [float(v) for v in self.base_position()]

    def trunk_qpos(self) -> np.ndarray:
        idx = _to_numpy(self.robot.trunk_control_idx).astype(int).reshape(-1)
        return _to_numpy(self.robot.get_joint_positions())[idx].astype(np.float64)

    def objects(self) -> dict[str, dict[str, list[float]]]:
        """Pose of every task-relevant object, by its BDDL scope name —
        privileged, for metrics and scripted debugging only
        (``tasks/behavior_task.py`` ``object_scope``)."""
        out: dict[str, dict[str, list[float]]] = {}
        for name, entity in (getattr(self.task, "object_scope", {}) or {}).items():
            try:
                if entity is None or entity.is_system or not entity.exists:
                    continue
                position, quat = entity.get_position_orientation()
            except Exception:
                continue
            out[str(name)] = {"position": [float(v) for v in _to_numpy(position).reshape(3)],
                              "rotation": _xyzw_to_wxyz(quat).tolist()}
        return out
