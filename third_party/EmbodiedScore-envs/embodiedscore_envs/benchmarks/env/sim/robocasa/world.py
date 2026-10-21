"""RobocasaWorld — the simulator facade of the RoboCasa engine (layer L0): the
only module that imports robocasa / robosuite / mujoco, lazily, at
construction.

It owns one robosuite kitchen env at a time (rebuilt when the task, the
layout, the style or the object split changes — every kitchen is its own
generated MuJoCo model), seeds and resets it, runs ticks, and reads facts:
the frames, the end-effector pose, the arm's controller origin frame, the
gripper opening, the mobile base pose, the object poses, the episode's
language and RoboCasa's own success check. It never counts steps or budgets
(the env body does) and never decides an action (the bodies do).

Frames: robosuite world frame, metres, z up. Rotations leave as ``(w, x, y,
z)`` quaternions or 3x3 matrices; robosuite's own observation quaternions are
``(x, y, z, w)`` and are converted here.

The action this world ticks is RoboCasa's own 12-D gym action, transcribed
from ``robocasa/utils/env_utils.py`` ``convert_action`` (L142-154) and
``robocasa/wrappers/gym_wrapper.py`` ``PandaOmronKeyConverter.unmap_action``
(L120-140):

    a[0:3]   end-effector position delta   -> ``robot0_right[0:3]``   (OSC_POSE, arm base frame, unit 5 cm)
    a[3:6]   end-effector rotation delta   -> ``robot0_right[3:6]``   (axis-angle, arm base frame, unit 0.5 rad)
    a[6]     gripper_close                 -> ``robot0_right_gripper``: -1 (open) when < 0.5 else +1 (close)
    a[7:10]  base motion (x, y, yaw)       -> ``robot0_base``         (JOINT_VELOCITY)
    a[10]    torso                         -> ``robot0_torso``        (JOINT_POSITION)
    a[11]    control_mode                  -> ``robot0_base_mode``:   -1 (arm mode) when < 0.5 else +1 (base mode)

The raw vector robosuite's ``step`` takes is assembled through the live
``composite_controller._action_split_indexes``, exactly as RoboCasa's own
``RoboCasaGymEnv.step`` does (gym_wrapper.py L355-385), so the part order is
never assumed.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .body import RobocasaBody
from .scene import RobocasaSceneRef

ACTION_DIM = 12
A_EEF_POS = slice(0, 3)
A_EEF_ROT = slice(3, 6)
A_GRIPPER = 6
A_BASE = slice(7, 10)
A_TORSO = 10
A_MODE = 11

GRIPPER_CLOSE, GRIPPER_OPEN = 1.0, 0.0       # RoboCasa's gripper_close flag, thresholded at 0.5
BASE_MODE_ON, BASE_MODE_OFF = 1.0, 0.0       # RoboCasa's control_mode flag, thresholded at 0.5
PANDA_FINGER_OPEN_QPOS = 0.04                # the Panda finger joint's upper limit; read from the model when available


def _import_robocasa() -> tuple[Any, Any]:
    try:
        import robocasa  # registers the kitchen environments with robosuite
        import robosuite
    except ImportError as e:
        raise ImportError(
            "the RoboCasa lines need the `robocasa` package (robosuite 1.5 + mujoco) in this interpreter — "
            "see INSTALL-robocasa.md"
        ) from e
    return robocasa, robosuite


def _controller_config(robot: str) -> Any:
    """RoboCasa's controller for the robot: ``load_composite_controller_config``
    on robosuite >= 1.5 (``robocasa/utils/env_utils.py`` L81-84), the 1.4
    ``load_controller_config(default_controller="OSC_POSE")`` before it
    (RoboCasa v0.2's ``robocasa/utils/eval_utils.py``)."""
    import robosuite.controllers as C

    if hasattr(C, "load_composite_controller_config"):
        return C.load_composite_controller_config(controller=None, robot=robot)
    return C.load_controller_config(default_controller="OSC_POSE")


def _patch_render_context() -> None:
    """robosuite's ``MjRenderContext.__del__`` assumes ``self.con`` exists;
    during interpreter teardown it may not, and the traceback is noise."""
    try:
        from robosuite.utils import binding_utils

        def safe_del(self):
            con = getattr(self, "con", None)
            if con is not None:
                try:
                    con.free()
                except Exception:
                    pass

        binding_utils.MjRenderContext.__del__ = safe_del
    except Exception:
        pass


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def mat_to_wxyz(m: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(np.asarray(m, dtype=np.float64).reshape(3, 3)).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def zero_action(gripper: float = GRIPPER_OPEN) -> np.ndarray:
    """A 12-D action that commands no motion and the given gripper flag."""
    a = np.zeros(ACTION_DIM, dtype=np.float64)
    a[A_GRIPPER] = float(gripper)
    return a


class RobocasaWorld:
    def __init__(self, body: RobocasaBody, gpu_id: int = 0) -> None:
        self.body = body
        self.gpu_id = int(gpu_id)
        self._env: Any = None
        self._key: tuple | None = None
        self._obs: dict[str, Any] | None = None
        self._success = False
        self._lang = ""
        os.environ.setdefault("MUJOCO_GL", "egl")

    # ---- lifecycle ----------------------------------------------------------------------
    def load_scene(self, scene: RobocasaSceneRef, horizon: int) -> None:
        """Build the scenario's env (once per task / layout / style / object
        split); ``horizon`` is robosuite's own tick cap, set above ours — and
        RoboCasa runs with ``ignore_done=True`` anyway, so its done never
        masquerades as success."""
        if self._env is not None and self._key == scene.model_key:
            self._env.horizon = int(horizon)
            return
        self.close()
        _robocasa, robosuite = _import_robocasa()
        _patch_render_context()
        cams = list(self.body.cameras)
        self._env = robosuite.make(
            env_name=scene.task,
            robots=self.body.robot,
            controller_configs=_controller_config(self.body.robot),
            camera_names=cams,
            camera_widths=self.body.rgb.width,
            camera_heights=self.body.rgb.height,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=True,
            camera_depths=False,
            control_freq=self.body.control_freq,
            horizon=int(horizon),
            seed=int(scene.seed),
            obj_instance_split=scene.obj_instance_split,
            layout_and_style_ids=[(int(scene.layout_id), int(scene.style_id))],
            translucent_robot=False,
            render_gpu_device_id=self.gpu_id,
        )
        self._key = scene.model_key

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        self._env, self._key, self._obs = None, None, None

    @property
    def sim(self) -> Any:
        return self._env.sim

    @property
    def env(self) -> Any:
        return self._env

    # ---- episode ---------------------------------------------------------------------------
    def reset(self, seed: int) -> int:
        """Reset the kitchen under ``seed`` — the env RNG draws the objects,
        their placements and the robot spawn — then let the scene settle
        (``body.settle_ticks`` zero-motion ticks with the gripper open).
        Returns the ticks spent. Seeding through ``env.rng`` is RoboCasa's own
        (``robocasa/wrappers/gym_wrapper.py`` ``reset``, L342-345)."""
        self._env.rng = np.random.default_rng(int(seed))
        self._obs = self._env.reset()
        self._success = False
        self._lang = str((self._env.get_ep_meta() or {}).get("lang", "") or "")
        ticks = 0
        for _ in range(self.body.settle_ticks):
            self.tick(zero_action(GRIPPER_OPEN))
            ticks += 1
        return ticks

    def _raw_action(self, action: np.ndarray) -> np.ndarray:
        """RoboCasa's 12-D gym action -> the concatenated part vector robosuite
        steps (gym_wrapper.py L355-385)."""
        robot = self._env.robots[0]
        cc = robot.composite_controller
        raw = np.zeros(cc.action_limits[0].shape, dtype=np.float64)
        parts = {
            "right": np.concatenate([action[A_EEF_POS], action[A_EEF_ROT]]),
            "right_gripper": np.array([-1.0 if action[A_GRIPPER] < 0.5 else 1.0]),
            "base": action[A_BASE],
            "torso": action[A_TORSO:A_TORSO + 1],
        }
        for name, vec in parts.items():
            span = cc._action_split_indexes.get(name)
            if span is None:
                continue
            start, end = span
            raw[start:end] = np.asarray(vec, dtype=np.float64)[: end - start]
        if raw.shape[0]:      # HybridMobileBase appends the base-mode flag last
            raw[-1] = -1.0 if action[A_MODE] < 0.5 else 1.0
        return raw

    def tick(self, action: np.ndarray) -> bool:
        """One control tick. Returns RoboCasa's own success check."""
        a = np.nan_to_num(np.asarray(action, dtype=np.float64).reshape(ACTION_DIM))
        a[:A_GRIPPER] = np.clip(a[:A_GRIPPER], -1.0, 1.0)
        a[A_BASE] = np.clip(a[A_BASE], -1.0, 1.0)
        a[A_TORSO] = float(np.clip(a[A_TORSO], -1.0, 1.0))
        obs, _reward, _done, _info = self._env.step(self._raw_action(a))
        self._obs = obs
        self._success = bool(self._env._check_success())
        return self._success

    # ---- facts ------------------------------------------------------------------------------
    @property
    def success(self) -> bool:
        return self._success

    @property
    def language(self) -> str:
        """The episode's instruction, as RoboCasa words it for the sampled
        objects (``env.get_ep_meta()["lang"]``) — only known after a reset."""
        return self._lang

    def _frame(self, camera: str) -> np.ndarray:
        img = np.asarray(self._obs[f"{camera}_image"], dtype=np.uint8)
        return np.ascontiguousarray(img[::-1, :, :]) if self.body.upright else img

    def observe(self) -> dict[str, np.ndarray]:
        assert self._obs is not None, "reset() first"
        out = {"rgb": self._frame(self.body.rgb_camera)}
        if self.body.wrist is not None:
            out["wrist"] = self._frame(self.body.wrist_camera)
        if self.body.aux is not None:
            out["aux"] = self._frame(self.body.aux_camera)
        out["proprio"] = self.proprio()
        return out

    def proprio(self) -> np.ndarray:
        """eef position 3 + eef axis-angle 3 + gripper finger qpos 2 + base
        position 3 + base yaw 1 — the 12-D state of a mobile manipulator."""
        from scipy.spatial.transform import Rotation

        o = self._obs
        rotvec = Rotation.from_quat(np.asarray(o["robot0_eef_quat"], dtype=np.float64)).as_rotvec()
        base_pos, base_yaw = self.base_pose()
        return np.concatenate([
            np.asarray(o["robot0_eef_pos"], dtype=np.float64), rotvec,
            np.asarray(o["robot0_gripper_qpos"], dtype=np.float64)[:2],
            base_pos, [base_yaw],
        ]).astype(np.float32)

    def _arm_controller(self) -> Any:
        cc = self._env.robots[0].composite_controller
        for name in ("right", "arm", *cc.part_controllers):
            if name in cc.part_controllers:
                return cc.part_controllers[name]
        raise RuntimeError("no arm controller on this robot")

    def eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(position (3,), rotation matrix (3, 3)) of the end-effector site the
        OSC controller tracks, world frame."""
        c = self._arm_controller()
        name = c.ref_name if isinstance(c.ref_name, str) else list(c.ref_name)[0]
        sid = self.sim.model.site_name2id(name)
        return (np.array(self.sim.data.site_xpos[sid], dtype=np.float64),
                np.array(self.sim.data.site_xmat[sid], dtype=np.float64).reshape(3, 3))

    def arm_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """The frame the OSC's deltas are expressed in — the arm's mount site,
        which the composite controller feeds it as its origin
        (``composite_controller.get_controller_base_pose`` /
        ``update_origin``). A world-frame delta must be rotated into it."""
        c = self._arm_controller()
        pos, ori = getattr(c, "origin_pos", None), getattr(c, "origin_ori", None)
        if pos is None or ori is None:
            pos, ori = self._env.robots[0].composite_controller.get_controller_base_pose("right")
        return (np.array(pos, dtype=np.float64).reshape(3),
                np.array(ori, dtype=np.float64).reshape(3, 3))

    def gripper_open(self) -> float:
        """The gripper opening, 0 (closed) .. 100 (fully open), from the finger qpos."""
        q = float(abs(np.asarray(self._obs["robot0_gripper_qpos"], dtype=np.float64)[0]))
        return float(np.clip(q / PANDA_FINGER_OPEN_QPOS, 0.0, 1.0) * 100.0)

    def base_pose(self) -> tuple[np.ndarray, float]:
        """(position (3,), yaw) of the mobile base in the world frame."""
        from scipy.spatial.transform import Rotation

        o = self._obs
        pos = np.asarray(o["robot0_base_pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(o["robot0_base_quat"], dtype=np.float64).reshape(4)   # x, y, z, w
        yaw = float(Rotation.from_quat(quat).as_euler("xyz")[2])
        return pos, yaw

    def robot_base(self) -> list[float]:
        pos, _ = self.base_pose()
        return [float(v) for v in pos]

    def objects(self) -> dict[str, dict[str, list[float]]]:
        """Pose of every object RoboCasa placed — privileged, for metrics and
        debugging (``kitchen.py`` ``_create_obj_sensors`` publishes
        ``<name>_pos`` / ``<name>_quat`` for each)."""
        out: dict[str, dict[str, list[float]]] = {}
        for name in getattr(self._env, "obj_body_id", {}) or {}:
            pos, quat = self._obs.get(f"{name}_pos"), self._obs.get(f"{name}_quat")
            if pos is None or quat is None:
                continue
            out[str(name)] = {"position": [float(v) for v in pos],
                              "rotation": _xyzw_to_wxyz(quat).tolist()}
        return out

    def fixtures(self) -> dict[str, str]:
        """The kitchen's fixtures by name -> class — privileged scene facts a
        scripted policy needs (``kitchen.py`` ``get_ep_meta``)."""
        return {str(k): type(v).__name__ for k, v in (getattr(self._env, "fixtures", {}) or {}).items()}
