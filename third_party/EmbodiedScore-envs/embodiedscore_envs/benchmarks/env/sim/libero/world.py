"""LiberoWorld — the simulator facade of the LIBERO engine (layer L0): the
only module that imports libero / robosuite / mujoco, lazily, at
construction.

It owns one ``OffScreenRenderEnv`` at a time (rebuilt when the BDDL file
changes — every task is its own MuJoCo model), places an episode's init
state, runs ticks, and reads facts: the frames, the end-effector pose from
the gripper site, the gripper opening, the object poses and the BDDL
success predicate. It never counts steps or budgets (the env body does)
and never decides an action (the bodies do).

Frames: robosuite world frame, metres, z up. Rotations leave as ``(w, x,
y, z)`` quaternions or 3x3 matrices; robosuite's own observation
quaternions are ``(x, y, z, w)`` and are converted here.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .body import LiberoBody
from .scene import LiberoSceneRef

GRIPPER_OPEN_QPOS = 0.04       # finger qpos when fully open (closed ~0.0005): the 0-100 opening scale
GRIPPER_CLOSE, GRIPPER_OPEN = 1.0, -1.0     # robosuite's gripper action convention
_EEF_SITES = ("gripper0_grip_site", "robot0_eef_site", "robot0_grip_site")


def _import_libero() -> Any:
    try:
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as e:
        raise ImportError(
            "the LIBERO lines need the `libero` package (robosuite 1.4 + mujoco) in this interpreter — "
            "see INSTALL-libero.md"
        ) from e
    return OffScreenRenderEnv


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


def load_init_states(scene: LiberoSceneRef) -> np.ndarray:
    """The task's (N, state) init states — torch pickles in the libero release.

    A task with a single init state may store it as a flat vector rather than
    a (1, state) matrix: LIBERO-Plus's ``libero_newobj`` files do, and its
    ``Benchmark.get_task_init_states`` reshapes them to ``(1, -1)`` before
    indexing (``libero/libero/benchmark/__init__.py``, sylvestf/LIBERO-plus at
    ``4976dc3``). Same reshape here, so a 1-D file is one init state."""
    import torch

    try:
        states = torch.load(scene.init_states_file, weights_only=False)
    except TypeError:     # torch < 1.13 has no weights_only
        states = torch.load(scene.init_states_file)
    array = np.asarray(states)
    return array.reshape(1, -1) if array.ndim == 1 else array


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def mat_to_wxyz(m: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(np.asarray(m, dtype=np.float64).reshape(3, 3)).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


class LiberoWorld:
    def __init__(self, body: LiberoBody, gpu_id: int = 0) -> None:
        self.body = body
        self.gpu_id = int(gpu_id)
        self._env: Any = None
        self._bddl: str | None = None
        self._obs: dict[str, Any] | None = None
        self._success = False
        self._objects: tuple[str, ...] = ()
        os.environ.setdefault("MUJOCO_GL", "egl")

    # ---- lifecycle ----------------------------------------------------------------------
    def load_scene(self, scene: LiberoSceneRef, horizon: int) -> None:
        """Build the task's env (once per BDDL file); ``horizon`` is robosuite's
        own tick cap, set above ours so its done never masquerades as success."""
        if self._env is not None and self._bddl == scene.bddl_file:
            self._env.env.horizon = int(horizon)
            return
        self.close()
        OffScreenRenderEnv = _import_libero()
        _patch_render_context()
        self._env = OffScreenRenderEnv(
            bddl_file_name=scene.bddl_file,
            camera_heights=self.body.rgb.height,
            camera_widths=self.body.rgb.width,
            control_freq=self.body.control_freq,
            horizon=int(horizon),
            render_gpu_device_id=self.gpu_id,
        )
        self._bddl = scene.bddl_file
        inner = self._env.env
        self._objects = tuple(str(n) for n in getattr(inner, "obj_of_interest", ()) or ())
        if not self._objects:
            self._objects = tuple(str(n) for n in getattr(inner, "objects_dict", {}).keys())

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        self._env, self._bddl, self._obs = None, None, None

    @property
    def sim(self) -> Any:
        return self._env.env.sim

    # ---- episode ---------------------------------------------------------------------------
    def reset(self, init_state: np.ndarray) -> int:
        """Reset to ``init_state`` and let the scene settle (``body.settle_ticks``
        zero-motion ticks with the gripper open, as OpenVLA's evaluator does).
        Returns the ticks spent."""
        self._env.reset()
        self._obs = self._env.set_init_state(np.asarray(init_state))
        self._success = False
        ticks = 0
        for _ in range(self.body.settle_ticks):
            self.tick(np.array([0, 0, 0, 0, 0, 0, GRIPPER_OPEN], dtype=np.float64))
            ticks += 1
        return ticks

    def tick(self, action: np.ndarray) -> bool:
        """One OSC_POSE control step. Returns LIBERO's done — the BDDL goal
        holds (robosuite's horizon is kept out of reach)."""
        a = np.clip(np.nan_to_num(np.asarray(action, dtype=np.float64).reshape(7)), -1.0, 1.0)
        obs, _reward, done, _info = self._env.step(a.tolist())
        self._obs = obs
        self._success = bool(self._env.env._check_success())
        return self._success

    # ---- facts ------------------------------------------------------------------------------
    @property
    def success(self) -> bool:
        return self._success

    def _frame(self, key: str) -> np.ndarray:
        img = np.asarray(self._obs[key], dtype=np.uint8)
        return np.ascontiguousarray(img[::-1, ::-1]) if self.body.upright else img

    def observe(self) -> dict[str, np.ndarray]:
        assert self._obs is not None, "reset() first"
        out = {"rgb": self._frame("agentview_image")}
        if self.body.wrist is not None:
            out["wrist"] = self._frame("robot0_eye_in_hand_image")
        out["proprio"] = self.proprio()
        return out

    def proprio(self) -> np.ndarray:
        """eef position 3 + eef axis-angle 3 + gripper finger qpos 2 (LIBERO's 8-D state)."""
        from scipy.spatial.transform import Rotation

        o = self._obs
        q = np.asarray(o["robot0_eef_quat"], dtype=np.float64)        # x, y, z, w
        rotvec = Rotation.from_quat(q).as_rotvec()
        return np.concatenate([o["robot0_eef_pos"], rotvec, o["robot0_gripper_qpos"]]).astype(np.float32)

    def eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(position (3,), rotation matrix (3, 3)) of the gripper site, world frame."""
        sim = self.sim
        for name in _EEF_SITES:
            try:
                sid = sim.model.site_name2id(name)
            except (ValueError, KeyError):
                continue
            return (np.array(sim.data.site_xpos[sid], dtype=np.float64),
                    np.array(sim.data.site_xmat[sid], dtype=np.float64).reshape(3, 3))
        raise RuntimeError(f"no end-effector site among {_EEF_SITES}")

    def gripper_open(self) -> float:
        """The gripper opening, 0 (closed) .. 100 (fully open), from the finger qpos."""
        q = float(abs(np.asarray(self._obs["robot0_gripper_qpos"], dtype=np.float64)[0]))
        return float(np.clip(q / GRIPPER_OPEN_QPOS, 0.0, 1.0) * 100.0)

    def objects(self) -> dict[str, dict[str, list[float]]]:
        """Pose of every BDDL object of interest — privileged, for metrics and debugging."""
        out: dict[str, dict[str, list[float]]] = {}
        for name in self._objects:
            pos, quat = self._obs.get(f"{name}_pos"), self._obs.get(f"{name}_quat")
            if pos is None or quat is None:
                continue
            out[name] = {"position": [float(v) for v in pos], "rotation": _xyzw_to_wxyz(quat).tolist()}
        return out

    def robot_base(self) -> list[float]:
        robot = self._env.env.robots[0]
        return [float(v) for v in np.asarray(robot.base_pos, dtype=np.float64)]
