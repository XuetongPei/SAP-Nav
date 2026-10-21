"""CalvinWorld — the simulator facade of the CALVIN engine (layer L0): the
only module that imports ``calvin_env`` / pybullet, lazily, at construction.

It owns one ``PlayTableSimEnv`` (built once from the release's own hydra
config), puts the scene into an episode's initial state, runs ticks, reads
facts (frames, the TCP pose, the gripper opening, the scene state) and
answers the one question the chain protocol needs: **which of a set of
sub-tasks the scene now satisfies**, through CALVIN's own task oracle. It
never counts steps or budgets (the env body does) and never decides an
action (the bodies do).

## The action, transcribed

``PlayTableSimEnv.step(action)`` (``calvin_env/envs/play_table_env.py``:225)
is one control tick: ``robot.apply_action(action)``, then ``action_repeat``
(= ``bullet_time_step // control_freq`` = 240 / 30 = 8) physics steps, then
``scene.step()``. ``Robot.apply_action`` (``calvin_env/robot/robot.py``:244)
takes **either** form:

* a length-7 vector -> ``relative_to_absolute`` (:228): ``[dx, dy, dz]`` are
  scaled by ``max_rel_pos * magic_scaling_factor_pos`` = 0.02 * 1 = **2 cm**
  per unit and ``[dax, day, daz]`` by ``max_rel_orn * magic_scaling_factor_orn``
  = 0.05 * 1 = **0.05 rad** per unit; with ``use_target_pose: true`` (the
  release's ``conf/robot/panda.yaml``:21) they accumulate on an internal
  target pose rather than on the measured TCP. This is the action CALVIN's
  policies emit and what ``CalvinEnv`` exposes.
* a length-3 tuple ``(position, orientation, gripper)`` -> applied directly
  (:311 ``if not len(action) == 3``). The orientation may be a 3-vector,
  which :318 turns into a quaternion with ``p.getQuaternionFromEuler`` —
  pybullet's extrinsic x-y-z Euler, the same convention ``robot_obs[3:6]``
  is reported in (``euler_obs: true``). This is CALVIN's **own absolute
  end-effector interface**, and it is what ``CalvinPoseEnv``'s macro
  protocol drives: no bounded-delta closed loop is needed here (the LIBERO
  and RoboCasa engines have to build one because robosuite's OSC only takes
  deltas), the target is simply held until the TCP arrives.

Either way the target goes through ``MixedIK.get_ik`` (pybullet's damped
least squares with null-space limits; ``use_ik_fast: false`` in the
release's config) into ``POSITION_CONTROL`` at ``max_velocity`` 2.

The gripper is binary and **inverted relative to robosuite**:
``control_gripper`` (:355) opens on ``+1`` and closes on anything else, and
``apply_action`` asserts the value is in ``(-1, 1)``.

## Success, transcribed

CALVIN has no goal predicate on a single state — a task is a *change*
between two scene states. ``evaluate_policy.rollout`` (:147) snapshots
``start_info = env.get_info()`` before the sub-task's first action, and on
every step asks

    task_oracle.get_task_info_for_set(start_info, current_info, {subtask})

(``calvin_env/envs/tasks.py``:35), which returns the subset of the named
tasks whose predicate over the (start, current) pair holds. The predicates
live in ``calvin_env/conf/tasks/new_playtable_tasks.yaml`` — the same table
the evaluator loads from ``calvin_models/conf/callbacks/rollout/tasks/
new_playtable_tasks.yaml`` and the same one the dataset's
``merged_config.yaml`` carries (all three verified identical, 2026-09-10;
``tests/test_calvin_contracts.py`` re-checks the installed one).
``snapshot()`` / ``achieved()`` below are that pair of calls.

Frames: pybullet's world frame, metres, z up. Rotations leave as
``(w, x, y, z)`` quaternions or 3x3 matrices.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

import numpy as np

from .body import CalvinBody
from .scene import CalvinSceneRef

GRIPPER_OPEN, GRIPPER_CLOSE = 1.0, -1.0   # CALVIN's convention (robot.py control_gripper) — robosuite's is inverted
GRIPPER_MAX_WIDTH = 0.08                  # gripper_joint_limits [0, 0.04] on two fingers: the 0-100 opening scale
# What the world serves; anything else the release's config declares (the DIGIT `tactile`
# sensor, which needs the `tacto` package) is dropped by get_env's own obs_space filter.
OBS_SPACE = {"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []}
TASKS_CONF = ("conf", "tasks", "new_playtable_tasks.yaml")   # inside the installed calvin_env checkout


def _import_calvin() -> Any:
    """The ``calvin_env`` module that owns ``PlayTableSimEnv``. Importing it
    installs ``rich``'s traceback hook (``play_table_env.py``:26,
    ``install(show_locals=True)``), which would reformat every traceback in
    the interpreter that serves these lines; the hook is undone here."""
    hook = sys.excepthook
    try:
        from calvin_env.envs import play_table_env
    except ImportError as e:
        raise ImportError(
            "the CALVIN lines need the `calvin_env` package (pybullet + hydra) in this interpreter — "
            "see INSTALL-calvin.md"
        ) from e
    finally:
        sys.excepthook = hook
    _patch_del(play_table_env)
    return play_table_env


def _patch_del(module: Any) -> None:
    """``PlayTableSimEnv.__del__`` calls ``close()`` unconditionally
    (:77), so a world that closed cleanly disconnects a second time at
    collection and pybullet raises "Not connected to physics server" —
    printed as an ignored exception, pure noise. (Same shape as the
    robosuite render-context patch in the LIBERO world.)"""
    cls = module.PlayTableSimEnv
    if getattr(cls, "_embodiedscore_del_patched", False):
        return
    original = cls.__del__

    def safe_del(self):
        try:
            original(self)
        except Exception:
            pass

    cls.__del__ = safe_del
    cls._embodiedscore_del_patched = True


def task_oracle() -> Any:
    """CALVIN's ``Tasks`` oracle, instantiated from the task table of the
    installed ``calvin_env`` (``conf/tasks/new_playtable_tasks.yaml``)."""
    import calvin_env
    import hydra
    from omegaconf import OmegaConf

    root = os.path.dirname(os.path.dirname(os.path.abspath(calvin_env.__file__)))
    path = os.path.join(root, *TASKS_CONF)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CALVIN task table not found at {path} — is calvin_env an editable checkout?")
    return hydra.utils.instantiate(OmegaConf.load(path))


def mat_to_wxyz(m: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(np.asarray(m, dtype=np.float64).reshape(3, 3)).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def rotvec_to_euler(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle (the package's macro convention) -> pybullet's extrinsic
    x-y-z Euler, which ``Robot.apply_action`` feeds to
    ``p.getQuaternionFromEuler``."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_euler("xyz")


class CalvinWorld:
    def __init__(self, body: CalvinBody, gpu_id: int = 0) -> None:
        self.body = body
        self.gpu_id = int(gpu_id)
        self._env: Any = None
        self._config: str | None = None
        self._oracle: Any = None
        self._obs: dict[str, Any] | None = None
        self._info: dict[str, Any] | None = None

    # ---- lifecycle ----------------------------------------------------------------------
    def load_scene(self, scene: CalvinSceneRef) -> None:
        """Build the play table from the release's own hydra config, once —
        every CALVIN episode of a line runs in the same scene, so this is a
        no-op after the first call.

        This is ``calvin_env.envs.play_table_env.get_env`` (:270) inlined,
        for one reason: the body's camera sizes have to be in the config
        *before* the environment is constructed. ``PlayTableSimEnv.__init__``
        sizes the EGL window from ``max(cameras[c].width)`` (:56) and creates
        the pybullet context with it, so a camera enlarged afterwards would
        render off-context. Everything else is upstream's: the same file, the
        same ``obs_space`` camera filter, the same ``instantiate`` call with
        ``show_gui=False, use_vr=False, use_scene_info=True``.

        Only the raster size changes — both cameras have ``aspect: 1`` and
        their projection is built from ``fov`` and ``aspect`` alone
        (``static_camera.py``:44, ``gripper_camera.py``:35), so the view is
        the release's at a different resolution.
        """
        if self._env is not None and self._config == scene.config_file:
            return
        self.close()
        _import_calvin()
        import hydra
        from omegaconf import OmegaConf

        conf = OmegaConf.load(scene.config_file)
        keep = {re.split("_", k)[1] for k in OBS_SPACE["rgb_obs"] + OBS_SPACE["depth_obs"]}
        if self.body.wrist is None:
            keep.discard("gripper")
        for name in set(conf.cameras.keys()) - keep:      # `tactile` (a DIGIT sensor needing `tacto`)
            del conf.cameras[name]
        for name, cam in (("static", self.body.rgb), ("gripper", self.body.wrist)):
            if cam is not None and name in conf.cameras:
                conf.cameras[name].width, conf.cameras[name].height = int(cam.width), int(cam.height)
        # (upstream also calls ``hydra.initialize(".")`` here; ``instantiate`` needs no global
        #  hydra, and claiming the process-wide one from inside an env server would be rude.)
        self._env = hydra.utils.instantiate(conf.env, show_gui=False, use_vr=False, use_scene_info=True)
        self._config = scene.config_file
        self._oracle = task_oracle()

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        self._env, self._config, self._oracle = None, None, None
        self._obs, self._info = None, None

    @property
    def env(self) -> Any:
        return self._env

    @property
    def n_tasks(self) -> int:
        return int(self._oracle.num_tasks)

    # ---- episode ---------------------------------------------------------------------------
    def reset(self, robot_obs: np.ndarray, scene_obs: np.ndarray) -> int:
        """Put the table into an episode's initial state, exactly as
        ``evaluate_policy.evaluate_sequence`` does (:128-129), and let it
        settle for ``body.settle_ticks`` (0 upstream). Returns the ticks
        spent."""
        self._obs = self._env.reset(robot_obs=np.asarray(robot_obs, dtype=np.float64),
                                    scene_obs=np.asarray(scene_obs, dtype=np.float64))
        self._info = self._env.get_info()
        ticks = 0
        if self.body.settle_ticks:
            pos, rot = self.eef_pose()
            euler = rotvec_to_euler(_rotvec_of_matrix(rot))
            for _ in range(self.body.settle_ticks):
                self.tick_absolute(pos, euler, GRIPPER_OPEN)
                ticks += 1
        return ticks

    def tick_relative(self, action: np.ndarray) -> None:
        """One control tick on CALVIN's own 7-D relative action
        ``[dx, dy, dz, dax, day, daz, gripper]`` (unit = 2 cm / 0.05 rad of
        target shift; gripper +1 open, -1 close)."""
        a = np.nan_to_num(np.asarray(action, dtype=np.float64).reshape(7))
        a[6] = GRIPPER_OPEN if a[6] > 0 else GRIPPER_CLOSE       # apply_action asserts the value is +1 / -1
        obs, _reward, _done, info = self._env.step(a)
        self._obs, self._info = obs, info

    def tick_absolute(self, position: np.ndarray, euler: np.ndarray, gripper: float) -> None:
        """One control tick holding an ABSOLUTE end-effector target —
        CALVIN's own three-argument action (see the module docstring).
        ``euler`` is pybullet's extrinsic x-y-z.

        The gripper must be a python ``int`` on this path: ``apply_action``
        (``robot.py``:321) length-checks anything that is not one, and the
        relative path only gets away with a float because ``np.split`` hands
        it a length-1 array."""
        grip = int(GRIPPER_OPEN if float(gripper) > 0 else GRIPPER_CLOSE)
        action = (np.asarray(position, dtype=np.float64).reshape(3),
                  np.asarray(euler, dtype=np.float64).reshape(3), grip)
        obs, _reward, _done, info = self._env.step(action)
        self._obs, self._info = obs, info

    # ---- the task oracle -------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """``start_info`` for the oracle: the scene state a sub-task's
        success is measured *against* (``evaluate_policy.rollout``:158)."""
        return self._env.get_info()

    def achieved(self, start_info: dict[str, Any], names: set[str]) -> set[str]:
        """Which of ``names`` the change from ``start_info`` to now satisfies
        — ``Tasks.get_task_info_for_set``, the evaluator's own check
        (``evaluate_policy.rollout``:172)."""
        assert self._info is not None, "reset() first"
        return set(self._oracle.get_task_info_for_set(start_info, self._info, set(names)))

    # ---- facts ------------------------------------------------------------------------------
    def observe(self) -> dict[str, np.ndarray]:
        assert self._obs is not None, "reset() first"
        rgb = self._obs["rgb_obs"]
        out = {"rgb": np.ascontiguousarray(np.asarray(rgb["rgb_static"], dtype=np.uint8))}
        if self.body.wrist is not None:
            out["wrist"] = np.ascontiguousarray(np.asarray(rgb["rgb_gripper"], dtype=np.uint8))
        out["proprio"] = self.proprio()
        return out

    def proprio(self) -> np.ndarray:
        """CALVIN's own 15-D ``robot_obs`` (``robot.py`` ``get_observation``
        with ``euler_obs: true``): tcp position 3 + tcp Euler 3 + gripper
        opening width 1 + arm joint states 7 + gripper action 1."""
        return np.asarray(self._obs["robot_obs"], dtype=np.float32)

    def scene_state(self) -> np.ndarray:
        """CALVIN's 24-D ``scene_obs``: slide + drawer + button + switch +
        lightbulb + led, then position 3 and Euler 3 for each of the three
        blocks (``play_table_scene.py`` ``get_obs`` / ``parse_scene_obs``)."""
        return np.asarray(self._obs["scene_obs"], dtype=np.float32)

    def eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(position (3,), rotation matrix (3, 3)) of the TCP link, world frame."""
        import pybullet as p

        robot = self._env.robot
        pos, orn = p.getLinkState(robot.robot_uid, robot.tcp_link_id, physicsClientId=self._env.cid)[:2]
        rot = np.asarray(p.getMatrixFromQuaternion(orn), dtype=np.float64).reshape(3, 3)
        return np.asarray(pos, dtype=np.float64), rot

    def gripper_open(self) -> float:
        """The gripper opening, 0 (closed) .. 100 (fully open), from the
        finger width CALVIN reports."""
        width = float(self._info["robot_info"]["gripper_opening_width"])
        return float(np.clip(width / GRIPPER_MAX_WIDTH, 0.0, 1.0) * 100.0)

    def objects(self) -> dict[str, Any]:
        """The scene's movable objects and articulations — privileged, for
        metrics and debugging only."""
        info = (self._info or {}).get("scene_info") or {}
        out: dict[str, Any] = {}
        for name, obj in (info.get("movable_objects") or {}).items():
            out[name] = {"position": [float(v) for v in obj["current_pos"]],
                         "rotation": _xyzw_to_wxyz(obj["current_orn"]).tolist()}
        for name, door in (info.get("doors") or {}).items():
            out[name] = {"joint_state": float(np.asarray(door["current_state"]).reshape(-1)[0])}
        for name, light in (info.get("lights") or {}).items():
            out[name] = {"on": int(light["logical_state"])}
        return out

    def robot_base(self) -> list[float]:
        return [float(v) for v in np.asarray(self._env.robot.base_position, dtype=np.float64)]


def _xyzw_to_wxyz(q: Any) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _rotvec_of_matrix(m: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(np.asarray(m, dtype=np.float64).reshape(3, 3)).as_rotvec()
