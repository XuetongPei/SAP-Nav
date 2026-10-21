"""RobotwinWorld — the simulator facade of the RoboTwin engine (layer L0):
the only module that imports RoboTwin / sapien, lazily, at construction.

RoboTwin is not a pip package — it is a checkout whose ``envs/`` directory
holds one module per task (``envs/beat_block_hammer.py`` defines
``class beat_block_hammer(Base_Task)``). The checkout's root goes on
``sys.path`` here and nowhere else; ``EMBODIEDSCORE_ROBOTWIN_ROOT`` says
where it is (INSTALL-robotwin.md).

RoboTwin resolves its meshes and textures against the WORKING DIRECTORY, not
against its own ``__file__``: ``envs/utils/create_actor.py:451`` opens
``Path("./assets/objects") / modelname``, ``envs/_base_task.py:278`` reads
``./assets/background_texture/…``, ``envs/utils/rand_create_cluttered_actor.py:17``
reads a manifest at import time. Its own entrypoints ``cd`` to the checkout
first (``scripts/eval_policy.sh``, ``collect_data.sh``). This module does the
same, but only for the duration of a call (:func:`_in_root`), so nothing else
in the process sees the working directory move. Everything else RoboTwin
resolves — configs, embodiments, the camera table — comes off its own
``__file__`` (``envs/_GLOBAL_CONFIGS.py``); the embodiment directory is the one
place a relative path appears in a config and is made absolute here.

This world owns one task instance at a time (rebuilt when the task changes —
each task is its own python class and its own SAPIEN scene), puts an
episode's seed into it, runs actions, and reads facts: the frames, both
arms' end-effector poses, both gripper openings, the actor poses and the
task's own success predicate. It never counts budgets (the env body does)
and never decides an action (the bodies do).

Frames: SAPIEN world frame, metres, z up. Rotations are ``(w, x, y, z)``
quaternions throughout — sapien's own convention and RoboTwin's
(``envs/robot/robot.py:588`` ``_trans_endpose`` returns
``[x, y, z, qw, qx, qy, qz]``). Grippers are RoboTwin's normalised value,
0 closed .. 1 open (``envs/robot/robot.py:628``), exposed here on the
package's 0-100 scale.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from typing import Any

import numpy as np

from .body import CAMERA_TYPES, RobotwinBody
from .scene import RobotwinSceneRef

ARMS = ("left", "right")
UNSTABLE = "UnStableError"     # envs/utils/create_actor.py — RoboTwin's "this seed did not settle"
GRIPPER_OPEN, GRIPPER_CLOSE = 1.0, 0.0     # RoboTwin's normalised gripper command (robot.py:628 "gripper_val in [0,1]")
ROOT_ENV = "EMBODIEDSCORE_ROBOTWIN_ROOT"

# env_cfg/task_config/demo_clean.yml and demo_randomized.yml agree on this block; only
# `rgb` and the proprioception channels are read back by this port.
DATA_TYPE = {
    "rgb": True, "third_view": False, "depth": False, "pointcloud": False, "observer": False,
    "endpose": True, "qpos": True, "mesh_segmentation": False, "actor_segmentation": False,
}


def robotwin_root(explicit: str | os.PathLike | None = None) -> str:
    """The RoboTwin checkout. ``EMBODIEDSCORE_ROBOTWIN_ROOT`` unless given."""
    p = explicit or os.environ.get(ROOT_ENV)
    if not p:
        raise ValueError(f"the RoboTwin lines need the RoboTwin checkout — set {ROOT_ENV} (see INSTALL-robotwin.md)")
    p = str(p)
    if not os.path.isdir(os.path.join(p, "envs")):
        raise FileNotFoundError(f"{p} is not a RoboTwin checkout (no envs/ directory)")
    return p


def task_instruction(scene: RobotwinSceneRef, root: str | os.PathLike | None = None) -> str:
    """The task's language, from ``description/task_instruction/<task>.json``.

    RoboTwin's per-episode instructions are templates with ``{A}`` / ``{a}``
    placeholders that only its scripted expert can fill (the expert writes
    ``info["info"]``, and ``description/utils/generate_episode_instructions.py``
    substitutes it); 49 of the 50 tasks have no placeholder-free template at
    all. ``full_description`` is the one per-task statement that stands on
    its own, so it is what this port hands the agent; RoboTwin's own
    fallback when it has no episode info is the task name with underscores
    replaced (``scripts/collect_data.py:26``), which is what is used if the
    file is missing a description. The ``<>`` markers RoboTwin uses to bracket
    sub-goals inside ``full_description`` are stripped.
    """
    path = os.path.join(robotwin_root(root), scene.instruction_file)
    text = ""
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            text = str(json.load(f).get("full_description") or "").strip()
    if not text:
        return scene.task.replace("_", " ")
    text = text.replace("<", "").replace(">", "").strip()
    return text[0].upper() + text[1:] if text else text


@contextlib.contextmanager
def _in_root(root: str):
    """Run with the RoboTwin checkout as the working directory, then restore
    it — RoboTwin opens its meshes and textures through ``./assets/…``."""
    previous = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def _import_task(task: str, root: str) -> Any:
    """RoboTwin's task class, imported lazily with the checkout on sys.path."""
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import importlib

        with _in_root(root):     # envs/utils/rand_create_cluttered_actor.py reads a manifest at import time
            module = importlib.import_module(f"envs.{task}")
    except ImportError as e:
        raise ImportError(
            "the RoboTwin lines need the RoboTwin checkout's dependencies (sapien 3, mplib, curobo, "
            "toppra, torch) in this interpreter — see INSTALL-robotwin.md"
        ) from e
    try:
        return getattr(module, task)
    except AttributeError as e:
        raise ImportError(f"envs/{task}.py has no class named {task!r}") from e


def _embodiment(body: RobotwinBody, root: str) -> tuple[str, dict[str, Any]]:
    """(absolute embodiment directory, its config.yml) for the body's embodiment."""
    import yaml

    table = os.path.join(root, "env_cfg", "task_config", "_embodiment_config.yml")
    with open(table, encoding="utf-8") as f:
        rows = yaml.safe_load(f)
    if body.embodiment not in rows:
        raise ValueError(f"unknown embodiment {body.embodiment!r}; {table} has {sorted(rows)}")
    # The table stores paths relative to the checkout root ("./assets/embodiments/...").
    directory = os.path.normpath(os.path.join(root, rows[body.embodiment]["file_path"]))
    with open(os.path.join(directory, "config.yml"), encoding="utf-8") as f:
        return directory, yaml.safe_load(f)


class RobotwinWorld:
    def __init__(self, body: RobotwinBody, gpu_id: int = 0, root: str | os.PathLike | None = None) -> None:
        self.body = body
        self.gpu_id = int(gpu_id)
        self.root = robotwin_root(root)
        self._task: Any = None
        self._task_name: str | None = None
        self._scene: RobotwinSceneRef | None = None
        self._obs: dict[str, Any] | None = None
        self._live = False
        os.environ.setdefault("SAPIEN_HEADLESS", "1")

    # ---- lifecycle ----------------------------------------------------------------------
    def load_scene(self, scene: RobotwinSceneRef) -> None:
        """Hold the task's class (built once per task — each task is its own
        python class and its own SAPIEN scene). The scene itself is built by
        :meth:`reset`, which RoboTwin does per episode."""
        if self._task is not None and self._task_name == scene.task:
            self._scene = scene
            return
        self.close()
        self._task = _import_task(scene.task, self.root)()
        self._task_name = scene.task
        self._scene = scene

    def close(self) -> None:
        self._close_episode()
        self._task, self._task_name, self._scene = None, None, None

    def _close_episode(self) -> None:
        """Drop the episode's SAPIEN scene. Called before every setup, and
        after one that failed: a seed rejected by RoboTwin's stability check
        raises AFTER ``setup_scene`` built the scene (``envs/_base_task.py:90``
        vs ``:138``), so the scene has to be closed even though the episode
        never went live — RoboTwin's own evaluator does the same on a skipped
        seed (``scripts/eval_policy_xpolicylab.py:634``)."""
        if self._task is not None:
            try:
                with _in_root(self.root):
                    self._task.close_env()
            except Exception:
                pass
        self._live, self._obs = False, None

    @property
    def task(self) -> Any:
        if self._task is None or not self._live:
            raise RuntimeError("RobotwinWorld: reset() first")
        return self._task

    # ---- episode ---------------------------------------------------------------------------
    def reset(self, seed: int, episode_index: int, step_limit: int, need_plan: bool = False,
              seed_window: int = 1) -> tuple[int, int]:
        """Build the episode's scene and settle it. Returns
        ``(seed actually used, seeds tried)``.

        RoboTwin's own setup runs 2500 physics steps of settling and raises
        ``UnStableError`` if an actor is still moving
        (``envs/_base_task.py:186-192``); its evaluator answers that by
        skipping the seed and taking the next
        (``scripts/eval_policy_xpolicylab.py:633``). ``seed_window`` is how
        many consecutive seeds from ``seed`` this episode may spend before
        giving up, so the walk stays inside the episode's own window and the
        episode remains reproducible on its own.

        ``body.settle_ticks`` adds further physics steps. ``step_limit``
        replaces the cap RoboTwin reads from ``_eval_step_limit.yml`` so the
        env body owns the budget.
        """
        if self._task is None or self._scene is None:
            raise RuntimeError("RobotwinWorld: load_scene() first")
        last: Exception | None = None
        for attempt in range(max(1, int(seed_window))):
            candidate = int(seed) + attempt
            self._close_episode()
            args = self._task_args(self._scene, candidate, int(episode_index), bool(need_plan))
            try:
                with _in_root(self.root):
                    self._task.setup_demo(**args)
                    self._live = True
                    self._task.step_lim = int(step_limit)
                    for _ in range(self.body.settle_ticks):
                        self._task.scene.step()
                    self._obs = self._task.get_obs()
            except Exception as e:                      # RoboTwin's UnStableError, raised from deep inside setup
                self._close_episode()
                if type(e).__name__ != UNSTABLE:
                    raise
                last = e
                continue
            self._check_camera_rig()
            return candidate, attempt + 1
        raise RuntimeError(
            f"{self._scene.task}: no seed in [{seed}, {int(seed) + int(seed_window)}) produced a settled scene"
        ) from last

    def _task_args(self, scene: RobotwinSceneRef, seed: int, episode_index: int, need_plan: bool) -> dict[str, Any]:
        """RoboTwin's ``setup_demo`` keyword block, assembled from the body
        instead of read from ``env_cfg/task_config/<config>.yml``: the
        declaration is the single source of truth, and the body's
        randomisation / camera presets transcribe that file (presets/bodies.py).
        The collection-only keys (``save_data``, ``collect_data``,
        ``save_freq``, ``render_freq``, ``eval_video_save_dir``) are off —
        this port evaluates, it does not collect."""
        directory, config = _embodiment(self.body, self.root)
        return {
            "task_name": scene.task,
            "task_config": scene.task_config,
            "embodiment_name": scene.embodiment,
            "embodiment": [scene.embodiment],
            "left_robot_file": directory,
            "right_robot_file": directory,
            "left_embodiment_config": config,
            "right_embodiment_config": config,
            "dual_arm_embodied": True,
            "dual_arm": True,
            "domain_randomization": self.body.randomization.as_dict(),
            "camera": self.body.camera_block(),
            "data_type": dict(DATA_TYPE),
            "pcd_down_sample_num": 1024,          # demo_clean.yml; unread with data_type.pointcloud false
            "pcd_crop": True,                     # demo_clean.yml
            "render_freq": 0,                     # demo_clean.yml — no on-screen viewer
            "save_freq": None,                    # nothing is written to disk
            "save_data": False,
            "collect_data": False,
            "eval_mode": True,                    # _base_task.py:141 — RoboTwin's own evaluation branch
            "eval_video_log": False,
            "eval_video_save_dir": None,
            "eval_instruction": "seen",
            "need_plan": need_plan,
            "save_path": os.path.join(self.root, "data"),   # unused: save_data is off
            "now_ep_num": episode_index,
            "seed": seed,
            "is_test": True,
        }

    def _check_camera_rig(self) -> None:
        """The body transcribes ``env_cfg/task_config/_camera_config.yml``;
        assert the transcription against what RoboTwin actually loaded."""
        want = {"head_camera": self.body.head}
        if self.body.wrist is not None:
            want["left_camera"] = want["right_camera"] = self.body.wrist
        for name, spec in want.items():
            if spec.name not in CAMERA_TYPES:
                continue
            frame = self._frame(name)
            if frame is None:
                raise RuntimeError(f"RoboTwin did not serve camera {name!r}")
            if frame.shape[:2] != (spec.height, spec.width):
                raise RuntimeError(
                    f"camera {name!r}: RoboTwin rendered {frame.shape[1]}x{frame.shape[0]}, "
                    f"the body declares {spec.width}x{spec.height} (camera type {spec.name!r})"
                )

    def take_action(self, action: np.ndarray, action_type: str = "qpos") -> bool:
        """One RoboTwin action — ``Base_Task.take_action``
        (``envs/_base_task.py:1486``). The whole planned motion runs inside
        this call; ``check_success`` is polled at every physics step and
        latches ``eval_success`` (``envs/_base_task.py:1664``). Returns the
        latched success.
        """
        if action_type not in ("qpos", "ee"):
            raise ValueError(f"action_type must be 'qpos' or 'ee', got {action_type!r}")
        a = np.nan_to_num(np.asarray(action, dtype=np.float64), posinf=0.0, neginf=0.0).reshape(-1)
        with _in_root(self.root):
            self.task.take_action(a.tolist(), action_type=action_type)
            self._obs = self.task.get_obs()
        return self.success

    def play_expert(self) -> dict[str, Any]:
        """RoboTwin's own scripted expert for this task (``play_once``) —
        privileged: it reads actor poses straight out of the simulator. Used
        to validate a seed the way RoboTwin's evaluator does
        (``scripts/eval_policy_xpolicylab.py:630``), and to source waypoints
        for a scripted proof episode. Needs ``need_plan=True`` at reset."""
        with _in_root(self.root):
            info = self.task.play_once()
            self._obs = self.task.get_obs()
        return dict(info or {})

    # ---- facts ------------------------------------------------------------------------------
    @property
    def success(self) -> bool:
        """The task's own predicate, latched — RoboTwin latches it into
        ``eval_success`` inside ``take_action``; polled here as well so a goal
        that already holds is not missed."""
        if not self._live:
            return False
        if bool(getattr(self._task, "eval_success", False)):
            return True
        return bool(self._task.check_success())

    @property
    def ticks(self) -> int:
        """RoboTwin's own action counter — what ``_eval_step_limit.yml`` caps."""
        return int(getattr(self._task, "take_action_cnt", 0)) if self._live else 0

    @property
    def step_limit(self) -> int:
        return int(getattr(self._task, "step_lim", 0) or 0) if self._live else 0

    @property
    def plan_success(self) -> bool:
        """False once a planner call inside the expert failed (RoboTwin's own flag)."""
        return bool(getattr(self._task, "plan_success", True)) if self._live else True

    def _frame(self, camera: str) -> np.ndarray | None:
        obs = self._obs or {}
        cam = (obs.get("observation") or {}).get(camera) or {}
        rgb = cam.get("rgb")
        return None if rgb is None else np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))

    def observe(self) -> dict[str, np.ndarray]:
        """``rgb`` is the head camera; the wrist cameras ride alongside under
        the arm they are mounted on. RoboTwin renders upright — unlike
        robosuite, no flip is needed."""
        if self._obs is None:
            raise RuntimeError("RobotwinWorld: reset() first")
        out = {"rgb": self._frame("head_camera")}
        if self.body.wrist is not None:
            out["left_wrist"] = self._frame("left_camera")
            out["right_wrist"] = self._frame("right_camera")
        if self.body.front is not None:
            front = self._frame("front_camera")
            if front is not None:
                out["front"] = front
        out["proprio"] = self.proprio()
        return {k: v for k, v in out.items() if v is not None}

    def proprio(self) -> np.ndarray:
        """RoboTwin's own joint vector: left arm joints + left gripper +
        right arm joints + right gripper (``envs/_base_task.py:496``
        ``joint_action.vector``), ``2 * (arm_dof + 1)`` wide."""
        vector = ((self._obs or {}).get("joint_action") or {}).get("vector")
        if vector is None:
            robot = self.task.robot
            vector = robot.get_left_arm_jointState() + robot.get_right_arm_jointState()
        return np.asarray(vector, dtype=np.float32).reshape(-1)

    def arm_pose(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        """(position (3,), rotation ``(w, x, y, z)``) of the arm's end-effector,
        world frame — ``Base_Task.get_arm_pose`` (``envs/_base_task.py:1404``),
        the same frame ``take_action(action_type='ee')`` targets."""
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        pose = np.asarray(self.task.get_arm_pose(arm), dtype=np.float64).reshape(7)
        return pose[:3].copy(), pose[3:].copy()

    def gripper_value(self, arm: str) -> float:
        """RoboTwin's normalised gripper value, 0 closed .. 1 open."""
        robot = self.task.robot
        return float(robot.get_left_gripper_val() if arm == "left" else robot.get_right_gripper_val())

    def gripper_open(self, arm: str) -> float:
        """The gripper opening on the package's scale, 0 (closed) .. 100 (open)."""
        return float(np.clip(self.gripper_value(arm), 0.0, 1.0) * 100.0)

    def objects(self) -> dict[str, dict[str, list[float]]]:
        """Pose of every actor the task placed — privileged, for metrics and
        debugging. RoboTwin has no manifest of "objects of interest", so this
        is every non-static actor of the SAPIEN scene under its own name."""
        out: dict[str, dict[str, list[float]]] = {}
        if not self._live:
            return out
        for actor in self._task.scene.get_all_actors():
            name = str(actor.get_name() or "")
            if not name or name in out:
                continue
            pose = actor.get_pose()
            out[name] = {"position": [float(v) for v in pose.p], "rotation": [float(v) for v in pose.q]}
        return out

    def robot_base(self) -> dict[str, list[float]]:
        robot = self.task.robot
        return {
            "left": [float(v) for v in robot.left_entity_origion_pose.p],
            "right": [float(v) for v in robot.right_entity_origion_pose.p],
        }

    def instruction(self) -> str:
        if self._scene is None:
            raise RuntimeError("RobotwinWorld: load_scene() first")
        return task_instruction(self._scene, self.root)
