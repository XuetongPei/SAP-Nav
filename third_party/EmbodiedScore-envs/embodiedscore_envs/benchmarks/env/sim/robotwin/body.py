"""Embodiment parameters for the RoboTwin engine — pure data (layer L0, no
sapien import).

The robot is RoboTwin's: a dual-arm tabletop platform (the default
``aloha-agilex``: two 6-DoF arms with a one-DoF parallel gripper each,
mounted on a 0.74 m table) under SAPIEN 3 at a 1/250 s physics step
(``envs/_base_task.py:223``). Motion is planned, not servoed — one action is
a target the planner drives the arm to, time-parameterised at 250 Hz and run
to completion (``envs/_base_task.py:1486`` ``take_action``). None of that is
a knob here; it is the benchmark's physics.

What a body fixes:

* ``embodiment`` — the row of ``env_cfg/task_config/_embodiment_config.yml``
  whose URDF, joint names, gripper scale and camera mounts are loaded.
  ``arm_dof`` and ``gripper_dof`` transcribe that embodiment's
  ``arm_joints_name`` / ``gripper_name`` so the action width is known
  without loading the simulator (aloha-agilex: 6 + 1 per arm,
  ``assets/embodiments/aloha-agilex/config.yml``).
* the camera rig — the head camera (third person, mounted over the table),
  the two wrist cameras, and the optional front camera. RoboTwin names camera
  *types* and looks their geometry up in
  ``env_cfg/task_config/_camera_config.yml``; the specs here transcribe that
  file, and the world asserts the transcription against what RoboTwin loads.
* ``settle_ticks`` — extra zero-motion physics steps after a reset. RoboTwin
  already settles every scene for 2500 steps inside its own stability check
  (``envs/_base_task.py:186-191``, 2000 + 500 at 1/250 s = 10 s of simulated
  rest), so the default is 0.
* ``randomization`` — the ``domain_randomization`` block of the task config.
  RoboTwin ships two, ``demo_clean.yml`` and ``demo_randomized.yml``; each is
  transcribed as a preset in ``presets/bodies.py`` and each is a benchmark
  line.

There is no depth camera in the rig: RoboTwin's shipped configs collect RGB
only (``data_type.depth: false`` in both ``demo_clean.yml`` and
``demo_randomized.yml``), so ``depth`` is None — the field every engine's
Body has, which lets ``make()`` skip DepthClip.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotwinCameraSpec:
    """One RoboTwin camera type. ``name`` is the key the task config names
    (``head_camera_type`` / ``wrist_camera_type``); the geometry transcribes
    that key's row of ``env_cfg/task_config/_camera_config.yml``."""

    name: str
    width: int
    height: int
    fovy_deg: float


# env_cfg/task_config/_camera_config.yml, verbatim.
D435 = RobotwinCameraSpec("D435", 320, 240, 37.0)
LARGE_D435 = RobotwinCameraSpec("Large_D435", 640, 480, 37.0)
L515 = RobotwinCameraSpec("L515", 320, 180, 45.0)
LARGE_L515 = RobotwinCameraSpec("Large_L515", 640, 360, 45.0)

CAMERA_TYPES = {c.name: c for c in (D435, LARGE_D435, L515, LARGE_L515)}


@dataclass(frozen=True)
class RobotwinRandomization:
    """The ``domain_randomization`` block of a RoboTwin task config, verbatim
    (``env_cfg/task_config/demo_clean.yml`` / ``demo_randomized.yml``). Read
    by ``Base_Task._init_task_env_`` (``envs/_base_task.py:75-84``)."""

    random_background: bool = False
    cluttered_table: bool = False
    clean_background_rate: float = 1.0
    random_head_camera_dis: float = 0.0
    random_table_height: float = 0.0
    random_light: bool = False
    crazy_random_light_rate: float = 0.0

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "random_background": self.random_background,
            "cluttered_table": self.cluttered_table,
            "clean_background_rate": self.clean_background_rate,
            "random_head_camera_dis": self.random_head_camera_dis,
            "random_table_height": self.random_table_height,
            "random_light": self.random_light,
            "crazy_random_light_rate": self.crazy_random_light_rate,
        }


@dataclass(frozen=True)
class RobotwinBody:
    embodiment: str = "aloha-agilex"                     # a key of env_cfg/task_config/_embodiment_config.yml
    head: RobotwinCameraSpec = D435                      # the third-person camera over the table
    wrist: RobotwinCameraSpec | None = D435              # both wrist cameras; None = not served
    front: RobotwinCameraSpec | None = None              # the embodiment's front camera (RoboTwin's third_view)
    arm_dof: int = 6                                     # joints per arm (aloha-agilex config.yml arm_joints_name)
    gripper_dof: int = 1                                 # gripper command width per arm, RoboTwin's normalised [0, 1]
    settle_ticks: int = 0                                # extra zero-motion physics steps after a reset
    randomization: RobotwinRandomization = RobotwinRandomization()

    def __post_init__(self) -> None:
        if self.arm_dof < 1 or self.gripper_dof != 1:
            raise ValueError("RobotwinBody: arm_dof >= 1 and gripper_dof == 1 "
                             "(RoboTwin commands one normalised value per gripper)")
        if self.head is None:
            raise ValueError("RobotwinBody: the head camera is the observation — it cannot be None")

    @property
    def depth(self) -> None:
        """No depth camera — the field every engine's Body has, so ``make()`` skips DepthClip."""
        return None

    @property
    def rgb(self) -> RobotwinCameraSpec:
        """The observation camera, under the name every engine's Body uses."""
        return self.head

    @property
    def has_tilt(self) -> bool:
        return False

    @property
    def qpos_dim(self) -> int:
        """RoboTwin's ``action_type='qpos'`` action width: arm + gripper, per
        arm (``envs/_base_task.py:1504-1528``)."""
        return 2 * (self.arm_dof + self.gripper_dof)

    @property
    def ee_dim(self) -> int:
        """RoboTwin's ``action_type='ee'`` action width: a 7-vector pose plus
        the gripper, per arm (``envs/_base_task.py:1504-1505`` pins the arm
        block at 7)."""
        return 2 * (7 + self.gripper_dof)

    def camera_block(self) -> dict[str, object]:
        """The ``camera`` block of a RoboTwin task config this rig asks for."""
        return {
            "head_camera_type": self.head.name,
            "wrist_camera_type": (self.wrist or self.head).name,
            "collect_head_camera": True,
            "collect_wrist_camera": self.wrist is not None,
        }
