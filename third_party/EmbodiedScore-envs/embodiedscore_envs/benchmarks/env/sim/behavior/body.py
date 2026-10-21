"""Embodiment parameters for the BEHAVIOR engine — pure data (layer L0, no
omnigibson import).

The robot is the one the 2025 BEHAVIOR Challenge fixes: **R1 Pro**, asserted
by the challenge's own evaluator —
``OmniGibson/omnigibson/learning/eval.py`` L128,
``assert robot_type == "R1Pro", f"Got invalid robot type: {robot_type}, only
R1Pro is supported."`` — a Galaxea R1 Pro: a holonomic wheeled base (three
omni wheels, six virtual joints), a four-joint torso, two 7-DOF arms and a
two-finger parallel gripper on each, 28 DOF in all
(``omnigibson/robots/r1pro.py``; ``R1Pro(R1)``, ``R1(HolonomicBaseRobot,
ArticulatedTrunkRobot, MobileManipulationRobot)`` in ``robots/r1.py`` L11).

None of the robot's physics is a knob here — the challenge fixes it:

* control at 30 Hz, rendering at 30 Hz, physics at 120 Hz
  (``learning/utils/eval_utils.py`` ``generate_basic_environment_config``
  L256-261: ``action_frequency: 30, rendering_frequency: 30,
  physics_frequency: 120``);
* the controllers of ``joylo/gello/robots/sim_robot/og_teleop_cfg.py``
  ``R1_CONTROLLER_CONFIG`` L181-232 — base ``HolonomicBaseJointController``
  (velocity, output ±[0.75, 0.75, 1.0] m/s · m/s · rad/s), trunk and both
  arms ``JointController`` (position, ``use_delta_commands: false`` — i.e.
  ABSOLUTE joint angles, ``pos_kp: 150``), both grippers
  ``MultiFingerGripperController`` in ``mode: "smooth"``;
* ``grasping_mode: "assisted"``, ``self_collisions: True``
  (``joylo/gello/robots/sim_robot/og_teleop_utils.py`` L1503, L1507).

What a body fixes is the camera rig — the three cameras the challenge serves
and their resolution — and how many zero-motion control ticks follow a reset.

Cameras are the robot's own, named as the challenge names them
(``learning/utils/eval_utils.py`` ``ROBOT_CAMERA_NAMES["R1Pro"]`` L11-15):
a ZED head camera and a RealSense in each wrist. ``head_aperture_mm`` is the
horizontal aperture both shipped evaluation wrappers force onto the head
camera to match the data collection rig (``learning/wrappers/
rgb_low_res_wrapper.py`` L24 and ``default_wrapper.py`` L23:
``horizontal_aperture = 40.0``).

BEHAVIOR's own settling happens inside the instance load — 25 physics steps
with every task-relevant entity held still (``eval.py`` L281-286) — so
``settle_ticks`` (extra zero-motion CONTROL ticks after a reset, as the
LIBERO and RoboCasa bodies have) adds nothing and is 0 by default.
"""

from __future__ import annotations

from dataclasses import dataclass

HEAD_CAMERA = "head"
LEFT_WRIST_CAMERA = "left_wrist"
RIGHT_WRIST_CAMERA = "right_wrist"

# eval_utils.py ROBOT_CAMERA_NAMES["R1Pro"] L11-15 — the prim paths of the three served cameras.
CAMERA_PRIMS = {
    HEAD_CAMERA: "robot_r1::robot_r1:zed_link:Camera:0",
    LEFT_WRIST_CAMERA: "robot_r1::robot_r1:left_realsense_link:Camera:0",
    RIGHT_WRIST_CAMERA: "robot_r1::robot_r1:right_realsense_link:Camera:0",
}
ROBOT_NAME = "robot_r1"          # og_teleop_cfg.py L111-112: the scene name of the challenge robot
ARMS = ("left", "right")         # r1.py L215-221 — arm_names

# eval_utils.py L19-20 — the resolutions the raw evaluation rig renders at.
HEAD_RESOLUTION = (720, 720)
WRIST_RESOLUTION = (480, 480)
HEAD_APERTURE_MM = 40.0          # rgb_low_res_wrapper.py L24 / default_wrapper.py L23


@dataclass(frozen=True)
class BehaviorCameraSpec:
    width: int
    height: int


@dataclass(frozen=True)
class BehaviorBody:
    rgb: BehaviorCameraSpec = BehaviorCameraSpec(224, 224)                # the head view served as obs["rgb"]
    wrist: BehaviorCameraSpec | None = BehaviorCameraSpec(224, 224)       # obs["wrist"] (left) and obs["wrist_right"]
    rgb_camera: str = HEAD_CAMERA
    wrist_camera: str = LEFT_WRIST_CAMERA
    aux_camera: str = RIGHT_WRIST_CAMERA
    robot: str = "R1Pro"
    head_aperture_mm: float = HEAD_APERTURE_MM
    action_freq: int = 30           # eval_utils.generate_basic_environment_config: action_frequency
    render_freq: int = 30           # ... rendering_frequency
    physics_freq: int = 120         # ... physics_frequency
    settle_ticks: int = 0           # BEHAVIOR settles inside the instance load (eval.py L281-286)
    partial_scene_load: bool = False  # learning/configs/base_config.yaml: partial_scene_load: false

    @property
    def arms(self) -> tuple[str, ...]:
        """The robot's arm names, in the challenge's controller order (r1.py L166-170)."""
        return ARMS

    @property
    def cameras(self) -> tuple[str, ...]:
        """The camera ids the world asks OmniGibson to render, in obs order."""
        names = [self.rgb_camera]
        if self.wrist is not None:
            names += [self.wrist_camera, self.aux_camera]
        return tuple(names)

    def resolution_of(self, camera: str) -> tuple[int, int]:
        """(width, height) for a camera id — the head camera's rig, or the wrists'."""
        spec = self.rgb if camera == self.rgb_camera else self.wrist
        if spec is None:
            raise KeyError(f"BehaviorBody: camera {camera!r} is not served")
        return (spec.width, spec.height)

    @property
    def depth(self) -> None:
        """No depth camera on the challenge rig — the field every engine's Body
        has, so ``make()`` skips DepthClip. (The challenge's ``DefaultWrapper``
        does serve depth; this port serves the RGB modalities only.)"""
        return None

    @property
    def has_tilt(self) -> bool:
        return False
