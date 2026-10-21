"""Embodiment parameters for the RoboCasa engine — pure data (layer L0, no
robosuite import).

The robot is RoboCasa's: a Franka Panda on an Omron mobile base
(``PandaOmron``; ``robocasa/utils/env_utils.py`` ``create_env`` defaults
``robots="PandaOmron"``) under robosuite's ``HYBRID_MOBILE_BASE`` composite
controller — the arm on ``OSC_POSE`` at 20 Hz with a unit action moving the
end-effector goal 5 cm / 0.5 rad **in the arm's base frame** (kp 150,
critical damping, position and orientation uncoupled), the torso on
``JOINT_POSITION`` and the mobile base on ``JOINT_VELOCITY``
(``robosuite/controllers/config/robots/default_pandaomron.json``). None of
that is a knob here — it is the benchmark's physics.

What a body fixes is the camera rig (RoboCasa renders one resolution per
env, so every served camera shares it), how many settle ticks follow a
reset, and whether the frames are turned upright (robosuite's offscreen
renderer returns them flipped; RoboCasa's own gym wrapper flips the rows
back — ``robocasa/wrappers/gym_wrapper.py`` ``get_basic_observation``,
``np.copy(img[::-1, :, :])``).

Camera names are RoboCasa's own (``robocasa/utils/camera_utils.py``
``DEFAULT_LAYOUT_CAM``): ``robot0_agentview_center`` / ``_left`` / ``_right``
are the third-person views mounted on the mobile base, ``robot0_eye_in_hand``
the wrist camera.
"""

from __future__ import annotations

from dataclasses import dataclass

HEAD_CAMERA = "robot0_agentview_center"
LEFT_CAMERA = "robot0_agentview_left"
RIGHT_CAMERA = "robot0_agentview_right"
WRIST_CAMERA = "robot0_eye_in_hand"


@dataclass(frozen=True)
class RobocasaCameraSpec:
    width: int
    height: int


@dataclass(frozen=True)
class RobocasaBody:
    rgb: RobocasaCameraSpec = RobocasaCameraSpec(256, 256)             # the third-person view served as obs["rgb"]
    wrist: RobocasaCameraSpec | None = RobocasaCameraSpec(256, 256)    # obs["wrist"]; None = not served
    aux: RobocasaCameraSpec | None = None                              # obs["aux"]: RoboCasa's second agentview
    rgb_camera: str = HEAD_CAMERA
    wrist_camera: str = WRIST_CAMERA
    aux_camera: str = RIGHT_CAMERA
    robot: str = "PandaOmron"
    control_freq: int = 20          # robocasa/environments/kitchen/kitchen.py __init__ default
    settle_ticks: int = 10          # zero-motion, gripper-open ticks after a reset (objects come to rest)
    upright: bool = True            # flip the renderer's rows, as RoboCasa's gym wrapper does

    def __post_init__(self) -> None:
        for name in ("wrist", "aux"):
            cam = getattr(self, name)
            if cam is not None and (cam.width, cam.height) != (self.rgb.width, self.rgb.height):
                # robosuite renders every camera at one size per env
                raise ValueError(f"RobocasaBody: the {name} camera must share the third-person resolution")

    @property
    def cameras(self) -> tuple[str, ...]:
        """The camera names the world asks robosuite to render, in obs order."""
        names = [self.rgb_camera]
        if self.wrist is not None:
            names.append(self.wrist_camera)
        if self.aux is not None:
            names.append(self.aux_camera)
        return tuple(names)

    @property
    def depth(self) -> None:
        """No depth camera — the field every engine's Body has, so ``make()`` skips DepthClip."""
        return None

    @property
    def has_tilt(self) -> bool:
        return False
