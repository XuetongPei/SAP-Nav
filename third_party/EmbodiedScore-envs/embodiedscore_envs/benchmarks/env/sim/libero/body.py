"""Embodiment parameters for the LIBERO engine — pure data (layer L0, no
robosuite import).

The robot is LIBERO's: a Franka Panda under robosuite's OSC_POSE controller
(20 Hz; a unit action moves the end-effector goal 5 cm / 0.5 rad; kp 150,
critical damping, position and orientation uncoupled; +1 closes the gripper,
-1 opens it). None of that is a knob here — it is the benchmark's physics.
What a body fixes is the camera rig (the third-person ``agentview`` and the
``robot0_eye_in_hand`` wrist camera), how many settle ticks follow a reset
(OpenVLA's evaluator and LIBERO's own use 10), and whether the frames are
turned upright (robosuite's offscreen renderer returns them upside down;
every LIBERO consumer flips them).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiberoCameraSpec:
    width: int
    height: int


@dataclass(frozen=True)
class LiberoBody:
    rgb: LiberoCameraSpec = LiberoCameraSpec(256, 256)            # agentview
    wrist: LiberoCameraSpec | None = LiberoCameraSpec(256, 256)   # robot0_eye_in_hand; None = not served
    control_freq: int = 20
    settle_ticks: int = 10          # zero-motion, gripper-open ticks after a reset (objects come to rest)
    upright: bool = True            # flip the renderer's frames so the table is at the bottom

    def __post_init__(self) -> None:
        if self.wrist is not None and (self.wrist.width, self.wrist.height) != (self.rgb.width, self.rgb.height):
            # robosuite renders every camera at one size per env
            raise ValueError("LiberoBody: the wrist camera must share the agentview resolution")

    @property
    def depth(self) -> None:
        """No depth camera — the field every engine's Body has, so ``make()`` skips DepthClip."""
        return None

    @property
    def has_tilt(self) -> bool:
        return False
