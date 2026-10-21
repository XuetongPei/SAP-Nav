"""Embodiment parameters for the CALVIN engine — pure data (layer L0, no
pybullet import).

The robot is CALVIN's: a Franka Panda with a parallel gripper on the play
table, driven at 30 Hz over 240 Hz pybullet steps (``calvin_env/conf/env/
play_table_env.yaml``: ``bullet_time_step: 240.0``, ``control_freq: 30`` —
so one control tick is 8 physics steps). The end-effector command goes
through pybullet's IK and joint position control (``calvin_env/robot/robot.py``
``apply_action`` -> ``MixedIK.get_ik`` -> ``setJointMotorControl2``); the
gripper is a binary command, ``+1`` open and ``-1`` close
(``robot.py:355`` ``control_gripper``: ``+1`` drives the fingers to
``gripper_joint_limits[1]``, the OPPOSITE sign of robosuite's convention,
which the LIBERO and RoboCasa engines use). None of that is a knob here —
it is the benchmark's physics.

What a body fixes is the camera rig and how many settle ticks follow a
reset. Both cameras' geometry is the dataset's hydra config, which is also
``calvin_env/conf/cameras/cameras/{static,gripper}.yaml``:

    static    200x200, fov 10 (a telephoto third-person view of the table)
    gripper    84x84,  fov 75 (wrist camera, ``gripper_cam_link`` 12)

The third camera the config declares, ``tactile``, needs the ``tacto``
package and serves a DIGIT sensor no line here reads; the world drops it
through ``get_env``'s own ``obs_space`` filter (see ``world.py``).

``settle_ticks`` is 0: CALVIN's evaluator resets and starts the rollout on
the next tick (``calvin_models/calvin_agent/evaluation/evaluate_policy.py``
:129 ``env.reset(robot_obs=..., scene_obs=...)`` then :154 ``env.get_obs()``),
and ``PlayTableSimEnv.reset`` already runs one ``stepSimulation``. The field
exists because every manipulation body has it.

The environment letter (A / B / C / D) is NOT here: A-D are *scenes*, not
rigs — the letter rides ``CalvinSceneRef``, as the LIBERO suite rides
``LiberoSceneRef``. The camera geometry above is identical in all four.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalvinCameraSpec:
    width: int
    height: int


@dataclass(frozen=True)
class CalvinBody:
    rgb: CalvinCameraSpec = CalvinCameraSpec(200, 200)          # the static third-person camera
    wrist: CalvinCameraSpec | None = CalvinCameraSpec(84, 84)   # the gripper camera; None = not served
    control_freq: int = 30          # calvin_env/conf/env/play_table_env.yaml
    settle_ticks: int = 0           # CALVIN's evaluator settles for none (see the module docstring)

    @property
    def depth(self) -> None:
        """No depth camera served — the field every engine's Body has, so
        ``make()`` skips DepthClip. (calvin_env renders depth for both
        cameras; no line here asks for it.)"""
        return None

    @property
    def has_tilt(self) -> bool:
        return False
