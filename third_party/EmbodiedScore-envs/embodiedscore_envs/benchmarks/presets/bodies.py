"""Embodiment presets — every number an upstream config or the EmbodiedScore
protocol fixes, nothing derived.

STANDARD  EmbodiedScore's shared body: the AgentCanvas workspace's bare VLN-CE R2R
          rig (``coding-agent/configs/habitat/std_task.yaml``: 0.25 m / 15°, 1.5 m /
          0.1 m agent, sliding on, RGB 512² + depth 256² at hfov 90 and 1.25 m,
          scene navmesh file) plus a 30° tilt clamped to ±60° so every line has
          LOOK_UP / LOOK_DOWN. Decided 2026-09-07.
VLNCE     VLN-CE ``habitat_extensions/config/vlnce_task.yaml`` (R2R-CE; IVLN-CE
          reuses it): as STANDARD but RGB 224² and no tilt.
RXR_CE    VLN-CE ``rxr_vlnce_english_task.yaml``: the LoCoBot rig on habitat-lab
          0.1.7 — 0.25 m / 30°, tilt 30° unlimited, 0.88 m / 0.18 m, sliding off,
          RGB-D 640x480 hfov 79 at 0.88 m, scene navmesh file (0.1.7 never
          recomputes).
LOCOBOT   habitat-lab ``benchmark/nav/objectnav/objectnav_hm3d.yaml`` (== mp3d):
          RXR_CE's rig with the navmesh recomputed for the 0.18 / 0.88 agent
          (habitat-lab 0.2.4 behaviour).
STRETCH   goat-bench ``goat_stretch_hm3d.yaml`` == OVON ``objectnav_stretch_hm3d.yaml``:
          0.25 m / 30°, tilt 30° unlimited, 1.41 m / 0.17 m, sliding off, RGB
          360x640 (portrait) hfov 42 at 1.31 m, no depth sensor, navmesh
          recomputed with climb 0.1 / cell 0.05 (every released experiment config
          of both repos).
EXPLORE_EQA  explore-eqa ``cfg/vlm_exp.yaml`` (HM-EQA; MemoryEQA's MT-HM3D reuses it):
          RGB-D 640x480 hfov 120 at 1.5 m, camera pitched -30° at start, the
          whole agent pitches (``pitch_moves_camera``), free-pose teleport
          locomotion; tilt 30° clamped to ±60°.
EXPRESS   EXPRESS-Bench ``fine_eqa.yaml``: RGB-D 512² hfov 90 at 1.5 m, camera
          level, free-pose teleport locomotion, no tilt.
LANGMAP   LangMap (bo-miao/LangMap ``vlnce_task_planavid_langmap.yaml`` + ``navmesh_utils.py``
          "stretch"): STANDARD's rig and actions, navmesh REBUILT for the Stretch
          (radius 0.17, height 1.41, climb 0.10, cell 0.05) — the mesh its episodes
          were generated on. LANGMAP_UPSTREAM is the evaluator's own rig: 0.25 m /
          30°, no tilt, habitat default agent 0.1 / 1.5, sliding on, RGB 640x480
          hfov 120 at 1.25 m, no depth, same rebuilt navmesh. Decided 2026-10-15.

Isaac engine (VLNverse; the agent is a camera on an occupancy grid — no
navmesh, no agent cylinder, no tilt: the render worker draws yaw-only poses):

VLNVERSE_STANDARD  the STANDARD numbers Isaac can express: 0.25 m / 15°, camera
          1.25 m above the floor, RGB 512² + depth 256² at hfov 90 (depth is the
          RGB camera's frame resampled at pixel centres), collision by occupancy
          snapping with a 0.1 m threshold in place of the navmesh. Decided 2026-09-08.
VLNVERSE  the zero-shot line's kinematic agent (billzhao1030/vlnverse_emr_zero_shot,
          carried through the AgentCanvas ``env_vlnverse`` nodeset): 0.25 m / 15°,
          camera 1.2 m, RGB-D 1024² at hfov 90 (focal 10 mm on a 20 mm aperture),
          occupancy collision 0.1 m (the line ships it switched off; the nodeset
          switched it on). The original evaluator moved a Unitree H1 in
          InternUtopia "flash" mode with a 256² camera — not reproduced.

LIBERO engine (a Franka Panda under robosuite's OSC_POSE controller — the
benchmark's physics, not a body knob; a body fixes the camera rig and the
settle ticks after a reset):

LIBERO_STANDARD  agentview 256² + wrist 256², frames upright, 10 settle ticks.
          Decided 2026-09-10 (256² is what the AgentCanvas env_libero nodeset served).
LIBERO    LIBERO's own evaluator rig (``libero/configs/data/default.yaml``:
          img_h / img_w 128) as OpenVLA's ``run_libero_eval.py`` runs it:
          agentview 128² + wrist 128², upright, 10 settle ticks (its num_steps_wait).

RoboTwin engine (a dual-arm tabletop robot under SAPIEN 3 with planned motion
— the benchmark's physics, not a body knob; a body fixes the embodiment, the
camera rig, the settle steps and the domain randomisation. RoboTwin's own
settling is 2500 physics steps inside its stability check, so ``settle_ticks``
adds nothing by default):

CLEAN / RANDOMIZED  the ``domain_randomization`` blocks of
          ``env_cfg/task_config/demo_clean.yml`` and ``demo_randomized.yml``,
          verbatim.
ROBOTWIN / ROBOTWIN_RANDOMIZED  RoboTwin's own evaluation rig: the
          ``aloha-agilex`` embodiment with head and wrist cameras of type D435
          (320x240, fovy 37 — ``env_cfg/task_config/_camera_config.yml``), which
          is what both shipped task configs name, under their respective
          randomisation.
ROBOTWIN_STANDARD / ROBOTWIN_STANDARD_RANDOMIZED  the same rig at RoboTwin's own
          larger camera type, ``Large_D435`` (640x480, the same fovy 37), so the
          macro protocol's agent reads the scene at the resolution the rest of
          EmbodiedScore's standard bodies serve. Decided 2026-09-10.

RoboCasa engine (a Franka Panda on an Omron mobile base under robosuite's
HYBRID_MOBILE_BASE controller — the benchmark's physics, not a body knob):

ROBOCASA_STANDARD  robot0_agentview_center 256² + wrist 256², frames upright,
          10 settle ticks. Decided 2026-09-10, matching LIBERO_STANDARD.
ROBOCASA  RoboCasa's own rig (``robocasa/utils/env_utils.py`` ``create_env``:
          robot0_agentview_left + robot0_agentview_right + robot0_eye_in_hand at
          128²) — the left view as obs["rgb"], the right one as obs["aux"].

CALVIN engine (a Franka Panda on the play table under pybullet IK + joint
position control at 30 Hz — the benchmark's physics, not a body knob; a body
fixes the two cameras' raster size and the settle ticks, which CALVIN's
evaluator has none of):

CALVIN_STANDARD  static 256² + gripper 256², 0 settle ticks. Decided
          2026-09-10, matching LIBERO_STANDARD and ROBOCASA_STANDARD.
CALVIN    CALVIN's own rig (``calvin_env/conf/cameras/cameras/{static,gripper}.yaml``,
          identical in every released dataset's ``merged_config.yaml``): the
          static camera 200x200 at fov 10, the gripper camera 84x84 at fov 75.
          Both cameras' geometry is the release's either way — only the raster
          size differs (both have ``aspect: 1``).

BEHAVIOR engine (a Galaxea R1 Pro — holonomic base, 4-joint torso, two 7-DOF
arms with parallel grippers — under the controllers the 2025 BEHAVIOR
Challenge fixes, at 30 Hz control / 120 Hz physics: the benchmark's physics,
not a body knob. A body fixes the camera rig; all three cameras are the
robot's own, ``ROBOT_CAMERA_NAMES["R1Pro"]``):

BEHAVIOR  the challenge's default evaluation rig, its ``RGBLowResWrapper``
          (``omnigibson/learning/wrappers/rgb_low_res_wrapper.py`` L16-29): the
          ZED head camera and both RealSense wrist cameras at 224², RGB only,
          the head camera's horizontal aperture forced to 40 mm to match the
          data collection. Both challenge tracks use it by default.
BEHAVIOR_STANDARD  the same three cameras at 256², the resolution the rest of
          EmbodiedScore's standard manipulation bodies serve (LIBERO_STANDARD,
          ROBOCASA_STANDARD), aperture unchanged. Decided 2026-09-10.
"""

from __future__ import annotations

from ..env import (BehaviorBody, BehaviorCameraSpec, Body, CalvinBody, CalvinCameraSpec, CameraSpec, IsaacBody,
                   IsaacCameraSpec, LiberoBody, LiberoCameraSpec, NavMesh, RobocasaBody, RobocasaCameraSpec, RobotwinBody,
                   RobotwinRandomization)
from ..env.sim.robocasa.body import LEFT_CAMERA, RIGHT_CAMERA
from ..env.sim.robotwin import D435, LARGE_D435

STANDARD = Body(forward_step_m=0.25, turn_deg=15.0, tilt_deg=30.0, tilt_limit_deg=60.0,
                agent_height_m=1.5, agent_radius_m=0.1, allow_sliding=True,
                rgb=CameraSpec(512, 512, 90.0, (0.0, 1.25, 0.0)), depth=CameraSpec(256, 256, 90.0, (0.0, 1.25, 0.0)),
                navmesh=NavMesh.file())

VLNCE = Body(forward_step_m=0.25, turn_deg=15.0, agent_height_m=1.5, agent_radius_m=0.1, allow_sliding=True,
             rgb=CameraSpec(224, 224, 90.0, (0.0, 1.25, 0.0)), depth=CameraSpec(256, 256, 90.0, (0.0, 1.25, 0.0)),
             navmesh=NavMesh.file())

RXR_CE = Body(forward_step_m=0.25, turn_deg=30.0, tilt_deg=30.0, tilt_limit_deg=None,
              agent_height_m=0.88, agent_radius_m=0.18, allow_sliding=False,
              rgb=CameraSpec(640, 480, 79.0, (0.0, 0.88, 0.0)), depth=CameraSpec(640, 480, 79.0, (0.0, 0.88, 0.0)),
              navmesh=NavMesh.file())

LOCOBOT = Body(forward_step_m=0.25, turn_deg=30.0, tilt_deg=30.0, tilt_limit_deg=None,
               agent_height_m=0.88, agent_radius_m=0.18, allow_sliding=False,
               rgb=CameraSpec(640, 480, 79.0, (0.0, 0.88, 0.0)), depth=CameraSpec(640, 480, 79.0, (0.0, 0.88, 0.0)),
               navmesh=NavMesh.recompute(agent_radius=0.18, agent_height=0.88))

STRETCH = Body(forward_step_m=0.25, turn_deg=30.0, tilt_deg=30.0, tilt_limit_deg=None,
               agent_height_m=1.41, agent_radius_m=0.17, allow_sliding=False,
               rgb=CameraSpec(360, 640, 42.0, (0.0, 1.31, 0.0)), depth=None,
               navmesh=NavMesh.recompute(agent_radius=0.17, agent_height=1.41, agent_max_climb=0.1, cell_height=0.05))

EXPLORE_EQA = Body(forward_step_m=0.25, turn_deg=30.0, tilt_deg=30.0, tilt_limit_deg=60.0, camera_pitch_deg=-30.0,
                   pitch_moves_camera=True, locomotion="teleport",
                   agent_height_m=1.5, agent_radius_m=0.1, allow_sliding=True,
                   rgb=CameraSpec(640, 480, 120.0, (0.0, 1.5, 0.0)), depth=CameraSpec(640, 480, 120.0, (0.0, 1.5, 0.0)),
                   navmesh=NavMesh.file())

EXPRESS = Body(forward_step_m=0.25, turn_deg=30.0, tilt_deg=None, camera_pitch_deg=0.0, locomotion="teleport",
               agent_height_m=1.5, agent_radius_m=0.1, allow_sliding=True,
               rgb=CameraSpec(512, 512, 90.0, (0.0, 1.5, 0.0)), depth=CameraSpec(512, 512, 90.0, (0.0, 1.5, 0.0)),
               navmesh=NavMesh.file())

LANGMAP = Body(forward_step_m=0.25, turn_deg=15.0, tilt_deg=30.0, tilt_limit_deg=60.0,
               agent_height_m=1.5, agent_radius_m=0.1, allow_sliding=True,
               rgb=CameraSpec(512, 512, 90.0, (0.0, 1.25, 0.0)), depth=CameraSpec(256, 256, 90.0, (0.0, 1.25, 0.0)),
               navmesh=NavMesh.recompute(agent_radius=0.17, agent_height=1.41, agent_max_climb=0.1, cell_height=0.05))

LANGMAP_UPSTREAM = Body(forward_step_m=0.25, turn_deg=30.0, tilt_deg=None,
                        agent_height_m=1.5, agent_radius_m=0.1, allow_sliding=True,
                        rgb=CameraSpec(640, 480, 120.0, (0.0, 1.25, 0.0)), depth=None,
                        navmesh=NavMesh.recompute(agent_radius=0.17, agent_height=1.41, agent_max_climb=0.1, cell_height=0.05))

VLNVERSE_STANDARD = IsaacBody(forward_step_m=0.25, turn_deg=15.0, camera_height_m=1.25,
                              collision="occupancy", collision_threshold_m=0.1,
                              rgb=IsaacCameraSpec(512, 512, 90.0), depth=IsaacCameraSpec(256, 256, 90.0))

VLNVERSE = IsaacBody(forward_step_m=0.25, turn_deg=15.0, camera_height_m=1.2,
                     collision="occupancy", collision_threshold_m=0.1,
                     rgb=IsaacCameraSpec(1024, 1024, 90.0), depth=IsaacCameraSpec(1024, 1024, 90.0))

LIBERO_STANDARD = LiberoBody(rgb=LiberoCameraSpec(256, 256), wrist=LiberoCameraSpec(256, 256), settle_ticks=10)

LIBERO = LiberoBody(rgb=LiberoCameraSpec(128, 128), wrist=LiberoCameraSpec(128, 128), settle_ticks=10)

# env_cfg/task_config/demo_clean.yml `domain_randomization`, verbatim.
CLEAN = RobotwinRandomization(random_background=False, cluttered_table=False, clean_background_rate=1.0,
                              random_head_camera_dis=0.0, random_table_height=0.0, random_light=False,
                              crazy_random_light_rate=0.0)

# env_cfg/task_config/demo_randomized.yml `domain_randomization`, verbatim.
RANDOMIZED = RobotwinRandomization(random_background=True, cluttered_table=True, clean_background_rate=0.02,
                                   random_head_camera_dis=0.0, random_table_height=0.03, random_light=True,
                                   crazy_random_light_rate=0.02)

ROBOTWIN = RobotwinBody(embodiment="aloha-agilex", head=D435, wrist=D435, randomization=CLEAN)

ROBOTWIN_RANDOMIZED = RobotwinBody(embodiment="aloha-agilex", head=D435, wrist=D435, randomization=RANDOMIZED)

ROBOTWIN_STANDARD = RobotwinBody(embodiment="aloha-agilex", head=LARGE_D435, wrist=LARGE_D435, randomization=CLEAN)

ROBOTWIN_STANDARD_RANDOMIZED = RobotwinBody(embodiment="aloha-agilex", head=LARGE_D435, wrist=LARGE_D435,
                                            randomization=RANDOMIZED)

ROBOCASA_STANDARD = RobocasaBody(rgb=RobocasaCameraSpec(256, 256), wrist=RobocasaCameraSpec(256, 256),
                                 aux=None, settle_ticks=10)

ROBOCASA = RobocasaBody(rgb=RobocasaCameraSpec(128, 128), wrist=RobocasaCameraSpec(128, 128),
                        aux=RobocasaCameraSpec(128, 128), rgb_camera=LEFT_CAMERA, aux_camera=RIGHT_CAMERA,
                        settle_ticks=10)

CALVIN_STANDARD = CalvinBody(rgb=CalvinCameraSpec(256, 256), wrist=CalvinCameraSpec(256, 256), settle_ticks=0)

CALVIN = CalvinBody(rgb=CalvinCameraSpec(200, 200), wrist=CalvinCameraSpec(84, 84), settle_ticks=0)

BEHAVIOR = BehaviorBody(rgb=BehaviorCameraSpec(224, 224), wrist=BehaviorCameraSpec(224, 224))

BEHAVIOR_STANDARD = BehaviorBody(rgb=BehaviorCameraSpec(256, 256), wrist=BehaviorCameraSpec(256, 256))

__all__ = ["STANDARD", "VLNCE", "RXR_CE", "LOCOBOT", "STRETCH", "EXPLORE_EQA", "EXPRESS",
           "VLNVERSE_STANDARD", "VLNVERSE", "LIBERO_STANDARD", "LIBERO",
           "CLEAN", "RANDOMIZED", "ROBOTWIN", "ROBOTWIN_RANDOMIZED", "ROBOTWIN_STANDARD",
           "ROBOTWIN_STANDARD_RANDOMIZED", "ROBOCASA_STANDARD", "ROBOCASA",
           "CALVIN_STANDARD", "CALVIN", "BEHAVIOR", "BEHAVIOR_STANDARD"]
