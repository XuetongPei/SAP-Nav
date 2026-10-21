"""Layer L1 — what every benchmark is built on: the environment bodies (one
per engine: habitat, isaac, libero, robotwin, robocasa, calvin, behavior), the episode / goal vocabulary, the metric and observation
wrappers, and the simulator facades underneath (``env.sim``).

``SimWorld`` / ``Follower`` / ``FollowerError`` are the habitat_sim importers
and resolve lazily (``sim.__getattr__``); everything else imports without a
simulator installed.
"""

from __future__ import annotations

from .habitat_env import HabitatEnv, HabitatPoseEnv
from .isaac_env import IsaacEnv, IsaacPolarEnv
from .libero_env import LiberoEnv, LiberoPoseEnv
from .robotwin_env import RobotwinEnv, RobotwinPoseEnv
from .robocasa_env import RobocasaEnv, RobocasaPoseEnv
from .behavior_env import BehaviorEnv, BehaviorPoseEnv
from .calvin_env import CalvinEnv, CalvinPoseEnv
from .metrics import (BEHAVIOR_KEYS, CHAIN_KEYS, MANIP_KEYS, NAV_KEYS, OBJECTNAV_KEYS, VLN_KEYS, VLNVERSE_EVALUATOR_NAMES,
                      VLNVERSE_KEYS, BehaviorMetrics, ChainMetrics, ManipMetrics, NavMetrics, SequenceNavMetrics,
                      VLNVerseMetrics)
from .schema import (ENGINES, Act, Benchmark, DepthSpec, Episode, Goal, GoalSequence, ImageGoal, ManipGoal, ObjectGoal,
                     ObjectInstance, PointGoal, Question, TextGoal, action_prefix, data_root, scene_root,
                     targets_of, to_dict)
from .sim import (BehaviorBody, BehaviorCameraSpec, BehaviorSceneRef, BehaviorWorld, Body, CalvinBody, CalvinCameraSpec,
                  CalvinSceneRef, CalvinWorld, CameraSpec, IsaacBody, IsaacCameraSpec, IsaacSceneRef, IsaacSettings,
                  IsaacWorld, LiberoBody, LiberoCameraSpec, LiberoSceneRef, LiberoWorld, NavMesh, RobocasaBody,
                  RobocasaCameraSpec, RobocasaSceneRef, RobocasaWorld, RobotwinBody, RobotwinCameraSpec,
                  RobotwinRandomization, RobotwinSceneRef, RobotwinWorld, SceneRef)
from .wrappers import DepthClip, DynamicTimeLimit

_SIM_LAZY = ("SimWorld", "Follower", "FollowerError")


def __getattr__(name: str):
    if name in _SIM_LAZY:
        from . import sim
        return getattr(sim, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HabitatEnv", "HabitatPoseEnv", "IsaacEnv", "IsaacPolarEnv", "LiberoEnv", "LiberoPoseEnv",
    "RobotwinEnv", "RobotwinPoseEnv",
    "RobocasaEnv", "RobocasaPoseEnv", "CalvinEnv", "CalvinPoseEnv", "BehaviorEnv", "BehaviorPoseEnv",
    "NavMetrics", "SequenceNavMetrics", "VLNVerseMetrics", "ManipMetrics", "ChainMetrics", "BehaviorMetrics",
    "CHAIN_KEYS", "BEHAVIOR_KEYS", "NAV_KEYS", "VLN_KEYS", "OBJECTNAV_KEYS",
    "VLNVERSE_KEYS", "VLNVERSE_EVALUATOR_NAMES", "MANIP_KEYS",
    "BehaviorMetrics", "BEHAVIOR_KEYS",
    "ENGINES", "Act", "Benchmark", "DepthSpec", "Episode", "Goal", "GoalSequence", "ImageGoal", "ManipGoal",
    "ObjectGoal", "ObjectInstance", "PointGoal", "Question", "TextGoal", "action_prefix", "data_root", "scene_root",
    "targets_of", "to_dict",
    "Body", "CameraSpec", "NavMesh", "SceneRef", "SimWorld", "Follower", "FollowerError",
    "IsaacBody", "IsaacCameraSpec", "IsaacSceneRef", "IsaacSettings", "IsaacWorld",
    "LiberoBody", "LiberoCameraSpec", "LiberoSceneRef", "LiberoWorld",
    "RobotwinBody", "RobotwinCameraSpec", "RobotwinRandomization", "RobotwinSceneRef", "RobotwinWorld",
    "RobocasaBody", "RobocasaCameraSpec", "RobocasaSceneRef", "RobocasaWorld",
    "CalvinBody", "CalvinCameraSpec", "CalvinSceneRef", "CalvinWorld",
    "BehaviorBody", "BehaviorCameraSpec", "BehaviorSceneRef", "BehaviorWorld",
    "DepthClip", "DynamicTimeLimit",
]
