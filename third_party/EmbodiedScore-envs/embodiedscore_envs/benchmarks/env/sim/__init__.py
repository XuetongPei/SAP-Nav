"""Layer L0 — the simulator facades, one per engine.

``habitat/``  habitat-sim 0.3.3 in process: Body · SceneRef · SimWorld (+ Follower).
``isaac/``    Isaac Sim 5.1 out of process: IsaacBody · IsaacSceneRef · IsaacWorld
              (render worker under Isaac's python, freemap kinematics on this side).
``libero/``   LIBERO (robosuite 1.4 + MuJoCo) in process: LiberoBody · LiberoSceneRef · LiberoWorld.
``robotwin/`` RoboTwin 2.0 (SAPIEN 3) in process: RobotwinBody · RobotwinSceneRef · RobotwinWorld.
``robocasa/`` RoboCasa / RoboCasa365 (robosuite 1.5 + MuJoCo) in process:
              RobocasaBody · RobocasaSceneRef · RobocasaWorld.
``calvin/``   CALVIN (pybullet) in process: CalvinBody · CalvinSceneRef · CalvinWorld.
``behavior/`` BEHAVIOR-1K (OmniGibson on Isaac Sim 4.5) in process, Isaac booted
              once: BehaviorBody · BehaviorSceneRef · BehaviorWorld.

The engines never import each other. The pure-data types of both are
importable without either simulator installed; ``SimWorld`` and its
companions (the habitat_sim importers) resolve lazily, so ``import
embodiedscore_envs`` works on a host that has only Isaac (or neither) — a
habitat line then fails at construction with the ImportError that names the
build to install.
"""

from __future__ import annotations

from .habitat import Body, CameraSpec, NavMesh, SceneRef
from .isaac import IsaacBody, IsaacCameraSpec, IsaacSceneRef, IsaacSettings, IsaacWorld
from .libero import LiberoBody, LiberoCameraSpec, LiberoSceneRef, LiberoWorld
from .robotwin import RobotwinBody, RobotwinCameraSpec, RobotwinRandomization, RobotwinSceneRef, RobotwinWorld
from .robocasa import RobocasaBody, RobocasaCameraSpec, RobocasaSceneRef, RobocasaWorld
from .calvin import CalvinBody, CalvinCameraSpec, CalvinSceneRef, CalvinWorld
from .behavior import BehaviorBody, BehaviorCameraSpec, BehaviorSceneRef, BehaviorWorld

_HABITAT_LAZY = ("SimWorld", "Follower", "FollowerError", "FORWARD", "LEFT", "RIGHT", "LOOK_UP", "LOOK_DOWN")


def __getattr__(name: str):
    if name in _HABITAT_LAZY:
        from .habitat import world
        return getattr(world, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Body", "CameraSpec", "NavMesh", "SceneRef", "SimWorld", "Follower", "FollowerError",
           "FORWARD", "LEFT", "RIGHT", "LOOK_UP", "LOOK_DOWN",
           "IsaacBody", "IsaacCameraSpec", "IsaacSceneRef", "IsaacSettings", "IsaacWorld",
           "LiberoBody", "LiberoCameraSpec", "LiberoSceneRef", "LiberoWorld",
           "RobotwinBody", "RobotwinCameraSpec", "RobotwinRandomization", "RobotwinSceneRef", "RobotwinWorld",
           "RobocasaBody", "RobocasaCameraSpec", "RobocasaSceneRef", "RobocasaWorld",
           "CalvinBody", "CalvinCameraSpec", "CalvinSceneRef", "CalvinWorld",
           "BehaviorBody", "BehaviorCameraSpec", "BehaviorSceneRef", "BehaviorWorld"]
