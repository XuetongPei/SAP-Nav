"""BEHAVIOR engine facade (layer L0): ``BehaviorBody`` · ``BehaviorSceneRef``
are pure data; ``BehaviorWorld`` is the only place that imports omnigibson /
isaacsim, and does so lazily — ``import embodiedscore_envs`` works without
them."""

from __future__ import annotations

from .body import ARMS, CAMERA_PRIMS, BehaviorBody, BehaviorCameraSpec
from .scene import SCENE_MODELS, BehaviorSceneRef
from .world import BehaviorWorld

__all__ = ["ARMS", "CAMERA_PRIMS", "SCENE_MODELS", "BehaviorBody", "BehaviorCameraSpec", "BehaviorSceneRef",
           "BehaviorWorld"]
