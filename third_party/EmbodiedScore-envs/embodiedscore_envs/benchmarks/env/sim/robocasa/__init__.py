"""RoboCasa engine facade (layer L0): ``RobocasaBody`` · ``RobocasaSceneRef``
are pure data; ``RobocasaWorld`` is the only place that imports robocasa /
robosuite / mujoco, and does so lazily — ``import embodiedscore_envs`` works
without them."""

from __future__ import annotations

from .body import RobocasaBody, RobocasaCameraSpec
from .scene import RobocasaSceneRef
from .world import RobocasaWorld

__all__ = ["RobocasaBody", "RobocasaCameraSpec", "RobocasaSceneRef", "RobocasaWorld"]
