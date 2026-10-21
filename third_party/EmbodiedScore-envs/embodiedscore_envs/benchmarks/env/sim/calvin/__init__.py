"""CALVIN engine facade (layer L0): ``CalvinBody`` · ``CalvinSceneRef`` are
pure data; ``CalvinWorld`` is the only place that imports calvin_env /
pybullet, and does so lazily — ``import embodiedscore_envs`` works without
them."""

from __future__ import annotations

from .body import CalvinBody, CalvinCameraSpec
from .scene import CalvinSceneRef
from .world import CalvinWorld, task_oracle

__all__ = ["CalvinBody", "CalvinCameraSpec", "CalvinSceneRef", "CalvinWorld", "task_oracle"]
