"""The Isaac Sim 5.1 engine (layer L0). The only place in the package that
talks to Isaac, and it does so out of process: ``worker.py`` runs inside
Isaac's python (standalone — never imported here), ``backend.py`` drives it
over msgpack frames, ``freemap.py`` is the agent's motion (kinematic teleport
+ occupancy snapping), ``world.py`` puts the three behind one object."""

from .body import IsaacBody, IsaacCameraSpec
from .scene import IsaacSceneRef
from .world import IsaacSettings, IsaacWorld

__all__ = ["IsaacBody", "IsaacCameraSpec", "IsaacSceneRef", "IsaacSettings", "IsaacWorld"]
