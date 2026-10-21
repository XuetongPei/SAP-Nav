"""RobocasaSceneRef — one RoboCasa evaluation scenario (pure data, layer L0).

RoboCasa has no scene files: a kitchen is *generated* from a layout id (the
floor plan) and a style id (the textures and fixture styles), and the objects
and their placements are drawn from an object split by the environment's own
RNG under a seed. A scenario is therefore the four numbers below, and the
episode is fully determined by them:

    task              the robocasa environment name, e.g. ``CloseDrawer``
    layout_id         the floor plan (``robocasa/utils/scene_registry.py``)
    style_id          the kitchen style
    seed              the env RNG seed — objects, placements, robot spawn
    obj_instance_split which object instances may be drawn ("target" /
                      "pretrain" on RoboCasa365, "A" / "B" on RoboCasa v0.2)

``release`` records which RoboCasa the scenario belongs to; the world only
uses it for the error message when the installed package is the other one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobocasaSceneRef:
    task: str                              # the robocasa env name (robosuite REGISTERED_ENVS key)
    layout_id: int
    style_id: int
    seed: int
    obj_instance_split: str | None = None
    release: str = "robocasa365"           # "robocasa365" (v1.0.x) | "robocasa" (v0.2)

    @property
    def scene_file(self) -> str:
        """What identifies the scene, as ``scene_file`` does on the other engines."""
        return f"{self.task}/layout{self.layout_id}/style{self.style_id}"

    @property
    def model_key(self) -> tuple[str, int, int, str | None]:
        """What a rebuilt MuJoCo model depends on — the seed does not (it is
        applied at reset, and every RoboCasa reset resamples the scene anyway)."""
        return (self.task, self.layout_id, self.style_id, self.obj_instance_split)
