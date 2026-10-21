"""Embodiment parameters for the Isaac engine — pure data (layer L0, no Isaac import).

An :class:`IsaacBody` says what the agent is: step / turn amounts, where the
camera sits, how collisions are decided, and the camera rig. Every number the
world uses comes from here; the presets live in ``presets/bodies.py``.

The agent is a camera. There is no navmesh, no agent cylinder and no tilt:
the render worker draws yaw-only poses, and the floor is the scene's
occupancy grid (``freemap.npy``) rather than a mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

COLLISION_MODES = ("occupancy", "none")


@dataclass(frozen=True)
class IsaacCameraSpec:
    width: int
    height: int
    hfov_deg: float = 90.0     # the worker derives Isaac's aperture from focal length and this


@dataclass(frozen=True)
class IsaacBody:
    forward_step_m: float = 0.25
    turn_deg: float = 15.0
    camera_height_m: float = 1.2          # the agent is a camera: its z is pinned here after every placement
    collision: str = "occupancy"          # "occupancy": the intended point is snapped to the nearest reachable
                                          #   freemap cell; a snap farther than collision_threshold_m counts as a
                                          #   collision. "none": pure arithmetic, walls are ignored.
    collision_threshold_m: float = 0.1
    rgb: IsaacCameraSpec = IsaacCameraSpec(1024, 1024, 90.0)
    depth: IsaacCameraSpec | None = IsaacCameraSpec(1024, 1024, 90.0)
                                          # None = no depth in obs. The worker renders depth from the RGB camera,
                                          # so depth.hfov_deg must equal rgb.hfov_deg; a smaller resolution is
                                          # served by resampling that frame at pixel centres (nearest neighbour).

    def __post_init__(self) -> None:
        if self.collision not in COLLISION_MODES:
            raise ValueError(f"collision must be one of {COLLISION_MODES}, got {self.collision!r}")
        if self.depth is not None and self.depth.hfov_deg != self.rgb.hfov_deg:
            raise ValueError("IsaacBody: depth is rendered from the RGB camera, so depth.hfov_deg must equal rgb.hfov_deg")

    @property
    def has_tilt(self) -> bool:
        return False
