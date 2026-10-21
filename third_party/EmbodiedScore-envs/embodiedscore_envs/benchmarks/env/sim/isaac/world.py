"""IsaacWorld — one object in front of the render worker and the freemap
kinematics (layer L0). It exposes the primitives the environment body needs
and nothing about episodes, budgets or metrics.

What the simulator is here: Isaac Sim renders; it does not move the agent.
The agent is a camera whose pose is plain state (``freemap.py``): a move is
computed arithmetically, snapped to the scene's occupancy grid, and the new
pose is sent to the worker to render from. That is the zero-shot line's
protocol (billzhao1030/vlnverse_emr_zero_shot), not the InternUtopia physics
loop of the original evaluator — see README "Fidelity".

Depth comes from the RGB camera (one render per view). When the body asks for
a smaller depth frame than its RGB frame — the STANDARD body's 256² next to
RGB 512² — the rendered depth is resampled at the pixel centres of the
smaller grid (nearest neighbour, :func:`_resample_nearest`): each output
pixel is the depth of one ray, never an average across an edge.

Conventions (Isaac world frame): positions in metres, z up; ``heading`` is
the yaw about z in radians, counter-clockwise positive, 0 = +x; rotations are
``(w, x, y, z)`` quaternions. Positions carry the camera height as z.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .body import IsaacBody
from .freemap import FreemapKinematics
from .quat import euler_angles_to_quat
from .scene import IsaacSceneRef

# Action ids the body sends to move() — a mirror of schema.Act without importing L1.
FORWARD, LEFT, RIGHT = 1, 2, 3

_DEFAULT_ISAAC_PYTHON = "~/isaacsim5.1/python.sh"


def _resample_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """``frame[H, W, ...]`` sampled at the pixel centres of an ``height x width``
    grid — the pixel a same-hfov camera of that resolution would sample."""
    h, w = frame.shape[:2]
    if (h, w) == (height, width):
        return frame
    rows = np.minimum(((np.arange(height) + 0.5) * h / height).astype(np.intp), h - 1)
    cols = np.minimum(((np.arange(width) + 0.5) * w / width).astype(np.intp), w - 1)
    return frame[rows[:, None], cols[None, :]]


@dataclass(frozen=True)
class IsaacSettings:
    """Renderer knobs (not embodiment): what the zero-shot line's
    ``_build_sim_cfg`` fixed. ``from_env()`` reads ``EMBODIEDSCORE_ISAAC_HEADLESS`` /
    ``EMBODIEDSCORE_ISAAC_RENDERER`` overrides."""
    headless: bool = True
    renderer: str = "RaytracedLighting"
    anti_aliasing: int = 0
    denoiser: bool = False
    focal_length_mm: float = 10.0     # aperture follows from the body's hfov: 2 * f * tan(hfov / 2)
    warmup_steps: int = 10            # render steps per view before the frame is read

    @classmethod
    def from_env(cls) -> "IsaacSettings":
        return cls(headless=os.environ.get("EMBODIEDSCORE_ISAAC_HEADLESS", "1") != "0",
                   renderer=os.environ.get("EMBODIEDSCORE_ISAAC_RENDERER", "RaytracedLighting"))


class IsaacWorld:
    def __init__(self, body: IsaacBody, gpu_id: int = 0, settings: IsaacSettings | None = None) -> None:
        self.body = body
        self._gpu_id = int(gpu_id)
        self._settings = settings or IsaacSettings.from_env()
        self._backend: Any = None            # created on the first load_scene (Isaac boot is minute-scale)
        self._kin = FreemapKinematics(collision_threshold=body.collision_threshold_m,
                                      use_occupancy_collision=(body.collision == "occupancy"))
        self._scene: IsaacSceneRef | None = None
        self._view: Any = None               # cached front view; dropped whenever the pose changes

    # ---- lifecycle ---------------------------------------------------------------
    def _sim_cfg(self) -> dict[str, Any]:
        s, cam = self._settings, self.body.rgb
        focal = float(s.focal_length_mm)
        aperture = 2.0 * focal * math.tan(math.radians(cam.hfov_deg) / 2.0)
        return {
            "headless": s.headless, "gpu_id": self._gpu_id, "isaac_multi_gpu": False,
            "renderer": s.renderer, "anti_aliasing": int(s.anti_aliasing), "denoiser": bool(s.denoiser),
            "camera_resolution": (int(cam.width), int(cam.height)),
            "focal_length": focal, "aperture": aperture,
        }

    def _make_backend(self) -> Any:
        """Socket client when ``$EMBODIEDSCORE_ISAAC_SOCKET`` names a pre-started
        worker (``worker.py --listen``); else spawn the worker under the Isaac
        launcher ``$EMBODIEDSCORE_ISAAC_PYTHON``."""
        from .backend import SocketIsaacSimBackend, SubprocessIsaacSimBackend
        socket_path = os.environ.get("EMBODIEDSCORE_ISAAC_SOCKET")
        if socket_path:
            return SocketIsaacSimBackend(socket_path)
        isaac_python = os.path.expanduser(os.environ.get("EMBODIEDSCORE_ISAAC_PYTHON", _DEFAULT_ISAAC_PYTHON))
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")
        return SubprocessIsaacSimBackend(isaac_python, worker)

    def _ensure_backend(self) -> Any:
        if self._backend is None:
            backend = self._make_backend()
            backend.start(self._sim_cfg())
            self._backend = backend
        return self._backend

    def load_scene(self, scene: IsaacSceneRef) -> None:
        """Load ``scene`` into the worker and its freemap into the kinematics;
        a no-op when it is already the current scene."""
        if self._scene is not None and self._scene.scan == scene.scan:
            return
        backend = self._ensure_backend()
        backend.load_scene(scene.scan, scene.scene_file)
        if self.body.collision == "occupancy":
            self._kin.load_freemap(scene.freemap_file)
        self._scene = scene
        self._view = None

    def close(self) -> None:
        if self._backend is not None:
            try:
                self._backend.shutdown()
            finally:
                self._backend = None
        self._scene = None
        self._view = None

    def __enter__(self) -> "IsaacWorld":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def scene(self) -> IsaacSceneRef | None:
        return self._scene

    @property
    def backend(self) -> Any:
        """The worker client — for tools that need extra renders."""
        return self._ensure_backend()

    @property
    def kinematics(self) -> FreemapKinematics:
        return self._kin

    # ---- pose ----------------------------------------------------------------------
    def place(self, position: Any, rotation_wxyz: Any) -> None:
        """Seat the agent: xy as given, z pinned to the body's camera height."""
        pos = np.asarray(position, dtype=np.float64).copy()
        pos[2] = self.body.camera_height_m
        self._kin.place(pos, np.asarray(rotation_wxyz, dtype=np.float64))
        self._view = None

    def teleport(self, position: Any, yaw_rad: float) -> None:
        pos = np.asarray(position, dtype=np.float64).copy()
        pos[2] = self.body.camera_height_m
        self._kin.place(pos, euler_angles_to_quat(np.array([0.0, 0.0, float(yaw_rad)])))
        self._view = None

    def move(self, action: int) -> bool:
        """One discrete action; returns whether it collided."""
        b = self.body
        turn = math.radians(b.turn_deg)
        if action == FORWARD:
            angle, distance = 0.0, b.forward_step_m
        elif action == LEFT:
            angle, distance = turn, 0.0
        elif action == RIGHT:
            angle, distance = -turn, 0.0
        else:
            raise ValueError(f"move(): not a movement action: {action}")
        collided, _ = self._kin.move(angle, 0.0, distance)
        self._view = None
        return bool(collided)

    def move_polar(self, angle_rad: float, distance_m: float, elevation_deg: float = 0.0) -> tuple[bool, float]:
        """VLNverse's macro action: rotate by ``angle_rad``, then advance
        ``distance_m`` at ``elevation_deg`` (the upstream rad / deg asymmetry
        is kept). Returns ``(collided, snap_deviation_m)``."""
        collided, deviation = self._kin.move(float(angle_rad), float(elevation_deg), float(distance_m))
        self._view = None
        return bool(collided), float(deviation)

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._kin.agent_position.copy(), self._kin.agent_rotation_quat.copy()

    def heading(self) -> float:
        return float(self._kin.agent_heading)

    def pitch(self) -> float:
        return 0.0        # the worker renders yaw-only poses

    # ---- rendering -----------------------------------------------------------------
    def observe(self) -> dict[str, np.ndarray]:
        """The front view at the current pose: ``{"rgb": HxWx3 uint8[, "depth":
        HxWx1 float32 metres]}``. Cached until the pose changes, so repeated
        reads are free and a read after STOP serves the terminal frame."""
        if self._view is None:
            self._view = self.render_views(1)[0]
        return dict(self._view)

    def render_views(self, n_views: int) -> list[dict[str, np.ndarray]]:
        """``n_views`` views from the current pose, view *i* facing
        ``heading + i * 360 / n`` degrees (counter-clockwise), i.e. a panorama
        whose view 0 is the front view. Rendering does not move the agent."""
        if self._scene is None:
            raise RuntimeError("IsaacWorld: load_scene() first")
        backend = self._ensure_backend()
        n = max(1, int(n_views))
        rendered = backend.capture_panoramic(self._kin.agent_position, self._kin.agent_heading,
                                             num_views=n, warmup_steps=self._settings.warmup_steps)
        out = []
        d = self.body.depth
        for v in rendered:
            obs = {"rgb": np.asarray(v.rgb, dtype=np.uint8)}
            if d is not None:
                obs["depth"] = _resample_nearest(np.asarray(v.depth, dtype=np.float32), d.height, d.width)[..., None]
            out.append(obs)
        if out:
            self._view = out[0]
        return out

    # ---- geometry ------------------------------------------------------------------
    def distance(self, start: Any, targets: Sequence[Any] | np.ndarray) -> float:
        """Euclidean distance in the ground plane from ``start`` to the nearest
        of ``targets`` — VLNverse's ``NE``. There is no navmesh and no
        geodesic in this benchmark."""
        s = np.asarray(start, dtype=np.float64)[:2]
        t = np.asarray(targets, dtype=np.float64).reshape(-1, 3)[:, :2]
        return float(np.min(np.linalg.norm(t - s, axis=1)))

    def snap(self, point: Any) -> np.ndarray | None:
        """The nearest reachable freemap cell to ``point`` (xy; z kept), or
        None when nothing reachable lies within the search radius."""
        p = np.asarray(point, dtype=np.float64)
        if self._kin.occupancy is None:
            return p.copy()
        hit = self._kin.find_nearest_reachable(float(p[0]), float(p[1]))
        if hit is None:
            return None
        return np.array([hit[0], hit[1], p[2] if p.shape[0] > 2 else self.body.camera_height_m])

    def is_navigable(self, point: Any) -> bool:
        p = np.asarray(point, dtype=np.float64)
        if self._kin.occupancy is None:
            return True
        hit = self._kin.find_nearest_reachable(float(p[0]), float(p[1]))
        return hit is not None and hit[2] == 0
