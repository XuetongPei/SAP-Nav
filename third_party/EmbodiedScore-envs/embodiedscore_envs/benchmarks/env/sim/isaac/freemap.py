"""The agent's motion: kinematic teleport + occupancy collision on the scene freemap.

Lineage: the zero-shot line's ``IsaacVLNBaseEnvironment`` (billzhao1030/vlnverse_emr_zero_shot,
``vlnverse/projects/internutopia_vln_extension/envs/isaac_base_env.py``, movement
subset), carried through the AgentCanvas ``env_vlnverse`` nodeset. The upstream
class fuses episode management, Isaac observation capture and pure-python
motion; only the motion + occupancy geometry lives here — episode seating is
``IsaacWorld.place``, rendering is ``backend.py``. The upstream oracle helpers
(``_analyze_surrounding_occupancy``, ``cand_dist_to_goal``,
``get_next_reference_waypoint``) have no consumer here and were dropped.

Deviations from upstream:

- torch / Isaac / episode selection removed. No sim handle anywhere: the
  agent pose is plain state, exactly as upstream computes it (movement is
  kinematic teleportation with occupancy snapping — Isaac only renders).
- ``reset_from_episode`` (goal / reference-path bookkeeping) replaced by
  :meth:`place` — distance to goal is the world's business, not the mover's.
- ``print`` diagnostics → ``logging`` (upstream spams stdout per move).
- EVAL collision semantics only: a collision is *recorded* and the episode
  continues (upstream's training/eval branching lives in its trainer, not in
  the move helpers — there is no behavior change here, just no flag).
- ``move()`` unifies the two upstream paths behind one call and always
  returns ``(collision, deviation_m)``; upstream's no-occupancy path returns
  a bare ``False``.
- Missing freemap with occupancy enabled raises ``RuntimeError`` instead of
  ``AttributeError`` deep in ``find_nearest_reachable``.

Unit conventions (upstream-faithful — mind the mismatch):
- ``angle`` is **radians** (yaw delta, +CCW);
- ``elevation`` is **degrees** (converted via ``np.radians`` before use);
- ``distance`` is meters.

Freemap convention (``<scene_root>/<scan>/freemap.npy``): row 0 stores the
world-x coordinate of every column, column 0 stores the world-y coordinate of
every row; interior cells are 0 = obstacle, 1 = reachable, 2 = out-of-bounds.
World→grid goes through nearest-coordinate lookup on those axis vectors;
grid→world reads the axis values back (so occupancy-snapped positions land on
cell-center coordinates).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from .quat import _quat_mul, euler_angles_to_quat, quat_to_euler_angles

log = logging.getLogger(__name__)


class FreemapKinematics:
    """Pure-python agent pose + movement on a freemap.

    Owns: position (np [x, y, z]), heading (rad), rotation quat (w, x, y, z)
    and the current scene's occupancy map. Nothing about episodes or goals.
    """

    def __init__(
        self,
        collision_threshold: float = 0.1,
        use_occupancy_collision: bool = True,
    ) -> None:
        # Agent state (upstream defaults, isaac_base_env.py:44-47)
        self.agent_position = np.array([0.0, 0.0, 1.5])
        self.agent_heading = 0.0
        self.agent_rotation_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]

        # Collision config (isaac_base_env.py:49-52)
        self.use_occupancy_collision = bool(use_occupancy_collision)
        self.collision_threshold = float(collision_threshold)
        self.collision_occurred = False  # any collision this episode

        # Scene occupancy (load_freemap)
        self.occupancy: Optional[np.ndarray] = None

    # ==================== pose wiring ====================

    def place(self, position: np.ndarray, rotation_wxyz: np.ndarray) -> None:
        """Seat the agent: position as given, heading = the quaternion's yaw
        (upstream ``reset()`` pose part: ``quat_to_euler_angles(q)[2]``)."""
        self.collision_occurred = False
        self.agent_position = np.asarray(position, dtype=np.float64).copy()
        self.agent_rotation_quat = self._normalize_quat(np.asarray(rotation_wxyz, dtype=np.float64))
        self.agent_heading = float(quat_to_euler_angles(self.agent_rotation_quat, degrees=False)[2])

    def load_freemap(self, path: str) -> None:
        """Load the scene occupancy grid (``freemap.npy``)."""
        self.occupancy = np.load(path)

    def set_pose(self, position: np.ndarray, heading: float) -> None:
        """Upstream ``_set_agent_pose`` — pure state update, no camera here."""
        self.agent_position = np.asarray(position, dtype=np.float64).copy()
        self.agent_heading = float(heading)

    # ==================== movement ====================

    def move(self, angle: float, elevation: float, distance: float) -> Tuple[bool, float]:
        """Execute a MOVE (action 4): rotate by ``angle`` rad, advance
        ``distance`` m at ``elevation`` deg. Dispatches on
        ``use_occupancy_collision`` exactly like upstream ``step()``.

        Returns ``(collision, deviation_m)`` where deviation is the
        occupancy-snap distance (0.0 on the no-occupancy path).
        """
        if self.use_occupancy_collision:
            collision, deviation = self._execute_move_with_occupancy_collision(
                angle, elevation, distance
            )
        else:
            collision, deviation = self._execute_move(angle, elevation, distance), 0.0
        if collision:
            self.collision_occurred = True
        return collision, deviation

    def _apply_rotation(self, angle: float) -> None:
        """Quaternion-composed yaw update (no cumulative drift) — shared head
        of both upstream move paths."""
        delta_quat = euler_angles_to_quat(np.array([0, 0, angle]), degrees=False)
        self.agent_rotation_quat = _quat_mul(self.agent_rotation_quat, delta_quat)
        self.agent_rotation_quat = self._normalize_quat(self.agent_rotation_quat)
        euler = quat_to_euler_angles(self.agent_rotation_quat, degrees=False)
        self.agent_heading = float(euler[2])

    @staticmethod
    def _movement_vector(heading: float, elevation: float, distance: float) -> np.ndarray:
        """(Δx, Δy, Δz) for advancing ``distance`` m at ``elevation`` deg."""
        elevation_rad = np.radians(elevation)
        h_dist = distance * np.cos(elevation_rad)
        v_dist = distance * np.sin(elevation_rad)
        return np.array(
            [h_dist * np.cos(heading), h_dist * np.sin(heading), v_dist]
        )

    def _execute_move(self, angle: float, elevation: float, distance: float) -> bool:
        """No-collision path (upstream lines 414-456): rotate, then teleport."""
        self._apply_rotation(angle)
        new_position = self.agent_position + self._movement_vector(
            self.agent_heading, elevation, distance
        )
        log.debug("moving to: %s", new_position)
        self.set_pose(new_position, self.agent_heading)
        return False

    def _execute_move_with_occupancy_collision(
        self,
        angle: float,
        elevation: float,
        distance: float,
        step_size: float = 0.05,  # unused, kept for upstream signature parity
    ) -> Tuple[bool, float]:
        """Occupancy path (upstream lines 188-264): rotate, compute the
        intended waypoint, snap to the nearest reachable freemap cell.

        Collision ⇔ the intended cell was not directly reachable AND the
        snap moved the agent farther than ``collision_threshold``. When no
        reachable cell exists within the search radius the agent stays put
        (heading already rotated) and it counts as a collision.
        """
        if self.occupancy is None:
            raise RuntimeError(
                "use_occupancy_collision=True but no freemap loaded — call "
                "load_freemap(<scene>/freemap.npy) at scene load"
            )

        initial_position = self.agent_position.copy()
        self._apply_rotation(angle)

        intended_position = initial_position + self._movement_vector(
            self.agent_heading, elevation, distance
        )
        log.debug("intended waypoint: %s", intended_position)

        result = self.find_nearest_reachable(intended_position[0], intended_position[1])
        if result is None:
            log.warning("no reachable position within search radius — staying put")
            return True, 0.0

        reachable_x, reachable_y, grid_distance = result
        actual_distance = float(
            np.sqrt(
                (reachable_x - intended_position[0]) ** 2
                + (reachable_y - intended_position[1]) ** 2
            )
        )
        collision_detected = (grid_distance > 0) and actual_distance > self.collision_threshold
        if collision_detected:
            log.debug(
                "collision: intended cell unreachable, snapped %.3fm (%d cells)",
                actual_distance,
                grid_distance,
            )

        # Reachable XY (cell-center coords) + intended Z.
        target_position = np.array([reachable_x, reachable_y, intended_position[2]])
        self.set_pose(target_position, self.agent_heading)
        log.debug("agent moved to: %s", self.agent_position)
        return collision_detected, actual_distance

    # ==================== occupancy queries ====================

    def find_nearest_reachable(
        self, x: float, y: float, max_search_radius: int = 30
    ) -> Optional[Tuple[float, float, int]]:
        """Nearest reachable freemap cell to world (x, y) — upstream lines
        266-350, verbatim BFS: minimum grid steps first, Euclidean distance
        as the tie-break among equally-near cells.

        Returns ``(world_x, world_y, grid_steps)`` or ``None``.
        """
        occ = self.occupancy
        idx_x = int(np.argmin(np.abs(occ[0, :] - x)))
        idx_y = int(np.argmin(np.abs(occ[:, 0] - y)))

        if occ[idx_y, idx_x] == 1:
            return (float(occ[0, idx_x]), float(occ[idx_y, 0]), 0)

        height, width = occ.shape

        queue = deque([(idx_y, idx_x, 0)])
        visited = {(idx_y, idx_x)}
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

        min_dist_found = float("inf")
        reachable_candidates: List[Tuple[int, int]] = []

        while queue:
            cy, cx, dist = queue.popleft()
            if dist > max_search_radius or dist >= min_dist_found:
                break
            next_dist = dist + 1
            for dy, dx in directions:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    if occ[ny, nx] == 1:
                        if next_dist < min_dist_found:
                            min_dist_found = next_dist
                            reachable_candidates = []
                        if next_dist == min_dist_found:
                            reachable_candidates.append((ny, nx))
                    if next_dist < min_dist_found:
                        queue.append((ny, nx, next_dist))

        if reachable_candidates:
            best_candidate = None
            min_euclidean_dist_sq = float("inf")
            for ny, nx in reachable_candidates:
                candidate_x = float(occ[0, nx])
                candidate_y = float(occ[ny, 0])
                dist_sq = (candidate_x - x) ** 2 + (candidate_y - y) ** 2
                if dist_sq < min_euclidean_dist_sq:
                    min_euclidean_dist_sq = dist_sq
                    best_candidate = (candidate_x, candidate_y, int(min_dist_found))
            return best_candidate

        return None

    # ==================== quaternion utils ====================

    @staticmethod
    def _normalize_quat(q: np.ndarray) -> np.ndarray:
        norm = np.sqrt(np.sum(q ** 2))
        return np.array([1.0, 0.0, 0.0, 0.0]) if norm < 1e-10 else q / norm
