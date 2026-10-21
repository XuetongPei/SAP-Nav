"""Navmesh local controller for waypoints selected by VLFM (no goal labels)."""
import numpy as np
import quaternion
from embodiedscore_envs.benchmarks.env.sim.habitat.world import FollowerError


def waypoint_to_world(goal_xy, start_position, start_rotation, current_height):
    """Invert VLFM's episodic (forward, left) map coordinates."""
    xy = np.asarray(goal_xy, dtype=np.float64)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError(f"Invalid VLFM waypoint: {goal_xy}")
    q = np.quaternion(start_rotation[3], *start_rotation[:3])
    local = np.quaternion(0, -xy[1], 0, -xy[0])
    position = np.asarray(start_position, dtype=np.float64) + (q * local * q.inverse()).imag
    # VLFM's map is 2D. Project on the agent's current floor, also on stairs.
    position[1] = current_height
    return position


class ShortestPathLocalPolicy:
    def __init__(self, world, start_position, start_rotation, goal_radius=0.25):
        self.world = world
        self.start_position = np.asarray(start_position)
        self.start_rotation = np.asarray(start_rotation)
        self.follower = world.follower(goal_radius, fresh=True)

    def act(self, goal_xy, stop):
        current, _ = self.world.pose()
        requested = waypoint_to_world(goal_xy, self.start_position, self.start_rotation, current[1])
        target = self.world.snap(requested)
        record = {"requested_waypoint": requested.tolist(), "waypoint_xy": list(goal_xy),
                  "object_waypoint": bool(stop), "local_policy": "shortest_path"}
        if not np.isfinite(target).all():
            return {"action": 0, "local_status": "unsnappable_waypoint", **record}
        record["snapped_waypoint"] = target.tolist()
        distance = self.world.geodesic(current, target.reshape(1, 3))
        if not np.isfinite(distance):
            return {"action": 0, "local_status": "unreachable_waypoint", **record}
        record["waypoint_distance"] = float(distance)
        try:
            action = self.follower.next_action(target)
        except FollowerError:
            # Match Habitat ShortestPathFollower's default stop_on_error behavior.
            return {"action": 0, "local_status": "follower_error", **record}
        if action is None:
            # Reaching an exploration waypoint should produce another observation
            # so VLFM can update its frontiers instead of ending the episode.
            action = 0 if stop else 2
            status = "object_waypoint_reached" if stop else "frontier_waypoint_reached"
        else:
            status = "following"
        return {"action": int(action), "local_status": status, **record}
