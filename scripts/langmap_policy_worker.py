"""Run upstream VLFM in its Python 3.9 runtime over a private Unix socket.

The simulator runs in the existing Python 3.11 Habitat environment. Only RGB,
normalized depth, proprioception and the current goal text cross this boundary.
Local actions are requested from a navmesh shortest-path follower in that
simulator. Ground-truth target positions never cross the policy boundary.
"""
import argparse
import os
import traceback
from dataclasses import asdict
from multiprocessing.connection import Listener

import numpy as np
import torch

from vlfm.policy.base_objectnav_policy import VLFMConfig
from vlfm.policy.habitat_policies import HabitatITMPolicyV2
from vlfm.policy.itm_policy import ITMPolicyV2
from scripts.langmap_goal_adapter import apply_goal_attributes


class ExternalLocalPolicy:
    """No neural local model or CUDA context for navmesh-only local control."""
    device = torch.device("cpu")

    def reset(self):
        pass

    def act(self, *args, **kwargs):
        raise RuntimeError("Local actions must come from the shortest-path follower")


class ShortestPathVLFMPolicy(HabitatITMPolicyV2):
    """Keep VLFM's global policy and replace its PointNav action inference."""
    def _make_pointnav_policy(self, checkpoint_path):
        return ExternalLocalPolicy()

    def _pointnav(self, goal, stop=False):
        # Preserve the upstream object-arrival rule; frontier navigation must
        # continue until the local waypoint is reached.
        self._last_goal = np.asarray(goal).copy()
        distance = float(np.linalg.norm(goal - self._observations_cache["robot_xy"]))
        if stop and distance < self._pointnav_stop_radius:
            self._called_stop = True
            self.local_status = "vlfm_object_stop_radius"
            return self._stop_action
        self.connection.send({"op": "shortest_path", "goal_xy": np.asarray(goal).tolist(), "stop": bool(stop)})
        reply = self.connection.recv()
        action = int(reply["action"])
        if action == 0 and stop:
            self._called_stop = True
        return torch.tensor([[action]], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = asdict(VLFMConfig())
    cfg.pop("name")
    policy = ShortestPathVLFMPolicy(
        **cfg, camera_height=0.88, min_depth=0.5, max_depth=5.0,
        camera_fov=79.0, image_width=640, visualize=False,
    )
    os.umask(0o077)
    with Listener(args.socket, family="AF_UNIX") as listener:
        with listener.accept() as conn:
            policy.connection = conn
            conn.send({"ready": True, "policy": "HabitatITMPolicyV2", "local_policy": "shortest_path"})
            while True:
                request = conn.recv()
                if request.get("op") == "close":
                    return
                try:
                    policy.local_status = None
                    if request["first"]:
                        seed = int(request["seed"])
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                    # Original HabitatMixin.act only maps six/twenty-one numeric
                    # category IDs to text. Bypass that lookup for LangMap text;
                    # upstream mapping, perception and global waypoint selection
                    # remain intact, with shortest-path local actions.
                    attributes = request["goal_attributes"]
                    if request["first"]:
                        apply_goal_attributes(policy, attributes)
                    # Habitat's trainer normally batches observations on the
                    # policy device; preserve that contract across the socket.
                    device = policy._pointnav_policy.device
                    obs = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in request["observations"].items()}
                    obs["objectgoal"] = attributes["object_category"]
                    masks = torch.tensor([[not request["first"]]], dtype=torch.bool, device=device)
                    try:
                        with torch.inference_mode():
                            action, _ = ITMPolicyV2.act(policy, obs, None, None, masks, deterministic=True)
                        action_id = int(action.item())
                        reason = "policy"
                    except StopIteration:
                        action_id, reason = 0, "upstream_stop_iteration"
                    conn.send({"action": action_id, "reason": reason,
                               "local_status": policy.local_status,
                               "target_detected": bool(policy._policy_info.get("target_detected", False))})
                except Exception:
                    conn.send({"error": traceback.format_exc()})


if __name__ == "__main__":
    main()
