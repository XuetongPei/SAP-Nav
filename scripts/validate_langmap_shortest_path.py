"""Check coordinate conversion and physical shortest-path navigation on a scene."""
import json
import os
from pathlib import Path

import numpy as np
from scripts.eval_langmap import load_episodes, proprioception
from scripts.langmap_shortest_path import ShortestPathLocalPolicy
from scripts.langmap_world import EVAL_BODY, VLFMWorld

world = VLFMWorld(EVAL_BODY, seed=42)
try:
    ep = load_episodes("object", "val")[0]
    world.load_scene(ep.scene)
    world.place(ep.start_position, ep.start_rotation)
    start, rotation = world.pose()
    # This target is sampled from geometry, not from any benchmark goal labels.
    for _ in range(100):
        target = world.random_navigable_near(start, 5)
        if not np.isfinite(target).all() or abs(target[1] - start[1]) > 0.3:
            continue
        initial_distance = world.geodesic(start, target.reshape(1, 3))
        if 2 < initial_distance < 8:
            break
    else:
        raise RuntimeError("No suitable reachable validation waypoint")
    gps = proprioception(target, rotation, start, rotation)["gps"]
    waypoint = [float(gps[0]), -float(gps[1])]
    local = ShortestPathLocalPolicy(world, start, rotation)
    trace = []
    for step in range(200):
        reply = local.act(waypoint, stop=True)
        trace.append(reply)
        if reply["action"] == 0:
            assert reply["local_status"] == "object_waypoint_reached", reply
            break
        world.move(reply["action"])
    else:
        raise RuntimeError("Shortest-path follower failed to arrive")
    final, _ = world.pose()
    distance = world.geodesic(final, target.reshape(1, 3))
    assert distance < 0.25, distance
    result = {"status": "passed", "scene": ep.info["langmap_scene"], "steps": len(trace),
              "initial_distance": initial_distance, "final_distance": distance, "trace": trace}
    path = Path(os.environ["VLFM_RUNTIME"]) / "shortest-path-validation.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k != "trace"})
finally:
    world.close()
