"""Build LangMap navmeshes with VLFM agent dimensions and validate all scenes."""
import json
import os
from pathlib import Path

import numpy as np
from embodiedscore_envs.benchmarks.env.schema import targets_of
from scripts.eval_langmap import load_episodes, proprioception
from scripts.langmap_world import EVAL_BODY, VLFMWorld, navmesh_protocol

world = VLFMWorld(EVAL_BODY)
rows, seen = [], set()
try:
    for ep in load_episodes("object", "val"):
        if ep.scene.scene_file in seen:
            continue
        seen.add(ep.scene.scene_file)
        world.load_scene(ep.scene)
        world.place(ep.start_position, ep.start_rotation)
        obs = world.observe()
        assert obs["rgb"].shape == (480, 640, 3)
        assert obs["depth"].shape == (480, 640, 1)
        assert np.isfinite(obs["depth"]).all()
        pos, rot = world.pose()
        np.testing.assert_allclose(proprioception(pos, rot, pos, rot)["gps"], [0, 0], atol=1e-6)
        distance = world.geodesic(pos, targets_of(ep.goal))
        world.move(2)
        pos2, rot2 = world.pose()
        np.testing.assert_allclose(proprioception(pos2, rot2, pos, rot)["compass"], [np.pi / 6], atol=1e-5)
        settings = world.pathfinder.nav_mesh_settings
        assert abs(settings.agent_radius - 0.18) < 1e-6
        assert abs(settings.agent_height - 0.88) < 1e-6
        assert abs(settings.cell_size - 0.05) < 1e-6
        assert abs(settings.cell_height - 0.05) < 1e-6
        assert abs(settings.agent_max_climb - 0.1) < 1e-6
        assert abs(settings.agent_max_slope - 45.0) < 1e-6
        rows.append({"scene": ep.info["langmap_scene"], "render_and_turn_passed": True,
                     "first_episode_start_distance": distance if np.isfinite(distance) else None,
                     "navigable_area_m2": world.pathfinder.navigable_area,
                     "settings": {k: getattr(settings, k) for k in (
                         "agent_radius", "agent_height", "agent_max_climb", "agent_max_slope", "cell_size", "cell_height")}})
        print(len(rows), rows[-1], flush=True)
finally:
    world.close()
Path(os.environ["VLFM_RUNTIME"], "scenes-langmap-navmesh-validation.json").write_text(
    json.dumps({"navmesh_protocol": navmesh_protocol(), "scenes": rows}, indent=2) + "\n")
print("SCENES_PASSED", len(rows), flush=True)
