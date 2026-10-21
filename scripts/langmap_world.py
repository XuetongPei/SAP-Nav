"""LangMap's official navmesh construction with VLFM's agent dimensions."""
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import habitat_sim
from embodiedscore_envs.benchmarks.env.sim.habitat.world import SimWorld
from embodiedscore_envs.benchmarks.env.sim.habitat.body import NavMesh
from embodiedscore_envs.benchmarks.presets import bodies
from third_party.langmap.navmesh_utils import NAVMESH_PRESETS

LANGMAP_COMMIT = "3f2cd6e37a7f06818f515b480b0e81293a8269af"
NAVMESH_CACHE = "langmap-vlfm-agent"
_preset = dict(NAVMESH_PRESETS["stretch"], agent_radius=0.18, agent_height=0.88)
EVAL_BODY = replace(bodies.LOCOBOT, navmesh=NavMesh.recompute(
    agent_radius=_preset["agent_radius"], agent_height=_preset["agent_height"],
    agent_max_climb=_preset["agent_max_climb"], cell_height=_preset["cell_height"]))


def navmesh_settings():
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    for name, value in _preset.items():
        setattr(settings, name, value)
    # Habitat-Sim 0.3.3 moved LangMap's include_static_objects call argument
    # onto NavMeshSettings. Preserve the same construction semantics.
    settings.include_static_objects = True
    return settings


def navmesh_protocol():
    settings = navmesh_settings()
    values = {name: getattr(settings, name) for name in dir(settings)
              if not name.startswith("_") and isinstance(getattr(settings, name), (int, float, bool))}
    return {"source_commit": LANGMAP_COMMIT, "preset": "stretch",
            "agent_overrides": {"agent_radius": 0.18, "agent_height": 0.88},
            "include_static_objects": True, "settings": values}


class VLFMWorld(SimWorld):
    def _apply_navmesh(self, scene, spec):
        if spec != EVAL_BODY.navmesh:
            raise ValueError("VLFMWorld requires EVAL_BODY's LangMap navmesh specification")
        root = Path(os.environ["VLFM_RUNTIME"]) / "navmeshes" / NAVMESH_CACHE
        root.mkdir(parents=True, exist_ok=True)
        source = Path(scene.scene_file)
        mesh = root / (source.parent.name + ".navmesh")
        metadata_path = mesh.with_suffix(".json")
        metadata = {"source": str(source.resolve()), "source_size": source.stat().st_size,
                    "source_mtime_ns": source.stat().st_mtime_ns, "spec": asdict(spec),
                    "habitat_sim": habitat_sim.__version__, "navmesh_protocol": navmesh_protocol()}
        if mesh.exists() and metadata_path.exists() and json.loads(metadata_path.read_text()) == metadata:
            if not self.pathfinder.load_nav_mesh(str(mesh)):
                raise RuntimeError(f"Failed to load cached navmesh: {mesh}")
            return
        # Match LangMap's rebuild_navmesh, including static objects; only the
        # agent radius and height differ from its Stretch preset.
        if not self._sim.recompute_navmesh(self.pathfinder, navmesh_settings()):
            raise RuntimeError(f"Navmesh rebuild failed for {source}")
        if not self.pathfinder.is_loaded:
            raise RuntimeError(f"Navmesh is not loaded for {source}")
        # Multiple evaluation workers may encounter the same new scene at once.
        # Each writes private temporary files, then atomically publishes them.
        temp_mesh = mesh.with_name(mesh.name + f".{os.getpid()}.tmp")
        self.pathfinder.save_nav_mesh(str(temp_mesh))
        temp_mesh.replace(mesh)
        temp_meta = metadata_path.with_name(metadata_path.name + f".{os.getpid()}.tmp")
        temp_meta.write_text(json.dumps(metadata, indent=2) + "\n")
        temp_meta.replace(metadata_path)
