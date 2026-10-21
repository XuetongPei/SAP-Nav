"""Check the text boundary and the exact official navmesh construction knobs."""
from types import SimpleNamespace

import pytest

from scripts.langmap_goal_adapter import apply_goal_attributes
from scripts.langmap_world import EVAL_BODY, navmesh_protocol, navmesh_settings
from third_party.langmap.navmesh_utils import NAVMESH_PRESETS


def test_category_detection_keeps_full_description_out_of_class_filter():
    policy = SimpleNamespace()
    for category, text in (
        ("chair", "Seems like there is a chair in the Bedroom ahead."),
        ("sofa", "Seems like there is a red sofa beside the window ahead."),
    ):
        actual = apply_goal_attributes(policy, {"object_category": category, "clip_goal": text})
        assert actual == category
        assert policy._non_coco_caption == category + " ."
        assert policy._text_prompt == text


def test_navmesh_matches_official_except_vlfm_agent_dimensions():
    import habitat_sim
    official = habitat_sim.NavMeshSettings()
    official.set_defaults()
    for name, value in NAVMESH_PRESETS["stretch"].items():
        setattr(official, name, value)
    official.include_static_objects = True
    actual = navmesh_settings()
    for name, value in navmesh_protocol()["settings"].items():
        if name not in {"agent_radius", "agent_height"}:
            assert value == pytest.approx(getattr(official, name))
    assert actual.agent_radius == pytest.approx(0.18)
    assert actual.agent_height == pytest.approx(0.88)
    assert EVAL_BODY.navmesh.cell_height == pytest.approx(0.05)
    assert EVAL_BODY.navmesh.agent_max_climb == pytest.approx(0.10)
    assert navmesh_protocol()["include_static_objects"] is True


def test_rebuild_includes_static_objects_and_cache_loads_without_rebuilding(tmp_path, monkeypatch):
    import json
    from scripts.langmap_world import VLFMWorld

    class Pathfinder:
        is_loaded = True
        loads = 0
        def save_nav_mesh(self, filename):
            from pathlib import Path
            Path(filename).write_bytes(b"test mesh")
        def load_nav_mesh(self, filename):
            self.loads += 1
            return True

    pf = Pathfinder()
    calls = []
    def recompute(pathfinder, settings):
        assert pathfinder is pf
        assert settings.cell_height == pytest.approx(0.05)
        assert settings.include_static_objects is True
        calls.append(1)
        return True

    source = tmp_path / "scene.basis.glb"
    source.write_bytes(b"test scene")
    scene = SimpleNamespace(scene_file=str(source))
    world = SimpleNamespace(pathfinder=pf, _sim=SimpleNamespace(recompute_navmesh=recompute))
    monkeypatch.setenv("VLFM_RUNTIME", str(tmp_path))
    VLFMWorld._apply_navmesh(world, scene, EVAL_BODY.navmesh)
    VLFMWorld._apply_navmesh(world, scene, EVAL_BODY.navmesh)
    assert len(calls) == 1
    assert pf.loads == 1
    meta = next((tmp_path / "navmeshes/langmap-vlfm-agent").glob("*.json"))
    assert json.loads(meta.read_text())["navmesh_protocol"]["include_static_objects"] is True
