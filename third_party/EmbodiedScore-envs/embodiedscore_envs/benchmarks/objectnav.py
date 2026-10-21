"""ObjectNav (habitat-lab 0.2.4 ObjectNav-v1 task) on HM3D v1 / v2 and MP3D v1,
and HM3D-OVON — the same shard format and loader on the open-vocabulary
episode files (declared here because everything but the files and the goal
widening is shared).

Data (under ``EMBODIEDSCORE_DATA_ROOT/objectnav``):
    {hm3d/v1 | hm3d/v2 | mp3d/v1}/{split}/{split}.json.gz              shell (category tables; episodes
                                                                        only when there is no content/)
    {...}/{split}/content/<scene>.json.gz                                one shard per scene, name-sorted
Each shard is ``{episodes, goals_by_category, category_to_*}``; an episode's
goals are ``goals_by_category[f"{basename(scene_id)}_{object_category}"]`` —
every instance of that category in the scene, each with its view points.
``episode_id`` is re-numbered ``str(i)`` within a shard, as habitat-lab does
(unique per scene, not per split); the flat-file case numbers over the file.

Scenes: HM3D ``hm3d/val/<id>/<short>.basis.glb`` with the annotated
scene-dataset config (v2 episodes say ``hm3d_v0.2/{val,minival}`` — provide
those as links to ``hm3d/val``); MP3D ``mp3d/<scan>/<scan>.glb`` bare.

Variants. ``objectnav-*`` / ``ovon`` (default) run on the shared STANDARD body
(``presets.bodies.STANDARD``). ``-upstream``: ObjectNav on habitat-lab's
``objectnav_hm3d.yaml`` LoCoBot rig (``presets.bodies.LOCOBOT``, navmesh
recomputed for 0.18 / 0.88, depth 0.5-5 m normalised); OVON on the Stretch rig
every released OVON experiment config uses (``objectnav_stretch_hm3d`` +
``OVONSim-v0``: ``presets.bodies.STRETCH``, RGB only, navmesh climb 0.1 /
cell 0.05). Actions 0-5, 500 steps in all.

Task semantics (both variants). ObjectNav: distance_to_goal to the nearest view
point of the category's instances, success 0.1 m, spl, soft_spl (habitat's stock
measures). OVON: the upstream protocol — ``OVONDistanceToGoal`` widens the
target set with the view points of every ``children_object_categories`` entry
that exists in the shard's goal table (``ovon/measurements/nav.py``), and
``success_distance`` is 0.25 (every released experiment config). OVON
``episode_id`` is kept as authored (its loader does not renumber); its three
val splits have directory / file names that differ (``_OVON_FILES``) plus the
derived ``mip{100,60}_*`` files.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

from .env import (OBJECTNAV_KEYS, Benchmark, Episode, NavMetrics, ObjectGoal, ObjectInstance, SceneRef, data_root,
                  scene_root)
from .presets import actions, bodies, depth

SUCCESS_DISTANCE = 0.1          # habitat-lab ObjectNav
OVON_SUCCESS_DISTANCE = 0.25    # every released OVON experiment config
_SCENE_PREFIX = "data/scene_datasets/"

_DATASETS = {"hm3d-v1": "hm3d/v1", "hm3d-v2": "hm3d/v2", "mp3d-v1": "mp3d/v1"}
_OVON_MIP_N = (100, 60)
_OVON_FILES = {"val_seen": "val_seen", "val_seen_synonyms": "val_unseen_easy", "val_unseen": "val_unseen_hard",
               **{f"mip{n}_{suf}": f"mip{n}_{suf}" for n in _OVON_MIP_N for suf in ("seen", "seen_synonyms", "unseen")}}


def _load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _scene_ref(scene_id: str, dataset_config: str | None, sroot: Path) -> SceneRef:
    rel = scene_id[len(_SCENE_PREFIX):] if scene_id.startswith(_SCENE_PREFIX) else scene_id
    rel = rel.replace("//", "/")
    cfg = None
    if dataset_config and dataset_config != "default":
        c = dataset_config[len("./"):] if dataset_config.startswith("./") else dataset_config
        c = c[len(_SCENE_PREFIX):] if c.startswith(_SCENE_PREFIX) else c
        cfg = str(sroot / c)
    return SceneRef(scene_file=str(sroot / rel), dataset_config=cfg)


def parse_instances(goals: list[dict]) -> tuple[ObjectInstance, ...]:
    """habitat-lab's ObjectGoal list -> ObjectInstance tuple (view points = agent positions)."""
    out = []
    for g in goals:
        vps = tuple(tuple(float(x) for x in vp["agent_state"]["position"]) for vp in g.get("view_points", []))
        out.append(ObjectInstance(object_id=str(g.get("object_id")), category=str(g.get("object_category")),
                                  position=tuple(float(x) for x in g["position"]), view_points=vps))
    return tuple(out)


def _episodes_from(raw: dict, sroot: Path, start_index: int, renumber: bool = True,
                   children_widen: bool = False) -> list[Episode]:
    """``children_widen`` = OVON's ``OVONDistanceToGoal``: the instances of every
    ``children_object_categories`` entry present in the goal table join the goal."""
    gbc = {k: parse_instances(v) for k, v in raw.get("goals_by_category", {}).items()}
    out: list[Episode] = []
    for i, e in enumerate(raw["episodes"]):
        scene_id = e["scene_id"]
        key = f"{os.path.basename(scene_id)}_{e['object_category']}"
        instances = gbc[key]
        if children_widen:
            for child in e.get("children_object_categories") or []:
                ck = f"{os.path.basename(scene_id)}_{child}"
                if ck in gbc:
                    instances = instances + gbc[ck]
        info = dict(e.get("info") or {})
        info.update({"scene_id": scene_id, "object_category": e["object_category"], "goals_key": key,
                     "shortest_paths": e.get("shortest_paths"),
                     "children_object_categories": e.get("children_object_categories")})
        out.append(Episode(
            index=start_index + len(out),
            episode_id=str(i) if renumber else str(e["episode_id"]),
            scene=_scene_ref(scene_id, e.get("scene_dataset_config"), sroot),
            start_position=tuple(float(x) for x in e["start_position"]),
            start_rotation=tuple(float(x) for x in e["start_rotation"]),
            goal=ObjectGoal(category=str(e["object_category"]), instances=instances),
            info=info,
        ))
    return out


def load_split(base: Path, split: str, sroot: Path, renumber: bool = True, shell_name: str | None = None,
               children_widen: bool = False) -> list[Episode]:
    """habitat-lab PointNavDatasetV1 loading order: the shell, then every
    ``content/*.json.gz`` shard sorted by name; without content/, the shell's episodes."""
    shell = base / split / f"{shell_name or split}.json.gz"
    if not shell.is_file():
        shell = base / f"{shell_name or split}.json.gz"
    raw = _load_json_gz(shell)
    content = shell.parent / "content"
    out: list[Episode] = []
    if content.is_dir():
        shards = sorted(p for p in os.listdir(content) if p.endswith(".json.gz") and not p.startswith("._"))
        for name in shards:
            out.extend(_episodes_from(_load_json_gz(content / name), sroot, len(out), renumber, children_widen))
    else:
        out.extend(_episodes_from(raw, sroot, 0, renumber, children_widen))
    return out


def load_episodes(dataset: str, split: str, data_root_=None, scene_root_=None) -> list[Episode]:
    if dataset == "ovon":
        return load_split(data_root(data_root_) / "ovon" / "hm3d", split, scene_root(scene_root_),
                          renumber=False, shell_name=_OVON_FILES.get(split, split), children_widen=True)
    return load_split(data_root(data_root_) / "objectnav" / _DATASETS[dataset], split, scene_root(scene_root_))


def _decl(dataset: str, gym_id: str, splits: tuple[str, ...], upstream: bool, name: str | None = None) -> Benchmark:
    ovon = dataset == "ovon"
    sd = OVON_SUCCESS_DISTANCE if ovon else SUCCESS_DISTANCE
    if upstream:
        body, acts, dspec = (bodies.STRETCH, actions.NAV_LOOK, None) if ovon else (bodies.LOCOBOT, actions.NAV_LOOK, depth.LOCOBOT)
    else:
        body, acts, dspec = bodies.STANDARD, actions.STANDARD, depth.STANDARD
    line = name or f"objectnav-{dataset}"
    return Benchmark(
        name=line + ("-upstream" if upstream else ""),
        gym_id=gym_id.replace("-v0", "-Upstream-v0") if upstream else gym_id,
        body=body, actions=acts, splits=splits,
        episodes=lambda split, data_root=None, scene_root=None, **kw: load_episodes(dataset, split, data_root, scene_root),
        metrics=lambda env, **o: NavMetrics(env, success_distance=o.get("success_distance", sd), keys=OBJECTNAV_KEYS),
        depth=dspec, max_episode_steps=500, variant="upstream" if upstream else "standard",
        description=(f"HM3D-OVON ({'Stretch rig' if upstream else 'STANDARD body'}; success 0.25)" if ovon
                     else f"ObjectNav {dataset} ({'LoCoBot rig' if upstream else 'STANDARD body'})"),
    )


_LINES = (
    ("hm3d-v1", "EmbodiedScore/ObjectNav-HM3Dv1-v0", ("train", "val", "val_mini"), None),
    ("hm3d-v2", "EmbodiedScore/ObjectNav-HM3Dv2-v0", ("train", "val", "val_mini"), None),
    ("mp3d-v1", "EmbodiedScore/ObjectNav-MP3D-v0", ("train", "val", "val_mini"), None),
    ("ovon", "EmbodiedScore/OVON-v0", tuple(_OVON_FILES), "ovon"),
)
BENCHMARKS = tuple(_decl(d, g, sp, upstream=u, name=n) for d, g, sp, n in _LINES for u in (False, True))
