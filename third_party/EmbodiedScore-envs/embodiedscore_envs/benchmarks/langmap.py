"""LangMap (bo-miao/LangMap, NeurIPS 2026) — hierarchical open-vocabulary goal
navigation on 36 HM3D val scenes: four single-goal levels (object / room /
region / instance) and five-goal sequences.

Data (under ``EMBODIEDSCORE_DATA_ROOT/langmap/annotations``): one file per scene,
``<id>-<short>.json`` or ``.json.gz`` (the HF dataset ``bo-miao/LangMap``), with
``goals`` (habitat-lab ObjectGoal shape: object_id / object_category / position /
view_points[agent_state] + the annotators' descriptions), ``episodes_by_{object,
room,region,instance}_level``, ``episode_by_sequence`` (``task_sequence`` =
``[[level, index], ...]`` into those lists, by position) and ``region_annotation``.

Lines — one per level, so each has its own goal type, action table and metrics:

    langmap-object / -room / -region / -instance   single goal, actions 0-5, 500 steps,
                                                   NavMetrics(success_distance=0.25, OBJECTNAV_KEYS)
    langmap-sequence                               GoalSequence of 5, actions 0-6 (SUBTASK_STOP), 2500 steps,
                                                   SequenceNavMetrics(0.25), distance_to_goal every step

Protocol, replicated from the released evaluator (run.py / run_seq.py /
VLN_CE/habitat_extensions/task.py ``LangMapDatasetV1``):

* success = geodesic distance from the stopping pose to the nearest goal view
  point < 0.25 m, whatever ended the episode; SPL = gt / max(path, gt). An
  instance without view points counts its own position as the one view point.
* goals = every ``target_object_ids`` entry of the episode (object level: all
  instances of the category in the scene; room / region: the instances in that
  room / region; instance: the one).
* the navmesh is REBUILT with the Stretch preset (radius 0.17, height 1.41, max
  climb 0.10, cell height 0.05) — the mesh the starts, view points and
  ``min_dist`` were generated on; the shipped ``.basis.navmesh`` (0.1 / 1.5 agent)
  has ~27 % more walkable area and shorter geodesics. ``presets.bodies.LANGMAP``
  is EmbodiedScore's STANDARD body with that navmesh; ``LANGMAP_UPSTREAM`` is the
  evaluator's own rig (RGB 640x480 hfov 120 at 1.25 m, turn 30°, no tilt, no depth).
* instruction text, by level (the sentence the evaluator shows; ``Episode.instruction``
  carries it WITHOUT the leading "Find the " and trailing "." so an object-goal
  briefing can say ``Target object: "<phrase>"``; ``info["instruction_official"]``
  is the full sentence):
      object    Find the {category}.
      room      Find the {category} in the {room_name}
      region    Find the {category} in the {region_category} that has {description}
      instance  Find the {annot_unique_*_description}
  ``room_name`` / ``region_category`` / ``description`` containing "tie:" append
  ", 'Tie: A & B' means the area spans both rooms A and B". ``description`` is
  the concise one on split ``val`` (the evaluator's choice when all four levels run
  together — its reported numbers) and the detailed one on ``val_detailed`` (its
  choice when one level runs alone). Sequence sub-goals always use concise.
* the evaluator's ``black_task_ids`` (109 single-goal episodes) are dropped, as are
  episodes with no resolvable goal, a NaN / inf start, or a non-unit start quaternion.
  Sequence episodes are not black-listed (the evaluator does not either); a sub-goal
  with no resolvable goal is skipped inside its sequence.

``episode_id`` is the evaluator's unique id ``<scene>_<level>_<id>``; ``info``
keeps the raw LangMap fields (ids, min_dist, room / region / instance keys).
Sequence sub-goals: object / room / region -> ObjectGoal whose ``category`` is the
level's phrase (that is the text the sub-goal is announced with), instance ->
TextGoal(description=phrase) on the one instance.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path
from typing import Any

from .env import (OBJECTNAV_KEYS, Benchmark, Episode, GoalSequence, NavMetrics, ObjectGoal, ObjectInstance, SceneRef,
                  SequenceNavMetrics, TextGoal, data_root, scene_root)
from .presets import actions, bodies, depth

SUCCESS_DISTANCE = 0.25                       # run.py SUCCESS_DISTANCE: geodesic to the nearest goal view point
MAX_STEPS = 500                               # vlnce_task_planavid_langmap.yaml ENVIRONMENT.MAX_EPISODE_STEPS
SEQUENCE_LEN = 5
MAX_STEPS_SEQUENCE = SEQUENCE_LEN * MAX_STEPS  # the evaluator caps each sub-task at 500; one static budget here
LEVELS = ("object", "room", "region", "instance")
TIE_NOTE = ", 'Tie: A & B' means the area spans both rooms A and B"

# VLN_CE/habitat_extensions/task.py black_task_ids_ (bo-miao/LangMap @ main, 2026-10): the evaluator skips these.
BLACK_TASK_IDS = frozenset({
    '00800-TEEsavR23oF_room_74', '00810-CrMo8WxCyVb_object_30', '00810-CrMo8WxCyVb_object_31', '00810-CrMo8WxCyVb_region_75',
    '00810-CrMo8WxCyVb_region_76', '00810-CrMo8WxCyVb_room_53', '00810-CrMo8WxCyVb_room_54', '00820-mL8ThkuaVTM_object_29',
    '00820-mL8ThkuaVTM_region_47', '00820-mL8ThkuaVTM_room_39', '00823-7MXmsvcQjpJ_room_100', '00823-7MXmsvcQjpJ_room_101',
    '00823-7MXmsvcQjpJ_room_74', '00823-7MXmsvcQjpJ_room_98', '00823-7MXmsvcQjpJ_room_99', '00829-QaLdnwvtxbs_room_14',
    '00829-QaLdnwvtxbs_room_16', '00829-QaLdnwvtxbs_room_21', '00829-QaLdnwvtxbs_room_22', '00829-QaLdnwvtxbs_room_23',
    '00829-QaLdnwvtxbs_room_24', '00829-QaLdnwvtxbs_room_25', '00829-QaLdnwvtxbs_room_5', '00829-QaLdnwvtxbs_room_7',
    '00829-QaLdnwvtxbs_room_9', '00832-qyAac8rV8Zk_region_18', '00832-qyAac8rV8Zk_room_18', '00839-zt1RVoi7PcG_object_25',
    '00839-zt1RVoi7PcG_region_70', '00839-zt1RVoi7PcG_room_54', '00862-LT9Jq6dN3Ea_object_46', '00862-LT9Jq6dN3Ea_region_135',
    '00862-LT9Jq6dN3Ea_region_37', '00862-LT9Jq6dN3Ea_room_21', '00862-LT9Jq6dN3Ea_room_93', '00871-VBzV5z6i1WS_object_21',
    '00871-VBzV5z6i1WS_object_28', '00871-VBzV5z6i1WS_object_29', '00871-VBzV5z6i1WS_object_38', '00871-VBzV5z6i1WS_object_8',
    '00871-VBzV5z6i1WS_region_32', '00871-VBzV5z6i1WS_region_58', '00871-VBzV5z6i1WS_region_67', '00871-VBzV5z6i1WS_region_68',
    '00871-VBzV5z6i1WS_region_77', '00871-VBzV5z6i1WS_room_21', '00871-VBzV5z6i1WS_room_39', '00871-VBzV5z6i1WS_room_46',
    '00871-VBzV5z6i1WS_room_47', '00871-VBzV5z6i1WS_room_56', '00873-bxsVRursffK_object_29', '00873-bxsVRursffK_region_52',
    '00873-bxsVRursffK_room_39', '00876-mv2HUxq3B53_object_16', '00876-mv2HUxq3B53_object_31', '00876-mv2HUxq3B53_object_40',
    '00876-mv2HUxq3B53_region_33', '00876-mv2HUxq3B53_region_51', '00876-mv2HUxq3B53_region_69', '00876-mv2HUxq3B53_region_81',
    '00876-mv2HUxq3B53_region_92', '00876-mv2HUxq3B53_region_93', '00876-mv2HUxq3B53_room_19', '00876-mv2HUxq3B53_room_31',
    '00876-mv2HUxq3B53_room_45', '00876-mv2HUxq3B53_room_50', '00876-mv2HUxq3B53_room_59', '00877-4ok3usBNeis_object_44',
    '00877-4ok3usBNeis_region_58', '00877-4ok3usBNeis_region_8', '00877-4ok3usBNeis_room_52', '00877-4ok3usBNeis_room_9',
    '00878-XB4GS9ShBRE_object_18', '00878-XB4GS9ShBRE_region_33', '00878-XB4GS9ShBRE_room_32', '00880-Nfvxx8J5NCo_object_13',
    '00880-Nfvxx8J5NCo_object_30', '00880-Nfvxx8J5NCo_object_31', '00880-Nfvxx8J5NCo_region_22', '00880-Nfvxx8J5NCo_region_45',
    '00880-Nfvxx8J5NCo_region_46', '00880-Nfvxx8J5NCo_room_23', '00880-Nfvxx8J5NCo_room_44', '00880-Nfvxx8J5NCo_room_45',
    '00890-6s7QHgap2fW_region_19', '00890-6s7QHgap2fW_room_16', '00891-cvZr5TUy5C5_object_12', '00891-cvZr5TUy5C5_object_43',
    '00891-cvZr5TUy5C5_object_44', '00891-cvZr5TUy5C5_region_109', '00891-cvZr5TUy5C5_region_110', '00891-cvZr5TUy5C5_region_111',
    '00891-cvZr5TUy5C5_region_112', '00891-cvZr5TUy5C5_region_39', '00891-cvZr5TUy5C5_region_40', '00891-cvZr5TUy5C5_region_44',
    '00891-cvZr5TUy5C5_room_34', '00891-cvZr5TUy5C5_room_35', '00891-cvZr5TUy5C5_room_36', '00891-cvZr5TUy5C5_room_92',
    '00891-cvZr5TUy5C5_room_93', '00891-cvZr5TUy5C5_room_94', '00891-cvZr5TUy5C5_room_95', '00891-cvZr5TUy5C5_room_96',
    '00891-cvZr5TUy5C5_room_97', '00894-HY1NcmCgn3n_instance_7',
})


# ---- files -------------------------------------------------------------------------

def annotation_files(root: Path) -> list[tuple[str, Path]]:
    """``[(scene, path)]`` sorted by scene; ``.json.gz`` preferred when both exist."""
    found: dict[str, Path] = {}
    for p in sorted(root.iterdir()):
        if not p.name.endswith((".json", ".json.gz")) or p.name.startswith("._"):
            continue
        scene = p.name.split(".json")[0]
        if scene not in found or p.suffix == ".gz":
            found[scene] = p
    return sorted(found.items())


def _load(path: Path) -> dict:
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _scene_ref(scene: str, sroot: Path) -> SceneRef:
    short = scene.split("-", 1)[1] if "-" in scene else scene
    return SceneRef(scene_file=str(sroot / "hm3d" / "val" / scene / f"{short}.basis.glb"))


# ---- goals -------------------------------------------------------------------------

def _instance(g: dict) -> ObjectInstance:
    pos = tuple(float(x) for x in g["position"])
    vps = tuple(tuple(float(x) for x in vp["agent_state"]["position"])
                for vp in g.get("view_points", []) if vp.get("agent_state", {}).get("position"))
    # task.py: an instance without view points is reached at its own position
    return ObjectInstance(object_id=str(g["object_id"]), category=str(g.get("object_category", "object")),
                          position=pos, view_points=vps or (pos,))


def _targets(e: dict, goals: dict[str, dict]) -> tuple[ObjectInstance, ...]:
    return tuple(_instance(goals[t]) for t in e.get("target_object_ids", []) if t in goals)


def _phrase(e: dict, level: str, goals: dict[str, dict], regions: dict[str, dict], style: str) -> str:
    """The instruction minus "Find the " (task.py / run_seq.py templates, verbatim otherwise)."""
    desc_key = "concise_description" if style == "concise" else "detailed_description"
    inst_key = f"annot_unique_{style}_description"
    if level in ("region", "room", "object"):
        cat = str(e["object_category"])
    else:
        first = e["target_object_ids"][0]
        cat = str(goals.get(first, {}).get("object_category", "object"))
    if level == "object":
        return cat
    if level == "room":
        room = str(e.get("room_name", "room"))
        out = f"{cat} in the {room}"
        return out + TIE_NOTE if "tie:" in room.lower() else out
    if level == "region":
        rid = e.get("region_id", "")
        if rid in regions:
            info = regions[rid]
            rcat = str(info.get("region_category", "region")).lower()
            desc = str(info.get(desc_key, ""))
            out = f"{cat} in the {rcat} that has {desc}"
            return out + TIE_NOTE if ("tie:" in rcat or "tie:" in desc.lower()) else out
        return f"{cat} in the target region"
    iid = e.get("instance_id", "")
    if iid in goals:
        return str(goals[iid].get(inst_key, "object"))
    return "target instance"


def _official(level: str, phrase: str, sequence: bool) -> str:
    # task.py puts a full stop on the object-level sentence; run_seq.py (sub-goals) does not
    return "Find the " + phrase + ("." if level == "object" and not sequence else "")


def _valid_start(e: dict) -> bool:
    pos, rot = e.get("start_position", [0, 0, 0]), e.get("start_rotation", [0, 0, 0, 1])
    if any(math.isnan(x) or math.isinf(x) for x in pos) or any(math.isnan(x) or math.isinf(x) for x in rot):
        return False
    return abs(math.sqrt(sum(x * x for x in rot)) - 1.0) <= 0.1


def _split_parts(split: str) -> tuple[str | None, str]:
    """``val`` / ``val_detailed`` -> (None, style); ``<subset>`` / ``<subset>_detailed``
    -> (subset, style). A subset is a file ``<data_root>/langmap/splits/<subset>.json``
    written by MIP's ``scripts/langmap_subsets.py`` (``dev36`` / ``test108`` ...: nested,
    seeded, per-scene stratified samples of the sequences plus the single-goal episodes
    their sub-goals reference) or ``scripts/langmap_single_subsets.py`` (``single_dev36``
    / ``single_test108`` ...: the four single-goal levels sampled independently from
    their own pools, ``levels`` only — the sequence line refuses them)."""
    style = "concise"
    name = split
    if split.endswith("_detailed"):
        style, name = "detailed", split[: -len("_detailed")]
    return (None if name == "val" else name), style


def _style(split: str) -> str:
    return _split_parts(split)[1]


def _subset(data_root_: Path, name: str) -> dict:
    path = data_root_ / "langmap" / "splits" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"langmap: unknown split {name!r} (val | val_detailed | a subset file under {path.parent})")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _subset_ids(data_root_: Path, name: str, key: str) -> list[str]:
    """``key`` = ``sequences`` (sequence line) or ``levels`` (single-goal lines); a subset
    file that lacks it was built for the other kind of line."""
    doc = _subset(data_root_, name)
    if key not in doc:
        raise ValueError(f"langmap subset {name!r} has no {key!r}: it was built for the "
                         + ("single-goal" if key == "sequences" else "sequence") + " line(s) only")
    return doc[key]


def _select(episodes: list[Episode], ids: list[str]) -> list[Episode]:
    """The subset's episodes in the SUBSET's order (ranks nest), re-indexed."""
    by_id = {e.episode_id: e for e in episodes}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"langmap subset names {len(missing)} episode(s) the loader does not produce, e.g. {missing[:3]}")
    out = []
    for i in ids:
        e = by_id[i]
        out.append(Episode(index=len(out), episode_id=e.episode_id, scene=e.scene, start_position=e.start_position,
                           start_rotation=e.start_rotation, goal=e.goal, instruction=e.instruction, info=e.info))
    return out


# ---- loaders -----------------------------------------------------------------------

def load_level(level: str, split: str, data_root_=None, scene_root_=None) -> list[Episode]:
    subset, style = _split_parts(split)
    if subset is not None:
        full = load_level(level, "val_detailed" if style == "detailed" else "val", data_root_, scene_root_)
        return _select(full, list(_subset_ids(data_root(data_root_), subset, "levels")[level]))
    root = data_root(data_root_) / "langmap" / "annotations"
    sroot = scene_root(scene_root_)
    out: list[Episode] = []
    for scene, path in annotation_files(root):
        ann = _load(path)
        goals = {g["object_id"]: g for g in ann["goals"]}
        regions = ann.get("region_annotation", {})
        for e in ann.get(f"episodes_by_{level}_level", []):
            eid = f"{scene}_{level}_{e['episode_id']}"
            if eid in BLACK_TASK_IDS or not _valid_start(e):
                continue
            inst = _targets(e, goals)
            if not inst:
                continue
            phrase = _phrase(e, level, goals, regions, style)
            cat = inst[0].category if level == "instance" else str(e["object_category"])
            md = (e.get("min_dist") or [{}])[0]
            out.append(Episode(
                index=len(out), episode_id=eid, scene=_scene_ref(scene, sroot),
                start_position=tuple(float(x) for x in e["start_position"]),
                start_rotation=tuple(float(x) for x in e["start_rotation"]),
                goal=ObjectGoal(category=cat, instances=inst),
                instruction=phrase,
                info={"scene_id": _scene_ref(scene, sroot).scene_file, "langmap_scene": scene, "level": level,
                      "style": style, "langmap_episode_id": e["episode_id"], "instruction_official": _official(level, phrase, False),
                      "target_object_ids": list(e.get("target_object_ids", [])),
                      "children_object_categories": list(e.get("children_object_categories") or []),
                      "room_name": e.get("room_name"), "region_id": e.get("region_id"), "instance_id": e.get("instance_id"),
                      "geodesic_distance": md.get("geodesic_distance"), "euclidean_distance": md.get("euclidean_distance")},
            ))
    return out


def _subgoal(level: str, e: dict, goals: dict[str, dict], regions: dict[str, dict]):
    """run_seq.py's sub-task as a goal: the phrase is what the sub-goal is announced with."""
    inst = _targets(e, goals)
    if not inst:
        return None, None
    phrase = _phrase(e, level, goals, regions, "concise")
    if level == "instance":
        return TextGoal(instance=inst[0], description=phrase), phrase
    return ObjectGoal(category=phrase, instances=inst), phrase


def load_sequence(split: str, data_root_=None, scene_root_=None) -> list[Episode]:
    subset, style = _split_parts(split)
    if style != "concise":
        raise ValueError(f"langmap-sequence: sub-goals always use concise descriptions; no {split!r}")
    if subset is not None:
        full = load_sequence("val", data_root_, scene_root_)
        return _select(full, list(_subset_ids(data_root(data_root_), subset, "sequences")))
    root = data_root(data_root_) / "langmap" / "annotations"
    sroot = scene_root(scene_root_)
    out: list[Episode] = []
    for scene, path in annotation_files(root):
        ann = _load(path)
        goals = {g["object_id"]: g for g in ann["goals"]}
        regions = ann.get("region_annotation", {})
        tables = {lv: ann.get(f"episodes_by_{lv}_level", []) for lv in LEVELS}
        for e in ann.get("episode_by_sequence", []):
            eid = f"{scene}_sequence_{e['episode_id']}"
            seq = e.get("task_sequence") or []
            if not seq or not _valid_start(e):
                continue
            subs, texts, refs = [], [], []
            for level, idx in seq:
                table = tables.get(level, [])
                if not (0 <= int(idx) < len(table)):
                    continue  # run_seq.py: "Invalid task reference ... skipping"
                g, phrase = _subgoal(level, table[int(idx)], goals, regions)
                if g is None:
                    continue
                subs.append(g)
                texts.append(_official(level, phrase, True))
                refs.append({"level": level, "index": int(idx), "langmap_episode_id": table[int(idx)]["episode_id"],
                             "target_object_ids": list(table[int(idx)].get("target_object_ids", []))})
            if not subs:
                continue
            md = (e.get("min_dist") or [{}])[0]
            out.append(Episode(
                index=len(out), episode_id=eid, scene=_scene_ref(scene, sroot),
                start_position=tuple(float(x) for x in e["start_position"]),
                start_rotation=tuple(float(x) for x in e["start_rotation"]),
                goal=GoalSequence(goals=tuple(subs)),
                info={"scene_id": _scene_ref(scene, sroot).scene_file, "langmap_scene": scene, "level": "sequence",
                      "langmap_episode_id": e["episode_id"], "task_sequence": [list(x) for x in seq], "subgoals": refs,
                      "instructions_official": texts, "n_filtered_subgoals": len(seq) - len(subs),
                      "geodesic_distance": md.get("geodesic_distance"), "euclidean_distance": md.get("euclidean_distance")},
            ))
    return out


# ---- declarations ------------------------------------------------------------------

def _decl_level(level: str, upstream: bool) -> Benchmark:
    return Benchmark(
        name=f"langmap-{level}" + ("-upstream" if upstream else ""),
        gym_id=f"EmbodiedScore/LangMap-{level.capitalize()}" + ("-Upstream-v0" if upstream else "-v0"),
        body=bodies.LANGMAP_UPSTREAM if upstream else bodies.LANGMAP,
        actions=actions.NAV if upstream else actions.STANDARD,
        splits=("val", "val_detailed"),
        episodes=lambda split, data_root=None, scene_root=None, **kw: load_level(level, split, data_root, scene_root),
        metrics=lambda env, **o: NavMetrics(env, success_distance=o.get("success_distance", SUCCESS_DISTANCE), keys=OBJECTNAV_KEYS),
        depth=None if upstream else depth.STANDARD, max_episode_steps=MAX_STEPS,
        variant="upstream" if upstream else "standard",
        description=f"LangMap {level}-level goals (" + ("evaluator rig: RGB 640x480 hfov 120, turn 30, no tilt" if upstream else "STANDARD body")
                    + ", Stretch navmesh, success 0.25)",
    )


def _decl_sequence(upstream: bool) -> Benchmark:
    return Benchmark(
        name="langmap-sequence" + ("-upstream" if upstream else ""),
        gym_id="EmbodiedScore/LangMap-Sequence" + ("-Upstream-v0" if upstream else "-v0"),
        body=bodies.LANGMAP_UPSTREAM if upstream else bodies.LANGMAP,
        actions=actions.GOAT,  # SUBTASK_STOP closes a sub-goal on both variants (the evaluator has no such action: it re-seats per sub-task)
        splits=("val",),
        episodes=lambda split, data_root=None, scene_root=None, **kw: load_sequence(split, data_root, scene_root),
        metrics=lambda env, **o: SequenceNavMetrics(env, success_distance=o.get("success_distance", SUCCESS_DISTANCE)),
        depth=None if upstream else depth.STANDARD, max_episode_steps=MAX_STEPS_SEQUENCE, dtg_policy="every_step",
        variant="upstream" if upstream else "standard",
        description="LangMap five-goal sequences (" + ("evaluator rig" if upstream else "STANDARD body")
                    + ", Stretch navmesh, success 0.25 per sub-goal, 2500 steps)",
    )


BENCHMARKS = tuple(_decl_level(lv, u) for lv in LEVELS for u in (False, True)) + (_decl_sequence(False), _decl_sequence(True))
