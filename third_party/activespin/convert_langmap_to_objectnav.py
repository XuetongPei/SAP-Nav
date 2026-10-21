"""Convert LangMap_Annotations into habitat-lab datasets for vanilla-VLFM eval.

Emits one dataset per level (object / room / region / instance):

    <dst>/<level>_level/val/val.json.gz            (empty episodes + category map)
    <dst>/<level>_level/val/content/<hash>.json.gz (per-scene episodes + goals)

Episodes carry `goal_attributes` with:
    object_category : detection target for GroundingDINO/YOLOv7
    clip_goal       : full-sentence BLIP2-ITM prompt, built EXACTLY like
                      modules/run.py::load_langmap_episode builds `clip_goal`
                      (description_type='unique_concise') so the VLFM baseline
                      sees the same text as the ActiveSpin agent.
    instruction     : human-readable instruction string (logging only)

Success setup mirrors the ActiveSpin harness: goals = every instance in the
episode's target_object_ids, each with its LangMap view_points, evaluated with
DISTANCE_TO=VIEW_POINTS and SUCCESS_DISTANCE=1.0 over 500 steps.
"""

import argparse
import gzip
import json
import os
import re
from collections import defaultdict
from glob import glob

LEVELS = {
    "object": "episodes_by_object_level",
    "room": "episodes_by_room_level",
    "region": "episodes_by_region_level",
    "instance": "episodes_by_instance_level",
}


def scene_short_from_fname(path):
    return os.path.basename(path).replace(".json.gz", "").replace(".json", "")


def scene_glb_path(scene_short):
    scene_hash = scene_short.split("-", 1)[1]
    return f"hm3d/val/{scene_short}/{scene_hash}.basis.glb"


def load_scene(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def resolve_category(ep, target_ids, goals_by_id):
    """Mirror modules/run.py: prefer most-specific annot_manual_category."""
    cat = ep.get("object_category")
    if target_ids and target_ids[0] in goals_by_id:
        mc = goals_by_id[target_ids[0]].get("annot_manual_category", [])
        if mc and len(mc) > 1:
            cat = mc[-1]
        elif mc and not cat:
            cat = mc[0]
    if cat is None:
        iid = ep.get("instance_id", "")
        if iid and "_" in iid:
            cat = "_".join(iid.split("_")[:-1])
        elif target_ids and "_" in target_ids[0]:
            cat = "_".join(target_ids[0].split("_")[:-1])
    return cat or "object"


def build_texts(level, ep, cat, target_ids, goals_by_id, regions):
    """Return (instruction, clip_goal) — same strings as modules/run.py."""
    if level == "object":
        return (
            f"Find a {cat}",
            f"Seems like there is a {cat} ahead.",
        )

    if level == "room":
        room_name = ep.get("room_name", "room")
        return (
            f"Find a {cat} in the {room_name}",
            f"Seems like there is a {cat} in the {room_name} ahead.",
        )

    if level == "region":
        region_info = regions.get(ep.get("region_id", ""), {})
        region_category = region_info.get("region_category", "")
        desc = region_info.get("concise_description", "")
        if region_category and desc:
            return (
                f"Find a {cat} in the {region_category} with {desc}",
                f"Seems like there is a {cat} in the {region_category} with {desc} ahead.",
            )
        if region_category:
            return (
                f"Find a {cat} in the {region_category}",
                f"Seems like there is a {cat} in the {region_category} ahead.",
            )
        return (f"Find a {cat}", f"Seems like there is a {cat} ahead.")

    # instance
    desc = ""
    room = ""
    for tid in target_ids:
        if tid in goals_by_id:
            g = goals_by_id[tid]
            desc = g.get("annot_unique_concise_description") or g.get(
                "annot_appearance_description") or ""
            region_info = regions.get(g.get("annot_region_id", ""), {})
            room = region_info.get("region_category", "")
            break
    if desc and room:
        instruction = f"Find the {cat} in the {room}: {desc}"
    elif desc:
        instruction = f"Find the {cat}: {desc}"
    elif room:
        instruction = f"Find the {cat} in the {room}"
    else:
        instruction = f"Find the {cat}"
    if desc:
        clip_goal = f"Seems like there is {desc} ahead."
    else:
        clip_goal = f"Seems like there is a {cat} ahead."
    return instruction, clip_goal


def convert_scene_level(data, scene_short, level, max_per_scene=None):
    """Returns (episodes, goals_dict, n_skipped) for one scene at one level."""
    scene_id = scene_glb_path(scene_short)
    scene_hash = scene_short.split("-", 1)[1]
    goals_by_id = {g["object_id"]: g for g in data["goals"]}
    regions = data.get("region_annotation", {})

    episodes, goals_dict = [], {}
    skipped = 0
    raw_eps = data.get(LEVELS[level], [])
    if max_per_scene:
        raw_eps = raw_eps[:max_per_scene]

    for ep in raw_eps:
        target_ids = ep.get("target_object_ids") or [ep.get("instance_id")]
        valid_ids = sorted(t for t in target_ids if t in goals_by_id)
        if not valid_ids:
            skipped += 1
            continue

        cat = resolve_category(ep, valid_ids, goals_by_id)
        instruction, clip_goal = build_texts(
            level, ep, cat, valid_ids, goals_by_id, regions)

        goal_object_id = "+".join(valid_ids)
        goals_key = f"{scene_hash}_{goal_object_id}"
        if goals_key not in goals_dict:
            goals_dict[goals_key] = [
                {
                    "position": goals_by_id[t]["position"],
                    "object_id": t,
                    "object_name": t,
                    "object_category": goals_by_id[t].get("object_category", cat),
                    "object_surface_area": None,
                    "view_points": goals_by_id[t].get("view_points", []),
                    "image_goals": None,
                }
                for t in valid_ids
            ]

        min_dist = (ep.get("min_dist") or [{}])[0]
        episodes.append({
            "episode_id": str(ep["episode_id"]),
            "scene_id": scene_id,
            "start_position": ep["start_position"],
            "start_rotation": ep["start_rotation"],
            "info": {
                "geodesic_distance": min_dist.get("geodesic_distance", 0.0),
                "euclidean_distance": min_dist.get("euclidean_distance", 0.0),
                "langmap_episode_id": ep["episode_id"],
                "level": level,
            },
            "goals": [],
            "shortest_paths": None,
            "start_room": None,
            "object_category": cat,
            "goal_object_id": goal_object_id,
            "goal_attributes": {
                "object_category": cat,
                "clip_goal": clip_goal,
                "instruction": instruction,
                "level": level,
            },
        })

    return episodes, goals_dict, skipped


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default=os.path.join(
            os.path.dirname(repo_root), "datasets/data/LangMap_Annotations"),
    )
    ap.add_argument(
        "--dst",
        default=os.path.join(
            os.path.dirname(repo_root), "datasets/data/langmap_vlfm"),
    )
    ap.add_argument("--levels", default="all",
                    help='comma-separated subset of object,room,region,instance')
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--max-per-scene", type=int, default=None)
    args = ap.parse_args()

    levels = list(LEVELS) if args.levels == "all" else [
        l.strip() for l in args.levels.split(",")]
    for l in levels:
        assert l in LEVELS, f"unknown level {l}"

    files = sorted(glob(os.path.join(args.src, "*.json*")))
    by_scene = {}
    for f in files:  # dedupe .json vs .json.gz
        scene_short = scene_short_from_fname(f)
        # Guard against stray upload duplicates like "00832-xxx 2.json" whose
        # name is not a valid <id>-<hash> scene and yields a broken glb path.
        if not re.fullmatch(r"\d{5}-\w+", scene_short):
            print(f"[skip] not a valid scene filename: {os.path.basename(f)}")
            continue
        by_scene.setdefault(scene_short, f)
    if args.scenes:
        by_scene = {k: v for k, v in by_scene.items() if k in set(args.scenes)}

    for level in levels:
        all_cats = set()
        totals = defaultdict(int)
        out_dir = os.path.join(args.dst, f"{level}_level", "val")
        content_dir = os.path.join(out_dir, "content")
        os.makedirs(content_dir, exist_ok=True)

        scene_payloads = {}
        for scene_short in sorted(by_scene):
            data = load_scene(by_scene[scene_short])
            eps, goals, skipped = convert_scene_level(
                data, scene_short, level, args.max_per_scene)
            if not eps:
                continue
            all_cats.update(e["object_category"] for e in eps)
            totals["eps"] += len(eps)
            totals["skipped"] += skipped
            scene_payloads[scene_short] = (eps, goals)

        cat_map = {c: i for i, c in enumerate(sorted(all_cats))}

        for scene_short, (eps, goals) in scene_payloads.items():
            scene_hash = scene_short.split("-", 1)[1]
            payload = {
                "episodes": eps,
                "goals": goals,
                "goals_by_category": goals,
                "category_to_task_category_id": cat_map,
                "category_to_scene_annotation_category_id": cat_map,
            }
            with gzip.open(
                    os.path.join(content_dir, f"{scene_hash}.json.gz"), "wt") as f:
                json.dump(payload, f)

        with gzip.open(os.path.join(out_dir, "val.json.gz"), "wt") as f:
            json.dump({
                "episodes": [],
                "goals": {},
                "category_to_task_category_id": cat_map,
                "category_to_scene_annotation_category_id": cat_map,
            }, f)

        print(f"[{level}] {totals['eps']} eps across "
              f"{len(scene_payloads)} scenes, {len(cat_map)} categories"
              + (f", {totals['skipped']} skipped" if totals["skipped"] else "")
              + f" -> {out_dir}")


if __name__ == "__main__":
    main()
