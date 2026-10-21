"""ActiveSpin/vlfm's category and full-sentence LangMap goal interface.

The original converter functions are vendored unchanged. Only their textual
outputs cross the policy boundary; target IDs and geometry stay in evaluation.
"""
import gzip
import json
import os
from functools import lru_cache
from pathlib import Path

from third_party.activespin.convert_langmap_to_objectnav import build_texts, resolve_category

ACTIVESPIN_COMMIT = "2a4ca1fe68f15581b1706bce8690d3fe0c500df8"


@lru_cache(maxsize=2)
def _scene_records(annotation_root, scene):
    root = Path(annotation_root)
    plain = root / (scene + ".json")
    path = plain if plain.exists() else root / (scene + ".json.gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        data = json.load(stream)
    records = {
        (level, str(ep["episode_id"])): ep
        for level in ("object", "room", "region", "instance")
        for ep in data.get(f"episodes_by_{level}_level", [])
    }
    return records, {g["object_id"]: g for g in data["goals"]}, data.get("region_annotation", {})


def goal_attributes(episode, level, annotation_root=None):
    if annotation_root is None:
        annotation_root = Path(os.environ["EMBODIEDSCORE_DATA_ROOT"]) / "langmap/annotations"
    records, goals, regions = _scene_records(str(annotation_root), episode.info["langmap_scene"])
    raw = records[(level, str(episode.info["langmap_episode_id"]))]
    target_ids = raw.get("target_object_ids") or [raw.get("instance_id")]
    valid_ids = sorted(t for t in target_ids if t in goals)
    if not valid_ids:
        raise ValueError(f"No resolvable targets for {episode.episode_id}")
    category = resolve_category(raw, valid_ids, goals)
    instruction, clip_goal = build_texts(level, raw, category, valid_ids, goals, regions)
    return {"object_category": category, "clip_goal": clip_goal,
            "instruction": instruction, "level": level}


def apply_goal_attributes(policy, attributes):
    """Same three assignments as ActiveSpin's LangMapITMPolicyV2.act()."""
    category = attributes["object_category"]
    policy._non_coco_caption = category + " ."
    policy._text_prompt = attributes["clip_goal"]
    return category
