"""VLNverse: the fine-grained and coarse-grained instruction lines, on Isaac Sim.

Data (under ``EMBODIEDSCORE_DATA_ROOT/vlnverse``):
    raw_data/final_splits/{fine,coarse}_{train,val,val_unseen,test}.json[.gz]
        {"episodes": [{episode_id, trajectory_id, scan, scene_id, start_position,
                       start_rotation (w, x, y, z), instruction {instruction_text},
                       reference_path, goals {position, radius}, info {geodesic_distance}}]}
        test carries only episode_id / trajectory_id / scan / scene_id / start_* /
        instruction (ground truth withheld). coarse instruction_text is a dict
        {formal, natural, casual}; ``instruction_type`` picks one (the evaluator's
        default is formal), the others stay in ``info["instruction_variants"]``.
    ``challenge`` = the 150 test episodes per granularity listed in
    ``challenge_subset.txt`` next to this file (the leaderboard split).
Scenes: ``EMBODIEDSCORE_SCENE_ROOT/vlnverse/<scan>/`` — the USD stage with its
``Meshes/`` and ``Materials/`` (Eyz/VLNVerse_scene) plus ``freemap.npy``, the
occupancy grid the kinematics move on (not on Hugging Face at the time of writing).

Poses stay in the split files' own frame — Isaac world, z up, ``(w, x, y, z)``
quaternions — and so do the facts and metrics.

The evaluator's loader filters — ``filter_same_trajectory`` (keep the first
episode of each trajectory_id) and ``filter_stairs`` (drop episodes whose
instruction mentions stairs and whose reference path climbs >= 0.3 m, or whose
path has a > 0.3 m height step) — are available as loader keyword arguments,
both off by default; the evaluator's shipped configs also ran with
``filter_stairs=False``. Episodes are returned in file order.

Variants: ``vlnverse-fine`` / ``vlnverse-coarse`` run on ``bodies.VLNVERSE_STANDARD``
(the STANDARD numbers on the Isaac camera: 0.25 m / 15°, 1.25 m, RGB 512² +
depth 256², depth 0–10 m normalised) with the discrete table 0–3 (no LOOK —
the worker renders yaw-only poses); ``-upstream`` runs the zero-shot line's
rig (``bodies.VLNVERSE``: RGB-D 1024² at 1.2 m, depth left in metres) behind
that line's macro action, ``Box[angle_rad, distance_m, elevation_deg]``
(``IsaacPolarEnv``). Task semantics in both: point goal at the last
reference-path point, success radius 3.0 m, budget 500 steps (the evaluator's
``max_step``), VLNverse's own metric formulas (``VLNVerseMetrics``).
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

from .env import Benchmark, Episode, IsaacSceneRef, PointGoal, VLNVerseMetrics, data_root, scene_root
from .presets import actions, bodies, depth

GRANULARITIES = ("fine", "coarse")
SPLITS = ("train", "val", "val_unseen", "test", "challenge")
SUCCESS_DISTANCE = 3.0
MAX_EPISODE_STEPS = 500
_SUBSET_FILE = Path(__file__).with_name("challenge_subset.txt")


def _split_file(granularity: str, split: str, root: str | os.PathLike | None) -> str:
    d = data_root(root) / "vlnverse" / "raw_data" / "final_splits"
    base = d / f"{granularity}_{split}.json"
    for p in (base, base.with_suffix(".json.gz")):
        if p.is_file():
            return str(p)
    raise FileNotFoundError(f"no episode file for {granularity}/{split} under {d} (tried {base}[.gz])")


def _read(path: str) -> Any:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def challenge_ids(granularity: str) -> set[str]:
    """Episode ids of the challenge split (``challenge_subset.txt``:
    ``granularity  trajectory_id  episode_id  scene_id``)."""
    out: set[str] = set()
    with open(_SUBSET_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            g, _tid, eid, _sid = line.rstrip("\n").split("\t")
            if g == granularity:
                out.add(eid)
    return out


def _instruction(raw: Any, instruction_type: str) -> tuple[str, dict[str, str] | None]:
    text = raw.get("instruction_text") if isinstance(raw, dict) else raw
    if isinstance(text, dict):
        if instruction_type not in text:
            raise KeyError(f"instruction_type {instruction_type!r} not in {sorted(text)}")
        return str(text[instruction_type]), {k: str(v) for k, v in text.items()}
    return str(text or ""), None


def has_stairs(item: dict[str, Any], height_threshold: float = 0.3) -> bool:
    """The evaluator's ``has_stairs``: 'stair' in the instruction and the reference path climbs."""
    text = item["instruction"]["instruction_text"]
    if isinstance(text, dict):
        text = " ".join(str(v) for v in text.values())
    if "stair" not in text:
        return False
    path = item["reference_path"]
    latest = path[0][-1]
    for p in path[1:]:
        if abs(p[-1] - latest) >= height_threshold:
            return True
        latest = p[-1]
    return False


def different_height(item: dict[str, Any]) -> bool:
    """The evaluator's ``different_height``: a > 0.3 m step between consecutive reference points."""
    path = item["reference_path"]
    return any(abs(path[i + 1][2] - path[i][2]) > 0.3 for i in range(len(path) - 1))


def load_episodes(granularity: str, split: str, data_root_: str | os.PathLike | None = None,
                  scene_root_: str | os.PathLike | None = None, instruction_type: str = "formal",
                  filter_same_trajectory: bool = False, filter_stairs: bool = False) -> list[Episode]:
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES}, got {granularity!r}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    raw = _read(_split_file(granularity, "test" if split == "challenge" else split, data_root_))["episodes"]
    if split == "challenge":
        keep = challenge_ids(granularity)
        raw = [e for e in raw if e["episode_id"] in keep]
    if filter_same_trajectory:
        seen: set[str] = set()
        kept = []
        for e in raw:
            if e["trajectory_id"] in seen:
                continue
            seen.add(e["trajectory_id"])
            kept.append(e)
        raw = kept
    if filter_stairs:
        raw = [e for e in raw if not e.get("reference_path") or not (has_stairs(e) or different_height(e))]
    sroot = scene_root(scene_root_) / "vlnverse"
    out: list[Episode] = []
    for e in raw:
        text, variants = _instruction(e.get("instruction"), instruction_type)
        goal = e.get("goals")
        ref = e.get("reference_path")
        info = {"granularity": granularity, "trajectory_id": str(e.get("trajectory_id", "")),
                "scene_id": e.get("scene_id"), "scan": e["scan"],
                "geodesic_distance": (e.get("info") or {}).get("geodesic_distance"),
                "instruction_variants": variants}
        out.append(Episode(
            index=len(out), episode_id=str(e["episode_id"]),
            scene=IsaacSceneRef(scan=e["scan"], scene_dir=str(sroot / e["scan"])),
            start_position=tuple(float(x) for x in e["start_position"]),
            start_rotation=tuple(float(x) for x in e["start_rotation"]),
            goal=PointGoal(tuple(float(x) for x in goal["position"]),
                           float(goal["radius"]) if goal.get("radius") is not None else SUCCESS_DISTANCE)
            if isinstance(goal, dict) and "position" in goal else None,
            instruction=text,
            reference_path=tuple(tuple(float(x) for x in p) for p in ref) if ref else None,
            info=info,
        ))
    return out


def _decl(granularity: str, upstream: bool) -> Benchmark:
    gym_id = f"EmbodiedScore/VLNverse-{granularity.capitalize()}-v0"
    return Benchmark(
        name=f"vlnverse-{granularity}" + ("-upstream" if upstream else ""),
        gym_id=gym_id.replace("-v0", "-Upstream-v0") if upstream else gym_id,
        body=bodies.VLNVERSE if upstream else bodies.VLNVERSE_STANDARD,
        actions=actions.NAV, splits=SPLITS,
        episodes=lambda split, data_root=None, scene_root=None, **kw: load_episodes(granularity, split, data_root, scene_root, **kw),
        metrics=lambda env, **o: VLNVerseMetrics(env, success_distance=o.get("success_distance", SUCCESS_DISTANCE)),
        depth=None if upstream else depth.STANDARD, max_episode_steps=MAX_EPISODE_STEPS,
        engine="isaac", polar=upstream, variant="upstream" if upstream else "standard",
        description=f"VLNverse {granularity}-grained instructions " + (
            "on the zero-shot line's rig and macro action" if upstream else "on the STANDARD body"),
    )


BENCHMARKS = (_decl("fine", upstream=False), _decl("fine", upstream=True),
              _decl("coarse", upstream=False), _decl("coarse", upstream=True))
