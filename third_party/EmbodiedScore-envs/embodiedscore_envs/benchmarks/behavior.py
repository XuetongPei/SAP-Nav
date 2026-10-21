"""BEHAVIOR-1K: long-horizon household activities on a two-armed mobile
manipulator in OmniGibson (NVIDIA Isaac Sim). One engine, one line.

* **BEHAVIOR-1K** (Li et al., arXiv:2403.09227 — "BEHAVIOR-1K: A
  Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and
  Realistic Simulation"; a preliminary version at CoRL 2022) is 1 000 activity
  definitions written in BDDL, of which 50 are scored by the **2025 BEHAVIOR
  Challenge** (NeurIPS 2025). This port pins the challenge release,
  ``StanfordVL/BEHAVIOR-1K`` tag ``v3.7.2`` (commit
  ``88454bd04f75dc57c00ab1f1a00bcde1ff505950``, "the post release of the 2025
  BEHAVIOR Challenge at NeurIPS"). ``main`` has since moved to the 2026
  challenge (Python 3.11, Isaac Sim 5.1, ``omnigibson/eval/``); everything
  transcribed here is at v3.7.2 — see INSTALL-behavior.md.

Data: the encrypted asset bundle plus the challenge's task instances, under
``$OMNIGIBSON_DATA_PATH`` (OmniGibson's own root — ``omnigibson/macros.py``
``determine_data_path`` L110-128), NOT under ``EMBODIEDSCORE_DATA_ROOT``. The
loader reads two files of the ``2025-challenge-task-instances`` release and
nothing else; ``behavior_root`` overrides the root for one call.

Robot: **R1 Pro**, the only robot the challenge accepts (``eval.py`` L128
asserts it) — see ``env/sim/behavior/body.py`` for its morphology, cameras and
control rates, all transcribed.

Lines — ONE, ``behavior-1k``. The challenge's leaderboard is a single number:
``score_utils.py`` ``compute_final_q_score`` averages the per-task Q over its
10 instances (L114) and then over all 50 tasks (L122,
``overall_q_score = sum(q_score_avg.values()) / 50``); no per-scene and no
per-group figure is computed anywhere in the release, so splitting the 50
tasks into lines would report something the benchmark never reports. The
B10/B20/B30/B40/B50 comments in ``TASK_NAMES_TO_INDICES`` are a labelling of
the task list, not scored groups.

Episodes: task-major, one per **task instance** — the challenge's own
evaluation unit. For each of the 50 tasks in its own index order
(``learning/utils/eval_utils.py`` ``TASK_NAMES_TO_INDICES`` L185-241), the 10
public-test instances the release names in
``metadata/test_instances.csv`` (``eval.py`` L437-446 reads that row and maps
eval index -> real instance id; ``m.NUM_EVAL_INSTANCES = 10``, ``eval.py``
L50). 50 x 10 = 500 rollouts, which is exactly the volume of an official
submission. Episode id: ``<task>/<instance_id>``.

Splits: ``all`` (all 500) and ``mini`` (the first public-test instance of each
task: 50). A BEHAVIOR episode is expensive — the challenge's own throughput
note (``docs/challenge/evaluation.md``, RTX 4090) is 150-300 s to load a scene
and 20-25 FPS after that — so ``mini`` is one instance per task rather than a
prefix of the task list, keeping all 50 activities represented.

Budgets: BEHAVIOR's budget is per task and comes from its human demonstrations
— ``eval.py`` L146-153, ``max_steps = int(mean human demo length x 2)`` over
that task's 200 demonstrations in the release's ``metadata/episodes.jsonl``.
The loader computes it from that file, so it is the release's number, not a
transcription. It ranges from 4 299 (``hanging_pictures``: 4 780;
``turning_on_radio``: 4 299) to 52 120 (``assembling_gift_baskets``) control
ticks at 30 Hz, i.e. 2.4 to 29 minutes of simulated time.

Metrics: ``BehaviorMetrics`` — the challenge's own set (``env/metrics.py``
documents each key's source): ``q_score`` (the ranking metric), ``success``,
and the efficiency terms it normalises by the human average of that task.

Variants — the same deliberate bend as the other manipulation lines:

* ``behavior-1k-upstream`` is BEHAVIOR as the challenge's own evaluator runs
  it: ``BehaviorEnv``, one 30 Hz control tick per step, the fixed 23-D action
  (absolute joint angles for the torso and both arms, a base velocity, a
  finger target per gripper), the 224² three-camera rig of
  ``RGBLowResWrapper`` (``bodies.BEHAVIOR``), the task's own step budget as
  the tick cap, and done on success.
* ``behavior-1k`` (standard) is the macro protocol for agents that plan in
  poses: ``BehaviorPoseEnv``, one step = one base move, one absolute
  end-effector target per arm, or one gripper hold, driven through the
  ``InverseKinematicsController`` in ``absolute_pose`` mode the challenge
  documents as a permitted action-space substitution
  (``docs/challenge/evaluation.md`` § "Configure Robot Action Space"); a 256²
  rig (``bodies.BEHAVIOR_STANDARD``) and a budget of ``MACRO_STEPS`` macro
  moves (gym TimeLimit). The tick cap stays the challenge's own per-task
  budget on both variants (``TICK_SCALE`` is 1.0): only the unit the AGENT's
  budget counts bends, never the simulator's.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from .env import BehaviorMetrics, Benchmark, BehaviorSceneRef, Episode, ManipGoal
from .presets import bodies

# learning/utils/eval_utils.py TASK_NAMES_TO_INDICES L185-241 — the 50 tasks the 2025
# challenge scores, in the index order the release's metadata rows are keyed by. The scene
# each is sampled into is joylo/sampled_task/available_tasks.yaml <task>/0/scene_model.
TASKS: tuple[tuple[str, str], ...] = (
    # B10
    ("turning_on_radio", "house_double_floor_lower"),
    ("picking_up_trash", "house_double_floor_lower"),
    ("putting_away_Halloween_decorations", "house_double_floor_lower"),
    ("cleaning_up_plates_and_food", "house_double_floor_lower"),
    ("can_meat", "house_single_floor"),
    ("setting_mousetraps", "house_double_floor_upper"),
    ("hiding_Easter_eggs", "house_double_floor_lower"),
    ("picking_up_toys", "house_single_floor"),
    ("rearranging_kitchen_furniture", "house_double_floor_lower"),
    ("putting_up_Christmas_decorations_inside", "house_single_floor"),
    # B20
    ("set_up_a_coffee_station_in_your_kitchen", "house_single_floor"),
    ("putting_dishes_away_after_cleaning", "house_single_floor"),
    ("preparing_lunch_box", "house_single_floor"),
    ("loading_the_car", "house_double_floor_lower"),
    ("carrying_in_groceries", "house_double_floor_lower"),
    ("bringing_in_wood", "house_double_floor_lower"),
    ("moving_boxes_to_storage", "house_double_floor_lower"),
    ("bringing_water", "house_single_floor"),
    ("tidying_bedroom", "house_single_floor"),
    ("outfit_a_basic_toolbox", "house_single_floor"),
    # B30
    ("sorting_vegetables", "house_single_floor"),
    ("collecting_childrens_toys", "house_single_floor"),
    ("putting_shoes_on_rack", "house_double_floor_lower"),
    ("boxing_books_up_for_storage", "house_double_floor_upper"),
    ("storing_food", "house_single_floor"),
    ("clearing_food_from_table_into_fridge", "house_double_floor_lower"),
    ("assembling_gift_baskets", "house_double_floor_lower"),
    ("sorting_household_items", "house_single_floor"),
    ("getting_organized_for_work", "house_double_floor_upper"),
    ("clean_up_your_desk", "house_single_floor"),
    # B40
    ("setting_the_fire", "house_double_floor_lower"),
    ("clean_boxing_gloves", "house_single_floor"),
    ("wash_a_baseball_cap", "house_single_floor"),
    ("wash_dog_toys", "house_single_floor"),
    ("hanging_pictures", "house_double_floor_lower"),
    ("attach_a_camera_to_a_tripod", "house_double_floor_upper"),
    ("clean_a_patio", "house_double_floor_lower"),
    ("clean_a_trumpet", "house_double_floor_upper"),
    ("spraying_for_bugs", "house_double_floor_lower"),
    ("spraying_fruit_trees", "house_double_floor_lower"),
    # B50
    ("make_microwave_popcorn", "house_double_floor_lower"),
    ("cook_cabbage", "house_single_floor"),
    ("chop_an_onion", "house_double_floor_lower"),
    ("slicing_vegetables", "house_single_floor"),
    ("chopping_wood", "house_double_floor_lower"),
    ("cook_hot_dogs", "house_single_floor"),
    ("cook_bacon", "house_single_floor"),
    ("freeze_pies", "house_single_floor"),
    ("canning_food", "house_single_floor"),
    ("make_pizza", "house_double_floor_lower"),
)
TASK_INDEX = {task: i for i, (task, _scene) in enumerate(TASKS)}
SCENE_OF = dict(TASKS)

LINE = "behavior-1k"
RELEASE = "2025-challenge"
INSTANCE_DIR = "2025-challenge-task-instances"          # asset_utils.py download_2025_challenge_task_instances
DEFINITION_ID = 0                                       # every challenge task ships problem0.bddl only
EVAL_INSTANCES = 10                                     # eval.py L50, m.NUM_EVAL_INSTANCES
TRAIN_INSTANCES = 200                                   # eval.py L49, the demonstrated instances
CSV_INSTANCES = 20                                      # the row holds 20; the public leaderboard scores the first 10
BUDGET_MULTIPLE = 2                                     # eval.py L146-153: max_steps = 2 x mean human demo length
MINI_INSTANCES = 1                                      # the mini split: one public-test instance per task
MACRO_STEPS = 150                                       # standard protocol: macro moves per episode (2026-09-10)
TICK_SCALE = 1.0                                        # both variants keep BEHAVIOR's own per-task tick cap
SPLITS = ("all", "mini")


def data_path(explicit: str | os.PathLike | None = None) -> Path:
    """OmniGibson's data root: the loader argument, else ``OMNIGIBSON_DATA_PATH``
    (``omnigibson/macros.py`` ``determine_data_path`` L110-128 reads the same
    variable). The BEHAVIOR data does not live under ``EMBODIEDSCORE_DATA_ROOT``
    — OmniGibson resolves its own assets by this root, and the two must agree."""
    root = explicit or os.environ.get("OMNIGIBSON_DATA_PATH")
    if not root:
        raise ValueError("behavior: pass behavior_root=... or set OMNIGIBSON_DATA_PATH (INSTALL-behavior.md § Data)")
    return Path(os.path.expanduser(str(root)))


def metadata_dir(root: str | os.PathLike | None = None) -> Path:
    return data_path(root) / INSTANCE_DIR / "metadata"


def public_test_instances(root: str | os.PathLike | None = None) -> dict[str, list[int]]:
    """Each task's row of the release's ``metadata/test_instances.csv`` — the
    file ``eval.py`` L437-446 reads, indexed by ``TASK_NAMES_TO_INDICES``, whose
    third column is a comma-separated instance list. The row holds
    ``CSV_INSTANCES`` (20) ids and the evaluator scores the FIRST
    ``EVAL_INSTANCES`` of them: ``instances_to_run`` defaults to
    ``range(m.NUM_EVAL_INSTANCES)`` (``eval.py`` L430-446) and indexes into the
    row. ``docs/challenge/evaluation.md`` says the same in prose — the first 10
    are the public leaderboard's, the other 10 are the participant's own dev
    set. The whole row is returned; the loader takes the prefix."""
    path = metadata_dir(root) / "test_instances.csv"
    if not path.exists():
        raise FileNotFoundError(f"behavior: no {path} — the {INSTANCE_DIR} release is missing "
                                "(INSTALL-behavior.md § Data)")
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))[1:]     # ["Task ID", "Task", "Public Test Instance IDs"]
    out: dict[str, list[int]] = {}
    for task, index in TASK_INDEX.items():
        row = rows[index]
        if row[1].strip() != task:
            raise ValueError(f"behavior: {path} row {index} names {row[1]!r}, the declaration names {task!r}")
        ids = [int(v) for v in row[2].strip().split(",")]
        if len(ids) < EVAL_INSTANCES:
            raise ValueError(f"behavior: {path} row {index} names {len(ids)} instances of {task}, "
                             f"fewer than the {EVAL_INSTANCES} the leaderboard scores")
        out[task] = ids
    return out


def human_stats(root: str | os.PathLike | None = None) -> dict[str, dict[str, float]]:
    """Per task, the mean of its 200 human demonstrations over the four fields
    the challenge's metrics normalise by — ``eval.py`` L115-123 reads the same
    ``metadata/episodes.jsonl`` and selects a task's demos by
    ``episode["episode_index"] // 1e4 == task_idx``."""
    path = metadata_dir(root) / "episodes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"behavior: no {path} — the {INSTANCE_DIR} release is missing "
                                "(INSTALL-behavior.md § Data)")
    fields = ("length", "distance_traveled", "left_eef_displacement", "right_eef_displacement")
    sums: dict[int, dict[str, float]] = {i: dict.fromkeys(fields, 0.0) for i in range(len(TASKS))}
    counts: dict[int, int] = dict.fromkeys(range(len(TASKS)), 0)
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            episode = json.loads(line)
            index = int(episode["episode_index"] // 1e4)
            if index not in sums:
                continue
            for field in fields:
                sums[index][field] += float(episode[field])
            counts[index] += 1
    out: dict[str, dict[str, float]] = {}
    for task, index in TASK_INDEX.items():
        n = counts[index]
        if not n:
            raise ValueError(f"behavior: {path} holds no demonstrations for {task} (task index {index})")
        out[task] = {field: sums[index][field] / n for field in fields}
    return out


def goal_predicates(task: str, definition_id: int = DEFINITION_ID) -> tuple[tuple[str, ...], ...]:
    """The activity's BDDL goal conditions, one entry per top-level predicate of
    ``bddl/activity_definitions/<task>/problem<id>.bddl``, flattened to strings
    (``bddl.parsing.parse_problem`` — a pure parse, no simulator and no
    backend). Falls back to naming the check when ``bddl`` is not installed,
    as the RoboTwin lines do."""
    try:
        from bddl.parsing import parse_problem
    except ImportError:
        return (("behavior_bddl_goal", task),)
    _name, _objects, _init, goal = parse_problem(task, definition_id, "omnigibson")
    return tuple(tuple(_flatten(condition)) for condition in goal)


def _flatten(node: Any) -> list[str]:
    if isinstance(node, (list, tuple)):
        return [token for child in node for token in _flatten(child)]
    return [str(node)]


def instruction_of(task: str) -> str:
    """``turning_on_radio`` -> ``turning on radio``: the placeholder the
    declaration carries. BDDL's own natural-language goal for the activity is
    only known once the task is loaded — the env body puts it in
    ``info["language"]``."""
    return task.replace("_", " ")


def load_episodes(split: str, data_root: str | os.PathLike | None = None,
                  scene_root: str | os.PathLike | None = None, behavior_root: str | os.PathLike | None = None,
                  **_unused: Any) -> list[Episode]:
    """``data_root`` / ``scene_root`` are accepted for the shared signature and
    ignored: BEHAVIOR resolves its assets by OmniGibson's own root, which
    ``behavior_root`` or ``OMNIGIBSON_DATA_PATH`` names."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    instances = public_test_instances(behavior_root)
    stats = human_stats(behavior_root)
    per_task = EVAL_INSTANCES if split == "all" else MINI_INSTANCES
    out: list[Episode] = []
    for task, scene_model in TASKS:
        demo = stats[task]
        max_steps = int(demo["length"] * BUDGET_MULTIPLE)      # eval.py L146-153
        human_distance = {"base": demo["distance_traveled"], "left": demo["left_eef_displacement"],
                          "right": demo["right_eef_displacement"]}
        for slot in range(per_task):
            instance_id = int(instances[task][slot])   # the row's prefix: eval index -> real instance id
            scene = BehaviorSceneRef(task=task, scene_model=scene_model, instance_id=instance_id,
                                     definition_id=DEFINITION_ID, release=RELEASE)
            out.append(Episode(
                index=len(out), episode_id=f"{task}/{instance_id}", scene=scene,
                start_position=None, start_rotation=None,
                goal=ManipGoal(goal_predicates(task)),
                instruction=instruction_of(task),
                info={"line": LINE, "release": RELEASE, "task": task, "task_index": TASK_INDEX[task],
                      "scene_model": scene_model, "instance_id": instance_id, "eval_slot": slot,
                      "max_steps": max_steps, "human_steps": demo["length"], "human_distance": human_distance},
            ))
    return out


def _decl(upstream: bool) -> Benchmark:
    body = bodies.BEHAVIOR if upstream else bodies.BEHAVIOR_STANDARD
    return Benchmark(
        name=LINE + ("-upstream" if upstream else ""),
        gym_id="EmbodiedScore/BEHAVIOR-1K-" + ("Upstream-v0" if upstream else "v0"),
        body=body,
        actions=(), splits=SPLITS,
        episodes=lambda split, data_root=None, scene_root=None, **kw: load_episodes(split, data_root,
                                                                                     scene_root, **kw),
        # simulator_time = ticks x the rendering dt (task_metric.gather_results); the body fixes that rate.
        metrics=lambda env, **o: BehaviorMetrics(env, render_hz=float(body.render_freq), **o),
        depth=None,
        # The tick cap is the episode's own budget (BEHAVIOR's is per task, from its human demos), so a
        # static TimeLimit only bounds the per-tick variant: the largest budget of the 50 tasks, which is
        # assembling_gift_baskets at 52 120 ticks. The macro variant's TimeLimit is its macro-step budget.
        max_episode_steps=None if upstream else MACRO_STEPS,
        ticks=None, tick_scale=TICK_SCALE,
        engine="behavior", macro=not upstream, variant="upstream" if upstream else "standard",
        description="BEHAVIOR-1K 2025 Challenge, 50 household activities on an R1 Pro " + (
            "as the challenge's evaluator runs it: per-tick 23-D actions on the 224² three-camera rig"
            if upstream else "on the macro pose + base protocol and the 256² rig"),
    )


BENCHMARKS = tuple(_decl(up) for up in (False, True))
