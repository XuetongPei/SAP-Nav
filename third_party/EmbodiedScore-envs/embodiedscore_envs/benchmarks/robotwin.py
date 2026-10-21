"""RoboTwin 2.0: the bimanual manipulation benchmark (Chen et al. 2026,
arXiv:2506.18088, ICML 2026; RoboTwin 1.0 is arXiv:2504.13059, the early
version arXiv:2409.02920) on SAPIEN 3 — 50 tabletop tasks, a dual-arm robot
(``aloha-agilex`` by default), a scripted expert per task, domain
randomisation, and a per-task success predicate written in the task's own
module.

Repo: github.com/RoboTwin-Platform/RoboTwin at ``96c1fea`` (branch ``main``);
every file and line cited below is that commit's.

Data: nothing under ``EMBODIEDSCORE_DATA_ROOT``. The tasks, the configs and
the language ship inside the checkout, the meshes and embodiments beside it
(``EMBODIEDSCORE_ROBOTWIN_ROOT``, INSTALL-robotwin.md). The pre-collected
trajectories are not read — evaluation needs none.

Lines (one per shipped task config):

    robotwin-clean        50 tasks, no domain randomisation   (env_cfg/task_config/demo_clean.yml)
    robotwin-randomized   50 tasks, RoboTwin's randomisation  (env_cfg/task_config/demo_randomized.yml)

The task config is the axis RoboTwin's own evaluation is parameterised on
(``--task-config demo_clean | demo_randomized``) and the axis RoboTwin 2.0
reports clean and randomised success separately along; the 50 tasks are
episode families inside a line, because RoboTwin's own eval config lists all
50 flat (``env_cfg/eval/all_tasks.yml``) and its leaderboard is a per-task
success rate with an average over them. There is no task-family grouping in
the repo to mirror.

Episodes: task-major, ``episode_id`` = ``<task>/<seed>``. A RoboTwin episode
is a task plus a scene seed — the seed places every actor
(``envs/_base_task.py:58``). RoboTwin's evaluator starts at
``st_seed = 100000 * (1 + seed)`` with ``--seed`` defaulting to 0, i.e.
100000, and walks upward (``scripts/eval_policy_xpolicylab.py:305``), taking
the first ``test_num`` (default 100) seeds whose scene settles and skipping
the rest (``:633``) — how many are skipped is very task-dependent
(``beat_block_hammer`` loses none of its first six, ``adjust_bottle`` loses
half).

That walk draws from one shared queue, so upstream's episode ``k`` is
whichever seed the queue happened to hand a worker: not reproducible on its
own. This port partitions the same seed range instead. Episode ``k`` of a
task owns the window ``[SEED_BASE + k * SEED_STRIDE, + SEED_STRIDE)`` and
takes the first seed in it whose scene settles; ``SEED_STRIDE`` is
RoboTwin's own per-episode seed allowance (``max_seed_attempts`` defaults to
``test_num * 50``, ``scripts/eval_policy_xpolicylab.py:370``, i.e. 50 seeds
per episode). ``episode_id`` names the window start; ``info["seed"]`` after
a reset names the seed actually used. This is the port's one deviation from
RoboTwin's episode identity, and it is what makes an episode index mean the
same scene on every run.

Splits: ``all`` (``SEEDS_PER_TASK`` = RoboTwin's own ``test_num``: 100
windows x 50 tasks = 5000 episodes) and ``mini`` (``MINI_SEEDS`` = 2 windows
of every task: 100 episodes, the size of the LIBERO lines' mini split, still
covering all 50 tasks).

The instruction is the task's ``full_description``
(``description/task_instruction/<task>.json``); the goal is
``ManipGoal((("check_success", <task>),))`` — RoboTwin's goal is python, not
a declarative predicate list, so the goal names the check rather than
restating it. Success is the task's own ``check_success``
(``ManipMetrics``).

Variants — the one place the package's rule "task semantics do not change
between variants" is bent, deliberately, exactly as on the LIBERO lines:

* ``<line>-upstream`` is RoboTwin as its evaluator runs it
  (``scripts/eval_policy_xpolicylab.py:696-716``): ``RobotwinEnv``, one
  ``take_action`` call per step with RoboTwin's own flat joint vector
  (``action_type="qpos"``, its default at ``envs/_base_task.py:1486``), the
  D435 rig of ``demo_clean.yml`` / ``demo_randomized.yml``
  (``bodies.ROBOTWIN`` / ``ROBOTWIN_RANDOMIZED``), and the task's own step
  cap from ``env_cfg/task_config/_eval_step_limit.yml`` (``STEP_LIMIT``,
  400-1700 depending on the task).
* ``<line>`` (standard) is the macro protocol for agents that plan in poses:
  ``RobotwinPoseEnv``, one step = an absolute end-effector target per arm (or
  a gripper hold), the same D435 rig upscaled to ``Large_D435``
  (``bodies.ROBOTWIN_STANDARD`` / ``ROBOTWIN_STANDARD_RANDOMIZED``), a budget
  of ``MACRO_STEPS`` macro moves (gym TimeLimit) and the task's own step cap
  as the tick guard underneath. The budget unit therefore differs (macro
  moves vs RoboTwin actions): a language agent cannot emit a 14-vector of
  joint targets, and the LIBERO lines and the Isaac polar protocol are the
  same kind of concession. Unlike LIBERO, one macro move costs about ONE
  upstream tick rather than a hundred — a RoboTwin action is already a
  planned motion — so no multiplier is applied to the guard.
"""

from __future__ import annotations

import os
from typing import Any

from .env import Benchmark, Episode, ManipGoal, ManipMetrics, RobotwinSceneRef
from .env.sim.robotwin import task_instruction
from .presets import bodies

# env_cfg/task_config/_eval_step_limit.yml, verbatim — the cap RoboTwin puts on
# take_action calls per episode (read at envs/_base_task.py:142-148, enforced at
# envs/_base_task.py:1487). Its key set is exactly env_cfg/eval/all_tasks.yml's.
STEP_LIMIT = {
    "adjust_bottle": 400,
    "beat_block_hammer": 400,
    "blocks_ranking_rgb": 1200,
    "blocks_ranking_size": 1200,
    "click_alarmclock": 400,
    "click_bell": 400,
    "dump_bin_bigbin": 600,
    "grab_roller": 400,
    "handover_block": 800,
    "handover_mic": 600,
    "hanging_mug": 900,
    "lift_pot": 400,
    "move_can_pot": 400,
    "move_pillbottle_pad": 400,
    "move_playingcard_away": 400,
    "move_stapler_pad": 400,
    "open_laptop": 700,
    "open_microwave": 1500,
    "pick_diverse_bottles": 400,
    "pick_dual_bottles": 400,
    "place_a2b_left": 400,
    "place_a2b_right": 400,
    "place_bread_basket": 700,
    "place_bread_skillet": 500,
    "place_burger_fries": 500,
    "place_can_basket": 700,
    "place_cans_plasticbox": 800,
    "place_container_plate": 400,
    "place_dual_shoes": 600,
    "place_empty_cup": 500,
    "place_fan": 400,
    "place_mouse_pad": 400,
    "place_object_basket": 700,
    "place_object_scale": 400,
    "place_object_stand": 400,
    "place_phone_stand": 400,
    "place_shoe": 500,
    "press_stapler": 400,
    "put_bottles_dustbin": 1700,
    "put_object_cabinet": 700,
    "rotate_qrcode": 400,
    "scan_object": 500,
    "shake_bottle": 700,
    "shake_bottle_horizontally": 700,
    "stack_blocks_three": 1200,
    "stack_blocks_two": 800,
    "stack_bowls_three": 1200,
    "stack_bowls_two": 900,
    "stamp_seal": 400,
    "turn_switch": 400,
}
TASKS = tuple(STEP_LIMIT)                    # env_cfg/eval/all_tasks.yml order (alphabetical, as shipped)

CONFIGS = {                                  # line word -> the env_cfg/task_config/ file it transcribes
    "clean": "demo_clean",
    "randomized": "demo_randomized",
}
BODIES = {                                   # line word -> (standard body, upstream body)
    "clean": (bodies.ROBOTWIN_STANDARD, bodies.ROBOTWIN),
    "randomized": (bodies.ROBOTWIN_STANDARD_RANDOMIZED, bodies.ROBOTWIN_RANDOMIZED),
}
SEED_BASE = 100000           # scripts/eval_policy_xpolicylab.py:305 — st_seed = 100000 * (1 + 0)
SEEDS_PER_TASK = 100         # ... :306 — test_num, RoboTwin's own default episode count per task
SEED_STRIDE = 50             # ... :370 — max_seed_attempts / test_num: RoboTwin's own seed allowance per episode
MINI_SEEDS = 2               # the mini split: 2 seeds x 50 tasks = 100 episodes. Decided 2026-09-10.
MACRO_STEPS = 100            # standard protocol: macro moves per episode, as on the LIBERO lines. Decided 2026-09-10.
SPLITS = ("all", "mini")
EMBODIMENT = "aloha-agilex"  # env_cfg/task_config/demo_clean.yml `embodiment: [aloha-agilex]`


def step_budget(world: Any, episode: Episode) -> int:
    """The episode's cap on RoboTwin actions — the task's own number."""
    return int(episode.info["step_limit"])


def load_episodes(config: str, split: str, data_root: str | os.PathLike | None = None,
                  scene_root: str | os.PathLike | None = None, robotwin_root: str | os.PathLike | None = None,
                  embodiment: str = EMBODIMENT, **_unused: Any) -> list[Episode]:
    """``data_root`` / ``scene_root`` are accepted for the shared signature and
    ignored; the RoboTwin checkout is ``robotwin_root`` or
    ``EMBODIEDSCORE_ROBOTWIN_ROOT``."""
    if config not in CONFIGS.values():
        raise ValueError(f"config must be one of {sorted(CONFIGS.values())}, got {config!r}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    n_seeds = MINI_SEEDS if split == "mini" else SEEDS_PER_TASK
    out: list[Episode] = []
    for task in TASKS:
        scene = RobotwinSceneRef(task=task, embodiment=embodiment, task_config=config,
                                 step_limit=STEP_LIMIT[task])
        goal = ManipGoal((("check_success", task),))
        instruction = task_instruction(scene, robotwin_root)
        for k in range(n_seeds):
            seed = SEED_BASE + k * SEED_STRIDE
            out.append(Episode(
                index=len(out), episode_id=f"{task}/{seed}", scene=scene,
                start_position=None, start_rotation=None, goal=goal,
                instruction=instruction,
                info={"task": task, "task_config": config, "embodiment": embodiment, "seed": seed,
                      "seed_window": SEED_STRIDE, "seed_index": k, "step_limit": STEP_LIMIT[task]},
            ))
    return out


def _decl(word: str, upstream: bool) -> Benchmark:
    config = CONFIGS[word]
    gym_id = f"EmbodiedScore/RoboTwin-{word.capitalize()}-v0"
    standard_body, upstream_body = BODIES[word]
    return Benchmark(
        name=f"robotwin-{word}" + ("-upstream" if upstream else ""),
        gym_id=gym_id.replace("-v0", "-Upstream-v0") if upstream else gym_id,
        body=upstream_body if upstream else standard_body,
        actions=(), splits=SPLITS,
        episodes=lambda split, data_root=None, scene_root=None, **kw: load_episodes(config, split, data_root,
                                                                                    scene_root, **kw),
        metrics=lambda env, **o: ManipMetrics(env),
        depth=None,
        max_episode_steps=None if upstream else MACRO_STEPS,
        budget=step_budget, truncate_at_budget=upstream,
        ticks=None,                       # per episode: the task's own cap, through ``budget``
        engine="robotwin", macro=not upstream, variant="upstream" if upstream else "standard",
        description=f"RoboTwin 2.0 {config} " + ("as its evaluator runs it: RoboTwin's own joint-target action "
                                                 "on the D435 rig"
                                                 if upstream else
                                                 "on the bimanual macro pose protocol and the Large_D435 rig"),
    )


BENCHMARKS = tuple(_decl(w, up) for w in CONFIGS for up in (False, True))
