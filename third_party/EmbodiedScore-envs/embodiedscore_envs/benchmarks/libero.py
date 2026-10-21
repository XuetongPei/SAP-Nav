"""LIBERO: the five manipulation suites (Liu et al. 2023, arXiv:2306.03310) on
robosuite 1.4 / MuJoCo — a Franka Panda on a table, one BDDL task per
episode family, 50 init states per task.

Data: nothing under ``EMBODIEDSCORE_DATA_ROOT`` — the BDDL files and the init
states ship inside the ``libero`` package (``libero.libero.get_libero_path``).
The HDF5 demonstrations are not read (evaluation needs none).

Lines (one per suite):

    libero-spatial   10 tasks   same objects, layouts differ — spatial language
    libero-object    10 tasks   same layout, objects differ
    libero-goal      10 tasks   same scene, goals differ
    libero-10        10 tasks   the long-horizon pairs (LIBERO-LONG)
    libero-90        90 tasks   the lifelong pretraining suite

Episodes: task-major, ``episode_id`` = ``<suite>/<task_id>/<init_state>``.
Splits: ``all`` (every init state: 500 episodes, 4500 on libero-90) and
``mini`` (init states 0-9 of every task: 100 / 900). The instruction is the
task's language; the goal is the BDDL goal state (``ManipGoal``); success is
the simulator's own predicate check (``ManipMetrics``).

Variants — the one place the package's rule "task semantics do not change
between variants" is bent, deliberately:

* ``<line>-upstream`` is LIBERO as its evaluators run it (``libero/lifelong``,
  OpenVLA's ``run_libero_eval.py``): ``LiberoEnv``, one control tick per step,
  the OSC_POSE delta as the action, 128² agentview + wrist (``bodies.LIBERO``),
  10 settle ticks, and OpenVLA's per-suite tick cap (``TICKS``: the longest
  training demo plus slack).
* ``<line>`` (standard) is the macro protocol for agents that plan in poses:
  ``LiberoPoseEnv``, one step = one absolute end-effector target (or a gripper
  hold), 256² cameras (``bodies.LIBERO_STANDARD``), a budget of
  ``MACRO_STEPS`` macro moves (gym TimeLimit) and a tick guard of
  ``TICK_GUARD`` x the upstream cap. The budget unit therefore differs
  (macro moves vs ticks): a language agent cannot emit 20 Hz deltas, and the
  nav lines' Isaac polar protocol is the same kind of concession.
"""

from __future__ import annotations

import os
from typing import Any

from .env import Benchmark, Episode, LiberoSceneRef, ManipGoal, ManipMetrics
from .presets import bodies

SUITES = {   # line word -> libero suite name
    "spatial": "libero_spatial", "object": "libero_object", "goal": "libero_goal", "10": "libero_10", "90": "libero_90",
}
TICKS = {    # OpenVLA-OFT experiments/robot/libero/run_libero_eval.py TASK_MAX_STEPS
    "libero_spatial": 220,   # longest training demo has 193 steps
    "libero_object": 280,    # 254
    "libero_goal": 300,      # 270
    "libero_10": 520,        # 505
    "libero_90": 400,        # 373
}
MACRO_STEPS = 100            # standard protocol: macro moves per episode. Decided 2026-09-10.
TICK_GUARD = 10              # standard protocol: tick cap = TICK_GUARD x the upstream cap
SPLITS = ("all", "mini")
MINI_INIT_STATES = 10


def _libero_paths() -> tuple[str, str]:
    try:
        from libero.libero import get_libero_path
    except ImportError as e:
        raise ImportError("the LIBERO lines need the `libero` package in this interpreter — see INSTALL-libero.md") from e
    return str(get_libero_path("bddl_files")), str(get_libero_path("init_states"))


def _goal_predicates(bddl_file: str) -> tuple[tuple[str, ...], ...]:
    from libero.libero.envs import bddl_utils

    problem = bddl_utils.robosuite_parse_problem(bddl_file)
    return tuple(tuple(str(t) for t in pred) for pred in problem.get("goal_state", []))


def load_episodes(suite: str, split: str, data_root: str | os.PathLike | None = None,
                  scene_root: str | os.PathLike | None = None, **_unused: Any) -> list[Episode]:
    """``data_root`` / ``scene_root`` are accepted for the shared signature and ignored."""
    if suite not in SUITES.values():
        raise ValueError(f"suite must be one of {sorted(SUITES.values())}, got {suite!r}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    from libero.libero import benchmark as libero_benchmark

    bddl_root, init_root = _libero_paths()
    suite_obj = libero_benchmark.get_benchmark_dict()[suite]()
    out: list[Episode] = []
    for task_id in range(suite_obj.n_tasks):
        task = suite_obj.get_task(task_id)
        bddl = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
        init = os.path.join(init_root, task.problem_folder, task.init_states_file)
        if not os.path.isfile(bddl):
            raise FileNotFoundError(f"{suite}/{task_id}: no BDDL file at {bddl}")
        if not os.path.isfile(init):
            raise FileNotFoundError(f"{suite}/{task_id}: no init states at {init}")
        n_states = len(suite_obj.get_task_init_states(task_id))
        if split == "mini":
            n_states = min(n_states, MINI_INIT_STATES)
        scene = LiberoSceneRef(suite=suite, task_id=task_id, bddl_file=bddl, init_states_file=init)
        goal = ManipGoal(_goal_predicates(bddl))
        for k in range(n_states):
            out.append(Episode(
                index=len(out), episode_id=f"{suite}/{task_id}/{k}", scene=scene,
                start_position=None, start_rotation=None, goal=goal,
                instruction=str(task.language),
                info={"suite": suite, "task_id": task_id, "task_name": str(task.name),
                      "init_state_index": k, "bddl_file": task.bddl_file},
            ))
    return out


def _decl(word: str, upstream: bool) -> Benchmark:
    suite = SUITES[word]
    gym_id = f"EmbodiedScore/LIBERO-{word.capitalize()}-v0"
    ticks = TICKS[suite]
    return Benchmark(
        name=f"libero-{word}" + ("-upstream" if upstream else ""),
        gym_id=gym_id.replace("-v0", "-Upstream-v0") if upstream else gym_id,
        body=bodies.LIBERO if upstream else bodies.LIBERO_STANDARD,
        actions=(), splits=SPLITS,
        episodes=lambda split, data_root=None, scene_root=None, **kw: load_episodes(suite, split, data_root, scene_root, **kw),
        metrics=lambda env, **o: ManipMetrics(env),
        depth=None,
        max_episode_steps=ticks if upstream else MACRO_STEPS,
        ticks=ticks if upstream else ticks * TICK_GUARD,
        engine="libero", macro=not upstream, variant="upstream" if upstream else "standard",
        description=f"LIBERO {suite} " + ("as its evaluators run it: per-tick OSC deltas on the 128² rig"
                                          if upstream else "on the macro pose protocol and the 256² rig"),
    )


BENCHMARKS = tuple(_decl(w, up) for w in SUITES for up in (False, True))
