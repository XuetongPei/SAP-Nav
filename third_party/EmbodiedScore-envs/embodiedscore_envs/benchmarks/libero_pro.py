"""LIBERO-PRO: LIBERO's four evaluation suites with the tasks perturbed
(Zhou et al. 2025, arXiv:2510.03827 — "LIBERO-PRO: Towards Robust and Fair
Evaluation of Vision-Language-Action Models Beyond Memorization").

The benchmark is a fork of LIBERO (Zxy-MLlab/LIBERO-PRO, master ``eafdb80``,
a GitHub fork of Lifelong-Robot-Learning/LIBERO) whose ``perturbation.py``
rewrites a suite's BDDL files along one dimension and regenerates the init
states. Nothing else changes: same Franka Panda, same OSC_POSE controller,
same success predicate. The released task files are the HuggingFace dataset
``zhouxueyang/LIBERO-Pro`` (676 files, 15 MB), unpacked into the fork's
``bddl_files`` / ``init_files`` so the fork's own benchmark registry resolves
them (INSTALL-libero.md § LIBERO-PRO).

Lines — one per LIBERO suite, as the base ``libero-*`` lines are, each
carrying that suite's ten tasks under all four released perturbation
dimensions (40 tasks). The suite is what fixes the tick budget
(``TASK_MAX_STEPS`` in the repo's README, lines 253-278: 220 / 280 / 300 /
520 for spatial / object / goal / 10, the base suite's cap repeated for every
perturbed suite), which is why the line is the suite and the dimension is a
folder inside it:

    libero-pro-spatial   libero_spatial_{object,swap,lan,task}   40 tasks
    libero-pro-object    libero_object_{object,swap,lan,task}    40 tasks
    libero-pro-goal      libero_goal_{object,swap,lan,task}      40 tasks
    libero-pro-10        libero_10_{object,swap,lan,task}        40 tasks

The four dimensions (``DIMENSIONS``; the fork's folder suffix, the paper's
column name):

    object   ``_object``  the target and its receptacle become other object
                          categories (akita_black_bowl -> black_bowl, plate ->
                          yellow_plate); the goal predicate follows.
    position ``_swap``    two objects trade the regions they are placed in;
                          the instruction and the goal are unchanged, so the
                          spatial language now points elsewhere.
    semantic ``_lan``     the instruction is paraphrased (three candidates per
                          task in ``libero_ood/ood_language.yaml``); scene and
                          goal unchanged.
    task     ``_task``    the goal predicate is replaced (a different object,
                          a negated relation) and the instruction with it.

The fifth dimension the paper names, ``environment`` (``_env``), is declared
by the fork's registry but ships no task files, in the repo or in the
HuggingFace release, and the repo's own README warns that replacing the arena
lets table objects "move randomly"; it is not a line here. The ``_temp``,
``_object_ood``, ``_relation_ood`` and ``_semantic_ood`` folders of the
registry are the paper's ablations and ship no files either.

Data: the fork's ``libero`` package (``libero.libero.get_libero_path``), like
the base LIBERO lines — nothing under ``EMBODIEDSCORE_DATA_ROOT``. It must be
the LIBERO-PRO fork: the ``_object`` dimension uses object categories only
that fork registers, and its assets are the fork's.

Episodes: dimension-major, then the suite's task order, then the init state —
``episode_id`` = ``<fork suite>/<task_id>/<init_state>``, e.g.
``libero_spatial_swap/3/7``. 50 init states per task, as LIBERO releases them
and as LIBERO-PRO evaluates them (it changes nothing about the trial count).
Splits: ``all`` (every init state: 2000 episodes per line) and ``mini``
(init states 0-9: 400).

The instruction is read from the BDDL ``(:language ...)`` field, NOT from the
fork's benchmark registry: that registry derives the language from the task
*filename* (``benchmark/__init__.py::grab_language_from_filename``) and the
perturbed files keep the base task's filename, so the registry still reports
the unperturbed instruction — which would erase the ``semantic`` dimension
entirely and half of ``task``. The BDDL field is where ``perturbation.py``
writes the perturbation (``LanguagePerturbator`` / ``TaskPerturbator``), so
the BDDL is the source of truth.

Variants are the base LIBERO lines': ``<line>`` is the macro pose protocol on
the 256² rig, ``<line>-upstream`` the per-tick OSC protocol on the 128² rig
with the suite's tick cap. See ``libero.py`` for why the budget unit differs.
"""

from __future__ import annotations

import os
from typing import Any

from .env import Benchmark, Episode, LiberoSceneRef, ManipGoal, ManipMetrics
from .presets import bodies

SUITES = {   # line word -> the base libero suite the perturbed folders are built from
    "spatial": "libero_spatial", "object": "libero_object", "goal": "libero_goal", "10": "libero_10",
}
DIMENSIONS = (   # (line word for the perturbation, the fork's folder suffix, the paper's column)
    ("object", "object", "Object"),
    ("position", "swap", "Position"),
    ("semantic", "lan", "Semantic"),
    ("task", "task", "Task"),
)
TICKS = {    # LIBERO-PRO README TASK_MAX_STEPS (lines 253-278): the base suite's cap on every perturbed folder
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
MACRO_STEPS = 100            # standard protocol: macro moves per episode, as the base LIBERO lines
TICK_GUARD = 10              # standard protocol: tick cap = TICK_GUARD x the upstream cap
SPLITS = ("all", "mini")
MINI_INIT_STATES = 10


def _libero_paths() -> tuple[str, str]:
    try:
        from libero.libero import get_libero_path
    except ImportError as e:
        raise ImportError(
            "the LIBERO-PRO lines need the LIBERO-PRO fork installed as `libero` in this interpreter — "
            "see INSTALL-libero.md § LIBERO-PRO"
        ) from e
    return str(get_libero_path("bddl_files")), str(get_libero_path("init_states"))


def _problem(bddl_file: str) -> tuple[tuple[tuple[str, ...], ...], str]:
    """(goal predicates, instruction) of a BDDL file. The instruction is the
    file's own ``(:language ...)`` — see the module docstring."""
    from libero.libero.envs import bddl_utils

    problem = bddl_utils.robosuite_parse_problem(bddl_file)
    goal = tuple(tuple(str(t) for t in pred) for pred in problem.get("goal_state", []))
    return goal, str(bddl_utils.get_problem_info(bddl_file)["language_instruction"])


def load_episodes(suite: str, split: str, data_root: str | os.PathLike | None = None,
                  scene_root: str | os.PathLike | None = None, **_unused: Any) -> list[Episode]:
    """``data_root`` / ``scene_root`` are accepted for the shared signature and ignored."""
    if suite not in SUITES.values():
        raise ValueError(f"suite must be one of {sorted(SUITES.values())}, got {suite!r}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    from libero.libero import benchmark as libero_benchmark

    bddl_root, init_root = _libero_paths()
    suites = libero_benchmark.get_benchmark_dict()
    out: list[Episode] = []
    for word, folder_suffix, column in DIMENSIONS:
        folder = f"{suite}_{folder_suffix}"
        if folder not in suites:
            raise KeyError(f"{folder!r} is not a suite of the installed `libero` — is it the LIBERO-PRO fork?")
        suite_obj = suites[folder]()
        for task_id in range(suite_obj.n_tasks):
            task = suite_obj.get_task(task_id)
            bddl = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
            init = os.path.join(init_root, task.problem_folder, task.init_states_file)
            if not os.path.isfile(bddl):
                raise FileNotFoundError(f"{folder}/{task_id}: no BDDL file at {bddl} — unpack the LIBERO-PRO release")
            if not os.path.isfile(init):
                raise FileNotFoundError(f"{folder}/{task_id}: no init states at {init}")
            n_states = len(suite_obj.get_task_init_states(task_id))
            if split == "mini":
                n_states = min(n_states, MINI_INIT_STATES)
            scene = LiberoSceneRef(suite=folder, task_id=task_id, bddl_file=bddl, init_states_file=init)
            predicates, instruction = _problem(bddl)
            goal = ManipGoal(predicates)
            for k in range(n_states):
                out.append(Episode(
                    index=len(out), episode_id=f"{folder}/{task_id}/{k}", scene=scene,
                    start_position=None, start_rotation=None, goal=goal,
                    instruction=instruction,
                    info={"suite": folder, "base_suite": suite, "perturbation": word, "column": column,
                          "task_id": task_id, "task_name": str(task.name), "init_state_index": k,
                          "bddl_file": task.bddl_file},
                ))
    return out


def _decl(word: str, upstream: bool) -> Benchmark:
    suite = SUITES[word]
    gym_id = f"EmbodiedScore/LIBERO-PRO-{word.capitalize()}-v0"
    ticks = TICKS[suite]
    dims = ", ".join(w for w, _, _ in DIMENSIONS)
    return Benchmark(
        name=f"libero-pro-{word}" + ("-upstream" if upstream else ""),
        gym_id=gym_id.replace("-v0", "-Upstream-v0") if upstream else gym_id,
        body=bodies.LIBERO if upstream else bodies.LIBERO_STANDARD,
        actions=(), splits=SPLITS,
        episodes=lambda split, data_root=None, scene_root=None, **kw: load_episodes(suite, split, data_root,
                                                                                    scene_root, **kw),
        metrics=lambda env, **o: ManipMetrics(env),
        depth=None,
        max_episode_steps=ticks if upstream else MACRO_STEPS,
        ticks=ticks if upstream else ticks * TICK_GUARD,
        engine="libero", macro=not upstream, variant="upstream" if upstream else "standard",
        description=f"LIBERO-PRO {suite} perturbed along {dims} " + (
            "as its evaluators run it: per-tick OSC deltas on the 128² rig"
            if upstream else "on the macro pose protocol and the 256² rig"),
    )


BENCHMARKS = tuple(_decl(w, up) for w in SUITES for up in (False, True))
