"""LIBERO-Plus: seven kinds of perturbation over LIBERO's four evaluation
suites (Fei et al. 2025, arXiv:2510.13626 — "LIBERO-Plus: In-depth Robustness
Analysis of Vision-Language-Action Models").

The benchmark is a drop-in replacement for the ``libero`` package
(sylvestf/LIBERO-plus, main ``4976dc3``): same import name, same
``OffScreenRenderEnv``, a benchmark registry of 10 030 tasks over the four
suites, and its own assets (a 6.4 GB ``assets.zip`` of textures, scenes and
new objects). It is NOT a GitHub fork of LIBERO — it must be installed as
``libero`` in its own interpreter (INSTALL-libero.md § LIBERO-Plus).

Where each perturbation lives (``libero/libero/envs/env_wrapper.py``,
``ControlEnv.__init__``, and ``benchmark/task_classification.json``):

* Background Textures, Light Conditions and Objects Layout are real ``.bddl``
  files (name suffixes ``_table_N`` / ``_tb_N``, ``_light_N``, ``_add_N`` /
  ``_levelN_sampleM`` / ``_moved``).
* Camera Viewpoints, Robot Initial States, Language Instructions and Sensor
  Noise carry their parameters IN THE TASK NAME —
  ``…_view_<horizon>_<vertical>_<fov%>_<end_rot>_<end_vert>_initstate_<N>[_noise_<M>]``
  — and ``ControlEnv`` decodes them: the view five drive the camera, a nonzero
  ``initstate`` selects a ``Panda<N>`` robot variant (500 mounted + 500 ground
  init poses), and ``noise`` corrupts the agentview frame inside ``step()``
  (severity bands: <=10 motion blur, <=20 gaussian, <=30 zoom, <=40 fog,
  <=50 glass). The name IS the scene reference, so ``LiberoSceneRef.bddl_file``
  is the suffixed path and ``LiberoWorld`` hands it to the wrapper unchanged.

Nothing of that needs a hook in ``LiberoBody`` / ``LiberoWorld``: the fork
does all seven inside its own ``OffScreenRenderEnv``, which is the one the
World already builds.

Lines — one per perturbation kind, which is how the paper and the repo
leaderboard report (line word -> the exact ``category`` string of
``task_classification.json``, and the released task count):

    libero-plus-camera      Camera Viewpoints       1599 tasks
    libero-plus-noise       Sensor Noise            1601
    libero-plus-robot       Robot Initial States    1550
    libero-plus-language    Language Instructions   1537
    libero-plus-layout      Objects Layout          1525
    libero-plus-light       Light Conditions        1142
    libero-plus-background  Background Textures     1076
                                                   10030 = the paper's Table 7

Each line spans all four base suites; the base suite is the first field of
``episode_id`` and fixes that episode's tick budget, so the loader carries it
in ``info["max_ticks"]`` (``LiberoEnv.episode_ticks``) rather than the line
carrying one — the declaration's ``ticks`` is the largest of the four.

Episodes: suite-major (spatial, object, goal, 10), then the registry's task
order inside the category — ``episode_id`` = ``<suite>/<task_id>/<init_state>``
with ``task_id`` the index into that suite's 2402 / 2518 / 2591 / 2519 tasks,
so it names the row of ``task_classification.json`` exactly. Splits:

    all    every task of the category, init state 0 — LIBERO-Plus's own
           protocol, "adjusting num_trials_per_task from 50 to 1" (README);
           the line's episode count is its task count above.
    mini   the first MINI_TASKS_PER_SUITE tasks of the category in each base
           suite, init state 0: 40 episodes.

Most tasks resolve to the base task's 50-state ``.pruned_init`` file; the
layout ones (``_add_`` / ``_level``) have a single state of their own under
``init_files/libero_newobj/``. ``init_states_file`` here is the resolved path
(``_resolve_init``, a transcription of the fork's ``get_task_init_states``),
because ``LiberoWorld`` loads it directly.

The instruction is the base task's canonical LIBERO instruction, EXCEPT on
the language line, where it is the fork's (the paraphrase it reads from
``<base>_language_N.bddl``). The fork derives every other task's language
from the task *filename* (``grab_language_from_filename``), which leaves the
encoded suffix in the sentence — "… place it on the plate view 0 0 100 2 352
initstate 0". That is not an instruction, and it leaks the perturbation to
the agent, so the suffix is stripped (``_base_task_name``) and the sentence
is the one the same function returns for an unperturbed name.

Variants are the base LIBERO lines': ``<line>`` is the macro pose protocol on
the 256² rig, ``<line>-upstream`` the per-tick OSC protocol on the 128² rig.
See ``libero.py`` for why the budget unit differs between them.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .env import Benchmark, Episode, LiberoSceneRef, ManipGoal, ManipMetrics
from .presets import bodies

LINES = {    # line word -> the category string of task_classification.json
    "camera": "Camera Viewpoints",
    "noise": "Sensor Noise",
    "robot": "Robot Initial States",
    "language": "Language Instructions",
    "layout": "Objects Layout",
    "light": "Light Conditions",
    "background": "Background Textures",
}
BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TICKS = {    # OpenVLA-OFT run_libero_eval.py TASK_MAX_STEPS, keyed by the base suite (LIBERO-Plus keeps them)
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
MACRO_STEPS = 100            # standard protocol: macro moves per episode, as the base LIBERO lines
TICK_GUARD = 10              # standard protocol: tick cap = TICK_GUARD x the upstream cap
SPLITS = ("all", "mini")
MINI_TASKS_PER_SUITE = 10
INIT_STATE = 0               # LIBERO-Plus evaluates one trial per task (README § Evaluation)
CLASSIFICATION = ("benchmark", "task_classification.json")

# The perturbation suffixes the fork appends to a base task name, longest first.
_SUFFIXES = (
    r"_view_-?\d+_-?\d+_\d+_-?\d+_-?\d+_initstate_\d+(?:_noise_\d+)?$",   # camera / robot / noise (and language's tail)
    r"_language_\d+$",
    r"_(?:table|tb|light|add)_\d+$",
    r"_level\d+_sample\d+$",
    r"_moved$",
)


def _libero_paths() -> tuple[str, str, str]:
    try:
        from libero.libero import get_libero_path
    except ImportError as e:
        raise ImportError(
            "the LIBERO-Plus lines need LIBERO-Plus installed as `libero` in this interpreter — "
            "see INSTALL-libero.md § LIBERO-Plus"
        ) from e
    return (str(get_libero_path("bddl_files")), str(get_libero_path("init_states")),
            str(get_libero_path("benchmark_root")))


def classification(benchmark_root: str) -> dict[str, list[dict[str, Any]]]:
    """``task_classification.json``: suite -> one row per task ({id, name,
    category, difficulty_level}), in the registry's task order."""
    path = os.path.join(benchmark_root, *CLASSIFICATION)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no {os.path.join(*CLASSIFICATION)} at {path} — is `libero` the LIBERO-Plus package?")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _base_task_name(name: str) -> str:
    """The unperturbed task name behind a perturbed one — the suffixes the
    fork appends, stripped. Every one of the 10 030 names reduces to one of
    its suite's ten base tasks."""
    for pattern in _SUFFIXES:
        name = re.sub(pattern, "", name)
    return name


def _resolve_bddl(bddl_root: str, folder: str, bddl_file: str) -> str:
    """The BDDL path ``ControlEnv.__init__`` ends up parsing: a name carrying a
    ``_view_`` block names the file before it (``env_wrapper.py``: split on
    ``_view_``, add ``.bddl``); anything else is a file of its own."""
    path = os.path.join(bddl_root, folder, bddl_file)
    return path.split("_view_")[0] + ".bddl" if "_view_" in path else path


def _resolve_init(init_root: str, folder: str, init_file: str) -> str:
    """The init-state path the fork's ``Benchmark.get_task_init_states``
    reaches — its branch order kept, with the fallback it leaves out (a name
    carrying no suffix is its own file)."""
    ext = "." + init_file.split(".")[-1]
    if "_language_" in init_file:
        return os.path.join(init_root, folder, init_file.split("_language_")[0] + ext)
    if "_view_" in init_file:
        return os.path.join(init_root, folder, init_file.split("_view_")[0] + ext)
    path = os.path.join(init_root, folder, init_file)
    if "_table_" in init_file:
        path = os.path.join(init_root, folder, re.sub(r"_table_\d+", "", init_file))
    if "_tb_" in init_file:
        path = os.path.join(init_root, folder, re.sub(r"_tb_\d+", "", init_file))
    if "_light_" in init_file:
        path = os.path.join(init_root, folder, init_file.split("_light_")[0] + ext)
    if "_add_" in init_file or "_level" in init_file:
        path = os.path.join(init_root, "libero_newobj", folder, init_file)
    return path


def _goal_predicates(bddl_file: str) -> tuple[tuple[str, ...], ...]:
    from libero.libero.envs import bddl_utils

    problem = bddl_utils.robosuite_parse_problem(bddl_file)
    return tuple(tuple(str(t) for t in pred) for pred in problem.get("goal_state", []))


def load_episodes(line: str, split: str, data_root: str | os.PathLike | None = None,
                  scene_root: str | os.PathLike | None = None, tick_guard: int = 1,
                  **_unused: Any) -> list[Episode]:
    """``data_root`` / ``scene_root`` are accepted for the shared signature and
    ignored. ``tick_guard`` multiplies the per-episode tick cap: 1 on the
    upstream variant (the cap IS the budget), ``TICK_GUARD`` on the standard
    one (where the budget is macro moves and the cap is only a guard) — the
    declaration binds it, since the loader cannot see the variant."""
    if line not in LINES:
        raise ValueError(f"line must be one of {sorted(LINES)}, got {line!r}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    from libero.libero import benchmark as libero_benchmark

    category = LINES[line]
    bddl_root, init_root, benchmark_root = _libero_paths()
    rows_by_suite = classification(benchmark_root)
    suites = libero_benchmark.get_benchmark_dict()
    goals: dict[str, ManipGoal] = {}
    out: list[Episode] = []
    for suite in BASE_SUITES:
        suite_obj = suites[suite]()
        rows = rows_by_suite[suite]
        if len(rows) != suite_obj.n_tasks:
            raise RuntimeError(f"{suite}: {len(rows)} classified tasks but {suite_obj.n_tasks} registered")
        picked = [i for i, row in enumerate(rows) if row["category"] == category]
        if split == "mini":
            picked = picked[:MINI_TASKS_PER_SUITE]
        for task_id in picked:
            task = suite_obj.get_task(task_id)
            bddl = _resolve_bddl(bddl_root, task.problem_folder, task.bddl_file)
            init = _resolve_init(init_root, task.problem_folder, task.init_states_file)
            if not os.path.isfile(bddl):
                raise FileNotFoundError(f"{suite}/{task_id}: no BDDL file at {bddl}")
            if not os.path.isfile(init):
                raise FileNotFoundError(f"{suite}/{task_id}: no init states at {init}")
            if bddl not in goals:
                goals[bddl] = ManipGoal(_goal_predicates(bddl))
            base = _base_task_name(str(task.name))
            scene = LiberoSceneRef(suite=suite, task_id=task_id,
                                   bddl_file=os.path.join(bddl_root, task.problem_folder, task.bddl_file),
                                   init_states_file=init)
            out.append(Episode(
                index=len(out), episode_id=f"{suite}/{task_id}/{INIT_STATE}", scene=scene,
                start_position=None, start_rotation=None, goal=goals[bddl],
                instruction=str(task.language) if line == "language" else " ".join(base.split("_")),
                info={"suite": suite, "base_suite": suite, "perturbation": line, "category": category,
                      "difficulty_level": rows[task_id]["difficulty_level"], "task_id": task_id,
                      "task_name": str(task.name), "base_task_name": base, "init_state_index": INIT_STATE,
                      "bddl_file": task.bddl_file, "max_ticks": TICKS[suite] * tick_guard},
            ))
    return out


def _decl(word: str, upstream: bool) -> Benchmark:
    gym_id = f"EmbodiedScore/LIBERO-Plus-{word.capitalize()}-v0"
    ticks = max(TICKS.values())     # the longest base suite's cap; each episode uses its own via info["max_ticks"]
    return Benchmark(
        name=f"libero-plus-{word}" + ("-upstream" if upstream else ""),
        gym_id=gym_id.replace("-v0", "-Upstream-v0") if upstream else gym_id,
        body=bodies.LIBERO if upstream else bodies.LIBERO_STANDARD,
        actions=(), splits=SPLITS,
        episodes=lambda split, data_root=None, scene_root=None, **kw: load_episodes(
            word, split, data_root, scene_root, tick_guard=1 if upstream else TICK_GUARD, **kw),
        metrics=lambda env, **o: ManipMetrics(env),
        depth=None,
        max_episode_steps=ticks if upstream else MACRO_STEPS,
        ticks=ticks if upstream else ticks * TICK_GUARD,
        engine="libero", macro=not upstream, variant="upstream" if upstream else "standard",
        description=f"LIBERO-Plus {LINES[word]!r} over the four LIBERO suites " + (
            "as its evaluators run it: per-tick OSC deltas on the 128² rig"
            if upstream else "on the macro pose protocol and the 256² rig"),
    )


BENCHMARKS = tuple(_decl(w, up) for w in LINES for up in (False, True))
