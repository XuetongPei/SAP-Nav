"""RoboCasa: kitchen manipulation on a mobile manipulator (a Franka Panda on
an Omron base) in robosuite / MuJoCo. Two releases of one benchmark, one
engine:

* **RoboCasa** (Nasiriany et al., RSS 2024, arXiv:2406.02523) — 100 tasks,
  25 of them atomic, grouped by the paper's *eight foundational skills*.
  Repo tag ``v0.2``.
* **RoboCasa365** (ICLR 2026, arXiv:2603.04356) — 365 tasks (65 atomic, 300
  composite) in 2500 generated kitchens. Repo tag ``v1.0`` / version 1.0.1,
  the repo's ``main``.

The two are the same pip distribution (``robocasa``) at incompatible pins
(mujoco 3.2.6 / numpy 1.23.3 / py3.10 against mujoco 3.3.1 / numpy 2.2.5 /
py3.11) and they renamed the task classes, so they need one conda env each —
see INSTALL-robocasa.md. The engine code is shared; only these declarations
differ.

Data: nothing under ``EMBODIEDSCORE_DATA_ROOT``. A RoboCasa kitchen is
generated from a layout id and a style id; the assets it draws from are
downloaded into the ``robocasa`` checkout by
``robocasa/scripts/download_kitchen_assets.py``. The demonstration datasets
are not read (evaluation needs none).

Lines — one per group the benchmark itself scores:

    RoboCasa365 (``robocasa/utils/dataset_registry.py`` ``TASK_SET_REGISTRY``,
    the three sets the leaderboard reports and whose concatenation is
    ``target50``):

        robocasa365-atomic-seen        18 atomic tasks seen in the target data
        robocasa365-composite-seen     16 composite tasks seen in the target data
        robocasa365-composite-unseen   16 composite tasks held out (zero-shot)

    RoboCasa v0.2 (the paper's eight foundational skills, as
    ``docs/_versions/v0.2/tasks_scenes_assets/atomic_tasks.md`` groups the 25
    atomic tasks):

        robocasa-pnp 8 · robocasa-doors 4 · robocasa-drawers 2 · robocasa-levers 3
        robocasa-knobs 2 · robocasa-insertion 2 · robocasa-buttons 3 · robocasa-navigate 1

Episodes: task-major, one per evaluation scenario. A scenario is
``(task, layout_id, style_id, seed)`` — the RoboCasa evaluation protocol's
unit:

* RoboCasa365 (``docs/benchmarking/benchmarking_overview.md``): the ``target``
  split is "a disjoint set of 10 target kitchen scenes and a disjoint set of
  target objects", and "for all experiments, we randomly sample 50 scenarios
  to run evaluation rollouts on ... We report average task success rates
  across these 50 rollouts". The 10 scenes are
  ``layout_and_style_ids = list(zip(range(1, 11), range(1, 11)))`` in
  ``robocasa/utils/env_utils.py`` ``create_env`` under ``split="target"``,
  with ``obj_instance_split="target"``.
* RoboCasa v0.2 (paper § 5): "we evaluate the model performance across 50
  trials across five fixed evaluation scenes, each with a distinct floor plan
  and style"; the five are
  ``layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))`` with
  ``obj_instance_split="B"`` (unseen object instances) in
  ``robocasa/utils/eval_utils.py`` ``create_eval_env``.

Neither upstream fixes *which* scenario is which — it draws the scene at
random per rollout. This port deals the 50 scenarios round-robin over the
line's scenes and uses the scenario index as the env seed, so an episode id
names a reproducible kitchen: ``<task>/<layout>-<style>/<seed>``. That is the
one place the protocol is pinned down rather than transcribed (§ Deviations
of the README). A task that rules a scene out (``EXCLUDE_LAYOUTS`` /
``EXCLUDE_STYLES`` on its class — nine of RoboCasa365's composite tasks do) is
dealt only the scenes it can be generated in; ``*_EXCLUDED_SCENES`` below is
that table.

Splits: ``all`` (every scenario: 50 per task) and ``mini`` (scenarios 0-9).

Variants — the same deliberate bend as the LIBERO lines:

* ``<line>-upstream`` is RoboCasa as its own harness runs it
  (``robocasa/wrappers/gym_wrapper.py``, ``robocasa/utils/env_utils.py``
  ``run_random_rollouts``): ``RobocasaEnv``, one 20 Hz control tick per step,
  RoboCasa's 12-D gym action, the three-camera 128² rig of ``create_env``,
  the task's own horizon as the tick cap, and done on success.
* ``<line>`` (standard) is the macro protocol for agents that plan in poses:
  ``RobocasaPoseEnv``, one step = one base move, one absolute end-effector
  target or one gripper hold, a 256² third-person + wrist rig
  (``bodies.ROBOCASA_STANDARD``), a budget of ``MACRO_STEPS`` macro moves
  (gym TimeLimit) and a tick guard of ``TICK_GUARD`` x the task's horizon.
"""

from __future__ import annotations

import os
import re
from typing import Any

from .env import Benchmark, Episode, ManipGoal, ManipMetrics, RobocasaSceneRef
from .presets import bodies

# ---- RoboCasa365 (repo main / v1.0.1) --------------------------------------------------
# Task -> horizon, from robocasa/utils/dataset_registry.py (ATOMIC_TASK_DATASETS /
# COMPOSITE_TASK_DATASETS "horizon"), for the three sets of TASK_SET_REGISTRY the
# leaderboard scores. Horizons are per task and were raised 1.5x in v1.0.1
# ("Updated horizon lengths (1.5x increase) across all tasks").
R365_ATOMIC_SEEN = {
    "CloseBlenderLid": 900, "CloseFridge": 900, "CloseToasterOvenDoor": 450, "CoffeeSetupMug": 600,
    "NavigateKitchen": 450, "OpenCabinet": 1050, "OpenDrawer": 750, "OpenStandMixerHead": 450,
    "PickPlaceCounterToCabinet": 750, "PickPlaceCounterToStove": 600, "PickPlaceDrawerToCounter": 750,
    "PickPlaceSinkToCounter": 900, "PickPlaceToasterToCounter": 600, "SlideDishwasherRack": 450,
    "TurnOffStove": 750, "TurnOnElectricKettle": 450, "TurnOnMicrowave": 450, "TurnOnSinkFaucet": 600,
}
R365_COMPOSITE_SEEN = {
    "DeliverStraw": 2550, "GetToastedBread": 3000, "KettleBoiling": 1500, "LoadDishwasher": 1800,
    "PackIdenticalLunches": 3900, "PreSoakPan": 2400, "PrepareCoffee": 1800, "RinseSinkBasin": 1350,
    "ScrubCuttingBoard": 1200, "SearingMeat": 4350, "SetUpCuttingStation": 2400, "StackBowlsCabinet": 2100,
    "SteamInMicrowave": 2100, "StirVegetables": 2400, "StoreLeftoversInBowl": 2550, "WashLettuce": 1650,
}
R365_COMPOSITE_UNSEEN = {
    "ArrangeBreadBasket": 4350, "ArrangeTea": 2250, "BreadSelection": 1950, "CategorizeCondiments": 1650,
    "CuttingToolSelection": 1200, "GarnishPancake": 2700, "GatherTableware": 2250, "HeatKebabSandwich": 2700,
    "MakeIceLemonade": 3000, "PanTransfer": 1800, "PortionHotDogs": 2250, "RecycleBottlesByType": 2850,
    "SeparateFreezerRack": 2400, "WaffleReheat": 4050, "WashFruitColander": 3150, "WeighIngredients": 3000,
}

# ---- RoboCasa v0.2 ----------------------------------------------------------------------
# The 25 atomic tasks grouped by the paper's eight foundational skills
# (docs/_versions/v0.2/tasks_scenes_assets/atomic_tasks.md), each with the horizon of
# SINGLE_STAGE_TASK_DATASETS in robocasa/utils/dataset_registry.py at tag v0.2.
R02_PNP = {
    "PnPCounterToCab": 500, "PnPCabToCounter": 500, "PnPCounterToSink": 700, "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600, "PnPMicrowaveToCounter": 500, "PnPCounterToStove": 500, "PnPStoveToCounter": 500,
}
R02_DOORS = {"OpenSingleDoor": 500, "CloseSingleDoor": 500, "OpenDoubleDoor": 1000, "CloseDoubleDoor": 700}
R02_DRAWERS = {"OpenDrawer": 500, "CloseDrawer": 500}
R02_LEVERS = {"TurnOnSinkFaucet": 500, "TurnOffSinkFaucet": 500, "TurnSinkSpout": 500}
R02_KNOBS = {"TurnOnStove": 500, "TurnOffStove": 500}
R02_INSERTION = {"CoffeeSetupMug": 600, "CoffeeServeMug": 600}
R02_BUTTONS = {"CoffeePressButton": 300, "TurnOnMicrowave": 500, "TurnOffMicrowave": 500}
R02_NAVIGATE = {"NavigateKitchen": 500}

# The evaluation scenes of each release, and the object split they draw instances from.
R365_SCENES = tuple(zip(range(1, 11), range(1, 11)))                     # env_utils.create_env, split="target"
R365_OBJ_SPLIT = "target"

# A RoboCasa task class may declare EXCLUDE_LAYOUTS / EXCLUDE_STYLES, and Kitchen.__init__
# filters layout_and_style_ids by them; a scenario pinned to an excluded scene would leave
# the env nothing to sample from, and it raises on reset. These are the evaluation scenes
# each task rules out, read off the installed release (test_robocasa_contracts.py
# ``test_excluded_scenes_match_the_installed_release`` checks the table); a task not listed
# rules out none, and RoboCasa v0.2 rules out none of its five scenes at all.
R365_EXCLUDED_SCENES = {
    "ArrangeBreadBasket": ((1, 1), (3, 3), (5, 5), (6, 6)),
    "DeliverStraw": ((1, 1), (3, 3), (5, 5), (6, 6)),
    "GarnishPancake": ((1, 1), (3, 3), (5, 5), (6, 6)),
    "GetToastedBread": ((1, 1), (3, 3), (5, 5), (6, 6)),
    "PortionHotDogs": ((1, 1), (3, 3), (5, 5), (6, 6)),
    "RecycleBottlesByType": ((1, 1), (3, 3), (5, 5), (6, 6), (8, 8)),
    "SeparateFreezerRack": ((1, 1), (3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (10, 10)),
    "StoreLeftoversInBowl": ((1, 1), (3, 3), (5, 5), (6, 6)),
    "WaffleReheat": ((9, 9),),
}
R02_SCENES = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))                   # eval_utils.create_eval_env
R02_OBJ_SPLIT = "B"
R02_EXCLUDED_SCENES: dict[str, tuple[tuple[int, int], ...]] = {}

SCENARIOS = 50               # rollouts per task, both releases (benchmarking_overview.md / the v0.2 paper)
MINI_SCENARIOS = 10          # the mini split
MACRO_STEPS = 100            # standard protocol: macro moves per episode. Decided 2026-09-10, as the LIBERO lines.
TICK_GUARD = 10              # standard protocol: tick cap = TICK_GUARD x the task's own horizon
SPLITS = ("all", "mini")

LINES: dict[str, dict[str, Any]] = {
    "robocasa365-atomic-seen": dict(release="robocasa365", tasks=R365_ATOMIC_SEEN, gym="RoboCasa365-AtomicSeen"),
    "robocasa365-composite-seen": dict(release="robocasa365", tasks=R365_COMPOSITE_SEEN,
                                       gym="RoboCasa365-CompositeSeen"),
    "robocasa365-composite-unseen": dict(release="robocasa365", tasks=R365_COMPOSITE_UNSEEN,
                                         gym="RoboCasa365-CompositeUnseen"),
    "robocasa-pnp": dict(release="robocasa", tasks=R02_PNP, gym="RoboCasa-PnP"),
    "robocasa-doors": dict(release="robocasa", tasks=R02_DOORS, gym="RoboCasa-Doors"),
    "robocasa-drawers": dict(release="robocasa", tasks=R02_DRAWERS, gym="RoboCasa-Drawers"),
    "robocasa-levers": dict(release="robocasa", tasks=R02_LEVERS, gym="RoboCasa-Levers"),
    "robocasa-knobs": dict(release="robocasa", tasks=R02_KNOBS, gym="RoboCasa-Knobs"),
    "robocasa-insertion": dict(release="robocasa", tasks=R02_INSERTION, gym="RoboCasa-Insertion"),
    "robocasa-buttons": dict(release="robocasa", tasks=R02_BUTTONS, gym="RoboCasa-Buttons"),
    "robocasa-navigate": dict(release="robocasa", tasks=R02_NAVIGATE, gym="RoboCasa-Navigate"),
}

RELEASES = {
    "robocasa365": dict(scenes=R365_SCENES, obj_split=R365_OBJ_SPLIT, excluded=R365_EXCLUDED_SCENES),
    "robocasa": dict(scenes=R02_SCENES, obj_split=R02_OBJ_SPLIT, excluded=R02_EXCLUDED_SCENES),
}

_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")


def humanise(task: str) -> str:
    """``PickPlaceCounterToCabinet`` -> ``pick place counter to cabinet``: the
    placeholder instruction. RoboCasa words the real one for the objects it
    sampled, so it is only known after a reset — the env body puts it in
    ``info["language"]`` and over ``info["episode"]["instruction"]``."""
    return _CAMEL.sub(" ", task).lower()


def load_episodes(line: str, split: str, data_root: str | os.PathLike | None = None,
                  scene_root: str | os.PathLike | None = None, **_unused: Any) -> list[Episode]:
    """``data_root`` / ``scene_root`` are accepted for the shared signature and
    ignored: a RoboCasa kitchen is generated, not loaded from a scene file."""
    if line not in LINES:
        raise ValueError(f"line must be one of {sorted(LINES)}, got {line!r}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    decl = LINES[line]
    release = RELEASES[decl["release"]]
    obj_split = release["obj_split"]
    n = SCENARIOS if split == "all" else MINI_SCENARIOS
    out: list[Episode] = []
    for task, horizon in decl["tasks"].items():
        scenes = scenes_of(task, release)
        for k in range(n):
            layout, style = scenes[k % len(scenes)]
            scene = RobocasaSceneRef(task=task, layout_id=int(layout), style_id=int(style), seed=k,
                                     obj_instance_split=obj_split, release=decl["release"])
            out.append(Episode(
                index=len(out), episode_id=f"{task}/{layout}-{style}/{k}", scene=scene,
                start_position=None, start_rotation=None,
                # RoboCasa's goal is the task class's own _check_success, not a declarative
                # predicate list; the single entry names the check the simulator runs.
                goal=ManipGoal((("robocasa_check_success", task),)),
                instruction=humanise(task),
                info={"release": decl["release"], "line": line, "task": task, "scenario": k,
                      "layout_id": int(layout), "style_id": int(style), "horizon": int(horizon),
                      "obj_instance_split": obj_split},
            ))
    return out


def scenes_of(task: str, release: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    """The release's evaluation scenes this task can actually be generated in."""
    ruled_out = set(release["excluded"].get(task, ()))
    scenes = tuple(s for s in release["scenes"] if s not in ruled_out)
    if not scenes:
        raise ValueError(f"{task}: every evaluation scene of the release is excluded by the task")
    return scenes


def installed_exclusions(release: str) -> dict[str, tuple[tuple[int, int], ...]]:
    """What the installed ``robocasa`` rules out, for the contract test that
    checks the ``*_EXCLUDED_SCENES`` tables above against it."""
    import robocasa  # noqa: F401  (registers the kitchen environments with robosuite)
    from robosuite.environments.base import REGISTERED_ENVS

    scenes = RELEASES[release]["scenes"]
    out: dict[str, tuple[tuple[int, int], ...]] = {}
    for decl in LINES.values():
        if decl["release"] != release:
            continue
        for task in decl["tasks"]:
            cls = REGISTERED_ENVS[task]
            bad_l = set(getattr(cls, "EXCLUDE_LAYOUTS", ()) or ())
            bad_s = set(getattr(cls, "EXCLUDE_STYLES", ()) or ())
            ruled_out = tuple((la, st) for (la, st) in scenes if la in bad_l or st in bad_s)
            if ruled_out:
                out[task] = ruled_out
    return out


def installed_horizons(release: str) -> dict[str, int]:
    """The horizons of the installed ``robocasa``, for the contract test that
    checks the tables above against the release in this interpreter."""
    from robocasa.utils import dataset_registry as reg

    if release == "robocasa365":
        tables = (reg.ATOMIC_TASK_DATASETS, reg.COMPOSITE_TASK_DATASETS)
    else:
        tables = (reg.SINGLE_STAGE_TASK_DATASETS, reg.MULTI_STAGE_TASK_DATASETS)
    return {k: int(v["horizon"]) for table in tables for k, v in table.items()}


def _decl(line: str, upstream: bool) -> Benchmark:
    decl = LINES[line]
    horizon_cap = max(decl["tasks"].values())
    return Benchmark(
        name=line + ("-upstream" if upstream else ""),
        gym_id=f"EmbodiedScore/{decl['gym']}-" + ("Upstream-v0" if upstream else "v0"),
        body=bodies.ROBOCASA if upstream else bodies.ROBOCASA_STANDARD,
        actions=(), splits=SPLITS,
        episodes=lambda split, data_root=None, scene_root=None, **kw: load_episodes(line, split, data_root,
                                                                                    scene_root, **kw),
        metrics=lambda env, **o: ManipMetrics(env),
        depth=None,
        # The tick cap is the episode's own horizon (RoboCasa's is per task), so the static
        # TimeLimit is only the line's ceiling: the longest horizon of its tasks.
        max_episode_steps=horizon_cap if upstream else MACRO_STEPS,
        ticks=None, tick_scale=1.0 if upstream else float(TICK_GUARD),
        engine="robocasa", macro=not upstream, variant="upstream" if upstream else "standard",
        description=f"RoboCasa {line} " + ("as its own harness runs it: per-tick 12-D actions on the 128² 3-camera rig"
                                           if upstream else "on the macro pose + base protocol and the 256² rig"),
    )


BENCHMARKS = tuple(_decl(line, up) for line in LINES for up in (False, True))
