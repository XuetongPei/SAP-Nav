"""Layer L2 — the benchmark declarations, and ``make()`` that turns one into a
Gymnasium stack. Each member module exposes a ``BENCHMARKS`` tuple; nothing
else in the package knows their names.

Every line declares two variants: ``<line>`` (the default — EmbodiedScore's shared
body, ``presets.bodies.STANDARD``, or its Isaac transcription for the VLNverse
lines) and ``<line>-upstream`` (the line's own evaluator: its rig, action table,
depth post-processing and, for the EQA and VLNverse lines, its own action
protocol). Task semantics — loader, goal, success distance, budget, metric keys
— are the line's and do not change between variants. ``Benchmark.engine`` picks
the simulator: habitat-sim in process, Isaac Sim through its render worker,
robosuite / MuJoCo in process (the LIBERO and RoboCasa manipulation lines),
SAPIEN 3 in process (the RoboTwin bimanual manipulation lines), pybullet in
process (the CALVIN chain line), or OmniGibson on Isaac Sim in process (the
BEHAVIOR-1K line). Every manipulation engine's variants differ in protocol — see
``libero.py``, ``robotwin.py``, ``robocasa.py``, ``calvin.py`` and ``behavior.py``
for the one deviation from the rule above."""

from __future__ import annotations

import importlib
from typing import Any

import gymnasium as gym

from .env import (BehaviorEnv, BehaviorPoseEnv, Benchmark, CalvinEnv, CalvinPoseEnv, DepthClip, DynamicTimeLimit, HabitatEnv, HabitatPoseEnv,
                  IsaacEnv, IsaacPolarEnv, LiberoEnv, LiberoPoseEnv, RobocasaEnv, RobocasaPoseEnv, RobotwinEnv,
                  RobotwinPoseEnv)

_MEMBERS = ("vlnce", "ivlnce", "objectnav", "goat", "langmap", "hmeqa", "express", "vlnverse", "libero", "libero_pro",
            "libero_plus", "robotwin", "robocasa", "calvin", "behavior")

BENCHMARKS: dict[str, Benchmark] = {}
for _m in _MEMBERS:
    try:
        _mod = importlib.import_module(f"{__name__}.{_m}")
    except ModuleNotFoundError as _e:   # a member not written yet
        if _e.name != f"{__name__}.{_m}":
            raise
        continue
    for _b in _mod.BENCHMARKS:
        if _b.name in BENCHMARKS:
            raise RuntimeError(f"duplicate benchmark name {_b.name}")
        BENCHMARKS[_b.name] = _b
del _m, _mod, _b


_SUFFIX = "-upstream"


def resolve(name: str, variant: str | None = None) -> str:
    """The declaration name for ``name`` under ``variant``: ``None`` keeps the
    name as given, ``"standard"`` strips ``-upstream``, ``"upstream"`` appends it."""
    if variant is None:
        return name
    if variant not in ("standard", "upstream"):
        raise ValueError(f"variant must be 'standard' or 'upstream', got {variant!r}")
    line = name[: -len(_SUFFIX)] if name.endswith(_SUFFIX) else name
    return line + _SUFFIX if variant == "upstream" else line


def benchmark(name: str, variant: str | None = None) -> Benchmark:
    key = resolve(name, variant)
    try:
        return BENCHMARKS[key]
    except KeyError:
        raise KeyError(f"unknown benchmark {key!r}; known: {sorted(BENCHMARKS)}") from None


def build_env(benchmark_name: str, split: str, data_root: str | None = None, scene_root: str | None = None,
              body: Any = None, actions: Any = None, gpu_id: int = 0, sim_seed: int = 42,
              render_mode: str | None = None, polar: bool | None = None, robotwin_root: str | None = None,
              behavior_root: str | None = None, **loader_kwargs: Any) -> gym.Env:
    """The bare body for a benchmark (what the gym ids point at). ``body`` /
    ``actions`` override the declaration (a different rig or action table on
    the same episodes and measures — how the legacy nodesets' per-graph
    sensor knobs and upstream RxR's body are served); ``polar`` overrides an
    Isaac declaration's action interface (``IsaacPolarEnv`` vs the discrete
    body). ``sim_seed`` is habitat's; the Isaac worker takes none, MuJoCo is
    deterministic given the init state."""
    b = benchmark(benchmark_name)
    if b.engine == "behavior" and behavior_root:
        # BEHAVIOR resolves its assets by OmniGibson's own data root, not EMBODIEDSCORE_DATA_ROOT
        loader_kwargs.setdefault("behavior_root", behavior_root)
    if b.engine == "robotwin" and robotwin_root:
        # the RoboTwin loader reads the task's language out of the checkout too
        loader_kwargs.setdefault("robotwin_root", robotwin_root)
    episodes = b.episodes(split, data_root=data_root, scene_root=scene_root, **loader_kwargs)
    if not episodes:
        raise ValueError(f"{benchmark_name}/{split}: no episodes")
    body = body or b.body
    if b.engine == "robotwin":
        common = dict(episodes=episodes, body=body, budget=b.budget, gpu_id=gpu_id, render_mode=render_mode,
                      root=robotwin_root)
        return RobotwinPoseEnv(**common) if b.macro else RobotwinEnv(**common)
    if b.engine == "libero":
        common = dict(episodes=episodes, body=body, max_ticks=b.ticks, gpu_id=gpu_id, render_mode=render_mode)
        return LiberoPoseEnv(**common) if b.macro else LiberoEnv(**common)
    if b.engine == "robocasa":
        common = dict(episodes=episodes, body=body, max_ticks=b.ticks, tick_scale=b.tick_scale, gpu_id=gpu_id,
                      render_mode=render_mode)
        return RobocasaPoseEnv(**common) if b.macro else RobocasaEnv(**common)
    if b.engine == "calvin":
        # ``ticks`` and the macro-move cap are PER SUB-TASK on this engine: a CALVIN chain
        # gives every instruction its own budget and ends at the first failure (calvin.py).
        common = dict(episodes=episodes, body=body, max_ticks=b.ticks, gpu_id=gpu_id, render_mode=render_mode)
        if b.macro:
            from .calvin import MACRO_STEPS_PER_SUBTASK
            return CalvinPoseEnv(max_moves_per_subtask=MACRO_STEPS_PER_SUBTASK, **common)
        return CalvinEnv(**common)
    if b.engine == "behavior":
        common = dict(episodes=episodes, body=body, max_ticks=b.ticks, tick_scale=b.tick_scale, gpu_id=gpu_id,
                      render_mode=render_mode, data_path=behavior_root)
        return BehaviorPoseEnv(**common) if b.macro else BehaviorEnv(**common)
    if b.engine == "isaac":
        common = dict(episodes=episodes, body=body, gpu_id=gpu_id, render_mode=render_mode)
        if b.polar if polar is None else polar:
            return IsaacPolarEnv(**common)
        return IsaacEnv(actions=actions or b.actions, **common)
    common = dict(episodes=episodes, body=body, budget=b.budget, dtg_policy=b.dtg_policy, gpu_id=gpu_id,
                  sim_seed=sim_seed, render_mode=render_mode)
    if b.pose:
        return HabitatPoseEnv(snap=b.pose_snap, **common)
    return HabitatEnv(actions=actions or b.actions, **common)


def make(name: str, split: str, *, variant: str | None = None, metrics: bool = True, depth: bool = True,
         depth_spec: Any = None, metric_overrides: dict[str, Any] | None = None, **kwargs: Any) -> gym.Env:
    """The standard stack: gym.make (PassiveEnvChecker / OrderEnforcing / TimeLimit
    when the benchmark has a static budget) -> the benchmark's metric wrapper ->
    DepthClip -> DynamicTimeLimit (per-episode budgets). ``variant`` picks the
    declaration (``"upstream"`` == ``make(f"{name}-upstream", ...)``; ``None``
    takes ``name`` literally, so a bare line name is its standard variant).
    ``kwargs`` go to :func:`build_env` (data_root, scene_root, body, actions,
    gpu_id, loader options); ``depth_spec`` replaces the declared depth
    post-processing; ``metric_overrides`` (e.g. ``success_distance``) reach the
    metric factory."""
    b = benchmark(name, variant)
    env = gym.make(b.gym_id, split=split, **kwargs)
    if metrics and b.metrics is not None:
        env = b.metrics(env, **(metric_overrides or {}))
    spec = depth_spec if depth_spec is not None else b.depth
    body = kwargs.get("body") or b.body
    if depth and spec is not None and body.depth is not None:
        env = DepthClip(env, spec.min_m, spec.max_m, spec.normalize)
    if b.budget is not None and b.truncate_at_budget:
        env = DynamicTimeLimit(env)
    return env


__all__ = ["BENCHMARKS", "benchmark", "build_env", "make", "resolve"]
