"""The vocabulary every benchmark is written in (layer L1, pure data).

* :class:`Act` — the global action ids. A benchmark's action table is a prefix
  of this sequence, so the integer an agent sends means the same thing in every
  environment of the package.
* Goals — what an episode asks the agent to reach, as typed data. The body
  derives ``distance_to_goal`` from :func:`targets_of`; the metric wrappers
  never look inside a goal beyond its ``kind``.
* :class:`Episode` — one evaluation unit: where it starts, what its goal is,
  what the dataset says about it. Poses are in the engine's own frame — the
  frame the split files store them in: habitat lines y up with ``(x, y, z, w)``
  quaternions, Isaac lines z up with ``(w, x, y, z)``. A manipulation episode
  (LIBERO) has no start pose — its start is an init state, indexed in ``info``;
  a CALVIN episode's start is the state its symbolic initial condition builds,
  carried in ``info`` too.
* :class:`Benchmark` — a declaration (no code of its own) that ``make()`` turns
  into a Gymnasium stack; ``engine`` names the simulator it runs on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np

from .sim import (BehaviorBody, BehaviorSceneRef, Body, CalvinBody, CalvinSceneRef, IsaacBody, IsaacSceneRef, LiberoBody, LiberoSceneRef,
                  RobocasaBody, RobocasaSceneRef, RobotwinBody, RobotwinSceneRef, SceneRef)

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]     # habitat: x, y, z, w — Isaac: w, x, y, z

ENGINES = ("habitat", "isaac", "libero", "robotwin", "robocasa", "calvin", "behavior")


class Act(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOK_UP = 4
    LOOK_DOWN = 5
    SUBTASK_STOP = 6


def action_prefix(n: int) -> tuple[Act, ...]:
    """The first ``n`` actions: (STOP,) (STOP, FORWARD, LEFT, RIGHT), ..."""
    return tuple(Act(i) for i in range(n))


# ---- goals -------------------------------------------------------------------

@dataclass(frozen=True)
class PointGoal:
    position: Vec3
    radius: float = 3.0
    kind: str = field(default="point", init=False)


@dataclass(frozen=True)
class ObjectInstance:
    object_id: str
    category: str
    position: Vec3
    view_points: tuple[Vec3, ...]        # agent positions from which the object counts as reached


@dataclass(frozen=True)
class ObjectGoal:
    """Reach any instance of ``category`` in the scene (ObjectNav / OVON / GOAT-object)."""
    category: str
    instances: tuple[ObjectInstance, ...]
    kind: str = field(default="object", init=False)


@dataclass(frozen=True)
class ImageGoal:
    """Reach the instance shown from a stored camera pose (GOAT-image)."""
    instance: ObjectInstance
    camera_position: Vec3
    camera_rotation: Quat
    hfov_deg: float
    image_size: tuple[int, int]          # (height, width)
    kind: str = field(default="image", init=False)


@dataclass(frozen=True)
class TextGoal:
    """Reach the instance a description refers to (GOAT-description)."""
    instance: ObjectInstance
    description: str
    kind: str = field(default="description", init=False)


@dataclass(frozen=True)
class GoalSequence:
    """Ordered sub-goals, reached one at a time. *Who* closes a sub-goal is
    the line's rule, not the type's: GOAT's the agent declares with
    SUBTASK_STOP, CALVIN's the task oracle closes automatically the tick its
    predicate holds (``benchmarks/calvin.py`` argues the case). The sub-goals
    are the line's own goal type — nav goals on GOAT, ``ManipGoal`` on
    CALVIN."""
    goals: tuple["Goal", ...]
    kind: str = field(default="sequence", init=False)


@dataclass(frozen=True)
class Question:
    """Answer a question about the scene (HM-EQA / MT-HM3D / EXPRESS). The
    answer is not an environment concept — it is compared outside. ``target``
    is the optional place the question is about (EXPRESS's goal_position)."""
    text: str
    answer: str
    choices: tuple[str, ...] | None = None
    target: PointGoal | None = None
    kind: str = field(default="question", init=False)


@dataclass(frozen=True)
class ManipGoal:
    """Make the scene satisfy a set of predicates: LIBERO's BDDL goal state,
    e.g. ``(("On", "akita_black_bowl_1", "plate_1"),)``, or BEHAVIOR's BDDL
    goal conditions, one entry per top-level predicate of the activity's
    ``problem0.bddl``. RoboTwin writes its
    goal as python rather than as declarative predicates, so its lines name
    the check instead — ``(("check_success", "beat_block_hammer"),)``. Either
    way success is the simulator's own check; the goal has no place, so
    ``targets_of`` is None."""
    predicates: tuple[tuple[str, ...], ...]
    kind: str = field(default="manip", init=False)


Goal = Union[PointGoal, ObjectGoal, ImageGoal, TextGoal, GoalSequence, Question, ManipGoal]


def targets_of(goal: Goal | None, dtype: Any = np.float32) -> np.ndarray | None:
    """The (N, 3) point set ``distance_to_goal`` is measured to, or None when
    the goal has no place (a Question without target). float32 is habitat's
    working precision; the Isaac body asks for float64 (its distance is
    plain arithmetic, and the evaluator's NE is computed in float64)."""
    if goal is None:
        return None
    if isinstance(goal, PointGoal):
        return np.asarray([goal.position], dtype=dtype)
    if isinstance(goal, ObjectGoal):
        pts = [vp for inst in goal.instances for vp in inst.view_points]
        return np.asarray(pts, dtype=dtype).reshape(-1, 3)
    if isinstance(goal, (ImageGoal, TextGoal)):
        return np.asarray(goal.instance.view_points, dtype=dtype).reshape(-1, 3)
    if isinstance(goal, Question):
        return targets_of(goal.target, dtype)
    if isinstance(goal, ManipGoal):
        return None
    if isinstance(goal, GoalSequence):
        raise TypeError("targets_of(GoalSequence): pass the current sub-goal")
    raise TypeError(f"not a goal: {goal!r}")


def to_dict(obj: Any) -> Any:
    """dataclasses -> plain dicts (goals keep their ``kind``); tuples -> lists."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---- episode -----------------------------------------------------------------

@dataclass(frozen=True)
class Episode:
    index: int                           # position in the loaded list
    episode_id: str
    scene: SceneRef | IsaacSceneRef | LiberoSceneRef | RobotwinSceneRef | RobocasaSceneRef | CalvinSceneRef | BehaviorSceneRef
    start_position: Vec3 | None          # the engine's frame (habitat y up; Isaac z up); None on a manipulation line
    start_rotation: Quat | None          # habitat x, y, z, w — Isaac w, x, y, z; None on a manipulation line
    goal: Goal | None                    # None when the split withholds it (VLNverse test / challenge)
    instruction: str | None = None       # VLN: what to follow
    reference_path: tuple[Vec3, ...] | None = None   # VLN: waypoints from the dataset
    gt_path: tuple[Vec3, ...] | None = None          # VLN: dense oracle locations (nDTW reference)
    info: dict[str, Any] = field(default_factory=dict)   # everything else the dataset says

    def as_dict(self) -> dict[str, Any]:
        return to_dict(self)


# ---- benchmark declaration ---------------------------------------------------

@dataclass(frozen=True)
class DepthSpec:
    """How depth leaves the stack: clip to [min, max] metres, optionally scaled to [0, 1]."""
    min_m: float
    max_m: float
    normalize: bool


@dataclass(frozen=True)
class Benchmark:
    name: str                                    # make() key, e.g. "objectnav-hm3d-v1" / "objectnav-hm3d-v1-upstream"
    gym_id: str                                  # e.g. "EmbodiedScore/ObjectNav-HM3Dv1-v0"
    # the engine's Body type, checked against ``engine`` in __post_init__
    body: Body | IsaacBody | LiberoBody | RobotwinBody | RobocasaBody | CalvinBody | BehaviorBody
    actions: tuple[Act, ...]
    splits: tuple[str, ...]
    episodes: Callable[..., list[Episode]]      # (split, data_root=None, scene_root=None, **kw) -> episodes
    metrics: Callable[..., Any] | None = None    # (env, **overrides) -> wrapped env
    depth: DepthSpec | None = None
    max_episode_steps: int | None = None         # static budget -> gym TimeLimit
    budget: Callable[[Any, Episode], int] | None = None   # (world, episode) -> per-episode budget
    dtg_policy: str = "when_moved"               # "when_moved" (habitat-lab stock) | "every_step" (goat-bench)
    truncate_at_budget: bool = False             # DynamicTimeLimit on info["step_budget"] (pose protocols)
    pose: bool = False                           # habitat: HabitatPoseEnv (Box [x, z, yaw] teleport) instead of HabitatEnv
    pose_snap: bool = False                      # pose env snaps targets to the navmesh
    polar: bool = False                          # isaac: IsaacPolarEnv (Box [angle, distance, elevation]) instead of IsaacEnv
    macro: bool = False                          # manipulation engines: the pose protocol (absolute end-effector targets, closed loop) instead of the upstream per-tick one
    ticks: int | None = None                     # libero: the control-tick cap (truncation); the macro protocol's guard, the per-tick protocol's budget. None on robotwin — its cap is per task, through ``budget``
    tick_scale: float = 1.0                      # robocasa / behavior: with ``ticks`` None the cap is this
                                                 # multiple of the episode's own horizon (both are per task)
    engine: str = "habitat"                      # "habitat" (habitat-sim, in process) | "isaac" (Isaac Sim render worker) |
                                                 # "libero" / "robocasa" (robosuite / MuJoCo, in process) | "robotwin"
                                                 # (SAPIEN 3) | "behavior" (OmniGibson on Isaac Sim, in process)
    description: str = ""
    line: str = ""                               # the benchmark line both variants belong to, e.g. "objectnav-hm3d-v1"
    variant: str = "standard"                    # "standard" (EmbodiedScore's shared body) | "upstream" (the line's own evaluator)

    def __post_init__(self) -> None:
        if self.variant not in ("standard", "upstream"):
            raise ValueError(f"{self.name}: variant must be 'standard' or 'upstream'")
        if self.engine not in ENGINES:
            raise ValueError(f"{self.name}: engine must be one of {ENGINES}")
        want = {"habitat": Body, "isaac": IsaacBody, "libero": LiberoBody, "robotwin": RobotwinBody,
                "robocasa": RobocasaBody, "calvin": CalvinBody, "behavior": BehaviorBody}[self.engine]
        if not isinstance(self.body, want):
            raise TypeError(f"{self.name}: engine {self.engine!r} needs a {want.__name__}, got {type(self.body).__name__}")
        if (self.pose and self.engine != "habitat" or self.polar and self.engine != "isaac"
                or self.macro and self.engine not in ("libero", "robotwin", "robocasa", "calvin", "behavior")):
            raise ValueError(f"{self.name}: pose is a habitat protocol, polar an isaac one, "
                             "macro a manipulation one (libero / robotwin / robocasa / calvin / behavior)")
        if not self.line:
            object.__setattr__(self, "line", self.name[: -len("-upstream")] if self.name.endswith("-upstream") else self.name)


# ---- data roots ----------------------------------------------------------------

def data_root(explicit: str | os.PathLike | None = None) -> Path:
    p = explicit or os.environ.get("EMBODIEDSCORE_DATA_ROOT")
    if not p:
        raise ValueError("data_root not given and EMBODIEDSCORE_DATA_ROOT unset")
    return Path(p)


def scene_root(explicit: str | os.PathLike | None = None) -> Path:
    p = explicit or os.environ.get("EMBODIEDSCORE_SCENE_ROOT")
    if not p:
        raise ValueError("scene_root not given and EMBODIEDSCORE_SCENE_ROOT unset")
    return Path(p)
