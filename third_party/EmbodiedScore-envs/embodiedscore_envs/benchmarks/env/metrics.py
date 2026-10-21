"""Metric wrappers (layer L1). They read only the facts the body puts in
``info`` — never the simulator — and write ``info["metrics"]`` after reset and
after every step.

``NavMetrics`` is the one formula set shared by every point / object goal
benchmark, transcribed from habitat-lab (``habitat/tasks/nav/nav.py``:
DistanceToGoal, Success, SPL, SoftSPL — identical in 0.1.7 and 0.2.4) and
VLN-CE (``habitat_extensions/measures.py``: NDTW, PathLength, OracleSuccess,
StepsTaken), quirks included:

* ``distance_to_goal``  the body's fact (geodesic to the nearest target), every step
* ``success``           stop_called and distance_to_goal < success_distance
* ``spl``               success * d0 / max(d0, L); d0 = distance_to_goal at reset
                        (from the simulator, not the dataset), L = sum of *euclidean*
                        step lengths
* ``soft_spl``          max(0, 1 - dtg / d0) * d0 / max(d0, L), every step
* ``path_length``       L (upstream's docstring says geodesic; the code is euclidean and
                        the code is what scored the boards; the norm is float32 like
                        habitat's); ``path_length_from="euclid64"`` is the same in float64
                        (EXPRESS's own accounting), ``"step_geodesic"`` sums the pose env's
                        per-step geodesics instead (EXPRESS pose protocol)
* ``ndtw``              FastDTW(agent locations, episode.gt_path, euclidean, radius 1) —
                        ``dtw.py``, a transcription of the fastdtw package's Cython build,
                        the one that scored the boards; exp(-dtw / (len(gt) *
                        success_distance)); the location list starts with the start
                        position and skips a step whose position equals the previous one
* ``oracle_success``    1.0 once distance_to_goal < success_distance at any point, reset included
* ``steps_taken``       number of step() calls, STOP included

A benchmark picks the keys it reports. ``SequenceNavMetrics`` is the GOAT-Bench
accounting for a GoalSequence (per-sub-goal restart of d0 and L), transcribed
from goat-bench's ``measures.py`` (GoatSuccess / GoatSPL / GoatSoftSPL).
``VLNVerseMetrics`` is the VLNverse evaluator's own formula set (its
docstring lists where it departs from the definitions above). ``ManipMetrics``
is the manipulation lines' success rate; ``ChainMetrics`` is CALVIN's
long-horizon accounting over a chain of sub-tasks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np

NAV_KEYS = ("distance_to_goal", "success", "spl", "soft_spl", "ndtw", "path_length",
            "oracle_success", "steps_taken")
VLN_KEYS = ("distance_to_goal", "success", "spl", "ndtw", "path_length", "oracle_success", "steps_taken")
OBJECTNAV_KEYS = ("distance_to_goal", "success", "spl", "soft_spl")
MANIP_KEYS = ("success", "steps_taken", "ticks")
# BEHAVIOR scores more than success — see ``BehaviorMetrics`` for the source of each number.
BEHAVIOR_KEYS = ("success", "q_score", "steps_taken", "ticks", "simulator_time_s", "normalized_time",
                 "agent_distance_base", "agent_distance_left", "agent_distance_right",
                 "normalized_agent_distance_base", "normalized_agent_distance_left",
                 "normalized_agent_distance_right")


def _euclid(a, b) -> float:
    """habitat's ``_euclidean_distance`` on float32 agent positions (the norm is
    taken in float32, the running sum in float64 — mirrored exactly)."""
    return float(np.linalg.norm(np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32), ord=2))


def _euclid64(a, b) -> float:
    return float(np.linalg.norm(np.array(b) - np.array(a), ord=2))


def _div(a: float, b: float) -> float:
    """a / b with IEEE semantics (inf / inf = nan, x / 0 = inf|nan) instead of raising."""
    with np.errstate(all="ignore"):
        return float(np.float64(a) / np.float64(b))


def _ratio(d0: float, L: float) -> float:
    """habitat's ``start_end / max(start_end, agent_distance)`` — argument order kept,
    so an inf d0 gives nan and a nan d0 stays nan."""
    return _div(d0, max(d0, L))


def _fastdtw():
    from .dtw import fastdtw  # numba; imported here so lines without nDTW never JIT
    return fastdtw


class NavMetrics(gym.Wrapper):
    def __init__(self, env: gym.Env, success_distance: float = 3.0, keys: Sequence[str] = NAV_KEYS,
                 path_length_from: str = "euclid") -> None:
        super().__init__(env)
        self._sd = float(success_distance)
        self._keys = tuple(keys)
        unknown = set(self._keys) - set(NAV_KEYS)
        if unknown:
            raise ValueError(f"unknown metric keys {sorted(unknown)}")
        if path_length_from not in ("euclid", "euclid64", "step_geodesic"):
            raise ValueError("path_length_from must be 'euclid', 'euclid64' or 'step_geodesic'")
        self._pl_from = path_length_from
        self._dtw = _fastdtw() if "ndtw" in self._keys else None
        self._m: dict[str, float] = {}
        self._d0 = math.nan
        self._prev: list[float] = []
        self._L = 0.0
        self._steps = 0.0
        self._oracle = 0.0
        self._locations: list[list[float]] = []
        self._gt: list[list[float]] = []

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self._m)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        pos = list(info["position"])
        dtg = info.get("distance_to_goal")
        self._d0 = math.nan if dtg is None else float(dtg)
        self._prev = pos
        self._L = 0.0
        self._steps = 0.0
        self._oracle = float(self._d0 < self._sd) if not math.isnan(self._d0) else 0.0
        self._locations = []
        self._gt = list(info["episode"].get("gt_path") or []) if "ndtw" in self._keys else []
        self._m = self._compute(pos, self._d0, stop=False)
        return obs, self._annotate(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = list(info["position"])
        dtg = info.get("distance_to_goal")
        dtg = math.nan if dtg is None else float(dtg)
        if self._pl_from == "euclid":
            self._L += _euclid(self._prev, pos)
        elif self._pl_from == "euclid64":
            self._L += _euclid64(self._prev, pos)
        else:
            g = float(info.get("step_geodesic", math.inf))
            if math.isfinite(g):
                self._L += g
        self._prev = pos
        self._steps += 1.0
        if dtg < self._sd:
            self._oracle = 1.0
        self._m = self._compute(pos, dtg, stop=bool(info.get("stop_called", False)))
        return obs, reward, terminated, truncated, self._annotate(info)

    # ---- formulas ------------------------------------------------------------
    def _compute(self, pos: list[float], dtg: float, *, stop: bool) -> dict[str, float]:
        d0, L = self._d0, self._L
        out: dict[str, float] = {}
        success = float(stop and dtg < self._sd)
        for k in self._keys:
            if k == "distance_to_goal":
                out[k] = dtg
            elif k == "success":
                out[k] = success
            elif k == "spl":
                out[k] = success * _ratio(d0, L)                      # habitat: success * d0 / max(d0, L)
            elif k == "soft_spl":
                out[k] = max(0.0, 1.0 - _div(dtg, d0)) * _ratio(d0, L)
            elif k == "ndtw":
                out[k] = self._ndtw_update(pos)
            elif k == "path_length":
                out[k] = L
            elif k == "oracle_success":
                out[k] = self._oracle
            elif k == "steps_taken":
                out[k] = self._steps
        return out

    def _ndtw_update(self, pos: list[float]) -> float:
        if self._locations and pos == self._locations[-1]:
            return self._m.get("ndtw", 0.0)
        self._locations.append(pos)
        if not self._gt:
            return math.nan
        dtw_distance = self._dtw(self._locations, self._gt)[0]   # euclidean in float64, as _euclid64
        return float(np.exp(-dtw_distance / (len(self._gt) * self._sd)))

    def _annotate(self, info: dict[str, Any]) -> dict[str, Any]:
        info["metrics"] = dict(self._m)
        info["metrics_valid"] = math.isfinite(self._d0)
        return info


_MODALITY = {"object": "object", "image": "image", "description": "description"}


class SequenceNavMetrics(gym.Wrapper):
    """GOAT-Bench accounting over a GoalSequence.

    Per sub-goal *i* (closed by SUBTASK_STOP): ``subtask_success[i] =
    prev_dtg < success_distance`` where prev_dtg is the distance to the sub-goal
    being closed at the stopping pose (the fact of the step *before* the
    switch); ``spl_by_subtask[i] = subtask_success[i] * se / max(se, walked)``
    with ``se`` the distance to that sub-goal from where it started and
    ``walked`` the euclidean path since — both restart at every switch.
    ``success`` = partial_success (successes / sub-goals), ``spl`` =
    composite_spl (mean of per-sub-goal SPL), ``soft_spl`` = composite_softspl
    (accumulated only on closing steps). ``task_success`` additionally needs
    STOP; ``{object,image,description}_success`` are per-modality rates,
    short-circuited to 0.0 when that modality has no success (upstream quirk:
    an absent modality reads the same as a failed one — the ``n_*_subtasks``
    counts tell them apart).
    """

    def __init__(self, env: gym.Env, success_distance: float = 0.25) -> None:
        super().__init__(env)
        self._sd = float(success_distance)
        self._m: dict[str, Any] = {}

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self._m)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        goals = info["episode"]["goal"]["goals"]
        self._mods = [_MODALITY[g["kind"]] for g in goals]
        self._n = len(goals)
        self._counts = {m: self._mods.count(m) for m in _MODALITY.values()}
        self._succ = [0.0] * self._n
        self._spl = [0.0] * self._n
        self._succ_by = {m: 0.0 for m in _MODALITY.values()}
        self._spl_by = {m: 0.0 for m in _MODALITY.values()}
        self._soft_by = {m: 0.0 for m in _MODALITY.values()}
        dtg = info.get("distance_to_goal")
        self._dtg = math.nan if dtg is None else float(dtg)
        self._prev_dtg = self._dtg
        self._se = self._dtg
        self._walked = 0.0
        self._prev = list(info["position"])
        self._idx = 0
        self._steps = 0.0
        self._stop = False
        self._m = self._compute()
        return obs, self._annotate(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = list(info["position"])
        self._walked += _euclid(self._prev, pos)
        self._prev = pos
        self._steps += 1.0
        self._stop = bool(info.get("stop_called", False))
        dtg = info.get("distance_to_goal")
        new_dtg = self._dtg if dtg is None else float(dtg)
        if info.get("subtask_closed") and self._idx < self._n:
            i, mod = self._idx, self._mods[self._idx]
            ok = float(self._prev_dtg < self._sd)
            self._succ[i] = ok
            self._succ_by[mod] += ok
            se, walked = self._se, self._walked
            spl = ok * _ratio(se, walked)
            self._spl[i] = spl
            self._spl_by[mod] += spl
            soft = max(0.0, 1.0 - _div(self._prev_dtg, se)) * _ratio(se, walked)
            self._soft_by[mod] += soft
            self._idx += 1
            self._se = new_dtg          # the next sub-goal, measured from where the agent stands
            self._walked = 0.0
        self._prev_dtg = new_dtg
        self._dtg = new_dtg
        self._m = self._compute()
        return obs, reward, terminated, truncated, self._annotate(info)

    def _rate(self, num: dict[str, float], mod: str) -> float:
        return num[mod] / self._counts[mod] if num[mod] else 0.0

    def _compute(self) -> dict[str, Any]:
        n = float(self._n)
        all_ok = sum(self._succ) == self._n
        return {
            "success": sum(self._succ_by.values()) / n,
            "spl": sum(self._spl_by.values()) / n,
            "soft_spl": sum(self._soft_by.values()) / n,
            "composite_success": float(all_ok),
            "task_success": float(all_ok and self._stop),
            "subtask_success": list(self._succ),
            "spl_by_subtask": list(self._spl),
            "object_success": self._rate(self._succ_by, "object"),
            "image_success": self._rate(self._succ_by, "image"),
            "description_success": self._rate(self._succ_by, "description"),
            "n_object_subtasks": self._counts["object"],
            "n_image_subtasks": self._counts["image"],
            "n_description_subtasks": self._counts["description"],
            "distance_to_target": self._dtg if math.isfinite(self._dtg) else None,
            "n_subtasks": self._n,
            "subtasks_completed": self._idx,
            "steps_taken": self._steps,
        }

    def _annotate(self, info: dict[str, Any]) -> dict[str, Any]:
        info["metrics"] = dict(self._m)
        info["metrics_valid"] = True
        return info


# ---- VLNverse -------------------------------------------------------------------

VLNVERSE_KEYS = ("distance_to_goal", "success", "oracle_success", "spl", "ndtw", "path_length",
                 "steps_taken", "shortest_path_length")
VLNVERSE_EVALUATOR_NAMES = {"distance_to_goal": "NE", "success": "success", "oracle_success": "osr", "spl": "spl",
                            "ndtw": "ndtw", "path_length": "TL", "steps_taken": "steps",
                            "shortest_path_length": "shortest_path_length"}
_VLNVERSE_GT_KEYS = ("distance_to_goal", "success", "oracle_success", "spl", "ndtw", "shortest_path_length")


class VLNVerseMetrics(gym.Wrapper):
    """The VLNverse evaluator's formula set, transcribed from its ``VLNPEMetrics``
    (``vlnverse/projects/internutopia_vln_extension/metrics/vln_pe_metrics.py``
    in billzhao1030/VLNVerse at ``d444c04``), quirks included. The keys are
    named like ``NavMetrics`` so agents can be shared; ``VLNVERSE_EVALUATOR_NAMES``
    maps them back to the evaluator's own names. Positions are read from
    ``info["position"]`` in the Isaac frame (z up); every formula works in xy.

      key (evaluator name)     formula, as the evaluator computes it
      ---------------------    ------------------------------------------------------------
      distance_to_goal (NE)    ground-plane euclidean distance from the agent to the last
                               point of ``reference_path`` (== goals.position in every split)
      success                  NE < success_distance (3.0 m), STOP or not
      oracle_success (osr)     min NE over the positions *after* each action < 3.0 m — the
                               start pose is not considered
      spl                      success * S / max(TL, S), S = ``info.geodesic_distance`` when
                               > 0 else the polyline length of ``reference_path`` (the released
                               splits carry -1, so it is always the polyline); 0 when TL == 0
      ndtw                     mean over the agent's positions (start included) of
                               exp(-d_min^2 / (2 * 3.0^2)), d_min = distance to the nearest
                               reference-path point (a waypoint, not the segment between two)
                               — a nearest-point similarity, not a DTW
      path_length (TL)         sum of ground-plane euclidean displacements between consecutive
                               positions
      steps_taken (steps)      number of ``step()`` calls, STOP included (the evaluator counts
                               simulator updates, warm-up included — not comparable 1:1)
      shortest_path_length     S above

    Where this differs from ``NavMetrics`` (VLN-CE / habitat-lab; Anderson et
    al. 2018 for SPL, Ilharco et al. 2019 for nDTW):

      distance_to_goal   there: geodesic on the navmesh. Here: straight line in xy.
      success            there: STOP called and distance < 3 m. Here: STOP is not required.
      oracle_success     there: any position on the trajectory, start included. Here: start
                         excluded.
      spl                there: S = geodesic start-goal distance, and spl == success when the
                         agent never moved. Here: S = reference-path polyline; spl = 0 at TL == 0.
      ndtw               there: exp(-DTW(agent, reference) / (|reference| * 3 m)) with DTW the
                         dynamic-time-warping alignment cost. Here: no alignment — the per-point
                         nearest-reference Gaussian similarity, averaged.
      path_length        there: 3-D euclidean. Here: xy (identical on flat floors).

    A third formula set exists: the zero-shot line (billzhao1030/vlnverse_emr_zero_shot,
    ``EnvBridge.final_metrics``) uses success <= 3 m, SPL against the straight-line
    start-goal distance, a true DTW for nDTW, path_length = commanded distance, and
    extra keys (stop_success, colrate, success_efficiency). It is not implemented
    here.

    Test / challenge episodes have no reference path: every ground-truth-derived
    key is -1 (the evaluator's placeholder) and ``info["metrics_valid"]`` is False.
    """

    def __init__(self, env: gym.Env, success_distance: float = 3.0, keys: Sequence[str] = VLNVERSE_KEYS) -> None:
        super().__init__(env)
        self._sd = float(success_distance)
        self._keys = tuple(keys)
        unknown = set(self._keys) - set(VLNVERSE_KEYS)
        if unknown:
            raise ValueError(f"unknown metric keys {sorted(unknown)}")
        self._m: dict[str, float] = {}
        self._valid = False
        self._ref = np.zeros((0, 2))
        self._goal = np.zeros(2)
        self._shortest = 0.0
        self._traj: list[np.ndarray] = []
        self._prev = np.zeros(2)
        self._L = 0.0
        self._steps = 0
        self._ne = math.nan
        self._min_ne = math.inf

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self._m)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ep = info["episode"]
        ref = ep.get("reference_path")
        pos = np.asarray(info["position"], dtype=np.float64)[:2]
        self._valid = bool(ref)
        self._traj = [pos]
        self._prev = pos
        self._L = 0.0
        self._steps = 0
        self._min_ne = math.inf           # the evaluator seeds this with 999 from its config; same outcome
        if self._valid:
            self._ref = np.asarray(ref, dtype=np.float64)
            geo = float(((ep.get("info") or {}).get("geodesic_distance")) or -1.0)
            self._shortest = geo if geo > 0 else float(np.sum(np.linalg.norm(np.diff(self._ref, axis=0), axis=1)))
            self._goal = self._ref[-1, :2]
            # The evaluator has no NE before the first action; a value at reset is this wrapper's addition.
            self._ne = float(np.linalg.norm(pos - self._goal))
        else:
            self._ne = math.nan
        self._m = self._compute()
        return obs, self._annotate(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = np.asarray(info["position"], dtype=np.float64)[:2]
        self._L += float(np.linalg.norm(pos - self._prev))
        self._prev = pos
        self._traj.append(pos)
        self._steps += 1
        if self._valid:
            self._ne = float(np.linalg.norm(pos - self._goal))
            self._min_ne = min(self._min_ne, self._ne)
        self._m = self._compute()
        return obs, reward, terminated, truncated, self._annotate(info)

    # ---- formulas --------------------------------------------------------------------
    def _compute(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if not self._valid:
            for k in self._keys:
                out[k] = -1.0 if k in _VLNVERSE_GT_KEYS else (self._L if k == "path_length" else float(self._steps))
            return out
        success = float(self._ne < self._sd)
        for k in self._keys:
            if k == "distance_to_goal":
                out[k] = self._ne
            elif k == "success":
                out[k] = success
            elif k == "oracle_success":
                out[k] = float(self._min_ne < self._sd)
            elif k == "spl":
                out[k] = success * self._shortest / max(self._L, self._shortest) if self._L > 0 else 0.0
            elif k == "ndtw":
                out[k] = self._ndtw()
            elif k == "path_length":
                out[k] = self._L
            elif k == "steps_taken":
                out[k] = float(self._steps)
            elif k == "shortest_path_length":
                out[k] = self._shortest
        return out

    def _ndtw(self) -> float:
        traj = np.asarray(self._traj, dtype=np.float64)
        ref = self._ref[:, :2]
        if len(traj) == 0:
            return 0.0
        d = np.linalg.norm(traj[:, None, :] - ref[None, :, :], axis=2).min(axis=1)
        return float(np.mean(np.exp(-(d ** 2) / (2.0 * self._sd ** 2))))

    def _annotate(self, info: dict[str, Any]) -> dict[str, Any]:
        info["metrics"] = dict(self._m)
        info["metrics_valid"] = self._valid
        return info


class ManipMetrics(gym.Wrapper):
    """The manipulation lines' accounting (LIBERO): ``success`` is the
    simulator's own goal check — the body's ``info["success"]``, 1.0 from
    the step it first holds (the episode terminates there, so there is no
    separate oracle key); ``steps_taken`` counts ``step()`` calls (macro
    moves on the standard protocol, control ticks on LIBERO's own);
    ``ticks`` is the body's control-tick count. LIBERO reports success rate
    and nothing else; nothing else is computed here."""

    def __init__(self, env: gym.Env, keys: Sequence[str] = MANIP_KEYS) -> None:
        super().__init__(env)
        self.keys = tuple(keys)
        self._steps = 0
        self._success = 0.0

    def _write(self, info: dict[str, Any]) -> dict[str, Any]:
        if info.get("success"):
            self._success = 1.0
        m = {"success": self._success, "steps_taken": self._steps, "ticks": int(info.get("ticks", 0))}
        info["metrics"] = {k: m[k] for k in self.keys}
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._steps, self._success = 0, 0.0
        return obs, self._write(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        return obs, reward, terminated, truncated, self._write(info)


CHAIN_KEYS = ("success", "chain_length", "success_1", "success_2", "success_3", "success_4", "success_5",
              "n_subtasks", "subtasks_completed", "steps_taken", "ticks")


class ChainMetrics(gym.Wrapper):
    """CALVIN's long-horizon accounting — ``ManipMetrics`` for an episode
    that is a *chain* of sub-tasks rather than one task.

    CALVIN scores an episode with a single number: how many of the five
    chained instructions were solved consecutively before the first failure
    (``evaluate_policy.evaluate_sequence``:144 returns ``success_counter``),
    and reports two aggregates over the 1000 episodes
    (``evaluation/utils.py`` ``count_success``:77, ``print_and_save``:87):

        avg_seq_len   np.mean(results)                 "Average successful sequence length"
        chain_sr[i]   fraction of episodes with result >= i, for i = 1..5
                      "Success rates for i instructions in a row"

    Both are means of per-episode facts, so this wrapper reports the facts
    and the run's aggregation takes the mean:

        chain_length          the episode's result, 0..5 — its mean IS avg_seq_len
        success_1..success_5  1.0 when chain_length >= i — their means ARE chain_sr[1..5]
        success               == success_1, so every manipulation line has the key
        n_subtasks            the chain's length (5)
        subtasks_completed    == chain_length, under the package's sub-goal name
        steps_taken           step() calls (macro moves on the standard protocol,
                              control ticks on CALVIN's own)
        ticks                 the body's control-tick count

    Like every wrapper here it reads only ``info``: ``chain_length`` is the
    body's fact, advanced there by CALVIN's task oracle."""

    def __init__(self, env: gym.Env, keys: Sequence[str] = CHAIN_KEYS) -> None:
        super().__init__(env)
        self.keys = tuple(keys)
        unknown = set(self.keys) - set(CHAIN_KEYS)
        if unknown:
            raise ValueError(f"unknown metric keys {sorted(unknown)}")
        self._steps = 0

    def _write(self, info: dict[str, Any]) -> dict[str, Any]:
        chain = int(info.get("chain_length", 0))
        m: dict[str, Any] = {
            "success": float(chain >= 1),
            "chain_length": chain,
            "n_subtasks": int(info.get("n_goals", 0)),
            "subtasks_completed": chain,
            "steps_taken": self._steps,
            "ticks": int(info.get("ticks", 0)),
        }
        for i in range(1, 6):
            m[f"success_{i}"] = float(chain >= i)
        info["metrics"] = {k: m[k] for k in self.keys}
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._steps = 0
        return obs, self._write(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        return obs, reward, terminated, truncated, self._write(info)


class BehaviorMetrics(ManipMetrics):
    """The BEHAVIOR lines' accounting — the 2025 BEHAVIOR Challenge's own
    metric set, transcribed from the code that produced the leaderboard.

    Ranking metric (``docs/challenge/evaluation.md`` § "Metrics and Results":
    "The success score (Q) is the metric used for ranking submissions"; the
    leaderboard averages it over 10 instances x 50 tasks,
    ``learning/utils/score_utils.py`` L114, L122):

    * ``q_score``   ``omnigibson/metrics/task_metric.py`` ``end_callback``
                    L30-45 — 1.0 when the task succeeded, else the LARGEST
                    fraction, over the goal's groundings, of goal predicates
                    that were false at the episode's first state and are true
                    now ("the maximum progress made, across any of the
                    groundings, from the initial state of the task"). The body
                    evaluates it every step and it is reported live; the
                    challenge only reads it at the end of the rollout.
    * ``success``   the same run's binary task success, latched (the body's
                    ``info["success"]``): every goal predicate of some
                    grounding satisfied.

    Efficiency (secondary; the challenge normalises them by the human average
    over that task's 200 demonstrations and uses them only to break ties):

    * ``simulator_time_s``  ticks x the rendering dt (``task_metric.py``
                            ``gather_results``: ``simulator_steps *
                            og.sim.get_rendering_dt()``); here ticks / 30 Hz,
                            the challenge's ``rendering_frequency``.
    * ``normalized_time``   ``human_steps / agent_steps`` (``task_metric.py``
                            ``gather_results``). ``human_steps`` is the mean
                            length of that task's 200 demonstrations, from the
                            release's ``metadata/episodes.jsonl`` — the loader
                            carries it as ``episode.info["human_steps"]``.
                            The leaderboard's derived ``time_score`` is
                            ``2 - 1 / normalized_time`` (``score_utils.py``
                            L98); it is not recomputed here.
    * ``agent_distance_*``  accumulated per-step L2 displacement of the base
                            and of each end effector, in metres
                            (``omnigibson/metrics/agent_metric.py`` L26-45),
                            read off the body's ``info["travel_m"]``.
    * ``normalized_agent_distance_*``  ``human_distance / agent_distance``,
                            ``inf`` when the part never moved
                            (``agent_metric.py`` ``gather_results``); the human
                            distances are ``distance_traveled`` /
                            ``left_eef_displacement`` / ``right_eef_displacement``
                            averaged over the same 200 demonstrations, carried
                            as ``episode.info["human_distance"]``.

    Nothing is invented: a key whose human reference the episode does not carry
    is reported as ``None`` rather than guessed.
    """

    def __init__(self, env: gym.Env, keys: Sequence[str] | None = None, render_hz: float = 30.0) -> None:
        super().__init__(env, keys=BEHAVIOR_KEYS if keys is None else keys)
        self.render_hz = float(render_hz)

    def _reference(self) -> tuple[float | None, dict[str, float]]:
        """(human steps, human distances) of this episode's task, or (None, {})
        when the loader could not read the release's metadata."""
        info = getattr(self.env.unwrapped, "episode", None)
        info = dict(getattr(info, "info", {}) or {})
        steps = info.get("human_steps")
        return (float(steps) if steps else None, {k: float(v) for k, v in (info.get("human_distance") or {}).items()})

    def _write(self, info: dict[str, Any]) -> dict[str, Any]:
        if info.get("success"):
            self._success = 1.0
        ticks = int(info.get("ticks", 0))
        human_steps, human_distance = self._reference()
        travel = dict(info.get("travel_m") or {})
        m: dict[str, Any] = {
            "success": self._success,
            "q_score": 1.0 if self._success else float(info.get("q_score") or 0.0),
            "steps_taken": self._steps,
            "ticks": ticks,
            "simulator_time_s": ticks / self.render_hz if self.render_hz else None,
            "normalized_time": (human_steps / ticks) if (human_steps and ticks) else None,
        }
        for part in ("base", "left", "right"):
            distance = travel.get(part)
            m[f"agent_distance_{part}"] = None if distance is None else float(distance)
            reference = human_distance.get(part)
            m[f"normalized_agent_distance_{part}"] = (
                None if (reference is None or distance is None)
                else (float("inf") if distance == 0.0 else reference / float(distance))
            )
        info["metrics"] = {k: m[k] for k in self.keys}
        return info
