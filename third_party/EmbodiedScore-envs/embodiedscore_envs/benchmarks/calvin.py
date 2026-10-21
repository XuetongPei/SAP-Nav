"""CALVIN: language-conditioned long-horizon manipulation on a play table
(Mees et al., RA-L 2022, `arXiv:2112.03227 <https://arxiv.org/abs/2112.03227>`_;
`github.com/mees/calvin <https://github.com/mees/calvin>`_ at ``fa03f01``,
its ``calvin_env`` submodule at ``1431a46``). A Franka Panda with a parallel
gripper in pybullet, 34 language-conditioned tasks, and an evaluation that is
a *chain*: 1000 sequences of 5 instructions each, given one at a time.

## One line, not two

The published board rows are **ABC->D** and **ABCD->D**. Both name the
*training* split; **the evaluation is the same in both** — the same
environment D and the same 1000 sequences:

* ``evaluate_policy.make_env`` (:48) builds the environment from
  ``<dataset>/validation``, and the validation split of ``task_D_D``,
  ``task_ABC_D`` and ``task_ABCD_D`` is environment D in all three
  (``validation/scene_info.npy`` reads ``{'calvin_scene_D': [...]}``);
* the initial states are not read from the dataset at all — they are computed
  from each sequence's symbolic initial condition
  (``evaluation/utils.py`` ``get_env_state_for_initial_condition``:207);
* the 1000 sequences come from ``get_sequences(1000)``
  (``evaluation/multistep_sequences.py``:350), which takes no dataset.

A zero-shot agent has no training split, so ABC->D and ABCD->D would be the
identical benchmark twice. Hence one line, **``calvin-d``**, named for the
environment the board is actually scored in.

## The episode

``get_sequences(1000)`` returns 1000 ``(initial_condition, five task names)``
pairs, deterministically: 192 symbolic initial conditions (the filtered
product at :352-365), ``np.array_split`` of 1000 over them, per-condition
rejection sampling under ``np.random.seed(i)`` (:333), and one final
``np.random.shuffle`` under seed 0 (:379). ``eval_sequences`` below is that
function transcribed — the process pool replaced by a serial loop with the
same per-condition seeding, verified to produce the identical 1000 pairs
(2026-09-10). Episode index = position in that list; ``mini`` is the first
100.

There is **no sequence file**: the list is generated, not stored. It costs
about two minutes to build the first time (upstream parallelises it over
every core), so it is cached in-process and on disk under the data root.

The episode's goal is a ``GoalSequence`` of five ``ManipGoal``s. Its
sub-goals are closed **automatically by the task oracle**, not by the agent
with SUBTASK_STOP as GOAT's are:

* CALVIN's own semantics are exactly that. ``rollout`` (:147) polls
  ``Tasks.get_task_info_for_set`` every tick and returns the moment it holds
  (:172-176); ``evaluate_sequence`` (:138-144) then reveals the next
  instruction, or returns at the first failure. Nothing in the protocol lets
  the policy declare a sub-task done, and a policy that could would be
  scoring itself.
* The type carries no opinion about who closes a sub-goal —
  ``schema.GoalSequence`` is an ordered tuple of goals; SUBTASK_STOP is
  GOAT's closing rule, not the type's contract. Adding a second sequence
  type for the same data shape would buy nothing.

so the port keeps ``GoalSequence`` and closes it from the oracle, in
``env/calvin_env.py``.

## Metrics

CALVIN's own, through ``ChainMetrics``: the per-episode ``chain_length``
(0..5) whose mean is the evaluator's *average successful sequence length*,
and ``success_1..success_5`` whose means are its *success rates for i
instructions in a row* (``evaluation/utils.py`` ``count_success``:77 and
``print_and_save``:87).

## Variants — the same deliberate bend as the LIBERO and RoboCasa lines

* ``calvin-d-upstream`` is CALVIN as its evaluator runs it: ``CalvinEnv``,
  one 30 Hz control tick per step, the 7-D relative end-effector command,
  the release's 200x200 static + 84x84 gripper rig (``bodies.CALVIN``), and
  ``EP_LEN`` = 360 ticks per sub-task (``evaluate_policy.py``:38).
* ``calvin-d`` (standard) is the macro protocol for agents that plan in
  poses: ``CalvinPoseEnv``, one step = one absolute end-effector target (or
  a gripper hold), 256² cameras (``bodies.CALVIN_STANDARD``),
  ``MACRO_STEPS_PER_SUBTASK`` macro moves per sub-task under a tick guard of
  ``TICK_GUARD`` x ``EP_LEN``. The budget unit differs (macro moves vs
  ticks), as on the other manipulation lines — but the budget stays **per
  sub-task**, because that is what makes a CALVIN chain a chain: an
  instruction that is not solved in its own budget fails, and the chain ends
  there. A shared pool would let an agent spend the whole episode on the
  first instruction and score 1/5 by construction.

## Data

``<data_root>/calvin/calvin_debug_dataset`` — the release's smallest
download (``dataset/download_data.sh debug``, 1.3 GB unpacked). Only
``validation/.hydra/merged_config.yaml`` is read, and only to build the
environment, exactly as ``make_env`` does; ``task_D_D`` (166 GB) carries the
same file. The play-table meshes ship inside the ``calvin_env`` checkout.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
from itertools import product
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .env import Benchmark, CalvinSceneRef, ChainMetrics, Episode, GoalSequence, ManipGoal, data_root
from .presets import bodies

EP_LEN = 360                     # evaluate_policy.py:38 — control ticks per sub-task
NUM_SEQUENCES = 1000             # evaluate_policy.py:39
SEQ_LEN = 5                      # multistep_sequences.py:294 / :335
MACRO_STEPS_PER_SUBTASK = 20     # standard protocol: macro moves per sub-task. Decided 2026-09-10.
TICK_GUARD = 10                  # standard protocol: tick cap per sub-task = TICK_GUARD x EP_LEN
MACRO_STEPS = MACRO_STEPS_PER_SUBTASK * SEQ_LEN     # the episode ceiling (gym TimeLimit): 100, as on libero-*
SPLITS = ("all", "mini")
MINI_SEQUENCES = 100
ENV_LETTER = "D"                 # the environment every published CALVIN row is evaluated in
DATASET_DIR = "calvin_debug_dataset"     # under <data_root>/calvin/
CONFIG_REL = ("validation", ".hydra", "merged_config.yaml")

# ---- the task table, transcribed -------------------------------------------------------
# The 34 tasks. THE ORDER IS LOAD-BEARING: the sequence generator draws with
# ``np.random.choice(list(tasks.keys()), size=5, replace=False)``
# (``multistep_sequences.py``:340), which indexes into this list, so a different order
# would give different sequences. It is the key order of ``multistep_sequences.tasks``
# (:53-259), which is also the order ``calvin_env/conf/tasks/new_playtable_tasks.yaml``
# declares — hence also the oracle's ``Tasks.task_to_id``. The success predicates
# themselves are never copied: they are read from the installed calvin_env at runtime
# (``sim/calvin/world.py`` ``task_oracle``).
TASKS = (
    "rotate_red_block_right", "rotate_red_block_left", "rotate_blue_block_right", "rotate_blue_block_left",
    "rotate_pink_block_right", "rotate_pink_block_left", "push_red_block_right", "push_red_block_left",
    "push_blue_block_right", "push_blue_block_left", "push_pink_block_right", "push_pink_block_left",
    "move_slider_left", "move_slider_right", "open_drawer", "close_drawer",
    "lift_red_block_table", "lift_red_block_slider", "lift_red_block_drawer",
    "lift_blue_block_table", "lift_blue_block_slider", "lift_blue_block_drawer",
    "lift_pink_block_table", "lift_pink_block_slider", "lift_pink_block_drawer",
    "place_in_slider", "place_in_drawer", "stack_block", "unstack_block",
    "turn_on_lightbulb", "turn_off_lightbulb", "turn_on_led", "turn_off_led", "push_into_drawer",
)

# ---- the instructions, transcribed ------------------------------------------------------
# calvin_models/conf/annotations/new_playtable_validation.yaml — the single validation
# annotation of each task, which the evaluator hands the policy as the sub-task's goal
# (evaluate_policy.rollout:156 ``val_annotations[subtask][0]``). Kept here rather than read
# from the checkout because it lives in ``calvin_models``, whose imports need torch and
# pytorch-lightning; these lines install only ``calvin_env``. ``tests/test_calvin_contracts.py``
# checks the table against the checkout when one is present.
INSTRUCTIONS = {
    "rotate_red_block_right": "take the red block and rotate it to the right",
    "rotate_red_block_left": "take the red block and rotate it to the left",
    "rotate_blue_block_right": "take the blue block and rotate it to the right",
    "rotate_blue_block_left": "take the blue block and rotate it to the left",
    "rotate_pink_block_right": "take the pink block and rotate it to the right",
    "rotate_pink_block_left": "take the pink block and rotate it to the left",
    "push_red_block_right": "go push the red block right",
    "push_red_block_left": "go push the red block left",
    "push_blue_block_right": "go push the blue block right",
    "push_blue_block_left": "go push the blue block left",
    "push_pink_block_right": "go push the pink block right",
    "push_pink_block_left": "go push the pink block left",
    "move_slider_left": "push the sliding door to the left side",
    "move_slider_right": "push the sliding door to the right side",
    "open_drawer": "pull the handle to open the drawer",
    "close_drawer": "push the handle to close the drawer",
    "lift_red_block_table": "grasp and lift the red block",
    "lift_blue_block_table": "grasp and lift the blue block",
    "lift_pink_block_table": "grasp and lift the pink block",
    "lift_red_block_slider": "lift the red block from the sliding cabinet",
    "lift_blue_block_slider": "lift the blue block from the sliding cabinet",
    "lift_pink_block_slider": "lift the pink block from the sliding cabinet",
    "lift_red_block_drawer": "Take the red block from the drawer",
    "lift_blue_block_drawer": "Take the blue block from the drawer",
    "lift_pink_block_drawer": "Take the pink block from the drawer",
    "place_in_slider": "store the grasped block in the sliding cabinet",
    "place_in_drawer": "store the grasped block in the drawer",
    "push_into_drawer": "slide the block that it falls into the drawer",
    "stack_block": "stack the grasped block",
    "unstack_block": "remove the stacked block",
    "turn_on_lightbulb": "use the switch to turn on the light bulb",
    "turn_off_lightbulb": "use the switch to turn off the light bulb",
    "turn_on_led": "press the button to turn on the led light",
    "turn_off_led": "press the button to turn off the led light",
}

# ---- the sequence generator, transcribed ------------------------------------------------
# multistep_sequences.py. ``task_categories`` (:16) and ``tasks`` (:53) drive the symbolic
# planner that decides which chains are *possible* from a given initial condition; only the
# task-category map and the condition/effect table are needed for that, and both are copied
# verbatim. (The physical success check is the oracle's, at runtime — not this table.)

CATEGORIES = {
    "rotate_red_block_right": 1, "rotate_red_block_left": 1, "rotate_blue_block_right": 1,
    "rotate_blue_block_left": 1, "rotate_pink_block_right": 1, "rotate_pink_block_left": 1,
    "push_red_block_right": 1, "push_red_block_left": 1, "push_blue_block_right": 1,
    "push_blue_block_left": 1, "push_pink_block_right": 1, "push_pink_block_left": 1,
    "move_slider_left": 2, "move_slider_right": 2, "open_drawer": 3, "close_drawer": 3,
    "lift_red_block_table": 4, "lift_red_block_slider": 5, "lift_red_block_drawer": 6,
    "lift_blue_block_table": 4, "lift_blue_block_slider": 5, "lift_blue_block_drawer": 6,
    "lift_pink_block_table": 4, "lift_pink_block_slider": 5, "lift_pink_block_drawer": 6,
    "place_in_slider": 7, "place_in_drawer": 7, "turn_on_lightbulb": 8, "turn_off_lightbulb": 8,
    "turn_on_led": 8, "turn_off_led": 8, "push_into_drawer": 9, "stack_block": 10, "unstack_block": 11,
}


def _block_rules(colour: str) -> dict[str, list[dict]]:
    """The per-colour rows of ``multistep_sequences.tasks``, which differ only
    in the block's name (rotate / push / lift-from-table / -slider / -drawer)."""
    b = f"{colour}_block"
    return {
        f"rotate_{colour}_block_right": [{"condition": {b: "table", "grasped": 0}, "effect": {b: "table"}}],
        f"rotate_{colour}_block_left": [{"condition": {b: "table", "grasped": 0}, "effect": {b: "table"}}],
        f"push_{colour}_block_right": [{"condition": {b: "table", "grasped": 0}, "effect": {b: "table"}}],
        f"push_{colour}_block_left": [{"condition": {b: "table", "grasped": 0}, "effect": {b: "table"}}],
        f"lift_{colour}_block_table": [
            {"condition": {b: "table", "grasped": 0}, "effect": {b: "grasped", "grasped": 1}}],
        f"lift_{colour}_block_slider": [
            {"condition": {b: "slider_left", "slider": "right", "grasped": 0},
             "effect": {b: "grasped", "grasped": 1}},
            {"condition": {b: "slider_right", "slider": "left", "grasped": 0},
             "effect": {b: "grasped", "grasped": 1}}],
        f"lift_{colour}_block_drawer": [
            {"condition": {b: "drawer", "drawer": "open", "grasped": 0},
             "effect": {b: "grasped", "grasped": 1}}],
    }


COLOURS = ("red", "blue", "pink")


def _build_rules() -> dict[str, list[dict]]:
    rules: dict[str, list[dict]] = {}
    for colour in COLOURS:
        rules.update(_block_rules(colour))
    rules.update({
        "move_slider_left": [{"condition": {"slider": "right", "grasped": 0}, "effect": {"slider": "left"}}],
        "move_slider_right": [{"condition": {"slider": "left", "grasped": 0}, "effect": {"slider": "right"}}],
        "open_drawer": [{"condition": {"drawer": "closed", "grasped": 0}, "effect": {"drawer": "open"}}],
        "close_drawer": [{"condition": {"drawer": "open", "grasped": 0}, "effect": {"drawer": "closed"}}],
        "place_in_slider": [
            {"condition": {f"{c}_block": "grasped", "slider": side, "grasped": 1},
             "effect": {f"{c}_block": f"slider_{side}", "grasped": 0}}
            for c in COLOURS for side in ("right", "left")],
        "place_in_drawer": [
            {"condition": {f"{c}_block": "grasped", "drawer": "open", "grasped": 1},
             "effect": {f"{c}_block": "drawer", "grasped": 0}} for c in COLOURS],
        "stack_block": [
            {"condition": {f"{a}_block": "grasped", f"{b}_block": "table", "grasped": 1},
             "effect": {f"{a}_block": "stacked_top", f"{b}_block": "stacked_bottom", "grasped": 0}}
            for a in COLOURS for b in COLOURS if a != b],
        "unstack_block": [
            {"condition": {f"{a}_block": "stacked_top", f"{b}_block": "stacked_bottom", "grasped": 0},
             "effect": {f"{a}_block": "table", f"{b}_block": "table"}}
            for a in COLOURS for b in COLOURS if a != b],
        "turn_on_lightbulb": [{"condition": {"lightbulb": 0, "grasped": 0}, "effect": {"lightbulb": 1}}],
        "turn_off_lightbulb": [{"condition": {"lightbulb": 1, "grasped": 0}, "effect": {"lightbulb": 0}}],
        "turn_on_led": [{"condition": {"led": 0, "grasped": 0}, "effect": {"led": 1}}],
        "turn_off_led": [{"condition": {"led": 1, "grasped": 0}, "effect": {"led": 0}}],
        "push_into_drawer": [
            {"condition": {f"{c}_block": "table", "drawer": "open", "grasped": 0,
                           **{f"{o}_block": ["slider_right", "slider_left"] for o in COLOURS if o != c}},
             "effect": {f"{c}_block": "drawer", "grasped": 0}} for c in COLOURS],
    })
    # the order the upstream dict literal declares, which np.random.choice draws from
    return {name: rules[name] for name in TASKS}


RULES = _build_rules()

POSSIBLE_CONDITIONS = {      # multistep_sequences.get_sequences:352
    "led": [0, 1], "lightbulb": [0, 1], "slider": ["right", "left"], "drawer": ["closed", "open"],
    "red_block": ["table", "slider_right", "slider_left"],
    "blue_block": ["table", "slider_right", "slider_left"],
    "pink_block": ["table", "slider_right", "slider_left"], "grasped": [0],
}


@contextlib.contextmanager
def _temp_seed(seed: int) -> Iterator[None]:
    """``evaluation/utils.py`` :197. Also what gives each initial condition
    the isolation upstream's process pool gives it for free."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _check_condition(state: dict, condition: dict) -> bool:      # multistep_sequences.py:262
    for k, v in condition.items():
        if isinstance(v, (str, int)):
            if not state[k] == v:
                return False
        elif isinstance(v, list):
            if state[k] not in v:
                return False
        else:
            raise TypeError
    return True


def _valid_task(state: dict, rule: list[dict]) -> list[dict]:    # multistep_sequences.py:282
    out = []
    for one in rule:
        if _check_condition(state, one["condition"]):
            nxt = dict(state)
            nxt.update(one["effect"])
            out.append(nxt)
    return out


def check_sequence(state: dict, seq) -> bool:                    # multistep_sequences.py:323
    for name in seq:
        states = _valid_task(state, RULES[name])
        if len(states) != 1:
            return False
        state = states[0]
    cats = [CATEGORIES[name] for name in seq]
    return len(cats) == len(set(cats))


def initial_conditions() -> list[dict[str, Any]]:
    """The 192 symbolic initial conditions, ``get_sequences``:363-365."""
    def keep(values) -> bool:
        return (values.count("table") in [1, 2] and values.count("slider_right") < 2
                and values.count("slider_left") < 2)

    return [dict(zip(POSSIBLE_CONDITIONS.keys(), vals))
            for vals in filter(keep, product(*POSSIBLE_CONDITIONS.values()))]


def _sequences_for_condition(state: dict, n: int, i: int) -> list[tuple[str, ...]]:
    """``get_sequences_for_state2``:333 — rejection sampling under seed ``i``."""
    names = np.asarray(TASKS)          # np.random.choice draws from pop_size alone, so an array is the same stream
    out: list[tuple[str, ...]] = []
    np.random.seed(i)
    while len(out) < n:
        seq = np.random.choice(names, size=SEQ_LEN, replace=False)
        if check_sequence(state, seq):
            out.append(tuple(seq.tolist()))
    return out


def _generate(num_sequences: int) -> list[tuple[dict[str, Any], tuple[str, ...]]]:
    """``get_sequences``:350, serial. Upstream farms the per-condition draws
    out to a process pool, which isolates the global numpy RNG for free and
    leaves the parent's state — seeded 0 — untouched for the final shuffle;
    ``_temp_seed`` reproduces that isolation in one process. Verified to give
    the identical 1000 pairs (2026-09-10)."""
    states = initial_conditions()
    per = list(map(len, np.array_split(range(num_sequences), len(states))))
    with _temp_seed(0):
        drawn: list[tuple[str, ...]] = []
        for i, (state, n) in enumerate(zip(states, per)):
            with _temp_seed(i):
                drawn.extend(_sequences_for_condition(state, n, i))
        results = list(zip(np.repeat(states, per), drawn))
        np.random.shuffle(results)
    return [(dict(state), tuple(seq)) for state, seq in results]


def _cache_file(root: Path, num_sequences: int) -> Path:
    return root / "calvin" / f"eval_sequences_{num_sequences}.json"


@functools.lru_cache(maxsize=4)
def eval_sequences(num_sequences: int = NUM_SEQUENCES,
                   cache_dir: str | None = None) -> tuple[tuple[dict[str, Any], tuple[str, ...]], ...]:
    """CALVIN's evaluation sequences, ``(initial_condition, five task names)``
    in the evaluator's own order. Generated (~2 min the first time), then
    cached in this process and, when ``cache_dir`` is given, as JSON there."""
    path = _cache_file(Path(cache_dir), num_sequences) if cache_dir else None
    if path is not None and path.is_file():
        try:
            raw = json.loads(path.read_text())
            if len(raw) == num_sequences:
                return tuple((dict(c), tuple(s)) for c, s in raw)
        except (ValueError, TypeError, KeyError):
            pass          # a truncated or stale cache is regenerated, never trusted
    out = _generate(num_sequences)
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([[c, list(s)] for c, s in out]))
        except OSError:
            pass          # a read-only data root is not an error; the in-process cache still holds
    return tuple(out)


# ---- the initial state, transcribed ------------------------------------------------------

def fnv1_32(text: str) -> int:
    """``pyhash.fnv1_32()`` on a python ``str``, transcribed so the loaders do
    not need that (unbuildable on python >= 3.10) package.

    Two details of pyhash 0.9.3 matter and are easy to get wrong:

    * it hashes a ``str`` as **UTF-16** — ``Hash.h``:253 encodes with
      ``PyUnicode_AsUTF16String`` and strips the two BOM bytes, so an ASCII
      character contributes ``(c, 0x00)``;
    * ``fnv1_32_t`` calls ``fnv_32_buf(buf, len, seed)`` (``FNV1.h``:37) with
      the constructor's seed, which defaults to **0** — not the FNV-1 offset
      basis 0x811c9dc5. The mixing order is FNV-1's (multiply, then xor:
      ``fnv/hash_32.c``:103-109).
    """
    h = 0
    for byte in text.encode("utf-16-le"):
        h = (h * 0x01000193) & 0xFFFFFFFF
        h ^= byte
    return h


def env_state_for(condition: dict[str, Any]) -> tuple[list[float], list[float]]:
    """``(robot_obs, scene_obs)`` for a symbolic initial condition —
    ``evaluation/utils.py`` ``get_env_state_for_initial_condition``:207,
    transcribed verbatim (the block-position constants, the
    ``pi/2 +- pi/8`` rotation range, and the condition-derived seed that
    decides which of the two table slots each block takes)."""
    pi = np.pi
    robot_obs = [
        0.02586889, -0.2313129, 0.5712808, 3.09045411, -0.02908596, 1.50013585, 0.07999963,
        -1.21779124, 1.03987629, 2.11978254, -2.34205014, -0.87015899, 1.64119093, 0.55344928, 1.0,
    ]
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    seed = fnv1_32(str(condition.values()))
    with _temp_seed(seed):
        np.random.shuffle(block_table)
        scene_obs = np.zeros(24)
        if condition["slider"] == "left":
            scene_obs[0] = 0.28
        if condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = condition["lightbulb"]
        scene_obs[5] = condition["led"]
        if condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        if condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        if condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)
    return robot_obs, [float(v) for v in scene_obs]


# ---- the loader ---------------------------------------------------------------------------

def config_file(root: str | os.PathLike, dataset: str = DATASET_DIR) -> str:
    """``<data_root>/calvin/<dataset>/validation/.hydra/merged_config.yaml`` —
    what ``evaluate_policy.make_env`` hands ``get_env``."""
    path = Path(root) / "calvin" / dataset
    if not path.is_dir():
        path = Path(dataset) if os.path.isabs(dataset) else path
    return str(path.joinpath(*CONFIG_REL))


def load_episodes(split: str, data_root_: str | os.PathLike | None = None,
                  scene_root_: str | os.PathLike | None = None, calvin_dataset: str = DATASET_DIR,
                  **_unused: Any) -> list[Episode]:
    """``scene_root`` is accepted for the shared signature and ignored."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    root = data_root(data_root_)
    scene = CalvinSceneRef(env_letter=ENV_LETTER, config_file=config_file(root, calvin_dataset))
    pairs = eval_sequences(NUM_SEQUENCES, cache_dir=str(root))
    if split == "mini":
        pairs = pairs[:MINI_SEQUENCES]
    out: list[Episode] = []
    for i, (condition, sequence) in enumerate(pairs):
        robot_obs, scene_obs = env_state_for(condition)
        instructions = tuple(INSTRUCTIONS[name] for name in sequence)
        out.append(Episode(
            index=i, episode_id=f"{ENV_LETTER}/{i}", scene=scene,
            start_position=None, start_rotation=None,
            goal=GoalSequence(tuple(ManipGoal((("calvin_task", name),)) for name in sequence)),
            instruction=instructions[0],          # the chain reveals one instruction at a time
            info={"env_letter": ENV_LETTER, "sequence": list(sequence), "instructions": list(instructions),
                  "initial_condition": dict(condition), "robot_obs": robot_obs, "scene_obs": scene_obs,
                  "n_subtasks": len(sequence)},
        ))
    return out


def _decl(upstream: bool) -> Benchmark:
    return Benchmark(
        name="calvin-d" + ("-upstream" if upstream else ""),
        gym_id="EmbodiedScore/CALVIN-D-Upstream-v0" if upstream else "EmbodiedScore/CALVIN-D-v0",
        body=bodies.CALVIN if upstream else bodies.CALVIN_STANDARD,
        actions=(), splits=SPLITS,
        episodes=lambda split, data_root=None, scene_root=None, **kw: load_episodes(split, data_root, scene_root, **kw),
        metrics=lambda env, **o: ChainMetrics(env),
        depth=None,
        # the real budget is per sub-task (``ticks`` upstream, MACRO_STEPS_PER_SUBTASK here);
        # this is only the episode's ceiling — five sub-tasks' worth.
        max_episode_steps=EP_LEN * SEQ_LEN if upstream else MACRO_STEPS,
        ticks=EP_LEN if upstream else EP_LEN * TICK_GUARD,
        engine="calvin", macro=not upstream, variant="upstream" if upstream else "standard",
        description="CALVIN environment D, 1000 chains of 5 instructions — "
                    + ("as its evaluator runs it: per-tick relative end-effector commands on the 200x200 rig"
                       if upstream else "on the macro pose protocol and the 256² rig"),
    )


BENCHMARKS = (_decl(False), _decl(True))
